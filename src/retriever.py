"""
Advanced 4-stage retrieval pipeline:

  Stage 1 — Query Rewriting   : MultiQueryRetriever generates query variants
  Stage 2 — Hybrid Search     : Semantic (Chroma) + Keyword (BM25) via EnsembleRetriever
  Stage 3 — Re-ranking        : FlashRank cross-encoder selects top-N most relevant chunks
  Stage 4 — Result            : Returns ranked Document list with metadata

Retriever instances are cached per user_id and invalidated when new
documents are added.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import MultiQueryRetriever

from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import get_settings
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)
settings = get_settings()


def _build_llm():
    """Returns LLM for query rewriting (must match rag_engine LLM)."""
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=settings.groq_model,
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )


class RetrieverFactory:
    """
    Builds and caches the full retrieval pipeline per user.
    Call `.invalidate(user_id)` after adding new documents for that user.
    """

    def __init__(self, vs_manager: VectorStoreManager):
        self.vs_manager = vs_manager
        self.llm = _build_llm()
        self._cache: dict[str, ContextualCompressionRetriever] = {}

    def get(self, user_id: str):
        """Returns a cached retriever, building it if necessary."""
        if user_id not in self._cache:
            logger.info(f"Building retrieval pipeline for user '{user_id}'")
            self._cache[user_id] = self._build(user_id)
        return self._cache[user_id]

    def invalidate(self, user_id: str) -> None:
        """Clears cached retriever so next call rebuilds with fresh data."""
        self._cache.pop(user_id, None)
        logger.debug(f"Retriever cache invalidated for user '{user_id}'")

    # def _build(self, user_id: str):
    #     """
    #     Constructs the full 4-stage pipeline for a user.
    #     Falls back to semantic-only if no documents are found for BM25.
    #     """
    #     # ── Stage 1 base: Semantic retriever ──────────────────────────── #
    #     vector_retriever = self.vs_manager.get_base_retriever(user_id)

    #     # ── Stage 2: Hybrid search (Semantic + BM25) ──────────────────── #
    #     raw = self.vs_manager.get_user_documents_raw(user_id)
    #     texts = raw.get("documents", [])
    #     metadatas = raw.get("metadatas", [])

    #     if texts:
    #         bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    #         bm25.k = settings.bm25_k

    #         ensemble = EnsembleRetriever(
    #             retrievers=[vector_retriever, bm25],
    #             weights=[
    #                 settings.ensemble_vector_weight,
    #                 1 - settings.ensemble_vector_weight,
    #             ],
    #         )
    #         base_retriever = ensemble
    #         logger.debug(f"Hybrid retriever built for user '{user_id}'")
    #     else:
    #         # No documents yet — return a no-op retriever
    #         logger.warning(
    #             f"No documents found for user '{user_id}'. "
    #             "Using semantic-only retriever."
    #         )
    #         base_retriever = vector_retriever

    #     # ── Stage 3: Multi-query rewriting ────────────────────────────── #
    #     mq_retriever = MultiQueryRetriever.from_llm(
    #         retriever=base_retriever,
    #         llm=self.llm,
    #     )

    #     # ── Stage 4: Cross-encoder re-ranking ─────────────────────────── #
    #     reranker = ContextualCompressionRetriever(
    #         base_compressor=FlashrankRerank(top_n=settings.rerank_top_n),
    #         base_retriever=mq_retriever,
    #     )

    #     return reranker

    def _build(self, user_id: str):
        """
        Constructs the full 4-stage pipeline for a user using Cohere API for reranking.
        """
        from langchain_cohere import CohereRerank
        from langchain_classic.retrievers import ContextualCompressionRetriever

        # ── Stage 1: Semantic retriever ──────────────────────────────── #
        vector_retriever = self.vs_manager.get_base_retriever(user_id)

        # ── Stage 2: Hybrid search (Semantic + BM25) ─────────────────── #
        raw = self.vs_manager.get_user_documents_raw(user_id)
        texts = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        if texts:
            bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
            bm25.k = settings.bm25_k

            base_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25],
                weights=[
                    settings.ensemble_vector_weight,
                    1 - settings.ensemble_vector_weight,
                ],
            )
        else:
            base_retriever = vector_retriever

        # ── Stage 3: Multi-query rewriting ───────────────────────────── #
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )

        # ── Stage 4: Cohere API Re-ranking ────────────────────────────── #
        # This replaces FlashrankRerank to save local RAM
        compressor = CohereRerank(
            cohere_api_key=settings.cohere_api_key,
            model="rerank-english-v3.0",
            top_n=settings.rerank_top_n
        )

        reranker = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=mq_retriever,
        )

        return reranker

    def retrieve(self, user_id: str, query: str) -> list[Document]:
        """
        Convenience method: runs the full pipeline and returns ranked docs.
        """
        retriever = self.get(user_id)
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            logger.error(f"Retrieval failed for user '{user_id}': {e}")
            # Fallback to basic semantic search
            docs = self.vs_manager.get_base_retriever(user_id).invoke(query)
        return docs