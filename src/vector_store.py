"""
src/vector_store.py
────────────────────
VectorStoreManager wraps ChromaDB with:
  - HuggingFace (local, free) or OpenAI embeddings
  - Per-user document isolation via metadata filtering
  - Add / delete / list operations
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _build_embeddings() -> Embeddings:
    """Returns the configured embeddings model."""
    if settings.use_openai_embeddings:
        from langchain_openai import OpenAIEmbeddings
        logger.info("Using OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


class VectorStoreManager:
    """
    Singleton-style manager for the ChromaDB vector store.
    Handles multi-tenant isolation via user_id metadata filters.
    """

    def __init__(self):
        self.embeddings = _build_embeddings()
        self.vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name="documind",
        )
        logger.info(
            f"VectorStore initialised at '{settings.chroma_persist_dir}'"
        )

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def add_documents(self, chunks: list[Document], user_id: str) -> int:
        """
        Embeds and stores document chunks tagged with user_id.

        Returns:
            Number of chunks added.
        """
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id

        self.vectorstore.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks for user '{user_id}'")
        return len(chunks)

    def delete_user_documents(self, user_id: str, source: Optional[str] = None) -> None:
        """
        Delete all documents for a user, or only those from a specific source file.
        """
        where_filter: dict = {"user_id": user_id}
        if source:
            where_filter["source"] = source

        try:
            self.vectorstore.delete(where=where_filter)
            target = f"'{source}'" if source else "all documents"
            logger.info(f"Deleted {target} for user '{user_id}'")
        except Exception as e:
            logger.error(f"Delete failed for user '{user_id}': {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def get_user_documents_raw(self, user_id: str) -> dict:
        """
        Returns raw Chroma results for a user — used for BM25 construction.
        """
        return self.vectorstore.get(where={"user_id": user_id})

    def list_user_sources(self, user_id: str) -> list[str]:
        """
        Returns unique source filenames for a user.
        """
        raw = self.get_user_documents_raw(user_id)
        sources = {
            meta.get("source", "unknown")
            for meta in raw.get("metadatas", [])
            if meta
        }
        return sorted(sources)

    def get_base_retriever(self, user_id: str):
        """
        Returns a basic semantic retriever filtered to a single user.
        """
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": settings.vector_search_k,
                "filter": {"user_id": user_id},
            },
        )

    def document_count(self, user_id: str) -> int:
        """Returns the number of stored chunks for a user."""
        raw = self.get_user_documents_raw(user_id)
        return len(raw.get("ids", [])) 