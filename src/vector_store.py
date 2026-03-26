"""
VectorStoreManager wraps ChromaDB with:
  - HuggingFace or OpenAI embeddings
  - Per-user document isolation via metadata filtering
  - Add / delete / list operations
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# def _build_embeddings() -> Embeddings:
#     """Returns the configured embeddings model."""
#     if settings.use_openai_embeddings:
#         from langchain_openai import OpenAIEmbeddings
#         logger.info("Using OpenAI embeddings (text-embedding-3-small)")
#         return OpenAIEmbeddings(
#             model="text-embedding-3-small",
#             openai_api_key=settings.openai_api_key,
#         )
#     else:
#         from langchain_huggingface import HuggingFaceEmbeddings
#         logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
#         return HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )

def _build_embeddings() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    logger.info("Using Gemini Cloud Embeddings (text-embedding-004)")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.google_api_key ,# Ensure this is in your config
        max_retries=6,
        output_dimensionality=768
    )

def _build_chroma(embeddings: Embeddings) -> VectorStore:
    """Local ChromaDB — good for development."""
    from langchain_chroma import Chroma
    logger.info(f"Using ChromaDB at '{settings.chroma_persist_dir}'")
    return Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings,
        collection_name="documind",
    )

def _build_pinecone(embeddings: Embeddings) -> VectorStore:
    """Pinecone cloud vector store — good for production."""
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    if not settings.pinecone_api_key:
        raise ValueError(
            "PINECONE_API_KEY not set. Add it to your .env file."
        )

    # Initialise Pinecone client
    pc = Pinecone(api_key=settings.pinecone_api_key)

    # Create index if it doesn't exist yet
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing_indexes:
        logger.info(
            f"Creating Pinecone index '{settings.pinecone_index_name}'..."
        )
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=768,  # Must match all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created.")
    else:
        logger.info(
            f"Using existing Pinecone index '{settings.pinecone_index_name}'"
        )

    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
    )

def is_rate_limit_error(exception):
    return "429" in str(exception) or "RESOURCE_EXHAUSTED" in str(exception)

class VectorStoreManager:
    """
    Unified interface for ChromaDB and Pinecone.
    Switch between them by setting USE_PINECONE=true in .env.
    All other files (retriever.py, rag_engine.py) stay unchanged.
    """
    def __init__(self):
        self.embeddings = _build_embeddings()
        self.using_pinecone = settings.use_pinecone

        if self.using_pinecone:
            self.vectorstore = _build_pinecone(self.embeddings)
        else:
            self.vectorstore = _build_chroma(self.embeddings)

        logger.info(
            f"VectorStore ready — "
            f"{'Pinecone' if self.using_pinecone else 'ChromaDB'}"
        )

    @retry(
        retry=retry_if_exception_type(Exception), # Standard Pinecone/Google errors
        wait=wait_exponential(multiplier=2, min=10, max=60), # Wait 10s, then 20s, 40s...
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit! Retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def _safe_add_batch(self, batch):
        """Helper to add a single batch with retry logic."""
        self.vectorstore.add_documents(batch)

    #  Write                                                             
    def add_documents(self, chunks: list[Document], user_id: str) -> int:
        import time
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id

        # --- BATCHING FIX ---
        batch_size = 50  # Stay safely under the 100 RPM limit
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            # Use the retry-protected helper
            self._safe_add_batch(batch)
            
            logger.info(f"Successfully indexed batch {i//batch_size + 1}")
            
            # Mandatory breathing room between batches
            if i + batch_size < len(chunks):
                time.sleep(5)
        return len(chunks)
    
    def delete_user_documents(
        self, user_id: str, source: Optional[str] = None
    ) -> None:
        """Delete documents for a user, optionally filtered by source file."""
        if self.using_pinecone:
            self._delete_pinecone(user_id, source)
        else:
            self._delete_chroma(user_id, source)

    def _delete_chroma(
        self, user_id: str, source: Optional[str] = None
    ) -> None:
        where_filter: dict = {"user_id": user_id}
        if source:
            where_filter["source"] = source
        try:
            self.vectorstore.delete(where=where_filter)
        except Exception as e:
            logger.error(f"Chroma delete failed: {e}")
            raise

    def _delete_pinecone(
        self, user_id: str, source: Optional[str] = None
    ) -> None:
        """
        Pinecone deletion requires fetching IDs first, then deleting by ID.
        """
        try:
            index = self.vectorstore._index
            filter_dict: dict = {"user_id": {"$eq": user_id}}
            if source:
                filter_dict["source"] = {"$eq": source}

            # Fetch matching vector IDs
            results = index.query(
                vector=[0.0] * 768,  # Changed from 384
                filter=filter_dict,
                top_k=10000,
                include_values=False,
            )
            ids = [match["id"] for match in results.get("matches", [])]

            if ids:
                batch_size = 1000
                for i in range(0, len(ids), batch_size):
                    batch = ids[i : i + batch_size]
                    index.delete(ids=batch)
                    logger.info(f"Deleted batch of {len(batch)} vectors...")
            else:
                logger.info(f"No vectors found for user '{user_id}'")
        except Exception as e:
            logger.error(f"Pinecone delete failed: {e}")
            raise

    #  Read                                                                
    def get_user_documents_raw(self, user_id: str) -> dict:
        """
        Returns raw documents for BM25 construction.
        """
        if self.using_pinecone:
            return self._get_raw_pinecone(user_id)
        else:
            return self.vectorstore.get(where={"user_id": user_id})
        
    def _get_raw_pinecone(self, user_id: str) -> dict:
        """Fetch all document texts for a user from Pinecone."""
        try:
            index = self.vectorstore._index
            results = index.query(
                vector=[0.0] * 768,  # Changed from 384
                filter={"user_id": {"$eq": user_id}},
                top_k=10000,
                include_metadata=True,
                include_values=False,
            )
            matches = results.get("matches", [])
            documents = [
                m["metadata"].get("text", "") for m in matches
            ]
            metadatas = [m["metadata"] for m in matches]
            return {"documents": documents, "metadatas": metadatas}
        except Exception as e:
            logger.error(f"Pinecone raw fetch failed: {e}")
            return {"documents": [], "metadatas": []}

    def list_user_sources(self, user_id: str) -> list[str]:
        """Returns unique source filenames for a user."""
        raw = self.get_user_documents_raw(user_id)
        metadatas = raw.get("metadatas", [])
        if not metadatas:
            return []
            
        sources = {
            meta.get("source")
            for meta in metadatas
            if meta and meta.get("source")
        }
        return sorted(list(sources))

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

    def get_base_retriever(self, user_id: str):
        """Base semantic retriever filtered to a single user."""
        if self.using_pinecone:
            # Pinecone uses different filter syntax
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": settings.vector_search_k,
                    "filter": {"user_id": {"$eq": user_id}},
                },
            )
        else:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": settings.vector_search_k,
                    "filter": {"user_id": user_id},
                },
            )
        
    def document_count(self, user_id: str) -> int:
        """Returns the number of stored chunks for a user by querying metadata."""
        if self.using_pinecone:
            try:
                index = self.vectorstore._index
                # On Serverless, we use a standard query to count matches
                results = index.query(
                    vector=[0.0] * 768, # Changed from 384
                    filter={"user_id": {"$eq": user_id}},
                    top_k=10000,
                    include_values=False,
                    include_metadata=False
                )
                return len(results.get("matches", []))
            except Exception as e:
                logger.error(f"Error counting Pinecone docs: {e}")
                return 0
        else:
            # ChromaDB logic remains the same
            raw = self.vectorstore.get(where={"user_id": user_id})
            return len(raw.get("ids", []))