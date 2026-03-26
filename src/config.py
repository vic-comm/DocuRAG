"""
Centralised settings loaded from environment variables.
All other modules import from here — never import os.environ directly.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

import os

# Support Streamlit Cloud secrets
try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ.setdefault(key.upper(), str(value))
except Exception:
    pass 

class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    use_openai_embeddings: bool = Field(default=False, env="USE_OPENAI_EMBEDDINGS")

    # LLM model names
    groq_model: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")

    # Vector store
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")

    # Retrieval
    vector_search_k: int = Field(default=20, env="VECTOR_SEARCH_K")
    bm25_k: int = Field(default=10, env="BM25_K")
    rerank_top_n: int = Field(default=10, env="RERANK_TOP_N")
    ensemble_vector_weight: float = Field(default=0.6, env="ENSEMBLE_VECTOR_WEIGHT")

    # Chunking
    chunk_size: int = Field(default=1500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=300, env="CHUNK_OVERLAP")

    # Memory
    history_k: int = Field(default=6, env="HISTORY_K")

    # LangSmith
    langchain_tracing_v2: str = Field(default="false", env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="DocuMind", env="LANGCHAIN_PROJECT")
    langchain_api_key: str = Field(default="", env="LANGCHAIN_API_KEY")

    # App
    app_env: str = Field(default="development", env="APP_ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    pinecone_api_key: str = Field(default="", env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="documind", env="PINECONE_INDEX_NAME")
    use_pinecone: bool = Field(default=False, env="USE_PINECONE")
    google_api_key: str = Field(default='', env='GOOGLE_API_KEY')
    cohere_api_key: str = Field(default='', env='COHERE_API_KEY')

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def llm_provider(self) -> str:
        """Returns which LLM provider to use based on available keys."""
        if self.groq_api_key:
            return "groq"
        if self.openai_api_key:
            return "openai"
        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in .env"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()