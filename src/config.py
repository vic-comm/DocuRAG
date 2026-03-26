import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Try to pull from Streamlit secrets and force them into environment
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            # Force uppercase and ensure it's a string
            os.environ[key.upper()] = str(value)
except Exception:
    pass

class Settings(BaseSettings):
    # Core API Keys - Remove the default="" to force a check, 
    # or handle the empty string in the property/logic.
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    google_api_key: str = Field(default="", validation_alias="GOOGLE_API_KEY")
    cohere_api_key: str = Field(default="", validation_alias="COHERE_API_KEY")
    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    
    # Vector store
    use_pinecone: bool = Field(default=True, validation_alias="USE_PINECONE")
    pinecone_index_name: str = Field(default="documind", validation_alias="PINECONE_INDEX_NAME")
    chroma_persist_dir: str = Field(default="./chroma_db", validation_alias="CHROMA_PERSIST_DIR")

    # Retrieval Tuning
    vector_search_k: int = Field(default=20)
    bm25_k: int = Field(default=10)
    rerank_top_n: int = Field(default=10)
    ensemble_vector_weight: float = Field(default=0.4)

    # Chunking
    chunk_size: int = Field(default=1500)
    chunk_overlap: int = Field(default=300)

    # App Metadata
    langchain_project: str = Field(default="DocuMind")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    groq_model: str = Field(default="llama-3.1-8b-instant", validation_alias="GROQ_MODEL")
    openai_model: str = Field(default="gpt-4o", validation_alias="OPENAI_MODEL")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    use_openai_embeddings: bool = Field(default=False, validation_alias="USE_OPENAI_EMBEDDINGS")
    history_k: int = Field(default=6, validation_alias="HISTORY_K")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    langchain_tracing_v2: str = Field(default="false", validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", validation_alias="LANGCHAIN_API_KEY")

    def validate_keys(self):
        """Manual check to prevent the cryptic Pydantic errors in the engine."""
        missing = []
        if not self.google_api_key: missing.append("GOOGLE_API_KEY")
        if not self.groq_api_key: missing.append("GROQ_API_KEY")
        if self.use_pinecone and not self.pinecone_api_key: missing.append("PINECONE_API_KEY")
        
        if missing:
            import streamlit as st
            error_msg = f"Missing Secrets: {', '.join(missing)}. Please add them to Streamlit Cloud Settings."
            st.error(error_msg)
            raise ValueError(error_msg)

    def get_llm_provider(self) -> str:
        if self.groq_api_key and len(self.groq_api_key) > 0:
            return "groq"
        if self.openai_api_key and len(self.openai_api_key) > 0:
            return "openai"
        return "none" # Return a safe fallback
    
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # If in production (Streamlit Cloud), run the validation
    if os.getenv("STREAMLIT_RUNTIME_ENV", ""):
        settings.validate_keys()
    return settings