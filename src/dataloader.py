"""
Handles loading documents from:
  - Uploaded file bytes (FastAPI / Streamlit)
  - A directory on disk (batch ingestion)

Supported formats: PDF, DOCX, TXT, HTML, Markdown
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Supported file types and their loaders
LOADER_MAP: dict[str, type] = {
    ".pdf": PyMuPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


#  Core splitting logic                                                       
def _split_documents(docs: list[Document], source_name: str) -> list[Document]:
    """Split documents into chunks and attach source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    # Normalise metadata across all chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_name
        chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))
        chunk.metadata["chunk_index"] = i

    filtered = [c for c in chunks if c.page_content.strip()]
    logger.info(
        f"'{source_name}': {len(docs)} pages → {len(filtered)} chunks"
    )
    return filtered


#  File bytes → chunks  (used by FastAPI & Streamlit upload)                   
def process_uploaded_file(file_content: bytes, filename: str) -> list[Document]:
    """
    Saves raw bytes to a temp file, parses it with the appropriate loader,
    splits into chunks, and cleans up the temp file.

    Args:
        file_content: Raw bytes of the uploaded file.
        filename:     Original filename (used to determine extension and metadata).

    Returns:
        List of Document chunks ready for embedding.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in LOADER_MAP:
        supported = ", ".join(LOADER_MAP.keys())
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {supported}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        loader_cls = LOADER_MAP[suffix]
        loader = loader_cls(tmp_path)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No content could be extracted from '{filename}'.")

        return _split_documents(docs, source_name=filename)

    except Exception as e:
        logger.error(f"Failed to process '{filename}': {e}")
        raise
    finally:
        Path(tmp_path).unlink(missing_ok=True)


#  Directory → chunks  (used for batch ingestion)                              
def load_directory(data_dir: str) -> list[Document]:
    """
    Recursively loads all supported documents from a directory.

    Args:
        data_dir: Path to the directory containing documents.

    Returns:
        List of Document chunks from all files.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_chunks: list[Document] = []

    for suffix, loader_cls in LOADER_MAP.items():
        files = list(data_path.rglob(f"*{suffix}"))
        for file_path in files:
            try:
                loader = loader_cls(str(file_path))
                docs = loader.load()
                if docs:
                    chunks = _split_documents(docs, source_name=file_path.name)
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Skipping '{file_path.name}': {e}")
                continue

    logger.info(
        f"Directory load complete: {len(all_chunks)} total chunks "
        f"from {data_dir}"
    )
    return all_chunks