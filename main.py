"""
Endpoints:
  POST /ingest        — Upload and index a document
  POST /query         — Ask a question against indexed documents
  GET  /sources       — List uploaded documents for a user
  DELETE /sources/{source} — Remove a document from the index
  POST /session/clear — Clear conversation history for a session
  GET  /health        — Health check

"""
from __future__ import annotations

import logging
import os
import uuid

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from src.config import get_settings
from src.rag_engine import MultiTenantRAGEngine

#  Setup                                                                       
settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

#  App                                                                         
app = FastAPI(
    title="DocuRAG API",
    description="RAG-powered document intelligence backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: MultiTenantRAGEngine | None = None


def get_engine() -> MultiTenantRAGEngine:
    global _engine
    if _engine is None:
        _engine = MultiTenantRAGEngine()
    return _engine


#  Auth (mock — replace with real JWT decode)                           

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def get_current_user_id(token: str | None = Depends(oauth2_scheme)) -> str:
    """
    Mock auth: returns the token itself as user_id.
    In production: decode JWT and return user['sub'].
    """
    if not token:
        # For local dev, fall back to a default user
        return "local_dev_user"
    return token


#  Request / Response models                                                  
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None  # Auto-generated if not provided


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


class ClearSessionRequest(BaseModel):
    session_id: str


#  Endpoints                                                                    
@app.get("/health")
async def health():
    return {"status": "ok", "service": "DocuMind API"}


@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    engine: MultiTenantRAGEngine = Depends(get_engine),
):
    """
    Upload and index a document.
    Processing runs in the background to avoid request timeout.
    """
    content = await file.read()  # Read bytes before passing to background task

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    background_tasks.add_task(
        engine.ingest_file, content, file.filename, user_id
    )

    return {
        "message": f"'{file.filename}' queued for indexing.",
        "filename": file.filename,
        "size_bytes": len(content),
    }


@app.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    user_id: str = Depends(get_current_user_id),
    engine: MultiTenantRAGEngine = Depends(get_engine),
):
    """Ask a question against the user's indexed documents."""
    session_id = body.session_id or str(uuid.uuid4())

    try:
        result = engine.ask(
            user_id=user_id,
            session_id=session_id,
            question=body.question,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed for user '{user_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@app.get("/sources")
async def list_sources(
    user_id: str = Depends(get_current_user_id),
    engine: MultiTenantRAGEngine = Depends(get_engine),
):
    """List all uploaded documents for the current user."""
    sources = engine.list_sources(user_id)
    count = engine.document_count(user_id)
    return {"sources": sources, "total_chunks": count}


@app.delete("/sources/{source_name}")
async def delete_source(
    source_name: str,
    user_id: str = Depends(get_current_user_id),
    engine: MultiTenantRAGEngine = Depends(get_engine),
):
    """Remove a document from the user's knowledge base."""
    try:
        engine.delete_source(user_id, source_name)
        return {"message": f"'{source_name}' removed from knowledge base."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/session/clear")
async def clear_session(
    body: ClearSessionRequest,
    engine: MultiTenantRAGEngine = Depends(get_engine),
):
    """Clear conversation history for a session."""
    engine.clear_session(body.session_id)
    return {"message": f"Session '{body.session_id}' cleared."}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )