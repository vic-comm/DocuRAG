"""
src/rag_engine.py
──────────────────
MultiTenantRAGEngine:
  - Per-user document isolation (Chroma metadata filter)
  - Per-session conversational memory (summary buffer)
  - Full 4-stage retrieval pipeline (hybrid + rerank + rewrite)
  - Structured response: answer + source citations

Architecture:
  [Query]
      │
      ▼
  [ConversationHistory]  ←── session memory
      │
      ▼
  [RetrieverFactory]     ←── hybrid search → rerank
      │  (docs + context)
      ▼
  [ChatPromptTemplate]   ←── system + history + context + question
      │
      ▼
  [LLM]                  ←── Groq / OpenAI
      │
      ▼
  [Response: answer + sources]
"""
from __future__ import annotations

import logging

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import get_settings
from src.conversation_history import ConversationSummaryBufferMessageHistory
from src.dataloader import process_uploaded_file
from src.retriever import RetrieverFactory
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)
settings = get_settings()

# ──────────────────────────────────────────────────────────────────────────── #
#  Prompt                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

SYSTEM_PROMPT = """You are DocuMind, an expert document intelligence assistant.
Your job is to answer questions accurately and concisely using ONLY the provided context.

Rules:
- Answer strictly from the context below. Do not use prior knowledge.
- If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
- Always cite the source document and page number when referencing specific information.
- Be concise but complete. Use bullet points for multi-part answers.
- If asked a follow-up question, use the conversation history to maintain context.

Context from retrieved documents:
────────────────────────────────
{context}
────────────────────────────────"""


#  LLM factory                                                                  #

def _build_llm():
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


#  Engine                                                                        
class MultiTenantRAGEngine:
    """
    Production-ready RAG engine with multi-tenant isolation,
    conversational memory, and advanced retrieval.
    """

    def __init__(self):
        self.vs_manager = VectorStoreManager()
        self.retriever_factory = RetrieverFactory(self.vs_manager)
        self.llm = _build_llm()

        # Per-session conversation stores {session_id: history}
        self._session_store: dict[str, BaseChatMessageHistory] = {}

        # Build the chain once
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        self._chain = RunnableWithMessageHistory(
            self.prompt | self.llm | StrOutputParser(),
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        logger.info("MultiTenantRAGEngine initialised.")

    #  Document management                                                 
    def ingest_file(
        self, file_content: bytes, filename: str, user_id: str
    ) -> dict:
        """
        Process a raw file upload and add it to the user's knowledge base.

        Returns:
            {"chunks_added": int, "source": str}
        """
        chunks = process_uploaded_file(file_content, filename)
        count = self.vs_manager.add_documents(chunks, user_id)
        # Invalidate retriever so next query uses fresh data
        self.retriever_factory.invalidate(user_id)
        logger.info(f"Ingested '{filename}' for user '{user_id}': {count} chunks")
        return {"chunks_added": count, "source": filename}

    def delete_source(self, user_id: str, source: str) -> None:
        """Remove a specific document from a user's knowledge base."""
        self.vs_manager.delete_user_documents(user_id, source=source)
        self.retriever_factory.invalidate(user_id)

    def list_sources(self, user_id: str) -> list[str]:
        """Return all uploaded document names for a user."""
        return self.vs_manager.list_user_sources(user_id)

    def document_count(self, user_id: str) -> int:
        return self.vs_manager.document_count(user_id)

    #  Querying                                                            

    def ask(self, user_id: str, session_id: str, question: str) -> dict:
        """
        Run a full RAG query for a user's session.

        Args:
            user_id:    Identifies which documents to search.
            session_id: Identifies which conversation history to use.
            question:   The user's question.

        Returns:
            {
                "answer": str,
                "sources": [{"source": str, "page": int, "excerpt": str}],
                "session_id": str,
            }
        """
        if self.document_count(user_id) == 0:
            return {
                "answer": "No documents have been uploaded yet. "
                          "Please upload a PDF or document first.",
                "sources": [],
                "session_id": session_id,
            }

        # 1. Retrieve relevant chunks
        docs = self.retriever_factory.retrieve(user_id, question)

        # 2. Format context with clear source attribution
        context = self._format_context(docs)

        # 3. Generate answer with history
        answer = self._chain.invoke(
            {"context": context, "question": question},
            config={"configurable": {"session_id": session_id}},
        )

        # 4. Build source list for UI display
        sources = self._extract_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
        }

    def clear_session(self, session_id: str) -> None:
        """Clears conversation history for a session."""
        if session_id in self._session_store:
            self._session_store[session_id].clear()
            logger.info(f"Cleared history for session '{session_id}'")

    #  Private helpers                                                     
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._session_store:
            self._session_store[session_id] = (
                ConversationSummaryBufferMessageHistory(
                    llm=self.llm,
                    k=settings.history_k,
                )
            )
        return self._session_store[session_id]

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        """Formats retrieved documents into a readable context block."""
        if not docs:
            return "No relevant context found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            parts.append(
                f"[{i}] Source: {source} | Page: {page}\n{doc.page_content}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _extract_sources(docs: list[Document]) -> list[dict]:
        """Builds the source citation list returned to the client."""
        seen = set()
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": source,
                    "page": page,
                    "excerpt": doc.page_content[:250].strip() + "…",
                })
        return sources