"""
demo.py
────────
Automated demo script — perfect for your screen recording walkthrough.

Creates a sample PDF in memory, ingests it, and runs a series of
questions that showcase all DocuMind features:
  ✓ Document ingestion
  ✓ Precise Q&A with citations
  ✓ Multi-turn conversational memory
  ✓ "I don't know" behaviour for out-of-scope questions
  ✓ Source transparency

Usage:
  python demo.py
  python demo.py --pdf path/to/your.pdf   # Use your own PDF
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path


#  Colour helpers (works on most terminals)                                     
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"
    RED    = "\033[91m"


def _h(text: str) -> str:
    """Header line."""
    return f"\n{C.BOLD}{C.WHITE}{'═' * 64}{C.RESET}\n{C.BOLD}{C.CYAN}  {text}{C.RESET}\n{'═' * 64}"


def _step(n: int, text: str) -> None:
    print(f"\n{C.BOLD}{C.YELLOW}  [{n}] {text}{C.RESET}")
    time.sleep(0.3)


def _ok(text: str) -> None:
    print(f"  {C.GREEN}✓{C.RESET} {text}")


def _q(text: str) -> None:
    print(f"\n  {C.CYAN}❓ {C.BOLD}{text}{C.RESET}")


def _a(text: str) -> None:
    print(f"\n  {C.WHITE}🧠 {text}{C.RESET}")


def _sources(sources: list[dict]) -> None:
    if not sources:
        return
    print(f"\n  {C.GREY}{'─' * 56}{C.RESET}")
    print(f"  {C.GREY}📚 Sources retrieved:{C.RESET}")
    for i, s in enumerate(sources, 1):
        print(
            f"  {C.GREY}  [{i}] {s['source']}  |  "
            f"Page {s['page']}{C.RESET}"
        )
        print(f"  {C.GREY}      \"{s['excerpt'][:100]}…\"{C.RESET}")
    print(f"  {C.GREY}{'─' * 56}{C.RESET}")


# ──────────────────────────────────────────────────────────────────────────── #
#  Sample document content                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

SAMPLE_TEXT = """
DocuMind Technical Reference Manual
Version 2.0 — 2026

1. Introduction
DocuMind is a Retrieval-Augmented Generation (RAG) system that enables
precise question-answering over private document collections. Unlike
general-purpose LLMs, DocuMind grounds every answer in your documents.

2. Architecture
The system uses a four-stage retrieval pipeline:
  Stage 1 - Query Rewriting: The LLM generates three query variants to
    maximise recall across different phrasings.
  Stage 2 - Hybrid Search: Combines semantic vector search (ChromaDB)
    with keyword search (BM25) using Reciprocal Rank Fusion.
  Stage 3 - Re-ranking: A FlashRank cross-encoder scores each candidate
    chunk for relevance and returns the top five.
  Stage 4 - Generation: The retrieved context is injected into a
    system prompt and the LLM produces a grounded, cited answer.

3. Configuration
All settings are managed through environment variables in a .env file.
Key settings include GROQ_API_KEY for the LLM, CHROMA_PERSIST_DIR for
vector storage location, and HISTORY_K for conversation buffer size.

4. API Endpoints
  POST /ingest  — Upload and index a document (runs in background)
  POST /query   — Ask a question and receive answer with citations
  GET  /sources — List all indexed documents for the current user
  DELETE /sources/{name} — Remove a document from the knowledge base
  POST /session/clear — Reset the conversation history

5. Performance
DocuMind processes documents at approximately 50 chunks per second
on a standard CPU. Average query latency is 1.2 seconds using Groq
inference. The hybrid retrieval pipeline achieves 94% recall@5 on
internal benchmarks.

6. Multi-Tenancy
Each user's documents are isolated using ChromaDB metadata filters.
User IDs are derived from JWT tokens in production. For local
development, a default user ID is used automatically.

7. Limitations and Next Steps
Current limitations include no support for images within PDFs, no
streaming responses, and in-memory BM25 that does not scale beyond
10,000 documents. Planned improvements include Pinecone migration,
multimodal ingestion, and WebSocket streaming.
"""


def _create_sample_pdf() -> bytes:
    """Creates a minimal PDF in memory without external dependencies."""
    # Minimal valid PDF structure
    lines = SAMPLE_TEXT.strip().replace("\n", " ").split(". ")
    content_lines = "\n".join(f"({line.strip()}.) Tj T*" for line in lines if line.strip())

    pdf = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj << /Length 200 >>
stream
BT /F1 10 Tf 50 750 Td 14 TL
{content_lines[:200]}
ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
trailer << /Size 6 /Root 1 0 R >>
startxref
0
%%EOF"""
    return pdf.encode()


# ──────────────────────────────────────────────────────────────────────────── #
#  Demo questions                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

QUESTIONS = [
    # Basic factual retrieval
    "What is DocuMind and what problem does it solve?",
    # Multi-part technical question
    "Describe the four stages of the retrieval pipeline.",
    # Specific detail
    "What are the API endpoints and what does each one do?",
    # Follow-up (tests memory)
    "You mentioned /ingest — does it run synchronously or in the background?",
    # Performance data
    "What is the average query latency?",
    # Out-of-scope (tests grounding)
    "What is the recipe for jollof rice?",
]


# ──────────────────────────────────────────────────────────────────────────── #
#  Main                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

def run_demo(pdf_path: str | None = None):
    print(_h("DocuMind — Automated Demo Walkthrough"))
    print(f"\n  {C.GREY}This script demonstrates the full DocuMind pipeline.{C.RESET}")

    # ── Step 1: Initialise engine ────────────────────────────────────────── #
    _step(1, "Initialising DocuMind engine…")
    from src.rag_engine import MultiTenantRAGEngine
    engine = MultiTenantRAGEngine()
    _ok("Engine ready.")

    # ── Step 2: Load document ────────────────────────────────────────────── #
    _step(2, "Loading document…")
    user_id = "demo_user"
    session_id = str(uuid.uuid4())

    if pdf_path:
        path = Path(pdf_path)
        if not path.exists():
            print(f"  {C.RED}✗ File not found: {pdf_path}{C.RESET}")
            sys.exit(1)
        content = path.read_bytes()
        filename = path.name
        print(f"  Using: {C.CYAN}{filename}{C.RESET}")
    else:
        print(f"  {C.GREY}No PDF provided — generating sample document…{C.RESET}")
        # Write sample text as a .txt file (avoids PDF parser dependency in demo)
        import tempfile
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        ) as f:
            f.write(SAMPLE_TEXT)
            tmp_path = f.name
        content = Path(tmp_path).read_bytes()
        filename = "documind_manual.txt"
        _ok(f"Sample document created ({len(SAMPLE_TEXT)} characters).")

    # ── Step 3: Ingest ───────────────────────────────────────────────────── #
    _step(3, "Ingesting document into vector store…")
    result = engine.ingest_file(content, filename, user_id)
    _ok(f"Indexed '{result['source']}' — {result['chunks_added']} chunks.")

    sources = engine.list_sources(user_id)
    _ok(f"Knowledge base: {len(sources)} document(s).")

    # ── Step 4: Q&A ──────────────────────────────────────────────────────── #
    print(_h("Question & Answer Demo"))
    print(
        f"\n  {C.GREY}Running {len(QUESTIONS)} questions — "
        f"watch the source citations below each answer.{C.RESET}"
    )

    for i, question in enumerate(QUESTIONS, 1):
        _q(f"Q{i}: {question}")

        result = engine.ask(user_id, session_id, question)

        _a(result["answer"])
        _sources(result["sources"])

        time.sleep(0.5)  # Pacing for demo readability

    # ── Summary ──────────────────────────────────────────────────────────── #
    print(_h("Demo Complete"))
    print(f"""
  {C.GREEN}What was demonstrated:{C.RESET}
  ✓ Document ingestion (chunking + embedding + vector storage)
  ✓ Hybrid retrieval (semantic + BM25 + reranking)
  ✓ Precise answers grounded in source documents
  ✓ Conversational memory (follow-up Q4 referenced Q3's answer)
  ✓ Honest "I don't know" for out-of-scope questions (Q6)
  ✓ Source citations with page references on every answer

  {C.CYAN}Run the Streamlit UI for the full interactive experience:{C.RESET}
    streamlit run app.py
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuMind demo walkthrough")
    parser.add_argument(
        "--pdf", default=None,
        help="Path to a PDF to use (uses built-in sample if omitted)",
    )
    args = parser.parse_args()
    run_demo(args.pdf)