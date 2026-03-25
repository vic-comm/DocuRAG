"""
cli.py
───────
DocuMind command-line interface.
Useful for batch ingestion and quick testing without starting the UI.

Usage:
  python cli.py ingest ./data/             # Ingest all docs in a directory
  python cli.py ingest report.pdf          # Ingest a single file
  python cli.py ask "What is X?"           # Ask a one-shot question
  python cli.py chat                       # Start an interactive chat session
  python cli.py sources                    # List indexed documents
  python cli.py clear                      # Clear all documents for default user
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def _print_sources(sources: list[dict]) -> None:
    if not sources:
        return
    print(f"\n  {'─' * 60}")
    print(f"  📚 Sources ({len(sources)} retrieved)")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s['source']}  |  Page {s['page']}")
        print(f"      {s['excerpt'][:120]}…")
    print(f"  {'─' * 60}")


def _get_engine():
    from src.rag_engine import MultiTenantRAGEngine
    return MultiTenantRAGEngine()


# ──────────────────────────────────────────────────────────────────────────── #
#  Commands                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def cmd_ingest(args):
    """Ingest a file or directory."""
    engine = _get_engine()
    path = Path(args.path)
    user_id = args.user or "cli_user"

    if path.is_dir():
        from src.dataloader import load_directory
        print(f"📂 Loading all documents from '{path}'…")
        chunks = load_directory(str(path))
        if not chunks:
            print("⚠️  No supported documents found.")
            return
        count = engine.vs_manager.add_documents(chunks, user_id)
        engine.retriever_factory.invalidate(user_id)
        print(f"✓ Ingested {count} chunks from {len(set(c.metadata['source'] for c in chunks))} files.")

    elif path.is_file():
        print(f"📄 Ingesting '{path.name}'…")
        content = path.read_bytes()
        result = engine.ingest_file(content, path.name, user_id)
        print(f"✓ Ingested '{result['source']}' — {result['chunks_added']} chunks.")

    else:
        print(f"✗ Path not found: {path}")
        sys.exit(1)


def cmd_ask(args):
    """Ask a single question."""
    engine = _get_engine()
    user_id = args.user or "cli_user"
    session_id = str(uuid.uuid4())

    print(f"\n❓ {args.question}\n")
    result = engine.ask(user_id, session_id, args.question)
    print(f"🧠 {result['answer']}")
    _print_sources(result["sources"])


def cmd_chat(args):
    """Interactive multi-turn chat session."""
    engine = _get_engine()
    user_id = args.user or "cli_user"
    session_id = str(uuid.uuid4())

    print("\n" + "═" * 64)
    print("  🧠 DocuMind — Interactive Chat")
    print(f"  User: {user_id}  |  Session: {session_id[:8]}…")
    print("  Type 'exit' or Ctrl+C to quit, 'clear' to reset memory.")
    print("═" * 64 + "\n")

    while True:
        try:
            question = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            engine.clear_session(session_id)
            session_id = str(uuid.uuid4())
            print(f"  ✓ Session cleared. New session: {session_id[:8]}…\n")
            continue

        result = engine.ask(user_id, session_id, question)
        print(f"\nDocuMind > {result['answer']}")
        _print_sources(result["sources"])
        print()


def cmd_sources(args):
    """List all indexed documents for a user."""
    engine = _get_engine()
    user_id = args.user or "cli_user"
    sources = engine.list_sources(user_id)
    count = engine.document_count(user_id)

    if not sources:
        print(f"No documents indexed for user '{user_id}'.")
        return

    print(f"\n📚 Knowledge Base — {user_id}")
    print(f"   {count} total chunks across {len(sources)} document(s)\n")
    for src in sources:
        print(f"   📄 {src}")
    print()


def cmd_clear(args):
    """Delete all indexed documents for a user."""
    engine = _get_engine()
    user_id = args.user or "cli_user"
    confirm = input(
        f"⚠️  Delete ALL documents for user '{user_id}'? [y/N] "
    ).strip().lower()
    if confirm == "y":
        engine.vs_manager.delete_user_documents(user_id)
        engine.retriever_factory.invalidate(user_id)
        print("✓ Knowledge base cleared.")
    else:
        print("Cancelled.")


# ──────────────────────────────────────────────────────────────────────────── #
#  Entry point                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        prog="documind",
        description="DocuMind CLI — RAG Document Intelligence",
    )
    parser.add_argument(
        "--user", "-u",
        default="cli_user",
        help="User ID (default: cli_user)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a file or directory")
    p_ingest.add_argument("path", help="File or directory path")
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a one-shot question")
    p_ask.add_argument("question", help="Your question")
    p_ask.set_defaults(func=cmd_ask)

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive chat session")
    p_chat.set_defaults(func=cmd_chat)

    # sources
    p_sources = subparsers.add_parser("sources", help="List indexed documents")
    p_sources.set_defaults(func=cmd_sources)

    # clear
    p_clear = subparsers.add_parser("clear", help="Clear knowledge base")
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()