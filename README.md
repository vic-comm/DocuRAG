# 🧠 DocuMind

**RAG-powered Document Intelligence**

DocuMind lets you upload any document (PDF, Word, TXT, Markdown, HTML) and have a precise, grounded conversation with it. Every answer is backed by citations from your source material — no hallucinations.

---

## Architecture

```
User Query
    │
    ▼
ConversationMemory (Summary Buffer)
    │
    ▼
RetrieverFactory
    ├── Stage 1: MultiQueryRetriever  (LLM rewrites query into 3 variants)
    ├── Stage 2: EnsembleRetriever   (Semantic + BM25 hybrid search)
    └── Stage 3: FlashrankReranker   (Cross-encoder selects top-5 chunks)
    │
    ▼
LLM (Groq / OpenAI)
    │
    ▼
Answer + Source Citations
```

### Key Components

| File | Responsibility |
|------|---------------|
| `src/config.py` | Centralised settings from `.env` |
| `src/dataloader.py` | File parsing + chunking |
| `src/vector_store.py` | ChromaDB wrapper with multi-tenant isolation |
| `src/retriever.py` | 4-stage retrieval pipeline with caching |
| `src/conversation_history.py` | Summary buffer memory |
| `src/rag_engine.py` | Orchestrates everything |
| `main.py` | FastAPI REST API |
| `app.py` | Streamlit chat UI |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourname/documind.git
cd documind
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

Visit `http://localhost:8501`, upload a PDF, and start chatting.

### 4. Run the API (optional)

```bash
python main.py
# API docs at http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/ingest` | Upload and index a document |
| `POST` | `/query`  | Ask a question |
| `GET`  | `/sources` | List uploaded documents |
| `DELETE` | `/sources/{name}` | Remove a document |
| `POST` | `/session/clear` | Clear conversation history |

### Example: Query via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "session_id": "my-session"}'
```

---

## Tech Stack

- **LangChain** — orchestration framework
- **Groq / OpenAI** — LLM for generation and query rewriting
- **ChromaDB** — persistent vector store
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` (free, local)
- **BM25** — keyword search for hybrid retrieval
- **FlashRank** — cross-encoder reranking
- **FastAPI** — REST backend
- **Streamlit** — chat UI

---

## Retrieval Pipeline Detail

### Why 4 stages?

| Stage | Purpose | Why it matters |
|-------|---------|---------------|
| Multi-query rewriting | Generates 3 query variants | Catches documents that use different terminology |
| Hybrid search (Semantic + BM25) | Dense + sparse retrieval | BM25 catches exact keyword matches semantic search misses |
| Cross-encoder reranking | Scores doc-query relevance | Dramatically improves precision over bi-encoder alone |
| Summary buffer memory | Maintains conversation context | Handles follow-up questions naturally |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Next Steps (Production Hardening)

- [ ] Replace mock auth with real JWT (Auth0, Supabase)
- [ ] Migrate ChromaDB to Pinecone for scale
- [ ] Add streaming responses (`/query/stream`)
- [ ] Add document re-ingestion (update without full delete)
- [ ] Deploy FastAPI to Railway / Render
- [ ] Deploy Streamlit to Streamlit Cloud