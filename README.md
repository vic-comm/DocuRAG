# 🧠 DocuRAG
**Production-Grade RAG Document Intelligence**

DocuRAG lets you upload any document (PDF, Word, TXT, Markdown, HTML) and have a precise, grounded conversation with it. Every answer is backed by citations from your source material — no hallucinations, no general knowledge, strictly your documents.

🔗 **Live Demo:** [docurag-grxzmrohpnhxnd2znmrkqq.streamlit.app](https://docurag-grxzmrohpnhxnd2znmrkqq.streamlit.app)
📁 **GitHub:** [github.com/vic-comm/DocuRAG](https://github.com/vic-comm/DocuRAG)

---

## Architecture

```
User Query
    │
    ▼
ConversationMemory (Summary Buffer)   ←── per-session, isolated history
    │
    ▼
RetrieverFactory
    ├── Stage 1: MultiQueryRetriever  (LLM rewrites query into 3 variants)
    ├── Stage 2: EnsembleRetriever   (Semantic Pinecone + BM25 hybrid search)
    ├── Stage 3: FlashrankReranker   (Cross-encoder selects top-N chunks)
    └── [Cached per user — invalidated on new uploads]
    │
    ▼
ChatPromptTemplate                    ←── system + history + context + question
    │
    ▼
LLM (Groq llama-3.3-70b-versatile)
    │
    ▼
Answer + Source Citations             ←── document name, page number, excerpt
```

---

## Key Components

| File | Responsibility |
|------|---------------|
| `src/config.py` | Centralised settings — reads from `.env` locally, Streamlit secrets in production |
| `src/dataloader.py` | Multi-format document parsing + semantic chunking with metadata |
| `src/vector_store.py` | Pinecone (production) + ChromaDB (local) with multi-tenant isolation |
| `src/retriever.py` | 4-stage retrieval pipeline: rewrite → hybrid search → rerank (cached per user) |
| `src/conversation_history.py` | Summary buffer memory — keeps last k messages, summarises older ones |
| `src/rag_engine.py` | Orchestrates all components into a single clean interface |
| `app.py` | Streamlit chat UI with document management and source citation panel |
| `main.py` | FastAPI REST backend |
| `evaluate.py` | RAGAS evaluation framework — faithfulness, relevancy, precision, recall |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/vic-comm/DocuRAG.git
cd DocuRAG
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```bash
# Required
GROQ_API_KEY=your_groq_api_key        # Free at console.groq.com
GOOGLE_API_KEY=your_google_api_key    # For Gemini embeddings

# For production (Pinecone)
PINECONE_API_KEY=your_pinecone_key    # Free tier at pinecone.io
PINECONE_INDEX_NAME=documind
USE_PINECONE=true
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

Visit `http://localhost:8501`, upload a document, and start chatting.

### 4. Run the API (optional)

```bash
python main.py
# Interactive API docs at http://localhost:8000/docs
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | Groq `llama-3.3-70b-versatile` | Answer generation + query rewriting |
| Embeddings | Google `gemini-embedding-001` (768-dim) | Semantic vector representation |
| Vector Store | Pinecone (prod) / ChromaDB (local) | Persistent similarity search |
| Keyword Search | BM25 | Exact match retrieval |
| Reranking | FlashRank cross-encoder | Precision scoring of retrieved chunks |
| Orchestration | LangChain | Pipeline chaining and memory |
| Backend | FastAPI | REST API |
| Frontend | Streamlit | Chat UI |
| Evaluation | RAGAS | Retrieval + answer quality metrics |

---

## Retrieval Pipeline — Why 4 Stages?

| Stage | What it does | Why it matters |
|-------|-------------|---------------|
| **Query rewriting** | LLM generates 3 query variants | Catches documents using different terminology than the user |
| **Hybrid search** | Semantic (Pinecone) + keyword (BM25) via Reciprocal Rank Fusion | BM25 catches exact matches semantic search misses; semantic catches meaning keyword search misses |
| **Cross-encoder reranking** | FlashRank scores each chunk against the query jointly | Dramatically improves precision — picks the 5 best from the top 20 |
| **Summary buffer memory** | Keeps last k messages in full, summarises older ones | Handles long multi-turn conversations without exceeding context window |

---

## Evaluation Framework

DocuRAG includes a RAGAS-powered evaluation module that measures retrieval and answer quality scientifically.

```bash
# Run evaluation with built-in questions
python evaluate.py --user your_user_id

# Run with your own testset
python evaluate.py --user your_user_id --testset tests/eval_questions.yaml

# Save report to CSV
python evaluate.py --user your_user_id --output reports/eval.csv
```

### Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Is the answer grounded in retrieved context? | > 0.85 |
| **Answer Relevancy** | Does the answer address the question? | > 0.80 |
| **Context Precision** | Are retrieved chunks actually relevant? | > 0.75 |
| **Context Recall** | Were all relevant chunks retrieved? (needs ground truth) | > 0.70 |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Upload and index a document (async) |
| `POST` | `/query` | Ask a question, receive answer + citations |
| `GET` | `/sources` | List uploaded documents for current user |
| `DELETE` | `/sources/{name}` | Remove a document from the knowledge base |
| `POST` | `/session/clear` | Clear conversation history for a session |

### Example: Query via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "session_id": "my-session"}'
```

Response:
```json
{
  "answer": "The main findings include...",
  "sources": [
    {
      "source": "report.pdf",
      "page": 4,
      "excerpt": "...relevant excerpt from the document..."
    }
  ],
  "session_id": "my-session"
}
```

---

## Running Tests

```bash
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Deployment

### Streamlit Cloud (live demo)

The app is deployed at [docurag-grxzmrohpnhxnd2znmrkqq.streamlit.app](https://docurag-grxzmrohpnhxnd2znmrkqq.streamlit.app).

To deploy your own instance:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your fork
3. Set `app.py` as the main file
4. Add your secrets in **Advanced Settings**:
```toml
GROQ_API_KEY = "..."
GOOGLE_API_KEY = "..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "documind"
USE_PINECONE = "true"
```

---

## What's Next — OmniSupport

DocuRAG is the foundation. The next project, **OmniSupport**, extends this into a fully production-hardened system:

- [ ] Async document ingestion via Celery + Redis task queue
- [ ] PostgreSQL full-text search for persistent, scalable BM25
- [ ] Fine-tuning pipeline for domain-specific embedding adaptation
- [ ] Streaming responses via WebSocket
- [ ] Prometheus + Grafana monitoring with latency percentile tracking
- [ ] Real JWT authentication (Auth0 / Supabase)
- [ ] Citation verification — programmatic hallucination detection