"""
Features:
  - Multi-document upload (PDF, DOCX, TXT, HTML, Markdown)
  - Real-time chat with conversational memory
  - Source citation panel with page references
  - Document management (list + delete)
  - Session reset
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import streamlit as st
from streamlit_local_storage import LocalStorage
from src.config import get_settings
from src.rag_engine import MultiTenantRAGEngine

logger = logging.getLogger(__name__)
settings = get_settings()

#  Page config                                                                  

st.set_page_config(
    page_title="DocuRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────── #
#  Custom CSS                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:        #0d0f12;
    --surface:   #14171c;
    --surface2:  #1c2028;
    --border:    #2a2f3a;
    --accent:    #4ade80;
    --accent2:   #22d3ee;
    --text:      #e8eaed;
    --muted:     #6b7280;
    --user-bg:   #1a2535;
    --bot-bg:    #131a14;
    --danger:    #f87171;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Header */
  .dm-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 0 0 24px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
  }
  .dm-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--accent);
    letter-spacing: -0.02em;
  }
  .dm-tagline {
    font-size: 0.8rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Chat messages */
  .msg-user {
    background: var(--user-bg);
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 15%;
    font-size: 0.95rem;
    line-height: 1.6;
  }
  .msg-bot {
    background: var(--bot-bg);
    border: 1px solid #1e3a1e;
    border-left: 3px solid var(--accent);
    border-radius: 4px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 5%;
    font-size: 0.95rem;
    line-height: 1.7;
  }
  .msg-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
  }
  .msg-label.user { color: var(--accent2); }
  .msg-label.bot  { color: var(--accent); }

  /* Source citations */
  .source-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.82rem;
  }
  .source-file {
    font-family: 'DM Mono', monospace;
    color: var(--accent2);
    font-size: 0.78rem;
  }
  .source-excerpt {
    color: var(--muted);
    margin-top: 4px;
    font-style: italic;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 12px;
  }

  /* Doc chip */
  .doc-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
  }

  /* Status badge */
  .status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .badge-ready  { background: #14291a; color: var(--accent); border: 1px solid #1e3a1e; }
  .badge-empty  { background: #1f1a0e; color: #fbbf24; border: 1px solid #3a2f1e; }

  /* Input area */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
  }

  /* Buttons */
  .stButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
  }
  .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
  }

  /* Expander */
  details {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
  }
  summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
  }

  /* Hide Streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header[data-testid="stHeader"] { 
    background: transparent !important; 
  }
  .viewerBadge_container__1QSob { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────── #
#  Session state init                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

local_storage = LocalStorage()

def init_persistent_state(engine_instance=None):
    # Standard state defaults
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        
    # Persistent User ID logic
    stored_uid = local_storage.getItem("documind_user_id")
    if stored_uid:
        st.session_state.user_id = stored_uid
    else:
        new_uid = f"user_{uuid.uuid4().hex[:8]}"
        st.session_state.user_id = new_uid
        local_storage.setItem("documind_user_id", new_uid)
    
    # Sync documents from DB if engine is ready
    if engine_instance:
        try:
            st.session_state.uploaded_docs = engine_instance.list_sources(
                st.session_state.user_id
            )
        except Exception as e:
            logger.error(f"Sync failed: {e}")

@st.cache_resource(show_spinner="Initialising DocuRAG engine…")
def load_engine() -> MultiTenantRAGEngine:
    return MultiTenantRAGEngine()


engine = load_engine()

init_persistent_state(engine)

# ──────────────────────────────────────────────────────────────────────────── #
#  Sidebar                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.markdown("### 📂 Knowledge Base")
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "html"],
        accept_multiple_files=True,
        help="Supports PDF, Word, TXT, Markdown, HTML",
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_docs:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        result = engine.ingest_file(
                            uf.read(), uf.name,
                            st.session_state.user_id
                        )
                        st.session_state.uploaded_docs.append(uf.name)
                        st.success(
                            f"✓ {uf.name}  "
                            f"({result['chunks_added']} chunks)"
                        )
                    except Exception as e:
                        st.error(f"✗ {uf.name}: {e}")

    # Loaded documents list
    sources = engine.list_sources(st.session_state.user_id)
    if sources:
        st.markdown("### 📄 Loaded Documents")
        for src in sources:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f'<div class="doc-chip">📎 {src}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                    engine.delete_source(st.session_state.user_id, src)
                    if src in st.session_state.uploaded_docs:
                        st.session_state.uploaded_docs.remove(src)
                    st.rerun()

        chunk_count = engine.document_count(st.session_state.user_id)
        st.markdown(
            f'<span class="status-badge badge-ready">'
            f'● {len(sources)} doc{"s" if len(sources) > 1 else ""} · {chunk_count} chunks</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge badge-empty">⚠ No documents loaded</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### ⚙️ Session")

    if st.button("🗑 Clear conversation", use_container_width=True):
        engine.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown(
        f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
        f'color:var(--muted,#6b7280);margin-top:8px;">'
        f'session: {st.session_state.session_id[:8]}…</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────── #
#  Main chat area                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

# Header
st.markdown("""
<div class="dm-header">
  <div class="dm-logo">🧠 DocuRAG</div>
  <div class="dm-tagline">RAG · Document Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div class="msg-bot">
      <div class="msg-label bot">DocuRAG</div>
      Hello! I'm DocuRAG — upload your documents in the sidebar and ask me anything about them.
      I'll answer with precise citations from your sources.
      <br><br>
      <b>Try asking:</b><br>
      • "Summarise the key points of [document name]"<br>
      • "What does the document say about X?"<br>
      • "Compare the sections on A and B"
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">'
            f'<div class="msg-label user">You</div>'
            f'{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="msg-label bot">DocuRAG</div>'
            f'{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show source citations if available
        if msg.get("sources"):
            with st.expander(
                f"📚 {len(msg['sources'])} source(s) retrieved", expanded=False
            ):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-file">'
                        f'📄 {src["source"]}  |  Page {src["page"]}'
                        f'</div>'
                        f'<div class="source-excerpt">{src["excerpt"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ──────────────────────────────────────────────────────────────────────────── #
#  Input                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question…",
            placeholder="e.g. What are the main conclusions of the report?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("→", use_container_width=True)

if submitted and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Query engine
    with st.spinner("Searching documents…"):
        try:
            result = engine.ask(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                question=user_input,
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}",
                "sources": [],
            })

    st.rerun()