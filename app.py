"""
app.py - PhishingGuard-RAG Streamlit Application
Main entry point: streamlit run app.py
"""
import sys
import time
import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.database     import (init_db, save_query, save_evaluation,
                               get_recent_queries, get_latest_evaluation,
                               get_kb_stats, get_alerts)
from src.knowledge_base import build_knowledge_base
from src.rag_pipeline   import run_pipeline
from src.evaluator      import evaluate_response

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishingGuard-RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  .stApp { background-color: #0d1b2a; color: #e0e6ed; }
  section[data-testid="stSidebar"] { background-color: #112240; }
  section[data-testid="stSidebar"] * { color: #ccd6f6 !important; }

  /* Header */
  .pg-header {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(135deg,#112240,#0d1b2a);
    padding: 16px 24px; border-radius: 12px;
    border-bottom: 2px solid #1e90ff; margin-bottom: 20px;
  }
  .pg-logo { font-size: 2rem; }
  .pg-title { font-size: 1.6rem; font-weight: 700; color: #ccd6f6; }
  .pg-title span { color: #1e90ff; }

  /* Chat bubbles */
  .user-msg {
    background: #1e3a5f; border-radius: 12px 12px 4px 12px;
    padding: 12px 16px; margin: 8px 0; max-width: 85%;
    float: right; clear: both; color: #e0e6ed;
  }
  .ai-msg {
    background: #112240; border-radius: 12px 12px 12px 4px;
    padding: 14px 18px; margin: 8px 0; max-width: 90%;
    float: left; clear: both; color: #ccd6f6;
    border-left: 3px solid #1e90ff;
  }
  .verified-badge {
    display: inline-block; background: #0d3b2a;
    color: #2ecc71; border: 1px solid #2ecc71;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.78rem; margin-top: 8px;
  }
  .fail-badge {
    display: inline-block; background: #3b0d0d;
    color: #e74c3c; border: 1px solid #e74c3c;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.78rem; margin-top: 8px;
  }
  .context-card {
    background: #1a2a4a; border-radius: 8px;
    padding: 10px 14px; margin: 4px 0;
    font-size: 0.82rem; border-left: 3px solid #4a90d9;
    color: #aab8c8;
  }
  .intent-tag {
    background: #1a3a4a; border: 1px solid #4a90d9;
    border-radius: 6px; padding: 6px 12px;
    font-size: 0.82rem; color: #7ec8e3; margin-bottom: 8px;
  }

  /* Metric card */
  .metric-card {
    background: #112240; border-radius: 10px;
    padding: 12px; text-align: center; margin: 4px;
  }
  .metric-label { font-size: 0.78rem; color: #7f8c8d; }
  .metric-value { font-size: 1.5rem; font-weight: 700; }

  /* Alert */
  .alert-high   { border-left: 4px solid #e74c3c; padding: 8px 12px;
                  background:#1a0a0a; border-radius:6px; margin:4px 0; }
  .alert-medium { border-left: 4px solid #f39c12; padding: 8px 12px;
                  background:#1a140a; border-radius:6px; margin:4px 0; }
  .alert-low    { border-left: 4px solid #2ecc71; padding: 8px 12px;
                  background:#0a1a0d; border-radius:6px; margin:4px 0; }

  /* History item */
  .hist-item {
    background: #1a2a40; border-radius: 8px; padding: 8px 12px;
    margin: 4px 0; cursor: pointer; font-size: 0.82rem;
    border-left: 3px solid #2c4a6e; color: #ccd6f6;
  }
  .hist-item:hover { border-left-color: #1e90ff; }

  /* Scrollable chat area */
  .chat-scroll {
    max-height: 60vh; overflow-y: auto;
    padding: 8px; border-radius: 8px;
  }
  div[data-clearfix]::after { content: ""; display: table; clear: both; }

  /* Input row */
  .stTextInput > div > div > input {
    background: #112240 !important; color: #e0e6ed !important;
    border: 1px solid #2c4a6e !important; border-radius: 10px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Initialise ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base...")
def initialise():
    init_db()
    build_knowledge_base()
    return True

initialise()


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_page" not in st.session_state:
    st.session_state.active_page = "Chat"
if "last_eval" not in st.session_state:
    st.session_state.last_eval = get_latest_evaluation()


# ── Helper: gauge chart ───────────────────────────────────────────────────────
def gauge(value: int, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"size": 11, "color": "#aab8c8"}},
        number={"font": {"size": 22, "color": color}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#aab8c8",
                     "tickfont": {"size": 8}},
            "bar":  {"color": color},
            "bgcolor": "#1a2a4a",
            "bordercolor": "#2c4a6e",
            "steps": [
                {"range": [0,  50], "color": "#1a0a0a"},
                {"range": [50, 75], "color": "#1a140a"},
                {"range": [75, 100], "color": "#0a1a0d"},
            ],
        },
    ))
    fig.update_layout(
        height=140, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccd6f6",
    )
    return fig


def score_color(v: int) -> str:
    if v >= 80: return "#2ecc71"
    if v >= 60: return "#f39c12"
    return "#e74c3c"


def score_label(v: int) -> str:
    if v >= 80: return "Excellent"
    if v >= 60: return "Good"
    return "Needs Improvement"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo + title
    st.markdown("""
    <div class="pg-header">
      <span class="pg-logo">🛡️</span>
      <span class="pg-title">PhishingGuard<span>-RAG</span></span>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    pages = ["💬 Chat", "📚 Knowledge Base", "📊 Reports"]
    page  = st.radio("Navigation", pages, label_visibility="collapsed")
    st.session_state.active_page = page.split(" ", 1)[1]

    st.divider()

    # Recent queries
    st.markdown("**🕒 Recent Queries**")
    recent = get_recent_queries(8)
    example_qs = [
        "How to identify phishing links in emails?",
        "What are common phishing techniques?",
        "How to detect spear-phishing?",
        "Explain BEC attack and prevention.",
        "What indicators suggest a phishing page?",
        "Best practices for phishing awareness?",
        "How does URL obfuscation work?",
        "How to report phishing emails?",
    ]
    all_items = (
        [(r[0], r[1], str(r[2])[:10]) for r in recent]
        + [(None, q, "Demo") for q in example_qs]
    )[:8]

    for _, q, ts in all_items:
        short = (q[:40] + "…") if len(q) > 40 else q
        if st.button(f"📧 {short}", key=f"hist_{q[:20]}", use_container_width=True):
            st.session_state.prefill = q

    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Example prompts
    st.markdown("**💡 Example prompts**")
    ex_prompts = [
        "How to detect spear-phishing?",
        "What are phishing URL red flags?",
        "How to prevent BEC attacks?",
    ]
    for ep in ex_prompts:
        if st.button(f"↗ {ep}", key=f"ex_{ep[:15]}", use_container_width=True):
            st.session_state.prefill = ep


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
active = st.session_state.active_page

# ── CHAT PAGE ─────────────────────────────────────────────────────────────────
if active == "Chat":
    col_chat, col_right = st.columns([3, 1.2])

    # ── Chat column ──────────────────────────────────────────────────────────
    with col_chat:
        st.markdown("""
        <div class="pg-header">
          <span class="pg-logo">🛡️</span>
          <span class="pg-title">PhishingGuard<span>-RAG</span> AI Assistant</span>
        </div>
        """, unsafe_allow_html=True)

        # Render chat history
        chat_html = '<div class="chat-scroll"><div data-clearfix>'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f'<div class="user-msg">👤 {msg["content"]}</div>'
            else:
                badge = (
                    '<span class="verified-badge">✅ Ontology Verified</span>'
                    if msg.get("verified")
                    else '<span class="fail-badge">⚠️ Not Verified</span>'
                )
                conf  = msg.get("confidence", 0)
                intent_html = ""
                if msg.get("intent"):
                    intent_html = (
                        f'<div class="intent-tag">🎯 <b>Intent:</b> '
                        f'{msg["intent"][:120]}</div>'
                    )
                ctx_html = ""
                for i, c in enumerate(msg.get("contexts", [])[:2]):
                    ctx_html += (
                        f'<div class="context-card">'
                        f'[{i+1}] {c[:200]}…</div>'
                    )
                chat_html += (
                    f'<div class="ai-msg">'
                    f'{intent_html}'
                    f'<b>PhishingGuard-RAG AI</b><br>'
                    f'{msg["content"]}<br><br>'
                    f'{ctx_html}'
                    f'{badge} &nbsp; Confidence: {conf:.0%}'
                    f'</div>'
                )
        chat_html += '</div></div>'
        st.markdown(chat_html, unsafe_allow_html=True)

        # Input
        st.markdown("---")
        prefill = st.session_state.pop("prefill", "")
        with st.form("chat_form", clear_on_submit=True):
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                user_input = st.text_input(
                    "query",
                    value=prefill,
                    placeholder="Ask a question about phishing threats…",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("Send ✈️",
                                                  use_container_width=True)

        if submitted and user_input.strip():
            query = user_input.strip()
            st.session_state.messages.append({"role": "user",
                                               "content": query})

            with st.spinner("🔍 Analysing query…"):
                result = run_pipeline(query)

            with st.spinner("📊 Evaluating with RAGAS…"):
                scores = evaluate_response(
                    query, result["answer"], result["contexts"]
                )

            # Persist to DB
            qid = save_query(
                query, result["intent"], result["answer"],
                result["contexts"], result["ontology_verified"],
                result["confidence_score"], result["ontology_reasoning"],
            )
            save_evaluation(qid, **scores)

            # Update eval panel
            st.session_state.last_eval = {
                "faithfulness":      round(scores["faithfulness"] * 100),
                "answer_relevance":  round(scores["answer_relevance"] * 100),
                "context_relevance": round(scores["context_relevance"] * 100),
                "context_recall":    round(scores["context_recall"] * 100),
                "overall_score":     round(
                    sum(scores.values()) / 4 * 100, 1),
            }

            st.session_state.messages.append({
                "role":      "assistant",
                "content":   result["answer"],
                "intent":    result["intent"],
                "contexts":  result["contexts"],
                "verified":  result["ontology_verified"],
                "confidence": result["confidence_score"],
            })
            st.rerun()

    # ── Right panel ───────────────────────────────────────────────────────────
    with col_right:
        ev = st.session_state.last_eval

        st.markdown("### 📊 RAGAS Evaluation")
        overall = ev.get("overall_score", 0)
        color_o = score_color(int(overall))
        st.markdown(
            f"**Overall Score: "
            f"<span style='color:{color_o};font-size:1.3rem'>"
            f"{overall}/100</span>**",
            unsafe_allow_html=True,
        )

        metrics = [
            ("faithfulness",      "Faithfulness",
             "Factual consistency with context"),
            ("answer_relevance",  "Answer Relevance",
             "Addresses the question"),
            ("context_relevance", "Context Relevance",
             "Retrieved context quality"),
            ("context_recall",    "Context Recall",
             "Coverage of relevant info"),
        ]
        for key, label, desc in metrics:
            val = ev.get(key, 0)
            col = score_color(val)
            lbl = score_label(val)
            st.plotly_chart(gauge(val, label, col),
                            use_container_width=True, key=f"g_{key}")
            st.markdown(
                f"<small style='color:#7f8c8d'>{desc}</small> "
                f"<span style='color:{col};font-size:0.75rem'>{lbl}</span>",
                unsafe_allow_html=True,
            )

        st.divider()

        # KB Stats
        kb = get_kb_stats()
        st.markdown("### 📚 Knowledge Base")
        c1, c2 = st.columns(2)
        c1.metric("Documents", kb["documents"])
        c2.metric("Chunks",    kb["chunks"])
        c1.metric("Embeddings", kb["embeddings"])
        c2.metric("Queries",   kb["total_queries"])

        st.divider()

        # Alerts
        st.markdown("### 🚨 Security Alerts")
        alerts = get_alerts(4)
        for title, desc, sev, ts in alerts:
            css = f"alert-{sev}"
            st.markdown(
                f'<div class="{css}"><b>{title}</b><br>'
                f'<small style="color:#aaa">{desc}</small></div>',
                unsafe_allow_html=True,
            )


# ── KNOWLEDGE BASE PAGE ───────────────────────────────────────────────────────
elif active == "Knowledge Base":
    st.markdown("## 📚 Knowledge Base")
    st.markdown(
        "Documents are crawled from NIST and APWG, chunked into 512-token "
        "segments with 64-token overlap, and indexed in FAISS using "
        "**BAAI/bge-large-en-v1.5** embeddings."
    )

    from src.knowledge_base import PHISHING_KNOWLEDGE
    for doc in PHISHING_KNOWLEDGE:
        with st.expander(f"📄 {doc['title']} — *{doc['source']}*"):
            st.markdown(doc["content"])

    st.divider()
    if st.button("🔄 Rebuild Knowledge Base"):
        from src.knowledge_base import build_knowledge_base
        with st.spinner("Rebuilding…"):
            build_knowledge_base(force_rebuild=True)
        st.success("Knowledge base rebuilt successfully!")


# ── REPORTS PAGE ──────────────────────────────────────────────────────────────
elif active == "Reports":
    st.markdown("## 📊 Evaluation Reports")

    import sqlite3, pandas as pd
    from src.database import DB_PATH

    conn = sqlite3.connect(DB_PATH)

    try:
        df_eval = pd.read_sql_query("""
            SELECT q.question,
                   e.faithfulness, e.answer_relevance,
                   e.context_relevance, e.context_recall,
                   e.overall_score, e.timestamp
            FROM evaluations e
            JOIN queries q ON q.id = e.query_id
            ORDER BY e.timestamp DESC
            LIMIT 20
        """, conn)
    except Exception:
        df_eval = pd.DataFrame()
    conn.close()

    if df_eval.empty:
        st.info("No evaluation data yet. Start chatting to generate results!")
    else:
        st.markdown("### Recent RAGAS Scores")
        avg = df_eval[["faithfulness", "answer_relevance",
                        "context_relevance", "context_recall"]].mean()

        cols = st.columns(4)
        labels = ["Faithfulness", "Ans. Relevance",
                  "Ctx. Relevance", "Ctx. Recall"]
        keys   = ["faithfulness", "answer_relevance",
                  "context_relevance", "context_recall"]
        for col, lbl, key in zip(cols, labels, keys):
            val = round(avg[key] * 100, 1)
            col.metric(lbl, f"{val}%")

        # Trend chart
        import plotly.express as px
        df_plot = df_eval.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        fig = px.line(
            df_plot, x="timestamp",
            y=["faithfulness", "answer_relevance",
               "context_relevance", "context_recall"],
            title="RAGAS Score Trends",
            color_discrete_sequence=["#1e90ff", "#2ecc71",
                                      "#f39c12", "#e74c3c"],
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#112240")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Query History")
        st.dataframe(
            df_eval[["question", "faithfulness", "answer_relevance",
                     "context_relevance", "context_recall",
                     "overall_score", "timestamp"]],
            use_container_width=True,
        )

        # Export
        csv = df_eval.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv,
                           "phishingguard_results.csv", "text/csv")
