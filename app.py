"""
app.py - PhishingGuard-RAG Streamlit Application
Main entry point: streamlit run app.py
"""
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.database     import (init_db, save_query, save_evaluation,
                               get_recent_queries, get_latest_evaluation,
                               get_kb_stats, get_query_by_id)
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
/* ── Base ───────────────────────────────────────────────────── */
:root {
  --bg: #071629;
  --bg-soft: #0b1f35;
  --panel: #0d2239;
  --panel-2: #112843;
  --panel-3: #143252;
  --line: rgba(120, 166, 209, 0.18);
  --text: #e7f1fb;
  --muted: #8aa3be;
  --teal: #2fd0c3;
  --teal-strong: #23b8b0;
  --blue: #2f8fff;
  --green: #53d17d;
  --orange: #ffad42;
  --danger: #ff7b72;
  --shadow: 0 20px 48px rgba(0, 0, 0, 0.28);
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(47, 208, 195, 0.10), transparent 26%),
    radial-gradient(circle at top right, rgba(47, 143, 255, 0.10), transparent 22%),
    linear-gradient(180deg, #081a2d 0%, #061322 100%);
  color: var(--text);
  font-family: 'Segoe UI', 'SF Pro Display', system-ui, sans-serif;
}

[data-testid="stHeader"] {
  background: transparent !important;
  height: 0 !important;
  border: none !important;
}
[data-testid="stToolbar"] {
  right: 1rem !important;
  top: 0.45rem !important;
}
[data-testid="stDecoration"] {
  display: none !important;
}
[data-testid="stStatusWidget"] {
  display: none !important;
}
[data-testid="stAppViewContainer"] > .main {
  padding-top: 0 !important;
}

[data-testid="stAppViewContainer"] {
  background: transparent;
}

[data-testid="stMainBlockContainer"],
.block-container {
  padding-top: 0 !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  padding-bottom: 6.75rem !important;
}

[data-testid="stMainBlockContainer"] > div:first-child,
.block-container > div:first-child {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(11, 31, 53, 0.98), rgba(8, 23, 39, 0.98)) !important;
  border-right: 1px solid var(--line) !important;
  min-width: 18rem !important;
  max-width: 18rem !important;
}
section[data-testid="stSidebar"] > div {
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding-top: 0.45rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div {
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] .block-container {
  padding-top: 0.35rem !important;
}
section[data-testid="stSidebar"] > div > div {
  padding-top: 0.35rem !important;
}
section[data-testid="stSidebar"] * { color: #d9e7f6 !important; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
  gap: 0.35rem !important;
}
section[data-testid="stSidebar"] div[data-testid="stElementContainer"] {
  margin-bottom: 0.18rem !important;
}
section[data-testid="stSidebar"] .stMarkdown {
  margin-bottom: 0 !important;
}

/* Nav radio */
section[data-testid="stSidebar"] .stRadio input[type="radio"] {
  display: none !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
  display: grid !important;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}
section[data-testid="stSidebar"] .stRadio label {
  background: rgba(255,255,255,0.02) !important;
  border-radius: 14px !important;
  padding: 12px 12px !important;
  border: 1px solid rgba(255,255,255,0.05) !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
  cursor: pointer;
  transition: all 0.18s ease;
  display: flex !important;
  align-items: center;
  justify-content: center;
  min-height: 54px;
  text-align: center;
}
section[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(47,208,195,0.08) !important;
  border-color: rgba(47,208,195,0.22) !important;
  color: #f4fbff !important;
}

/* Sidebar buttons — uniform, no red focus ring */
section[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(180deg, rgba(17,40,67,0.98), rgba(11,31,53,0.98)) !important;
  border: 1px solid var(--line) !important;
  border-radius: 13px !important;
  color: #d7e8f7 !important;
  font-size: 0.72rem !important;
  font-weight: 560 !important;
  text-align: left !important;
  transition: all 0.18s ease !important;
  box-shadow: none !important;
  min-height: 42px !important;
  padding: 0.56rem 0.72rem !important;
  white-space: normal !important;
  line-height: 1.28 !important;
}
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"] p {
  font-size: inherit !important;
  font-weight: inherit !important;
  line-height: inherit !important;
  margin: 0 !important;
}
section[data-testid="stSidebar"] .recent-query-action button {
  min-height: 44px !important;
  padding: 0.65rem 0.9rem !important;
  border-radius: 18px !important;
  font-size: 0.84rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: linear-gradient(180deg, rgba(20,50,82,1), rgba(13,34,57,1)) !important;
  border-color: rgba(47,208,195,0.34) !important;
  color: white !important;
  transform: translateY(-1px) !important;
}
section[data-testid="stSidebar"] .stButton > button:focus,
section[data-testid="stSidebar"] .stButton > button:active {
  border-color: rgba(47,208,195,0.34) !important;
  box-shadow: none !important;
  outline: none !important;
  color: white !important;
  background: linear-gradient(180deg, rgba(20,50,82,1), rgba(13,34,57,1)) !important;
}

/* ── Header ──────────────────────────────────────────────────── */
.pg-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  background: linear-gradient(180deg, rgba(13,34,57,0.96), rgba(11,31,53,0.96));
  padding: 18px 22px;
  border-radius: 20px;
  border: 1px solid var(--line);
  margin-bottom: 18px;
  box-shadow: var(--shadow);
}
.pg-header-left {
  display: flex;
  align-items: center;
  gap: 14px;
}
.pg-logo {
  width: 46px;
  height: 46px;
  border-radius: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.45rem;
  line-height: 1;
  background: linear-gradient(180deg, rgba(47,208,195,0.14), rgba(47,143,255,0.08));
  border: 1px solid rgba(47,208,195,0.28);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
}
.pg-title-wrap { display: flex; flex-direction: column; gap: 2px; }
.pg-title {
  font-size: 1.55rem;
  font-weight: 800;
  color: #f7fbff;
  letter-spacing: -0.03em;
}
.pg-title span { color: var(--teal); }
.pg-subtitle {
  font-size: 0.9rem;
  color: var(--muted);
}
.pg-top-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.pg-chip {
  padding: 9px 14px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.02);
  color: var(--muted);
  font-size: 0.82rem;
}
.pg-chip.active {
  background: rgba(47,208,195,0.10);
  color: #ebfffd;
  border-color: rgba(47,208,195,0.28);
}
.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 9px;
  margin: 0 0 6px;
  padding: 0 0.1rem 0.18rem;
}
.sidebar-brand-mark {
  width: 2.2rem;
  height: 2.2rem;
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.08rem;
  background: linear-gradient(180deg, rgba(47,208,195,0.18), rgba(47,143,255,0.10));
  border: 1px solid rgba(47,208,195,0.24);
}
.sidebar-brand-copy {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.sidebar-brand-title {
  color: #f5fbff;
  font-size: 1.08rem;
  font-weight: 760;
  letter-spacing: -0.02em;
}
.sidebar-brand-subtitle {
  color: #8aa3be;
  font-size: 0.74rem;
}
.sidebar-section {
  margin-top: 0.62rem;
}
.sidebar-new-chat-heading {
  margin-top: 0.78rem;
}
.sidebar-new-chat-heading .sidebar-heading {
  margin-bottom: 10px;
}
.sidebar-new-chat-gap {
  height: 0.42rem;
}
.sidebar-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.35rem;
  padding: 0 0.15rem;
}
.sidebar-heading-title {
  color: #b9cce0;
  font-size: 0.8rem;
  font-weight: 780;
  letter-spacing: 0.075em;
  text-transform: uppercase;
}
.sidebar-heading-meta {
  color: #7894b0;
  font-size: 0.7rem;
}
.sidebar-divider {
  height: 1px;
  margin: 0.55rem 0 0.1rem;
  background: linear-gradient(90deg, rgba(120,166,209,0.14), rgba(120,166,209,0));
}
.sidebar-empty {
  padding: 0.55rem 0.15rem 0.25rem;
  color: #7d98b4;
  font-size: 0.84rem;
  line-height: 1.5;
}
.sidebar-nav-button button {
  min-height: 38px !important;
  padding: 0.48rem 0.68rem !important;
  border-radius: 13px !important;
  background: rgba(255,255,255,0.025) !important;
  border: 1px solid rgba(120,166,209,0.10) !important;
  box-shadow: none !important;
  color: #dcecff !important;
  font-size: 0.72rem !important;
  font-weight: 560 !important;
  line-height: 1.22 !important;
  text-align: left !important;
  justify-content: flex-start !important;
}
.sidebar-nav-button button:hover {
  background: rgba(255,255,255,0.055) !important;
  border-color: rgba(47,208,195,0.18) !important;
  color: #ffffff !important;
  transform: none !important;
}
.sidebar-nav-button.active button {
  background: rgba(47,208,195,0.10) !important;
  border-color: rgba(47,208,195,0.26) !important;
  color: #efffff !important;
}

/* ── Chat area ───────────────────────────────────────────────── */
.chat-scroll {
  min-height: 440px;
  max-height: calc(100vh - 25rem);
  overflow-y: auto;
  padding: 8px 4px 10rem;
  scrollbar-width: thin;
  scrollbar-color: #2c4666 transparent;
}
.chat-shell {
  background: linear-gradient(180deg, rgba(10,26,44,0.95), rgba(8,23,39,0.98));
  border: 1px solid var(--line);
  border-radius: 26px;
  padding: 14px 16px 18px;
  box-shadow: var(--shadow);
}
.chat-scroll::-webkit-scrollbar { width: 6px; }
.chat-scroll::-webkit-scrollbar-thumb { background: #2d4565; border-radius: 999px; }

div[data-clearfix]::after { content: ""; display: table; clear: both; }

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 380px;
  padding: 64px 24px;
  text-align: center;
  border: 1px dashed rgba(138,163,190,0.18);
  border-radius: 22px;
  background: rgba(255,255,255,0.015);
}
.es-icon  { font-size: 2.8rem; margin-bottom: 14px; opacity: 0.8; }
.es-title { font-size: 1.1rem; font-weight: 700; color: #eef7ff; margin-bottom: 8px; }
.es-desc  { font-size: 0.92rem; color: var(--muted); line-height: 1.7; max-width: 420px; }

/* User bubble */
.chat-row {
  display: flex;
  gap: 14px;
  margin: 14px 0;
  width: 100%;
}
.chat-row.user {
  justify-content: flex-end;
}
.chat-row.assistant {
  justify-content: flex-start;
}
.avatar {
  width: 44px;
  height: 44px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  flex: 0 0 auto;
  border: 1px solid rgba(255,255,255,0.08);
}
.avatar.assistant {
  background: linear-gradient(180deg, rgba(47,208,195,0.14), rgba(47,143,255,0.10));
  color: #dffffc;
}
.avatar.user {
  background: rgba(255,255,255,0.04);
  color: #eff7ff;
}
.message-card {
  max-width: 82%;
  padding: 16px 18px;
  border-radius: 20px;
  border: 1px solid var(--line);
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.message-card.user {
  background: linear-gradient(180deg, rgba(30,67,118,0.94), rgba(20,50,82,0.94));
  border-radius: 20px 20px 8px 20px;
}
.message-card.assistant {
  background: linear-gradient(180deg, rgba(11,46,68,0.96), rgba(11,37,57,0.96));
  border-left: 2px solid rgba(47,208,195,0.45);
  border-radius: 20px 20px 20px 8px;
}
.message-card.assistant.loading {
  border-left-color: rgba(47,208,195,0.72);
}
.message-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 0.84rem;
}
.message-author {
  font-weight: 700;
  color: #eff8ff;
}
.message-time {
  color: var(--muted);
  font-size: 0.78rem;
}
.message-body {
  color: #eff7ff;
  line-height: 1.72;
  font-size: 0.98rem;
}
.message-body code {
  color: #ffbf75;
  background: rgba(6, 19, 34, 0.75);
  padding: 2px 6px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.05);
}
.thinking-state {
  display: grid;
  gap: 12px;
}
.thinking-line {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #e8fbff;
  font-weight: 700;
}
.typing-dots {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  width: 32px;
}
.typing-dots span {
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #6ff2e7;
  opacity: 0.38;
  animation: typingPulse 1.15s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.16s; }
.typing-dots span:nth-child(3) { animation-delay: 0.32s; }
.thinking-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.thinking-step {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(120,166,209,0.14);
  background: rgba(255,255,255,0.035);
  color: #a8bdd3;
  font-size: 0.78rem;
  font-weight: 600;
}
.thinking-step.active {
  color: #9df1eb;
  border-color: rgba(47,208,195,0.24);
  background: rgba(47,208,195,0.08);
}
@keyframes typingPulse {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
  40% { transform: translateY(-4px); opacity: 1; }
}

/* AI bubble */
.context-stack {
  display: grid;
  gap: 8px;
  margin-top: 12px;
}

/* Badges */
.verified-badge {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(83,209,125,0.10); color: #8be6a8;
  border: 1px solid rgba(83,209,125,0.24);
  border-radius: 999px; padding: 5px 12px;
  font-size: 0.76rem; font-weight: 600; margin-top: 10px;
}
.fail-badge {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(255,123,114,0.08); color: #ffaaa3;
  border: 1px solid rgba(255,123,114,0.18);
  border-radius: 999px; padding: 5px 12px;
  font-size: 0.76rem; font-weight: 600; margin-top: 10px;
}
.message-footer {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.confidence-pill {
  padding: 5px 12px;
  border-radius: 999px;
  border: 1px solid rgba(47,143,255,0.22);
  background: rgba(47,143,255,0.08);
  color: #9dcaff;
  font-size: 0.76rem;
  font-weight: 600;
}

/* Context and intent */
.context-card {
  background: rgba(4, 16, 29, 0.48);
  border-radius: 14px;
  padding: 10px 14px;
  font-size: 0.8rem;
  border: 1px solid rgba(255,255,255,0.05);
  color: #a8bdd3;
  line-height: 1.55;
}
.intent-tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(47,208,195,0.07);
  border: 1px solid rgba(47,208,195,0.18);
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 0.78rem;
  color: #9df1eb;
  margin-bottom: 10px;
}

/* ── Input & buttons ─────────────────────────────────────────── */
.chat-composer {
  margin-top: 18px;
  padding: 14px;
  border-radius: 22px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(11,31,53,0.96), rgba(9,24,41,0.98));
  box-shadow: var(--shadow);
}
div[data-testid="stForm"] {
  position: sticky;
  bottom: 0.9rem;
  z-index: 20;
  margin-top: 18px;
  padding: 14px;
  border-radius: 22px;
  border: 1px solid rgba(47,208,195,0.18);
  background: linear-gradient(180deg, rgba(11,31,53,0.98), rgba(9,24,41,0.99));
  box-shadow: 0 24px 52px rgba(0,0,0,0.34);
  backdrop-filter: blur(12px);
}
.stTextInput > div > div > input {
  background: rgba(7,22,41,0.92) !important;
  color: var(--text) !important;
  border: 1px solid rgba(120,166,209,0.18) !important;
  border-radius: 16px !important;
  font-size: 1rem !important;
  min-height: 58px !important;
  caret-color: #ffffff !important;
  -webkit-text-fill-color: #eef7ff !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stTextInput > div > div > input:focus-visible,
.stTextInput > div > div > input:active {
  border-color: rgba(47,208,195,0.45) !important;
  box-shadow: 0 0 0 3px rgba(47,208,195,0.12) !important;
  outline: none !important;
}
.stTextInput [data-baseweb="input"] {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
.stTextInput [data-baseweb="input"]:focus-within {
  border: none !important;
  box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder { color: #6883a0 !important; }

.stFormSubmitButton > button {
  background: linear-gradient(135deg, var(--teal), var(--teal-strong)) !important;
  border: none !important;
  border-radius: 16px !important;
  color: #04202e !important;
  font-weight: 800 !important;
  min-height: 58px !important;
  transition: all 0.2s !important;
  box-shadow: none !important;
}
.stFormSubmitButton > button:hover {
  background: linear-gradient(135deg, #57e0d5, #2fd0c3) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 10px 26px rgba(47,208,195,0.24) !important;
}
.stFormSubmitButton > button:focus,
.stFormSubmitButton > button:active {
  box-shadow: 0 0 0 3px rgba(47,208,195,0.18) !important;
  outline: none !important;
}

/* Main area generic buttons */
.stButton > button {
  background: linear-gradient(180deg, rgba(13,34,57,0.95), rgba(11,31,53,0.95)) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  color: #cfe0f2 !important;
  transition: all 0.15s !important;
  box-shadow: none !important;
}
.stButton > button:hover {
  border-color: rgba(47,208,195,0.28) !important;
  color: #ffffff !important;
  background: linear-gradient(180deg, rgba(20,50,82,0.95), rgba(13,34,57,0.95)) !important;
}
.stButton > button:focus,
.stButton > button:active {
  border-color: rgba(47,208,195,0.28) !important;
  box-shadow: none !important;
  outline: none !important;
}

/* ── Cards ───────────────────────────────────────────────────── */
.side-section-title,
.section-title {
  font-size: 0.96rem;
  font-weight: 700;
  color: #f2f8ff;
  margin-bottom: 10px;
}
.side-note {
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: -6px;
  margin-bottom: 12px;
}
.recent-query-title {
  color: #eef7ff;
  font-size: 0.92rem;
  font-weight: 600;
  line-height: 1.35;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.recent-query-card-shell {
  display: block;
  padding: 11px 12px;
  border-radius: 14px;
  border: 1px solid transparent;
  background: transparent;
  margin-bottom: 8px;
  text-decoration: none !important;
  transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease;
  cursor: pointer;
}
.recent-query-card-shell.active {
  background: rgba(255,255,255,0.06);
  border-color: rgba(255,255,255,0.06);
}
.recent-query-card-shell:hover {
  background: rgba(255,255,255,0.04);
  border-color: rgba(255,255,255,0.04);
}
.sidebar-text-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-top: 0.12rem;
}
.sidebar-text-button button {
  min-height: 32px !important;
  padding: 0.36rem 0.56rem !important;
  border-radius: 9px !important;
  background: rgba(5, 17, 30, 0.58) !important;
  border: 1px solid rgba(120,166,209,0.08) !important;
  box-shadow: none !important;
  color: #cfe0f2 !important;
  font-size: 0.72rem !important;
  font-weight: 400 !important;
  line-height: 1.3 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
.sidebar-text-button button:hover {
  background: rgba(8, 24, 41, 0.82) !important;
  border-color: rgba(120,166,209,0.12) !important;
  color: #ffffff !important;
  transform: none !important;
}
.sidebar-text-button.active button {
  background: rgba(7, 28, 43, 0.92) !important;
  border-color: rgba(47,208,195,0.14) !important;
  color: #efffff !important;
}
.panel-card {
  background: linear-gradient(180deg, rgba(13,34,57,0.96), rgba(10,26,44,0.98));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
  margin-bottom: 10px;
}
.panel-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.panel-title {
  font-size: 0.98rem;
  font-weight: 700;
  color: #eef8ff;
}
.panel-subtitle {
  color: var(--muted);
  font-size: 0.74rem;
}
.overall-score {
  display: flex;
  align-items: baseline;
  gap: 6px;
  color: #f4fbff;
}
.overall-score strong {
  font-size: 1.72rem;
  line-height: 1;
}
.overall-score span {
  font-size: 0.84rem;
  color: var(--muted);
}
.metric-card {
  display: grid;
  grid-template-columns: 64px 1fr;
  gap: 12px;
  align-items: center;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 16px;
  padding: 10px 12px;
  margin-top: 8px;
  min-height: 92px;
}
.metric-ring {
  width: 58px;
  height: 58px;
  border-radius: 50%;
  background: conic-gradient(#18d59c 0%, rgba(255,255,255,0.08) 0%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow:
    0 0 0 8px rgba(255,255,255,0.025),
    inset 0 0 0 1px rgba(255,255,255,0.04);
  flex: 0 0 auto;
  position: relative;
}
.metric-ring::before {
  content: "";
  position: absolute;
  inset: 9px;
  border-radius: inherit;
  background: #0c1f34;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
}
.metric-ring span {
  color: #f4fbff;
  font-size: 0.98rem;
  font-weight: 800;
  line-height: 1;
  position: relative;
  z-index: 1;
}
.metric-copy h4 {
  margin: 0 0 4px 0;
  font-size: 0.86rem;
  color: #f2f8ff;
}
.metric-copy p {
  margin: 0;
  color: var(--muted);
  font-size: 0.72rem;
  line-height: 1.35;
}
.metric-pill {
  margin-top: 7px;
  display: inline-flex;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,0.08);
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.stat-card {
  border-radius: 18px;
  padding: 14px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
}
.stat-label {
  color: var(--muted);
  font-size: 0.8rem;
  margin-bottom: 8px;
}
.stat-value {
  color: #f5fbff;
  font-size: 1.3rem;
  font-weight: 800;
}
.prompt-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
  margin-bottom: 10px;
}
.prompt-chip {
  padding: 9px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  color: #bad0e6;
  font-size: 0.82rem;
}
.chat-bottom-spacer {
  height: 1rem;
}
.kb-shell, .report-shell {
  background: linear-gradient(180deg, rgba(13,34,57,0.94), rgba(8,23,39,0.98));
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 24px;
  box-shadow: var(--shadow);
}
.report-score-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin: 14px 0 18px;
}
.report-score-card {
  border-radius: 16px;
  padding: 14px 16px;
  border: 1px solid rgba(120,166,209,0.14);
  background: linear-gradient(180deg, rgba(16,39,63,0.88), rgba(10,26,44,0.96));
}
.report-score-label {
  color: #b8cbe0;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 10px;
}
.report-score-value {
  color: #f5fbff;
  font-size: 1.65rem;
  font-weight: 850;
}
.report-chart-title {
  color: #f1f8ff;
  font-size: 1rem;
  font-weight: 800;
  margin: 14px 0 8px;
}

/* ── Misc ────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 14px 0 !important; }
h3 { color: #f1f8ff !important; font-weight: 700 !important;
     font-size: 0.98rem !important; letter-spacing: -0.01em !important; }
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px; padding: 12px 14px;
}
[data-testid="metric-container"] label {
  color: #84a1be !important; font-size: 0.72rem !important;
  text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] { color: #f1f8ff !important; }
[data-testid="stDataFrame"] {
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.07);
}
div[data-testid="column"] .stButton > button {
  min-height: 62px;
}
</style>
<script>
document.addEventListener("DOMContentLoaded", function () {
  const disableSpellcheck = () => {
    document.querySelectorAll('.stTextInput input').forEach((el) => {
      el.setAttribute('spellcheck', 'false');
      el.setAttribute('autocorrect', 'off');
      el.setAttribute('autocapitalize', 'off');
    });
  };
  disableSpellcheck();
  new MutationObserver(disableSpellcheck).observe(document.body, {
    childList: true,
    subtree: true,
  });
});
</script>
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
if "active_query" not in st.session_state:
    st.session_state.active_query = ""
if "loaded_query_id" not in st.session_state:
    st.session_state.loaded_query_id = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pending_visible" not in st.session_state:
    st.session_state.pending_visible = False
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""


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


def format_query_time(ts) -> str:
    raw = str(ts).strip()
    if not raw or raw.lower() == "none":
        return "Just now"
    if raw == "Demo":
        return "Demo prompt"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(raw[:19], fmt)
            return dt.strftime("%b %d · %I:%M %p").replace(" 0", " ")
        except ValueError:
            continue
    return raw


def query_icon(query: str) -> str:
    q = query.lower()
    if "bec" in q or "business email compromise" in q:
        return "🎯"
    if "spear" in q:
        return "🎣"
    if "url" in q or "link" in q:
        return "🔗"
    if "report" in q:
        return "📨"
    if "awareness" in q or "best practice" in q:
        return "🛡️"
    if "page" in q or "site" in q:
        return "🌐"
    return "✉️"


def restore_query(query_id: int) -> bool:
    loaded = get_query_by_id(query_id)
    if not loaded:
        return False
    st.session_state.messages = [
        {
            "role": "user",
            "content": loaded["question"],
        },
        {
            "role": "assistant",
            "content": loaded["answer"],
            "intent": loaded["intent"],
            "contexts": loaded["contexts"],
            "verified": loaded["ontology_verified"],
            "confidence": loaded["confidence_score"],
        },
    ]
    st.session_state.active_query = loaded["question"]
    st.session_state.active_page = "Chat"
    st.session_state.last_eval = loaded["evaluation"]
    st.session_state.loaded_query_id = query_id
    st.session_state.pending_query = None
    st.session_state.pending_visible = False
    st.query_params["page"] = "Chat"
    return True


def queue_chat_query() -> None:
    query = st.session_state.get("chat_input", "").strip()
    if not query or st.session_state.get("pending_query"):
        return
    st.session_state.active_query = query
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Analysing your question",
        "loading": True,
    })
    st.session_state.pending_query = query
    st.session_state.pending_visible = False
    st.session_state.loaded_query_id = None
    st.session_state.chat_input = ""


requested_loaded_query = st.query_params.get("load_query")
if isinstance(requested_loaded_query, list):
    requested_loaded_query = requested_loaded_query[0]
if requested_loaded_query:
    try:
        requested_loaded_query_id = int(requested_loaded_query)
    except ValueError:
        requested_loaded_query_id = None
    if (
        requested_loaded_query_id
        and st.session_state.loaded_query_id != requested_loaded_query_id
    ):
        restore_query(requested_loaded_query_id)


def metric_card(value: int, label: str, desc: str) -> str:
    status = score_label(value)
    pct = max(0, min(100, int(value)))
    ring_color = score_color(pct)
    track_color = "rgba(255,255,255,0.08)"
    pill_bg = "rgba(83,209,125,0.10)" if value >= 80 else (
        "rgba(255,173,66,0.10)" if value >= 60 else "rgba(255,123,114,0.10)"
    )
    pill_fg = "#8be6a8" if value >= 80 else ("#ffbe68" if value >= 60 else "#ffaaa3")
    return f"""
    <div class="metric-card">
      <div class="metric-ring" style="background: conic-gradient({ring_color} 0 {pct}%, {track_color} {pct}% 100%);">
        <span>{pct}%</span>
      </div>
      <div class="metric-copy">
        <h4>{label}</h4>
        <p>{desc}</p>
        <span class="metric-pill" style="background:{pill_bg}; color:{pill_fg};">{status}</span>
      </div>
    </div>
    """


def stat_card(label: str, value) -> str:
    return f"""
    <div class="stat-card">
      <div class="stat-label">{label}</div>
      <div class="stat-value">{value}</div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
page_items = [
    ("Chat", "💬"),
    ("Knowledge Base", "📚"),
    ("Reports", "📊"),
]
requested_page = st.query_params.get("page", st.session_state.active_page)
if isinstance(requested_page, list):
    requested_page = requested_page[0]
if requested_page in {name for name, _ in page_items}:
    st.session_state.active_page = requested_page
active = st.session_state.active_page


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <span class="sidebar-brand-mark">🛡️</span>
      <div class="sidebar-brand-copy">
        <span class="sidebar-brand-title">PhishingGuard-RAG</span>
        <span class="sidebar-brand-subtitle">Threat analysis workspace</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section sidebar-new-chat-heading">
      <div class="sidebar-heading">
        <span class="sidebar-heading-title">New Chat</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✚  Start a new conversation", key="sidebar_new_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_query = ""
        st.session_state.prefill = ""
        st.session_state.chat_input = ""
        st.session_state.pending_query = None
        st.session_state.pending_visible = False
        st.session_state.active_page = "Chat"
        st.query_params["page"] = "Chat"
        st.rerun()

    st.markdown('<div class="sidebar-new-chat-gap"></div>', unsafe_allow_html=True)

    for nav_name, nav_icon in page_items[1:]:
        active_class = " active" if active == nav_name else ""
        st.markdown(
            f'<div class="sidebar-nav-button{active_class}">',
            unsafe_allow_html=True,
        )
        if st.button(
            f"{nav_icon}  {nav_name}",
            key=f"sidebar_nav_{nav_name}",
            use_container_width=True,
        ):
            st.session_state.active_page = nav_name
            st.query_params["page"] = nav_name
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
      <div class="sidebar-heading">
        <span class="sidebar-heading-title">Recent Chats</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    recent = get_recent_queries(8)
    recent_items = [(r[0], r[1], str(r[2])[:10]) for r in recent][:8]

    if not recent_items:
        st.markdown('<div class="sidebar-empty">No recent chats yet. Start a conversation and it will show up here.</div>', unsafe_allow_html=True)

    if recent_items:
        st.markdown('<div class="sidebar-text-list">', unsafe_allow_html=True)
        for query_id, q, ts in recent_items:
            short = (q[:42] + "…") if len(q) > 42 else q
            is_active_query = st.session_state.get("active_query") == q
            active_class = " active" if is_active_query else ""
            st.markdown(
                f'<div class="sidebar-text-button{active_class}">',
                unsafe_allow_html=True,
            )
            if st.button(short, key=f"recent_chat_{query_id}", use_container_width=True):
                restore_query(query_id)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
      <div class="sidebar-heading">
        <span class="sidebar-heading-title">Example Prompts</span>
        <span class="sidebar-heading-meta">Try one</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    ex_prompts = [
        "How to detect spear-phishing?",
        "What are phishing URL red flags?",
        "How to prevent BEC attacks?",
    ]
    st.markdown('<div class="sidebar-text-list">', unsafe_allow_html=True)
    for i, ep in enumerate(ex_prompts):
        st.markdown(
            '<div class="sidebar-text-button">',
            unsafe_allow_html=True,
        )
        if st.button(ep, key=f"ex_prompt_{i}", use_container_width=True):
            st.session_state.chat_input = ep
            st.session_state.active_page = "Chat"
            st.query_params["page"] = "Chat"
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
active = st.session_state.active_page
st.query_params["page"] = active

# ── CHAT PAGE ─────────────────────────────────────────────────────────────────
if active == "Chat":
    col_chat, col_right = st.columns([4, 1.2])

    # ── Chat column ──────────────────────────────────────────────────────────
    with col_chat:
        # Render chat history
        if not st.session_state.messages:
            chat_html = (
                '<div class="chat-shell"><div class="chat-scroll">'
                '<div class="empty-state">'
                '<div class="es-icon">🛡️</div>'
                '<div class="es-title">Ask me about phishing threats</div>'
                '<div class="es-desc">I can help identify phishing indicators, explain attack techniques, validate suspicious messages, and guide incident response.</div>'
                '</div></div></div>'
            )
        else:
            chat_html = '<div class="chat-shell"><div class="chat-scroll">'
            for msg in st.session_state.messages:
                msg_time = datetime.now().strftime("%I:%M %p").lstrip("0")
                if msg["role"] == "user":
                    chat_html += (
                        f'<div class="chat-row user">'
                        f'<div class="message-card user">'
                        f'<div class="message-meta">'
                        f'<span class="message-author">You</span>'
                        f'<span class="message-time">{msg_time}</span>'
                        f'</div>'
                        f'<div class="message-body">{msg["content"]}</div>'
                        f'</div>'
                        f'<div class="avatar user">👤</div>'
                        f'</div>'
                    )
                elif msg.get("loading"):
                    chat_html += (
                        f'<div class="chat-row assistant">'
                        f'<div class="avatar assistant">🛡️</div>'
                        f'<div class="message-card assistant loading">'
                        f'<div class="message-meta">'
                        f'<span class="message-author">PhishingGuard-RAG AI</span>'
                        f'<span class="message-time">{msg_time}</span>'
                        f'</div>'
                        f'<div class="message-body">'
                        f'<div class="thinking-state">'
                        f'<div class="thinking-line">'
                        f'<span>{msg.get("content", "Analysing your question")}</span>'
                        f'<span class="typing-dots"><span></span><span></span><span></span></span>'
                        f'</div>'
                        f'<div class="thinking-steps">'
                        f'<span class="thinking-step active">Intent</span>'
                        f'<span class="thinking-step">Retrieval</span>'
                        f'<span class="thinking-step">Answer</span>'
                        f'<span class="thinking-step">Evaluation</span>'
                        f'</div>'
                        f'</div>'
                        f'</div>'
                        f'</div>'
                        f'</div>'
                    )
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
                            f'<div class="intent-tag">🎯 <b>Intent</b> {msg["intent"][:120]}</div>'
                        )
                    ctx_html = ""
                    for i, c in enumerate(msg.get("contexts", [])[:3]):
                        ctx_html += (
                            f'<div class="context-card">'
                            f'[{i+1}] {c[:200]}…</div>'
                        )
                    chat_html += (
                        f'<div class="chat-row assistant">'
                        f'<div class="avatar assistant">🛡️</div>'
                        f'<div class="message-card assistant">'
                        f'<div class="message-meta">'
                        f'<span class="message-author">PhishingGuard-RAG AI</span>'
                        f'<span class="message-time">{msg_time}</span>'
                        f'</div>'
                        f'{intent_html}'
                        f'<div class="message-body">{msg["content"]}</div>'
                        f'<div class="context-stack">{ctx_html}</div>'
                        f'<div class="message-footer">{badge}'
                        f'<span class="confidence-pill">Confidence {conf:.0%}</span>'
                        f'</div>'
                        f'</div>'
                        f'</div>'
                    )
            chat_html += '</div></div>'
        st.markdown(chat_html, unsafe_allow_html=True)

        # Input
        prefill = st.session_state.pop("prefill", "")
        if prefill and not st.session_state.pending_query:
            st.session_state.chat_input = prefill
        st.markdown("""
        <div class="prompt-row">
          <span class="prompt-chip">How to detect spear-phishing?</span>
          <span class="prompt-chip">What are phishing URL red flags?</span>
          <span class="prompt-chip">How to prevent BEC attacks?</span>
        </div>
        """, unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                st.text_input(
                    "query",
                    key="chat_input",
                    placeholder="Ask a question about phishing threats…",
                    label_visibility="collapsed",
                    disabled=bool(st.session_state.pending_query),
                )
            with col_btn:
                st.form_submit_button(
                    "Send ✈️",
                    use_container_width=True,
                    disabled=bool(st.session_state.pending_query),
                    on_click=queue_chat_query,
                )
        st.markdown('<div class="chat-bottom-spacer"></div>', unsafe_allow_html=True)

        if st.session_state.pending_query:
            if not st.session_state.pending_visible:
                st.session_state.pending_visible = True
                st.rerun()

            query = st.session_state.pending_query

            try:
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

                assistant_msg = {
                    "role":      "assistant",
                    "content":   result["answer"],
                    "intent":    result["intent"],
                    "contexts":  result["contexts"],
                    "verified":  result["ontology_verified"],
                    "confidence": result["confidence_score"],
                }
            except Exception as exc:
                assistant_msg = {
                    "role": "assistant",
                    "content": (
                        "Sorry, the request failed while processing. "
                        f"Please try again. Error: {exc}"
                    ),
                    "intent": "",
                    "contexts": [],
                    "verified": False,
                    "confidence": 0,
                }

            if (
                st.session_state.messages
                and st.session_state.messages[-1].get("loading")
            ):
                st.session_state.messages[-1] = assistant_msg
            else:
                st.session_state.messages.append(assistant_msg)
            st.session_state.pending_query = None
            st.session_state.pending_visible = False
            st.rerun()

    # ── Right panel ───────────────────────────────────────────────────────────
    with col_right:
        ev = st.session_state.last_eval

        overall = ev.get("overall_score", 0)
        color_o = score_color(int(overall))
        st.markdown(f"""
        <div class="panel-card">
          <div class="panel-heading">
            <div>
              <div class="panel-title">📊 RAGAS Evaluation</div>
              <div class="panel-subtitle">Live retrieval quality signals</div>
            </div>
          </div>
          <div class="overall-score"><strong style="color:{color_o};">{overall}</strong><span>/100 overall</span></div>
        </div>
        """, unsafe_allow_html=True)

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
            st.markdown(metric_card(val, label, desc), unsafe_allow_html=True)

# ── KNOWLEDGE BASE PAGE ───────────────────────────────────────────────────────
elif active == "Knowledge Base":
    kb = get_kb_stats()
    kb_stats_html = "".join([
        stat_card("Documents", kb["documents"]),
        stat_card("Chunks", kb["chunks"]),
        stat_card("Embeddings", kb["embeddings"]),
        stat_card("Queries", kb["total_queries"]),
    ])
    st.markdown(f"""
    <div class="panel-card">
      <div class="panel-heading">
        <div>
          <div class="panel-title">📚 Knowledge Base Stats</div>
          <div class="panel-subtitle">Current indexed corpus snapshot</div>
        </div>
      </div>
      <div class="stat-grid">
        {kb_stats_html}
      </div>
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="report-shell">
      <div class="panel-title">📊 Evaluation Reports</div>
      <div class="panel-subtitle">Recent RAGAS performance trends and exportable query history.</div>
    </div>
    """, unsafe_allow_html=True)

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

        labels = ["Faithfulness", "Ans. Relevance",
                  "Ctx. Relevance", "Ctx. Recall"]
        keys   = ["faithfulness", "answer_relevance",
                  "context_relevance", "context_recall"]
        score_cards = ['<div class="report-score-grid">']
        for lbl, key in zip(labels, keys):
            val = round(avg[key] * 100, 1)
            score_cards.append(
                f'<div class="report-score-card">'
                f'<div class="report-score-label">{lbl}</div>'
                f'<div class="report-score-value">{val}%</div>'
                f'</div>'
            )
        score_cards.append('</div>')
        st.markdown("".join(score_cards), unsafe_allow_html=True)

        # Trend chart
        import plotly.express as px
        df_plot = df_eval.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        st.markdown('<div class="report-chart-title">RAGAS Score Trends</div>', unsafe_allow_html=True)
        fig = px.line(
            df_plot, x="timestamp",
            y=["faithfulness", "answer_relevance",
               "context_relevance", "context_recall"],
            color_discrete_sequence=["#1e90ff", "#2ecc71",
                                      "#f39c12", "#e74c3c"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#10233b",
            font=dict(color="#d9e7f6", size=13),
            legend=dict(
                title=dict(text="Metric", font=dict(color="#f1f8ff")),
                font=dict(color="#d9e7f6"),
                bgcolor="rgba(8,23,39,0.72)",
                bordercolor="rgba(120,166,209,0.18)",
                borderwidth=1,
            ),
            xaxis=dict(
                title=dict(font=dict(color="#d9e7f6")),
                tickfont=dict(color="#b8cbe0"),
                gridcolor="rgba(255,255,255,0.14)",
                zerolinecolor="rgba(255,255,255,0.18)",
            ),
            yaxis=dict(
                title=dict(font=dict(color="#d9e7f6")),
                tickfont=dict(color="#b8cbe0"),
                gridcolor="rgba(255,255,255,0.18)",
                zerolinecolor="rgba(255,255,255,0.18)",
            ),
            margin=dict(l=48, r=28, t=12, b=48),
        )
        fig.update_traces(line=dict(width=3))
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
