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
  min-width: 19rem !important;
  max-width: 19rem !important;
}
section[data-testid="stSidebar"] > div {
  padding-top: 0rem;
}
section[data-testid="stSidebar"] * { color: #d9e7f6 !important; }

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
  border-radius: 16px !important;
  color: #d7e8f7 !important;
  font-size: 0.95rem !important;
  text-align: left !important;
  transition: all 0.18s ease !important;
  box-shadow: none !important;
  min-height: 72px;
  padding: 0.95rem 1rem !important;
  white-space: normal !important;
  line-height: 1.45 !important;
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
.sidebar-hero {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 18px 18px 16px;
  border-radius: 22px;
  border: 1px solid rgba(120, 166, 209, 0.14);
  background:
    radial-gradient(circle at top left, rgba(47,208,195,0.10), transparent 34%),
    linear-gradient(180deg, rgba(17,40,67,0.98), rgba(10,26,44,0.98));
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.24);
  margin-bottom: 14px;
}
.sidebar-hero-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.sidebar-hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0.38rem 0.72rem;
  border-radius: 999px;
  background: rgba(47,208,195,0.10);
  border: 1px solid rgba(47,208,195,0.22);
  color: #eafffd;
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.sidebar-hero-mark {
  width: 3rem;
  height: 3rem;
  border-radius: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.4rem;
  background: linear-gradient(180deg, rgba(47,208,195,0.18), rgba(47,143,255,0.10));
  border: 1px solid rgba(47,208,195,0.26);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.sidebar-hero-copy {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.sidebar-hero-eyebrow {
  color: #8fb2cf;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.sidebar-hero-title {
  color: #f4f9ff;
  font-size: 1.22rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.15;
}
.sidebar-hero-title span {
  color: var(--teal);
}
.sidebar-hero-subtitle {
  color: #a8bdd3;
  font-size: 0.95rem;
  line-height: 1.6;
}
.sidebar-hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.sidebar-hero-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 34px;
  padding: 0.38rem 0.72rem;
  border-radius: 999px;
  border: 1px solid rgba(120,166,209,0.14);
  background: rgba(255,255,255,0.03);
  color: #cbdcf0;
  font-size: 0.78rem;
  font-weight: 700;
}
.sidebar-hero-chip.active {
  background: rgba(47,208,195,0.12);
  border-color: rgba(47,208,195,0.28);
  color: #efffff;
}
.top-nav-shell {
  position: sticky;
  top: 0.75rem;
  z-index: 30;
  margin: 0 0 20px;
  padding: 0.65rem;
  border-radius: 24px;
  border: 1px solid rgba(120, 166, 209, 0.16);
  background:
    linear-gradient(180deg, rgba(14, 35, 58, 0.96), rgba(8, 24, 41, 0.94)),
    radial-gradient(circle at top left, rgba(47, 208, 195, 0.10), transparent 45%);
  box-shadow: 0 20px 42px rgba(0, 0, 0, 0.26);
  backdrop-filter: blur(14px);
}
.top-nav-bar {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.top-nav-link {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 64px;
  padding: 0.95rem 1.1rem;
  border-radius: 18px;
  border: 1px solid rgba(120, 166, 209, 0.12);
  background: linear-gradient(180deg, rgba(16, 39, 63, 0.92), rgba(10, 26, 44, 0.98));
  color: #dcecff !important;
  text-decoration: none !important;
  font-size: 1rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}
.top-nav-link:hover {
  transform: translateY(-1px);
  border-color: rgba(47,208,195,0.30);
  background: linear-gradient(180deg, rgba(22, 52, 80, 0.96), rgba(12, 31, 50, 0.98));
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
  color: #f6fbff !important;
}
.top-nav-link.active {
  background:
    linear-gradient(180deg, rgba(38, 111, 141, 0.98), rgba(20, 63, 93, 0.98)),
    radial-gradient(circle at top left, rgba(47,208,195,0.18), transparent 55%);
  border-color: rgba(47,208,195,0.55);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.08),
    inset 0 -4px 0 rgba(47,208,195,0.90),
    0 16px 30px rgba(0, 0, 0, 0.24);
  color: #ffffff !important;
}
.top-nav-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  font-size: 1.05rem;
  line-height: 1;
}
.top-nav-link.active .top-nav-icon {
  background: rgba(255,255,255,0.16);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
}
.top-nav-label {
  white-space: nowrap;
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
.recent-query-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}
.recent-query-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  font-size: 1rem;
}
.recent-query-time {
  color: var(--muted);
  font-size: 0.76rem;
  white-space: nowrap;
}
.recent-query-title {
  color: #eef7ff;
  font-size: 0.92rem;
  font-weight: 700;
  line-height: 1.42;
}
.recent-query-sub {
  color: var(--muted);
  font-size: 0.76rem;
  margin-top: 6px;
}
.recent-query-card-shell {
  padding: 14px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.06);
  background: linear-gradient(180deg, rgba(17,40,67,0.98), rgba(11,31,53,0.98));
  margin-bottom: 8px;
}
.recent-query-card-shell.active {
  background: linear-gradient(180deg, rgba(24,58,95,0.98), rgba(17,40,67,0.98));
  border-color: rgba(47,208,195,0.22);
  box-shadow: inset 0 0 0 1px rgba(47,208,195,0.08);
}
.panel-card {
  background: linear-gradient(180deg, rgba(13,34,57,0.96), rgba(10,26,44,0.98));
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 18px;
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}
.panel-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.panel-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: #eef8ff;
}
.panel-subtitle {
  color: var(--muted);
  font-size: 0.78rem;
}
.overall-score {
  display: flex;
  align-items: baseline;
  gap: 6px;
  color: #f4fbff;
}
.overall-score strong {
  font-size: 2rem;
  line-height: 1;
}
.overall-score span {
  font-size: 0.95rem;
  color: var(--muted);
}
.metric-card {
  display: grid;
  grid-template-columns: 84px 1fr;
  gap: 14px;
  align-items: center;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 18px;
  padding: 14px;
  margin-top: 12px;
}
.metric-ring {
  --angle: 0deg;
  --ring-color: var(--teal);
  width: 72px;
  height: 72px;
  border-radius: 50%;
  background:
    radial-gradient(closest-side, #0c1f34 70%, transparent 71% 100%),
    conic-gradient(var(--ring-color) var(--angle), rgba(255,255,255,0.08) 0deg);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
}
.metric-ring span {
  color: #f4fbff;
  font-size: 1.45rem;
  font-weight: 800;
}
.metric-copy h4 {
  margin: 0 0 6px 0;
  font-size: 1rem;
  color: #f2f8ff;
}
.metric-copy p {
  margin: 0;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.55;
}
.metric-pill {
  margin-top: 10px;
  display: inline-flex;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 0.74rem;
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
.alert-card {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  padding: 12px 0;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.alert-card:first-of-type {
  border-top: none;
  padding-top: 4px;
}
.alert-main {
  display: flex;
  gap: 10px;
}
.alert-icon {
  width: 38px;
  height: 38px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  flex: 0 0 auto;
}
.alert-icon.high {
  background: rgba(255,123,114,0.10);
  color: #ff958b;
}
.alert-icon.medium {
  background: rgba(255,173,66,0.10);
  color: #ffbe68;
}
.alert-icon.low {
  background: rgba(83,209,125,0.10);
  color: #8be6a8;
}
.alert-copy strong {
  display: block;
  color: #eef8ff;
  margin-bottom: 4px;
}
.alert-copy span {
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.5;
}
.alert-time {
  color: var(--muted);
  font-size: 0.78rem;
  white-space: nowrap;
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


def metric_card(value: int, label: str, desc: str) -> str:
    color = score_color(value)
    status = score_label(value)
    angle = max(0, min(100, value)) * 3.6
    pill_bg = "rgba(83,209,125,0.10)" if value >= 80 else (
        "rgba(255,173,66,0.10)" if value >= 60 else "rgba(255,123,114,0.10)"
    )
    pill_fg = "#8be6a8" if value >= 80 else ("#ffbe68" if value >= 60 else "#ffaaa3")
    return f"""
    <div class="metric-card">
      <div class="metric-ring" style="--angle:{angle}deg; --ring-color:{color};">
        <span>{value}</span>
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


def alert_card(title: str, desc: str, sev: str, ts) -> str:
    sev = (sev or "low").lower()
    icon = {"high": "⚠️", "medium": "🛡️", "low": "✓"}.get(sev, "•")
    return f"""
    <div class="alert-card">
      <div class="alert-main">
        <div class="alert-icon {sev}">{icon}</div>
        <div class="alert-copy">
          <strong>{title}</strong>
          <span>{desc}</span>
        </div>
      </div>
      <div class="alert-time">{str(ts)}</div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo + title
    st.markdown("""
    <div class="sidebar-hero">
      <div class="sidebar-hero-top">
        <span class="sidebar-hero-mark">🛡️</span>
        <span class="sidebar-hero-badge">Secure Workspace</span>
      </div>
      <div class="sidebar-hero-copy">
        <span class="sidebar-hero-eyebrow">Phishing Analysis</span>
        <span class="sidebar-hero-title">PhishingGuard<span>-RAG</span> Assistant</span>
        <span class="sidebar-hero-subtitle">Investigate suspicious messages, URLs, and phishing indicators in one focused workspace.</span>
      </div>
      <div class="sidebar-hero-actions">
        <span class="sidebar-hero-chip active">Live Chat</span>
        <span class="sidebar-hero-chip">Knowledge Base</span>
        <span class="sidebar-hero-chip">RAGAS</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Recent queries
    st.markdown('<div class="side-section-title">🕒 Recent Queries</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-note">Resume recent investigations or load demo prompts.</div>', unsafe_allow_html=True)
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
        short = (q[:48] + "…") if len(q) > 48 else q
        icon = query_icon(q)
        stamp = format_query_time(ts)
        sub = "Recent investigation" if ts != "Demo" else "Try this example"
        is_active_query = st.session_state.get("active_query") == q
        card_html = f"""
        <div class="recent-query-card-shell{" active" if is_active_query else ""}">
        <div class="recent-query-meta">
          <span class="recent-query-badge">{icon}</span>
          <span class="recent-query-time">{stamp}</span>
        </div>
        <div class="recent-query-title">{short}</div>
        <div class="recent-query-sub">{sub}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        st.markdown('<div class="recent-query-action">', unsafe_allow_html=True)
        if st.button(f"{icon}  Load query", key=f"hist_{q[:20]}", use_container_width=True):
            st.session_state.prefill = q
            st.session_state.active_query = q
            st.session_state.active_page = "Chat"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️  Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_query = ""
        st.rerun()

    # Example prompts
    st.markdown('<div class="side-section-title">💡 Example Prompts</div>', unsafe_allow_html=True)
    ex_prompts = [
        "How to detect spear-phishing?",
        "What are phishing URL red flags?",
        "How to prevent BEC attacks?",
    ]
    for ep in ex_prompts:
        if st.button(f"↗  {ep}", key=f"ex_{ep[:15]}", use_container_width=True):
            st.session_state.prefill = ep


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
active = st.session_state.active_page
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
nav_html = '<div class="top-nav-shell"><div class="top-nav-bar">'
for name, icon in page_items:
    active_class = " active" if active == name else ""
    href = f"?page={name.replace(' ', '%20')}"
    nav_html += (
        f'<a class="top-nav-link{active_class}" href="{href}" target="_self">'
        f'<span class="top-nav-icon">{icon}</span>'
        f'<span class="top-nav-label">{name}</span>'
        '</a>'
    )
nav_html += '</div></div>'
st.markdown(nav_html, unsafe_allow_html=True)
st.query_params["page"] = active
active = st.session_state.active_page

# ── CHAT PAGE ─────────────────────────────────────────────────────────────────
if active == "Chat":
    col_chat, col_right = st.columns([3, 1.5])

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
                    for i, c in enumerate(msg.get("contexts", [])[:2]):
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
                user_input = st.text_input(
                    "query",
                    value=prefill,
                    placeholder="Ask a question about phishing threats…",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("Send ✈️",
                                                  use_container_width=True)
        st.markdown('<div class="chat-bottom-spacer"></div>', unsafe_allow_html=True)

        if submitted and user_input.strip():
            query = user_input.strip()
            st.session_state.active_query = query
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

        # KB Stats
        kb = get_kb_stats()
        st.markdown(f"""
        <div class="panel-card">
          <div class="panel-heading">
            <div>
              <div class="panel-title">📚 Knowledge Base Stats</div>
              <div class="panel-subtitle">Current indexed corpus snapshot</div>
            </div>
          </div>
          <div class="stat-grid">
            {stat_card("Documents", kb["documents"])}
            {stat_card("Chunks", kb["chunks"])}
            {stat_card("Embeddings", kb["embeddings"])}
            {stat_card("Queries", kb["total_queries"])}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Alerts
        st.markdown("""
        <div class="panel-card">
          <div class="panel-heading">
            <div>
              <div class="panel-title">🚨 Security Alerts</div>
              <div class="panel-subtitle">Recent high-signal activity</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        alerts = get_alerts(4)
        for title, desc, sev, ts in alerts:
            st.markdown(alert_card(title, desc, sev, ts), unsafe_allow_html=True)


# ── KNOWLEDGE BASE PAGE ───────────────────────────────────────────────────────
elif active == "Knowledge Base":
    st.markdown("""
    <div class="kb-shell">
      <div class="panel-title">📚 Knowledge Base</div>
      <div class="panel-subtitle">Indexed phishing reference material and source documents.</div>
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="report-shell">
      <div class="panel-title">📊 Evaluation Reports</div>
      <div class="panel-subtitle">Recent RAGAS performance trends and exportable query history.</div>
    </div>
    """, unsafe_allow_html=True)
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
