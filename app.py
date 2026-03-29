"""
app.py — MentionBot Quantitative Dashboard v3
===============================================
Professional trading terminal interface with Alpha Delta arbitrage,
regime detection, VIX/DXY feedback, and news context.

Launch:  python -m streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import pytz
import random
import math
from datetime import datetime, timedelta
import json
from pathlib import Path

from market_agent import run_agent_cycle
from market_impact import (
    get_sectors, get_topic_sector_matrix, get_impact_for_topic,
    simulate_market_reaction, generate_demo_trade_history, calculate_pnl,
    SECTORS,
)

EST = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Auto-Sync: compute real-time state from truth_archive.csv
# ---------------------------------------------------------------------------
DATA_PATHS = [
    Path(__file__).parent / "data" / "truth_archive.csv",
    Path(__file__).parent / "data" / "sample_posts.csv",
]


def _compute_live_state() -> dict:
    """
    Read the latest post data and compute what all the sidebar sliders
    should be set to right now, based on the current EST time.
    """
    now_est = datetime.now(EST)

    # Load posts
    df = None
    for p in DATA_PATHS:
        if p.exists():
            df = pd.read_csv(p)
            # Auto-detect timestamp column
            for col in ["created_at", "timestamp", "date"]:
                matches = [c for c in df.columns if c.lower() == col.lower()]
                if matches:
                    df["_ts"] = pd.to_datetime(df[matches[0]], utc=True, errors="coerce")
                    break
            if "_ts" not in df.columns:
                continue
            # Auto-detect text column
            for col in ["content", "text", "tweet", "body"]:
                matches = [c for c in df.columns if c.lower() == col.lower()]
                if matches:
                    df["_text"] = df[matches[0]].astype(str)
                    break
            df = df.dropna(subset=["_ts"])
            df["_ts_est"] = df["_ts"].dt.tz_convert(EST)
            df = df.sort_values("_ts_est")
            break

    if df is None or df.empty:
        # Fallback defaults if no data available
        return {
            "sim_hour": now_est.hour,
            "hours_since": 2.0,
            "post_count_24h": 5,
            "posting_velocity": 0.5,
            "news_volume": 10,
            "news_sentiment": 0.0,
            "vix_level": 18.0,
        }

    # Current hour
    current_hour = now_est.hour

    # Hours since last post
    last_post_time = df["_ts_est"].iloc[-1]
    hours_since_last = (now_est - last_post_time).total_seconds() / 3600
    hours_since_last = round(min(24.0, max(0.0, hours_since_last)), 2)
    # Round to nearest 0.25 for slider step
    hours_since_last = round(hours_since_last * 4) / 4

    # Rolling 24h post count
    cutoff_24h = now_est - timedelta(hours=24)
    posts_24h = int((df["_ts_est"] >= cutoff_24h).sum())
    posts_24h = min(50, posts_24h)

    # Posting velocity (posts in last 4 hours / 4)
    cutoff_4h = now_est - timedelta(hours=4)
    posts_4h = int((df["_ts_est"] >= cutoff_4h).sum())
    velocity = round(posts_4h / 4.0, 2)
    velocity = round(velocity * 4) / 4  # snap to 0.25 step
    velocity = min(5.0, velocity)

    # Simulated news/VIX based on current activity level and time
    # More active = more newsworthy = higher volume
    rng = random.Random(now_est.hour * 100 + now_est.minute)
    is_business_hours = 9 <= current_hour <= 17

    news_vol = rng.randint(5, 12)
    if is_business_hours:
        news_vol += rng.randint(3, 8)
    if posts_24h > 10:
        news_vol += rng.randint(5, 10)  # high activity = high news
    news_vol = min(50, news_vol)

    # Sentiment: slightly negative when posting is heavy
    sentiment = round(rng.gauss(0.05, 0.2), 2)
    if posts_24h > 15:
        sentiment -= 0.2
    sentiment = round(max(-1.0, min(1.0, sentiment)) * 20) / 20  # snap to 0.05 step

    # VIX: baseline 18, elevated if high activity
    vix = round(rng.gauss(18.0, 2.0), 1)
    if posts_24h > 10:
        vix += (posts_24h - 10) * 0.5
    if hours_since_last < 0.5:
        vix += 2.0  # very recent post = possible market event
    vix = round(max(10.0, min(50.0, vix)) * 2) / 2  # snap to 0.5 step

    # Recent tweets for the feed
    recent_texts = []
    if "_text" in df.columns:
        # Filter out RTs and get last 5
        real_posts = df[~df["_text"].str.startswith("RT @", na=False)]
        recent_texts = real_posts["_text"].tail(5).tolist()

    return {
        "sim_hour": current_hour,
        "hours_since": hours_since_last,
        "post_count_24h": posts_24h,
        "posting_velocity": velocity,
        "news_volume": news_vol,
        "news_sentiment": sentiment,
        "vix_level": vix,
        "recent_tweets": recent_texts,
    }


def _on_auto_sync():
    """Callback: compute live state and write it into session_state keys."""
    state = _compute_live_state()
    st.session_state["k_sim_hour"] = state["sim_hour"]
    st.session_state["k_hours_since"] = state["hours_since"]
    st.session_state["k_post_count_24h"] = state["post_count_24h"]
    st.session_state["k_posting_velocity"] = state["posting_velocity"]
    st.session_state["k_news_volume"] = state["news_volume"]
    st.session_state["k_news_sentiment"] = state["news_sentiment"]
    st.session_state["k_vix_level"] = state["vix_level"]
    if state.get("recent_tweets"):
        st.session_state["k_tweets_text"] = "\n".join(state["recent_tweets"])
    st.session_state["auto_synced"] = True

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MentionBot — Prediction Engine",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Quant Terminal Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* hide streamlit chrome — header stays for sidebar toggle */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    .viewerBadge_container__r5tak {display: none !important;}

    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* ── fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #111318;
        color: #cdd5e0;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #cdd5e0;
    }

    .mono {
        font-family: 'Fira Code', 'Consolas', monospace !important;
    }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #0d0f14;
        border-right: 1px solid #232730;
    }
    section[data-testid="stSidebar"] .stSlider > div > div {
        color: #7b8498;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextArea label,
    section[data-testid="stSidebar"] .stCheckbox label {
        font-family: 'Fira Code', monospace !important;
        font-size: 0.73rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #596175 !important;
    }

    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #4dd9cb;
        border-bottom: 1px solid #232730;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
        margin-top: 1.2rem;
    }

    /* ── header ── */
    .terminal-header {
        font-size: 1.55rem;
        font-weight: 800;
        color: #e4e9f2;
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
    }
    .terminal-header-accent {
        color: #4dd9cb;
        font-weight: 300;
    }
    .terminal-subtitle {
        font-size: 0.8rem;
        color: #596175;
        margin-bottom: 1.4rem;
    }

    /* ── probability gauge ── */
    .gauge-container {
        text-align: center;
        padding: 1.5rem 0;
    }
    .gauge-value {
        font-family: 'Fira Code', monospace;
        font-size: 4.2rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .gauge-hot  { color: #ef6b6b; }
    .gauge-warm { color: #edc04a; }
    .gauge-cold { color: #5cd8a0; }

    .gauge-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #596175;
        margin-top: 0.5rem;
    }

    .gauge-bar {
        width: 100%;
        height: 5px;
        background: #1a1d25;
        border-radius: 3px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .gauge-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* ── signal badges ── */
    .signal-tag {
        font-family: 'Fira Code', monospace;
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .signal-hot {
        background: rgba(239,107,107,0.12);
        color: #ef6b6b;
        border: 1px solid rgba(239,107,107,0.28);
    }
    .signal-warm {
        background: rgba(237,192,74,0.12);
        color: #edc04a;
        border: 1px solid rgba(237,192,74,0.28);
    }
    .signal-cold {
        background: rgba(92,216,160,0.12);
        color: #5cd8a0;
        border: 1px solid rgba(92,216,160,0.28);
    }

    /* ── alpha delta ── */
    .alpha-container {
        text-align: center;
        padding: 0.8rem 0;
        margin-top: 0.5rem;
    }
    .alpha-value {
        font-family: 'Fira Code', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }
    .alpha-positive { color: #5cd8a0; }
    .alpha-negative { color: #ef6b6b; }
    .alpha-neutral  { color: #edc04a; }
    .alpha-label {
        font-size: 0.67rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #596175;
        margin-top: 0.3rem;
    }

    /* ── data cards ── */
    .data-card {
        background: #181b22;
        border: 1px solid #232730;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .data-card-value {
        font-family: 'Fira Code', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: #e4e9f2;
    }
    .data-card-label {
        font-size: 0.66rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #596175;
        margin-top: 0.3rem;
    }

    /* ── order tickets ── */
    .order-ticket {
        background: #181b22;
        border: 1px solid #232730;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.15s;
    }
    .order-ticket:hover {
        border-color: #353a48;
    }
    .order-contract {
        font-size: 0.85rem;
        font-weight: 500;
        color: #cdd5e0;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }
    .order-prices {
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .order-yes {
        font-family: 'Fira Code', monospace;
        color: #5cd8a0;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .order-no {
        font-family: 'Fira Code', monospace;
        color: #ef6b6b;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .order-meta {
        font-family: 'Fira Code', monospace;
        color: #596175;
        font-size: 0.7rem;
    }
    .order-sector {
        font-family: 'Fira Code', monospace;
        display: inline-block;
        background: #232730;
        color: #7b8498;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-left: 0.5rem;
    }
    .order-alpha {
        font-family: 'Fira Code', monospace;
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-left: 0.5rem;
    }
    .order-alpha-pos {
        background: rgba(92,216,160,0.12);
        color: #5cd8a0;
        border: 1px solid rgba(92,216,160,0.22);
    }
    .order-alpha-neg {
        background: rgba(239,107,107,0.08);
        color: #ef6b6b;
        border: 1px solid rgba(239,107,107,0.18);
    }

    /* ── buttons ── */
    .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.73rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
        border-radius: 6px !important;
        padding: 0.4rem 1.1rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"] {
        background-color: rgba(77,217,203,0.12) !important;
        color: #4dd9cb !important;
        border: 1px solid rgba(77,217,203,0.32) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: rgba(77,217,203,0.22) !important;
        border-color: #4dd9cb !important;
    }
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: rgba(239,107,107,0.1) !important;
        color: #ef6b6b !important;
        border: 1px solid rgba(239,107,107,0.28) !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: rgba(239,107,107,0.2) !important;
        border-color: #ef6b6b !important;
    }

    /* ── impact tiles ── */
    .impact-tile {
        background: #181b22;
        border: 1px solid #232730;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .impact-bullish { border-left: 3px solid #5cd8a0; }
    .impact-bearish { border-left: 3px solid #ef6b6b; }
    .impact-neutral { border-left: 3px solid #596175; }

    /* ── topic tag ── */
    .topic-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* ── regime tag ── */
    .regime-tag {
        font-family: 'Fira Code', monospace;
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.68rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .regime-high {
        background: rgba(239,107,107,0.1);
        color: #ef6b6b;
        border: 1px solid rgba(239,107,107,0.22);
    }
    .regime-dormant {
        background: rgba(92,216,160,0.1);
        color: #5cd8a0;
        border: 1px solid rgba(92,216,160,0.22);
    }

    /* ── section headers ── */
    .section-header {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #7b8498;
        border-bottom: 1px solid #232730;
        padding-bottom: 0.45rem;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    /* ── tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #232730;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        padding: 0.6rem 1.4rem;
        color: #596175;
        border: none;
        border-bottom: 2px solid transparent;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #7b8498;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid #4dd9cb !important;
        color: #e4e9f2 !important;
    }

    /* ── misc ── */
    hr {
        border: none;
        border-top: 1px solid #232730;
        margin: 1rem 0;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #111318; }
    ::-webkit-scrollbar-thumb { background: #232730; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #353a48; }

    .stAlert { border-radius: 6px !important; }
    .stExpander { border: 1px solid #232730 !important; border-radius: 6px !important; }

    div[data-testid="stMetric"] {
        background: #181b22;
        border: 1px solid #232730;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.66rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #596175 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'Fira Code', monospace !important;
        color: #e4e9f2 !important;
    }

    .terminal-footer {
        text-align: center;
        color: #353a48;
        font-size: 0.7rem;
        letter-spacing: 0.01em;
        padding-top: 1.5rem;
        border-top: 1px solid #181b22;
        margin-top: 2rem;
    }

    .feed-item {
        font-size: 0.82rem;
        color: #7b8498;
        padding: 0.45rem 0;
        border-bottom: 1px solid #181b22;
        line-height: 1.5;
    }
    .feed-index {
        font-family: 'Fira Code', monospace;
        color: #353a48;
        font-size: 0.7rem;
        margin-right: 0.5rem;
    }

    .standby-box {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #353a48;
    }
    .standby-box .standby-label {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #5cd8a0;
        margin-bottom: 0.5rem;
    }
    .standby-box .standby-detail {
        font-size: 0.74rem;
        color: #596175;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Auto-Sync Button
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="sidebar-section">Automation</div>',
    unsafe_allow_html=True,
)
st.sidebar.button(
    "SYNC TO LIVE",
    on_click=_on_auto_sync,
    type="primary",
    use_container_width=True,
    help="Auto-set all parameters to current real-time values from the dataset",
)
if st.session_state.get("auto_synced"):
    now_label = datetime.now(EST).strftime("%H:%M:%S EST")
    st.sidebar.markdown(
        f"<span class='mono' style='color:#5cd8a0; font-size:0.65rem;'>"
        f"SYNCED @ {now_label}</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar — Parameter Controls (keyed for auto-sync)
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-section">System Parameters</div>', unsafe_allow_html=True)

sim_hour = st.sidebar.slider(
    "HOUR (EST)", min_value=0, max_value=23, value=10,
    key="k_sim_hour",
    help="Simulated hour of day in Eastern Time"
)
hours_since = st.sidebar.slider(
    "GAP (HRS SINCE LAST)", min_value=0.0, max_value=24.0, value=1.5, step=0.25,
    key="k_hours_since",
)
post_count_24h = st.sidebar.slider(
    "ROLLING COUNT (24H)", min_value=0, max_value=50, value=8,
    key="k_post_count_24h",
)
posting_velocity = st.sidebar.slider(
    "VELOCITY (POSTS/HR, 4H AVG)", min_value=0.0, max_value=5.0, value=0.5, step=0.25,
    key="k_posting_velocity",
)

st.sidebar.markdown('<div class="sidebar-section">Market Context</div>', unsafe_allow_html=True)
news_volume = st.sidebar.slider(
    "NEWS VOLUME (4H)", min_value=0, max_value=50, value=10,
    key="k_news_volume",
    help="Breaking news article count in last 4 hours"
)
news_sentiment = st.sidebar.slider(
    "NEWS SENTIMENT", min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
    key="k_news_sentiment",
    help="Aggregate news sentiment (-1=bearish, +1=bullish)"
)
vix_level = st.sidebar.slider(
    "VIX (4H TRAILING)", min_value=10.0, max_value=50.0, value=18.0, step=0.5,
    key="k_vix_level",
    help="Simulated CBOE Volatility Index"
)

st.sidebar.markdown('<div class="sidebar-section">Market Filter</div>', unsafe_allow_html=True)
sector_options = ["All"] + list(SECTORS.keys())
selected_sector = st.sidebar.selectbox("SECTOR", sector_options, index=0)

st.sidebar.markdown('<div class="sidebar-section">Input Feed</div>', unsafe_allow_html=True)
default_tweets = (
    "The fake news media is LYING about our great economy. Sad!\n"
    "Just spoke with President Xi. TREMENDOUS progress on trade!\n"
    "Tariffs on China are WORKING. Numbers are the best ever!\n"
    "The radical left wants to destroy everything we built.\n"
    "MAKE AMERICA GREAT AGAIN! Our country is coming back!"
)
tweets_text = st.sidebar.text_area(
    "RECENT POSTS (1 PER LINE)",
    value=default_tweets,
    height=160,
    key="k_tweets_text",
)
recent_tweets = [t.strip() for t in tweets_text.strip().split("\n") if t.strip()]

use_llm = st.sidebar.checkbox(
    "USE LLM TOPIC CLASSIFIER", value=False,
    help="Requires OPENAI_API_KEY env var"
)

# ---------------------------------------------------------------------------
# Run the agent cycle
# ---------------------------------------------------------------------------
result = run_agent_cycle(
    current_time_hour=sim_hour,
    hours_since_last=hours_since,
    post_count_24h=post_count_24h,
    recent_tweets=recent_tweets,
    use_llm=use_llm,
    sector_filter=selected_sector if selected_sector != "All" else None,
    posting_velocity_4h=posting_velocity,
    news_volume_4h=float(news_volume),
    news_sentiment=news_sentiment,
    simulated_vix_4h_vol=vix_level,
)

prob = result["probability"]
signal = result["signal"]
topic = result["predicted_topic"]
contracts = result["contracts"]
market_impacts = result.get("market_impacts", [])
arbitrage = result.get("arbitrage", {})
regime = result.get("regime", "DORMANT")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="terminal-header">'
    'MentionBot <span class="terminal-header-accent">|</span> Prediction Engine'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="terminal-subtitle">'
    'Real-time post probability &bull; prediction market routing &bull; alpha arbitrage'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["LIVE RADAR", "TIME TRAVEL", "MARKET IMPACT", "MODEL DIAGNOSTICS"])

# ===================================================================
# TAB 1 — LIVE RADAR
# ===================================================================
with tab1:
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_left:
        # Regime indicator
        regime_class = "regime-high" if regime == "HIGH-ACTIVITY" else "regime-dormant"
        st.markdown(
            f"<div style='text-align:center; padding:0.5rem 0;'>"
            f"<span class='regime-tag {regime_class}'>Regime: {regime}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # VIX indicator
        vix_color = "#ef6b6b" if vix_level > 30 else "#edc04a" if vix_level > 22 else "#5cd8a0"
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'><span class='mono' style='color:{vix_color}'>{vix_level:.1f}</span></div>"
            f"<div class='data-card-label'>VIX (4H Trailing)</div></div>",
            unsafe_allow_html=True,
        )

        # News sentiment
        sent_color = "#5cd8a0" if news_sentiment > 0.2 else "#ef6b6b" if news_sentiment < -0.2 else "#edc04a"
        st.markdown(
            f"<div class='data-card' style='margin-top:0.5rem;'>"
            f"<div class='data-card-value'><span class='mono' style='color:{sent_color}'>{news_sentiment:+.2f}</span></div>"
            f"<div class='data-card-label'>News Sentiment</div></div>",
            unsafe_allow_html=True,
        )

        # News volume
        st.markdown(
            f"<div class='data-card' style='margin-top:0.5rem;'>"
            f"<div class='data-card-value'><span class='mono'>{news_volume}</span></div>"
            f"<div class='data-card-label'>News Volume (4H)</div></div>",
            unsafe_allow_html=True,
        )

    with col_center:
        # Probability gauge
        if prob >= 0.60:
            gauge_class = "gauge-hot"
            bar_color = "#ef6b6b"
        elif prob >= 0.40:
            gauge_class = "gauge-warm"
            bar_color = "#edc04a"
        else:
            gauge_class = "gauge-cold"
            bar_color = "#5cd8a0"

        st.markdown(
            f'<div class="gauge-container">'
            f'<div class="gauge-value {gauge_class}">{prob:.0%}</div>'
            f'<div class="gauge-label">P(post) next 60 min</div>'
            f'<div class="gauge-bar">'
            f'<div class="gauge-bar-fill" style="width:{prob*100:.0f}%; background:{bar_color};"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        signal_class_map = {"HOT": "signal-hot", "WARM": "signal-warm", "COLD": "signal-cold"}
        signal_class = signal_class_map.get(signal, "signal-cold")
        st.markdown(
            f"<p style='text-align:center; margin-top:0.5rem;'>"
            f"<span class='signal-tag {signal_class}'>Signal: {signal}</span></p>",
            unsafe_allow_html=True,
        )

    with col_right:
        # Alpha Delta display
        alpha_delta = arbitrage.get("alpha_delta", 0.0)
        poly_baseline = arbitrage.get("polymarket_baseline", 0.5)
        has_alpha = arbitrage.get("has_alpha", False)

        if alpha_delta >= 0.10:
            alpha_class = "alpha-positive"
        elif alpha_delta > 0:
            alpha_class = "alpha-neutral"
        else:
            alpha_class = "alpha-negative"

        st.markdown(
            f"<div class='alpha-container'>"
            f"<div class='alpha-value {alpha_class}'>{alpha_delta:+.1%}</div>"
            f"<div class='alpha-label'>Alpha Delta</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Polymarket baseline
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'><span class='mono'>{poly_baseline:.0%}</span></div>"
            f"<div class='data-card-label'>Polymarket Baseline</div></div>",
            unsafe_allow_html=True,
        )

        # Alpha status
        alpha_status_color = "#5cd8a0" if has_alpha else "#596175"
        alpha_status_text = "EDGE DETECTED" if has_alpha else "NO EDGE"
        st.markdown(
            f"<div style='text-align:center; padding:0.5rem 0; margin-top:0.5rem;'>"
            f"<span class='mono' style='color:{alpha_status_color}; font-size:0.7rem; "
            f"font-weight:600; letter-spacing:0.1em;'>{alpha_status_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Metrics row
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'><span class='mono'>{sim_hour:02d}:00</span></div>"
            f"<div class='data-card-label'>Hour EST</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'><span class='mono'>{hours_since:.1f}h</span></div>"
            f"<div class='data-card-label'>Since Last Post</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'><span class='mono'>{post_count_24h}</span></div>"
            f"<div class='data-card-label'>Posts 24H</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'>{topic or '\u2014'}</div>"
            f"<div class='data-card-label'>Predicted Topic</div></div>",
            unsafe_allow_html=True,
        )
    with m5:
        sector_display = selected_sector if selected_sector != "All" else "ALL"
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'>{sector_display}</div>"
            f"<div class='data-card-label'>Sector Filter</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Two columns: Context + Execution
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="section-header">Input Context</div>', unsafe_allow_html=True)
        for i, tweet in enumerate(recent_tweets[:5], 1):
            st.markdown(
                f"<div class='feed-item'>"
                f"<span class='feed-index'>{i:02d}</span>"
                f"{tweet[:200]}{'...' if len(tweet) > 200 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if topic:
            st.markdown('<div class="section-header">Topic Classification</div>', unsafe_allow_html=True)
            topic_colors = {
                "Tariffs": "#ef6b6b", "Crypto": "#edc04a", "Media": "#4dd9cb",
                "Borders": "#a78bfa", "Fed": "#F0883E", "Cabinet": "#DB61A2",
                "Other": "#596175",
            }
            color = topic_colors.get(topic, "#596175")
            st.markdown(
                f"<span class='topic-tag' style='background: {color}22; color: {color}; border: 1px solid {color}55;'>{topic}</span>",
                unsafe_allow_html=True,
            )

            # Market impact preview
            if market_impacts:
                st.markdown("")
                st.markdown('<div class="section-header">Impact Preview</div>', unsafe_allow_html=True)
                for mi in market_impacts[:3]:
                    if "Bullish" in mi["direction"]:
                        dir_color = "#5cd8a0"
                        css = "impact-bullish"
                    elif "Bearish" in mi["direction"]:
                        dir_color = "#ef6b6b"
                        css = "impact-bearish"
                    else:
                        dir_color = "#596175"
                        css = "impact-neutral"
                    st.markdown(
                        f"<div class='impact-tile {css}'>"
                        f"<strong>{mi['sector']}</strong> "
                        f"<span class='mono' style='color:{dir_color}; font-weight:700;'>"
                        f"{mi['expected_move_pct']:+.2f}%</span> "
                        f"<span style='color:#596175; font-size:0.75rem;'>{mi['direction']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Below threshold. Topic classification inactive.")

    with right_col:
        st.markdown('<div class="section-header">Execution Engine</div>', unsafe_allow_html=True)

        if signal == "HOT" and contracts:
            # Show Alpha Delta banner at top of execution panel
            st.markdown(
                f"<div style='background: rgba(92,216,160,0.08); border: 1px solid rgba(92,216,160,0.25); "
                f"border-radius: 3px; padding: 0.6rem 1rem; margin-bottom: 0.8rem; text-align: center;'>"
                f"<span class='mono' style='color: #5cd8a0; font-size: 0.75rem; font-weight: 600; "
                f"letter-spacing: 0.1em;'>ALPHA DETECTED: {alpha_delta:+.1%} EDGE VS POLYMARKET</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<p style='font-family: Fira Code, monospace; font-size: 0.7rem; "
                f"color: #5cd8a0; letter-spacing: 0.1em; text-transform: uppercase;'>"
                f"ACTIVE &mdash; {len(contracts)} contracts matched for {topic}</p>",
                unsafe_allow_html=True,
            )
            for c in contracts:
                sector_tag = f"<span class='order-sector'>{c.get('sector', '')}</span>" if c.get('sector') else ""

                # Alpha tag per contract
                contract_alpha = c.get("contract_alpha", 0)
                if contract_alpha >= 0.10:
                    alpha_tag = f"<span class='order-alpha order-alpha-pos'>ALPHA {contract_alpha:+.0%}</span>"
                elif contract_alpha > 0:
                    alpha_tag = f"<span class='order-alpha order-alpha-neg'>+{contract_alpha:.0%}</span>"
                else:
                    alpha_tag = f"<span class='order-alpha order-alpha-neg'>{contract_alpha:+.0%}</span>"

                st.markdown(
                    f"<div class='order-ticket'>"
                    f"<div class='order-contract'>{c['contract']}{sector_tag}{alpha_tag}</div>"
                    f"<div class='order-prices'>"
                    f"<span class='order-yes'>YES <span class='mono'>{c['yes_price']:.0%}</span></span>"
                    f"<span class='order-no'>NO <span class='mono'>{c['no_price']:.0%}</span></span>"
                    f"<span class='order-meta'><span class='mono'>${c['volume']:,.0f}</span> vol // {c['market']}</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                btn_left, btn_right, _ = st.columns([1, 1, 2])
                contract_id = c["contract"][:30].replace(" ", "_")
                with btn_left:
                    if st.button(
                        f"BUY YES @ {c['yes_price']:.0%}",
                        key=f"yes_{contract_id}",
                        type="primary",
                    ):
                        st.success(f"ORDER FILLED: YES @ {c['yes_price']:.0%} on {c['market']}")
                with btn_right:
                    if st.button(
                        f"BUY NO @ {c['no_price']:.0%}",
                        key=f"no_{contract_id}",
                    ):
                        st.success(f"ORDER FILLED: NO @ {c['no_price']:.0%} on {c['market']}")

        elif signal == "WARM":
            st.markdown(
                f"<div style='background: rgba(237,192,74,0.08); border: 1px solid rgba(237,192,74,0.25); "
                f"border-radius: 3px; padding: 0.8rem 1rem; text-align: center;'>"
                f"<span class='mono' style='color: #edc04a; font-size: 0.75rem; font-weight: 600; "
                f"letter-spacing: 0.1em;'>SIGNAL WARM &mdash; NO ARBITRAGE EDGE</span><br>"
                f"<span class='mono' style='color: #596175; font-size: 0.65rem;'>"
                f"ML: {prob:.0%} vs Polymarket: {poly_baseline:.0%} &mdash; "
                f"Alpha: {alpha_delta:+.1%} (need +10%)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                "<div class='standby-box'>"
                "<div class='standby-label' style='color:#edc04a;'>Monitoring</div>"
                "<div class='standby-detail'>Probability above base threshold but insufficient alpha.</div>"
                "<div class='standby-detail' style='margin-top:0.3rem;'>Awaiting +10% edge over crowd consensus.</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        elif signal == "HOT":
            st.warning("Signal active. No contracts matched the current sector filter.")
        else:
            st.markdown(
                "<div class='standby-box'>"
                "<div class='standby-label'>Standby</div>"
                "<div class='standby-detail'>Probability below threshold. No orders routed.</div>"
                "<div class='standby-detail' style='margin-top:0.3rem;'>Engine monitoring. Awaiting signal.</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ===================================================================
# TAB 2 — TIME TRAVEL
# ===================================================================
with tab2:
    st.markdown('<div class="section-header">Historical Replay Engine</div>', unsafe_allow_html=True)

    try:
        from time_travel import get_available_dates, run_full_day_replay

        @st.cache_data(ttl=300)
        def _get_dates():
            return get_available_dates()

        available_dates = _get_dates()

        if not available_dates:
            st.warning("No historical data available for replay.")
        else:
            tt_col1, tt_col2 = st.columns([1, 2])
            with tt_col1:
                selected_date = st.selectbox(
                    "TARGET DATE",
                    available_dates[:60],
                    index=0,
                    format_func=lambda d: f"{d} ({pd.Timestamp(d).strftime('%A')})"
                )

                if st.button("EXECUTE REPLAY", type="primary"):
                    st.session_state["replay_date"] = selected_date

            replay_date = st.session_state.get("replay_date", None)
            if replay_date:
                with st.spinner(f"Replaying {replay_date}..."):
                    @st.cache_data(ttl=300)
                    def _run_replay(date):
                        return run_full_day_replay(date)

                    replay = _run_replay(replay_date)

                with tt_col2:
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    sm1.metric("Date", f"{replay['date']}")
                    sm2.metric("Total Posts", replay["total_posts"])
                    sm3.metric("Accuracy", f"{replay['accuracy']:.0%}")
                    sm4.metric("Day", replay["day_of_week"])

                st.markdown("---")

                # Timeline chart
                frames = replay["frames"]
                hours = [f["hour"] for f in frames]
                predictions = [f["prediction"] for f in frames]
                actual = [f["actual_posts"] for f in frames]
                hits = [f["hit"] for f in frames]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=hours, y=predictions,
                    mode="lines+markers",
                    name="Prediction",
                    line=dict(color="#4dd9cb", width=2),
                    marker=dict(size=6, color=["#5cd8a0" if h else "#ef6b6b" for h in hits]),
                    hovertemplate="Hour: %{x}:00<br>P(post): %{y:.1%}<extra></extra>",
                ))

                fig.add_hline(y=0.5, line_dash="dash", line_color="#edc04a",
                              annotation_text="THRESHOLD", annotation_position="top right",
                              annotation_font=dict(family="Fira Code", size=10, color="#edc04a"))

                fig.add_trace(go.Bar(
                    x=hours, y=[a / max(max(actual), 1) for a in actual],
                    name="Actual Posts (scaled)",
                    marker_color=["rgba(92,216,160,0.2)" if a > 0 else "rgba(128,128,128,0.05)" for a in actual],
                    hovertemplate="Hour: %{x}:00<br>Posts: " + str(actual) + "<extra></extra>",
                ))

                fig.update_layout(
                    title=dict(
                        text=f"REPLAY: {replay['date']}",
                        font=dict(family="Fira Code", size=14),
                    ),
                    xaxis_title="Hour (EST)",
                    yaxis_title="Probability",
                    template="plotly_dark",
                    paper_bgcolor="#111318",
                    plot_bgcolor="#111318",
                    height=400,
                    legend=dict(orientation="h", y=1.12, font=dict(family="Fira Code", size=10)),
                    xaxis=dict(dtick=1, range=[-0.5, 23.5], gridcolor="#181b22"),
                    yaxis=dict(range=[0, 1.05], gridcolor="#181b22"),
                    font=dict(family="Fira Code"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Hour-by-hour detail
                st.markdown('<div class="section-header">Hour-by-Hour Log</div>', unsafe_allow_html=True)
                for f in frames:
                    if f["actual_posted"] or f["prediction"] >= 0.4:
                        hit_mark = "HIT" if f["hit"] else "MISS"
                        hit_color = "#5cd8a0" if f["hit"] else "#ef6b6b"
                        post_info = f"**{f['actual_posts']} post(s)**" if f["actual_posted"] else "_(quiet)_"
                        st.markdown(
                            f"<span class='mono' style='color:{hit_color}; font-size:0.75rem;'>[{hit_mark}]</span> "
                            f"**{f['time_label']}** — "
                            f"Predicted: `{f['prediction']:.0%}` [{f['signal']}] | "
                            f"Actual: {post_info}",
                            unsafe_allow_html=True,
                        )
                        if f["tweets_this_hour"]:
                            for tweet in f["tweets_this_hour"][:2]:
                                st.caption(f"    > _{tweet[:150]}{'...' if len(tweet) > 150 else ''}_")

    except Exception as e:
        st.error(f"Replay module error: {e}")
        st.info("Ensure the model is trained: `python ml_engine.py`")


# ===================================================================
# TAB 3 — MARKET IMPACT
# ===================================================================
with tab3:
    st.markdown('<div class="section-header">Topic / Sector Impact Matrix</div>', unsafe_allow_html=True)

    impact_matrix = get_topic_sector_matrix()
    topics = list(impact_matrix.keys())
    sectors_list = list(SECTORS.keys())

    z_data = []
    hover_data = []
    for topic_name in topics:
        row = []
        hover_row = []
        for sector_name in sectors_list:
            impact = impact_matrix[topic_name].get(sector_name, {})
            move = impact.get("avg_move", 0)
            row.append(move)
            hover_row.append(
                f"Topic: {topic_name}<br>"
                f"Sector: {sector_name}<br>"
                f"Avg Move: {move:+.1f}%<br>"
                f"Confidence: {impact.get('confidence', 0):.0%}<br>"
                f"{impact.get('note', '')}"
            )
        z_data.append(row)
        hover_data.append(hover_row)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=z_data,
        x=sectors_list,
        y=topics,
        colorscale=[
            [0, "rgb(239,107,107)"],
            [0.35, "rgb(100,46,46)"],
            [0.5, "rgb(24,27,34)"],
            [0.65, "rgb(32,92,52)"],
            [1, "rgb(92,216,160)"],
        ],
        zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#cdd5e0", "family": "Fira Code"},
        hovertext=hover_data,
        hoverinfo="text",
        colorbar=dict(title=dict(text="Impact %", side="right")),
    ))

    fig_heatmap.update_layout(
        title=dict(
            text="IMPACT MATRIX: AVG % MOVE BY TOPIC / SECTOR",
            font=dict(family="Fira Code", size=13),
        ),
        template="plotly_dark",
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        height=400,
        xaxis_title="Sector",
        yaxis_title="Topic",
        font=dict(family="Fira Code"),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # Detailed impact for selected topic
    impact_topic = st.selectbox("DRILL INTO TOPIC:", topics, index=0)
    impacts = get_impact_for_topic(impact_topic)

    imp_cols = st.columns(min(len(impacts), 3))
    for idx, impact in enumerate(impacts[:6]):
        col = imp_cols[idx % len(imp_cols)]
        with col:
            if impact["direction"] > 0:
                dir_color = "#5cd8a0"
                dir_text = "BULLISH"
                css = "impact-bullish"
            elif impact["direction"] < 0:
                dir_color = "#ef6b6b"
                dir_text = "BEARISH"
                css = "impact-bearish"
            else:
                dir_color = "#596175"
                dir_text = "NEUTRAL"
                css = "impact-neutral"
            st.markdown(
                f"<div class='impact-tile {css}'>"
                f"<strong style='font-size:0.95rem; color:#cdd5e0;'>{impact['sector']}</strong><br>"
                f"<span class='mono' style='color:{dir_color}; font-size:1.6rem; font-weight:700;'>"
                f"{impact['avg_move']:+.1f}%</span><br>"
                f"<span class='mono' style='color:#596175; font-size:0.65rem;'>{dir_text} // "
                f"CONF {impact['confidence']:.0%}</span><br>"
                f"<span class='mono' style='color:#353a48; font-size:0.6rem;'>{', '.join(impact['tickers'])}</span><br>"
                f"<span style='color:#353a48; font-size:0.65rem;'>{impact['note']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # P&L Simulation
    st.markdown('<div class="section-header">Simulated Trading P&L</div>', unsafe_allow_html=True)

    demo_trades = generate_demo_trade_history(30)
    pnl = calculate_pnl(demo_trades)

    pnl_c1, pnl_c2, pnl_c3, pnl_c4 = st.columns(4)
    pnl_color = "#5cd8a0" if pnl["total_pnl"] > 0 else "#ef6b6b"
    pnl_c1.markdown(
        f"<div class='data-card'><div class='data-card-value'>"
        f"<span class='mono' style='color:{pnl_color}'>${pnl['total_pnl']:+.2f}</span></div>"
        f"<div class='data-card-label'>Total P&L</div></div>",
        unsafe_allow_html=True,
    )
    pnl_c2.markdown(
        f"<div class='data-card'><div class='data-card-value'>"
        f"<span class='mono'>{pnl['win_rate']:.0%}</span></div>"
        f"<div class='data-card-label'>Win Rate</div></div>",
        unsafe_allow_html=True,
    )
    pnl_c3.markdown(
        f"<div class='data-card'><div class='data-card-value'>"
        f"<span class='mono'>{pnl['wins']}</span></div>"
        f"<div class='data-card-label'>Wins</div></div>",
        unsafe_allow_html=True,
    )
    pnl_c4.markdown(
        f"<div class='data-card'><div class='data-card-value'>"
        f"<span class='mono'>{pnl['losses']}</span></div>"
        f"<div class='data-card-label'>Losses</div></div>",
        unsafe_allow_html=True,
    )

    # P&L timeline chart
    if pnl["pnl_timeline"]:
        fig_pnl = go.Figure()
        pnl_values = [p["cumulative_pnl"] for p in pnl["pnl_timeline"]]
        fig_pnl.add_trace(go.Scatter(
            x=list(range(len(pnl_values))),
            y=pnl_values,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#4dd9cb", width=2),
            fillcolor="rgba(77,217,203,0.08)",
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="#232730")
        fig_pnl.update_layout(
            title=dict(
                text="CUMULATIVE P&L",
                font=dict(family="Fira Code", size=13),
            ),
            xaxis_title="Trade #",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            paper_bgcolor="#111318",
            plot_bgcolor="#111318",
            height=300,
            font=dict(family="Fira Code"),
            xaxis=dict(gridcolor="#181b22"),
            yaxis=dict(gridcolor="#181b22"),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)


# ===================================================================
# TAB 4 — MODEL DIAGNOSTICS
# ===================================================================
with tab4:
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    try:
        from ml_engine import get_model_metrics, get_hourly_heatmap_data

        metrics = get_model_metrics()

        if metrics:
            mc1, mc2, mc3, mc4 = st.columns(4)
            acc_color = "#5cd8a0" if metrics.get("accuracy", 0) > 0.65 else "#edc04a"
            mc1.markdown(
                f"<div class='data-card'><div class='data-card-value'>"
                f"<span class='mono' style='color:{acc_color}'>{metrics.get('accuracy', 0):.1%}</span></div>"
                f"<div class='data-card-label'>Accuracy</div></div>",
                unsafe_allow_html=True,
            )
            mc2.markdown(
                f"<div class='data-card'><div class='data-card-value'>"
                f"<span class='mono'>{metrics.get('precision_post', 0):.1%}</span></div>"
                f"<div class='data-card-label'>Precision (Post)</div></div>",
                unsafe_allow_html=True,
            )
            mc3.markdown(
                f"<div class='data-card'><div class='data-card-value'>"
                f"<span class='mono'>{metrics.get('recall_post', 0):.1%}</span></div>"
                f"<div class='data-card-label'>Recall (Post)</div></div>",
                unsafe_allow_html=True,
            )
            mc4.markdown(
                f"<div class='data-card'><div class='data-card-value'>"
                f"<span class='mono'>{metrics.get('f1_post', 0):.1%}</span></div>"
                f"<div class='data-card-label'>F1 Score</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # Feature importance chart
            fi = metrics.get("feature_importances", {})
            if fi:
                features_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
                feat_names = [f[0].replace("_", " ").title() for f in features_sorted]
                feat_values = [f[1] for f in features_sorted]

                fig_fi = go.Figure(go.Bar(
                    x=feat_values,
                    y=feat_names,
                    orientation="h",
                    marker=dict(
                        color=feat_values,
                        colorscale=[[0, "#232730"], [0.5, "#4dd9cb"], [1, "#a78bfa"]],
                    ),
                    text=[f"{v:.1%}" for v in feat_values],
                    textposition="auto",
                    textfont=dict(family="Fira Code", size=10),
                ))
                fig_fi.update_layout(
                    title=dict(
                        text="FEATURE IMPORTANCE (GRADIENT BOOSTING — 16 FEATURES)",
                        font=dict(family="Fira Code", size=13),
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#111318",
                    plot_bgcolor="#111318",
                    height=500,
                    yaxis=dict(autorange="reversed", gridcolor="#181b22"),
                    xaxis_title="Importance",
                    font=dict(family="Fira Code"),
                    xaxis=dict(gridcolor="#181b22"),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # Confusion matrix
            cm = metrics.get("confusion_matrix")
            if cm:
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Pred: No Post", "Pred: Post"],
                    y=["Actual: No Post", "Actual: Post"],
                    colorscale=[[0, "#111318"], [1, "#4dd9cb"]],
                    text=[[str(v) for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont={"size": 16, "color": "#cdd5e0", "family": "Fira Code"},
                    showscale=False,
                ))
                fig_cm.update_layout(
                    title=dict(
                        text="CONFUSION MATRIX",
                        font=dict(family="Fira Code", size=13),
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#111318",
                    plot_bgcolor="#111318",
                    height=350,
                    font=dict(family="Fira Code"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            # Classification report
            report = metrics.get("classification_report", "")
            if report:
                with st.expander("FULL CLASSIFICATION REPORT"):
                    st.code(report)

            # Training data stats
            st.markdown("---")
            st.markdown('<div class="section-header">Training Data</div>', unsafe_allow_html=True)
            ts1, ts2, ts3 = st.columns(3)
            ts1.metric("Training Samples", f"{metrics.get('n_train', 0):,}")
            ts2.metric("Test Samples", f"{metrics.get('n_test', 0):,}")
            ts3.metric("Positive Rate", f"{metrics.get('positive_rate', 0):.1%}")

        else:
            st.warning("No model metrics found. Run `python ml_engine.py` to train the model.")

        st.markdown("---")

        # Posting frequency heatmap
        st.markdown('<div class="section-header">Posting Frequency Heatmap</div>', unsafe_allow_html=True)

        try:
            heatmap_data = get_hourly_heatmap_data(csv_path="data/truth_archive.csv")
            if not heatmap_data.empty:
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                hour_labels = [f"{h:02d}:00" for h in range(24)]

                fig_freq = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=day_names,
                    y=hour_labels,
                    colorscale=[[0, "#111318"], [0.3, "#181b22"], [0.6, "#4dd9cb"], [1, "#a78bfa"]],
                    text=[[f"{v:.1f}" for v in row] for row in heatmap_data.values],
                    texttemplate="%{text}",
                    textfont={"size": 9, "family": "Fira Code"},
                    colorbar=dict(title="Avg Posts"),
                ))
                fig_freq.update_layout(
                    title=dict(
                        text="POSTING FREQUENCY: AVG POSTS/HR BY DAY",
                        font=dict(family="Fira Code", size=13),
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#111318",
                    plot_bgcolor="#111318",
                    height=600,
                    xaxis_title="Day of Week",
                    yaxis_title="Hour (EST)",
                    yaxis=dict(autorange="reversed"),
                    font=dict(family="Fira Code"),
                )
                st.plotly_chart(fig_freq, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")

    except Exception as e:
        st.error(f"Error loading model diagnostics: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='terminal-footer'>"
    "mentionbot v3 &mdash; built at yhack 2025 &mdash; "
    "gradient boosting &bull; alpha arb &bull; regime detection &bull; vix feedback &bull; 6 sectors"
    "</div>",
    unsafe_allow_html=True,
)
