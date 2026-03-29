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
from datetime import datetime
import json
from pathlib import Path

from market_agent import run_agent_cycle
from market_impact import (
    get_sectors, get_topic_sector_matrix, get_impact_for_topic,
    simulate_market_reaction, generate_demo_trade_history, calculate_pnl,
    SECTORS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MENTIONBOT // Prediction Terminal",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Quant Terminal Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    .viewerBadge_container__r5tak {display: none !important;}

    /* Remove default top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* ===== GLOBAL DARK THEME ===== */
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }

    /* ===== TYPOGRAPHY ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        color: #C9D1D9;
    }

    /* Monospace for all data values */
    .mono {
        font-family: 'JetBrains Mono', 'Roboto Mono', 'Courier New', monospace !important;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #0B0E14;
        border-right: 1px solid #1E2228;
    }
    section[data-testid="stSidebar"] .stSlider > div > div {
        color: #8B949E;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextArea label,
    section[data-testid="stSidebar"] .stCheckbox label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6E7681 !important;
    }

    /* Sidebar section headers */
    .sidebar-section {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #484F58;
        border-bottom: 1px solid #1E2228;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
        margin-top: 1.2rem;
    }

    /* ===== TERMINAL HEADER ===== */
    .terminal-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #E6EDF3;
        letter-spacing: 0.04em;
        margin-bottom: 0.15rem;
    }
    .terminal-header-accent {
        color: #3FB950;
    }
    .terminal-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #484F58;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    /* ===== GAUGE / PROBABILITY DISPLAY ===== */
    .gauge-container {
        text-align: center;
        padding: 1.5rem 0;
    }
    .gauge-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4.5rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .gauge-hot { color: #F85149; }
    .gauge-warm { color: #D29922; }
    .gauge-cold { color: #3FB950; }

    .gauge-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #484F58;
        margin-top: 0.5rem;
    }

    .gauge-bar {
        width: 100%;
        height: 4px;
        background: #161B22;
        border-radius: 2px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .gauge-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.6s ease;
    }

    /* Signal badge */
    .signal-tag {
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 2px;
        font-weight: 600;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .signal-hot {
        background: rgba(248, 81, 73, 0.15);
        color: #F85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .signal-warm {
        background: rgba(210, 153, 34, 0.15);
        color: #D29922;
        border: 1px solid rgba(210, 153, 34, 0.3);
    }
    .signal-cold {
        background: rgba(63, 185, 80, 0.15);
        color: #3FB950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }

    /* ===== ALPHA DELTA DISPLAY ===== */
    .alpha-container {
        text-align: center;
        padding: 0.8rem 0;
        margin-top: 0.5rem;
    }
    .alpha-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }
    .alpha-positive { color: #3FB950; }
    .alpha-negative { color: #F85149; }
    .alpha-neutral { color: #D29922; }
    .alpha-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #484F58;
        margin-top: 0.3rem;
    }

    /* ===== DATA CARDS ===== */
    .data-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .data-card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: #E6EDF3;
    }
    .data-card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #484F58;
        margin-top: 0.3rem;
    }

    /* ===== ORDER TICKET (Contract Cards) ===== */
    .order-ticket {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 3px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .order-ticket:hover {
        border-color: #30363D;
    }
    .order-contract {
        font-size: 0.85rem;
        font-weight: 500;
        color: #C9D1D9;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }
    .order-prices {
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .order-yes {
        font-family: 'JetBrains Mono', monospace;
        color: #3FB950;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .order-no {
        font-family: 'JetBrains Mono', monospace;
        color: #F85149;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .order-meta {
        font-family: 'JetBrains Mono', monospace;
        color: #484F58;
        font-size: 0.7rem;
    }
    .order-sector {
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        background: #21262D;
        color: #8B949E;
        padding: 0.1rem 0.5rem;
        border-radius: 2px;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-left: 0.5rem;
    }
    .order-alpha {
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 2px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-left: 0.5rem;
    }
    .order-alpha-pos {
        background: rgba(63, 185, 80, 0.15);
        color: #3FB950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .order-alpha-neg {
        background: rgba(248, 81, 73, 0.1);
        color: #F85149;
        border: 1px solid rgba(248, 81, 73, 0.2);
    }

    /* ===== TRADE BUTTONS ===== */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border-radius: 2px !important;
        padding: 0.35rem 1rem !important;
        transition: all 0.15s ease !important;
    }

    /* Buy YES button (primary) */
    .stButton > button[kind="primary"] {
        background-color: rgba(63, 185, 80, 0.15) !important;
        color: #3FB950 !important;
        border: 1px solid rgba(63, 185, 80, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: rgba(63, 185, 80, 0.25) !important;
        border-color: #3FB950 !important;
    }

    /* Buy NO button (secondary) */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: rgba(248, 81, 73, 0.12) !important;
        color: #F85149 !important;
        border: 1px solid rgba(248, 81, 73, 0.35) !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: rgba(248, 81, 73, 0.22) !important;
        border-color: #F85149 !important;
    }

    /* ===== IMPACT CARDS ===== */
    .impact-tile {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 3px;
        padding: 0.8rem;
        text-align: center;
    }
    .impact-bullish { border-left: 3px solid #3FB950; }
    .impact-bearish { border-left: 3px solid #F85149; }
    .impact-neutral { border-left: 3px solid #484F58; }

    /* ===== TOPIC TAG ===== */
    .topic-tag {
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 2px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ===== REGIME TAG ===== */
    .regime-tag {
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 2px;
        font-weight: 600;
        font-size: 0.65rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .regime-high {
        background: rgba(248, 81, 73, 0.12);
        color: #F85149;
        border: 1px solid rgba(248, 81, 73, 0.25);
    }
    .regime-dormant {
        background: rgba(63, 185, 80, 0.12);
        color: #3FB950;
        border: 1px solid rgba(63, 185, 80, 0.25);
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #8B949E;
        border-bottom: 1px solid #21262D;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #21262D;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        padding: 0.6rem 1.4rem;
        color: #484F58;
        border: none;
        border-bottom: 2px solid transparent;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #8B949E;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid #58A6FF !important;
        color: #E6EDF3 !important;
    }

    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        border-top: 1px solid #21262D;
        margin: 1rem 0;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #21262D; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #30363D; }

    /* ===== MISC OVERRIDES ===== */
    .stAlert { border-radius: 3px !important; }
    .stExpander { border: 1px solid #21262D !important; border-radius: 3px !important; }

    /* Metric overrides for Streamlit native metrics */
    div[data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 3px;
        padding: 0.6rem 0.8rem;
    }
    div[data-testid="stMetric"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #484F58 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #E6EDF3 !important;
    }

    /* Footer bar */
    .terminal-footer {
        font-family: 'JetBrains Mono', monospace;
        text-align: center;
        color: #30363D;
        font-size: 0.6rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding-top: 1.5rem;
        border-top: 1px solid #161B22;
        margin-top: 2rem;
    }

    /* Context feed */
    .feed-item {
        font-size: 0.8rem;
        color: #8B949E;
        padding: 0.4rem 0;
        border-bottom: 1px solid #161B22;
        line-height: 1.5;
    }
    .feed-index {
        font-family: 'JetBrains Mono', monospace;
        color: #30363D;
        font-size: 0.7rem;
        margin-right: 0.5rem;
    }

    /* Standby state */
    .standby-box {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #30363D;
    }
    .standby-box .standby-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #3FB950;
        margin-bottom: 0.5rem;
    }
    .standby-box .standby-detail {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #484F58;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Parameter Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-section">System Parameters</div>', unsafe_allow_html=True)

sim_hour = st.sidebar.slider(
    "HOUR (EST)", min_value=0, max_value=23, value=10,
    help="Simulated hour of day in Eastern Time"
)
hours_since = st.sidebar.slider(
    "GAP (HRS SINCE LAST)", min_value=0.0, max_value=24.0, value=1.5, step=0.25,
)
post_count_24h = st.sidebar.slider(
    "ROLLING COUNT (24H)", min_value=0, max_value=50, value=8,
)
posting_velocity = st.sidebar.slider(
    "VELOCITY (POSTS/HR, 4H AVG)", min_value=0.0, max_value=5.0, value=0.5, step=0.25,
)

st.sidebar.markdown('<div class="sidebar-section">Market Context</div>', unsafe_allow_html=True)
news_volume = st.sidebar.slider(
    "NEWS VOLUME (4H)", min_value=0, max_value=50, value=10,
    help="Breaking news article count in last 4 hours"
)
news_sentiment = st.sidebar.slider(
    "NEWS SENTIMENT", min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
    help="Aggregate news sentiment (-1=bearish, +1=bullish)"
)
vix_level = st.sidebar.slider(
    "VIX (4H TRAILING)", min_value=10.0, max_value=50.0, value=18.0, step=0.5,
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
    'MENTIONBOT <span class="terminal-header-accent">//</span> PREDICTION TERMINAL'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="terminal-subtitle">'
    'Real-time posting probability engine &mdash; prediction market order routing &mdash; alpha arbitrage'
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
        vix_color = "#F85149" if vix_level > 30 else "#D29922" if vix_level > 22 else "#3FB950"
        st.markdown(
            f"<div class='data-card'>"
            f"<div class='data-card-value'><span class='mono' style='color:{vix_color}'>{vix_level:.1f}</span></div>"
            f"<div class='data-card-label'>VIX (4H Trailing)</div></div>",
            unsafe_allow_html=True,
        )

        # News sentiment
        sent_color = "#3FB950" if news_sentiment > 0.2 else "#F85149" if news_sentiment < -0.2 else "#D29922"
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
            bar_color = "#F85149"
        elif prob >= 0.40:
            gauge_class = "gauge-warm"
            bar_color = "#D29922"
        else:
            gauge_class = "gauge-cold"
            bar_color = "#3FB950"

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
        alpha_status_color = "#3FB950" if has_alpha else "#484F58"
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
                "Tariffs": "#F85149", "Crypto": "#D29922", "Media": "#58A6FF",
                "Borders": "#A371F7", "Fed": "#F0883E", "Cabinet": "#DB61A2",
                "Other": "#484F58",
            }
            color = topic_colors.get(topic, "#484F58")
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
                        dir_color = "#3FB950"
                        css = "impact-bullish"
                    elif "Bearish" in mi["direction"]:
                        dir_color = "#F85149"
                        css = "impact-bearish"
                    else:
                        dir_color = "#484F58"
                        css = "impact-neutral"
                    st.markdown(
                        f"<div class='impact-tile {css}'>"
                        f"<strong>{mi['sector']}</strong> "
                        f"<span class='mono' style='color:{dir_color}; font-weight:700;'>"
                        f"{mi['expected_move_pct']:+.2f}%</span> "
                        f"<span style='color:#484F58; font-size:0.75rem;'>{mi['direction']}</span>"
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
                f"<div style='background: rgba(63,185,80,0.08); border: 1px solid rgba(63,185,80,0.25); "
                f"border-radius: 3px; padding: 0.6rem 1rem; margin-bottom: 0.8rem; text-align: center;'>"
                f"<span class='mono' style='color: #3FB950; font-size: 0.75rem; font-weight: 600; "
                f"letter-spacing: 0.1em;'>ALPHA DETECTED: {alpha_delta:+.1%} EDGE VS POLYMARKET</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<p style='font-family: JetBrains Mono, monospace; font-size: 0.7rem; "
                f"color: #3FB950; letter-spacing: 0.1em; text-transform: uppercase;'>"
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
                f"<div style='background: rgba(210,153,34,0.08); border: 1px solid rgba(210,153,34,0.25); "
                f"border-radius: 3px; padding: 0.8rem 1rem; text-align: center;'>"
                f"<span class='mono' style='color: #D29922; font-size: 0.75rem; font-weight: 600; "
                f"letter-spacing: 0.1em;'>SIGNAL WARM &mdash; NO ARBITRAGE EDGE</span><br>"
                f"<span class='mono' style='color: #484F58; font-size: 0.65rem;'>"
                f"ML: {prob:.0%} vs Polymarket: {poly_baseline:.0%} &mdash; "
                f"Alpha: {alpha_delta:+.1%} (need +10%)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                "<div class='standby-box'>"
                "<div class='standby-label' style='color:#D29922;'>Monitoring</div>"
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
                    line=dict(color="#58A6FF", width=2),
                    marker=dict(size=6, color=["#3FB950" if h else "#F85149" for h in hits]),
                    hovertemplate="Hour: %{x}:00<br>P(post): %{y:.1%}<extra></extra>",
                ))

                fig.add_hline(y=0.5, line_dash="dash", line_color="#D29922",
                              annotation_text="THRESHOLD", annotation_position="top right",
                              annotation_font=dict(family="JetBrains Mono", size=10, color="#D29922"))

                fig.add_trace(go.Bar(
                    x=hours, y=[a / max(max(actual), 1) for a in actual],
                    name="Actual Posts (scaled)",
                    marker_color=["rgba(63,185,80,0.2)" if a > 0 else "rgba(128,128,128,0.05)" for a in actual],
                    hovertemplate="Hour: %{x}:00<br>Posts: " + str(actual) + "<extra></extra>",
                ))

                fig.update_layout(
                    title=dict(
                        text=f"REPLAY: {replay['date']}",
                        font=dict(family="JetBrains Mono", size=14),
                    ),
                    xaxis_title="Hour (EST)",
                    yaxis_title="Probability",
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=400,
                    legend=dict(orientation="h", y=1.12, font=dict(family="JetBrains Mono", size=10)),
                    xaxis=dict(dtick=1, range=[-0.5, 23.5], gridcolor="#161B22"),
                    yaxis=dict(range=[0, 1.05], gridcolor="#161B22"),
                    font=dict(family="JetBrains Mono"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Hour-by-hour detail
                st.markdown('<div class="section-header">Hour-by-Hour Log</div>', unsafe_allow_html=True)
                for f in frames:
                    if f["actual_posted"] or f["prediction"] >= 0.4:
                        hit_mark = "HIT" if f["hit"] else "MISS"
                        hit_color = "#3FB950" if f["hit"] else "#F85149"
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
            [0, "rgb(248,81,73)"],
            [0.35, "rgb(100,40,36)"],
            [0.5, "rgb(22,27,34)"],
            [0.65, "rgb(30,90,45)"],
            [1, "rgb(63,185,80)"],
        ],
        zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#C9D1D9", "family": "JetBrains Mono"},
        hovertext=hover_data,
        hoverinfo="text",
        colorbar=dict(title=dict(text="Impact %", side="right")),
    ))

    fig_heatmap.update_layout(
        title=dict(
            text="IMPACT MATRIX: AVG % MOVE BY TOPIC / SECTOR",
            font=dict(family="JetBrains Mono", size=13),
        ),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=400,
        xaxis_title="Sector",
        yaxis_title="Topic",
        font=dict(family="JetBrains Mono"),
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
                dir_color = "#3FB950"
                dir_text = "BULLISH"
                css = "impact-bullish"
            elif impact["direction"] < 0:
                dir_color = "#F85149"
                dir_text = "BEARISH"
                css = "impact-bearish"
            else:
                dir_color = "#484F58"
                dir_text = "NEUTRAL"
                css = "impact-neutral"
            st.markdown(
                f"<div class='impact-tile {css}'>"
                f"<strong style='font-size:0.95rem; color:#C9D1D9;'>{impact['sector']}</strong><br>"
                f"<span class='mono' style='color:{dir_color}; font-size:1.6rem; font-weight:700;'>"
                f"{impact['avg_move']:+.1f}%</span><br>"
                f"<span class='mono' style='color:#484F58; font-size:0.65rem;'>{dir_text} // "
                f"CONF {impact['confidence']:.0%}</span><br>"
                f"<span class='mono' style='color:#30363D; font-size:0.6rem;'>{', '.join(impact['tickers'])}</span><br>"
                f"<span style='color:#30363D; font-size:0.65rem;'>{impact['note']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # P&L Simulation
    st.markdown('<div class="section-header">Simulated Trading P&L</div>', unsafe_allow_html=True)

    demo_trades = generate_demo_trade_history(30)
    pnl = calculate_pnl(demo_trades)

    pnl_c1, pnl_c2, pnl_c3, pnl_c4 = st.columns(4)
    pnl_color = "#3FB950" if pnl["total_pnl"] > 0 else "#F85149"
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
            line=dict(color="#58A6FF", width=2),
            fillcolor="rgba(88,166,255,0.08)",
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="#21262D")
        fig_pnl.update_layout(
            title=dict(
                text="CUMULATIVE P&L",
                font=dict(family="JetBrains Mono", size=13),
            ),
            xaxis_title="Trade #",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=300,
            font=dict(family="JetBrains Mono"),
            xaxis=dict(gridcolor="#161B22"),
            yaxis=dict(gridcolor="#161B22"),
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
            acc_color = "#3FB950" if metrics.get("accuracy", 0) > 0.65 else "#D29922"
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
                        colorscale=[[0, "#21262D"], [0.5, "#58A6FF"], [1, "#A371F7"]],
                    ),
                    text=[f"{v:.1%}" for v in feat_values],
                    textposition="auto",
                    textfont=dict(family="JetBrains Mono", size=10),
                ))
                fig_fi.update_layout(
                    title=dict(
                        text="FEATURE IMPORTANCE (GRADIENT BOOSTING — 16 FEATURES)",
                        font=dict(family="JetBrains Mono", size=13),
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=500,
                    yaxis=dict(autorange="reversed", gridcolor="#161B22"),
                    xaxis_title="Importance",
                    font=dict(family="JetBrains Mono"),
                    xaxis=dict(gridcolor="#161B22"),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # Confusion matrix
            cm = metrics.get("confusion_matrix")
            if cm:
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Pred: No Post", "Pred: Post"],
                    y=["Actual: No Post", "Actual: Post"],
                    colorscale=[[0, "#0E1117"], [1, "#58A6FF"]],
                    text=[[str(v) for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont={"size": 16, "color": "#C9D1D9", "family": "JetBrains Mono"},
                    showscale=False,
                ))
                fig_cm.update_layout(
                    title=dict(
                        text="CONFUSION MATRIX",
                        font=dict(family="JetBrains Mono", size=13),
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=350,
                    font=dict(family="JetBrains Mono"),
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
                    colorscale=[[0, "#0E1117"], [0.3, "#161B22"], [0.6, "#58A6FF"], [1, "#A371F7"]],
                    text=[[f"{v:.1f}" for v in row] for row in heatmap_data.values],
                    texttemplate="%{text}",
                    textfont={"size": 9, "family": "JetBrains Mono"},
                    colorbar=dict(title="Avg Posts"),
                ))
                fig_freq.update_layout(
                    title=dict(
                        text="POSTING FREQUENCY: AVG POSTS/HR BY DAY",
                        font=dict(family="JetBrains Mono", size=13),
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=600,
                    xaxis_title="Day of Week",
                    yaxis_title="Hour (EST)",
                    yaxis=dict(autorange="reversed"),
                    font=dict(family="JetBrains Mono"),
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
    "MENTIONBOT v3.0 // YHACK 2025 &mdash; "
    "GradientBoosting (16 features) + Alpha Arbitrage + Regime Detection + VIX Feedback &mdash; "
    "6-sector impact analysis"
    "</div>",
    unsafe_allow_html=True,
)
