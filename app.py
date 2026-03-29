"""
app.py — MentionBot Demo Dashboard v2
=======================================
4-tab Streamlit dashboard with dark theme, charts, time travel, and market impact.

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
    page_title="MentionBot — Trump Post Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Professional Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Force dark backgrounds */
    .stApp { background-color: #0a0a1a; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }

    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main header gradient */
    .main-title {
        font-size: 2.2rem; font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .main-subtitle {
        text-align: center; color: #8b949e; font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Probability gauge */
    .prob-hot  { color: #ff6b6b; font-size: 5rem; font-weight: 900; text-align: center;
                 text-shadow: 0 0 40px rgba(255,107,107,0.4); }
    .prob-warm { color: #ffa726; font-size: 5rem; font-weight: 900; text-align: center;
                 text-shadow: 0 0 40px rgba(255,167,38,0.4); }
    .prob-cold { color: #4ecdc4; font-size: 5rem; font-weight: 900; text-align: center;
                 text-shadow: 0 0 40px rgba(78,205,196,0.4); }

    .signal-badge {
        display: inline-block; padding: 0.4rem 1.5rem; border-radius: 2rem;
        font-weight: 800; font-size: 1.1rem; letter-spacing: 0.05em;
    }
    .badge-hot  { background: linear-gradient(135deg, #ff6b6b, #ee5a24);
                  color: white; box-shadow: 0 4px 15px rgba(255,107,107,0.4); }
    .badge-cold { background: linear-gradient(135deg, #4ecdc4, #44bd9d);
                  color: white; box-shadow: 0 4px 15px rgba(78,205,196,0.3); }

    /* Contract cards */
    .contract-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748; border-radius: 1rem;
        padding: 1.2rem; margin-bottom: 0.75rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .contract-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.2);
    }
    .contract-title { font-weight: 700; font-size: 0.95rem; color: #e6e8eb; margin-bottom: 0.6rem; }
    .price-yes { color: #4ecdc4; font-weight: 800; font-size: 1.3rem; }
    .price-no  { color: #ff6b6b; font-weight: 800; font-size: 1.3rem; }
    .volume    { color: #6b7280; font-size: 0.85rem; }
    .sector-tag {
        display: inline-block; background: #2d3748; color: #a0aec0;
        padding: 0.15rem 0.6rem; border-radius: 0.5rem; font-size: 0.75rem;
        margin-left: 0.5rem;
    }

    /* Impact cards */
    .impact-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        border: 1px solid #2d3748; border-radius: 1rem;
        padding: 1rem; margin-bottom: 0.5rem; text-align: center;
    }
    .impact-positive { border-left: 4px solid #4ecdc4; }
    .impact-negative { border-left: 4px solid #ff6b6b; }
    .impact-neutral  { border-left: 4px solid #6b7280; }

    /* Topic badges */
    .topic-badge {
        display: inline-block; padding: 0.4rem 1.2rem; border-radius: 2rem;
        font-weight: 700; font-size: 1.2rem; color: white;
    }

    /* Timeline */
    .timeline-hit  { color: #4ecdc4; }
    .timeline-miss { color: #ff6b6b; }

    /* Metric card */
    .metric-glass {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 1rem; padding: 1rem; text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #e6e8eb; }
    .metric-label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase;
                    letter-spacing: 0.1em; }

    /* Footer */
    .footer {
        text-align: center; color: #4a5568; font-size: 0.8rem;
        padding-top: 2rem; border-top: 1px solid #2d3748; margin-top: 2rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.03);
        border-radius: 0.5rem; padding: 0.5rem 1.5rem;
        color: #8b949e; border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(102,126,234,0.15) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Agent Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🎯 Agent Controls")
st.sidebar.caption("Tune the simulation parameters")

sim_hour = st.sidebar.slider(
    "Current Hour (EST)", min_value=0, max_value=23, value=10,
    help="Simulated hour of day in Eastern Time"
)
hours_since = st.sidebar.slider(
    "Hours Since Last Post", min_value=0.0, max_value=24.0, value=1.5, step=0.25,
)
post_count_24h = st.sidebar.slider(
    "Rolling Post Count (24h)", min_value=0, max_value=50, value=8,
)
posting_velocity = st.sidebar.slider(
    "Posting Velocity (posts/hr, 4h avg)", min_value=0.0, max_value=5.0, value=0.5, step=0.25,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 Market Focus")
sector_options = ["All"] + list(SECTORS.keys())
selected_sector = st.sidebar.selectbox("Focus Sector", sector_options, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Recent Tweets")
default_tweets = (
    "The fake news media is LYING about our great economy. Sad!\n"
    "Just spoke with President Xi. TREMENDOUS progress on trade!\n"
    "Tariffs on China are WORKING. Numbers are the best ever!\n"
    "The radical left wants to destroy everything we built.\n"
    "MAKE AMERICA GREAT AGAIN! Our country is coming back!"
)
tweets_text = st.sidebar.text_area(
    "Enter 1-5 tweets (one per line):",
    value=default_tweets,
    height=160,
)
recent_tweets = [t.strip() for t in tweets_text.strip().split("\n") if t.strip()]

use_llm = st.sidebar.checkbox(
    "Use OpenAI for topic prediction", value=False,
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
)

prob = result["probability"]
signal = result["signal"]
topic = result["predicted_topic"]
contracts = result["contracts"]
market_impacts = result.get("market_impacts", [])

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">MentionBot — Live Posting Radar</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">AI agent that predicts Trump\'s Truth Social activity '
    'and surfaces prediction market trades</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Radar", "⏰ Time Travel", "📊 Market Impact", "🧠 Model Insights"])

# ===================================================================
# TAB 1 — LIVE RADAR
# ===================================================================
with tab1:
    # Probability gauge
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if prob >= 0.60:
            css_class = "prob-hot"
        elif prob >= 0.40:
            css_class = "prob-warm"
        else:
            css_class = "prob-cold"

        st.markdown(
            f'<div class="{css_class}">{prob:.0%}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; font-size:1.1rem; color:#8b949e; margin-top:-1rem;'>"
            "Chance of post in the next hour</p>",
            unsafe_allow_html=True,
        )

        badge_class = "badge-hot" if signal == "HOT" else "badge-cold"
        st.markdown(
            f"<p style='text-align:center'>"
            f"<span class='signal-badge {badge_class}'>⚡ Signal: {signal}</span></p>",
            unsafe_allow_html=True,
        )

    # Metrics row
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"<div class='metric-glass'><div class='metric-value'>{sim_hour}:00</div>"
            f"<div class='metric-label'>Hour (EST)</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='metric-glass'><div class='metric-value'>{hours_since:.1f}h</div>"
            f"<div class='metric-label'>Since Last Post</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='metric-glass'><div class='metric-value'>{post_count_24h}</div>"
            f"<div class='metric-label'>Posts (24h)</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"<div class='metric-glass'><div class='metric-value'>{topic or '—'}</div>"
            f"<div class='metric-label'>Predicted Topic</div></div>",
            unsafe_allow_html=True,
        )
    with m5:
        sector_display = selected_sector if selected_sector != "All" else "All"
        st.markdown(
            f"<div class='metric-glass'><div class='metric-value'>{sector_display}</div>"
            f"<div class='metric-label'>Market Focus</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Two columns: Context + Execution
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### 💬 Context Window")
        st.caption("Recent posts fed to the topic predictor")
        for i, tweet in enumerate(recent_tweets[:5], 1):
            st.markdown(f"**{i}.** {tweet[:200]}{'...' if len(tweet) > 200 else ''}")

        st.markdown("---")
        if topic:
            st.markdown("**Predicted Next Topic:**")
            topic_colors = {
                "Tariffs": "#ff6b6b", "Crypto": "#ffd93d", "Media": "#4ecdc4",
                "Borders": "#667eea", "Fed": "#ff922b", "Cabinet": "#cc5de8",
                "Other": "#6b7280",
            }
            color = topic_colors.get(topic, "#6b7280")
            st.markdown(
                f"<span class='topic-badge' style='background: {color};'>{topic}</span>",
                unsafe_allow_html=True,
            )

            # Market impact preview
            if market_impacts:
                st.markdown("#### Market Impact Preview")
                for mi in market_impacts[:3]:
                    direction_color = "#4ecdc4" if "Bullish" in mi["direction"] else "#ff6b6b" if "Bearish" in mi["direction"] else "#6b7280"
                    css_class = "impact-positive" if "Bullish" in mi["direction"] else "impact-negative" if "Bearish" in mi["direction"] else "impact-neutral"
                    st.markdown(
                        f"<div class='impact-card {css_class}'>"
                        f"<span style='font-size:1.5rem'>{mi['icon']}</span> "
                        f"<strong>{mi['sector']}</strong> "
                        f"<span style='color:{direction_color}; font-weight:700;'>"
                        f"{mi['expected_move_pct']:+.2f}%</span> "
                        f"<span style='color:#6b7280;'>{mi['direction']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Probability below threshold — topic prediction skipped.")

    with right_col:
        st.markdown("### ⚡ Execution Engine")
        if signal == "HOT" and contracts:
            st.caption(
                f"Signal is HOT — {len(contracts)} actionable contracts for **{topic}**"
            )
            for c in contracts:
                sector_tag = f"<span class='sector-tag'>{c.get('sector', '')}</span>" if c.get('sector') else ""
                st.markdown(
                    f"<div class='contract-card'>"
                    f"<div class='contract-title'>{c['contract']}{sector_tag}</div>"
                    f"<span class='price-yes'>YES {c['yes_price']:.0%}</span>"
                    f"&nbsp;&nbsp;&nbsp;"
                    f"<span class='price-no'>NO {c['no_price']:.0%}</span>"
                    f"&nbsp;&nbsp;&nbsp;"
                    f"<span class='volume'>${c['volume']:,.0f} vol · {c['market']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                btn_left, btn_right, _ = st.columns([1, 1, 2])
                contract_id = c["contract"][:30].replace(" ", "_")
                with btn_left:
                    if st.button(
                        f"Buy YES @ {c['yes_price']:.0%}",
                        key=f"yes_{contract_id}",
                        type="primary",
                    ):
                        st.success(f"✅ Trade executed! Bought YES @ {c['yes_price']:.0%} on {c['market']}")
                with btn_right:
                    if st.button(
                        f"Buy NO @ {c['no_price']:.0%}",
                        key=f"no_{contract_id}",
                    ):
                        st.success(f"✅ Trade executed! Bought NO @ {c['no_price']:.0%} on {c['market']}")

        elif signal == "HOT":
            st.warning("Signal is HOT but no contracts matched the sector filter.")
        else:
            st.markdown(
                "<div style='text-align:center; padding:3rem; color:#4a5568;'>"
                "<p style='font-size:2.5rem;'>😴</p>"
                "<p style='font-size:1.5rem; font-weight:600;'>Signal COLD</p>"
                "<p>Probability below 50% threshold.<br>"
                "No trades recommended. Agent is watching.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ===================================================================
# TAB 2 — TIME TRAVEL
# ===================================================================
with tab2:
    st.markdown("### ⏰ Time Travel Replay")
    st.markdown("*Step through a historical day and see what MentionBot would have predicted vs. what actually happened.*")

    try:
        from time_travel import get_available_dates, run_full_day_replay

        # Cache available dates
        @st.cache_data(ttl=300)
        def _get_dates():
            return get_available_dates()

        available_dates = _get_dates()

        if not available_dates:
            st.warning("No historical data available for time travel replay.")
        else:
            tt_col1, tt_col2 = st.columns([1, 2])
            with tt_col1:
                # Date selector — show most recent dates
                selected_date = st.selectbox(
                    "Select a date to replay",
                    available_dates[:60],
                    index=0,
                    format_func=lambda d: f"{d} ({pd.Timestamp(d).strftime('%A')})"
                )

                if st.button("🚀 Run Replay", type="primary"):
                    st.session_state["replay_date"] = selected_date

            # Run replay
            replay_date = st.session_state.get("replay_date", None)
            if replay_date:
                with st.spinner(f"Replaying {replay_date}..."):
                    @st.cache_data(ttl=300)
                    def _run_replay(date):
                        return run_full_day_replay(date)

                    replay = _run_replay(replay_date)

                with tt_col2:
                    # Summary metrics
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

                # Prediction probability line
                fig.add_trace(go.Scatter(
                    x=hours, y=predictions,
                    mode="lines+markers",
                    name="Prediction Probability",
                    line=dict(color="#667eea", width=3),
                    marker=dict(size=8, color=["#4ecdc4" if h else "#ff6b6b" for h in hits]),
                    hovertemplate="Hour: %{x}:00<br>Prediction: %{y:.1%}<extra></extra>",
                ))

                # Threshold line
                fig.add_hline(y=0.5, line_dash="dash", line_color="#ffa726",
                              annotation_text="50% Threshold", annotation_position="top right")

                # Actual posts (bars)
                fig.add_trace(go.Bar(
                    x=hours, y=[a / max(max(actual), 1) for a in actual],
                    name="Actual Posts (scaled)",
                    marker_color=["rgba(78,205,196,0.3)" if a > 0 else "rgba(128,128,128,0.1)" for a in actual],
                    hovertemplate="Hour: %{x}:00<br>Posts: " + str(actual) + "<extra></extra>",
                ))

                fig.update_layout(
                    title=f"Predictions vs Reality — {replay['date']}",
                    xaxis_title="Hour (EST)",
                    yaxis_title="Probability",
                    template="plotly_dark",
                    paper_bgcolor="#0a0a1a",
                    plot_bgcolor="#0a0a1a",
                    height=400,
                    legend=dict(orientation="h", y=1.12),
                    xaxis=dict(dtick=1, range=[-0.5, 23.5]),
                    yaxis=dict(range=[0, 1.05]),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Hour-by-hour detail
                st.markdown("#### Hour-by-Hour Detail")
                for f in frames:
                    if f["actual_posted"] or f["prediction"] >= 0.4:
                        symbol = "✅" if f["hit"] else "❌"
                        post_info = f"📢 **{f['actual_posts']} post(s)**" if f["actual_posted"] else "_(quiet)_"
                        st.markdown(
                            f"{symbol} **{f['time_label']}** — "
                            f"Predicted: `{f['prediction']:.0%}` [{f['signal']}] | "
                            f"Actual: {post_info}"
                        )
                        if f["tweets_this_hour"]:
                            for tweet in f["tweets_this_hour"][:2]:
                                st.caption(f"    💬 _{tweet[:150]}{'...' if len(tweet) > 150 else ''}_")

    except Exception as e:
        st.error(f"Time travel module error: {e}")
        st.info("Make sure the model is trained first: `python ml_engine.py`")


# ===================================================================
# TAB 3 — MARKET IMPACT
# ===================================================================
with tab3:
    st.markdown("### 📊 Market Impact Analysis")
    st.markdown("*How Trump's tweets historically move different market sectors*")

    # Impact heatmap
    impact_matrix = get_topic_sector_matrix()
    topics = list(impact_matrix.keys())
    sectors_list = list(SECTORS.keys())

    # Build heatmap data
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
            [0, "#ff6b6b"],
            [0.35, "#ff9f9f"],
            [0.5, "#1a1a2e"],
            [0.65, "#9fdfdb"],
            [1, "#4ecdc4"],
        ],
        zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hovertext=hover_data,
        hoverinfo="text",
        colorbar=dict(title=dict(text="Impact %", side="right")),
    ))

    fig_heatmap.update_layout(
        title="Topic → Sector Impact Matrix (Avg % Move)",
        template="plotly_dark",
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        height=400,
        xaxis_title="Market Sector",
        yaxis_title="Tweet Topic",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # Detailed impact for selected topic
    impact_topic = st.selectbox("Drill into topic:", topics, index=0)
    impacts = get_impact_for_topic(impact_topic)

    imp_cols = st.columns(min(len(impacts), 3))
    for idx, impact in enumerate(impacts[:6]):
        col = imp_cols[idx % len(imp_cols)]
        with col:
            direction_color = "#4ecdc4" if impact["direction"] > 0 else "#ff6b6b" if impact["direction"] < 0 else "#6b7280"
            dir_text = "Bullish" if impact["direction"] > 0 else "Bearish" if impact["direction"] < 0 else "Neutral"
            css = "impact-positive" if impact["direction"] > 0 else "impact-negative" if impact["direction"] < 0 else "impact-neutral"
            st.markdown(
                f"<div class='impact-card {css}'>"
                f"<span style='font-size:2rem'>{impact['icon']}</span><br>"
                f"<strong style='font-size:1.1rem'>{impact['sector']}</strong><br>"
                f"<span style='color:{direction_color}; font-size:1.8rem; font-weight:800;'>"
                f"{impact['avg_move']:+.1f}%</span><br>"
                f"<span style='color:#8b949e'>Confidence: {impact['confidence']:.0%}</span><br>"
                f"<span style='color:#6b7280; font-size:0.8rem'>{', '.join(impact['tickers'])}</span><br>"
                f"<span style='color:#4a5568; font-size:0.75rem'>{impact['note']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # P&L Simulation
    st.markdown("### 💰 Simulated Trading P&L")
    st.caption("If MentionBot had traded prediction markets over the last 60 days")

    demo_trades = generate_demo_trade_history(30)
    pnl = calculate_pnl(demo_trades)

    pnl_c1, pnl_c2, pnl_c3, pnl_c4 = st.columns(4)
    pnl_color = "#4ecdc4" if pnl["total_pnl"] > 0 else "#ff6b6b"
    pnl_c1.markdown(
        f"<div class='metric-glass'><div class='metric-value' style='color:{pnl_color}'>"
        f"${pnl['total_pnl']:+.2f}</div><div class='metric-label'>Total P&L</div></div>",
        unsafe_allow_html=True,
    )
    pnl_c2.markdown(
        f"<div class='metric-glass'><div class='metric-value'>{pnl['win_rate']:.0%}</div>"
        f"<div class='metric-label'>Win Rate</div></div>",
        unsafe_allow_html=True,
    )
    pnl_c3.markdown(
        f"<div class='metric-glass'><div class='metric-value'>{pnl['wins']}</div>"
        f"<div class='metric-label'>Wins</div></div>",
        unsafe_allow_html=True,
    )
    pnl_c4.markdown(
        f"<div class='metric-glass'><div class='metric-value'>{pnl['losses']}</div>"
        f"<div class='metric-label'>Losses</div></div>",
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
            line=dict(color="#667eea", width=2),
            fillcolor="rgba(102,126,234,0.1)",
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="#4a5568")
        fig_pnl.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Trade #",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            paper_bgcolor="#0a0a1a",
            plot_bgcolor="#0a0a1a",
            height=300,
        )
        st.plotly_chart(fig_pnl, use_container_width=True)


# ===================================================================
# TAB 4 — MODEL INSIGHTS
# ===================================================================
with tab4:
    st.markdown("### 🧠 Model Insights")
    st.markdown("*Under the hood: how MentionBot makes predictions*")

    try:
        from ml_engine import get_model_metrics, get_hourly_heatmap_data

        metrics = get_model_metrics()

        if metrics:
            # Model metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            acc_color = "#4ecdc4" if metrics.get("accuracy", 0) > 0.65 else "#ffa726"
            mc1.markdown(
                f"<div class='metric-glass'><div class='metric-value' style='color:{acc_color}'>"
                f"{metrics.get('accuracy', 0):.1%}</div><div class='metric-label'>Accuracy</div></div>",
                unsafe_allow_html=True,
            )
            mc2.markdown(
                f"<div class='metric-glass'><div class='metric-value'>"
                f"{metrics.get('precision_post', 0):.1%}</div><div class='metric-label'>Precision (Post)</div></div>",
                unsafe_allow_html=True,
            )
            mc3.markdown(
                f"<div class='metric-glass'><div class='metric-value'>"
                f"{metrics.get('recall_post', 0):.1%}</div><div class='metric-label'>Recall (Post)</div></div>",
                unsafe_allow_html=True,
            )
            mc4.markdown(
                f"<div class='metric-glass'><div class='metric-value'>"
                f"{metrics.get('f1_post', 0):.1%}</div><div class='metric-label'>F1 Score</div></div>",
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
                        colorscale=[[0, "#4a5568"], [0.5, "#667eea"], [1, "#f093fb"]],
                    ),
                    text=[f"{v:.1%}" for v in feat_values],
                    textposition="auto",
                ))
                fig_fi.update_layout(
                    title="Feature Importance (GradientBoosting)",
                    template="plotly_dark",
                    paper_bgcolor="#0a0a1a",
                    plot_bgcolor="#0a0a1a",
                    height=400,
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance",
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # Confusion matrix
            cm = metrics.get("confusion_matrix")
            if cm:
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted: No Post", "Predicted: Post"],
                    y=["Actual: No Post", "Actual: Post"],
                    colorscale=[[0, "#0a0a1a"], [1, "#667eea"]],
                    text=[[str(v) for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont={"size": 18, "color": "white"},
                    showscale=False,
                ))
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    template="plotly_dark",
                    paper_bgcolor="#0a0a1a",
                    plot_bgcolor="#0a0a1a",
                    height=350,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            # Classification report
            report = metrics.get("classification_report", "")
            if report:
                with st.expander("📋 Full Classification Report"):
                    st.code(report)

            # Training data stats
            st.markdown("---")
            st.markdown("#### Training Data Stats")
            ts1, ts2, ts3 = st.columns(3)
            ts1.metric("Training Samples", f"{metrics.get('n_train', 0):,}")
            ts2.metric("Test Samples", f"{metrics.get('n_test', 0):,}")
            ts3.metric("Positive Rate", f"{metrics.get('positive_rate', 0):.1%}")

        else:
            st.warning("No model metrics found. Run `python ml_engine.py` to train the model first.")

        st.markdown("---")

        # Posting frequency heatmap
        st.markdown("#### 📅 Posting Frequency Heatmap")
        st.caption("Average posts per hour-of-day × day-of-week (from 17,922 real posts)")

        try:
            heatmap_data = get_hourly_heatmap_data(csv_path="data/truth_archive.csv")
            if not heatmap_data.empty:
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                hour_labels = [f"{h:02d}:00" for h in range(24)]

                fig_freq = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=day_names,
                    y=hour_labels,
                    colorscale=[[0, "#0a0a1a"], [0.3, "#16213e"], [0.6, "#667eea"], [1, "#f093fb"]],
                    text=[[f"{v:.1f}" for v in row] for row in heatmap_data.values],
                    texttemplate="%{text}",
                    textfont={"size": 9},
                    colorbar=dict(title="Avg Posts"),
                ))
                fig_freq.update_layout(
                    title="When Does Trump Post? (Avg posts per hour, by day of week)",
                    template="plotly_dark",
                    paper_bgcolor="#0a0a1a",
                    plot_bgcolor="#0a0a1a",
                    height=600,
                    xaxis_title="Day of Week",
                    yaxis_title="Hour (EST)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_freq, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")

    except Exception as e:
        st.error(f"Error loading model insights: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='footer'>"
    "MentionBot v2.0 — YHack 2025 | "
    "GradientBoosting (12 features) trained on 17,922 real Truth Social posts | "
    "Market impact analysis across 6 sectors"
    "</div>",
    unsafe_allow_html=True,
)
