"""
app.py — MentionBot Demo Dashboard
====================================
Launch:  py -m streamlit run app.py
"""

import streamlit as st
from market_agent import run_agent_cycle

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MentionBot — Trump Post Predictor",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Gauge colours */
    .prob-hot  { color: #ff4b4b; font-size: 4.5rem; font-weight: 800; text-align: center; }
    .prob-warm { color: #ffa726; font-size: 4.5rem; font-weight: 800; text-align: center; }
    .prob-cold { color: #66bb6a; font-size: 4.5rem; font-weight: 800; text-align: center; }
    .signal-badge {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 1rem;
        font-weight: 700; font-size: 1.1rem; margin-top: 0.5rem;
    }
    .badge-hot  { background: #ff4b4b; color: white; }
    .badge-cold { background: #66bb6a; color: white; }
    .contract-card {
        background: #1e1e2f; border: 1px solid #333;
        border-radius: 0.75rem; padding: 1rem; margin-bottom: 0.75rem;
    }
    .contract-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.5rem; }
    .price-yes { color: #66bb6a; font-weight: 700; font-size: 1.3rem; }
    .price-no  { color: #ff4b4b; font-weight: 700; font-size: 1.3rem; }
    .volume    { color: #888; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Agent Controls & Simulation ("Time Travel")
# ---------------------------------------------------------------------------
st.sidebar.title("Agent Controls & Simulation")
st.sidebar.caption("Move the sliders to time-travel through Trump's posting patterns.")

sim_hour = st.sidebar.slider(
    "Current Hour (EST)", min_value=0, max_value=23, value=10,
    help="Simulated hour of day in Eastern Time"
)
hours_since = st.sidebar.slider(
    "Hours Since Last Post", min_value=0.0, max_value=24.0, value=1.5, step=0.25,
    help="How long ago Trump last posted"
)
post_count_24h = st.sidebar.slider(
    "Rolling Post Count (24h)", min_value=0, max_value=50, value=8,
    help="Total posts in the last 24 hours"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulated Recent Tweets")
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
    height=200,
)
recent_tweets = [t.strip() for t in tweets_text.strip().split("\n") if t.strip()]

use_llm = st.sidebar.checkbox(
    "Use OpenAI for topic prediction",
    value=False,
    help="Requires OPENAI_API_KEY env var. If off, uses keyword fallback."
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
)

prob = result["probability"]
signal = result["signal"]
topic = result["predicted_topic"]
contracts = result["contracts"]

# ---------------------------------------------------------------------------
# PANEL 1 — The Live Radar (Top)
# ---------------------------------------------------------------------------
st.markdown("## MentionBot — Live Posting Radar")
st.markdown("*AI agent that predicts Trump's Truth Social activity and surfaces prediction market trades.*")
st.markdown("---")

radar_col1, radar_col2, radar_col3 = st.columns([2, 3, 2])

with radar_col2:
    # Probability gauge
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
        f"<p style='text-align:center; font-size:1.1rem; margin-top:-0.5rem;'>"
        f"Chance of post in the next hour</p>",
        unsafe_allow_html=True,
    )

    badge_class = "badge-hot" if signal == "HOT" else "badge-cold"
    st.markdown(
        f"<p style='text-align:center'>"
        f"<span class='signal-badge {badge_class}'>Signal: {signal}</span></p>",
        unsafe_allow_html=True,
    )

# Key metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Simulated Hour (EST)", f"{sim_hour}:00")
m2.metric("Hours Since Last Post", f"{hours_since:.1f}h")
m3.metric("Posts in Last 24h", post_count_24h)
m4.metric("Predicted Topic", topic if topic else "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# PANEL 2 & 3 — Context Window + Execution Engine
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1])

# --- Left: Context Window ---
with left_col:
    st.markdown("### Context Window")
    st.caption("Recent posts fed to the LLM topic predictor")

    for i, tweet in enumerate(recent_tweets[:5], 1):
        st.markdown(
            f"**{i}.** {tweet[:200]}{'...' if len(tweet) > 200 else ''}"
        )

    st.markdown("---")

    if topic:
        st.markdown(f"**LLM Predicted Next Topic:**")
        topic_colors = {
            "Tariffs": "#ff6b6b", "Crypto": "#ffd93d", "Media": "#6bcb77",
            "Borders": "#4d96ff", "Fed": "#ff922b", "Cabinet": "#cc5de8",
            "Other": "#868e96",
        }
        color = topic_colors.get(topic, "#868e96")
        st.markdown(
            f"<span style='background:{color}; color:white; padding:0.4rem 1.2rem; "
            f"border-radius:2rem; font-weight:700; font-size:1.2rem;'>"
            f"{topic}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Probability below threshold — topic prediction skipped.")

# --- Right: Execution Engine ---
with right_col:
    st.markdown("### Execution Engine")

    if signal == "HOT" and contracts:
        st.caption(
            f"Signal is HOT — {len(contracts)} actionable contracts found for **{topic}**"
        )

        for c in contracts:
            st.markdown(
                f"<div class='contract-card'>"
                f"<div class='contract-title'>{c['contract']}</div>"
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
                    st.success(
                        f"Trade executed on {c['market']}! "
                        f"Bought YES @ {c['yes_price']:.0%}"
                    )
            with btn_right:
                if st.button(
                    f"Buy NO @ {c['no_price']:.0%}",
                    key=f"no_{contract_id}",
                ):
                    st.success(
                        f"Trade executed on {c['market']}! "
                        f"Bought NO @ {c['no_price']:.0%}"
                    )

    elif signal == "HOT":
        st.warning("Signal is HOT but no contracts matched the topic.")
    else:
        st.markdown(
            "<div style='text-align:center; padding:3rem; color:#888;'>"
            "<p style='font-size:2rem;'>Signal COLD</p>"
            "<p>Probability is below the 50% threshold.<br>"
            "No trades recommended. The agent is watching.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "MentionBot v1.0 — YHack 2025 | "
    "ML engine trained on 17,922 real Truth Social posts | "
    "RandomForest + GPT-4o-mini topic classifier</p>",
    unsafe_allow_html=True,
)
