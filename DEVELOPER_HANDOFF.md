# 🛠️ MentionBot — Developer Handoff & Architecture Guide

Welcome to the MentionBot codebase! This document is designed for developers taking over or extending the project. It covers the technical architecture, data flow, machine learning pipeline, and exactly how the different modules interact.

---

## 🏗️ 1. High-Level Architecture

MentionBot is a Python-first application built around a core Machine Learning engine that predicts Trump's Truth Social posting behavior. The system is split into distinct functional modules:

```text
trump_tweet_prediction/
├── ml_engine.py            # Core ML: Data ingestion, feature engineering, training, inference
├── market_agent.py         # Agent Logic: Predicts topics and queries market contracts
├── market_impact.py        # Financial Logic: Mappings connecting topics to market sectors
├── time_travel.py          # Simulation Engine: Replays historical data hour-by-hour
├── app.py                  # Frontend UI: Streamlit + Plotly dashboard
├── generate_sample_data.py # Utility: Synthesizes fake data (fallback)
├── data/
│   └── truth_archive.csv   # The grounded truth data (17,922 pristine posts)
└── models/
    ├── mentionbot_rf.joblib    # The serialized GradientBoosting model
    └── model_metrics.json      # Evaluation metrics and feature importance
```

---

## 🧠 2. The Machine Learning Engine (`ml_engine.py`)

This is the brain of the operation. It handles everything from raw CSV ingestion to serving live probability inferences.

### Data Ingestion & Cleanup
*   **Source:** `data/truth_archive.csv`
*   **Logic:** We filter out explicit retweets (`RT @...`) and drop exact duplicate texts to prevent data leakage.
*   **Windowing:** Data is clustered into **1-hour tumbling windows**. The target variable (`posted_in_next_hour`) is a strict boolean: 1 if `post_count > 0` in the *following* chronological hour, else 0.

### Feature Engineering
The model transforms timestamps into exactly **12 features**:
1.  **Temporal Cyclicals:** `hour_sin`, `hour_cos`, `day_sin`, `day_cos`. (Critical: AI models are linear. Sin/Cos transforms teach the model that 23:00 and 00:00 are adjacent).
2.  **Momentum Indicators:**
    *   `rolling_post_count_24h`: How active has the account been over the last day?
    *   `posting_velocity_4h`: Average posts per hour over the last 4 hours (detects incoming bursts).
    *   `hours_since_last_post`: The primary decay function.
    *   `gap_acceleration`: The first derivative of the decay (are the gaps between posts widening or shrinking?).
3.  **Averages & Flags:** `avg_posts_this_hour_hist` (historical heatmap overlay), `is_peak_hour` (6am-10am or 6pm-10pm EST), `is_weekend`.

### The Algorithm
*   We use a `GradientBoostingClassifier` instead of a Random Forest for superior handling of non-linear boosting.
*   **Class Imbalance:** Because actual posts only occur in ~22.5% of hours, we apply a custom `sample_weight` array (weighting positive hits 3.0x vs 1.0x for negatives) during the `fit()` stage. Without this, the model heavily biases toward "No Post".
*   *Note:* The final model is serialized to `models/mentionbot_rf.joblib`. The UI loads this natively.

---

## 🏦 3. The Market Agent & Impact Analysis

Once `ml_engine.py` generates a probability, the Agent decides what to do with it.

### `market_agent.py`
*   **The Threshold:** The agent enforces a strict `PROBABILITY_THRESHOLD = 0.35`. Because Gradient Boosting models output well-calibrated (but lower) raw probabilities, 35% represents a massive spike in momentum against the baseline.
*   **Topic Prediction:** If the signal is HOT (>35%), the agent feeds recent context tweets to an LLM (requires `OPENAI_API_KEY`) to extract a Topic (e.g., "Tariffs", "Crypto"). If OpenAI fails, it falls back to a deterministic keyword-matching heuristic (`_keyword_topic_fallback()`).
*   **Dynamic Pricing:** It queries `MOCK_CONTRACTS` and automatically shifts the YES/NO asset prices based directly on the model's confidence multiplier.

### `market_impact.py`
*   This module holds the financial rule engine.
*   **`IMPACT_MATRIX`**: A dictionary establishing the mathematical link between Topics and Sectors. E.g., If the topic is "Fed", it defines that US Equities have an `avg_move` of +1.1%, while Forex (DXY) drops by -0.8%.
*   **`simulate_market_reaction()`**: Generates gaussian noise around the expected moves to simulate live price action for the UI.
*   *Note on signs:* The `avg_move` float internally holds the sign (e.g., `-1.2`). Do not multiply it by a direction flag in the UI or you will double-negate the value.

---

## ⏪ 4. Time Travel (`time_travel.py`)

The simulation engine used to prove the model's efficacy over historical data.
*   **Flow:** Extracts all posts for a specific date (e.g., "2025-10-25"). It reconstructs the exact feature-state of the world (24h counts, 4h velocities) sequentially for hour 0 through 23.
*   It fires `predict_live_probability()` just as the live agent would, generating an array of predictions compared against the true ground reality (`actual_posted`).
*   The Streamlit app calls `run_full_day_replay()` and caches the result for instant rendering.

---

## 🖥️ 5. The Application UI (`app.py`)

Built on Streamlit (`st`) with Plotly Graph Objects (`go`).
*   **State Management:** Heavily utilizes `st.sidebar` for live agent parameters and `st.session_state` to retain Time Travel dates across re-renders.
*   **UI Hacks:** Streamlit is inherently static-looking. We inject massive blocks of custom CSS (`st.markdown("<style>...", unsafe_allow_html=True)`) to force dark mode, custom gradient typography, glassmorphism cards, and badge animations. Do not strip the CSS block out unless you are refactoring to a different framework.
*   **Plotly V6 Breakage:** We updated to Plotly 6.x. The `titleside` argument in `colorbar` is deprecated. Always use `colorbar=dict(title=dict(text="...", side="right"))`.

---

## 🚀 6. Next Steps for the Next Developer

If you are expanding MentionBot from a prototype to a production trading bot, focus on these three things:

1.  **Polymarket API Integration:** Deeply embed the `py_clob_client` into `market_agent.py`. Replace `MOCK_CONTRACTS` with live orderbook queries and trigger market-orders when the probability crosses the Kelly-criterion threshold.
2.  **Live Truth Social Scraper:** Right now, the sliders simulate current conditions. You need to build a cronjob/webhook that pings the Truth Social API/RSS feed continuously to construct the 24h/4h velocity counters autonomously.
3.  **Model Retraining Architecture:** The model degrades over time. Build an airflow DAG that appends new scrape data to `truth_archive.csv` and calls `train_model()` automatically every 48 hours to recalibrate the baseline averages.
