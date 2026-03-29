# 🛠️ MentionBot — Developer Handoff & Architecture Guide

Welcome to the MentionBot codebase! This document is designed for developers taking over or extending the project. It covers the technical architecture, data flow, machine learning pipeline, and exactly how the different modules interact.

---

## 🏗️ 1. High-Level Architecture

MentionBot is a Python-first application built around a core Machine Learning engine that predicts Trump's Truth Social posting behavior. The system is split into distinct functional modules:

```text
trump_tweet_prediction/
├── ml_engine.py            # Core ML: Data ingestion, feature engineering, training, inference
├── market_agent.py         # Agent Logic: Predicts topics, evaluates arbitrage, queries contracts
├── market_impact.py        # Financial Logic: Mappings connecting topics to market sectors
├── time_travel.py          # Simulation Engine: Replays historical data hour-by-hour
├── app.py                  # Frontend UI: Streamlit + Plotly dashboard
├── generate_sample_data.py # Utility: Synthesizes correlated mock data (news, VIX, DXY)
├── data/
│   └── truth_archive.csv   # The grounded truth data (17,922 pristine posts)
└── models/
    ├── mentionbot_rf.joblib    # The serialized GradientBoosting model (16 features)
    └── model_metrics.json      # Evaluation metrics and feature importance
```

---

## 🧠 2. The Machine Learning Engine (`ml_engine.py`)

This is the brain of the operation. It handles everything from raw CSV ingestion to serving live probability inferences.

### Data Ingestion & Cleanup
*   **Source:** `data/truth_archive.csv`
*   **Logic:** We filter out explicit retweets (`RT @...`) and drop exact duplicate texts to prevent data leakage.
*   **Extra columns:** The ingester now preserves `news_volume_4h`, `news_sentiment`, `simulated_vix_4h_vol`, and `simulated_dxy_4h_vol` if they exist in the source CSV.
*   **Windowing:** Data is clustered into **1-hour tumbling windows**. The target variable (`posted_in_next_hour`) is a strict boolean: 1 if `post_count > 0` in the *following* chronological hour, else 0.

### Feature Engineering
The model transforms timestamps and market context into exactly **16 features**:

1.  **Temporal Cyclicals (4):** `hour_sin`, `hour_cos`, `day_sin`, `day_cos`. (Critical: AI models are linear. Sin/Cos transforms teach the model that 23:00 and 00:00 are adjacent).
2.  **Momentum Indicators (4):**
    *   `rolling_post_count_24h`: How active has the account been over the last day?
    *   `posting_velocity_4h`: Average posts per hour over the last 4 hours (detects incoming bursts).
    *   `hours_since_last_post`: The primary decay function.
    *   `gap_acceleration`: The first derivative of the decay (are the gaps between posts widening or shrinking?).
3.  **Averages & Flags (3):** `avg_posts_this_hour_hist` (historical heatmap overlay), `is_peak_hour` (7-11am or 6-11pm EST), `is_weekend`.
4.  **`hour_of_day` (1):** Raw integer hour, top feature by importance.
5.  **News Context (2) — NEW:**
    *   `news_volume_4h`: Breaking news article count in the last 4 hours (integer 0-50). More news = more likely to post.
    *   `news_sentiment`: Aggregate sentiment of recent news articles (float -1.0 to +1.0). Negative sentiment correlates with reactive posting.
6.  **Regime Classification (1) — NEW:**
    *   `regime_flag`: Binary flag. `1` if `rolling_post_count_24h > 5` (High-Activity Regime), else `0` (Dormant Regime). A simple rule-based proxy for Hidden Markov Model state detection.
7.  **Market Feedback (1) — NEW:**
    *   `simulated_vix_4h_vol`: CBOE Volatility Index trailing 4-hour value (float 10-50). High VIX = market crisis = reactive posting.
    *   **Exponential VIX Boost:** After the model produces a raw probability, if VIX > 25, an exponential boost is applied: `prob += (exp((vix - 25) / 15) - 1) * 0.15`. This models the feedback loop where market turmoil drives urgent social media behavior.

### The Algorithm
*   We use a `GradientBoostingClassifier` instead of a Random Forest for superior handling of non-linear boosting.
*   **Class Imbalance:** Because actual posts only occur in ~22.5% of hours, we apply a custom `sample_weight` array (weighting positive hits 3.0x vs 1.0x for negatives) during the `fit()` stage. Without this, the model heavily biases toward "No Post".
*   *Note:* The final model is serialized to `models/mentionbot_rf.joblib`. The UI loads this natively.
*   **Current performance (v3, 16 features):** Accuracy 63.3%, Precision 33.2%, Recall 62.4%, F1 43.3%. See "Changelog" section below for comparison against v2.

---

## 🏦 3. The Market Agent & Impact Analysis

Once `ml_engine.py` generates a probability, the Agent decides what to do with it.

### `market_agent.py`

#### Signal Logic (Three-Tier)
The agent now uses a **three-tier signal system** instead of the previous binary HOT/COLD:

| Signal | Condition | Meaning |
|--------|-----------|---------|
| **HOT** | `probability >= 0.35` AND `alpha_delta >= +10%` | Model sees an edge the crowd doesn't. Execute trades. |
| **WARM** | `probability >= 0.35` AND `alpha_delta < +10%` | Model is bullish but the crowd already priced it in. Monitor only. |
| **COLD** | `probability < 0.35` | Below base threshold. Standby. |

#### Polymarket Arbitrage Evaluation — NEW
The `evaluate_arbitrage()` function compares the ML model's internal probability against the average Polymarket contract `yes_price` (the crowd's consensus):

*   **Alpha Delta** = `ML_probability - Polymarket_baseline`
*   The agent only triggers a **HOT** signal if Alpha Delta >= **+10%** (i.e., our model sees something the crowd hasn't priced in yet).
*   Each contract is individually tagged with its own `contract_alpha` so the dashboard shows per-contract edge.

#### Other Agent Logic (Unchanged)
*   **Topic Prediction:** If signal is WARM or HOT, the agent feeds recent context tweets to an LLM (requires `OPENAI_API_KEY`) to extract a Topic (e.g., "Tariffs", "Crypto"). Falls back to keyword matching if unavailable.
*   **Dynamic Pricing:** Queries `MOCK_CONTRACTS` and shifts YES/NO prices based on the model's confidence.
*   **New parameters:** `run_agent_cycle()` now accepts `news_volume_4h`, `news_sentiment`, and `simulated_vix_4h_vol` which it passes through to `predict_live_probability()`.

### `market_impact.py`
*   This module is **unchanged**. It holds the financial rule engine.
*   **`IMPACT_MATRIX`**: A dictionary establishing the mathematical link between Topics and Sectors.
*   **`simulate_market_reaction()`**: Generates gaussian noise around the expected moves.
*   *Note on signs:* The `avg_move` float internally holds the sign (e.g., `-1.2`). Do not multiply it by a direction flag in the UI.

---

## ⏪ 4. Time Travel (`time_travel.py`)

The simulation engine used to prove the model's efficacy over historical data. **Unchanged in v3.**
*   **Flow:** Extracts all posts for a specific date. Reconstructs the exact feature-state hour-by-hour and fires `predict_live_probability()` just as the live agent would.
*   The Streamlit app calls `run_full_day_replay()` and caches the result for instant rendering.

---

## 🖥️ 5. The Application UI (`app.py`)

Built on Streamlit (`st`) with Plotly Graph Objects (`go`).

### Layout Changes (v3)
The **Live Radar** tab now uses a three-column layout:
*   **Left column:** Regime indicator badge, VIX gauge (color-coded: green < 22, yellow 22-30, red > 30), News Sentiment indicator, News Volume counter.
*   **Center column:** Main probability gauge and three-tier signal badge (HOT/WARM/COLD).
*   **Right column:** Alpha Delta percentage display, Polymarket baseline, and edge status indicator.

### New Sidebar Controls
Three new sliders under "Market Context":
*   `NEWS VOLUME (4H)` — integer 0-50
*   `NEWS SENTIMENT` — float -1.0 to +1.0
*   `VIX (4H TRAILING)` — float 10.0 to 50.0

### Execution Engine Changes
*   **HOT signal:** Shows a green "ALPHA DETECTED" banner with the exact edge percentage. Each contract card now has an alpha badge (e.g., `ALPHA +15%`).
*   **WARM signal:** Shows a yellow "NO ARBITRAGE EDGE" box explaining the ML probability vs Polymarket baseline and the gap needed.
*   **COLD signal:** Unchanged standby state.

### Unchanged
*   **State Management:** Same `st.sidebar` + `st.session_state` pattern.
*   **CSS injection block:** Same dark-mode terminal theme with additions for `.signal-warm`, `.regime-tag`, `.order-alpha`, and `.alpha-container` classes.
*   **Plotly V6:** Same `colorbar=dict(title=dict(text="...", side="right"))` pattern.
*   **Tabs 2-4** (Time Travel, Market Impact, Model Diagnostics): Unchanged except the feature importance chart now shows 16 features and the footer reads v3.0.

---

## 📊 6. Data Generator (`generate_sample_data.py`)

The synthetic data generator now produces **8 columns** per row instead of 4:

| Column | Type | Range | Correlation Logic |
|--------|------|-------|-------------------|
| `timestamp` | str | ISO 8601 | Same as before |
| `text` | str | — | Same template system |
| `predicted_topic` | str | — | Empty (filled by pipeline) |
| `probability_score` | float | — | Empty (filled by pipeline) |
| `news_volume_4h` | int | 0-50 | Higher during business hours, spikes on crisis topics (Tariffs/Fed/Borders), +10% chance of breaking-news spike |
| `news_sentiment` | float | -1.0 to 1.0 | Inversely correlated with news volume (crisis = negative). Crypto slightly positive. |
| `simulated_vix_4h_vol` | float | 10-50 | Baseline ~18. Driven up by news volume and negative sentiment via exponential scaling. Weekend dampening. |
| `simulated_dxy_4h_vol` | float | -1.5 to 1.5 | USD index change. Tariffs strengthen (+), dovish Fed weakens (-). High VIX adds slight positive bias (flight to safety). |

All correlations are generated per-post via `_generate_market_context()` using a seeded `random.Random` instance for reproducibility.

---

## 📝 7. Changelog: v2 → v3

### What changed

| Area | Before (v2) | After (v3) |
|------|------------|------------|
| **Features** | 12 autoregressive/temporal | 16 (added news, regime, VIX) |
| **Signal tiers** | Binary: HOT / COLD | Three-tier: HOT / WARM / COLD |
| **Trade gating** | Fires on probability > 35% | Fires only if Alpha Delta >= +10% over Polymarket |
| **Data generator** | 4 columns (timestamp, text, topic, score) | 8 columns (+ news_volume, sentiment, VIX, DXY) |
| **Sidebar controls** | 4 sliders + sector + tweets | 7 sliders (+ VIX, news vol, news sentiment) |
| **Dashboard layout** | 2-column (gauge + signal) | 3-column (market context / gauge / alpha delta) |

### Accuracy comparison (trained on 17,922 real posts)

| Metric | v2 (12 features) | v3 (16 features) | Delta |
|--------|------------------|-------------------|-------|
| Accuracy | 62.17% | **63.25%** | **+1.08%** |
| Precision (post) | 33.49% | 33.20% | -0.29% |
| Recall (post) | 68.88% | 62.37% | -6.51% |
| F1 (post) | 45.07% | 43.33% | -1.74% |
| No-post Recall | 60.25% | **63.51%** | **+3.26%** |

**Interpretation:** The model is slightly more conservative — fewer false positives (good for trading) at the cost of missing some actual posting hours. Overall accuracy improved. The new features (especially `news_sentiment` at 13.7% importance and `simulated_vix_4h_vol` at 12.4%) absorbed explanatory power from the old autoregressive features.

### Files modified
*   `generate_sample_data.py` — Added `_generate_market_context()`, expanded CSV schema
*   `ml_engine.py` — Added 4 features to `FEATURE_COLS`, updated `prep_data()`, `predict_live_probability()`, and `ingest_real_data()`
*   `market_agent.py` — Added `evaluate_arbitrage()`, three-tier signal logic, new params on `run_agent_cycle()`
*   `app.py` — Three-column Live Radar, alpha delta display, regime/VIX/news indicators, WARM signal state, new sidebar sliders

### Files NOT modified
*   `market_impact.py` — No changes
*   `time_travel.py` — No changes
*   `requirements.txt` — No new dependencies

---

## 🚀 8. Next Steps for the Next Developer

If you are expanding MentionBot from a prototype to a production trading bot, focus on these:

1.  **Polymarket API Integration:** Replace `MOCK_CONTRACTS` with live orderbook queries via `py_clob_client`. The `evaluate_arbitrage()` function is already structured to accept real contract prices — just swap the data source.
2.  **Live News API:** Replace the simulated `news_volume_4h` and `news_sentiment` with a real feed (e.g., NewsAPI, GDELT, or a Bloomberg terminal). The `predict_live_probability()` function already accepts these as parameters.
3.  **Live VIX Feed:** Pull real CBOE VIX data via Yahoo Finance (`yfinance`) or Alpha Vantage. The exponential boost logic in `predict_live_probability()` is calibrated for real VIX ranges (10-50).
4.  **Live Truth Social Scraper:** Build a cronjob/webhook that pings the Truth Social API/RSS feed to construct the 24h/4h velocity counters autonomously.
5.  **Full HMM Regime Detection:** The current `regime_flag` is a simple threshold. For production, implement a proper Hidden Markov Model (e.g., `hmmlearn`) trained on posting cadence to detect Campaign, Crisis, Weekend, and Dormant states automatically.
6.  **Model Retraining Architecture:** The model degrades over time. Build an Airflow DAG that appends new scrape data to `truth_archive.csv` and calls `train_model()` automatically every 48 hours.
