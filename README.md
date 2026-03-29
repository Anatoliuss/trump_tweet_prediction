# 🎯 MentionBot — Trump Post Prediction & Market Impact Agent

An AI-powered agent that predicts when Trump will post on Truth Social and analyzes the potential impact on financial markets, surfacing actionable prediction market trades.

**Built for YHack 2025**

## Features

- **ML Prediction Engine** — GradientBoosting classifier trained on 17,922 real Truth Social posts with 12 engineered features
- **Market Impact Analysis** — Maps tweet topics to 6 market sectors (Equities, Crypto, Bonds, Forex, Commodities, China/EM)
- **Time Travel Replay** — Step through historical days and see what MentionBot would have predicted vs. reality
- **Prediction Market Integration** — Surfaces actionable Polymarket/Kalshi contracts based on predicted topic
- **Sector Selector** — Choose which market sector to focus on for targeted trade recommendations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the ML model (uses real data from data/truth_archive.csv)
python ml_engine.py

# Launch the dashboard
python -m streamlit run app.py
```

## Optional: LLM Topic Prediction
```bash
export OPENAI_API_KEY=your_key_here
```
The app works without an API key using keyword-based topic fallback.

## Architecture

```
├── app.py                  # Streamlit dashboard (4 tabs)
├── ml_engine.py            # ML training + inference + heatmap data
├── market_agent.py         # Agent decision cycle + contract search
├── market_impact.py        # Topic → sector impact analysis + P&L
├── time_travel.py          # Historical day replay engine
├── generate_sample_data.py # Synthetic data generator (fallback)
├── data/
│   ├── truth_archive.csv   # 29K real Truth Social posts (Feb 2022 – Oct 2025)
│   └── sample_posts.csv    # 100 synthetic posts
├── models/
│   ├── mentionbot_rf.joblib    # Trained model
│   └── model_metrics.json      # Accuracy metrics
└── requirements.txt
```

## Dashboard Tabs

1. **Live Radar** — Real-time prediction probability gauge with signal badges, contract cards, and market impact preview
2. **Time Travel** — Select any historical date, see predictions vs. reality hour-by-hour
3. **Market Impact** — Interactive heatmap showing topic→sector correlations, drill-down analysis, simulated P&L
4. **Model Insights** — Feature importance, confusion matrix, posting frequency heatmap

## Model

- **Algorithm**: GradientBoostingClassifier (300 trees, depth 5)
- **Features**: Cyclical hour/day encoding, posting velocity, gap acceleration, peak hour flag, weekend flag, historical hourly averages
- **Training Data**: 17,922 cleaned posts → 32,383 hourly time windows

## Team

Built at YHack 2025
