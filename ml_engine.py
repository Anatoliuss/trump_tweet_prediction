"""
ml_engine.py — MentionBot Predictive Engine v3
================================================
Enhanced with 16 features including news context, regime detection,
and market feedback loops (VIX/DXY).

Public API:
  - ingest_real_data(csv_path)                    -> pd.DataFrame
  - prep_data(csv_path, pre_ingested_df)          -> pd.DataFrame
  - train_model(df)                               -> GradientBoostingClassifier
  - predict_live_probability(...)                  -> float  (0.0 – 1.0)
  - predict_next_topic(last_5_tweets_list)         -> str
  - get_model_metrics()                            -> dict
  - get_hourly_heatmap_data(df)                    -> pd.DataFrame
"""

from __future__ import annotations

import os
import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EST = pytz.timezone("America/New_York")
MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "mentionbot_rf.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

FEATURE_COLS = [
    # Original 12 autoregressive features
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "is_peak_hour",
    "hours_since_last_post",
    "rolling_post_count_24h",
    "posting_velocity_4h",
    "gap_acceleration",
    "avg_posts_this_hour_hist",
    "hour_of_day",
    # NEW: News context features
    "news_volume_4h",
    "news_sentiment",
    # NEW: Regime classification
    "regime_flag",
    # NEW: Market feedback loops
    "simulated_vix_4h_vol",
]

TOPIC_LABELS = ["Tariffs", "Crypto", "Media", "Borders", "Fed", "Cabinet", "Other"]


# ===========================================================================
# STEP 1 — DATA PREP & FEATURE ENGINEERING
# ===========================================================================

def ingest_real_data(csv_path: str) -> pd.DataFrame:
    """
    Load any Trump Truth Social CSV, auto-detect columns, clean, and
    normalise to the canonical schema: (timestamp, text).
    """
    df = pd.read_csv(csv_path)
    original_count = len(df)

    # --- Auto-detect timestamp column ---
    ts_candidates = ["created_at", "timestamp", "date", "datetime", "created", "time"]
    ts_col = None
    for c in ts_candidates:
        matches = [col for col in df.columns if col.lower() == c.lower()]
        if matches:
            ts_col = matches[0]
            break
    if ts_col is None:
        raise ValueError(f"Cannot detect timestamp column. Columns: {list(df.columns)}")

    # --- Auto-detect text column ---
    text_candidates = ["content", "text", "tweet", "body", "message", "post"]
    text_col = None
    for c in text_candidates:
        matches = [col for col in df.columns if col.lower() == c.lower()]
        if matches:
            text_col = matches[0]
            break
    if text_col is None:
        raise ValueError(f"Cannot detect text column. Columns: {list(df.columns)}")

    print(f"[ingest] Detected columns: timestamp='{ts_col}', text='{text_col}'")

    df = df.rename(columns={ts_col: "timestamp", text_col: "text"})

    # --- Remove RT-prefixed posts ---
    df["text"] = df["text"].astype(str)
    before = len(df)
    df = df[~df["text"].str.startswith("RT @", na=False)].copy()
    if before - len(df) > 0:
        print(f"[ingest] Removed {before - len(df)} RT-prefixed reposts")

    # --- Remove empty/null text ---
    df = df.dropna(subset=["timestamp", "text"])
    df = df[df["text"].str.strip().str.len() > 0]

    # --- Remove duplicates by text ---
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    if before - len(df) > 0:
        print(f"[ingest] Removed {before - len(df)} duplicate posts")

    # --- Parse timestamps ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Keep extra columns if they exist (news_volume_4h, etc.)
    keep_cols = ["timestamp", "text"]
    extra_cols = ["news_volume_4h", "news_sentiment", "simulated_vix_4h_vol", "simulated_dxy_4h_vol"]
    for col in extra_cols:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].sort_values("timestamp").reset_index(drop=True)

    print(
        f"[ingest] {original_count} raw rows -> {len(df)} clean original posts "
        f"({df['timestamp'].min().strftime('%Y-%m-%d')} to "
        f"{df['timestamp'].max().strftime('%Y-%m-%d')})"
    )
    return df


def prep_data(csv_path: str, pre_ingested_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Load historical posts and return an hourly feature matrix with 16 features.
    Includes news context, regime detection, and market feedback features.
    """
    # ------------------------------------------------------------------
    # 1. Load & parse timestamps
    # ------------------------------------------------------------------
    if pre_ingested_df is not None:
        df_raw = pre_ingested_df.copy()
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    else:
        df_raw = pd.read_csv(csv_path)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)

    df_raw["timestamp_est"] = df_raw["timestamp"].dt.tz_convert(EST)
    df_raw = df_raw.sort_values("timestamp_est").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Build hourly spine
    # ------------------------------------------------------------------
    floor_start = df_raw["timestamp_est"].iloc[0].floor("h")
    floor_end = df_raw["timestamp_est"].iloc[-1].floor("h")

    hourly_index = pd.date_range(floor_start, floor_end, freq="h", tz=EST)
    hourly = pd.DataFrame(index=hourly_index)

    df_raw["hour_bucket"] = df_raw["timestamp"].dt.floor("h").dt.tz_convert(EST)
    post_counts = df_raw.groupby("hour_bucket").size().rename("post_count")
    hourly = hourly.join(post_counts).fillna(0)
    hourly["post_count"] = hourly["post_count"].astype(int)

    # ------------------------------------------------------------------
    # 3. Target: posted in next hour?
    # ------------------------------------------------------------------
    hourly["posted_in_next_hour"] = (
        hourly["post_count"].shift(-1).fillna(0).gt(0).astype(int)
    )

    # ------------------------------------------------------------------
    # 4. Basic time features
    # ------------------------------------------------------------------
    hourly["hour_of_day"] = hourly.index.hour
    hourly["day_of_week"] = hourly.index.dayofweek

    # ------------------------------------------------------------------
    # 5. Cyclical encoding (sin/cos) for hour and day
    # ------------------------------------------------------------------
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour_of_day"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour_of_day"] / 24)
    hourly["day_sin"] = np.sin(2 * np.pi * hourly["day_of_week"] / 7)
    hourly["day_cos"] = np.cos(2 * np.pi * hourly["day_of_week"] / 7)

    # ------------------------------------------------------------------
    # 6. Weekend flag
    # ------------------------------------------------------------------
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)

    # ------------------------------------------------------------------
    # 7. Peak hour flag (7-11am, 6-11pm EST)
    # ------------------------------------------------------------------
    hourly["is_peak_hour"] = (
        ((hourly["hour_of_day"] >= 7) & (hourly["hour_of_day"] <= 11))
        | ((hourly["hour_of_day"] >= 18) & (hourly["hour_of_day"] <= 23))
    ).astype(int)

    # ------------------------------------------------------------------
    # 8. hours_since_last_post
    # ------------------------------------------------------------------
    hours_since = []
    last_seen: Optional[datetime] = None
    for ts, row in hourly.iterrows():
        if last_seen is None:
            hours_since.append(np.nan)
        else:
            delta = (ts - last_seen).total_seconds() / 3600
            hours_since.append(round(delta, 2))
        if row["post_count"] > 0:
            last_seen = ts

    hourly["hours_since_last_post"] = hours_since
    median_gap = pd.Series(hours_since).median()
    hourly["hours_since_last_post"] = hourly["hours_since_last_post"].fillna(median_gap)

    # ------------------------------------------------------------------
    # 9. rolling_post_count_24h
    # ------------------------------------------------------------------
    hourly["rolling_post_count_24h"] = (
        hourly["post_count"]
        .rolling(window=24, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # 10. posting_velocity_4h (posts per hour in last 4 hours)
    # ------------------------------------------------------------------
    hourly["posting_velocity_4h"] = (
        hourly["post_count"]
        .rolling(window=4, min_periods=1)
        .mean()
        .shift(1)
        .fillna(0)
    )

    # ------------------------------------------------------------------
    # 11. gap_acceleration — change in hours_since_last_post
    # ------------------------------------------------------------------
    hourly["gap_acceleration"] = (
        hourly["hours_since_last_post"].diff().fillna(0)
    )

    # ------------------------------------------------------------------
    # 12. avg_posts_this_hour_hist — historical average for this hour
    # ------------------------------------------------------------------
    hour_means = hourly.groupby("hour_of_day")["post_count"].transform("mean")
    hourly["avg_posts_this_hour_hist"] = hour_means

    # ==================================================================
    # NEW FEATURES
    # ==================================================================

    # ------------------------------------------------------------------
    # 13-14. News context: news_volume_4h, news_sentiment
    # Aggregate per-post news data into hourly buckets, forward-fill gaps
    # ------------------------------------------------------------------
    has_news_data = "news_volume_4h" in df_raw.columns and "news_sentiment" in df_raw.columns
    if has_news_data:
        news_hourly = df_raw.groupby("hour_bucket").agg(
            news_volume_4h=("news_volume_4h", "max"),
            news_sentiment=("news_sentiment", "mean"),
        )
        hourly = hourly.join(news_hourly)
        hourly["news_volume_4h"] = hourly["news_volume_4h"].ffill().fillna(8).astype(float)
        hourly["news_sentiment"] = hourly["news_sentiment"].ffill().fillna(0.0)
    else:
        # Generate synthetic news features if not in source data
        rng = np.random.RandomState(42)
        n = len(hourly)
        hourly["news_volume_4h"] = np.clip(
            rng.normal(10, 5, n) + hourly["post_count"] * 2, 0, 50
        ).round(0)
        hourly["news_sentiment"] = np.clip(
            rng.normal(0.05, 0.3, n) - (hourly["news_volume_4h"] - 10) * 0.01,
            -1.0, 1.0,
        ).round(3)

    # ------------------------------------------------------------------
    # 15. Regime classification (rule-based heuristic)
    #     High-Activity Regime (1): rolling 24h posts > 5
    #     Dormant Regime (0): otherwise
    # ------------------------------------------------------------------
    hourly["regime_flag"] = (hourly["rolling_post_count_24h"] > 5).astype(int)

    # ------------------------------------------------------------------
    # 16. Market feedback: simulated_vix_4h_vol
    #     VIX trailing volatility — high VIX = crisis mode = more posts
    # ------------------------------------------------------------------
    has_vix_data = "simulated_vix_4h_vol" in df_raw.columns
    if has_vix_data:
        vix_hourly = df_raw.groupby("hour_bucket").agg(
            simulated_vix_4h_vol=("simulated_vix_4h_vol", "mean"),
        )
        hourly = hourly.join(vix_hourly)
        hourly["simulated_vix_4h_vol"] = hourly["simulated_vix_4h_vol"].ffill().fillna(18.0)
    else:
        # Generate synthetic VIX correlated with posting activity
        rng = np.random.RandomState(99)
        n = len(hourly)
        vix_base = rng.normal(18, 3, n)
        # Higher post velocity -> higher VIX (market turmoil drives posts)
        vix_base += hourly["posting_velocity_4h"].values * 2
        hourly["simulated_vix_4h_vol"] = np.clip(vix_base, 10, 50).round(2)

    # Drop the last row (target shifted off the end)
    hourly = hourly.iloc[:-1].copy()

    print(
        f"[prep_data] {len(hourly)} hourly rows | "
        f"positive rate: {hourly['posted_in_next_hour'].mean():.1%} | "
        f"features: {len(FEATURE_COLS)}"
    )
    return hourly


# ===========================================================================
# STEP 2 — TRAIN & INFERENCE
# ===========================================================================

def train_model(df: pd.DataFrame) -> GradientBoostingClassifier:
    """
    Train a GradientBoostingClassifier on the enhanced 16-feature matrix.
    Saves model + metrics to models/ directory.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    X = df[FEATURE_COLS].values
    y = df["posted_in_next_hour"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute sample weights to handle class imbalance
    sample_weights = np.where(y_train == 1, 3.0, 1.0)

    clf = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.08,
        min_samples_leaf=15,
        subsample=0.85,
        random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluation
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["no_post", "post"])
    print("[train_model] Classification report on held-out test set:")
    print(report)

    # Feature importance
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_))
    print("[train_model] Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:30s} {imp:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics for dashboard
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_post": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_post": float(recall_score(y_test, y_pred, pos_label=1)),
        "f1_post": float(f1_score(y_test, y_pred, pos_label=1)),
        "confusion_matrix": cm.tolist(),
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "classification_report": report,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(clf, MODEL_PATH)
    print(f"[train_model] Model saved -> {MODEL_PATH}")
    print(f"[train_model] Metrics saved -> {METRICS_PATH}")
    return clf


def predict_live_probability(
    current_time_hour: int = 10,
    hours_since_last: float = 1.5,
    post_count_24h: int = 8,
    posting_velocity_4h: float = 0.5,
    gap_acceleration: float = 0.0,
    day_of_week: int | None = None,
    news_volume_4h: float = 10.0,
    news_sentiment: float = 0.0,
    simulated_vix_4h_vol: float = 18.0,
) -> float:
    """
    Lightweight inference: load the saved model and return P(post in next hour).

    Now accepts news context, regime, and VIX market feedback features.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model() first."
        )

    clf = joblib.load(MODEL_PATH)

    hour = current_time_hour
    if day_of_week is None:
        day_of_week = datetime.now(tz=EST).weekday()

    # Compute all features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if (7 <= hour <= 11) or (18 <= hour <= 23) else 0

    # Historical average posts per hour (rough estimate)
    peak_hours_avg = {
        0: 0.4, 1: 0.2, 2: 0.05, 3: 0.02, 4: 0.01, 5: 0.02,
        6: 0.1, 7: 0.5, 8: 0.8, 9: 1.0, 10: 1.1, 11: 1.0,
        12: 0.8, 13: 0.7, 14: 0.7, 15: 0.6, 16: 0.6, 17: 0.7,
        18: 0.8, 19: 0.9, 20: 1.0, 21: 1.1, 22: 0.9, 23: 0.6,
    }
    avg_posts_this_hour_hist = peak_hours_avg.get(hour, 0.5)

    # Regime flag: high-activity if >5 posts in last 24h
    regime_flag = 1 if post_count_24h > 5 else 0

    features = np.array([[
        hour_sin,
        hour_cos,
        day_sin,
        day_cos,
        is_weekend,
        is_peak_hour,
        hours_since_last,
        post_count_24h,
        posting_velocity_4h,
        gap_acceleration,
        avg_posts_this_hour_hist,
        hour,
        # New features
        news_volume_4h,
        news_sentiment,
        regime_flag,
        simulated_vix_4h_vol,
    ]])

    prob: float = clf.predict_proba(features)[0][1]

    # VIX exponential scaling: if VIX is elevated (>25), boost probability
    # This models the feedback loop where market crisis drives reactive posting
    if simulated_vix_4h_vol > 25:
        vix_boost = (math.exp((simulated_vix_4h_vol - 25) / 15) - 1) * 0.15
        prob = min(0.99, prob + vix_boost)

    return round(float(prob), 4)


# ===========================================================================
# STEP 3 — LLM TOPIC PREDICTOR
# ===========================================================================

def predict_next_topic(last_5_tweets_list: list[str]) -> str:
    """
    Use Gemini to classify what topic Trump will post about next.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")

    import os
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
    model = genai.GenerativeModel("gemini-2.5-flash")

    numbered_tweets = "\n".join(
        f"{i+1}. {t}" for i, t in enumerate(last_5_tweets_list[-5:])
    )

    prompt = (
        "You are a political analyst specializing in Donald Trump's Truth Social posts. "
        "Your only job is to analyze recent posts and predict the single most likely topic "
        "of his NEXT post.\n\n"
        "You MUST respond with EXACTLY ONE word from this list — no punctuation, no explanation:\n"
        "Tariffs, Crypto, Media, Borders, Fed, Cabinet, Other\n\n"
        "Rules:\n"
        "- Output ONLY the single keyword.\n"
        "- Do NOT add any other text, punctuation, or newlines.\n"
        "- If context is ambiguous, choose the closest match or 'Other'.\n\n"
        f"Here are Trump's {len(last_5_tweets_list[-5:])} most recent Truth Social posts "
        f"(oldest to newest):\n\n{numbered_tweets}\n\n"
        "Based on these posts, what is the single most likely topic of his next post?"
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=10,
        ),
    )

    raw = response.text.strip()

    for label in TOPIC_LABELS:
        if label.lower() == raw.lower():
            return label
    for label in TOPIC_LABELS:
        if label.lower() in raw.lower():
            return label
    return "Other"


# ===========================================================================
# STEP 4 — MODEL METRICS & HEATMAP HELPERS
# ===========================================================================

def get_model_metrics() -> dict:
    """Load saved model metrics for dashboard display."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def get_hourly_heatmap_data(csv_path: str = None, pre_ingested_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build hour-of-day x day-of-week heatmap of posting frequency.
    Returns a 24x7 DataFrame (index=hour 0-23, columns=day 0-6).
    """
    if pre_ingested_df is not None:
        df = pre_ingested_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif csv_path:
        df = ingest_real_data(csv_path)
    else:
        return pd.DataFrame()

    df["ts_est"] = df["timestamp"].dt.tz_convert(EST)
    df["hour"] = df["ts_est"].dt.hour
    df["dow"] = df["ts_est"].dt.dayofweek

    heatmap = df.groupby(["hour", "dow"]).size().unstack(fill_value=0)
    # Normalize to average posts per hour per day
    total_weeks = max(1, (df["ts_est"].max() - df["ts_est"].min()).days / 7)
    heatmap = (heatmap / total_weeks).round(2)

    # Ensure all hours and days are present
    heatmap = heatmap.reindex(index=range(24), columns=range(7), fill_value=0.0)

    return heatmap


# ===========================================================================
# CLI — full pipeline with real data
# ===========================================================================

REAL_DATA_PATH = "data/truth_archive.csv"
FALLBACK_DATA_PATH = "data/sample_posts.csv"

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else REAL_DATA_PATH

    if Path(csv_path).exists():
        print(f"[ml_engine] Loading REAL dataset: {csv_path}")
        clean_df = ingest_real_data(csv_path)
    elif Path(FALLBACK_DATA_PATH).exists():
        print(f"[ml_engine] Real data not found. Using fallback: {FALLBACK_DATA_PATH}")
        clean_df = None
        csv_path = FALLBACK_DATA_PATH
    else:
        print("[ml_engine] No real dataset found. Generating synthetic data...")
        from generate_sample_data import generate_posts, save_csv
        save_csv(generate_posts(100), FALLBACK_DATA_PATH)
        clean_df = None
        csv_path = FALLBACK_DATA_PATH

    # Step 1: Feature engineering
    df = prep_data(csv_path, pre_ingested_df=clean_df)
    print()
    print("=== SUMMARY ===")
    print(f"  Real posts used:   {len(clean_df) if clean_df is not None else 'N/A (synthetic)'}")
    print(f"  Hourly rows:       {len(df)}")
    print(f"  Positive rate:     {df['posted_in_next_hour'].mean():.1%}")
    print(f"  Features:          {len(FEATURE_COLS)}")
    print()

    # Step 2: Train
    clf = train_model(df)

    # Step 3: Example predictions
    print("\n=== EXAMPLE LIVE PREDICTIONS ===")
    scenarios = [
        ("Morning burst (9am, 0.25h since, 12/24h, VIX=20)", 9, 0.25, 12, 2.0, -0.5, 15, -0.1, 20.0),
        ("Crisis mode (11pm, 1h since, 8/24h, VIX=35)", 23, 1.0, 8, 0.5, 0.5, 30, -0.5, 35.0),
        ("Dead zone (4am, 5h since, 3/24h, VIX=15)", 4, 5.0, 3, 0.0, 1.0, 3, 0.2, 15.0),
        ("High news vol (2pm, 3h since, 6/24h, VIX=22)", 14, 3.0, 6, 0.25, 0.5, 25, -0.3, 22.0),
        ("Evening + crisis (7pm, 0.3h since, 15/24h, VIX=40)", 19, 0.3, 15, 3.0, -1.0, 40, -0.7, 40.0),
    ]

    for label, hour, since, count, vel, accel, nv, ns, vix in scenarios:
        prob = predict_live_probability(
            hour, since, count, vel, accel,
            news_volume_4h=nv, news_sentiment=ns, simulated_vix_4h_vol=vix,
        )
        print(f"  {label}")
        print(f"    -> P(post next hour) = {prob:.2%}")
    print()

    # Step 4: Topic prediction (optional)
    if os.getenv("OPENAI_API_KEY") and clean_df is not None:
        last_5 = clean_df["text"].tail(5).tolist()
        topic = predict_next_topic(last_5)
        print(f"[predict_next_topic] Based on 5 most recent REAL posts: {topic}")
    else:
        print("[predict_next_topic] Skipped (set OPENAI_API_KEY to enable).")

    # Step 5: Heatmap data
    if clean_df is not None:
        heatmap = get_hourly_heatmap_data(pre_ingested_df=clean_df)
        print(f"\n[heatmap] Shape: {heatmap.shape}")
        print(heatmap.head())
