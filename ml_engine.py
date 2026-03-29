"""
ml_engine.py — MentionBot Predictive Engine
============================================
Public API (import these in your agent script):
  - prep_data(csv_path)                        -> pd.DataFrame
  - train_model(df)                            -> RandomForestClassifier
  - predict_live_probability(current_time,
                             last_post_time,
                             post_count_24h)   -> float  (0.0 – 1.0)
  - predict_next_topic(last_5_tweets_list)     -> str    (one of TOPIC_LABELS)

PlanetScale-compatible schema:
  timestamp       TEXT (ISO-8601 with tz)
  text            TEXT
  predicted_topic TEXT (one of TOPIC_LABELS | "")
  probability_score REAL (0.0 – 1.0 | NULL)
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EST = pytz.timezone("America/New_York")
MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "mentionbot_rf.joblib"

FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "hours_since_last_post",
    "rolling_post_count_24h",
]

TOPIC_LABELS = ["Tariffs", "Crypto", "Media", "Borders", "Fed", "Cabinet", "Other"]


# ===========================================================================
# STEP 1 — DATA PREP & FEATURE ENGINEERING
# ===========================================================================

def ingest_real_data(csv_path: str) -> pd.DataFrame:
    """
    Load any Trump Truth Social CSV, auto-detect columns, clean, and
    normalise to the canonical schema: (timestamp, text).

    Handles:
      - stiles/trump-truth-social-archive  (created_at, content)
      - Kaggle Trump Tweets                (date/timestamp, text/tweet)
      - Our own sample format              (timestamp, text)

    Filters:
      - Removes reblogs/retweets (RT prefix or reblogs_count > 0)
      - Removes duplicates by text content
      - Sorts chronologically
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
        raise ValueError(
            f"Cannot detect timestamp column. Columns: {list(df.columns)}"
        )

    # --- Auto-detect text column ---
    text_candidates = ["content", "text", "tweet", "body", "message", "post"]
    text_col = None
    for c in text_candidates:
        matches = [col for col in df.columns if col.lower() == c.lower()]
        if matches:
            text_col = matches[0]
            break
    if text_col is None:
        raise ValueError(
            f"Cannot detect text column. Columns: {list(df.columns)}"
        )

    print(f"[ingest] Detected columns: timestamp='{ts_col}', text='{text_col}'")

    # --- Rename to canonical schema ---
    df = df.rename(columns={ts_col: "timestamp", text_col: "text"})

    # --- Remove reblogs/retweets ---
    # Note: reblogs_count in the stiles archive = how many times OTHERS shared
    # the post, NOT whether the post itself is a reblog. Use RT prefix instead.

    # Remove RT-prefixed posts
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

    # --- Parse timestamps, coerce errors ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Keep only timestamp and text (+ any extra columns for PlanetScale)
    df = df[["timestamp", "text"]].sort_values("timestamp").reset_index(drop=True)

    print(
        f"[ingest] {original_count} raw rows -> {len(df)} clean original posts "
        f"({df['timestamp'].min().strftime('%Y-%m-%d')} to "
        f"{df['timestamp'].max().strftime('%Y-%m-%d')})"
    )
    return df


def prep_data(csv_path: str, pre_ingested_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Load historical Truth Social posts CSV and return an hourly feature matrix.

    Args:
        csv_path: path to CSV (used when pre_ingested_df is None)
        pre_ingested_df: optional pre-cleaned DataFrame with (timestamp, text)
                         — skips CSV loading + column detection if provided.

    Returns a DataFrame indexed by EST hour with columns:
      hour_of_day, day_of_week, hours_since_last_post,
      rolling_post_count_24h, posted_in_next_hour (target)
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

    # Localise to EST (handles DST automatically)
    df_raw["timestamp_est"] = df_raw["timestamp"].dt.tz_convert(EST)
    df_raw = df_raw.sort_values("timestamp_est").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Build a complete hourly spine covering the full date range
    # ------------------------------------------------------------------
    floor_start = df_raw["timestamp_est"].iloc[0].floor("h")
    floor_end   = df_raw["timestamp_est"].iloc[-1].floor("h")

    hourly_index = pd.date_range(floor_start, floor_end, freq="h", tz=EST)
    hourly = pd.DataFrame(index=hourly_index)

    # Posts per hour (raw count)
    # Floor in UTC first (no DST ambiguity), then convert to EST for grouping
    df_raw["hour_bucket"] = (
        df_raw["timestamp"].dt.floor("h").dt.tz_convert(EST)
    )
    post_counts = (
        df_raw.groupby("hour_bucket").size().rename("post_count")
    )
    hourly = hourly.join(post_counts).fillna(0)
    hourly["post_count"] = hourly["post_count"].astype(int)

    # ------------------------------------------------------------------
    # 3. Target: did any post appear in the *next* hour?
    # ------------------------------------------------------------------
    hourly["posted_in_next_hour"] = (
        hourly["post_count"].shift(-1).fillna(0).gt(0).astype(int)
    )

    # ------------------------------------------------------------------
    # 4. Feature: hour_of_day, day_of_week
    # ------------------------------------------------------------------
    hourly["hour_of_day"] = hourly.index.hour          # 0–23
    hourly["day_of_week"]  = hourly.index.dayofweek    # 0=Mon … 6=Sun

    # ------------------------------------------------------------------
    # 5. Feature: hours_since_last_post
    #    Walk forward tracking the last hour that had ≥1 post.
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
    # Fill the initial NaN with the dataset median so the model has a value
    median_gap = pd.Series(hours_since).median()
    hourly["hours_since_last_post"] = hourly["hours_since_last_post"].fillna(median_gap)

    # ------------------------------------------------------------------
    # 6. Feature: rolling_post_count_24h
    #    Sum of posts in the 24 hours *ending at* (not including) this hour.
    # ------------------------------------------------------------------
    hourly["rolling_post_count_24h"] = (
        hourly["post_count"]
        .rolling(window=24, min_periods=1)
        .sum()
        .shift(1)           # exclude current hour → look-back only
        .fillna(0)
        .astype(int)
    )

    # Drop the last row (target shifted off the end)
    hourly = hourly.iloc[:-1].copy()

    print(
        f"[prep_data] {len(hourly)} hourly rows | "
        f"positive rate: {hourly['posted_in_next_hour'].mean():.1%}"
    )
    return hourly


# ===========================================================================
# STEP 2 — TRAIN & INFERENCE
# ===========================================================================

def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the engineered hourly feature matrix.

    Saves the fitted model to models/mentionbot_rf.joblib.
    Returns the fitted classifier.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    X = df[FEATURE_COLS].values
    y = df["posted_in_next_hour"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",   # handles the natural class imbalance
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("[train_model] Classification report on held-out test set:")
    print(classification_report(y_test, y_pred, target_names=["no_post", "post"]))

    # Feature importance summary
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_))
    print("[train_model] Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:30s} {imp:.3f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"[train_model] Model saved -> {MODEL_PATH}")
    return clf


def predict_live_probability(
    current_time: datetime,
    last_post_time: datetime,
    post_count_24h: int,
) -> float:
    """
    Lightweight inference: load the saved model and return P(post in next hour).

    Args:
        current_time:   aware datetime (any tz; converted internally to EST)
        last_post_time: aware datetime of the most recent post
        post_count_24h: number of posts in the last 24 h (rolling)

    Returns:
        float in [0.0, 1.0] — e.g. 0.82 means 82 % probability
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model() first."
        )

    clf: RandomForestClassifier = joblib.load(MODEL_PATH)

    # Normalise to EST
    ct  = current_time.astimezone(EST)
    lpt = last_post_time.astimezone(EST)

    hour_of_day         = ct.hour
    day_of_week         = ct.weekday()
    hours_since_last    = (ct - lpt).total_seconds() / 3600
    rolling_post_count  = int(post_count_24h)

    features = np.array([[
        hour_of_day,
        day_of_week,
        hours_since_last,
        rolling_post_count,
    ]])

    prob: float = clf.predict_proba(features)[0][1]   # P(class=1)
    return round(float(prob), 4)


# ===========================================================================
# STEP 3 — LLM TOPIC PREDICTOR
# ===========================================================================

def predict_next_topic(last_5_tweets_list: list[str]) -> str:
    """
    Use GPT-4o-mini to classify what topic Trump will post about next.

    Args:
        last_5_tweets_list: list of up to 5 recent post strings (newest last)

    Returns:
        One of: Tariffs | Crypto | Media | Borders | Fed | Cabinet | Other
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI()  # reads OPENAI_API_KEY from environment

    numbered_tweets = "\n".join(
        f"{i+1}. {t}" for i, t in enumerate(last_5_tweets_list[-5:])
    )

    system_prompt = (
        "You are a political analyst specializing in Donald Trump's Truth Social posts. "
        "Your only job is to analyze recent posts and predict the single most likely topic "
        "of his NEXT post.\n\n"
        "You MUST respond with EXACTLY ONE word from this list — no punctuation, no explanation:\n"
        "Tariffs, Crypto, Media, Borders, Fed, Cabinet, Other\n\n"
        "Rules:\n"
        "- Output ONLY the single keyword.\n"
        "- Do NOT add any other text, punctuation, or newlines.\n"
        "- If context is ambiguous, choose the closest match or 'Other'."
    )

    user_prompt = (
        f"Here are Trump's {len(last_5_tweets_list[-5:])} most recent Truth Social posts "
        f"(oldest to newest):\n\n{numbered_tweets}\n\n"
        "Based on these posts, what is the single most likely topic of his next post?"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    raw = response.choices[0].message.content.strip()

    # Guard: ensure the model returned a valid label
    for label in TOPIC_LABELS:
        if label.lower() == raw.lower():
            return label

    # Fallback: partial match
    for label in TOPIC_LABELS:
        if label.lower() in raw.lower():
            return label

    return "Other"


# ===========================================================================
# CLI — full pipeline with real data
# ===========================================================================

REAL_DATA_PATH = "data/truth_archive.csv"
FALLBACK_DATA_PATH = "data/sample_posts.csv"

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else REAL_DATA_PATH

    # --- Prefer real data; fall back to synthetic only if unavailable ---
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

    # --- Step 1: Feature engineering ---
    df = prep_data(csv_path, pre_ingested_df=clean_df)
    print()
    print("=== SUMMARY ===")
    print(f"  Real posts used:   {len(clean_df) if clean_df is not None else 'N/A (synthetic)'}")
    print(f"  Hourly rows:       {len(df)}")
    print(f"  Positive rate:     {df['posted_in_next_hour'].mean():.1%}")
    print()

    # --- Step 2: Train ---
    clf = train_model(df)

    # --- 5 example live predictions across different scenarios ---
    print("\n=== 5 EXAMPLE LIVE PREDICTIONS ===")
    scenarios = [
        ("Morning burst (9am, posted 15min ago, 12 posts/24h)",
         datetime(2025, 3, 1, 9, 0, tzinfo=EST),
         datetime(2025, 3, 1, 8, 45, tzinfo=EST), 12),
        ("Late night (11pm, posted 1h ago, 8 posts/24h)",
         datetime(2025, 3, 1, 23, 0, tzinfo=EST),
         datetime(2025, 3, 1, 22, 0, tzinfo=EST), 8),
        ("Dead zone (4am, posted 5h ago, 3 posts/24h)",
         datetime(2025, 3, 1, 4, 0, tzinfo=EST),
         datetime(2025, 2, 28, 23, 0, tzinfo=EST), 3),
        ("Afternoon quiet (2pm, posted 3h ago, 6 posts/24h)",
         datetime(2025, 3, 1, 14, 0, tzinfo=EST),
         datetime(2025, 3, 1, 11, 0, tzinfo=EST), 6),
        ("Evening peak (7pm, posted 20min ago, 15 posts/24h)",
         datetime(2025, 3, 1, 19, 0, tzinfo=EST),
         datetime(2025, 3, 1, 18, 40, tzinfo=EST), 15),
    ]

    for label, now, last, count in scenarios:
        prob = predict_live_probability(now, last, count)
        print(f"  {label}")
        print(f"    -> P(post next hour) = {prob:.2%}")
    print()

    # --- Step 3 (optional — requires OPENAI_API_KEY) ---
    if os.getenv("OPENAI_API_KEY") and clean_df is not None:
        last_5 = clean_df["text"].tail(5).tolist()
        topic = predict_next_topic(last_5)
        print(f"[predict_next_topic] Based on 5 most recent REAL posts: {topic}")
    elif os.getenv("OPENAI_API_KEY"):
        sample_tweets = [
            "The fake news media is at it again. Total witch hunt!",
            "Bitcoin is the future. We love Crypto!",
            "Tariffs on China will make America RICH again.",
            "Our borders are being invaded. We will stop it!",
            "The Federal Reserve must lower rates NOW. Terrible leadership!",
        ]
        topic = predict_next_topic(sample_tweets)
        print(f"[predict_next_topic] Predicted next topic: {topic}")
    else:
        print("[predict_next_topic] Skipped (set OPENAI_API_KEY to enable).")
