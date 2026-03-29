"""
time_travel.py — Time Travel Replay Engine
============================================
Replays historical data hour-by-hour, comparing MentionBot
predictions against what actually happened.

Public API:
  - get_available_dates(csv_path)                  -> list[str]
  - load_replay_day(date_str, csv_path)            -> dict
  - run_full_day_replay(date_str, csv_path)         -> dict
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from ml_engine import (
    ingest_real_data,
    prep_data,
    predict_live_probability,
    EST,
    MODEL_PATH,
)

REAL_DATA_PATH = Path(__file__).parent / "data" / "truth_archive.csv"


def get_available_dates(csv_path: str | None = None) -> list[str]:
    """
    Return a list of dates (YYYY-MM-DD) that have at least 3 posts,
    making them interesting for replay. Returns up to 100 dates.
    """
    path = csv_path or str(REAL_DATA_PATH)
    if not Path(path).exists():
        return []

    df = ingest_real_data(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date_est"] = df["timestamp"].dt.tz_convert(EST).dt.date

    # Count posts per day
    daily_counts = df.groupby("date_est").size()
    # Only include days with 3+ posts (interesting days)
    active_days = daily_counts[daily_counts >= 3].index.tolist()
    # Sort descending (most recent first)
    active_days.sort(reverse=True)

    return [d.isoformat() for d in active_days[:200]]


def load_replay_day(
    date_str: str,
    csv_path: str | None = None,
    pre_ingested_df: pd.DataFrame | None = None,
) -> dict:
    """
    Load all data for a specific date and prepare hour-by-hour replay frames.

    Args:
        date_str: "YYYY-MM-DD" format
        csv_path: path to CSV (uses default if None)
        pre_ingested_df: optional pre-cleaned DataFrame

    Returns dict with:
        - date: the date string
        - total_posts: how many posts happened this day
        - frames: list of 24 dicts, one per hour (0-23), each containing:
            - hour: 0-23
            - prediction: float probability from model
            - actual_posts: int (how many posts actually happened this hour)
            - actual_posted: bool
            - hit: bool (prediction correct?)
            - tweets_this_hour: list of tweet texts that happened this hour
            - features: dict of feature values used
    """
    path = csv_path or str(REAL_DATA_PATH)

    # Load all data
    if pre_ingested_df is not None:
        df = pre_ingested_df.copy()
    else:
        df = ingest_real_data(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_est"] = df["timestamp"].dt.tz_convert(EST)

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Get posts for the target date
    day_mask = df["ts_est"].dt.date == target_date
    day_posts = df[day_mask].copy()

    # Get posts from the previous 24 hours for context
    target_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=EST)
    context_start = target_start - timedelta(hours=24)
    context_mask = (df["ts_est"] >= context_start) & (df["ts_est"] < target_start)
    context_posts = df[context_mask]

    # Build hour-by-hour frames
    frames = []
    running_post_count_24h = len(context_posts)  # Start with previous 24h count
    last_post_time = None

    # Find the last post before this day
    before_day = df[df["ts_est"] < target_start]
    if len(before_day) > 0:
        last_post_time = before_day["ts_est"].iloc[-1]

    # Track recent posts for velocity calculation
    recent_hours_posts = []  # list of (hour, count) for last 4 hours

    for hour in range(24):
        hour_start = target_start + timedelta(hours=hour)
        hour_end = hour_start + timedelta(hours=1)

        # Posts in this hour
        hour_mask = (day_posts["ts_est"] >= hour_start) & (day_posts["ts_est"] < hour_end)
        hour_posts = day_posts[hour_mask]
        actual_count = len(hour_posts)
        actual_posted = actual_count > 0

        # Calculate features
        if last_post_time is not None:
            hours_since = (hour_start - last_post_time).total_seconds() / 3600
        else:
            hours_since = 12.0  # default

        # Posting velocity (last 4 hours)
        recent_hours_posts.append(actual_count)
        if len(recent_hours_posts) > 4:
            recent_hours_posts.pop(0)
        velocity = sum(recent_hours_posts) / len(recent_hours_posts)

        # Day of week
        dow = hour_start.weekday()

        # Make prediction
        try:
            prediction = predict_live_probability(
                current_time_hour=hour,
                hours_since_last=min(hours_since, 24.0),
                post_count_24h=running_post_count_24h,
                posting_velocity_4h=velocity,
                gap_acceleration=0.0,
                day_of_week=dow,
            )
        except Exception:
            prediction = 0.5  # Fallback

        # Determine if prediction was correct
        threshold = 0.50
        predicted_hot = prediction >= threshold
        hit = predicted_hot == actual_posted

        tweets_this_hour = hour_posts["text"].tolist() if actual_posted else []

        frames.append({
            "hour": hour,
            "time_label": f"{hour:02d}:00",
            "prediction": prediction,
            "signal": "HOT" if predicted_hot else "COLD",
            "actual_posts": actual_count,
            "actual_posted": actual_posted,
            "hit": hit,
            "tweets_this_hour": tweets_this_hour[:5],  # Cap at 5
            "features": {
                "hours_since_last": round(hours_since, 2),
                "post_count_24h": running_post_count_24h,
                "posting_velocity_4h": round(velocity, 2),
                "day_of_week": dow,
            },
        })

        # Update running state
        if actual_posted and last_post_time is not None:
            last_post_time = hour_posts["ts_est"].iloc[-1]
        elif actual_posted:
            last_post_time = hour_posts["ts_est"].iloc[-1]

        # Update 24h count (add this hour's posts, remove posts from 24h ago)
        running_post_count_24h += actual_count

    # Calculate accuracy
    hits = sum(1 for f in frames if f["hit"])
    active_hours = sum(1 for f in frames if f["actual_posted"])

    return {
        "date": date_str,
        "total_posts": len(day_posts),
        "active_hours": active_hours,
        "frames": frames,
        "accuracy": round(hits / 24, 3),
        "hits": hits,
        "misses": 24 - hits,
        "day_of_week": target_date.strftime("%A"),
    }


def run_full_day_replay(date_str: str, csv_path: str | None = None) -> dict:
    """
    Convenience wrapper that loads and runs a full day replay.
    Returns the same dict as load_replay_day.
    """
    return load_replay_day(date_str, csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Get available dates
    dates = get_available_dates()
    print(f"[time_travel] {len(dates)} replay-worthy dates available")
    print(f"  Most recent: {dates[:5]}")
    print(f"  Oldest: {dates[-3:]}")

    # Pick a date (default: most recent active day)
    target = sys.argv[1] if len(sys.argv) > 1 else dates[0]
    print(f"\n[time_travel] Running replay for {target}...")

    result = run_full_day_replay(target)

    print(f"\n=== REPLAY: {result['date']} ({result['day_of_week']}) ===")
    print(f"  Total posts: {result['total_posts']}")
    print(f"  Active hours: {result['active_hours']}")
    print(f"  Accuracy: {result['accuracy']:.0%} ({result['hits']}/24 hours correct)")
    print()

    for frame in result["frames"]:
        symbol = "✅" if frame["hit"] else "❌"
        post_indicator = f"📢 {frame['actual_posts']} post(s)" if frame["actual_posted"] else "  (quiet)"
        print(
            f"  {frame['time_label']} | "
            f"Pred: {frame['prediction']:5.1%} [{frame['signal']:4s}] | "
            f"Actual: {post_indicator:20s} | "
            f"{symbol}"
        )
