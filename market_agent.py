"""
market_agent.py — MentionBot Prediction Market Agent
=====================================================
Public API:
  - search_market_contracts(topic_keyword)  -> list[dict]
  - run_agent_cycle(...)                    -> dict   (the full agent payload)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pytz

EST = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Mock Prediction Market Contracts
# ---------------------------------------------------------------------------
# Realistic Polymarket / Kalshi style contracts per topic.
# Each contract: title, yes_price (0-1), no_price (0-1), volume (USD)

MOCK_CONTRACTS: dict[str, list[dict]] = {
    "Tariffs": [
        {
            "contract": "Will Trump announce new tariffs on China by end of Q2 2025?",
            "yes_price": 0.72,
            "no_price": 0.28,
            "volume": 287_000,
            "market": "Polymarket",
        },
        {
            "contract": "US average tariff rate above 20% on July 1?",
            "yes_price": 0.58,
            "no_price": 0.42,
            "volume": 145_000,
            "market": "Kalshi",
        },
        {
            "contract": "Will EU retaliate with counter-tariffs before August 2025?",
            "yes_price": 0.64,
            "no_price": 0.36,
            "volume": 93_500,
            "market": "Polymarket",
        },
    ],
    "Crypto": [
        {
            "contract": "Will Trump mention Bitcoin in an official statement this month?",
            "yes_price": 0.45,
            "no_price": 0.55,
            "volume": 312_000,
            "market": "Polymarket",
        },
        {
            "contract": "Bitcoin above $120k on June 30 2025?",
            "yes_price": 0.38,
            "no_price": 0.62,
            "volume": 520_000,
            "market": "Kalshi",
        },
        {
            "contract": "Executive order on crypto regulation before Q3 2025?",
            "yes_price": 0.51,
            "no_price": 0.49,
            "volume": 178_000,
            "market": "Polymarket",
        },
    ],
    "Media": [
        {
            "contract": "Will Trump revoke a major media outlet's press credentials in 2025?",
            "yes_price": 0.33,
            "no_price": 0.67,
            "volume": 98_000,
            "market": "Polymarket",
        },
        {
            "contract": "Trump lawsuit against a media company filed by July 2025?",
            "yes_price": 0.41,
            "no_price": 0.59,
            "volume": 67_000,
            "market": "Kalshi",
        },
    ],
    "Borders": [
        {
            "contract": "Southern border crossings below 100k/month by June 2025?",
            "yes_price": 0.55,
            "no_price": 0.45,
            "volume": 203_000,
            "market": "Polymarket",
        },
        {
            "contract": "New executive order on immigration before May 2025?",
            "yes_price": 0.78,
            "no_price": 0.22,
            "volume": 156_000,
            "market": "Kalshi",
        },
        {
            "contract": "Will Congress fund border wall expansion in 2025?",
            "yes_price": 0.42,
            "no_price": 0.58,
            "volume": 134_000,
            "market": "Polymarket",
        },
    ],
    "Fed": [
        {
            "contract": "Fed rate cut before July 2025 FOMC meeting?",
            "yes_price": 0.61,
            "no_price": 0.39,
            "volume": 890_000,
            "market": "Kalshi",
        },
        {
            "contract": "Trump publicly calls for Powell's removal in Q2 2025?",
            "yes_price": 0.48,
            "no_price": 0.52,
            "volume": 245_000,
            "market": "Polymarket",
        },
    ],
    "Cabinet": [
        {
            "contract": "Cabinet reshuffle (any secretary replaced) by June 2025?",
            "yes_price": 0.35,
            "no_price": 0.65,
            "volume": 112_000,
            "market": "Polymarket",
        },
        {
            "contract": "New cabinet nominee confirmation hearing before May 2025?",
            "yes_price": 0.52,
            "no_price": 0.48,
            "volume": 89_000,
            "market": "Kalshi",
        },
    ],
    "Other": [
        {
            "contract": "Trump Truth Social post volume above 500 in April 2025?",
            "yes_price": 0.70,
            "no_price": 0.30,
            "volume": 54_000,
            "market": "Polymarket",
        },
        {
            "contract": "Trump approval rating above 50% on May 1 (538 avg)?",
            "yes_price": 0.44,
            "no_price": 0.56,
            "volume": 410_000,
            "market": "Kalshi",
        },
    ],
}


def search_market_contracts(topic_keyword: str) -> list[dict]:
    """
    Simulate a Polymarket/Kalshi API search for active contracts
    matching the given topic keyword.

    Args:
        topic_keyword: one of Tariffs, Crypto, Media, Borders, Fed, Cabinet, Other

    Returns:
        list of contract dicts with keys:
          contract, yes_price, no_price, volume, market
    """
    return MOCK_CONTRACTS.get(topic_keyword, MOCK_CONTRACTS["Other"])


# ---------------------------------------------------------------------------
# Core Agent Cycle
# ---------------------------------------------------------------------------

PROBABILITY_THRESHOLD = 0.50


def run_agent_cycle(
    current_time_hour: int,
    hours_since_last: float,
    post_count_24h: int,
    recent_tweets: list[str],
    use_llm: bool = True,
) -> dict:
    """
    Execute one full agent decision cycle:
      1. Predict P(post in next hour) via the RF model
      2. If above threshold, predict topic via LLM
      3. Search for actionable prediction market contracts

    Args:
        current_time_hour:  0-23 hour of day (EST)
        hours_since_last:   hours since Trump's last post
        post_count_24h:     rolling post count in last 24h
        recent_tweets:      list of recent tweet strings for topic prediction
        use_llm:            if False, skip OpenAI call (for demo without API key)

    Returns:
        dict with keys:
          probability, threshold, signal, predicted_topic, contracts, timestamp
    """
    from ml_engine import predict_live_probability, predict_next_topic

    # Build timezone-aware datetimes for the model
    now = datetime.now(tz=EST).replace(
        hour=current_time_hour, minute=0, second=0, microsecond=0
    )
    last_post_time = now - timedelta(hours=hours_since_last)

    # Step 1: RF probability
    probability = predict_live_probability(now, last_post_time, post_count_24h)

    # Step 2: topic prediction (only if signal is hot)
    predicted_topic = None
    contracts = []

    if probability >= PROBABILITY_THRESHOLD:
        signal = "HOT"

        if use_llm and os.getenv("OPENAI_API_KEY") and recent_tweets:
            predicted_topic = predict_next_topic(recent_tweets)
        elif recent_tweets:
            # Lightweight keyword fallback when no API key
            predicted_topic = _keyword_topic_fallback(recent_tweets)
        else:
            predicted_topic = "Other"

        # Step 3: find contracts
        contracts = search_market_contracts(predicted_topic)
    else:
        signal = "COLD"

    return {
        "probability": probability,
        "threshold": PROBABILITY_THRESHOLD,
        "signal": signal,
        "predicted_topic": predicted_topic,
        "contracts": contracts,
        "sim_hour": current_time_hour,
        "hours_since_last": hours_since_last,
        "post_count_24h": post_count_24h,
    }


def _keyword_topic_fallback(tweets: list[str]) -> str:
    """Simple keyword matching when OpenAI is unavailable."""
    text = " ".join(tweets).lower()
    scores = {
        "Tariffs": sum(w in text for w in ["tariff", "trade", "china", "import", "export"]),
        "Crypto":  sum(w in text for w in ["bitcoin", "crypto", "blockchain", "nft", "digital currency"]),
        "Media":   sum(w in text for w in ["fake news", "media", "cnn", "nbc", "press"]),
        "Borders": sum(w in text for w in ["border", "immigration", "wall", "deport", "ice"]),
        "Fed":     sum(w in text for w in ["fed", "rate", "powell", "inflation", "interest"]),
        "Cabinet": sum(w in text for w in ["cabinet", "secretary", "nominate", "confirm", "appoint"]),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"
