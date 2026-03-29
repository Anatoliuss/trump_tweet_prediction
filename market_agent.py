"""
market_agent.py — MentionBot Prediction Market Agent v2
========================================================
Enhanced with dynamic pricing, sector filtering, and P&L tracking.

Public API:
  - search_market_contracts(topic, sector, probability)  -> list[dict]
  - run_agent_cycle(...)                                  -> dict
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytz

EST = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Mock Prediction Market Contracts — Organized by Topic AND Sector
# ---------------------------------------------------------------------------

MOCK_CONTRACTS: dict[str, list[dict]] = {
    "Tariffs": [
        {
            "contract": "Will Trump announce new tariffs on China by end of Q2?",
            "yes_price": 0.72, "no_price": 0.28, "volume": 287_000,
            "market": "Polymarket", "sector": "US Equities",
        },
        {
            "contract": "US average tariff rate above 20% on July 1?",
            "yes_price": 0.58, "no_price": 0.42, "volume": 145_000,
            "market": "Kalshi", "sector": "US Equities",
        },
        {
            "contract": "Will EU retaliate with counter-tariffs before August?",
            "yes_price": 0.64, "no_price": 0.36, "volume": 93_500,
            "market": "Polymarket", "sector": "Forex",
        },
        {
            "contract": "FXI (China ETF) drops 5%+ within 48h of tariff tweet?",
            "yes_price": 0.41, "no_price": 0.59, "volume": 178_000,
            "market": "Kalshi", "sector": "China/EM",
        },
        {
            "contract": "Gold above $3,200 within 1 week of tariff announcement?",
            "yes_price": 0.55, "no_price": 0.45, "volume": 210_000,
            "market": "Polymarket", "sector": "Commodities",
        },
    ],
    "Crypto": [
        {
            "contract": "Will Trump mention Bitcoin in an official statement this month?",
            "yes_price": 0.45, "no_price": 0.55, "volume": 312_000,
            "market": "Polymarket", "sector": "Crypto",
        },
        {
            "contract": "Bitcoin above $120k on June 30?",
            "yes_price": 0.38, "no_price": 0.62, "volume": 520_000,
            "market": "Kalshi", "sector": "Crypto",
        },
        {
            "contract": "Executive order on crypto regulation before Q3?",
            "yes_price": 0.51, "no_price": 0.49, "volume": 178_000,
            "market": "Polymarket", "sector": "Crypto",
        },
        {
            "contract": "SOL/USD above $300 by end of month after Trump crypto tweet?",
            "yes_price": 0.29, "no_price": 0.71, "volume": 95_000,
            "market": "Kalshi", "sector": "Crypto",
        },
    ],
    "Media": [
        {
            "contract": "Will Trump revoke a major outlet's press credentials?",
            "yes_price": 0.33, "no_price": 0.67, "volume": 98_000,
            "market": "Polymarket", "sector": "US Equities",
        },
        {
            "contract": "Trump lawsuit against a media company filed this quarter?",
            "yes_price": 0.41, "no_price": 0.59, "volume": 67_000,
            "market": "Kalshi", "sector": "US Equities",
        },
    ],
    "Borders": [
        {
            "contract": "Southern border crossings below 100k/month by June?",
            "yes_price": 0.55, "no_price": 0.45, "volume": 203_000,
            "market": "Polymarket", "sector": "US Equities",
        },
        {
            "contract": "New executive order on immigration before May?",
            "yes_price": 0.78, "no_price": 0.22, "volume": 156_000,
            "market": "Kalshi", "sector": "US Equities",
        },
        {
            "contract": "USD/MXN above 18.5 within 1 week of border tweet?",
            "yes_price": 0.62, "no_price": 0.38, "volume": 134_000,
            "market": "Polymarket", "sector": "Forex",
        },
    ],
    "Fed": [
        {
            "contract": "Fed rate cut before July FOMC meeting?",
            "yes_price": 0.61, "no_price": 0.39, "volume": 890_000,
            "market": "Kalshi", "sector": "Bonds",
        },
        {
            "contract": "Trump publicly calls for Powell's removal this quarter?",
            "yes_price": 0.48, "no_price": 0.52, "volume": 245_000,
            "market": "Polymarket", "sector": "US Equities",
        },
        {
            "contract": "TLT (bond ETF) up 3%+ within 48h of dovish Trump tweet?",
            "yes_price": 0.44, "no_price": 0.56, "volume": 315_000,
            "market": "Kalshi", "sector": "Bonds",
        },
        {
            "contract": "DXY below 102 within 1 week of Fed pressure tweet?",
            "yes_price": 0.37, "no_price": 0.63, "volume": 188_000,
            "market": "Polymarket", "sector": "Forex",
        },
    ],
    "Cabinet": [
        {
            "contract": "Cabinet reshuffle (any secretary replaced) by June?",
            "yes_price": 0.35, "no_price": 0.65, "volume": 112_000,
            "market": "Polymarket", "sector": "US Equities",
        },
        {
            "contract": "New cabinet nominee confirmation hearing before May?",
            "yes_price": 0.52, "no_price": 0.48, "volume": 89_000,
            "market": "Kalshi", "sector": "US Equities",
        },
    ],
    "Other": [
        {
            "contract": "Trump Truth Social post volume above 500 this month?",
            "yes_price": 0.70, "no_price": 0.30, "volume": 54_000,
            "market": "Polymarket", "sector": "US Equities",
        },
        {
            "contract": "Trump approval rating above 50% on May 1 (538 avg)?",
            "yes_price": 0.44, "no_price": 0.56, "volume": 410_000,
            "market": "Kalshi", "sector": "US Equities",
        },
    ],
}


def search_market_contracts(
    topic_keyword: str,
    sector_filter: str | None = None,
    probability: float = 0.5,
) -> list[dict]:
    """
    Search for contracts matching topic + optional sector filter.
    Dynamically adjusts prices based on prediction probability.
    """
    contracts = MOCK_CONTRACTS.get(topic_keyword, MOCK_CONTRACTS["Other"])

    # Filter by sector
    if sector_filter and sector_filter != "All":
        contracts = [c for c in contracts if c.get("sector") == sector_filter]
        # If no contracts match the sector, show all for the topic
        if not contracts:
            contracts = MOCK_CONTRACTS.get(topic_keyword, MOCK_CONTRACTS["Other"])

    # Dynamically adjust prices based on probability
    adjusted = []
    for c in contracts:
        ac = c.copy()
        # Higher prediction probability → shift YES prices up slightly
        prob_shift = (probability - 0.5) * 0.08  # ±4% shift
        ac["yes_price"] = round(min(0.95, max(0.05, c["yes_price"] + prob_shift)), 2)
        ac["no_price"] = round(1.0 - ac["yes_price"], 2)
        # Volume increases with higher probability (more interest)
        vol_mult = 0.8 + probability * 0.4  # 0.8x - 1.2x
        ac["volume"] = int(c["volume"] * vol_mult)
        adjusted.append(ac)

    return adjusted


# ---------------------------------------------------------------------------
# Core Agent Cycle
# ---------------------------------------------------------------------------

PROBABILITY_THRESHOLD = 0.35


def run_agent_cycle(
    current_time_hour: int,
    hours_since_last: float,
    post_count_24h: int,
    recent_tweets: list[str],
    use_llm: bool = True,
    sector_filter: str | None = None,
    posting_velocity_4h: float = 0.5,
    gap_acceleration: float = 0.0,
) -> dict:
    """
    Execute one full agent decision cycle:
      1. Predict P(post in next hour) via the ML model
      2. If above threshold, predict topic via LLM
      3. Search for actionable prediction market contracts
      4. Get market impact analysis

    Returns dict with all results.
    """
    from ml_engine import predict_live_probability, predict_next_topic
    from market_impact import get_impact_for_topic, simulate_market_reaction

    # Step 1: ML probability
    probability = predict_live_probability(
        current_time_hour=current_time_hour,
        hours_since_last=hours_since_last,
        post_count_24h=post_count_24h,
        posting_velocity_4h=posting_velocity_4h,
        gap_acceleration=gap_acceleration,
    )

    # Step 2: topic prediction (only if signal is hot)
    predicted_topic = None
    contracts = []
    market_impacts = []

    if probability >= PROBABILITY_THRESHOLD:
        signal = "HOT"

        if use_llm and os.getenv("OPENAI_API_KEY") and recent_tweets:
            predicted_topic = predict_next_topic(recent_tweets)
        elif recent_tweets:
            predicted_topic = _keyword_topic_fallback(recent_tweets)
        else:
            predicted_topic = "Other"

        # Step 3: find contracts
        contracts = search_market_contracts(predicted_topic, sector_filter, probability)

        # Step 4: market impact
        market_impacts = simulate_market_reaction(
            predicted_topic, probability, sector_filter,
            seed=int(current_time_hour * 100 + post_count_24h)
        )
    else:
        signal = "COLD"

    return {
        "probability": probability,
        "threshold": PROBABILITY_THRESHOLD,
        "signal": signal,
        "predicted_topic": predicted_topic,
        "contracts": contracts,
        "market_impacts": market_impacts,
        "sim_hour": current_time_hour,
        "hours_since_last": hours_since_last,
        "post_count_24h": post_count_24h,
    }


def _keyword_topic_fallback(tweets: list[str]) -> str:
    """Simple keyword matching when OpenAI is unavailable."""
    text = " ".join(tweets).lower()
    scores = {
        "Tariffs": sum(w in text for w in ["tariff", "trade", "china", "import", "export", "duty", "duties"]),
        "Crypto":  sum(w in text for w in ["bitcoin", "crypto", "blockchain", "nft", "digital currency", "btc", "eth"]),
        "Media":   sum(w in text for w in ["fake news", "media", "cnn", "nbc", "press", "journalist"]),
        "Borders": sum(w in text for w in ["border", "immigration", "wall", "deport", "ice", "migrant", "illegal"]),
        "Fed":     sum(w in text for w in ["fed", "rate", "powell", "inflation", "interest", "federal reserve"]),
        "Cabinet": sum(w in text for w in ["cabinet", "secretary", "nominate", "confirm", "appoint", "resign"]),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"
