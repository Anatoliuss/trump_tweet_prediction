"""
market_agent.py — MentionBot Prediction Market Agent v3
========================================================
Enhanced with Polymarket arbitrage evaluation and Alpha Delta logic.

Public API:
  - search_market_contracts(topic, sector, probability)  -> list[dict]
  - run_agent_cycle(...)                                  -> dict
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import pytz
import requests

EST = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Polymarket API — Live Contract Search
# ---------------------------------------------------------------------------

GAMMA_API = "https://gamma-api.polymarket.com"

# Topic → search queries for Polymarket
TOPIC_SEARCH_QUERIES: dict[str, list[str]] = {
    "Tariffs": ["tariffs", "trump tariffs", "trade war"],
    "Crypto":  ["crypto", "bitcoin", "trump crypto"],
    "Media":   ["trump media", "free press", "censorship"],
    "Borders": ["immigration", "border", "deportation"],
    "Fed":     ["federal reserve", "interest rates", "fed rate"],
    "Cabinet": ["trump cabinet", "trump administration"],
    "Other":   ["trump", "politics"],
}

# Keywords that a contract must match (at least one) to be considered relevant
TOPIC_RELEVANCE_KEYWORDS: dict[str, list[str]] = {
    "Tariffs": ["tariff", "trade", "import", "export", "duty", "duties", "customs", "wto"],
    "Crypto":  ["crypto", "bitcoin", "btc", "eth", "blockchain", "token", "coin", "digital currency"],
    "Media":   ["media", "press", "news", "cnn", "nbc", "journalist", "censor", "free speech"],
    "Borders": ["border", "immigra", "deport", "migrant", "asylum", "ice ", "wall"],
    "Fed":     ["fed", "rate", "powell", "inflation", "interest", "monetary", "central bank"],
    "Cabinet": ["cabinet", "secretary", "nominate", "appoint", "confirm", "resign", "administration"],
    "Other":   ["trump", "president", "white house", "executive order", "congress", "senate"],
}

# Simple TTL cache for API responses
_contract_cache: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 120  # seconds


def search_market_contracts(
    topic_keyword: str,
    sector_filter: str | None = None,
    probability: float = 0.5,
) -> list[dict]:
    """
    Search Polymarket's Gamma API for live contracts related to the topic.
    Results are cached for 120s to avoid hammering the API on every slider move.
    """
    cache_key = f"{topic_keyword}:{sector_filter or 'all'}"
    now = time.time()

    if cache_key in _contract_cache:
        cached_time, cached_data = _contract_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data

    queries = TOPIC_SEARCH_QUERIES.get(topic_keyword, ["trump"])
    contracts: list[dict] = []
    seen_ids: set[str] = set()

    for query in queries:
        if len(contracts) >= 6:
            break
        try:
            resp = requests.get(
                f"{GAMMA_API}/public-search",
                params={"q": query, "limit_per_type": 12},
                timeout=5,
            )
            resp.raise_for_status()
            events = resp.json().get("events", [])

            for event in events:
                # Skip closed events
                if event.get("closed"):
                    continue

                for market in event.get("markets", []):
                    market_id = market.get("id", "")
                    if market_id in seen_ids:
                        continue
                    seen_ids.add(market_id)

                    # Skip closed/resolved markets
                    if market.get("closed"):
                        continue

                    outcomes = json.loads(market.get("outcomes", "[]"))
                    prices = json.loads(market.get("outcomePrices", "[]"))

                    if len(outcomes) < 2 or len(prices) < 2:
                        continue

                    yes_price = float(prices[0])
                    no_price = float(prices[1])

                    # Skip effectively resolved markets (>95% or <5%)
                    if yes_price < 0.05 or yes_price > 0.95:
                        continue

                    # Relevance filter: contract question must contain topic keywords
                    question = market.get("question", event.get("title", "")).lower()
                    relevance_words = TOPIC_RELEVANCE_KEYWORDS.get(topic_keyword, TOPIC_RELEVANCE_KEYWORDS["Other"])
                    if not any(kw in question for kw in relevance_words):
                        continue

                    contracts.append({
                        "contract": market.get("question", event.get("title", "Unknown")),
                        "yes_price": round(yes_price, 3),
                        "no_price": round(no_price, 3),
                        "volume": int(float(event.get("volume", 0))),
                        "market": "Polymarket",
                    })

                    if len(contracts) >= 6:
                        break
                if len(contracts) >= 6:
                    break
        except Exception:
            continue

    # Sort by volume descending — show the most liquid contracts first
    contracts.sort(key=lambda c: c["volume"], reverse=True)

    _contract_cache[cache_key] = (now, contracts)
    return contracts


# ---------------------------------------------------------------------------
# Signal Evaluation
# ---------------------------------------------------------------------------

HOT_THRESHOLD = 0.50   # P(post) >= 50% → HOT, show contracts


def evaluate_signal(
    ml_probability: float,
    contracts: list[dict],
) -> dict:
    """
    Simple signal evaluation: P(post) is the signal strength.
    If high enough, surface relevant Polymarket contracts.
    """
    is_hot = ml_probability >= HOT_THRESHOLD and len(contracts) > 0

    return {
        "ml_probability": round(ml_probability, 4),
        "has_alpha": is_hot,
        "contracts_with_alpha": contracts if is_hot else [],
    }


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
    session_length: int = 0,
    posts_today: int = 0,
    posted_1h_ago: int = 0,
    posted_2h_ago: int = 0,
    posted_3h_ago: int = 0,
    last_post_text: str = "",
    avg_post_length_session: float = 0.0,
    topic_streak: int = 1,
) -> dict:
    """
    Execute one full agent decision cycle:
      1. Predict P(post in next hour) via the ML model
      2. If above threshold, predict topic via LLM
      3. Search for actionable prediction market contracts
      4. Evaluate signal strength
      5. Get market impact analysis

    Returns dict with all results.
    """
    from ml_engine import predict_live_probability, predict_next_topic
    from market_impact import get_impact_for_topic, simulate_market_reaction

    regime_label = "HIGH-ACTIVITY" if post_count_24h > 5 else "DORMANT"

    probability = predict_live_probability(
        current_time_hour=current_time_hour,
        hours_since_last=hours_since_last,
        post_count_24h=post_count_24h,
        posting_velocity_4h=posting_velocity_4h,
        gap_acceleration=gap_acceleration,
        session_length=session_length,
        posts_today=posts_today,
        posted_1h_ago=posted_1h_ago,
        posted_2h_ago=posted_2h_ago,
        posted_3h_ago=posted_3h_ago,
        last_post_text=last_post_text,
        avg_post_length_session=avg_post_length_session,
        topic_streak=topic_streak,
    )

    # Step 2: topic prediction (only if base probability is above threshold)
    predicted_topic = None
    contracts = []
    market_impacts = []
    signal_eval = {
        "ml_probability": probability,
        "has_alpha": False,
        "contracts_with_alpha": [],
    }

    if probability >= PROBABILITY_THRESHOLD:
        if use_llm and recent_tweets:
            predicted_topic = predict_next_topic(recent_tweets)
        elif recent_tweets:
            predicted_topic = _keyword_topic_fallback(recent_tweets)
        else:
            predicted_topic = "Other"

        # Step 3: find live Polymarket contracts for this topic
        contracts = search_market_contracts(predicted_topic, sector_filter, probability)

        # Step 4: Signal evaluation — P(post) >= 50% + contracts found = HOT
        signal_eval = evaluate_signal(probability, contracts)

        if signal_eval["has_alpha"]:
            signal = "HOT"
            contracts = signal_eval["contracts_with_alpha"]
        elif probability >= HOT_THRESHOLD:
            signal = "HOT"  # high probability but no contracts found
        else:
            signal = "WARM"  # above base threshold but below HOT

        # Step 5: market impact
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
        "arbitrage": signal_eval,
        "regime": regime_label,
    }


def _keyword_topic_fallback(tweets: list[str]) -> str:
    """Simple keyword matching when Gemini is unavailable."""
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
