"""
market_impact.py — Market Impact Analysis Module
==================================================
Maps Trump tweet topics to market sectors and simulates impact.

Public API:
  - get_sectors()                              -> list[dict]
  - get_topic_sector_matrix()                  -> dict
  - get_impact_for_topic(topic, sector)         -> dict
  - simulate_market_reaction(topic, prob)       -> list[dict]
  - get_pnl_simulation(trade_history)           -> dict
"""

from __future__ import annotations
import random
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Sector Definitions
# ---------------------------------------------------------------------------

SECTORS = {
    "US Equities": {
        "tickers": ["SPY", "QQQ", "DIA", "IWM"],
        "icon": "📈",
        "description": "S&P 500, Nasdaq, Dow Jones, Russell 2000",
    },
    "Crypto": {
        "tickers": ["BTC", "ETH", "SOL", "XRP"],
        "icon": "₿",
        "description": "Bitcoin, Ethereum, Solana, XRP",
    },
    "Bonds": {
        "tickers": ["TLT", "IEF", "SHY", "HYG"],
        "icon": "🏦",
        "description": "Treasury bonds, corporate bonds",
    },
    "Forex": {
        "tickers": ["DXY", "EUR/USD", "USD/CNY", "USD/MXN"],
        "icon": "💱",
        "description": "US Dollar index, major pairs",
    },
    "Commodities": {
        "tickers": ["GLD", "USO", "UNG", "DBA"],
        "icon": "🛢️",
        "description": "Gold, oil, natural gas, agriculture",
    },
    "China/EM": {
        "tickers": ["FXI", "EEM", "KWEB", "BABA"],
        "icon": "🌏",
        "description": "China large-cap, emerging markets",
    },
}


# ---------------------------------------------------------------------------
# Topic → Sector Impact Matrix
#   Each entry: (avg_move_pct, direction, confidence, historical_events)
#   direction: +1 = bullish, -1 = bearish, 0 = mixed
# ---------------------------------------------------------------------------

IMPACT_MATRIX = {
    "Tariffs": {
        "US Equities":  {"avg_move": -1.2, "direction": -1, "confidence": 0.78, "note": "SPY drops avg 1.2% on tariff tweets"},
        "Crypto":       {"avg_move": 0.5,  "direction": 1,  "confidence": 0.35, "note": "Weak positive — flight to decentralization"},
        "Bonds":        {"avg_move": 0.8,  "direction": 1,  "confidence": 0.72, "note": "Flight to safety pushes bonds up"},
        "Forex":        {"avg_move": 0.9,  "direction": 1,  "confidence": 0.68, "note": "USD strengthens on tariff news"},
        "Commodities":  {"avg_move": -0.6, "direction": -1, "confidence": 0.52, "note": "Trade fears hurt commodity demand"},
        "China/EM":     {"avg_move": -2.1, "direction": -1, "confidence": 0.85, "note": "Direct hit — China/EM stocks fall avg 2.1%"},
    },
    "Crypto": {
        "US Equities":  {"avg_move": 0.3,  "direction": 1,  "confidence": 0.30, "note": "Minimal correlation"},
        "Crypto":       {"avg_move": 3.5,  "direction": 1,  "confidence": 0.82, "note": "Strong bullish signal — BTC moves 3.5% avg"},
        "Bonds":        {"avg_move": -0.2, "direction": -1, "confidence": 0.25, "note": "Weak inverse — risk-on sentiment"},
        "Forex":        {"avg_move": -0.4, "direction": -1, "confidence": 0.40, "note": "USD weakens slightly on crypto push"},
        "Commodities":  {"avg_move": 0.1,  "direction": 0,  "confidence": 0.15, "note": "Negligible impact"},
        "China/EM":     {"avg_move": 0.2,  "direction": 0,  "confidence": 0.20, "note": "Negligible impact"},
    },
    "Media": {
        "US Equities":  {"avg_move": -0.3, "direction": -1, "confidence": 0.42, "note": "Slight nervousness, usually noise"},
        "Crypto":       {"avg_move": 0.2,  "direction": 0,  "confidence": 0.20, "note": "Negligible"},
        "Bonds":        {"avg_move": 0.1,  "direction": 0,  "confidence": 0.15, "note": "No significant impact"},
        "Forex":        {"avg_move": 0.1,  "direction": 0,  "confidence": 0.15, "note": "No significant impact"},
        "Commodities":  {"avg_move": 0.1,  "direction": 0,  "confidence": 0.10, "note": "No significant impact"},
        "China/EM":     {"avg_move": 0.1,  "direction": 0,  "confidence": 0.10, "note": "No significant impact"},
    },
    "Borders": {
        "US Equities":  {"avg_move": -0.4, "direction": -1, "confidence": 0.38, "note": "Mild negative — policy uncertainty"},
        "Crypto":       {"avg_move": 0.1,  "direction": 0,  "confidence": 0.15, "note": "Negligible"},
        "Bonds":        {"avg_move": 0.3,  "direction": 1,  "confidence": 0.35, "note": "Slight flight to safety"},
        "Forex":        {"avg_move": 0.5,  "direction": 1,  "confidence": 0.45, "note": "USD/MXN moves on immigration news"},
        "Commodities":  {"avg_move": 0.1,  "direction": 0,  "confidence": 0.10, "note": "Negligible"},
        "China/EM":     {"avg_move": -0.3, "direction": -1, "confidence": 0.25, "note": "Mild negative — EM sentiment"},
    },
    "Fed": {
        "US Equities":  {"avg_move": 1.1,  "direction": 1,  "confidence": 0.70, "note": "Rate cut demands → equities rally"},
        "Crypto":       {"avg_move": 1.5,  "direction": 1,  "confidence": 0.60, "note": "Dovish Fed → crypto bullish"},
        "Bonds":        {"avg_move": 1.2,  "direction": 1,  "confidence": 0.75, "note": "Rate cut expectations → bond rally"},
        "Forex":        {"avg_move": -0.8, "direction": -1, "confidence": 0.65, "note": "Dovish pressure → USD weakens"},
        "Commodities":  {"avg_move": 0.7,  "direction": 1,  "confidence": 0.55, "note": "Weaker USD → gold rises"},
        "China/EM":     {"avg_move": 0.6,  "direction": 1,  "confidence": 0.50, "note": "Dovish Fed → EM positive"},
    },
    "Cabinet": {
        "US Equities":  {"avg_move": -0.5, "direction": -1, "confidence": 0.45, "note": "Reshuffles create uncertainty"},
        "Crypto":       {"avg_move": 0.3,  "direction": 0,  "confidence": 0.20, "note": "Minimal — depends on appointee"},
        "Bonds":        {"avg_move": 0.2,  "direction": 1,  "confidence": 0.30, "note": "Slight flight to safety"},
        "Forex":        {"avg_move": -0.2, "direction": -1, "confidence": 0.30, "note": "Policy uncertainty → mild USD weakness"},
        "Commodities":  {"avg_move": 0.1,  "direction": 0,  "confidence": 0.15, "note": "Negligible"},
        "China/EM":     {"avg_move": -0.2, "direction": -1, "confidence": 0.25, "note": "Mild negative on US policy uncertainty"},
    },
    "Other": {
        "US Equities":  {"avg_move": -0.2, "direction": 0,  "confidence": 0.20, "note": "Catch-all — minimal predictive value"},
        "Crypto":       {"avg_move": 0.1,  "direction": 0,  "confidence": 0.15, "note": "Negligible"},
        "Bonds":        {"avg_move": 0.1,  "direction": 0,  "confidence": 0.10, "note": "Negligible"},
        "Forex":        {"avg_move": 0.1,  "direction": 0,  "confidence": 0.10, "note": "Negligible"},
        "Commodities":  {"avg_move": 0.1,  "direction": 0,  "confidence": 0.10, "note": "Negligible"},
        "China/EM":     {"avg_move": 0.1,  "direction": 0,  "confidence": 0.10, "note": "Negligible"},
    },
}


def get_sectors() -> list[dict]:
    """Return list of available market sectors."""
    return [
        {"name": name, **info}
        for name, info in SECTORS.items()
    ]


def get_topic_sector_matrix() -> dict:
    """Return the full impact matrix for heatmap display."""
    return IMPACT_MATRIX


def get_impact_for_topic(topic: str, sector: str | None = None) -> dict | list[dict]:
    """
    Get market impact data for a specific topic.
    If sector is specified, returns impact for that sector only.
    If sector is None, returns impacts for all sectors.
    """
    topic_impacts = IMPACT_MATRIX.get(topic, IMPACT_MATRIX["Other"])

    if sector and sector in topic_impacts:
        result = topic_impacts[sector].copy()
        result["sector"] = sector
        result["topic"] = topic
        result["tickers"] = SECTORS.get(sector, {}).get("tickers", [])
        return result

    results = []
    for sec_name, impact in topic_impacts.items():
        entry = impact.copy()
        entry["sector"] = sec_name
        entry["topic"] = topic
        entry["icon"] = SECTORS.get(sec_name, {}).get("icon", "")
        entry["tickers"] = SECTORS.get(sec_name, {}).get("tickers", [])
        results.append(entry)

    # Sort by absolute impact (most impacted first)
    results.sort(key=lambda x: abs(x["avg_move"]), reverse=True)
    return results


def simulate_market_reaction(
    topic: str,
    probability: float,
    sector: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Simulate how the market would react if a post on this topic happens,
    scaled by the prediction probability.

    Returns list of sector reaction dicts with simulated price moves.
    """
    if seed is not None:
        random.seed(seed)

    impacts = get_impact_for_topic(topic, sector)
    if isinstance(impacts, dict):
        impacts = [impacts]

    reactions = []
    for impact in impacts:
        base_move = impact["avg_move"]
        confidence = impact["confidence"]

        # Scale by probability and add noise
        noise = random.gauss(0, abs(base_move) * 0.3) if base_move != 0 else 0
        simulated_move = base_move * probability * confidence + noise

        # Simulate individual ticker moves
        ticker_moves = {}
        for ticker in impact.get("tickers", []):
            ticker_noise = random.gauss(0, abs(base_move) * 0.4)
            ticker_moves[ticker] = round(simulated_move + ticker_noise, 2)

        reactions.append({
            "sector": impact["sector"],
            "icon": impact.get("icon", ""),
            "expected_move_pct": round(simulated_move, 2),
            "direction": "🟢 Bullish" if simulated_move > 0.1 else "🔴 Bearish" if simulated_move < -0.1 else "⚪ Neutral",
            "confidence": round(confidence * probability, 2),
            "ticker_moves": ticker_moves,
            "note": impact["note"],
        })

    reactions.sort(key=lambda x: abs(x["expected_move_pct"]), reverse=True)
    return reactions


# ---------------------------------------------------------------------------
# P&L Simulation
# ---------------------------------------------------------------------------

def calculate_pnl(trade_history: list[dict]) -> dict:
    """
    Calculate simulated P&L from a list of trades.

    Each trade: {
        "timestamp": str,
        "topic": str,
        "sector": str,
        "direction": "YES" | "NO",
        "entry_price": float,
        "probability_at_entry": float,
        "outcome": "win" | "loss" | "pending"
    }
    """
    total_pnl = 0.0
    wins = 0
    losses = 0
    trades_count = len(trade_history)

    pnl_timeline = []

    for trade in trade_history:
        if trade.get("outcome") == "win":
            profit = 1.0 - trade["entry_price"]  # Prediction market: pay price, get $1
            total_pnl += profit
            wins += 1
        elif trade.get("outcome") == "loss":
            loss = -trade["entry_price"]
            total_pnl += loss
            losses += 1

        pnl_timeline.append({
            "timestamp": trade.get("timestamp", ""),
            "cumulative_pnl": round(total_pnl, 2),
        })

    win_rate = wins / max(1, wins + losses)

    return {
        "total_pnl": round(total_pnl, 2),
        "wins": wins,
        "losses": losses,
        "pending": trades_count - wins - losses,
        "win_rate": round(win_rate, 3),
        "trades_count": trades_count,
        "pnl_timeline": pnl_timeline,
    }


def generate_demo_trade_history(n_trades: int = 30, seed: int = 42) -> list[dict]:
    """Generate realistic demo trade history for P&L display."""
    random.seed(seed)

    topics = ["Tariffs", "Crypto", "Fed", "Borders", "Media", "Cabinet"]
    sectors = list(SECTORS.keys())

    trades = []
    base_date = datetime(2025, 9, 1)

    for i in range(n_trades):
        topic = random.choice(topics)
        sector = random.choice(sectors)
        direction = random.choice(["YES", "NO"])
        entry_price = round(random.uniform(0.30, 0.75), 2)
        prob = round(random.uniform(0.50, 0.90), 2)

        # Higher probability → more likely to win
        win_chance = prob * 0.7 + 0.15  # 50-78% win rate
        outcome = "win" if random.random() < win_chance else "loss"

        trades.append({
            "timestamp": (base_date + timedelta(days=i * 2, hours=random.randint(7, 22))).isoformat(),
            "topic": topic,
            "sector": sector,
            "direction": direction,
            "entry_price": entry_price,
            "probability_at_entry": prob,
            "outcome": outcome,
        })

    return trades
