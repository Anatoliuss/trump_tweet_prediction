"""
generate_sample_data.py
Generates 100 rows of realistic mock Truth Social post data for pipeline testing.
Output schema matches the PlanetScale migration target:
  (timestamp TEXT, text TEXT, predicted_topic TEXT, probability_score REAL,
   news_volume_4h INT, news_sentiment REAL, simulated_vix_4h_vol REAL,
   simulated_dxy_4h_vol REAL)
"""

import random
import csv
import math
from datetime import datetime, timedelta
import pytz

EST = pytz.timezone("America/New_York")

TEMPLATES = [
    "The fake news media is LYING about {topic}. Sad!",
    "Just spoke with {name}. Great meeting. {topic} will be HUGE.",
    "WITCH HUNT! The radical left wants to destroy {topic}. Not on my watch!",
    "The {topic} numbers are the best we've ever seen. Nobody does it better!",
    "MAKE AMERICA GREAT AGAIN! {topic} is our top priority.",
    "The {topic} deal is coming together. TREMENDOUS progress!",
    "Breaking: {topic} - the REAL story the mainstream media won't tell you.",
    "Thank you to the GREAT people of America who support us on {topic}!",
    "{topic} has never been stronger. The numbers speak for themselves!",
    "The Democrats are DESTROYING {topic}. We will fight back!",
]

TOPICS = {
    "Tariffs": ["tariffs", "trade deal", "China tariffs", "import taxes", "trade war"],
    "Crypto": ["Bitcoin", "crypto", "digital currency", "blockchain", "NFT"],
    "Media": ["fake news", "mainstream media", "CNN", "New York Times", "NBC"],
    "Borders": ["border security", "immigration", "the wall", "ICE", "deportations"],
    "Fed": ["the Federal Reserve", "interest rates", "Jerome Powell", "inflation", "the Fed"],
    "Cabinet": ["my cabinet", "Secretary", "the administration", "our team", "the White House"],
    "Other": ["the radical left", "MAGA", "America", "the deep state", "our great nation"],
}

# Topics that tend to spike VIX and news volume
HIGH_VOL_TOPICS = {"Tariffs", "Fed", "Borders"}

NAMES = ["Marco Rubio", "JD Vance", "Elon Musk", "the President of France", "a great patriot"]


def _generate_market_context(topic_key: str, hour: int, day_of_week: int, rng: random.Random) -> dict:
    """
    Generate correlated mock market/news context for a single post.

    Correlations:
    - High-volatility topics (Tariffs, Fed, Borders) -> higher news volume, higher VIX
    - Negative sentiment news -> higher VIX
    - Weekend -> lower news volume, calmer VIX
    - High news volume -> more negative sentiment (crisis drives coverage)
    """
    is_weekend = day_of_week >= 5
    is_business_hours = 9 <= hour <= 17

    # --- news_volume_4h: integer 0-50 ---
    # Base: 5-15, higher during business hours, lower on weekends
    base_vol = rng.randint(5, 15)
    if is_business_hours:
        base_vol += rng.randint(2, 8)
    if is_weekend:
        base_vol = max(1, base_vol - rng.randint(3, 7))
    if topic_key in HIGH_VOL_TOPICS:
        base_vol += rng.randint(5, 15)  # crisis topics spike news
    # Occasional breaking news spike
    if rng.random() < 0.10:
        base_vol += rng.randint(15, 30)
    news_volume_4h = min(50, max(0, base_vol))

    # --- news_sentiment: float -1.0 to 1.0 ---
    # High news volume tends negative (crisis), low volume tends neutral/positive
    sentiment_base = rng.gauss(0.1, 0.35)
    if news_volume_4h > 25:
        sentiment_base -= rng.uniform(0.2, 0.5)  # crisis -> negative
    if topic_key == "Crypto":
        sentiment_base += rng.uniform(0.0, 0.3)  # crypto hype -> positive
    if topic_key in ("Tariffs", "Borders"):
        sentiment_base -= rng.uniform(0.0, 0.25)  # trade war fear
    news_sentiment = round(max(-1.0, min(1.0, sentiment_base)), 3)

    # --- simulated_vix_4h_vol: float ~12-45 ---
    # VIX baseline ~18, spikes with high news volume and negative sentiment
    vix_base = rng.gauss(18.0, 3.0)
    # News volume drives VIX up
    vix_base += (news_volume_4h - 10) * 0.3
    # Negative sentiment drives VIX exponentially
    if news_sentiment < -0.3:
        vix_base += math.exp(abs(news_sentiment) * 1.5) - 1
    if topic_key in HIGH_VOL_TOPICS:
        vix_base += rng.uniform(2.0, 6.0)
    if is_weekend:
        vix_base -= rng.uniform(1.0, 3.0)  # calmer on weekends
    simulated_vix_4h_vol = round(max(10.0, min(50.0, vix_base)), 2)

    # --- simulated_dxy_4h_vol: float ~-1.5 to +1.5 pct change ---
    # DXY strengthens on tariff news, weakens on dovish Fed talk
    dxy_base = rng.gauss(0.0, 0.3)
    if topic_key == "Tariffs":
        dxy_base += rng.uniform(0.1, 0.5)  # tariffs strengthen USD
    elif topic_key == "Fed":
        dxy_base -= rng.uniform(0.1, 0.4)  # dovish pressure weakens USD
    # High VIX -> slight DXY strength (flight to safety)
    if simulated_vix_4h_vol > 25:
        dxy_base += rng.uniform(0.05, 0.2)
    simulated_dxy_4h_vol = round(max(-1.5, min(1.5, dxy_base)), 3)

    return {
        "news_volume_4h": news_volume_4h,
        "news_sentiment": news_sentiment,
        "simulated_vix_4h_vol": simulated_vix_4h_vol,
        "simulated_dxy_4h_vol": simulated_dxy_4h_vol,
    }


def generate_posts(n: int = 100, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    posts = []

    # Simulate ~6 months of posting history ending today
    end_dt = datetime(2025, 3, 1, tzinfo=EST)
    start_dt = end_dt - timedelta(days=180)

    current = start_dt
    while len(posts) < n:
        # He mostly posts between 7am-1am EST; peaks at morning and late night
        hour_weights = [0] * 24
        for h in range(7, 12):   hour_weights[h] = 4   # morning burst
        for h in range(12, 17):  hour_weights[h] = 2   # afternoon
        for h in range(17, 24):  hour_weights[h] = 5   # evening/night peak
        hour_weights[0] = 2; hour_weights[1] = 1        # past midnight

        hour = rng.choices(range(24), weights=hour_weights)[0]
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)

        post_dt = current.replace(hour=hour, minute=minute, second=second)

        topic_key = rng.choice(list(TOPICS.keys()))
        topic_phrase = rng.choice(TOPICS[topic_key])
        template = rng.choice(TEMPLATES)
        text = template.format(topic=topic_phrase, name=rng.choice(NAMES))

        # Generate correlated market/news context
        day_of_week = post_dt.weekday()
        market_ctx = _generate_market_context(topic_key, hour, day_of_week, rng)

        posts.append({
            "timestamp": post_dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "text": text,
            "predicted_topic": "",        # filled by pipeline
            "probability_score": None,    # filled by pipeline
            **market_ctx,
        })

        # Advance by a random gap: burst-posting (minutes) or daily gap (hours)
        gap_minutes = rng.choices(
            [rng.randint(5, 45), rng.randint(60, 480), rng.randint(480, 1440)],
            weights=[40, 35, 25]
        )[0]
        current += timedelta(minutes=gap_minutes)

        if len(posts) >= n:
            break

    posts.sort(key=lambda x: x["timestamp"])
    return posts[:n]


def save_csv(posts: list[dict], path: str = "data/sample_posts.csv") -> None:
    fieldnames = [
        "timestamp", "text", "predicted_topic", "probability_score",
        "news_volume_4h", "news_sentiment", "simulated_vix_4h_vol", "simulated_dxy_4h_vol",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(posts)
    print(f"[generate_sample_data] Wrote {len(posts)} rows -> {path}")


if __name__ == "__main__":
    posts = generate_posts(100)
    save_csv(posts)
