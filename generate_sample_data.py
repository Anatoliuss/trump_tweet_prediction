"""
generate_sample_data.py
Generates 100 rows of realistic mock Truth Social post data for pipeline testing.
Output schema matches the PlanetScale migration target:
  (timestamp TEXT, text TEXT, predicted_topic TEXT, probability_score REAL)
"""

import random
import csv
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

NAMES = ["Marco Rubio", "JD Vance", "Elon Musk", "the President of France", "a great patriot"]


def generate_posts(n: int = 100, seed: int = 42) -> list[dict]:
    random.seed(seed)
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

        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        post_dt = current.replace(hour=hour, minute=minute, second=second)

        topic_key = random.choice(list(TOPICS.keys()))
        topic_phrase = random.choice(TOPICS[topic_key])
        template = random.choice(TEMPLATES)
        text = template.format(topic=topic_phrase, name=random.choice(NAMES))

        posts.append({
            "timestamp": post_dt.isoformat(),
            "text": text,
            "predicted_topic": "",        # filled by pipeline
            "probability_score": None,    # filled by pipeline
        })

        # Advance by a random gap: burst-posting (minutes) or daily gap (hours)
        gap_minutes = random.choices(
            [random.randint(5, 45), random.randint(60, 480), random.randint(480, 1440)],
            weights=[40, 35, 25]
        )[0]
        current += timedelta(minutes=gap_minutes)

        if len(posts) >= n:
            break

    posts.sort(key=lambda x: x["timestamp"])
    return posts[:n]


def save_csv(posts: list[dict], path: str = "data/sample_posts.csv") -> None:
    fieldnames = ["timestamp", "text", "predicted_topic", "probability_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(posts)
    print(f"[generate_sample_data] Wrote {len(posts)} rows -> {path}")


if __name__ == "__main__":
    posts = generate_posts(100)
    save_csv(posts)
