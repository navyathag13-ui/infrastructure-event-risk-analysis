from __future__ import annotations

from pathlib import Path
import random
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATA_DIR / "events.csv"

random.seed(42)


ASSET_GROUPS = ["Transformer", "Feeder", "Line", "Switch", "Breaker"]
REGIONS = ["North", "South", "East", "West", "Central"]
EVENT_CATEGORIES = ["Weather", "Equipment Failure", "Vegetation", "Overload", "Inspection Follow-up"]

ASSETS = []
for i in range(1, 41):
    ASSETS.append(
        {
            "asset_id": f"A{i:03d}",
            "asset_group": random.choice(ASSET_GROUPS),
            "region": random.choice(REGIONS),
        }
    )


def random_date() -> str:
    year = 2024
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def generate_rows(n: int = 250) -> list[dict]:
    rows = []

    for event_id in range(1, n + 1):
        asset = random.choice(ASSETS)
        category = random.choice(EVENT_CATEGORIES)

        base_duration = {
            "Weather": random.randint(60, 280),
            "Equipment Failure": random.randint(90, 320),
            "Vegetation": random.randint(30, 140),
            "Overload": random.randint(40, 180),
            "Inspection Follow-up": random.randint(15, 90),
        }[category]

        severity = {
            "Weather": random.randint(2, 5),
            "Equipment Failure": random.randint(3, 5),
            "Vegetation": random.randint(1, 4),
            "Overload": random.randint(2, 4),
            "Inspection Follow-up": random.randint(1, 3),
        }[category]

        customers_affected = {
            "Transformer": random.randint(50, 250),
            "Feeder": random.randint(120, 500),
            "Line": random.randint(80, 420),
            "Switch": random.randint(20, 150),
            "Breaker": random.randint(30, 180),
        }[asset["asset_group"]]

        rows.append(
            {
                "event_id": event_id,
                "event_date": random_date(),
                "asset_id": asset["asset_id"],
                "asset_group": asset["asset_group"],
                "region": asset["region"],
                "event_category": category,
                "duration_minutes": base_duration,
                "severity": severity,
                "customers_affected": customers_affected,
            }
        )

    return rows


def main() -> None:
    rows = generate_rows(250)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()