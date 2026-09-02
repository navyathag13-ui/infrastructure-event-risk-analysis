from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "events.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load event data from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "event_id",
        "event_date",
        "asset_id",
        "asset_group",
        "region",
        "event_category",
        "duration_minutes",
        "severity",
        "customers_affected",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw event data."""
    cleaned = df.copy()

    cleaned["event_date"] = pd.to_datetime(cleaned["event_date"], errors="coerce")
    cleaned["duration_minutes"] = pd.to_numeric(cleaned["duration_minutes"], errors="coerce")
    cleaned["severity"] = pd.to_numeric(cleaned["severity"], errors="coerce")
    cleaned["customers_affected"] = pd.to_numeric(cleaned["customers_affected"], errors="coerce")

    cleaned = cleaned.dropna(
        subset=[
            "event_date",
            "asset_id",
            "asset_group",
            "region",
            "event_category",
            "duration_minutes",
            "severity",
            "customers_affected",
        ]
    )

    cleaned = cleaned[cleaned["duration_minutes"] >= 0]
    cleaned = cleaned[cleaned["severity"] >= 0]
    cleaned = cleaned[cleaned["customers_affected"] >= 0]

    cleaned["year"] = cleaned["event_date"].dt.year
    cleaned["month"] = cleaned["event_date"].dt.month
    cleaned["quarter"] = cleaned["event_date"].dt.quarter
    cleaned["day_of_week"] = cleaned["event_date"].dt.day_name()

    cleaned["season"] = cleaned["month"].map(
        {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall",
        }
    )

    return cleaned.reset_index(drop=True)


def create_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly event trends for reporting."""
    monthly = (
        df.groupby(["year", "month", "region", "asset_group"], as_index=False)
        .agg(
            total_events=("event_id", "count"),
            total_duration_minutes=("duration_minutes", "sum"),
            avg_duration_minutes=("duration_minutes", "mean"),
            avg_severity=("severity", "mean"),
            total_customers_affected=("customers_affected", "sum"),
        )
        .sort_values(["year", "month", "region", "asset_group"])
    )
    return monthly


def create_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create asset-level features for prioritization."""
    asset = (
        df.groupby(["asset_id", "asset_group", "region"], as_index=False)
        .agg(
            event_count=("event_id", "count"),
            avg_duration_minutes=("duration_minutes", "mean"),
            max_duration_minutes=("duration_minutes", "max"),
            avg_severity=("severity", "mean"),
            max_severity=("severity", "max"),
            total_customers_affected=("customers_affected", "sum"),
            unique_event_categories=("event_category", "nunique"),
            active_months=("month", "nunique"),
        )
    )

    asset["events_per_active_month"] = asset["event_count"] / asset["active_months"].replace(0, 1)

    asset["rule_based_risk_score"] = (
        0.30 * asset["event_count"]
        + 0.20 * asset["avg_duration_minutes"]
        + 8.00 * asset["avg_severity"]
        + 0.01 * asset["total_customers_affected"]
        + 2.00 * asset["unique_event_categories"]
    )

    return asset


def train_priority_model(asset_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Train a simple model to learn a smoothed priority score.
    For this MVP, the target is the rule-based score.
    Later you can replace it with a real future-risk label.
    """
    model_df = asset_df.copy()

    feature_cols = [
        "event_count",
        "avg_duration_minutes",
        "max_duration_minutes",
        "avg_severity",
        "max_severity",
        "total_customers_affected",
        "unique_event_categories",
        "active_months",
        "events_per_active_month",
    ]

    X = model_df[feature_cols]
    y = model_df["rule_based_risk_score"]

    if len(model_df) < 4:
        model_df["model_risk_score"] = model_df["rule_based_risk_score"]
        model_df = model_df.sort_values("model_risk_score", ascending=False).reset_index(drop=True)
        model_df["priority_rank"] = np.arange(1, len(model_df) + 1)
        model_df["priority_band"] = "High"

        metrics = {
            "mae": 0.0,
            "r2": 1.0,
        }
        return model_df, metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }

    model_df["model_risk_score"] = model.predict(X)

    num_unique_scores = model_df["model_risk_score"].nunique()
    if num_unique_scores >= 4:
        model_df["priority_band"] = pd.qcut(
            model_df["model_risk_score"],
            q=4,
            labels=["Low", "Moderate", "High", "Critical"],
            duplicates="drop",
        )
    else:
        model_df["priority_band"] = "Review"

    model_df = model_df.sort_values("model_risk_score", ascending=False).reset_index(drop=True)
    model_df["priority_rank"] = np.arange(1, len(model_df) + 1)

    return model_df, metrics


def save_outputs(
    cleaned_df: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    asset_scores: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    """Save all outputs to CSV/text files."""
    cleaned_df.to_csv(OUTPUT_DIR / "cleaned_events.csv", index=False)
    monthly_summary.to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)
    asset_scores.to_csv(OUTPUT_DIR / "asset_risk_scores.csv", index=False)

    with open(OUTPUT_DIR / "model_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Infrastructure Event Pattern Analysis & Risk Prioritization\n")
        f.write("=" * 60 + "\n")
        f.write(f"Rows in cleaned dataset: {len(cleaned_df)}\n")
        f.write(f"Assets scored: {len(asset_scores)}\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"R^2: {metrics['r2']:.4f}\n")


def print_summary(
    cleaned_df: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    asset_scores: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    """Print console output so you know the project ran."""
    print("\nProject run complete.")
    print("-" * 60)
    print(f"Total cleaned events: {len(cleaned_df)}")
    print(f"Assets scored: {len(asset_scores)}")
    print(f"Monthly summary rows: {len(monthly_summary)}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R^2: {metrics['r2']:.4f}")

    print("\nTop priority assets:")
    cols = [
        "priority_rank",
        "asset_id",
        "asset_group",
        "region",
        "event_count",
        "avg_duration_minutes",
        "avg_severity",
        "total_customers_affected",
        "model_risk_score",
        "priority_band",
    ]
    print(asset_scores[cols].head(10).to_string(index=False))


def main() -> None:
    print("Starting Infrastructure Event Pattern Analysis project...")

    raw_df = load_data(DATA_PATH)
    print(f"Loaded raw data: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    cleaned_df = clean_data(raw_df)
    print(f"Cleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")

    monthly_summary = create_monthly_summary(cleaned_df)
    asset_features = create_asset_features(cleaned_df)
    asset_scores, metrics = train_priority_model(asset_features)

    save_outputs(cleaned_df, monthly_summary, asset_scores, metrics)
    print_summary(cleaned_df, monthly_summary, asset_scores, metrics)


if __name__ == "__main__":
    main()