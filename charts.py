from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = BASE_DIR / "charts"

CHART_DIR.mkdir(exist_ok=True)


def plot_monthly_events() -> None:
    monthly_path = OUTPUT_DIR / "monthly_summary.csv"
    if not monthly_path.exists():
        raise FileNotFoundError(f"Missing file: {monthly_path}")

    df = pd.read_csv(monthly_path)

    grouped = (
        df.groupby(["year", "month"], as_index=False)["total_events"]
        .sum()
        .sort_values(["year", "month"])
    )

    grouped["label"] = grouped["year"].astype(str) + "-" + grouped["month"].astype(str).str.zfill(2)

    plt.figure(figsize=(10, 5))
    plt.plot(grouped["label"], grouped["total_events"], marker="o")
    plt.title("Monthly Total Events")
    plt.xlabel("Month")
    plt.ylabel("Total Events")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "monthly_total_events.png", dpi=300)
    plt.close()


def plot_top_risk_assets() -> None:
    risk_path = OUTPUT_DIR / "asset_risk_scores.csv"
    if not risk_path.exists():
        raise FileNotFoundError(f"Missing file: {risk_path}")

    df = pd.read_csv(risk_path).head(10).copy()
    df = df.sort_values("model_risk_score", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["asset_id"], df["model_risk_score"])
    plt.title("Top 10 Priority Assets by Risk Score")
    plt.xlabel("Model Risk Score")
    plt.ylabel("Asset ID")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "top_10_priority_assets.png", dpi=300)
    plt.close()


def main() -> None:
    plot_monthly_events()
    plot_top_risk_assets()
    print("Charts saved in charts/")


if __name__ == "__main__":
    main()