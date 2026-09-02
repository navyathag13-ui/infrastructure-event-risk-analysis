# Infrastructure Event Pattern Analysis & Risk Prioritization

A Python-based pipeline that analyzes infrastructure event records to identify operational patterns and rank assets by risk — enabling data-driven maintenance prioritization.

## Overview

This project processes raw infrastructure event data through a full ML pipeline:
1. **Cleans and validates** raw event records
2. **Aggregates** monthly trends by region and asset group
3. **Engineers features** at the asset level (frequency, severity, duration, customer impact)
4. **Trains a Random Forest model** to score and rank assets by risk
5. **Generates visualizations** for dashboard reporting

---

## Project Structure

```
infrastructure_event_risk/
├── main.py              # Core analysis pipeline
├── generate_data.py     # Synthetic data generator (250 sample events)
├── charts.py            # Visualization generator
├── requirements.txt     # Python dependencies
├── data/
│   └── events.csv       # Input event data
├── outputs/
│   ├── cleaned_events.csv
│   ├── monthly_summary.csv
│   ├── asset_risk_scores.csv
│   └── model_metrics.txt
└── charts/
    ├── monthly_total_events.png
    └── top_10_priority_assets.png
```

---

## Installation

```bash
git clone https://github.com/navyathag13-ui/infrastructure-event-risk-analysis.git
cd infrastructure-event-risk-analysis
pip install -r requirements.txt
```

---

## Usage

```bash
# Step 1: Generate sample data (skip if you have real data in data/events.csv)
python generate_data.py

# Step 2: Run the main analysis pipeline
python main.py

# Step 3: Generate charts
python charts.py
```

---

## Input Data

The input CSV (`data/events.csv`) expects the following fields:

| Field | Description |
|---|---|
| `event_date` | Date of the event |
| `asset_id` | Unique asset identifier |
| `asset_group` | Asset type (e.g., Transformer, Feeder, Line) |
| `region` | Geographic region |
| `event_category` | Type of event (e.g., Weather, Equipment Failure) |
| `duration_minutes` | Duration of the event |
| `severity` | Severity score (numeric) |
| `customers_affected` | Number of customers impacted |

---

## How It Works

### Data Cleaning
- Parses dates and validates numeric fields
- Removes rows with missing critical fields or invalid values (negative durations/severities)
- Enriches records with temporal features: year, month, quarter, season, day of week

### Risk Scoring
Each asset is scored using a weighted formula:

```
Risk Score = 0.30 × event_count
           + 0.20 × avg_duration
           + 8.00 × avg_severity
           + 0.01 × customers_affected
           + 2.00 × unique_event_categories
```

### Machine Learning Model
- **Algorithm**: Random Forest Regressor (200 estimators, max depth 6)
- **Split**: 75% train / 25% test
- **Output**: Continuous risk score per asset, bucketed into priority bands: `Low`, `Moderate`, `High`, `Critical`

---

## Outputs

| File | Description |
|---|---|
| `cleaned_events.csv` | Validated event records with temporal features |
| `monthly_summary.csv` | Aggregated monthly metrics by region and asset group |
| `asset_risk_scores.csv` | Per-asset risk scores, priority bands, and rankings |
| `model_metrics.txt` | MAE and R² model performance metrics |
| `charts/monthly_total_events.png` | Time series of event trends |
| `charts/top_10_priority_assets.png` | Bar chart of highest-risk assets |

**Sample model performance** (on synthetic data): MAE ≈ 3.0, R² ≈ 0.76

---

## Technologies

- **Python 3**
- **pandas** — data loading, cleaning, aggregation
- **NumPy** — numerical operations
- **scikit-learn** — Random Forest model, metrics
- **matplotlib** — chart generation
