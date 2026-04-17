"""Horizon definition node — Node 2 in the offline pipeline.

Responsibilities:
- Encode the churn column to 0/1 if needed
- Generate synthetic time structure (start_date, churn_date)
- Build horizon labels: churn_30d, churn_60d, churn_90d
- Set the active churn target based on the selected horizon
- Drop leakage columns before passing data downstream
- Preserve df_master (with all horizon labels intact) for later reference
"""

from agents.state import PipelineState
from pipeline.config import DEFAULT_HORIZON, HORIZONS, SNAPSHOT_DATE
from utils.horizon_utils import build_horizon_labels, generate_synthetic_time

_LEAKAGE_COLS = [
    "start_date",
    "churn_date",
    "days_before_snapshot",
    "days_since_churn",
] + [f"churn_{h}d" for h in HORIZONS]


def horizon_definition_node(state: PipelineState) -> dict:
    df = state["raw_df"].copy()
    horizon = state.get("selected_horizon") or DEFAULT_HORIZON
    messages = list(state.get("progress_messages", []))

    # Encode churn label if stored as strings (e.g. "Yes" / "No")
    if df["churn"].dtype == object:
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
        messages.append("Encoded churn column: Yes→1, No→0")

    df = generate_synthetic_time(df, SNAPSHOT_DATE)
    messages.append(f"Generated synthetic time structure (snapshot: {SNAPSHOT_DATE.date()})")

    df = build_horizon_labels(df, SNAPSHOT_DATE, HORIZONS)
    label_cols = ", ".join(f"churn_{h}d" for h in HORIZONS)
    messages.append(f"Built horizon labels: {label_cols}")

    # Keep df_master before any leakage columns are dropped
    df_master = df.copy()

    # Overwrite churn with the selected horizon's label
    df["churn"] = df[f"churn_{horizon}d"]
    messages.append(f"Active churn target: churn_{horizon}d ({horizon}-day window)")

    # Drop columns that would leak future information into the model
    drop_cols = [c for c in _LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return {
        "raw_df": df,
        "df_master": df_master,
        "selected_horizon": horizon,
        "current_step": f"Horizon defined ({horizon}d)",
        "progress_messages": messages,
    }
