"""Agent 1 — Model Selection: LangGraph node functions.

Nodes:
  clean_data_node        – drop NaN rows, drop irrelevant columns
  run_model_pipeline_node – execute d6tflow WorkflowMulti (5 models)
  compute_shap_node      – SHAP explainability on the best model
"""

import uuid
from pathlib import Path
import pandas as pd
import d6tflow

from pipeline.config import (
    COLUMNS_TO_DROP,
    TARGET_COLUMN,
    MODEL_TYPES,
    MODEL_DISPLAY_NAMES,
)
from pipeline.tasks import PrepareData, TrainModel, set_dataframe, set_imbalance_config
from utils.shap_utils import compute_shap_values
from agents.state import PipelineState


# ---------------------------------------------------------------------------
# Node 1: Clean data
# ---------------------------------------------------------------------------
def clean_data_node(state: PipelineState) -> dict:
    df = state["raw_df"].copy()
    rows_before = len(df)

    # Drop columns that aren't useful for modelling
    # NaN handling was already done by missing_values_node upstream
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    rows_after = len(df)

    # Dataset summary for the insight agent later
    churn_rate = df[TARGET_COLUMN].mean() * 100
    summary = {
        "rows": rows_after,
        "columns": len(df.columns),
        "rows_dropped": rows_before - rows_after,
        "churn_rate_pct": round(churn_rate, 2),
        "features": [c for c in df.columns if c != TARGET_COLUMN],
        "numeric_features": df.select_dtypes(exclude=["object"]).columns.tolist(),
        "categorical_features": df.select_dtypes(include=["object"]).columns.tolist(),
    }

    return {
        "clean_df": df,
        "dataset_summary": summary,
        "current_step": "Data cleaned",
        "progress_messages": [
            f"Dropped columns: {cols_to_drop}",
            f"Rows: {rows_before} -> {rows_after} ({rows_before - rows_after} dropped)",
            f"Churn rate: {churn_rate:.1f}%",
        ],
    }


# ---------------------------------------------------------------------------
# Node 2: Run d6tflow model pipeline
# ---------------------------------------------------------------------------
def run_model_pipeline_node(state: PipelineState) -> dict:
    df_clean = state["clean_df"]

    # Inject the dataframe and imbalance config into the pipeline module
    set_dataframe(df_clean)
    set_imbalance_config(state.get("imbalance_config", {}))

    # Use a unique directory to avoid conflicts between runs
    session_dir = Path(f"/tmp/d6tflow_capstone_{uuid.uuid4().hex[:8]}/")
    d6tflow.settings.dirpath = session_dir

    # Run tasks directly (bypasses luigi scheduler which needs signal
    # handlers that don't work in Streamlit's threaded environment)
    prepare_task = PrepareData()
    prepare_task.run()
    prepare_task.output()  # mark complete

    comparison = []
    for name in MODEL_TYPES:
        task = TrainModel(model_type=name)
        task.run()

        meta = task.outputLoadMeta()
        comparison.append({
            "model": name,
            "display_name": MODEL_DISPLAY_NAMES[name],
            "roc_auc": round(meta["roc_auc"], 4),
            "pr_auc": round(meta["pr_auc"], 4),
            "f1": round(meta["f1"], 4),
            "runtime_sec": round(meta["runtime_sec"], 2),
            "best_params": meta["best_params"],
            "optimal_threshold": meta["optimal_threshold"],
            "expected_profit": meta["expected_profit"],
            "threshold_curve": meta["threshold_curve"],
            "profit_curve": meta["profit_curve"],
        })

    # Sort by ROC-AUC descending
    comparison.sort(key=lambda x: x["roc_auc"], reverse=True)
    best = comparison[0]

    # Load the best fitted pipeline
    best_task = TrainModel(model_type=best["model"])
    best_pipeline = best_task.output().load()

    # Store predictions for the best model (used by chart and simulation agents)
    data = PrepareData().output().load()
    y_prob = best_pipeline.predict_proba(data["X_test"])[:, 1]
    predictions = {
        "y_test": data["y_test"].tolist(),
        "y_prob": y_prob.tolist(),
    }

    return {
        "model_comparison": comparison,
        "best_model_name": best["model"],
        "best_model_metrics": best,
        "best_pipeline": best_pipeline,
        "predictions": predictions,
        "current_step": "Models trained",
        "progress_messages": state.get("progress_messages", []) + [
            f"Trained {len(MODEL_TYPES)} models",
            f"Best model: {MODEL_DISPLAY_NAMES[best['model']]} (ROC-AUC: {best['roc_auc']})",
        ],
    }


# ---------------------------------------------------------------------------
# Node 3: Compute SHAP values
# ---------------------------------------------------------------------------
def compute_shap_node(state: PipelineState) -> dict:
    best_pipeline = state["best_pipeline"]
    best_model_name = state["best_model_name"]

    # Load test data from d6tflow
    data = PrepareData().output().load()
    X_test = data["X_test"]

    shap_vals, feature_names, feature_importances = compute_shap_values(
        best_pipeline, X_test, best_model_name
    )

    return {
        "shap_values": shap_vals,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "current_step": "SHAP computed",
        "progress_messages": state.get("progress_messages", []) + [
            f"Computed SHAP values for {best_model_name}",
            f"Top feature: {feature_importances[0]['feature']}",
        ],
    }
