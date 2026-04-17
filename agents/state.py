"""LangGraph pipeline state definition."""

from __future__ import annotations
from typing import Any, TypedDict

import pandas as pd


class PipelineState(TypedDict, total=False):
    # ── Input ──
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame

    # ── Horizon Definition outputs ──
    selected_horizon: int           # e.g. 30, 60, or 90
    df_master: pd.DataFrame         # df with all horizon labels (churn_30d/60d/90d) before leakage drop

    # ── Progress ──
    current_step: str
    progress_messages: list[str]

    # ── Model Selection Agent outputs ──
    model_comparison: list[dict]       # one dict per model with metrics
    best_model_name: str
    best_model_metrics: dict
    best_pipeline: Any                 # fitted sklearn Pipeline
    predictions: dict                  # {"y_test": list, "y_prob": list} for best model on test set
    shap_values: Any                   # numpy array
    feature_names: list[str]
    feature_importances: list[dict]    # [{feature, importance}, ...]

    # ── Insight Agent outputs ──
    auto_insights: str                 # LLM-generated markdown
    chat_history: list[dict]           # [{role, content}, ...]

    # ── Class Imbalance Agent outputs ──
    imbalance_config: dict             # minority_ratio, is_imbalanced, class weights, primary_metric

    # ── Missing Values Agent outputs ──
    missing_profile: list[dict]        # one dict per column with missing values
    missing_strategies: list[dict]     # LLM-proposed strategy per column (surfaced in UI for review)

    # ── Dataset metadata ──
    dataset_summary: dict              # shape, churn rate, etc.
