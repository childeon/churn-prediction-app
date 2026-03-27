"""LangGraph pipeline state definition."""

from __future__ import annotations
from typing import Any, TypedDict

import pandas as pd


class PipelineState(TypedDict, total=False):
    # ── Input ──
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame

    # ── Progress ──
    current_step: str
    progress_messages: list[str]

    # ── Model Selection Agent outputs ──
    model_comparison: list[dict]       # one dict per model with metrics
    best_model_name: str
    best_model_metrics: dict
    best_pipeline: Any                 # fitted sklearn Pipeline
    shap_values: Any                   # numpy array
    feature_names: list[str]
    feature_importances: list[dict]    # [{feature, importance}, ...]

    # ── Insight Agent outputs ──
    auto_insights: str                 # LLM-generated markdown
    chat_history: list[dict]           # [{role, content}, ...]

    # ── Dataset metadata ──
    dataset_summary: dict              # shape, churn rate, etc.
