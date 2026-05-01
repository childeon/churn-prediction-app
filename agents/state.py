"""LangGraph pipeline state definition."""

from __future__ import annotations
from typing import Any, TypedDict

import pandas as pd


class PipelineState(TypedDict, total=False):
    # ── Input ──
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    project_overview: str           # user-provided project/dataset context (required in UI)

    # ── Schema (set in app.py before the graph runs) ──
    # Decouples the pipeline from any single dataset's column names.
    #   target_col      – the binary churn label column (e.g. "churn", "Exited", "Attrition")
    #   id_cols         – columns to drop before modelling (IDs, free text, geography)
    #   tenure_col      – numeric column representing customer age in months (or None)
    #   positive_label  – the raw value in target_col that means "churned" (e.g. "Yes", 1, True)
    schema: dict

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

    # ── Business Aggregates (Node: business_aggregates) ──
    # BA-friendly numbers computed from model predictions + schema-level config.
    # Used both by the insight agent (as context) and by the Executive Summary UI.
    business_aggregates: dict
    # {
    #   "at_risk_count": int,              customers predicted to churn at optimal threshold
    #   "at_risk_pct": float,              % of customer base
    #   "revenue_at_stake": float,         at_risk_count × customer_value
    #   "projected_profit": float,         expected profit from retention campaign at optimal threshold
    #   "risk_bucket_counts": {"high": int, "medium": int, "low": int},
    #   "top_at_risk_customers": list[dict],  top-N individual customers with highest predicted prob
    # }

    # ── Segments (Node: segment_discovery) ──
    # Interpretable customer clusters mined from a decision-tree surrogate of the
    # best model, then named/narrated by the LLM.
    segments: list[dict]
    # [{
    #   "name": str,                       LLM-generated human name
    #   "rule": str,                       deterministic rule expression ("tenure < 2 AND geography == 'Germany'")
    #   "size": int, "size_pct": float,
    #   "churn_rate": float,               observed rate in this leaf
    #   "avg_churn_prob": float,           predicted rate
    #   "narrative": str,                  1-2 sentence BA-friendly explanation
    #   "characteristics": list[str],      plain-English feature highlights
    #   "recommended_actions": list[str],  2-3 suggested playbook items
    # }, ...]

    # ── Insight Agent outputs ──
    # auto_insights: legacy markdown output (kept for the existing Insights tab and chat context)
    # structured_insights: new JSON-structured output that drives the Executive Summary / Why tabs
    auto_insights: str
    structured_insights: dict
    # {
    #   "executive_summary": str,            2-3 sentence paragraph, business language only
    #   "kpis": list[{"label","value","unit","context"}],
    #   "top_actions": list[{"title","description","expected_impact","effort","timeline"}],
    #   "driver_narratives": list[{"driver","narrative","suggested_action"}],
    # }
    chat_history: list[dict]           # [{role, content}, ...]

    # ── Class Imbalance Agent outputs ──
    imbalance_config: dict             # minority_ratio, is_imbalanced, class weights, primary_metric

    # ── Missing Values Agent outputs ──
    missing_profile: list[dict]        # one dict per column with missing values
    missing_strategies: list[dict]     # LLM-proposed strategy per column (surfaced in UI for review)

    # ── Dataset metadata ──
    dataset_summary: dict              # shape, churn rate, etc.

    # ── Customer Simulation Profiles (Node: run_model_pipeline) ──
    # Test-set rows packaged for customer-level what-if simulation.
    # Each dict has: _sim_id, _actual_label, _churn_prob, + all X_test feature columns.
    simulation_profiles: list[dict]
