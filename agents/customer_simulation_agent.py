"""Customer-level sensitivity and counterfactual simulator.

All functions are deterministic. No LLM calls. No retraining.

Public API:
  predict_profile_risk(pipeline, profile, feature_columns) -> float
  simulate_feature_sensitivity(pipeline, baseline_profile, feature_name,
                               candidate_values, feature_columns) -> dict
  rank_feature_sensitivity(pipeline, baseline_profile, feature_metadata,
                           feature_columns, max_features=8) -> list[dict]
  generate_counterfactuals(pipeline, baseline_profile, feature_metadata,
                           feature_columns, max_results=10) -> list[dict]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Fields injected alongside feature values in simulation_profiles — never sent to the model.
_META_FIELDS = {"_sim_id", "_actual_label", "_churn_prob"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_profile_risk(
    pipeline: Any,
    profile: dict,
    feature_columns: list[str],
) -> float:
    """Score a single customer profile dict. Returns churn probability in [0, 1]."""
    row = {col: profile.get(col) for col in feature_columns}
    df = pd.DataFrame([row])[feature_columns]
    try:
        return float(pipeline.predict_proba(df)[0, 1])
    except Exception:
        return float("nan")


def simulate_feature_sensitivity(
    pipeline: Any,
    baseline_profile: dict,
    feature_name: str,
    candidate_values: list,
    feature_columns: list[str],
) -> dict:
    """Vary one feature across candidate_values; hold all others at baseline.

    Returns:
        feature, baseline_prob, candidate_values, scenario_probs, deltas
    """
    baseline_prob = predict_profile_risk(pipeline, baseline_profile, feature_columns)

    scenario_values: list = []
    scenario_probs: list[float] = []
    deltas: list[float] = []

    for val in candidate_values:
        modified = {**baseline_profile, feature_name: val}
        try:
            prob = predict_profile_risk(pipeline, modified, feature_columns)
            if not np.isnan(prob):
                scenario_values.append(val)
                scenario_probs.append(prob)
                deltas.append(prob - baseline_prob)
        except Exception:
            pass

    return {
        "feature": feature_name,
        "baseline_prob": baseline_prob,
        "candidate_values": scenario_values,
        "scenario_probs": scenario_probs,
        "deltas": deltas,
    }


def rank_feature_sensitivity(
    pipeline: Any,
    baseline_profile: dict,
    feature_metadata: dict,
    feature_columns: list[str],
    max_features: int = 8,
) -> list[dict]:
    """Rank features by maximum absolute probability swing across candidate values."""
    baseline_prob = predict_profile_risk(pipeline, baseline_profile, feature_columns)
    results: list[dict] = []

    for feat, meta in feature_metadata.items():
        if feat not in feature_columns:
            continue
        candidates = _candidate_values(feat, meta, baseline_profile)
        if not candidates:
            continue

        max_swing = 0.0
        for val in candidates:
            modified = {**baseline_profile, feat: val}
            try:
                prob = predict_profile_risk(pipeline, modified, feature_columns)
                if not np.isnan(prob):
                    max_swing = max(max_swing, abs(prob - baseline_prob))
            except Exception:
                pass

        results.append({
            "feature": feat,
            "max_swing": max_swing,
            "actionability": meta.get("actionability", "neutral"),
        })

    results.sort(key=lambda x: x["max_swing"], reverse=True)
    return results[:max_features]


def generate_counterfactuals(
    pipeline: Any,
    baseline_profile: dict,
    feature_metadata: dict,
    feature_columns: list[str],
    max_results: int = 10,
) -> list[dict]:
    """One-feature counterfactuals ranked by churn risk reduction (largest drop first).

    Only returns scenarios where the predicted probability decreases.
    """
    baseline_prob = predict_profile_risk(pipeline, baseline_profile, feature_columns)
    counterfactuals: list[dict] = []

    for feat, meta in feature_metadata.items():
        if feat not in feature_columns:
            continue
        current_val = baseline_profile.get(feat)
        candidates = _candidate_values(feat, meta, baseline_profile)

        for val in candidates:
            if _values_equal(val, current_val):
                continue
            modified = {**baseline_profile, feat: val}
            try:
                new_prob = predict_profile_risk(pipeline, modified, feature_columns)
                if np.isnan(new_prob):
                    continue
                delta = new_prob - baseline_prob
                if delta < 0:
                    counterfactuals.append({
                        "feature": feat,
                        "old_value": current_val,
                        "new_value": val,
                        "baseline_prob": baseline_prob,
                        "new_prob": new_prob,
                        "delta": delta,
                        "actionability": meta.get("actionability", "neutral"),
                    })
            except Exception:
                pass

    counterfactuals.sort(key=lambda x: x["delta"])
    return counterfactuals[:max_results]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _candidate_values(feat: str, meta: dict, baseline_profile: dict) -> list:
    """Return candidate values for a feature based on its metadata."""
    ftype = meta.get("type", "numeric")

    if ftype in ("categorical", "boolean"):
        return list(meta.get("values", []))

    # Numeric: use stored quartiles + ±10% / ±25% from current value
    candidates: list = list(meta.get("quartiles", []))
    current = baseline_profile.get(feat)
    if current is not None and not _is_nan(current):
        try:
            v = float(current)
            fmin = float(meta.get("min", v * 0.5))
            fmax = float(meta.get("max", v * 1.5))
            for pct in (0.75, 0.9, 1.1, 1.25):
                cand = max(fmin, min(fmax, round(v * pct, 4)))
                candidates.append(cand)
        except (TypeError, ValueError):
            pass

    # Deduplicate while preserving order
    seen: set = set()
    unique: list = []
    for c in candidates:
        try:
            key = round(float(c), 6)
        except (TypeError, ValueError):
            key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _is_nan(val) -> bool:
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _values_equal(a, b) -> bool:
    """Loose equality that handles NaN, mixed numeric/string, and type differences."""
    if _is_nan(a) and _is_nan(b):
        return True
    if _is_nan(a) or _is_nan(b):
        return False
    try:
        return str(a) == str(b)
    except Exception:
        return a == b
