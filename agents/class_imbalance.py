"""Node 4 — Class Imbalance Agent: detect target imbalance and configure mitigation."""

from agents.state import PipelineState


def class_imbalance_node(state: PipelineState) -> dict:
    df = state["raw_df"]
    target = df["churn"]

    n_total = len(target)
    n_minority = int(target.sum())
    n_majority = n_total - n_minority
    minority_ratio = n_minority / n_total if n_total > 0 else 0.5

    is_imbalanced = minority_ratio < 0.20

    # XGBoost uses scale_pos_weight = majority_count / minority_count
    scale_pos_weight = round(n_majority / n_minority, 2) if n_minority > 0 else 1.0

    imbalance_config = {
        "minority_ratio": round(float(minority_ratio), 4),
        "minority_count": n_minority,
        "majority_count": n_majority,
        "is_imbalanced": is_imbalanced,
        # Use PR-AUC as primary CV metric when data is imbalanced (ROC-AUC is over-optimistic)
        "primary_metric": "average_precision" if is_imbalanced else "roc_auc",
        "logreg_class_weight": "balanced" if is_imbalanced else None,
        "rf_class_weight": "balanced" if is_imbalanced else None,
        "lgbm_class_weight": "balanced" if is_imbalanced else None,
        "xgb_scale_pos_weight": scale_pos_weight if is_imbalanced else 1.0,
    }

    status = "imbalanced" if is_imbalanced else "balanced"
    msg = (
        f"Class ratio: {minority_ratio:.1%} minority "
        f"({n_minority:,} churned / {n_total:,} total) — {status}"
    )
    if is_imbalanced:
        msg += (
            f". Mitigation: class_weight=balanced, "
            f"xgb_scale_pos_weight={scale_pos_weight}, "
            f"CV metric=average_precision"
        )

    return {
        "imbalance_config": imbalance_config,
        "current_step": "class_imbalance",
        "progress_messages": state.get("progress_messages", []) + [msg],
    }
