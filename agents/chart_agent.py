"""Chart agent — returns matplotlib figures from stored pipeline state.

Pure functions only: read from state, return Figure objects.
Called directly from app.py — not LangGraph nodes.

Charts:
  pr_curve_figure              – Precision-Recall curve with AUC
  probability_distribution_figure – Predicted probability histogram by actual label
  cumulative_gains_figure      – % churners captured vs % customers contacted
  lift_chart_figure            – Decile lift over random baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def _get_predictions(state: dict):
    """Extract y_test and y_prob arrays from state. Returns (None, None) if missing."""
    preds = state.get("predictions", {})
    y_test = preds.get("y_test")
    y_prob = preds.get("y_prob")
    if y_test is None or y_prob is None:
        return None, None
    return np.array(y_test), np.array(y_prob)


def pr_curve_figure(state: dict) -> plt.Figure | None:
    """Precision-Recall curve for the best model."""
    y_test, y_prob = _get_predictions(state)
    if y_test is None:
        return None

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val = auc(recall, precision)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="#7C3AED", lw=2, label=f"Model (AUC = {pr_auc_val:.3f})")
    ax.axhline(baseline, color="#C8D0E0", linestyle="--", lw=1, label=f"Random baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    plt.tight_layout()
    return fig


def probability_distribution_figure(state: dict) -> plt.Figure | None:
    """Predicted churn probability histogram, split by actual churn label."""
    y_test, y_prob = _get_predictions(state)
    if y_test is None:
        return None

    optimal_threshold = state.get("best_model_metrics", {}).get("optimal_threshold", 0.5)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_prob[y_test == 0], bins=40, alpha=0.6, color="#C8D0E0", label="Retained (actual)")
    ax.hist(y_prob[y_test == 1], bins=40, alpha=0.75, color="#2563EB", label="Churned (actual)")
    ax.axvline(
        optimal_threshold, color="red", linestyle="--", lw=1.5,
        label=f"Optimal threshold ({optimal_threshold:.3f})",
    )
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Customer Count")
    ax.set_title("Churn Probability Distribution")
    ax.legend()
    plt.tight_layout()
    return fig


def cumulative_gains_figure(state: dict) -> plt.Figure | None:
    """Cumulative gains: % of churners captured when contacting top X% of customers."""
    y_test, y_prob = _get_predictions(state)
    if y_test is None:
        return None

    sorted_idx = np.argsort(y_prob)[::-1]
    sorted_labels = y_test[sorted_idx]

    n = len(sorted_labels)
    total_churners = sorted_labels.sum()
    gains = np.cumsum(sorted_labels) / total_churners * 100
    pct_customers = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pct_customers, gains, color="#2563EB", lw=2, label="Model")
    ax.plot([0, 100], [0, 100], color="#C8D0E0", linestyle="--", lw=1, label="Random baseline")
    ax.fill_between(pct_customers, gains, pct_customers, alpha=0.08, color="#2563EB")
    ax.set_xlabel("% Customers Contacted (ranked by predicted risk)")
    ax.set_ylabel("% Churners Captured")
    ax.set_title("Cumulative Gains Chart")
    ax.legend()
    plt.tight_layout()
    return fig


def lift_chart_figure(state: dict) -> plt.Figure | None:
    """Decile lift chart: model lift over random baseline per risk decile."""
    y_test, y_prob = _get_predictions(state)
    if y_test is None:
        return None

    sorted_idx = np.argsort(y_prob)[::-1]
    sorted_labels = y_test[sorted_idx]
    n = len(sorted_labels)
    baseline_rate = sorted_labels.mean()

    n_deciles = 10
    decile_size = n // n_deciles
    lifts = []
    for i in range(n_deciles):
        chunk = sorted_labels[i * decile_size: (i + 1) * decile_size]
        rate = chunk.mean() if len(chunk) > 0 else 0.0
        lifts.append(rate / baseline_rate if baseline_rate > 0 else 1.0)

    labels = [f"D{i+1}" for i in range(n_deciles)]
    colors = ["#2563EB" if lv >= 1 else "#C8D0E0" for lv in lifts]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, lifts, color=colors)
    ax.axhline(1.0, color="red", linestyle="--", lw=1.5, label="Baseline (lift = 1.0)")
    ax.set_xlabel("Decile (D1 = highest predicted risk)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart by Decile")
    ax.legend()
    plt.tight_layout()
    return fig
