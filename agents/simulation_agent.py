"""Simulation agent — what-if business scenario analysis.

Recomputes the profit curve using stored predictions and new business constants.
No model retraining. All profit logic is deterministic Python; LLM narrates the result.

Public API:
  simulate_profit(y_test, y_prob, constants)  → SimulationResult dict
  explain_simulation(baseline, result, constants) → str
"""

import numpy as np
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Deterministic profit recomputation
# ---------------------------------------------------------------------------
def simulate_profit(
    y_test: list | np.ndarray,
    y_prob: list | np.ndarray,
    customer_value: float,
    contact_cost: float,
    retention_success_rate: float,
    missed_churn_loss: float,
) -> dict:
    """Recompute profit across thresholds with new business constants.

    Uses the same threshold grid as the training pipeline (0.05 → 0.95, 100 steps).
    Returns optimal threshold, expected profit, full curves, and confusion breakdown.
    """
    y_test = np.array(y_test)
    y_prob = np.array(y_prob)
    thresholds = np.linspace(0.05, 0.95, 100)

    profits = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        TP = int(((y_pred == 1) & (y_test == 1)).sum())
        FP = int(((y_pred == 1) & (y_test == 0)).sum())
        FN = int(((y_pred == 0) & (y_test == 1)).sum())
        profit = (
            TP * (retention_success_rate * customer_value - contact_cost)
            - FP * contact_cost
            - FN * missed_churn_loss
        )
        profits.append(float(profit))

    best_idx = int(np.argmax(profits))
    best_t = float(thresholds[best_idx])

    y_pred_best = (y_prob >= best_t).astype(int)
    TP = int(((y_pred_best == 1) & (y_test == 1)).sum())
    FP = int(((y_pred_best == 1) & (y_test == 0)).sum())
    FN = int(((y_pred_best == 0) & (y_test == 1)).sum())
    TN = int(((y_pred_best == 0) & (y_test == 0)).sum())

    return {
        "optimal_threshold": best_t,
        "expected_profit": profits[best_idx],
        "threshold_curve": thresholds.tolist(),
        "profit_curve": profits,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "tn": TN,
        "contacts_made": TP + FP,
        "churners_missed": FN,
    }


# ---------------------------------------------------------------------------
# LLM narrative
# ---------------------------------------------------------------------------
_EXPLAIN_PROMPT = """\
You are a churn analyst explaining a what-if scenario to a business stakeholder.

Baseline:
- Customer value: ${cv_base}, Contact cost: ${cc_base}, Retention rate: {rsr_base:.0%}, Missed churn loss: ${mcl_base}
- Optimal threshold: {t_base:.3f}, Expected profit: ${p_base:,.0f}
- Contacts made: {contacts_base}, Churners missed: {missed_base}

New scenario:
- Customer value: ${cv_new}, Contact cost: ${cc_new}, Retention rate: {rsr_new:.0%}, Missed churn loss: ${mcl_new}
- New optimal threshold: {t_new:.3f}, Expected profit: ${p_new:,.0f}  ({delta:+,.0f} vs baseline)
- Contacts made: {contacts_new}, Churners missed: {missed_new}

In 2-3 sentences explain: what changed, why the threshold and profit shifted, and what the business should take away."""


def explain_simulation(
    baseline_metrics: dict,
    baseline_constants: dict,
    new_result: dict,
    new_constants: dict,
) -> str:
    """Ask GPT-4.1 to narrate the simulation result."""
    prompt = _EXPLAIN_PROMPT.format(
        cv_base=baseline_constants["customer_value"],
        cc_base=baseline_constants["contact_cost"],
        rsr_base=baseline_constants["retention_success_rate"],
        mcl_base=baseline_constants["missed_churn_loss"],
        t_base=baseline_metrics["optimal_threshold"],
        p_base=baseline_metrics["expected_profit"],
        contacts_base=baseline_metrics.get("tp", 0) + baseline_metrics.get("fp", 0),
        missed_base=baseline_metrics.get("fn", 0),
        cv_new=new_constants["customer_value"],
        cc_new=new_constants["contact_cost"],
        rsr_new=new_constants["retention_success_rate"],
        mcl_new=new_constants["missed_churn_loss"],
        t_new=new_result["optimal_threshold"],
        p_new=new_result["expected_profit"],
        delta=new_result["expected_profit"] - baseline_metrics["expected_profit"],
        contacts_new=new_result["contacts_made"],
        missed_new=new_result["churners_missed"],
    )

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
