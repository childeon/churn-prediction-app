"""Agent 2 — Insight Generation: LangGraph node + chat function.

generate_insights_node  – auto-generates business insights after model training
handle_chat_question    – answers user questions about the results (standalone)
"""

from openai import OpenAI

from agents.state import PipelineState
from pipeline.config import MODEL_DISPLAY_NAMES


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Build the context string used by both auto-insights and chat
# ---------------------------------------------------------------------------
def _build_context(state: PipelineState) -> str:
    summary = state.get("dataset_summary", {})
    comparison = state.get("model_comparison", [])
    best = state.get("best_model_metrics", {})
    importances = state.get("feature_importances", [])

    # Model comparison table
    table_lines = ["Model | ROC-AUC | PR-AUC | F1 | Threshold | Expected Profit"]
    table_lines.append("--- | --- | --- | --- | --- | ---")
    for m in comparison:
        table_lines.append(
            f"{m['display_name']} | {m['roc_auc']} | {m['pr_auc']} | "
            f"{m['f1']} | {m['optimal_threshold']:.3f} | ${m['expected_profit']:,.0f}"
        )

    # Top features
    top_features = importances[:15]
    feat_lines = [f"  {i+1}. {f['feature']} (SHAP importance: {f['importance']:.4f})"
                  for i, f in enumerate(top_features)]

    context = f"""
DATASET SUMMARY:
- {summary.get('rows', '?')} customers, {summary.get('columns', '?')} features
- Churn rate: {summary.get('churn_rate_pct', '?')}%
- {summary.get('rows_dropped', 0)} rows dropped due to missing values
- Numeric features: {', '.join(summary.get('numeric_features', [])[:10])}
- Categorical features: {', '.join(summary.get('categorical_features', []))}

MODEL COMPARISON:
{chr(10).join(table_lines)}

BEST MODEL: {MODEL_DISPLAY_NAMES.get(best.get('model', ''), best.get('model', 'N/A'))}
- ROC-AUC: {best.get('roc_auc', 'N/A')}
- PR-AUC: {best.get('pr_auc', 'N/A')}
- F1 Score: {best.get('f1', 'N/A')}
- Optimal classification threshold: {best.get('optimal_threshold', 'N/A')}
- Expected profit at optimal threshold: ${best.get('expected_profit', 0):,.0f}
- Business assumptions: customer_value=$500, contact_cost=$10, retention_success_rate=25%, missed_churn_loss=$500

TOP FEATURES BY SHAP IMPORTANCE:
{chr(10).join(feat_lines)}
""".strip()

    return context


# ---------------------------------------------------------------------------
# Node: Generate auto-insights
# ---------------------------------------------------------------------------
INSIGHTS_SYSTEM_PROMPT = """You are a senior business analyst specializing in customer churn prediction.
You will receive ML model results and SHAP feature analysis from a churn prediction pipeline.
Provide clear, actionable business insights in markdown format.

Structure your response as:
## Key Findings
- Top 5 churn risk drivers with business explanations

## At-Risk Customer Segments
- Describe 3-4 customer segments most at risk, using feature ranges

## Retention Recommendations
- 3-5 specific, actionable retention strategies tied to the data

## Business Impact
- Expected profit impact based on the threshold optimization
- What the optimal threshold means in practical terms

Be specific — reference actual feature names and values from the data. Avoid generic advice."""


def generate_insights_node(state: PipelineState) -> dict:
    context = _build_context(state)

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.3,
    )

    insights = response.choices[0].message.content

    return {
        "auto_insights": insights,
        "chat_history": [],
        "current_step": "Insights generated",
        "progress_messages": state.get("progress_messages", []) + [
            "Generated business insights",
        ],
    }


# ---------------------------------------------------------------------------
# Standalone chat function (not a graph node)
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are a churn prediction analyst assistant. You have access to the results
of a machine learning pipeline that trained 5 models on customer churn data.

Answer questions about the model results, feature importance, customer segments,
and retention strategies. Be specific and reference the data provided.

If asked about something not covered by the data, say so clearly."""


def handle_chat_question(state: PipelineState, question: str) -> str:
    """Answer a user question using the pipeline results as context."""
    context = _build_context(state)
    chat_history = state.get("chat_history", [])

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here are the analysis results:\n\n{context}"},
        {"role": "assistant", "content": "I have the full analysis results. What would you like to know?"},
    ]

    # Add prior Q&A turns
    for msg in chat_history:
        messages.append(msg)

    # Add the new question
    messages.append({"role": "user", "content": question})

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.3,
    )

    return response.choices[0].message.content
