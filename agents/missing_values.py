"""Node 5 — Missing Values Agent: profile, reason (LLM), and impute missing data.

Sub-steps:
  5a. profile_missing   – pure Python: detect NaN columns, rates, dtypes, samples
  5b. reason_about_column – LLM: interpret missingness and suggest strategy per column
  5c. (human gate)      – strategies surfaced in the UI for analyst review
  5d. apply_strategies  – pure Python: execute the approved imputation rules
"""

import json

import pandas as pd
from openai import OpenAI

from agents.state import PipelineState


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# 5a: Profile missing values
# ---------------------------------------------------------------------------
def _profile_missing(df: pd.DataFrame) -> list[dict]:
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]

    rows = []
    for col in missing_cols.index:
        n_missing = int(missing_counts[col])
        rows.append({
            "column": col,
            "missing_count": n_missing,
            "missing_rate": round(n_missing / len(df), 4),
            "dtype": str(df[col].dtype),
            "sample_values": df[col].dropna().head(5).tolist(),
        })
    return rows


# ---------------------------------------------------------------------------
# 5b: LLM reasoning — one call per column
# ---------------------------------------------------------------------------
_REASON_PROMPT = """\
You are analysing a customer churn dataset. One column has missing values.

Column: {column}
Data type: {dtype}
Missing rate: {missing_rate:.1%} ({missing_count} rows)
Sample non-null values: {sample_values}

Return a JSON object with exactly these fields:
- "interpretation": one sentence explaining what missingness likely means in a business context
- "strategy": one of ["fill_constant", "fill_median", "fill_mode", "drop_rows"]
- "fill_value": the constant to use if strategy is "fill_constant", otherwise null
- "reasoning": one sentence justifying the strategy

Respond with only the JSON object."""


def _reason_about_column(row: dict) -> dict:
    prompt = _REASON_PROMPT.format(**row)
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    # Always carry the column name through for the executor
    result["column"] = row["column"]
    result["missing_rate"] = row["missing_rate"]
    result["dtype"] = row["dtype"]
    return result


# ---------------------------------------------------------------------------
# 5d: Apply the approved imputation rules
# ---------------------------------------------------------------------------
def _apply_strategies(df: pd.DataFrame, strategies: list[dict]) -> pd.DataFrame:
    df = df.copy()
    for s in strategies:
        col = s.get("column")
        if not col or col not in df.columns:
            continue
        strat = s.get("strategy", "drop_rows")
        if strat == "fill_constant":
            df[col] = df[col].fillna(s.get("fill_value", "Unknown"))
        elif strat == "fill_median":
            df[col] = df[col].fillna(df[col].median())
        elif strat == "fill_mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
        # drop_rows: NaNs in this column remain and are cleaned below
    return df.dropna()


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
def missing_values_node(state: PipelineState) -> dict:
    df = state["raw_df"]
    msgs = list(state.get("progress_messages", []))

    # 5a: Profile
    profile = _profile_missing(df)

    if not profile:
        msgs.append("No missing values detected — skipping imputation.")
        return {
            "missing_profile": [],
            "missing_strategies": [],
            "current_step": "missing_values",
            "progress_messages": msgs,
        }

    msgs.append(
        f"Found {len(profile)} column(s) with missing values — running LLM reasoning."
    )

    # 5b: LLM reasoning per column
    strategies = []
    for row in profile:
        strategy = _reason_about_column(row)
        strategies.append(strategy)
        fill_note = (
            f" → '{strategy['fill_value']}'"
            if strategy.get("fill_value") is not None
            else ""
        )
        msgs.append(
            f"  {row['column']} ({row['missing_rate']:.1%} missing): "
            f"{strategy['strategy']}{fill_note}"
        )

    # 5d: Apply rules (5c human gate is surfaced in the UI via missing_strategies)
    rows_before = len(df)
    clean_df = _apply_strategies(df, strategies)
    rows_dropped = rows_before - len(clean_df)
    if rows_dropped:
        msgs.append(f"Dropped {rows_dropped} rows after imputation.")

    return {
        "missing_profile": profile,
        "missing_strategies": strategies,
        "raw_df": clean_df,
        "current_step": "missing_values",
        "progress_messages": msgs,
    }
