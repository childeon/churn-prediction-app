"""Streamlit UI for the Churn Prediction Multi-Agent System."""

import sys
import os

# Ensure project root is on the path so imports work with `streamlit run app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from dotenv import load_dotenv

load_dotenv()

# Support Streamlit Cloud secrets (fallback to .env for local dev)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from agents.graph import build_graph
from agents.insight_generation import handle_chat_question
from pipeline.config import MODEL_DISPLAY_NAMES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("AI Churn Prediction Tool")
st.caption("LangGraph Multi-Agent System · d6tflow Pipeline · SHAP Explainability")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Sidebar — Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload your customer churn dataset. Must contain a 'churn' column.",
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")
        st.dataframe(raw_df.head(), height=200)

        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    else:
        raw_df = None
        run_btn = False

    st.divider()
    st.caption("Built with LangGraph + d6tflow + Streamlit")

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
if run_btn and raw_df is not None:
    # Reset previous results
    st.session_state.analysis_complete = False
    st.session_state.chat_history = []

    graph = build_graph()
    initial_state = {"raw_df": raw_df}

    step_labels = {
        "clean_data": "Step 1/4: Cleaning data...",
        "run_model_pipeline": "Step 2/4: Training 5 models with Bayesian optimization (this may take a few minutes)...",
        "compute_shap": "Step 3/4: Computing SHAP explanations...",
        "generate_insights": "Step 4/4: Generating business insights with AI...",
    }

    final_state = dict(initial_state)

    with st.status("Running churn analysis pipeline...", expanded=True) as status:
        for event in graph.stream(initial_state):
            node_name = list(event.keys())[0]
            node_output = event[node_name]
            final_state.update(node_output)

            label = step_labels.get(node_name, node_name)
            st.write(f"✓ {label.replace('...', ' — done!')}")

            # Show progress messages
            for msg in node_output.get("progress_messages", []):
                if msg not in (final_state.get("_shown_msgs") or []):
                    st.caption(f"  {msg}")

        status.update(label="Analysis complete!", state="complete", expanded=False)

    st.session_state.pipeline_state = final_state
    st.session_state.analysis_complete = True

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.analysis_complete:
    state = st.session_state.pipeline_state

    tab_results, tab_insights, tab_chat = st.tabs([
        "📊 Model Results",
        "💡 Insights",
        "💬 Ask Questions",
    ])

    # ── Tab 1: Model Results ──
    with tab_results:
        st.subheader("Model Comparison")
        comparison = state.get("model_comparison", [])
        if comparison:
            comp_df = pd.DataFrame(comparison)
            display_df = comp_df[["display_name", "roc_auc", "pr_auc", "f1", "runtime_sec", "optimal_threshold", "expected_profit"]].copy()
            display_df.columns = ["Model", "ROC-AUC", "PR-AUC", "F1", "Runtime (s)", "Optimal Threshold", "Expected Profit ($)"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        best = state.get("best_model_metrics", {})
        if best:
            st.success(
                f"**Best Model: {MODEL_DISPLAY_NAMES.get(best['model'], best['model'])}** — "
                f"ROC-AUC: {best['roc_auc']} | "
                f"Expected Profit: ${best['expected_profit']:,.0f}"
            )

        # ROC-AUC bar chart
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ROC-AUC Comparison")
            if comparison:
                fig, ax = plt.subplots(figsize=(6, 4))
                names = [c["display_name"] for c in comparison]
                aucs = [c["roc_auc"] for c in comparison]
                colors = ["#2563EB" if c["model"] == state["best_model_name"] else "#C8D0E0" for c in comparison]
                ax.barh(names, aucs, color=colors)
                ax.set_xlim(0.6, 1.0)
                ax.set_xlabel("ROC-AUC")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            st.subheader("Profit vs Threshold")
            if best and "threshold_curve" in best:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(best["threshold_curve"], best["profit_curve"], color="#2563EB")
                ax.axvline(best["optimal_threshold"], color="red", linestyle="--", label=f"Optimal: {best['optimal_threshold']:.3f}")
                ax.set_xlabel("Classification Threshold")
                ax.set_ylabel("Expected Profit ($)")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Feature Importance
        st.subheader("Top Feature Drivers (SHAP)")
        importances = state.get("feature_importances", [])
        if importances:
            top_n = importances[:15]
            fig, ax = plt.subplots(figsize=(8, 6))
            features = [f["feature"] for f in reversed(top_n)]
            values = [f["importance"] for f in reversed(top_n)]
            ax.barh(features, values, color="#7C3AED")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Top 15 Features — {MODEL_DISPLAY_NAMES.get(state['best_model_name'], '')}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # SHAP Summary Plot
        shap_vals = state.get("shap_values")
        feature_names = state.get("feature_names")
        if shap_vals is not None and feature_names is not None:
            st.subheader("SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_vals, feature_names=feature_names, show=False, max_display=20)
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 2: Insights ──
    with tab_insights:
        st.subheader("AI-Generated Business Insights")
        insights = state.get("auto_insights", "")
        if insights:
            st.markdown(insights)
        else:
            st.info("No insights generated yet.")

    # ── Tab 3: Chat ──
    with tab_chat:
        st.subheader("Ask Questions About the Analysis")
        st.caption("Ask about churn drivers, at-risk segments, model performance, retention strategies, etc.")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the churn analysis..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Build state with chat history for context
                    chat_state = dict(state)
                    chat_state["chat_history"] = st.session_state.chat_history[:-1]  # exclude current question
                    answer = handle_chat_question(chat_state, prompt)

                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    # Landing page
    st.info("👈 Upload a CSV dataset in the sidebar to get started.")

    with st.expander("How it works"):
        st.markdown("""
1. **Upload** your customer churn dataset (CSV with a `churn` column)
2. **Click "Run Analysis"** — the system will:
   - Clean the data (drop missing values)
   - Train 5 ML models using d6tflow + Hyperopt Bayesian optimization
   - Select the best model by ROC-AUC
   - Compute SHAP feature explanations
   - Generate business insights using AI
3. **Explore results** — model comparison, feature importance, SHAP plots
4. **Read AI insights** — automated analysis of churn drivers and retention strategies
5. **Ask questions** — chat with the AI about your specific analysis
        """)
