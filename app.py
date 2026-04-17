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
from agents.chart_agent import (
    pr_curve_figure,
    probability_distribution_figure,
    cumulative_gains_figure,
    lift_chart_figure,
)
from agents.simulation_agent import simulate_profit, explain_simulation
from pipeline.config import MODEL_DISPLAY_NAMES, BUSINESS_CONSTANTS

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
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "sim_explanation" not in st.session_state:
    st.session_state.sim_explanation = None
if "sim_constants" not in st.session_state:
    st.session_state.sim_constants = None

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

        st.subheader("Churn Horizon")
        selected_horizon = st.selectbox(
            "Define churn window",
            options=[30, 60, 90],
            format_func=lambda h: f"{h}-day churn",
            help="Customers who churned within this window will be labelled as churned (1). Changing this redefines the supervised learning problem.",
        )

        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    else:
        raw_df = None
        selected_horizon = 30
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
    initial_state = {"raw_df": raw_df, "selected_horizon": selected_horizon}

    step_labels = {
        "horizon_definition": f"Step 1/7: Defining {selected_horizon}-day churn horizon...",
        "class_imbalance": "Step 2/7: Checking class imbalance...",
        "missing_values": "Step 3/7: Profiling and imputing missing values...",
        "clean_data": "Step 4/7: Cleaning data...",
        "run_model_pipeline": "Step 5/7: Training 5 models with Bayesian optimization (this may take a few minutes)...",
        "compute_shap": "Step 6/7: Computing SHAP explanations...",
        "generate_insights": "Step 7/7: Generating business insights with AI...",
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

    tab_profile, tab_results, tab_charts, tab_sim, tab_insights, tab_chat = st.tabs([
        "🔍 Data Profile",
        "📊 Model Results",
        "📈 Charts",
        "🔬 Simulation",
        "💡 Insights",
        "💬 Ask Questions",
    ])

    # ── Tab 0: Data Profile ──
    with tab_profile:
        # Class Imbalance
        st.subheader("Class Imbalance Check")
        imb = state.get("imbalance_config", {})
        if imb:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Minority class (churn=1)", f"{imb['minority_ratio']:.1%}")
            col_b.metric("Churned customers", f"{imb['minority_count']:,}")
            col_c.metric("Non-churned customers", f"{imb['majority_count']:,}")

            if imb.get("is_imbalanced"):
                st.warning(
                    f"**Imbalance detected.** Mitigation applied: "
                    f"`class_weight=balanced` (LogReg, RF, LightGBM), "
                    f"`scale_pos_weight={imb['xgb_scale_pos_weight']}` (XGBoost), "
                    f"CV metric switched to `{imb['primary_metric']}`."
                )
            else:
                st.success("Classes are balanced — no mitigation needed.")

        st.divider()

        # Missing Values
        st.subheader("Missing Values — LLM Reasoning")
        strategies = state.get("missing_strategies", [])
        profile = state.get("missing_profile", [])
        if not strategies:
            st.success("No missing values were found in the dataset.")
        else:
            st.caption(
                f"{len(strategies)} column(s) had missing values. "
                "The LLM proposed an imputation strategy for each. "
                "Rules were auto-applied before model training."
            )
            for s in strategies:
                with st.expander(
                    f"**{s['column']}** — {s['missing_rate']:.1%} missing · "
                    f"strategy: `{s['strategy']}`"
                    + (f" → `{s['fill_value']}`" if s.get("fill_value") is not None else ""),
                    expanded=False,
                ):
                    st.markdown(f"**Interpretation:** {s.get('interpretation', '—')}")
                    st.markdown(f"**Reasoning:** {s.get('reasoning', '—')}")

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

    # ── Tab 2: Charts ──
    with tab_charts:
        st.subheader("Model Diagnostics")
        st.caption("Four charts built from the best model's test-set predictions.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Precision-Recall Curve**")
            fig = pr_curve_figure(state)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            st.markdown("**Churn Probability Distribution**")
            fig = probability_distribution_figure(state)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Cumulative Gains**")
            fig = cumulative_gains_figure(state)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

        with col4:
            st.markdown("**Lift Chart by Decile**")
            fig = lift_chart_figure(state)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

    # ── Tab 3: Simulation ──
    with tab_sim:
        st.subheader("What-If Business Simulation")
        st.caption(
            "Adjust business assumptions and re-run the profit optimisation. "
            "No retraining — uses the trained model's test-set predictions."
        )

        baseline_metrics = state.get("best_model_metrics", {})
        predictions = state.get("predictions", {})

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("**Adjust assumptions**")
            cv = st.slider(
                "Customer lifetime value ($)",
                min_value=100, max_value=2000,
                value=BUSINESS_CONSTANTS["customer_value"], step=50,
            )
            cc = st.slider(
                "Contact cost ($)",
                min_value=1, max_value=100,
                value=BUSINESS_CONSTANTS["contact_cost"], step=1,
            )
            rsr = st.slider(
                "Retention success rate (%)",
                min_value=5, max_value=80,
                value=int(BUSINESS_CONSTANTS["retention_success_rate"] * 100), step=5,
            )
            mcl = st.slider(
                "Missed churn loss ($)",
                min_value=100, max_value=2000,
                value=BUSINESS_CONSTANTS["missed_churn_loss"], step=50,
            )

            run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

        if run_sim and predictions:
            new_constants = {
                "customer_value": cv,
                "contact_cost": cc,
                "retention_success_rate": rsr / 100,
                "missed_churn_loss": mcl,
            }
            with st.spinner("Computing..."):
                result = simulate_profit(
                    y_test=predictions["y_test"],
                    y_prob=predictions["y_prob"],
                    **new_constants,
                )
                explanation = explain_simulation(
                    baseline_metrics=baseline_metrics,
                    baseline_constants=BUSINESS_CONSTANTS,
                    new_result=result,
                    new_constants=new_constants,
                )
            st.session_state.sim_result = result
            st.session_state.sim_explanation = explanation
            st.session_state.sim_constants = new_constants

        with col_r:
            st.markdown("**Results**")
            sim = st.session_state.sim_result
            baseline_profit = baseline_metrics.get("expected_profit", 0)
            baseline_threshold = baseline_metrics.get("optimal_threshold", 0.5)

            if sim:
                delta_profit = sim["expected_profit"] - baseline_profit
                delta_threshold = sim["optimal_threshold"] - baseline_threshold

                m1, m2 = st.columns(2)
                m1.metric(
                    "Optimal Threshold",
                    f"{sim['optimal_threshold']:.3f}",
                    delta=f"{delta_threshold:+.3f}",
                )
                m2.metric(
                    "Expected Profit",
                    f"${sim['expected_profit']:,.0f}",
                    delta=f"${delta_profit:+,.0f}",
                )
                m3, m4 = st.columns(2)
                m3.metric("Contacts Made", f"{sim['contacts_made']:,}")
                m4.metric("Churners Missed", f"{sim['churners_missed']:,}")

                # Overlay profit curves: baseline vs new scenario
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(
                    baseline_metrics["threshold_curve"],
                    baseline_metrics["profit_curve"],
                    color="#C8D0E0", lw=1.5, label="Baseline",
                )
                ax.plot(
                    sim["threshold_curve"],
                    sim["profit_curve"],
                    color="#2563EB", lw=2, label="Simulation",
                )
                ax.axvline(baseline_threshold, color="#C8D0E0", linestyle="--", lw=1)
                ax.axvline(sim["optimal_threshold"], color="#2563EB", linestyle="--", lw=1)
                ax.set_xlabel("Threshold")
                ax.set_ylabel("Expected Profit ($)")
                ax.set_title("Profit Curve: Baseline vs Simulation")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if st.session_state.sim_explanation:
                    st.info(st.session_state.sim_explanation)
            else:
                st.markdown(
                    f"Baseline — threshold: **{baseline_threshold:.3f}** · "
                    f"profit: **${baseline_profit:,.0f}**"
                )
                st.caption("Adjust the sliders and click Run Simulation to see what changes.")

    # ── Tab 4: Insights ──
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
2. **Select a churn horizon** (30 / 60 / 90 days) — this defines what "churned" means for the model
3. **Click "Run Analysis"** — the system runs a 7-step offline pipeline:
   - **Horizon definition** — builds `churn_30d/60d/90d` labels and sets the active target
   - **Class imbalance agent** — checks target ratio, configures class weights and CV metric
   - **Missing values agent** — profiles each column, uses LLM to reason about missingness, auto-imputes
   - **Data cleaning** — drops identifier columns, computes dataset summary
   - **Model training** — 5 ML models via d6tflow + Hyperopt Bayesian optimization
   - **SHAP explainability** — computes feature importances for the best model
   - **Insight generation** — AI-written business narrative
4. **Data Profile tab** — view imbalance config and LLM imputation reasoning
5. **Model Results tab** — model comparison, feature importance, SHAP plots
6. **Insights tab** — automated analysis of churn drivers and retention strategies
7. **Ask questions** — chat with the AI about your specific analysis
        """)
