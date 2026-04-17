"""LangGraph StateGraph — wires the churn prediction pipeline."""

from langgraph.graph import StateGraph, END

from agents.state import PipelineState
from agents.horizon import horizon_definition_node
from agents.class_imbalance import class_imbalance_node
from agents.missing_values import missing_values_node
from agents.model_selection import (
    clean_data_node,
    run_model_pipeline_node,
    compute_shap_node,
)
from agents.insight_generation import generate_insights_node


def build_graph():
    """Build and compile the churn prediction pipeline graph.

    Flow:
        horizon_definition
        -> class_imbalance
        -> missing_values
        -> clean_data
        -> run_model_pipeline
        -> compute_shap
        -> generate_insights
        -> END
    """
    graph = StateGraph(PipelineState)

    graph.add_node("horizon_definition", horizon_definition_node)
    graph.add_node("class_imbalance", class_imbalance_node)
    graph.add_node("missing_values", missing_values_node)
    graph.add_node("clean_data", clean_data_node)
    graph.add_node("run_model_pipeline", run_model_pipeline_node)
    graph.add_node("compute_shap", compute_shap_node)
    graph.add_node("generate_insights", generate_insights_node)

    graph.set_entry_point("horizon_definition")
    graph.add_edge("horizon_definition", "class_imbalance")
    graph.add_edge("class_imbalance", "missing_values")
    graph.add_edge("missing_values", "clean_data")
    graph.add_edge("clean_data", "run_model_pipeline")
    graph.add_edge("run_model_pipeline", "compute_shap")
    graph.add_edge("compute_shap", "generate_insights")
    graph.add_edge("generate_insights", END)

    return graph.compile()
