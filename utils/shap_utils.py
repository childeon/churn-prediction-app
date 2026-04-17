"""SHAP computation utilities."""

import numpy as np
import shap

from pipeline.config import SHAP_SAMPLE_SIZE


def compute_shap_values(best_pipeline, X_test, model_type: str):
    """Compute SHAP values and feature names for the best model.

    Returns (shap_values, feature_names, feature_importances).
    feature_importances is a sorted list of dicts: [{feature, importance}, ...]
    """
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    # Sample for speed
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    # Transform through the preprocessor
    X_processed = preprocessor.transform(X_sample)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Build feature names from the ColumnTransformer
    feature_names = _get_feature_names(preprocessor)

    # Pick the right SHAP explainer
    if model_type in ("rf", "gb", "xgb", "lgbm"):
        # XGBoost 3.x stores base_score as '[value]' which older SHAP builds
        # cannot parse. Reset it to a plain float as a compatibility shim.
        if model_type == "xgb" and hasattr(model, "get_booster"):
            booster = model.get_booster()
            cfg = booster.save_config()
            import json, re
            raw = json.loads(cfg)
            raw_score = raw.get("learner", {}).get("learner_model_param", {}).get("base_score", "0.5")
            clean = re.sub(r"[\[\]]", "", str(raw_score))
            booster.set_param("base_score", float(clean))
        explainer = shap.TreeExplainer(model)
    else:
        # Logistic Regression — use LinearExplainer
        X_train_processed = preprocessor.transform(X_test)
        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()
        explainer = shap.LinearExplainer(model, X_train_processed)

    shap_vals = explainer.shap_values(X_processed)

    # For binary classification, some explainers return a list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Mean absolute SHAP → feature importance ranking
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_pairs = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    feature_importances = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in importance_pairs
    ]

    return shap_vals, feature_names, feature_importances


def _get_feature_names(preprocessor):
    """Extract feature names from a fitted ColumnTransformer."""
    names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(columns))
        else:
            # passthrough
            names.extend(columns)
    return list(names)
