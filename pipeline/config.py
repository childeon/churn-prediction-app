"""Constants for the churn prediction pipeline."""

# Business value constants for threshold optimization
BUSINESS_CONSTANTS = {
    "customer_value": 500,
    "contact_cost": 10,
    "retention_success_rate": 0.25,
    "missed_churn_loss": 500,
}

# Models to train
MODEL_TYPES = ["logreg", "rf", "gb", "xgb", "lgbm"]

MODEL_DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

# Hyperopt settings
HYPEROPT_MAX_EVALS = 8

# Columns to drop before modelling
COLUMNS_TO_DROP = ["customer_id", "city"]

# Target column
TARGET_COLUMN = "churn"

# SHAP settings
SHAP_SAMPLE_SIZE = 300
