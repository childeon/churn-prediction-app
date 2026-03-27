"""d6tflow tasks for churn model training pipeline.

Adapted from churn_v2.ipynb. Key change: uses a module-level _dataframe
variable instead of a global df, so the Streamlit app can set it dynamically.
"""

import time
import numpy as np
import pandas as pd
import d6tflow

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

from pipeline.config import (
    BUSINESS_CONSTANTS,
    HYPEROPT_MAX_EVALS,
    TARGET_COLUMN,
)

# ---------------------------------------------------------------------------
# Module-level DataFrame holder — set before running the pipeline
# ---------------------------------------------------------------------------
_dataframe: pd.DataFrame = None


def set_dataframe(df: pd.DataFrame):
    """Call this before flow.run() to inject the clean dataset."""
    global _dataframe
    _dataframe = df


# ---------------------------------------------------------------------------
# Task 1: Prepare Data (train/test split)
# ---------------------------------------------------------------------------
class PrepareData(d6tflow.tasks.TaskPickle):

    def run(self):
        df = _dataframe.copy()

        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        self.save({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        })


# ---------------------------------------------------------------------------
# Task 2: Train a single model (parameterised by model_type)
# ---------------------------------------------------------------------------
@d6tflow.requires(PrepareData)
class TrainModel(d6tflow.tasks.TaskPickle):

    model_type = d6tflow.Parameter()

    def run(self):
        print(f"\n===== Training model: {self.model_type} =====")
        start_time = time.time()

        data = self.inputLoad()
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        categorical_cols = X_train.select_dtypes(include=["object"]).columns
        numeric_cols = X_train.select_dtypes(exclude=["object"]).columns

        # Default preprocessor (no scaling)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols),
            ]
        )

        # ---- Model + param space selection ----
        if self.model_type == "logreg":
            model = LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                    ("num", StandardScaler(), numeric_cols),
                ]
            )
            param_dist = {"model__C": np.logspace(-3, 3, 10)}

        elif self.model_type == "rf":
            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
            )
            param_dist = {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            }

        elif self.model_type == "gb":
            model = GradientBoostingClassifier()
            param_dist = {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
            }

        elif self.model_type == "xgb":
            model = XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=9,
            )
            param_dist = {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.01, 0.1],
            }

        elif self.model_type == "lgbm":
            model = LGBMClassifier(class_weight="balanced", verbosity=-1)
            param_dist = {
                "model__n_estimators": [200, 500],
                "model__num_leaves": [31, 50],
                "model__learning_rate": [0.01, 0.1],
            }

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # ---- Hyperopt Bayesian Optimization ----
        def objective(params):
            pipeline.set_params(**params)
            score = cross_val_score(
                pipeline, X_train, y_train,
                scoring="roc_auc", cv=3, n_jobs=-1,
            ).mean()
            return {"loss": -score, "status": STATUS_OK}

        space = {k: hp.choice(k, v) for k, v in param_dist.items()}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=HYPEROPT_MAX_EVALS,
            trials=trials,
            rstate=np.random.default_rng(42),
            show_progressbar=False,
        )

        best_params = space_eval(space, best)
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # ---- Threshold Optimisation (Business Value) ----
        cv = BUSINESS_CONSTANTS["customer_value"]
        cc = BUSINESS_CONSTANTS["contact_cost"]
        rsr = BUSINESS_CONSTANTS["retention_success_rate"]
        mcl = BUSINESS_CONSTANTS["missed_churn_loss"]

        thresholds = np.linspace(0.05, 0.95, 100)
        profits = []

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            TP = ((y_pred == 1) & (y_test == 1)).sum()
            FP = ((y_pred == 1) & (y_test == 0)).sum()
            FN = ((y_pred == 0) & (y_test == 1)).sum()

            profit = (
                TP * (rsr * cv - cc)
                - FP * cc
                - FN * mcl
            )
            profits.append(profit)

        best_idx = np.argmax(profits)
        best_threshold = float(thresholds[best_idx])
        best_profit = float(profits[best_idx])

        y_pred = (y_prob >= best_threshold).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_prob)

        runtime = time.time() - start_time

        print(f"  ROC AUC: {auc:.4f} | F1: {f1:.4f} | Threshold: {best_threshold:.3f} | Profit: ${best_profit:.0f}")

        self.save(pipeline)
        self.saveMeta({
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
            "f1": float(f1),
            "runtime_sec": float(runtime),
            "best_params": best_params,
            "optimal_threshold": best_threshold,
            "expected_profit": best_profit,
            "threshold_curve": list(thresholds),
            "profit_curve": profits,
        })
