"""
train.py
--------
Trains Logistic Regression and Random Forest classifiers.
- Global model trained on all 500 records
- Per-tier models trained on each school selectivity tier subset
- All runs logged to MLflow
- Models saved to /models/
"""

import os
import sys
import warnings
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import FEATURE_COLS, get_feature_matrix
from preprocessing import TIER_ORDER

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42
N_SPLITS     = 5

os.makedirs(MODELS_DIR, exist_ok=True)


def build_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=RANDOM_STATE,
        )),
    ])


def build_random_forest():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "random_forest":       build_random_forest,
}


def cross_validate_model(pipeline, X, y):
    """
    Stratified k-fold cross-validation. Returns mean metrics dict.
    Handles tiers with severe class imbalance (e.g. >95% admitted) by
    computing AUC manually per fold and skipping folds with only one class.
    """
    from sklearn.metrics import (
        f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {"f1": [], "auc_roc": [], "precision": [], "recall": [], "accuracy": []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Skip fold if test set has only one class (can't compute AUC)
        if y_test.nunique() < 2:
            continue

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics["auc_roc"].append(roc_auc_score(y_test, y_prob))
        fold_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))

    # If all folds were skipped, return zeros
    if not fold_metrics["auc_roc"]:
        return {"f1": 0, "auc_roc": 0, "precision": 0, "recall": 0, "accuracy": 0}

    return {k: round(float(np.mean(v)), 4) for k, v in fold_metrics.items()}


def train_and_save(pipeline, X, y, save_name: str, run_name: str, params: dict):
    """Cross-validate, fit on full data, log to MLflow, save to disk."""
    metrics = cross_validate_model(pipeline, X, y)

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, save_name)

    path = os.path.join(MODELS_DIR, f"{save_name}.pkl")
    joblib.dump(pipeline, path)

    return metrics, pipeline


def run_training(df):
    """Full training pipeline — global + per-tier for both models."""
    mlflow.set_experiment("college_admission_predictor")
    X_all, y_all = get_feature_matrix(df)

    summary = {}

    with mlflow.start_run(run_name="full_pipeline"):
        for model_name, build_fn in MODEL_BUILDERS.items():
            print(f"\n[train] ── {model_name} ──")

            # Global model
            params = {"model": model_name, "tier": "global", "n_samples": len(df)}
            metrics, pipeline = train_and_save(
                build_fn(), X_all, y_all,
                save_name=f"{model_name}_global",
                run_name=f"{model_name}__global",
                params=params,
            )
            print(f"  {'GLOBAL':<20} | AUC={metrics['auc_roc']:.3f} | F1={metrics['f1']:.3f} | n={len(df)}")
            summary[model_name] = {"global": {"metrics": metrics, "model": pipeline}, "tiers": {}}

            # Per-tier models
            for tier in TIER_ORDER:
                tier_df = df[df["school_tier"] == tier]
                if len(tier_df) < 20 or tier_df["admitted"].nunique() < 2:
                    print(f"  {tier:<20} | Skipped (n={len(tier_df)})")
                    continue

                X_t, y_t = get_feature_matrix(tier_df.copy())
                tier_key  = tier.replace(" ", "_")
                t_params  = {"model": model_name, "tier": tier, "n_samples": len(tier_df)}
                t_metrics, t_pipeline = train_and_save(
                    build_fn(), X_t, y_t,
                    save_name=f"{model_name}_{tier_key}",
                    run_name=f"{model_name}__{tier}",
                    params=t_params,
                )
                print(f"  {tier:<20} | AUC={t_metrics['auc_roc']:.3f} | F1={t_metrics['f1']:.3f} | n={len(tier_df)}")
                summary[model_name]["tiers"][tier] = {"metrics": t_metrics, "model": t_pipeline}

    print("\n" + "="*55)
    print(f"{'Model':<25} {'AUC-ROC':>8} {'F1':>8} {'Pass?':>8}")
    print("="*55)
    for name, data in summary.items():
        m = data["global"]["metrics"]
        passed = "✓ PASS" if m["auc_roc"] >= 0.70 and m["f1"] >= 0.70 else "✗ FAIL"
        print(f"{name:<25} {m['auc_roc']:>8.3f} {m['f1']:>8.3f} {passed:>8}")
    print("="*55)

    return summary


def get_feature_importances(pipeline):
    """Extract feature importances from a fitted pipeline."""
    import pandas as pd
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
    else:
        return None
    return pd.Series(vals, index=FEATURE_COLS).sort_values(ascending=False)


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data_loader import load_data
    from preprocessing import run_preprocessing
    from feature_engineering import build_all_features
    df = build_all_features(run_preprocessing(load_data()))
    run_training(df)