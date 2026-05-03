"""
evaluate.py
-----------
Evaluates trained models on a holdout test set and per school tier.
Saves JSON results and feature importance CSV to /results/.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import FEATURE_COLS, get_feature_matrix
from preprocessing import TIER_ORDER
from train import MODEL_BUILDERS

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path} — run train.py first.")
    return joblib.load(path)


def evaluate_global(df, model_name: str):
    """Evaluate the global model on a held-out 20% test set."""
    X, y = get_feature_matrix(df.copy())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = load_model(f"{model_name}_global")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc_roc = round(float(roc_auc_score(y_test, y_prob)), 4)
    f1      = round(float(f1_score(y_test, y_pred)),      4)
    report  = classification_report(y_test, y_pred, target_names=["Not Admitted", "Admitted"])

    print(f"\n[evaluate] {model_name} — Holdout Test Set (n={len(y_test)})")
    print(f"  AUC-ROC : {auc_roc:.4f}  {'✓' if auc_roc >= 0.70 else '✗'} (target ≥ 0.70)")
    print(f"  F1 Score: {f1:.4f}  {'✓' if f1 >= 0.70 else '✗'} (target ≥ 0.70)")
    print(report)

    return {
        "model":            model_name,
        "n_test":           int(len(y_test)),
        "auc_roc":          auc_roc,
        "f1":               f1,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report":           classification_report(y_test, y_pred, output_dict=True),
        "y_test":           y_test.tolist(),
        "y_prob":           y_prob.tolist(),
    }, pipeline


def evaluate_per_tier(df, model_name: str):
    """Evaluate per school tier using tier-specific models (fall back to global)."""
    tier_results = {}
    X_all, y_all = get_feature_matrix(df.copy())

    # Fit global model on all data for fallback predictions
    global_pipeline = load_model(f"{model_name}_global")
    global_pipeline.fit(X_all, y_all)

    print(f"\n[evaluate] {model_name} — Per-Tier Breakdown")
    print(f"{'Tier':<22} {'N':>5} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 68)

    for tier in TIER_ORDER:
        tier_df = df[df["school_tier"] == tier].copy()
        if len(tier_df) < 10 or tier_df["admitted"].nunique() < 2:
            continue

        X_t, y_t = get_feature_matrix(tier_df)

        # Try tier-specific model first, fall back to global
        tier_key = tier.replace(" ", "_")
        try:
            pipeline = load_model(f"{model_name}_{tier_key}")
            pipeline.fit(X_t, y_t)
        except FileNotFoundError:
            pipeline = global_pipeline

        y_pred = pipeline.predict(X_t)
        y_prob = pipeline.predict_proba(X_t)[:, 1]

        rep = classification_report(y_t, y_pred, output_dict=True, zero_division=0)

        tier_results[tier] = {
            "n":         int(len(tier_df)),
            "auc_roc":   round(float(roc_auc_score(y_t, y_prob)), 4),
            "f1":        round(float(f1_score(y_t, y_pred, zero_division=0)), 4),
            "precision": round(float(rep.get("1", {}).get("precision", 0.0)), 4),
            "recall":    round(float(rep.get("1", {}).get("recall", 0.0)),    4),
        }
        m = tier_results[tier]
        print(f"{tier:<22} {m['n']:>5} {m['auc_roc']:>8.3f} {m['f1']:>8.3f} "
              f"{m['precision']:>10.3f} {m['recall']:>8.3f}")

    return tier_results


def save_results(results: dict, tier_results: dict, importances, model_name: str):
    """Save evaluation outputs to /results/."""
    # JSON — strip non-serializable fields
    out = {k: v for k, v in results.items() if k not in ("y_test", "y_prob")}
    out["tier_breakdown"] = tier_results

    json_path = os.path.join(RESULTS_DIR, f"{model_name}_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[evaluate] Saved: {json_path}")

    # Feature importances CSV
    if importances is not None:
        imp_path = os.path.join(RESULTS_DIR, f"{model_name}_feature_importances.csv")
        importances.reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        ).to_csv(imp_path, index=False)
        print(f"[evaluate] Saved: {imp_path}")


def run_evaluation(df):
    """Full evaluation pipeline for all trained models."""
    all_results = {}

    for model_name in MODEL_BUILDERS.keys():
        print(f"\n{'='*55}\n Evaluating: {model_name}\n{'='*55}")
        try:
            results, pipeline   = evaluate_global(df, model_name)
            tier_results        = evaluate_per_tier(df, model_name)
            clf = pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
            elif hasattr(clf, "coef_"):
                importances = pd.Series(np.abs(clf.coef_[0]), index=FEATURE_COLS).sort_values(ascending=False)
            else:
                importances = None
            if importances is not None:
                print(f"\n[evaluate] Feature Importances — {model_name}")
                print(importances.to_string())

            passed = results["auc_roc"] >= 0.70 and results["f1"] >= 0.70
            print(f"\n[evaluate] Success criteria: {'PASS ✓' if passed else 'FAIL ✗'}")

            save_results(results, tier_results, importances, model_name)
            all_results[model_name] = results

        except FileNotFoundError as e:
            print(f"[evaluate] Skipping {model_name}: {e}")

    return all_results


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data_loader import load_data
    from preprocessing import run_preprocessing
    from feature_engineering import build_all_features
    df = build_all_features(run_preprocessing(load_data()))
    run_evaluation(df)