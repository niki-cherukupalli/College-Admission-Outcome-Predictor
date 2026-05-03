"""
visualize.py
------------
Generates all plots for the final report.
All figures saved as PNGs to /visualizations/.

Plots:
  1. Data distributions (class balance + tier breakdown)
  2. ASI distribution by admission outcome
  3. Selectivity gap by outcome and tier
  4. Feature correlation heatmap
  5. Feature importances (LR + RF side by side)
  6. ROC curves per school tier
  7. Confusion matrices
  8. Model comparison bar chart (F1 + AUC)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import FEATURE_COLS, get_feature_matrix
from preprocessing import TIER_ORDER
from train import get_feature_importances, MODEL_BUILDERS

VIZ_DIR    = os.path.join(os.path.dirname(__file__), "..", "visualizations")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42

os.makedirs(VIZ_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

TIER_COLORS = {
    "Ivy":              "#8e44ad",
    "Highly Selective": "#2980b9",
    "Selective":        "#27ae60",
    "Less Selective":   "#e67e22",
}


def save(name):
    path = os.path.join(VIZ_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved: {path}")


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ── 1. Data distributions ──────────────────────────────────────────────────
def plot_data_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold")

    # Admission outcome
    counts = df["admitted"].value_counts().rename({0: "Not Admitted", 1: "Admitted"})
    bars = axes[0].bar(counts.index, counts.values,
                       color=["#e74c3c", "#2ecc71"], edgecolor="white")
    axes[0].bar_label(bars, fmt="%d")
    axes[0].set_title("Admission Outcome Distribution")
    axes[0].set_ylabel("Count")

    # Tier breakdown
    tier_counts = df["school_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    colors = [TIER_COLORS[t] for t in tier_counts.index]
    bars2 = axes[1].bar(tier_counts.index, tier_counts.values, color=colors, edgecolor="white")
    axes[1].bar_label(bars2, fmt="%d")
    axes[1].set_title("Applications by School Tier")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save("1_data_distributions")


# ── 2. ASI distribution ────────────────────────────────────────────────────
def plot_asi_distribution(df):
    if "asi" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for outcome, label, color in [(1, "Admitted", "#2ecc71"), (0, "Not Admitted", "#e74c3c")]:
        subset = df[df["admitted"] == outcome]["asi"]
        subset.plot.kde(ax=ax, label=f"{label} (median={subset.median():.1f})",
                        color=color, linewidth=2)
        ax.axvline(subset.median(), color=color, linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Academic Strength Index (ASI)")
    ax.set_ylabel("Density")
    ax.set_title("ASI Distribution by Admission Outcome", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save("2_asi_distribution")


# ── 3. Selectivity gap ────────────────────────────────────────────────────
def plot_selectivity_gap(df):
    if "selectivity_gap" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_df = df.copy()
    plot_df["outcome"] = plot_df["admitted"].map({0: "Not Admitted", 1: "Admitted"})
    sns.boxplot(data=plot_df, x="outcome", y="selectivity_gap",
                order=["Not Admitted", "Admitted"],
                palette={"Not Admitted": "#e74c3c", "Admitted": "#2ecc71"}, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Selectivity Gap by Outcome", fontweight="bold")
    axes[0].set_ylabel("Selectivity Gap (ASI − Tier Median ASI)")

    tier_order = [t for t in TIER_ORDER if t in df["school_tier"].values]
    sns.boxplot(data=df, x="school_tier", y="selectivity_gap",
                order=tier_order, palette=TIER_COLORS, ax=axes[1])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Selectivity Gap by School Tier", fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save("3_selectivity_gap")


# ── 4. Correlation heatmap ────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    X, y = get_feature_matrix(df.copy())
    corr_df = X.copy()
    corr_df["admitted"] = y.values
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=15)
    plt.tight_layout()
    save("4_correlation_heatmap")


# ── 5. Feature importances ────────────────────────────────────────────────
def plot_feature_importances(df):
    X, y = get_feature_matrix(df.copy())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature Importances by Model", fontsize=14, fontweight="bold")

    for ax, model_name in zip(axes, ["logistic_regression", "random_forest"]):
        pipeline = load_model(f"{model_name}_global")
        if pipeline is None:
            ax.text(0.5, 0.5, f"{model_name}\n(not found)", ha="center", va="center")
            continue
        pipeline.fit(X, y)
        importances = get_feature_importances(pipeline)
        if importances is None:
            continue
        colors = ["#3498db"] * len(importances)
        ax.barh(importances.index[::-1], importances.values[::-1], color=colors)
        ax.set_title(model_name.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("|Coefficient|" if "logistic" in model_name else "Gini Importance")

    plt.tight_layout()
    save("5_feature_importances")


# ── 6. ROC curves per tier ────────────────────────────────────────────────
def plot_roc_curves(df, model_name="random_forest"):
    X_all, y_all = get_feature_matrix(df.copy())
    pipeline = load_model(f"{model_name}_global")
    if pipeline is None:
        print(f"[visualize] {model_name}_global not found, skipping ROC curves")
        return
    pipeline.fit(X_all, y_all)

    fig, ax = plt.subplots(figsize=(8, 6))
    for tier in TIER_ORDER:
        tier_df = df[df["school_tier"] == tier]
        if len(tier_df) < 10 or tier_df["admitted"].nunique() < 2:
            continue
        X_t, y_t = get_feature_matrix(tier_df.copy())
        y_prob = pipeline.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{tier} (AUC={roc_auc:.3f})",
                color=TIER_COLORS.get(tier, "gray"), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline")
    ax.axhline(0.70, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Target AUC 0.70")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves by School Tier\n({model_name.replace('_',' ').title()})",
                 fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save(f"6_roc_curves_{model_name}")


# ── 7. Confusion matrices ──────────────────────────────────────────────────
def plot_confusion_matrices(df):
    X, y = get_feature_matrix(df.copy())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices (Holdout Test Set)", fontsize=13, fontweight="bold")

    for ax, model_name in zip(axes, ["logistic_regression", "random_forest"]):
        pipeline = load_model(f"{model_name}_global")
        if pipeline is None:
            ax.text(0.5, 0.5, f"{model_name}\nnot found", ha="center", va="center")
            continue
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Not Admitted", "Admitted"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(model_name.replace("_", " ").title(), fontweight="bold")

    plt.tight_layout()
    save("7_confusion_matrices")


# ── 8. Model comparison ───────────────────────────────────────────────────
def plot_model_comparison(eval_results):
    if not eval_results:
        return
    models   = list(eval_results.keys())
    f1_vals  = [eval_results[m]["f1"]      for m in models]
    auc_vals = [eval_results[m]["auc_roc"] for m in models]

    x     = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, f1_vals,  width, label="F1 Score",  color="#3498db")
    b2 = ax.bar(x + width/2, auc_vals, width, label="AUC-ROC",   color="#9b59b6")
    ax.bar_label(b1, fmt="%.3f", padding=3)
    ax.bar_label(b2, fmt="%.3f", padding=3)
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1.2, label="Target (0.70)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save("8_model_comparison")


# ── Full pipeline ─────────────────────────────────────────────────────────
def run_visualizations(df, eval_results=None):
    print("[visualize] Generating all plots...")
    plot_data_distributions(df)
    plot_asi_distribution(df)
    plot_selectivity_gap(df)
    plot_correlation_heatmap(df)
    plot_feature_importances(df)
    plot_roc_curves(df, "random_forest")
    plot_roc_curves(df, "logistic_regression")
    plot_confusion_matrices(df)
    if eval_results:
        plot_model_comparison(eval_results)
    print(f"[visualize] All plots saved to {VIZ_DIR}/")


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data_loader import load_data
    from preprocessing import run_preprocessing
    from feature_engineering import build_all_features
    df = build_all_features(run_preprocessing(load_data()))
    run_visualizations(df)