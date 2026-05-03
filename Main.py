"""
main.py
-------
End-to-end pipeline runner for the College Admission Outcome Predictor.
CS 210: Data Management for Data Science
Partners: Nikitha Cherukupalli and Varunavi Krishna

Data sources:
  - graduate_admissions.csv  (500 individual applicant records)
  - adm_data.csv             (500 individual applicant records, same schema)
  - college_admissions.csv   (1534 institutional school profiles — used for tier lookup)

Run:
    python main.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_all_data
from preprocessing import run_preprocessing
from feature_engineering import build_all_features
from train import run_training
from evaluate import run_evaluation
from visualize import run_visualizations


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║      College Admission Outcome Predictor                 ║
║      CS 210: Data Management for Data Science            ║
║      Nikitha Cherukupalli & Varunavi Krishna             ║
╚══════════════════════════════════════════════════════════╝
""")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-path",         type=str, default=None)
    parser.add_argument("--adm-path",          type=str, default=None)
    parser.add_argument("--institutional-path",type=str, default=None)
    parser.add_argument("--skip-train",        action="store_true")
    parser.add_argument("--skip-viz",          action="store_true")
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()
    start = time.time()

    # ── Step 1: Load ────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print(" STEP 1: Loading Data")
    print("─"*55)
    raw_df, tier_lookup = load_all_data(
        grad_path=args.grad_path,
        adm_path=args.adm_path,
        institutional_path=args.institutional_path,
    )
    print(f"  Raw applicant records: {raw_df.shape[0]} rows")

    # ── Step 2: Preprocess ──────────────────────────────────────────────
    print("\n" + "─"*55)
    print(" STEP 2: Preprocessing")
    print("─"*55)
    clean_df = run_preprocessing(raw_df, tier_lookup)

    # ── Step 3: Feature Engineering ─────────────────────────────────────
    print("\n" + "─"*55)
    print(" STEP 3: Feature Engineering")
    print("─"*55)
    featured_df = build_all_features(clean_df)

    if not args.skip_train:
        # ── Step 4: Train ────────────────────────────────────────────────
        print("\n" + "─"*55)
        print(" STEP 4: Model Training")
        print("─"*55)
        run_training(featured_df)

    # ── Step 5: Evaluate ────────────────────────────────────────────────
    print("\n" + "─"*55)
    print(" STEP 5: Evaluation")
    print("─"*55)
    eval_results = run_evaluation(featured_df)

    if not args.skip_viz:
        # ── Step 6: Visualize ────────────────────────────────────────────
        print("\n" + "─"*55)
        print(" STEP 6: Visualization")
        print("─"*55)
        run_visualizations(featured_df, eval_results)

    elapsed = time.time() - start
    print(f"\n{'─'*55}")
    print(f" Pipeline complete in {elapsed:.1f}s")
    print(f" Results:        ./results/")
    print(f" Visualizations: ./visualizations/")
    print(f" Models:         ./models/")
    print(f" MLflow UI:      mlflow ui")
    print("─"*55)


if __name__ == "__main__":
    main()