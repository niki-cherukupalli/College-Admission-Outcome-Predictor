"""
Ingests and merges the two Kaggle datasets into a unified schema.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_graduate_admissions(path: str = None) -> pd.DataFrame:
    # Load the Kaggle Graduate Admissions dataset.

    path = path or os.path.join(DATA_DIR, "graduate_admissions.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

    rename_map = {
        "gre_score": "gre_score",
        "toefl_score": "toefl_score",
        "university_rating": "school_tier_raw",
        "sop": "sop_strength",
        "lor_": "lor_strength",
        "lor": "lor_strength",
        "cgpa": "gpa_raw",
        "research": "has_research",
        "chance_of_admit": "admit_probability",
        "chance_of_admit_": "admit_probability",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "admit_probability" in df.columns:
        df["admitted"] = (df["admit_probability"] >= 0.5).astype(int)

    df["dataset_source"] = "graduate"
    print(f"[data_loader] Graduate admissions: {len(df)} records, columns: {list(df.columns)}")
    return df


def load_college_admissions(path: str = None) -> pd.DataFrame:
   # Load the Kaggle College Admissions (undergraduate) dataset.

    path = path or os.path.join(DATA_DIR, "college_admissions.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

    rename_map = {
        "sat_score": "sat_score",
        "sat": "sat_score",
        "act_score": "act_score",
        "act": "act_score",
        "gpa": "gpa_raw",
        "weighted_gpa": "gpa_raw",
        "unweighted_gpa": "gpa_raw",
        "extracurriculars": "extracurricular_raw",
        "extracurricular": "extracurricular_raw",
        "ec": "extracurricular_raw",
        "decision": "decision_raw",
        "result": "decision_raw",
        "status": "decision_raw",
        "school": "school_name",
        "college": "school_name",
        "university": "school_name",
        "tier": "school_tier_raw",
        "selectivity": "school_tier_raw",
        "major": "intended_major",
        "intended_major": "intended_major",
        "legacy": "is_legacy",
        "first_gen": "is_first_gen",
        "first_generation": "is_first_gen",
        "in_state": "is_instate",
        "instate": "is_instate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "decision_raw" in df.columns:
        decision_map = {
            "admit": 1, "admitted": 1, "acceptance": 1, "yes": 1, "accepted": 1,
            "deny": 0, "denied": 0, "rejected": 0, "rejection": 0, "no": 0,
            "waitlist": 2, "waitlisted": 2, "wl": 2,
        }
        df["admitted"] = (
            df["decision_raw"]
            .str.lower()
            .str.strip()
            .map(decision_map)
        )
        # For binary classification, drop waitlists (keep as optional extension)
        df = df[df["admitted"].isin([0, 1])].copy()

    df["dataset_source"] = "undergrad"
    print(f"[data_loader] College admissions: {len(df)} records, columns: {list(df.columns)}")
    return df


def merge_datasets(grad_df: pd.DataFrame, undergrad_df: pd.DataFrame) -> pd.DataFrame:
    #Merge both datasets into a one df. Columns not in one source are filled with NaN.

    merged = pd.concat([grad_df, undergrad_df], ignore_index=True, sort=False)
    print(f"[data_loader] Merged dataset: {len(merged)} records")
    return merged


def load_all_data(grad_path: str = None, undergrad_path: str = None) -> pd.DataFrame:
   
    grad = load_graduate_admissions(grad_path)
    undergrad = load_college_admissions(undergrad_path)
    return merge_datasets(grad, undergrad)


if __name__ == "__main__":
    df = load_all_data()
    print(df.head())
    print(df.dtypes)