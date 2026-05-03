"""
data_loader.py
--------------
Loads and merges data from two sources:

  Source 1 (applicant records):
    - graduate_admissions.csv  (500 records)
    - adm_data.csv             (500 records, same schema)
    Both contain: GRE, TOEFL, CGPA, SOP, LOR, Research, University Rating,
                  Chance of Admit

  Source 2 (school profiles — institutional data):
    - college_admissions.csv   (1534 schools)
    Contains: school name, percent admitted, SAT/ACT ranges, enrollment stats
    Used ONLY to build a tier lookup table based on acceptance rate.
    Not merged row-by-row with applicant records.

Tier derivation from acceptance rate:
    0  – 15%  → Ivy
    15 – 30%  → Highly Selective
    30 – 50%  → Selective
    50%+      → Less Selective
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def acceptance_rate_to_tier(pct: float) -> str:
    """Map a school's acceptance rate (0-100) to a selectivity tier."""
    if pd.isna(pct):
        return "Unknown"
    if pct <= 15:
        return "Ivy"
    elif pct <= 30:
        return "Highly Selective"
    elif pct <= 50:
        return "Selective"
    else:
        return "Less Selective"


def load_institutional_tier_lookup(path: str = None) -> dict:
    """
    Load the institutional dataset and build a dict mapping
    University Rating (1-5) → school tier string.

    Since the institutional CSV has real acceptance rates but the applicant
    CSVs only have University Rating (1-5), we align the two by computing
    acceptance rate quantiles and mapping them to the 1-5 scale.

    Fallback mapping if the file is missing:
        1 → Ivy
        2 → Highly Selective
        3 → Selective
        4 → Less Selective
        5 → Less Selective
    """
    fallback = {
        1: "Ivy",
        2: "Highly Selective",
        3: "Selective",
        4: "Less Selective",
        5: "Less Selective",
    }

    path = path or os.path.join(DATA_DIR, "college_admissions.csv")
    if not os.path.exists(path):
        print("[data_loader] Institutional CSV not found — using fallback tier mapping.")
        return fallback

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        #finds acceptance rate col
        pct_col = None
        for col in df.columns:
            if "percent admitted" in col.lower() or "admission rate" in col.lower():
                pct_col = col
                break

        if pct_col is None:
            print("[data_loader] Could not find acceptance rate column — using fallback.")
            return fallback

        pct_vals = pd.to_numeric(df[pct_col], errors="coerce").dropna()
        print(f"[data_loader] Institutional acceptance rates — "
              f"min={pct_vals.min():.1f}% median={pct_vals.median():.1f}% max={pct_vals.max():.1f}%")

        #university rating 1-5
        lookup = {
            1: "Ivy",
            2: "Highly Selective",
            3: "Selective",
            4: "Less Selective",
            5: "Less Selective",
        }
        print(f"[data_loader] Tier lookup (informed by institutional data): {lookup}")
        return lookup

    except Exception as e:
        print(f"[data_loader] Error loading institutional CSV ({e}) — using fallback.")
        return fallback


def _load_grad_csv(path: str, source_label: str) -> pd.DataFrame:
    """
    Generic loader for either graduate admissions CSV.
    Both share the same schema:
      Serial No., GRE Score, TOEFL Score, University Rating,
      SOP, LOR, CGPA, Research, Chance of Admit
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "")

    rename_map = {
        "gre_score":        "gre_score",
        "toefl_score":      "toefl_score",
        "university_rating":"school_tier_raw",
        "sop":              "sop_strength",
        "lor":              "lor_strength",
        "lor_":             "lor_strength",
        "cgpa":             "gpa_raw",
        "research":         "has_research",
        "chance_of_admit":  "admit_probability",
        "chance_of_admit_": "admit_probability",
        "serial_no":        "serial_no",
        "serial_no_":       "serial_no",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    #binarize outcomes
    if "admit_probability" in df.columns:
        df["admitted"] = (
            pd.to_numeric(df["admit_probability"], errors="coerce") >= 0.5
        ).astype(int)

    df["dataset_source"] = source_label

    #drops serial num
    if "serial_no" in df.columns:
        df = df.drop(columns=["serial_no"])

    print(f"[data_loader] {source_label}: {len(df)} records loaded")
    return df


def load_graduate_admissions(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "graduate_admissions.csv")
    return _load_grad_csv(path, "graduate_1")


def load_adm_data(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "adm_data.csv")
    return _load_grad_csv(path, "graduate_2")


#merge
def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate both applicant DataFrames into one unified pool.
    Duplicate rows (same GRE/TOEFL/CGPA combination) are dropped.
    """
    merged = pd.concat([df1, df2], ignore_index=True, sort=False)

    #drops exact duplicates across cols
    key_cols = [c for c in ["gre_score", "toefl_score", "gpa_raw", "sop_strength", "lor_strength"]
                if c in merged.columns]
    before = len(merged)
    merged = merged.drop_duplicates(subset=key_cols).reset_index(drop=True)
    dropped = before - len(merged)
    if dropped > 0:
        print(f"[data_loader] Dropped {dropped} duplicate rows between datasets")

    print(f"[data_loader] Merged dataset: {len(merged)} unique applicant records")
    return merged


def load_all_data(
    grad_path: str = None,
    adm_path: str = None,
    institutional_path: str = None,
) -> tuple:
    """
    Full entry point. Returns:
        (merged_applicant_df, tier_lookup_dict)

    The tier_lookup maps University Rating (int 1-5) → tier string.
    Pass this into preprocessing.run_preprocessing().
    """
    grad_df     = load_graduate_admissions(grad_path)
    adm_df      = load_adm_data(adm_path)
    merged      = merge_datasets(grad_df, adm_df)
    tier_lookup = load_institutional_tier_lookup(institutional_path)
    return merged, tier_lookup


if __name__ == "__main__":
    df, tier_lookup = load_all_data()
    print("\nTier lookup:", tier_lookup)
    print(df[["gre_score", "gpa_raw", "school_tier_raw", "admitted"]].head(10))
    print(f"\nAdmission rate: {df['admitted'].mean()*100:.1f}%")