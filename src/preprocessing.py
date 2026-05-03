"""
Data cleaning, standardization, and encoding pipeline.

  - Standardize GPA to 4.0 scale
  - Convert SAT/ACT to unified percentile rank
  - Median imputation for extracurriculars by school tier
  - Encode categorical variables
  - Map school tiers to ordinal labels
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

SAT_PERCENTILES = {
    1600: 99, 1550: 99, 1500: 98, 1450: 96, 1400: 94,
    1350: 91, 1300: 87, 1250: 81, 1200: 74, 1150: 67,
    1100: 58, 1050: 49, 1000: 40, 950: 31, 900: 23,
    850: 16, 800: 10, 750: 5,  700: 2,
}

ACT_PERCENTILES = {
    36: 100, 35: 99, 34: 99, 33: 98, 32: 97, 31: 96, 30: 94,
    29: 92, 28: 90, 27: 87, 26: 83, 25: 79, 24: 74, 23: 68,
    22: 62, 21: 56, 20: 49, 19: 42, 18: 35, 17: 29, 16: 22,
    15: 16, 14: 10, 13: 6,
}

TIER_MAP = {
    # Numeric (graduate dataset uses 1-5)
    1: "Ivy", 2: "Highly Selective", 3: "Selective",
    4: "Less Selective", 5: "Less Selective",
    # String labels (undergrad dataset)
    "ivy": "Ivy", "t20": "Ivy", "top 20": "Ivy",
    "highly selective": "Highly Selective", "t50": "Highly Selective", "top 50": "Highly Selective",
    "selective": "Selective", "t100": "Selective",
    "less selective": "Less Selective", "other": "Less Selective",
}

TIER_ORDER = ["Ivy", "Highly Selective", "Selective", "Less Selective"]
TIER_ORDINAL = {t: i for i, t in enumerate(TIER_ORDER)}


def standardize_gpa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "gpa_raw" not in df.columns:
        print("[preprocessing] Warning: 'gpa_raw' column not found. Skipping GPA standardization.")
        return df

    def _normalize(row):
        val = row["gpa_raw"]
        if pd.isna(val):
            return np.nan
        if row.get("dataset_source") == "graduate":
            return round(min(val / 2.5, 4.0), 2)
        elif val > 4.0:
            return round(min(val * 0.8, 4.0), 2)
        return round(val, 2)

    df["gpa"] = df.apply(_normalize, axis=1)
    df["gpa"] = df["gpa"].clip(0, 4.0)
    return df


def sat_to_percentile(score: float) -> float:
    if pd.isna(score):
        return np.nan
    score = int(round(score / 10) * 10)  # round to nearest 10
    if score >= 1600:
        return 99.0
    if score <= 700:
        return 1.0
    
    keys = sorted(SAT_PERCENTILES.keys())
    
    for i in range(len(keys) - 1):
        if keys[i] <= score <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            lo_p, hi_p = SAT_PERCENTILES[lo], SAT_PERCENTILES[hi]
            frac = (score - lo) / (hi - lo)
            return round(lo_p + frac * (hi_p - lo_p), 1)
    return np.nan


def act_to_percentile(score: float) -> float:
    if pd.isna(score):
        return np.nan
    score = int(round(score))
    return float(ACT_PERCENTILES.get(score, np.nan))


def standardize_test_scores(df: pd.DataFrame) -> pd.DataFrame:
    #Convert SAT and ACT scores to a one test_percentile col. takes the higher percentile.
    
    df = df.copy()
    sat_pct = pd.Series(np.nan, index=df.index)
    act_pct = pd.Series(np.nan, index=df.index)

    if "sat_score" in df.columns:
        sat_pct = df["sat_score"].apply(sat_to_percentile)
    if "act_score" in df.columns:
        act_pct = df["act_score"].apply(act_to_percentile)

    gre_pct = pd.Series(np.nan, index=df.index)
    if "gre_score" in df.columns:
        gre_pct = df["gre_score"].apply(
            lambda x: round(max(0, min(99, (x - 260) / (340 - 260) * 99)), 1)
            if not pd.isna(x) else np.nan
        )

    df["test_percentile"] = (
        pd.DataFrame({"sat": sat_pct, "act": act_pct, "gre": gre_pct})
        .max(axis=1)
    )
    return df


def standardize_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "school_tier_raw" not in df.columns:
        df["school_tier"] = "Unknown"
        return df

    def _map_tier(val):
        if pd.isna(val):
            return "Unknown"
        if isinstance(val, (int, float)):
            return TIER_MAP.get(int(val), "Less Selective")
        return TIER_MAP.get(str(val).strip().lower(), "Less Selective")

    df["school_tier"] = df["school_tier_raw"].apply(_map_tier)
    df["school_tier_ordinal"] = df["school_tier"].map(TIER_ORDINAL).fillna(3)
    return df


def impute_extracurriculars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing extracurricular scores using median imputation per school tier.
    """
    df = df.copy()
    if "extracurricular_raw" not in df.columns:
        df["extracurricular_score"] = np.nan
        return df

    # Attempt to coerce to numeric (some datasets use 1-10 scale already)
    df["extracurricular_score"] = pd.to_numeric(df["extracurricular_raw"], errors="coerce")

    # Clip to 0-10 range
    df["extracurricular_score"] = df["extracurricular_score"].clip(0, 10)

    # Median imputation by school tier
    tier_medians = df.groupby("school_tier")["extracurricular_score"].median()
    global_median = df["extracurricular_score"].median()

    def _impute(row):
        if not pd.isna(row["extracurricular_score"]):
            return row["extracurricular_score"]
        return tier_medians.get(row.get("school_tier", "Unknown"), global_median)

    df["extracurricular_score"] = df.apply(_impute, axis=1)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns as binary or ordinal integers.
    """
    df = df.copy()

    # Binary flags
    bool_cols = {
        "is_legacy": ["yes", "true", "1", 1],
        "is_first_gen": ["yes", "true", "1", 1],
        "is_instate": ["yes", "true", "1", 1],
        "has_research": [1, "1", "yes", "true"],
    }
    for col, truthy in bool_cols.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: 1 if str(x).strip().lower() in [str(t).lower() for t in truthy] else (0 if not pd.isna(x) else 0)
            ).astype(int)
        else:
            df[col] = 0

    # Major → STEM flag
    stem_keywords = ["computer", "engineering", "biology", "chemistry", "physics",
                     "math", "statistics", "neuroscience", "data", "cs", "ee", "me", "bme"]
    if "intended_major" in df.columns:
        df["is_stem"] = df["intended_major"].str.lower().str.contains(
            "|".join(stem_keywords), na=False
        ).astype(int)
    else:
        df["is_stem"] = 0

    return df


def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows missing both GPA and test scores — not enough signal.
    Drop rows missing the target label.
    """
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=["admitted"])
    df = df[~(df["gpa"].isna() & df["test_percentile"].isna())]
    print(f"[preprocessing] Dropped {before - len(df)} unusable rows. Remaining: {len(df)}")
    return df


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Call this after data_loader.load_all_data().
    Returns a clean, feature-ready DataFrame.
    """
    print("[preprocessing] Starting preprocessing pipeline...")
    df = standardize_gpa(df)
    df = standardize_test_scores(df)
    df = standardize_tiers(df)
    df = impute_extracurriculars(df)
    df = encode_categoricals(df)
    df = drop_unusable_rows(df)

    # Fill remaining numeric NaNs with column medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    print(f"[preprocessing] Done. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    from data_loader import load_all_data
    raw = load_all_data()
    clean = run_preprocessing(raw)
    print(clean[["gpa", "test_percentile", "school_tier", "admitted"]].head(10))