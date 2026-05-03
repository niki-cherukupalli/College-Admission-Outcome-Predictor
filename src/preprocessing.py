"""
preprocessing.py
----------------
Data cleaning, standardization, and encoding pipeline.

Steps:
  1. Standardize GPA (CGPA on 10-pt scale → 4.0 scale)
  2. Convert GRE/TOEFL scores to unified test_percentile
  3. Map University Rating (1-5) → school tier string using tier_lookup
  4. Impute missing values
  5. Encode categorical/binary variables
"""

import pandas as pd
import numpy as np

TIER_ORDER = ["Ivy", "Highly Selective", "Selective", "Less Selective"]
TIER_ORDINAL = {t: i for i, t in enumerate(TIER_ORDER)}

#default tier lookup
DEFAULT_TIER_LOOKUP = {
    1: "Ivy",
    2: "Highly Selective",
    3: "Selective",
    4: "Less Selective",
    5: "Less Selective",
}


#gpa standardization
def standardize_gpa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CGPA (10-point scale) to 4.0 scale.
    Formula: gpa = CGPA / 2.5, capped at 4.0.
    """
    df = df.copy()
    if "gpa_raw" not in df.columns:
        print("[preprocessing] Warning: 'gpa_raw' not found. Skipping GPA standardization.")
        df["gpa"] = np.nan
        return df

    df["gpa"] = (
        pd.to_numeric(df["gpa_raw"], errors="coerce")
        .div(2.5)
        .clip(0, 4.0)
        .round(2)
    )
    return df


#test score to percentile
def gre_to_percentile(score: float) -> float:
    """
    Convert GRE total score (260-340) to approximate percentile.
    Linear mapping: 260 → 1st, 340 → 99th.
    """
    if pd.isna(score):
        return np.nan
    return round(max(1.0, min(99.0, (score - 260) / (340 - 260) * 98 + 1)), 1)


def toefl_to_percentile(score: float) -> float:
    """
    Convert TOEFL iBT score (0-120) to approximate percentile.
    Linear mapping: 0 → 1st, 120 → 99th.
    """
    if pd.isna(score):
        return np.nan
    return round(max(1.0, min(99.0, score / 120 * 98 + 1)), 1)


def standardize_test_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unified test_percentile from GRE and TOEFL.
    Takes the average of both when both are present.
    """
    df = df.copy()
    gre_pct  = pd.Series(np.nan, index=df.index)
    toefl_pct = pd.Series(np.nan, index=df.index)

    if "gre_score" in df.columns:
        gre_pct = df["gre_score"].apply(gre_to_percentile)
    if "toefl_score" in df.columns:
        toefl_pct = df["toefl_score"].apply(toefl_to_percentile)

    combined = pd.DataFrame({"gre": gre_pct, "toefl": toefl_pct})
    df["test_percentile"] = combined.mean(axis=1, skipna=True).round(2)
    return df


#school tier mapping
def standardize_tiers(df: pd.DataFrame, tier_lookup: dict = None) -> pd.DataFrame:
    """
    Map University Rating (1-5) → school tier string → ordinal int.
    Uses tier_lookup dict built from institutional data in data_loader.
    """
    df = df.copy()
    lookup = tier_lookup or DEFAULT_TIER_LOOKUP

    if "school_tier_raw" not in df.columns:
        df["school_tier"] = "Less Selective"
        df["school_tier_ordinal"] = 3
        return df

    def _map(val):
        if pd.isna(val):
            return "Less Selective"
        try:
            return lookup.get(int(float(val)), "Less Selective")
        except (ValueError, TypeError):
            return "Less Selective"

    df["school_tier"] = df["school_tier_raw"].apply(_map)
    df["school_tier_ordinal"] = df["school_tier"].map(TIER_ORDINAL).fillna(3).astype(int)
    return df


#extracuricular imputation
def impute_extracurriculars(df: pd.DataFrame) -> pd.DataFrame:
    """
    This dataset doesn't have extracurricular data, so we derive a proxy
    from SOP and LOR strength (both rated 1-5 in the dataset).
    Extracurricular score = average of SOP and LOR, normalized to 0-10.
    """
    df = df.copy()
    sop = pd.to_numeric(df.get("sop_strength", pd.Series(np.nan, index=df.index)), errors="coerce")
    lor = pd.to_numeric(df.get("lor_strength", pd.Series(np.nan, index=df.index)), errors="coerce")

    #avg SOP and LOR
    proxy = pd.DataFrame({"sop": sop, "lor": lor}).mean(axis=1, skipna=True)
    df["extracurricular_score"] = (proxy * 2).round(2)  # scale 1-5 → 2-10

    max_ec = df["extracurricular_score"].max()
    df["extracurricular_score_norm"] = (
        (df["extracurricular_score"] / max_ec).round(4) if max_ec > 0 else 0.0
    )
    return df


#categorical encoding
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary flags. Since this dataset doesn't have legacy/first-gen/
    major/in-state columns, those default to 0. has_research is present.
    """
    df = df.copy()

    if "has_research" in df.columns:
        df["has_research"] = pd.to_numeric(df["has_research"], errors="coerce").fillna(0).astype(int)
    else:
        df["has_research"] = 0

    for col in ["is_legacy", "is_first_gen", "is_stem", "is_instate"]:
        df[col] = 0

    return df


#drop unusable rows
def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing both the target label and core academic features."""
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=["admitted"])
    df = df[~(df["gpa"].isna() & df["test_percentile"].isna())]
    print(f"[preprocessing] Dropped {before - len(df)} unusable rows. Remaining: {len(df)}")
    return df


#pipeline
def run_preprocessing(df: pd.DataFrame, tier_lookup: dict = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Call with the merged DataFrame and tier_lookup from data_loader.load_all_data().
    """
    print("[preprocessing] Starting preprocessing pipeline...")
    df = standardize_gpa(df)
    df = standardize_test_scores(df)
    df = standardize_tiers(df, tier_lookup)
    df = impute_extracurriculars(df)
    df = encode_categoricals(df)
    df = drop_unusable_rows(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    print(f"[preprocessing] Done. Final shape: {df.shape}")
    print(f"[preprocessing] Tier distribution:\n{df['school_tier'].value_counts().to_string()}")
    print(f"[preprocessing] Admission rate: {df['admitted'].mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_all_data
    df, tier_lookup = load_all_data()
    clean = run_preprocessing(df, tier_lookup)
    print(clean[["gpa", "test_percentile", "school_tier", "admitted"]].head(10))