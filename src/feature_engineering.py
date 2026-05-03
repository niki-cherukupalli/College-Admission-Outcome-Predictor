"""
feature_engineering.py
-----------------------
Builds all four engineered features described in the project proposal:

  1. Academic Strength Index (ASI)
       Composite of GPA percentile + test score percentile + course rigor (AP/IB)

  2. School Selectivity Gap
       Difference between applicant's ASI and the target school tier's
       median admitted student ASI (reach vs. match predictor)

  3. Extracurricular Score
       Already built in preprocessing; normalized here to 0-1 range

  4. Demographic Flags
       First-gen, legacy, STEM major, in-state (binary; built in preprocessing)
"""

import pandas as pd
import numpy as np

# Tier median ASI benchmarks (estimated from public admissions data)
# Used to compute the School Selectivity Gap
TIER_MEDIAN_ASI = {
    "Ivy":              92.0,
    "Highly Selective": 82.0,
    "Selective":        68.0,
    "Less Selective":   52.0,
    "Unknown":          60.0,
}

# AP/IB course rigor bonus (mapped from course_rigor column if present, else 0)
MAX_RIGOR_BONUS = 10.0  # maximum points added to ASI from course rigor


def compute_gpa_percentile(df: pd.DataFrame) -> pd.Series:
    """
    Convert GPA (0-4.0 scale) to a 0-100 percentile using a
    piecewise linear approximation based on national GPA distributions.
    """
    def _gpa_to_pct(gpa):
        if pd.isna(gpa):
            return np.nan
        # Piecewise: 4.0=99, 3.7=93, 3.5=85, 3.3=75, 3.0=60, 2.7=45, 2.5=35, 2.0=20
        breakpoints = [(4.0, 99), (3.7, 93), (3.5, 85), (3.3, 75),
                       (3.0, 60), (2.7, 45), (2.5, 35), (2.0, 20), (0.0, 1)]
        for i in range(len(breakpoints) - 1):
            hi_gpa, hi_pct = breakpoints[i]
            lo_gpa, lo_pct = breakpoints[i + 1]
            if lo_gpa <= gpa <= hi_gpa:
                frac = (gpa - lo_gpa) / (hi_gpa - lo_gpa)
                return round(lo_pct + frac * (hi_pct - lo_pct), 1)
        return 1.0

    return df["gpa"].apply(_gpa_to_pct)


def compute_rigor_bonus(df: pd.DataFrame) -> pd.Series:
    """
    Compute a course rigor bonus (0 to MAX_RIGOR_BONUS).
    If a 'course_rigor' or 'ap_courses' column exists, normalize it.
    Otherwise returns a series of zeros.
    """
    for col in ["course_rigor", "ap_courses", "num_ap", "ap_ib"]:
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce").fillna(0)
            # Normalize: assume max of 10 AP/IB courses = full bonus
            return (raw.clip(0, 10) / 10 * MAX_RIGOR_BONUS).round(2)
    return pd.Series(0.0, index=df.index)


def build_academic_strength_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: Academic Strength Index (ASI)

    ASI = 0.45 * gpa_percentile + 0.45 * test_percentile + 0.10 * rigor_bonus

    Weights reflect typical admissions emphasis on academics (90%)
    with a modest course rigor signal (10%).
    Range: 0 - 100
    """
    df = df.copy()
    gpa_pct = compute_gpa_percentile(df)
    test_pct = df["test_percentile"].fillna(df["test_percentile"].median())
    rigor = compute_rigor_bonus(df)

    df["gpa_percentile"] = gpa_pct
    df["asi"] = (
        0.45 * gpa_pct.fillna(gpa_pct.median()) +
        0.45 * test_pct +
        0.10 * rigor
    ).round(2)

    print(f"[feature_eng] ASI stats: mean={df['asi'].mean():.1f}, std={df['asi'].std():.1f}")
    return df


def build_selectivity_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: School Selectivity Gap

    gap = applicant_ASI - tier_median_ASI

    Positive gap → applicant is above the school's typical admit profile (safety/match)
    Negative gap → applicant is below the profile (reach)
    """
    df = df.copy()
    tier_median = df["school_tier"].map(TIER_MEDIAN_ASI).fillna(60.0)
    df["selectivity_gap"] = (df["asi"] - tier_median).round(2)
    print(f"[feature_eng] Selectivity gap: mean={df['selectivity_gap'].mean():.1f}")
    return df


def normalize_extracurricular_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: Normalize extracurricular score to 0-1 range for
    consistent scale with other features.
    """
    df = df.copy()
    if "extracurricular_score" in df.columns:
        max_ec = df["extracurricular_score"].max()
        if max_ec > 0:
            df["extracurricular_score_norm"] = (
                df["extracurricular_score"] / max_ec
            ).round(4)
        else:
            df["extracurricular_score_norm"] = 0.0
    else:
        df["extracurricular_score_norm"] = 0.0
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns DataFrame with all engineered columns added.

    Final feature set used for modeling:
      - asi                      (Academic Strength Index)
      - gpa_percentile           (Component of ASI)
      - test_percentile          (Standardized test score percentile)
      - selectivity_gap          (ASI vs. school tier median)
      - extracurricular_score_norm
      - is_legacy                (Binary demographic flag)
      - is_first_gen             (Binary demographic flag)
      - is_stem                  (Binary: STEM vs humanities major)
      - is_instate               (Binary demographic flag)
      - has_research             (Binary: research experience)
      - school_tier_ordinal      (0=Ivy ... 3=Less Selective)
      - sop_strength             (Statement of Purpose — graduate only)
      - lor_strength             (Letter of Recommendation — graduate only)
    """
    print("[feature_eng] Building all features...")
    df = build_academic_strength_index(df)
    df = build_selectivity_gap(df)
    df = normalize_extracurricular_score(df)
    print(f"[feature_eng] Done. Shape: {df.shape}")
    return df


# ── Final feature column list for model input ──────────────────────────────
FEATURE_COLS = [
    "asi",
    "gpa_percentile",
    "test_percentile",
    "selectivity_gap",
    "extracurricular_score_norm",
    "is_legacy",
    "is_first_gen",
    "is_stem",
    "is_instate",
    "has_research",
    "school_tier_ordinal",
    "sop_strength",
    "lor_strength",
]

TARGET_COL = "admitted"


def get_feature_matrix(df: pd.DataFrame):
    """
    Return X (feature matrix) and y (target vector).
    Only include columns that exist in the DataFrame.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[feature_eng] Note: columns not in data (filled with 0): {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[FEATURE_COLS].copy().fillna(0)
    y = df[TARGET_COL].copy()
    return X, y


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_all_data
    from preprocessing import run_preprocessing

    raw = load_all_data()
    clean = run_preprocessing(raw)
    featured = build_all_features(clean)
    X, y = get_feature_matrix(featured)
    print(X.head())
    print(f"\nFeature matrix: {X.shape}, Target: {y.value_counts().to_dict()}")