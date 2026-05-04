# College Admission Outcome Predictor

**CS 210: Data Management for Data Science**
**Partners:** Nikitha Cherukupalli and Varunavi Krishna

---

## Project Overview

The college admissions process in the United States is a stressful and opaque experience, especially for students without access to expensive counseling services. This project builds a machine learning pipeline that predicts the likelihood of admission to a university based on a student's academic profile, extracurricular involvement, and demographic background.

By training on anonymized, self-reported admissions outcomes from Kaggle, the pipeline identifies which features most strongly correlate with acceptance decisions — providing data-driven transparency to prospective applicants who lack institutional support.

### Problem
Students with nearly identical credentials often receive inconsistent results across schools with similar admissions rates. Existing prediction tools (Naviance, CollegeVine) are proprietary and inaccessible. This project builds an open, reproducible alternative.

### Approach
The pipeline combines two Kaggle datasets into a unified schema, engineers four custom features, and trains two classifiers stratified by school selectivity tier. All experiments are logged with MLflow for full reproducibility.

### Engineered Features
- **Academic Strength Index (ASI)** — composite of GPA percentile, test score percentile, and course rigor
- **School Selectivity Gap** — difference between an applicant's ASI and their target school tier's median admitted profile (reach vs. match indicator)
- **Extracurricular Score** — normalized proxy derived from SOP and LOR strength ratings
- **Demographic Flags** — research experience, legacy status, first-gen status, STEM major, in-state status

### Models
- **Logistic Regression** — interpretable baseline; coefficients show which features drive admission odds
- **Random Forest** — ensemble model capturing non-linear feature interactions; feature importances extracted and visualized
- Both models trained globally and stratified per school tier (Ivy, Highly Selective, Selective, Less Selective)

### Results
Both models exceeded the project success criteria of AUC-ROC ≥ 0.70 and F1 ≥ 0.70:

| Model | AUC-ROC | F1 Score |
|---|---|---|
| Logistic Regression | 0.923 | 0.874 |
| Random Forest | 0.937 | 0.979 |

---

## Project Structure

```
college_admission_predictor/
├── data/                        # Kaggle CSVs (not tracked in git — see setup below)
├── src/
│   ├── data_loader.py           # Loads and merges both datasets into unified schema
│   ├── preprocessing.py         # GPA standardization, test score percentiles, encoding
│   ├── feature_engineering.py   # Builds ASI, selectivity gap, EC score, demographic flags
│   ├── train.py                 # Model training + MLflow logging, global + per-tier
│   ├── evaluate.py              # AUC-ROC, F1, per-tier breakdown, feature importances
│   └── visualize.py             # 8 publication-quality plots saved to /visualizations/
├── models/                      # Saved .pkl model files (not tracked in git)
├── visualizations/              # Saved PNG plots (not tracked in git)
├── results/                     # JSON + CSV evaluation outputs (not tracked in git)
├── main.py                      # Full pipeline runner — runs all 6 steps end to end
├── requirements.txt
└── README.md
```

---

## Setup and Run Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd college_admission_predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Datasets
Download the following CSVs from Kaggle and place them in a `/data/` folder:

| File | Source |
|---|---|
| `graduate_admissions.csv` | [Graduate Admissions Dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) |
| `adm_data.csv` | [Admission in the University](https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university) |
| `college_admissions.csv` | [College Admissions (institutional)](https://www.kaggle.com/datasets/samsonqian/college-admissions) |

```
data/
├── graduate_admissions.csv
├── adm_data.csv
└── college_admissions.csv
```

### 4. Run the Full Pipeline
```bash
python main.py
```

This runs all 6 steps in sequence: data loading → preprocessing → feature engineering → training → evaluation → visualization. Outputs are saved to `/models/`, `/results/`, and `/visualizations/`.

### 5. Skip Retraining (if models already exist)
```bash
python main.py --skip-train
```

### 6. Use Custom Data Paths
```bash
python main.py --grad-path path/to/grad.csv --adm-path path/to/adm.csv --institutional-path path/to/schools.csv
```

### 7. View MLflow Experiment Runs
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser to see all logged runs, hyperparameters, and metrics.

---

## Outputs

After running the pipeline, the following are generated:

**`/results/`**
- `logistic_regression_evaluation.json` — AUC-ROC, F1, per-tier breakdown
- `random_forest_evaluation.json` — AUC-ROC, F1, per-tier breakdown
- `*_feature_importances.csv` — ranked feature importance scores

**`/visualizations/`**
- `1_data_distributions.png` — class imbalance and tier distribution
- `2_asi_distribution.png` — ASI distributions by admission outcome
- `3_selectivity_gap.png` — selectivity gap by outcome and tier
- `4_correlation_heatmap.png` — feature correlation matrix
- `5_feature_importances.png` — side-by-side LR and RF importances
- `6_roc_curves_logistic_regression.png` — ROC curves per tier
- `6_roc_curves_random_forest.png` — ROC curves per tier
- `7_confusion_matrices.png` — confusion matrices on holdout test set
