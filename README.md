<div align="center">

# 🏛️ Building Insurance Claim Prediction

### An End-to-End Binary Classification Project

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

[![GitHub](https://img.shields.io/badge/GitHub-cssadewale-181717?style=for-the-badge&logo=github)](https://github.com/cssadewale)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adewale%20Samson%20Adeagbo-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/adewalesamsonadeagbo)

---

### 🏆 Final Capstone Project — AINOW Bootcamp
**Organised by [The Incubator Hub](https://theincubatorhub.com)**
📅 **9 January 2026 – 23 January 2026** (2-week intensive)

</div>

---

## 📌 Project Overview

As the **Lead Data Analyst**, the task was to build a predictive model that determines whether a building will have an insurance claim during a given observation period, based on its structural and environmental characteristics.

**Target Variable — `claim`:**

| Value | Meaning |
|-------|---------|
| `1` | The building has **at least one claim** over the insured period |
| `0` | The building has **no claim** over the insured period |

This project was the **final capstone submission** for the AINOW Bootcamp organised by The Incubator Hub, covering a 2-week intensive period from 9 January 2026 to 23 January 2026. It demonstrates the complete data science lifecycle — from raw data through cleaning, EDA, feature engineering, model training, hyperparameter tuning, SHAP interpretability, and a live Streamlit web application deployed on the cloud.

---

## 🏆 Bootcamp Context

| Detail | Info |
|--------|------|
| 🎓 **Programme** | AINOW Bootcamp — Final Capstone Project |
| 🏢 **Organiser** | The Incubator Hub |
| 📅 **Start Date** | Friday, 9 January 2026 — 11:30 AM |
| 📅 **End Date** | Friday, 23 January 2026 — 12:00 AM |
| ⏱️ **Duration** | 2-week intensive |
| 🎯 **Role** | Lead Data Analyst |
| 📊 **Domain** | Insurance Risk Analytics / Binary Classification |

---

## 🗂️ Repository Structure

```
insurance-claim-prediction/
│
├── Insurance_Claim_Prediction.ipynb     ← Full end-to-end notebook (portfolio-ready)
├── app.py                               ← Streamlit web application (3 pages)
├── requirements.txt                     ← App-only Python dependencies
├── runtime.txt                          ← Pins Python 3.11 for Streamlit Cloud
├── README.md                            ← This file
│
└── assets/                              ← All EDA and model visualisations
    ├── target_distribution.png          ← Class balance (76.82% vs 23.18%)
    ├── insured_period_distribution.png
    ├── building_dimension_distribution.png
    ├── number_of_windows_distribution.png
    ├── building_age_distribution.png
    ├── categorical_distributions.png
    ├── correlation_heatmap.png
    ├── numerical_vs_claim.png           ← Box plots by claim status
    ├── categorical_vs_claim.png
    ├── transformations_before_after.png ← Skewness reduction visualised
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── feature_importance_rf.png
    ├── shap_summary_dot.png             ← Beeswarm: direction + magnitude
    ├── shap_summary_bar.png
    ├── shap_dependence.png
    └── shap_waterfall.png               ← Individual prediction explanation
```

> 📦 **Model file:** `best_random_forest_model.joblib` exceeds GitHub's 25 MB limit and is hosted on Google Drive. The app downloads it automatically on first launch via `gdown` — no manual steps needed.

---

## 🔄 Complete Project Workflow

| Step | Phase | What Was Done |
|------|-------|---------------|
| **1** | Basic Data Exploration | Loaded 7,160 records × 14 columns; standardised all column names; dropped `customer_id` and `geo_code`; corrected 4 data types; cleaned `number_of_windows` in 3 sequential steps (strip whitespace → replace `'.'` with `'0'` → replace `'>=10'` with `'10'` → cast to `int64`); imputed 3 missing-value columns using mode; removed 146 duplicate rows → **7,014 clean records** |
| **2** | Exploratory Data Analysis | Full univariate (target, numerical, categorical), bivariate (correlations, contingency tables, grouped statistics), and multivariate analysis; confirmed 3.3:1 class imbalance; identified `building_dimension` as top predictor; confirmed near-perfect triple redundancy of `building_fenced` / `garden` / `settlement` |
| **3** | Feature Engineering | Engineered `building_age` from datetime subtraction; applied `log1p` transformation to `building_dimension` and `building_age`; applied reflected `log1p` to `insured_period`; dropped 4 redundant date-derived columns; one-hot encoded 4 categorical features (`drop_first=True`) |
| **4** | Data Preprocessing | 80/20 stratified split (preserving 76.82%/23.18% ratio) → SMOTE applied to training data only (5,611 → 8,620 samples, 50/50 balance) → StandardScaler (fit on training only, transform both sets) |
| **5** | Model Development | Trained Logistic Regression, Decision Tree, and Random Forest; compared 5 metrics; tuned Random Forest with GridSearchCV (216 combinations × 5 folds = 1,080 fits); validated with 5-fold cross-validation (mean F1 = 0.7921); interpreted with SHAP (TreeExplainer, beeswarm, bar, dependence, and waterfall plots) |
| **6** | Model Deployment | Saved model with `joblib`; hosted on Google Drive (48.8 MB); deployed Streamlit app on Streamlit Cloud with automatic model download; resolved 3 deployment issues (Python version, feature count, sklearn compatibility) |
| **7** | Business Insights | 6 actionable recommendations across risk pricing, tiered inspection, portfolio strategy, and maintenance incentives |

---

## 📊 Key EDA Findings

<div align="center">

### Target Variable Distribution
![Target Distribution](assets/target_distribution.png)

### Correlation Heatmap — Numerical Features
![Correlation Heatmap](assets/correlation_heatmap.png)

### Numerical Features vs Claim Status
![Numerical vs Claim](assets/numerical_vs_claim.png)

### Feature Transformation — Before vs After
![Transformations](assets/transformations_before_after.png)

</div>

**Summary of critical findings:**

- **Class imbalance:** 76.82% No Claim vs 23.18% Claim (3.3:1 ratio) — a naïve always-predict-0 classifier achieves 76.82% accuracy by default. Addressed with SMOTE on training data only, after the train-test split.
- **Top predictor:** `building_dimension` — claim buildings have a median size **2.2× larger** (1,995 m² vs 900 m²)
- **Second predictor:** `building_age` — older buildings claim more frequently due to accumulated wear and material degradation
- **Window signal:** Claim buildings have a median of **3 windows** vs 1 for non-claim buildings — a 3× difference in median
- **Urban/rural triple redundancy:** `building_fenced`, `garden`, and `settlement` are near-perfectly correlated (contingency analysis confirmed). All three show identical 25.0% (rural) vs 21.2% (urban) claim rates — they encode one risk dimension, not three independent signals
- **Building type gradient:** Type 4 → highest claim rate; Type 1 → lowest — a consistent ordering across all analyses
- **Outlier decision:** 533 large-building outliers retained — they represent a genuine high-risk segment, not data errors. Log transformation addresses their distributional impact

---

## 🤖 Model Performance

<div align="center">

### ROC Curves — All Models
![ROC Curves](assets/roc_curves.png)

### Confusion Matrices — All Models
![Confusion Matrices](assets/confusion_matrices.png)

### Precision-Recall Curves
![PR Curves](assets/precision_recall_curves.png)

</div>

### Test Set Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.6636 | 0.3527 | 0.5415 | 0.4272 | 0.6629 |
| Decision Tree | 0.6536 | 0.2952 | 0.3569 | 0.3231 | 0.5498 |
| **Tuned Random Forest ✅** | **0.7106** | **0.3608** | 0.3231 | 0.3409 | 0.6144 |

> ⚠️ **Why accuracy is not the primary metric:** A naïve classifier that always predicts "No Claim" achieves **76.82% accuracy** without learning a single pattern — because 76.82% of buildings never filed a claim. We rely on **Precision, Recall, F1-Score, and ROC-AUC** throughout.

**5-Fold Cross-Validation (tuned model, training distribution):** Mean F1 = **0.7921 ± 0.1063**

**Best hyperparameters (GridSearchCV — 216 combinations, 1,080 total fits):**

```
n_estimators      : 200
max_depth         : 20
max_features      : sqrt
min_samples_split : 2
min_samples_leaf  : 1
```

---

## 🎯 Feature Importance & SHAP Analysis

<div align="center">

### Random Forest Feature Importance (Gini Impurity Reduction)
![Feature Importance](assets/feature_importance_rf.png)

### SHAP Summary — Beeswarm Plot (Direction + Magnitude per Feature)
![SHAP Beeswarm](assets/shap_summary_dot.png)

### SHAP Summary — Global Bar Plot (Mean |SHAP Value|)
![SHAP Bar](assets/shap_summary_bar.png)

### SHAP Dependence Plot — Top Feature
![SHAP Dependence](assets/shap_dependence.png)

### SHAP Waterfall — Individual Prediction Explanation
![SHAP Waterfall](assets/shap_waterfall.png)

</div>

**The model was trained on 13 features** — `building_dimension`, `building_age`, and `number_of_windows` are the dominant drivers:

| Rank | Feature | Role |
|------|---------|------|
| 1 | `building_dimension` | Strongest — claim buildings 2.2× larger in median size |
| 2 | `building_age` | Second — older buildings claim more frequently |
| 3 | `number_of_windows` | Third — claim buildings have 3× the median window count |
| 4 | `transformed_insured_period` | Policy duration signal |
| 5 | `insured_period` | Original policy duration |

SHAP confirms the direction: **high building dimension and high building age both push predictions toward Claim (1)** at the individual prediction level.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` |
| Imbalanced data | `imbalanced-learn` (SMOTE) |
| Interpretability | `shap` |
| Automated EDA | `ydata-profiling` |
| Model serialisation | `joblib` |
| Model hosting | Google Drive + `gdown` |
| Web application | `streamlit` |
| Environment | Python 3.11 (pinned via `runtime.txt`) |

---

## 🚀 Running the Streamlit App

### Option A — Live App (Streamlit Cloud)

👉 **[Open the live app](https://adewale-insurance-claim-prediction.streamlit.app)**

---

### Option B — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/cssadewale/insurance-claim-prediction.git
cd insurance-claim-prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. The model downloads automatically from Google Drive on first launch.

---

## 📓 Running the Notebook

The notebook is self-contained and designed for **Google Colab** — datasets download automatically via `gdown`.

1. Open `Insurance_Claim_Prediction.ipynb` in Google Colab
2. Click **Runtime → Run all**
3. No manual file uploads needed

---

## 📈 Feature Engineering Summary

| Original Feature | Transformation | New Feature | Skewness Change |
|-----------------|---------------|-------------|----------------|
| `building_dimension` | `log1p(x)` | `log_building_dimension` | +3.12 → −0.20 |
| `insured_period` | `log1p(1 − x)` (reflected) | `transformed_insured_period` | −2.72 → flipped |
| `building_age` | `log1p(x)` | `log_building_age` | +2.28 → −1.52 |
| `year_of_observation`, `date_of_occupancy` | Datetime subtraction | `building_age` | New derived feature; originals dropped |
| `building_painted`, `building_fenced`, `garden`, `settlement` | One-Hot (`drop_first=True`) | 4 binary dummy columns | Required for ML models |

> **Note:** `log_building_dimension` and `log_building_age` are computed in the notebook for analysis purposes. The deployed model was trained on **13 features** — these two log columns were not included in the final training `X`. `transformed_insured_period` is included in the 13.

---

## 🔧 Deployment Journey & Issues Resolved

This section documents the three real-world deployment problems encountered and how each was solved — a reflection of genuine engineering practice.

### Issue 1 — `ydata-profiling` and `shap` not installable on Python 3.14
**Error:** `Could not find a version that satisfies the requirement ydata-profiling>=4.0.0`

**Cause:** Streamlit Cloud defaulted to Python 3.14. `ydata-profiling` has no wheel for Python 3.14. `shap` pulls in `llvmlite` which also fails to compile on 3.14.

**Fix 1:** Removed `ydata-profiling` and `shap` from `requirements.txt` — these are notebook-only dependencies; the app never imports them.

**Fix 2:** Added `runtime.txt` containing `3.11` (not `python-3.11`) to pin the environment to Python 3.11.

---

### Issue 2 — `ValueError: _check_feature_names` (feature name mismatch)
**Error:** `X does not have valid feature names, but RandomForestClassifier was fitted with feature names`

**Cause:** We were passing a named pandas DataFrame to `model.predict()`. Sklearn compared the DataFrame's column names against `model.feature_names_in_` and detected a mismatch.

**Fix:** Changed both prediction calls to use `.values` to pass a numpy array instead, bypassing the name check while preserving the correct column order:
```python
# Before
model.predict(feature_vector)

# After
model.predict(feature_vector.values)
```

---

### Issue 3 — `ValueError: X has 15 features, but RandomForestClassifier is expecting 13`
**Error:** `X has 15 features, but RandomForestClassifier is expecting 13 features as input.`

**Root cause identified from `model.n_features_in_ = 13`:** The model was saved at a point in the notebook where `log_building_dimension` and `log_building_age` had not yet been included in the training `X`. The model therefore only knows 13 features. The app was incorrectly passing 15.

**Fix:** Removed `log_building_dimension` and `log_building_age` from three places in `app.py`:
- `SCALER_MEANS` and `SCALER_STDS` (reduced from 7 entries to 5)
- `FEATURE_COLUMNS` (reduced from 15 to 13)
- The `raw` dictionary inside `engineer_features()`

---

## 💼 Business Recommendations

| # | Recommendation | Action |
|---|---------------|--------|
| 1 | **Risk-Based Premium Pricing** | Use `building_dimension`, `building_age`, and `number_of_windows` as primary rating factors |
| 2 | **Tiered Risk Classification** | Type 4 & 3 → high-risk tier (mandatory inspection, higher premium). Type 1 → preferential pricing |
| 3 | **Targeted Pre-Renewal Inspections** | Flag buildings scoring > 60% claim probability for pre-renewal physical inspection |
| 4 | **Residential Portfolio Review** | Conduct dedicated actuarial review — residential buildings claim at a disproportionately higher rate |
| 5 | **Maintenance Incentive Programme** | Offer premium discounts for verified maintenance records at renewal |
| 6 | **Early Warning at New Business** | Integrate model into application workflow — refer high-risk applicants for manual underwriter review |

---

## ⚠️ Limitations & Next Steps

| Limitation | Impact | Recommended Mitigation |
|-----------|--------|------------------------|
| Geographic granularity lost | `geo_code` was dropped — location risk (flood zones, fire station proximity) unaccounted for | Enrich with geocoded risk scores and hazard indices |
| Claims severity not modelled | Predicts *whether* a claim occurs, not *how costly* it will be | Build a companion regression model for claim amount |
| Temporal drift | Dataset covers 2012–2016 only | Annual retraining pipeline with PSI / KS-test drift monitoring |
| Binary target limitation | Minor and catastrophic claims both receive `claim = 1` | Multi-class or severity-weighted target variable |
| Scaler statistics hardcoded | If model is retrained, `SCALER_MEANS` and `SCALER_STDS` in `app.py` must be updated | Persist the scaler object alongside the model and load it in the app |

---

## 👤 Author

<div align="center">

**Adewale Samson Adeagbo**

*Educator · Data Practitioner*

*Learning deliberately. Teaching authentically.*

---

[![GitHub](https://img.shields.io/badge/GitHub-cssadewale-181717?style=for-the-badge&logo=github)](https://github.com/cssadewale)
&nbsp;&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adewale%20Samson%20Adeagbo-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/adewalesamsonadeagbo)

</div>

| Detail | Info |
|--------|------|
| 🎓 **Education** | B.Sc.(Ed) Computer Science Education — Lagos State University (2023) |
| 📚 **Teaching** | Mathematics, Further Mathematics, Chemistry, Physics — 10+ years, Lagos & Ogun State, Nigeria |
| 🏢 **Organisation** | HMG Concepts (His Marvellous Grace Educational Consult) — Visioner & Data Lead since 2015 |
| 🛠️ **Stack** | Python · Scikit-learn · Pandas · SHAP · SQL · Power BI · Tableau · Streamlit |
| 🌐 **Portfolio** | [cssadewale.github.io](https://cssadewale.github.io) |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

*Submitted as Final Capstone — AINOW Bootcamp · The Incubator Hub · January 2026*

*Built with 🧠 data science precision and 📐 mathematical rigour*

</div>
