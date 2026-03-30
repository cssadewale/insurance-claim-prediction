# 🏛️ Building Insurance Claim Prediction
### End-to-End Binary Classification | Machine Learning Portfolio Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cssadewale/insurance-claim-prediction/blob/main/Insurance_Claim_Prediction_COMPLETE_FINAL.ipynb)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cssadewale-insurance-claim.streamlit.app)

---

## 📌 Project Overview

This project builds a **predictive model** to determine whether a building will file an insurance claim during a given observation period. As Lead Data Analyst, the objective is to use building characteristics to predict claim likelihood — providing the insurer with a data-driven tool for smarter underwriting, risk pricing, and resource allocation.

**Target Variable:**
- `1` → The building filed **at least one claim** during the insured period
- `0` → The building filed **no claim** during the insured period

---

## 🎯 Business Problem

Insurance companies face two key challenges:
1. **Adverse selection** — high-risk buildings may be priced at low-risk premiums
2. **Inefficient inspections** — blanket checks waste resources that should be focused on high-risk properties

A reliable claim prediction model solves both by scoring each building at the point of application or renewal.

---

## 📂 Repository Structure

```
insurance-claim-prediction/
│
├── Insurance_Claim_Prediction_COMPLETE_FINAL.ipynb   ← Main project notebook
├── app.py                                            ← Streamlit web application
├── README.md                                         ← This file
├── requirements.txt                                  ← Python dependencies
└── .gitignore                                        ← Files excluded from Git
```

> **Note on the trained model file:**
> The saved model (`best_random_forest_model.joblib`) exceeds GitHub's 25 MB file size limit and is therefore **not stored in this repository**. It is hosted on Google Drive and downloaded automatically at runtime. See [How to Run](#️-how-to-run-this-project) below for details.

---

## 🔄 Project Workflow

| Step | Phase | Description |
|------|-------|-------------|
| **1** | Basic Data Exploration | Data loading, inspection, cleaning |
| **2** | Exploratory Data Analysis | Univariate, Bivariate analysis |
| **3** | Feature Engineering & Preprocessing | Transformation, encoding, scaling |
| **4** | Model Development | Training, evaluation, tuning, interpretability |
| **5** | Business Insights | Findings, recommendations, next steps |

---

## 📊 Dataset Description

| Feature | Type | Description |
|---------|------|-------------|
| `YearOfObservation` | Datetime | Year the building was observed |
| `Insured_Period` | Float (0–1) | Duration of insurance coverage |
| `Residential` | Categorical | Residential (1) or non-residential (0) |
| `Building_Painted` | Categorical | Painted (V) or not painted (N) |
| `Building_Fenced` | Categorical | Fenced (V) or not fenced (N) |
| `Garden` | Categorical | Has garden (V) or no garden (O) |
| `Settlement` | Categorical | Rural (R) or Urban (U) |
| `Building_Type` | Categorical | Building type (1–4) |
| `Date_of_Occupancy` | Datetime | Year first occupied |
| `NumberOfWindows` | Integer | Number of windows |
| `Building_Dimension` | Float | Size of building in m² |
| **`Claim`** | **Binary (0/1)** | **Target variable** |

**Dataset:** 7,160 raw records → **7,014 clean records** (after removing 146 duplicates)

---

## 🔑 Key Findings

| Finding | Detail |
|---------|--------|
| **Class Imbalance** | 76.82% No Claim vs 23.18% Claim — 3.3:1 ratio |
| **Top predictor** | `building_dimension` — claim buildings have 2.2× larger median size (1,995.5 m² vs 900.0 m²) |
| **2nd predictor** | `building_age` — together with dimension, explains 68.73% of model decisions |
| **3rd predictor** | `number_of_windows` — claim buildings have 3× the median window count (3.0 vs 1.0) |
| **Urban/Rural signal** | Rural/unfenced/no-garden buildings claim at 25.0% vs 21.2% for urban equivalents |

---

## 🤖 Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.6636 | 0.3527 | **0.5415** | **0.4272** | **0.6629** |
| Decision Tree | 0.6536 | 0.2952 | 0.3569 | 0.3231 | 0.5498 |
| **Random Forest (Tuned) ✅** | **0.7106** | **0.3608** | 0.3231 | 0.3409 | 0.6144 |

**Selected Model:** Tuned Random Forest — `max_depth=20`, `n_estimators=200`, `max_features=sqrt`
**Cross-validated F1-Score:** 0.7921 ± 0.1063 (5-fold CV on training data)

---

## 🛠️ Technical Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalanced Learning | imbalanced-learn (SMOTE) |
| Interpretability | SHAP |
| Automated EDA | ydata-profiling |
| Model Persistence | Joblib |
| Web App | Streamlit |

---

## ⚙️ How to Run This Project

### Option 1 — Live Web App (No Setup Required)
Click the Streamlit badge at the top of this README to open the deployed app directly in your browser.

### Option 2 — Google Colab (Recommended for the full notebook)
Click the **"Open in Colab"** badge at the top. The notebook downloads data directly from Google Drive — no manual setup needed.

### Option 3 — Run Locally

**Step 1: Clone the repository**
```bash
git clone https://github.com/cssadewale/insurance-claim-prediction.git
cd insurance-claim-prediction
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Launch Jupyter**
```bash
jupyter notebook Insurance_Claim_Prediction_COMPLETE_FINAL.ipynb
```

> The dataset and trained model are loaded from Google Drive inside the notebook — no manual downloads needed.

---

## 📈 Business Recommendations

1. **Risk-Based Premium Pricing** — Use `building_dimension`, `building_age`, and `number_of_windows` as primary rating factors
2. **Tiered Risk Classification by Building Type** — Type 4 = high-risk tier; Type 1 = low-risk tier
3. **Targeted Pre-Renewal Inspections** — Flag buildings with predicted claim probability > 60%
4. **Residential Portfolio Review** — Residential buildings claim at 36.0% vs 29.3% for non-residential
5. **Maintenance Incentive Programme** — Offer premium discounts for verified property maintenance
6. **Early Warning Scoring System** — Integrate model into new business application workflow

---

## 🔮 Next Steps

- [ ] Build a **claim severity model** (regression) to predict claim amount
- [ ] Enrich dataset with external data (weather events, flood zones, building permits)
- [ ] Deploy as a full **Streamlit web dashboard** for underwriter use
- [ ] Establish an **annual model retraining pipeline**
- [ ] Explore **XGBoost / LightGBM** as potential performance improvements

---

## 👤 About the Author

**Adewale Samson Adeagbo** — Secondary school educator (Mathematics, Further Mathematics, Chemistry, Physics) and Data Science practitioner.

- 🎓 B.Sc.(Ed) Computer Science Education, Lagos State University (2023)
- 🏫 Founder & Director, HMG Concepts Educational Consult (est. 2015)
- 🌍 Lagos/Ogun State, Nigeria
- 💼 [LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo) | [GitHub](https://github.com/cssadewale)

*"Learning deliberately. Teaching authentically."*

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
