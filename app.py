# ─────────────────────────────────────────────────────────────────────────────
# Building Insurance Claim Prediction — Streamlit Web Application
# Author  : Adewale Samson Adeagbo
# GitHub  : https://github.com/cssadewale
# LinkedIn: https://linkedin.com/in/adewalesamsonadeagbo
# ─────────────────────────────────────────────────────────────────────────────

import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Building Insurance Claim Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
        padding: 2rem 2.5rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0 0; font-size: 0.95rem; opacity: 0.88; }

    .result-card {
        padding: 1.5rem 2rem; border-radius: 10px; text-align: center;
        font-size: 1rem; font-weight: 600; margin: 1rem 0; line-height: 1.7;
    }
    .risk-high { background:#fff0f0; border:2px solid #e53e3e; color:#c53030; }
    .risk-low  { background:#f0fff4; border:2px solid #38a169; color:#276749; }
    .risk-med  { background:#fffbeb; border:2px solid #d97706; color:#92400e; }

    .metric-tile {
        background:#f7fafc; border:1px solid #e2e8f0;
        border-radius:8px; padding:1rem 1.2rem; text-align:center;
    }
    .metric-tile .mt-label {
        font-size:0.75rem; color:#718096;
        text-transform:uppercase; letter-spacing:0.06em;
    }
    .metric-tile .mt-value { font-size:1.55rem; font-weight:700; color:#2d3748; }

    .sec-head {
        font-size:1rem; font-weight:700; color:#2d3748;
        border-left:4px solid #2e6da4; padding-left:0.7rem;
        margin: 1.4rem 0 0.6rem 0;
    }

    .info-box {
        background:#ebf8ff; border:1px solid #bee3f8; border-radius:8px;
        padding:0.9rem 1.1rem; font-size:0.88rem; color:#2c5282; margin:0.6rem 0;
    }
    .warn-box {
        background:#fffbeb; border:1px solid #fbd38d; border-radius:8px;
        padding:0.9rem 1.1rem; font-size:0.88rem; color:#744210; margin:0.6rem 0;
    }

    .footer {
        background:#f7fafc; border-top:1px solid #e2e8f0; border-radius:8px;
        padding:1.2rem; text-align:center; margin-top:2rem;
        font-size:0.82rem; color:#718096;
    }
    .footer a { color:#2e6da4; text-decoration:none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# The model file exceeds GitHub's 25 MB limit, so it is hosted on Google Drive.
# On first run (e.g., on Streamlit Cloud), gdown downloads it automatically.
# On subsequent runs, the cached file on disk is used directly.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_random_forest_model.joblib"
GDRIVE_URL = "https://drive.google.com/uc?id=1AlshXgY5MlJYE5JbnXq8mWoy0eteeVAa"


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the trained Random Forest model.

    If the .joblib file is not present locally (e.g., first launch on
    Streamlit Cloud), it is downloaded from Google Drive via gdown before
    loading. The @st.cache_resource decorator ensures this runs only once
    per session regardless of user interactions.
    """
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading model from Google Drive — this only happens once..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        st.error(
            "❌ Model download failed. "
            "Please check your internet connection or the Google Drive link in app.py."
        )
        st.stop()

    return joblib.load(MODEL_PATH)


model = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING HELPERS
# These must exactly mirror the preprocessing pipeline used in the notebook.
# Any deviation will cause the model to receive different inputs than it
# was trained on, producing unreliable predictions.
# ─────────────────────────────────────────────────────────────────────────────

# StandardScaler statistics learned from X_train_resampled during training.
# To refresh these after retraining, run the following in the notebook
# immediately after fitting the scaler:
#
#   for feat, mean, std in zip(numerical_cols_for_scaling,
#                               scaler.mean_, scaler.scale_):
#       print(f'"{feat}": {mean:.6f},  # std={std:.6f}')
#
SCALER_MEANS = {
    "insured_period":              0.9083,
    "building_dimension":       1884.80,
    "number_of_windows":           2.2223,
    "building_age":               49.10,
    "log_building_dimension":      6.80,
    "transformed_insured_period":  0.105,
    "log_building_age":            3.55,
}
SCALER_STDS = {
    "insured_period":              0.257,
    "building_dimension":       2283.0,
    "number_of_windows":           2.44,
    "building_age":               52.0,
    "log_building_dimension":      1.25,
    "transformed_insured_period":  0.255,
    "log_building_age":            1.10,
}

# Exact feature columns in the order the model expects
# (matches X_train_resampled.columns from the notebook)
FEATURE_COLUMNS = [
    "insured_period",
    "building_dimension",
    "number_of_windows",
    "building_age",
    "log_building_dimension",
    "transformed_insured_period",
    "log_building_age",
    "residential_1",
    "building_type_2",
    "building_type_3",
    "building_type_4",
    "building_painted_V",
    "building_fenced_V",
    "garden_V",
    "settlement_U",
]


def engineer_features(
    insured_period, building_dimension, number_of_windows, building_age,
    residential, building_type, building_painted, building_fenced, garden, settlement,
):
    """
    Apply the full preprocessing pipeline to raw user inputs.

    Steps (must match the notebook in this exact order):
      1. Log-transform skewed numerical features
      2. One-hot encode categorical features (drop_first=True, same reference groups)
      3. Standardise all numerical features using training-set μ and σ
      4. Return the final ordered feature DataFrame
    """

    # ── Step 1: Log transformations ───────────────────────────────────────────
    # log1p(x) = log(1 + x) — safe when x = 0 (avoids log(0) = -∞)
    log_building_dimension     = np.log1p(building_dimension)
    transformed_insured_period = np.log1p(1.0 - insured_period)   # reflected: left → right → compress
    log_building_age           = np.log1p(max(float(building_age), 0.0))

    # ── Step 2: One-hot encoding (same reference groups as pd.get_dummies, drop_first=True) ──
    # Reference groups (dropped during training):
    #   residential   → 0 (non-residential)
    #   building_type → 1
    #   building_painted → 'N' (not painted)
    #   building_fenced  → 'N' (unfenced)
    #   garden           → 'O' (no garden)
    #   settlement       → 'R' (rural)
    residential_1      = 1.0 if residential    == 1   else 0.0
    building_type_2    = 1.0 if building_type  == 2   else 0.0
    building_type_3    = 1.0 if building_type  == 3   else 0.0
    building_type_4    = 1.0 if building_type  == 4   else 0.0
    building_painted_V = 1.0 if building_painted == "V" else 0.0
    building_fenced_V  = 1.0 if building_fenced  == "V" else 0.0
    garden_V           = 1.0 if garden           == "V" else 0.0
    settlement_U       = 1.0 if settlement       == "U" else 0.0

    # ── Step 3: Assemble raw feature dictionary ───────────────────────────────
    raw = {
        "insured_period":             float(insured_period),
        "building_dimension":         float(building_dimension),
        "number_of_windows":          float(number_of_windows),
        "building_age":               float(building_age),
        "log_building_dimension":     log_building_dimension,
        "transformed_insured_period": transformed_insured_period,
        "log_building_age":           log_building_age,
        "residential_1":              residential_1,
        "building_type_2":            building_type_2,
        "building_type_3":            building_type_3,
        "building_type_4":            building_type_4,
        "building_painted_V":         building_painted_V,
        "building_fenced_V":          building_fenced_V,
        "garden_V":                   garden_V,
        "settlement_U":               settlement_U,
    }

    # ── Step 4: Standardise numerical features using training-set statistics ──
    for feat in SCALER_MEANS:
        raw[feat] = (raw[feat] - SCALER_MEANS[feat]) / SCALER_STDS[feat]

    # ── Step 5: Return ordered DataFrame (model requires this exact column order) ──
    return pd.DataFrame([raw])[FEATURE_COLUMNS]


def risk_tier(prob: float):
    """Map a claim probability to a (label, css_class, recommendation) tuple."""
    if prob >= 0.70:
        return (
            "🔴 Very High Risk", "risk-high",
            "This building is highly likely to file a claim. "
            "Immediate underwriter review and pre-policy inspection are strongly recommended.",
        )
    elif prob >= 0.50:
        return (
            "🟠 High Risk", "risk-high",
            "This building has an elevated claim probability. "
            "Consider mandatory pre-renewal inspection and a risk-adjusted premium.",
        )
    elif prob >= 0.30:
        return (
            "🟡 Moderate Risk", "risk-med",
            "This building carries moderate claim risk. "
            "Standard policy terms apply; schedule a review at the next renewal cycle.",
        )
    else:
        return (
            "🟢 Low Risk", "risk-low",
            "This building is unlikely to file a claim. "
            "Eligible for standard or preferential pricing.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Navigation")
    page = st.radio(
        "Go to",
        ["🔮 Predict Claim Risk", "📊 Model Information", "📖 About the Project"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
**Built by**

**Adewale Samson Adeagbo**

*Learning deliberately.*
*Teaching authentically.*

---

[![GitHub](https://img.shields.io/badge/GitHub-cssadewale-181717?logo=github&style=flat-square)](https://github.com/cssadewale)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin&style=flat-square)](https://linkedin.com/in/adewalesamsonadeagbo)
""")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔮 Predict Claim Risk":

    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Building Insurance Claim Predictor</h1>
        <p>
          Enter the building's structural and environmental characteristics below.
          The model returns an instant claim probability, risk tier, and underwriter recommendation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    ℹ️ <strong>How to use:</strong> Fill in every field in the three sections below,
    then click <strong>Predict Claim Risk</strong>. All inputs reflect the same variables
    the model was trained on — the more accurate your inputs, the more reliable the prediction.
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ─────────────────────────────────────────────────────────
    with st.form("prediction_form"):

        st.markdown('<div class="sec-head">📐 Physical Characteristics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            building_dimension = st.number_input(
                "Building Dimension (m²)",
                min_value=1.0, max_value=25000.0, value=1000.0, step=50.0,
                help=(
                    "Total floor area in square metres. "
                    "The strongest single predictor — claim buildings have a median of "
                    "1,995 m² vs 900 m² for non-claim buildings (a 2.2× difference)."
                ),
            )
        with c2:
            number_of_windows = st.slider(
                "Number of Windows",
                min_value=0, max_value=10, value=2,
                help=(
                    "Total number of windows, capped at 10. "
                    "Claim buildings have a median of 3 windows vs 1 for non-claim buildings."
                ),
            )
        with c3:
            building_age = st.number_input(
                "Building Age (years)",
                min_value=0, max_value=400, value=30, step=1,
                help=(
                    "Years since the building was first occupied. "
                    "Dataset spans 0–316 years, mean = 49.1 years. "
                    "Older buildings claim more frequently due to accumulated wear."
                ),
            )

        st.markdown('<div class="sec-head">📋 Policy & Classification</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            insured_period = st.slider(
                "Insured Period (fraction of year)",
                min_value=0.0, max_value=1.0, value=1.0, step=0.01, format="%.2f",
                help=(
                    "Proportion of the year the building was insured. "
                    "1.0 = full year. Most policies in the dataset are full-year (median = 1.0)."
                ),
            )
        with c5:
            building_type = st.selectbox(
                "Building Type",
                options=[1, 2, 3, 4], index=1,
                format_func=lambda x: f"Type {x}",
                help="Structural classification. Type 4 has the highest claim rate; Type 1 the lowest.",
            )
        with c6:
            residential = st.selectbox(
                "Residential Status",
                options=[0, 1], index=0,
                format_func=lambda x: "Residential" if x == 1 else "Non-Residential",
                help="30.5% of buildings in the dataset are residential. Residential buildings show a slightly higher claim rate.",
            )

        st.markdown('<div class="sec-head">🏡 Structural & Location Features</div>', unsafe_allow_html=True)
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            building_painted = st.selectbox(
                "Building Painted?",
                options=["V", "N"], index=0,
                format_func=lambda x: "Yes — Painted" if x == "V" else "No — Not Painted",
                help="75.68% of buildings are painted. Painted buildings show a 2.6 pp higher claim rate.",
            )
        with c8:
            building_fenced = st.selectbox(
                "Building Fenced?",
                options=["V", "N"], index=0,
                format_func=lambda x: "Yes — Fenced" if x == "V" else "No — Unfenced",
                help="Near-perfectly correlated with garden and settlement — all three encode the same urban/rural dimension.",
            )
        with c9:
            garden = st.selectbox(
                "Has Garden?",
                options=["V", "O"], index=0,
                format_func=lambda x: "Yes — Has Garden" if x == "V" else "No Garden",
                help="Near-perfectly correlated with building_fenced and settlement.",
            )
        with c10:
            settlement = st.selectbox(
                "Settlement Type",
                options=["U", "R"], index=0,
                format_func=lambda x: "Urban" if x == "U" else "Rural",
                help="Rural buildings show a 3.8 percentage point higher claim rate (25.0% vs 21.2%).",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔮  Predict Claim Risk",
            use_container_width=True,
            type="primary",
        )

    # ── Prediction Output ───────────────────────────────────────────────────
    if submitted:
        with st.spinner("Running prediction pipeline..."):
            feature_vector = engineer_features(
                insured_period=insured_period,
                building_dimension=building_dimension,
                number_of_windows=number_of_windows,
                building_age=building_age,
                residential=residential,
                building_type=building_type,
                building_painted=building_painted,
                building_fenced=building_fenced,
                garden=garden,
                settlement=settlement,
            )
            prediction    = int(model.predict(feature_vector)[0])
            claim_prob    = float(model.predict_proba(feature_vector)[0][1])
            no_claim_prob = 1.0 - claim_prob
            label, css, recommendation = risk_tier(claim_prob)

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        # Main result card
        st.markdown(
            f'<div class="result-card {css}">'
            f'<div style="font-size:1.1rem;">{label}</div>'
            f'<div style="font-size:2.4rem; font-weight:800; margin:0.3rem 0;">{claim_prob:.1%}</div>'
            f'<div style="font-size:0.85rem; font-weight:400; opacity:0.85;">Predicted Claim Probability</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Four metric tiles
        m1, m2, m3, m4 = st.columns(4)
        for col, lbl, val in zip(
            [m1, m2, m3, m4],
            ["Claim Probability", "No-Claim Probability", "Model Decision", "Decision Threshold"],
            [f"{claim_prob:.1%}", f"{no_claim_prob:.1%}",
             "⚠️ Claim" if prediction == 1 else "✅ No Claim", "50.0%"],
        ):
            col.markdown(
                f'<div class="metric-tile">'
                f'<div class="mt-label">{lbl}</div>'
                f'<div class="mt-value">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Probability bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">📈 Probability Breakdown</div>', unsafe_allow_html=True)
        ba, bb = st.columns(2)
        with ba:
            st.markdown(f"**Claim Probability — {claim_prob:.1%}**")
            st.progress(claim_prob)
        with bb:
            st.markdown(f"**No-Claim Probability — {no_claim_prob:.1%}**")
            st.progress(no_claim_prob)

        # Recommendation
        st.markdown('<div class="sec-head">💡 Underwriter Recommendation</div>', unsafe_allow_html=True)
        st.info(f"**{label}** — {recommendation}")

        # Expandable summaries
        with st.expander("📋 View Input Summary"):
            st.dataframe(
                pd.DataFrame({
                    "Feature": [
                        "Insured Period", "Building Dimension (m²)", "Number of Windows",
                        "Building Age (years)", "Building Type", "Residential",
                        "Building Painted", "Building Fenced", "Has Garden", "Settlement",
                    ],
                    "Value Entered": [
                        f"{insured_period:.2f}", f"{building_dimension:,.0f}",
                        str(number_of_windows), str(building_age),
                        f"Type {building_type}",
                        "Residential" if residential == 1 else "Non-Residential",
                        "Yes" if building_painted == "V" else "No",
                        "Yes" if building_fenced  == "V" else "No",
                        "Yes" if garden == "V" else "No",
                        "Urban" if settlement == "U" else "Rural",
                    ],
                }),
                use_container_width=True, hide_index=True,
            )

        with st.expander("⚙️ View Engineered Feature Vector (what the model actually receives)"):
            st.markdown("""
The model never receives raw inputs directly. They pass through the same preprocessing pipeline
used during training — log transformations, one-hot encoding, and StandardScaler.
The table below shows the final scaled feature vector passed to the model.
            """)
            st.dataframe(feature_vector.round(4), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — MODEL INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Information":

    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Information</h1>
        <p>Performance metrics, feature importances, cross-validation results, and the full preprocessing pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🤖 Recommended Model — Tuned Random Forest</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Best Hyperparameters (GridSearchCV)**")
        st.dataframe(
            pd.DataFrame({
                "Hyperparameter": ["n_estimators", "max_depth", "max_features",
                                   "min_samples_split", "min_samples_leaf"],
                "Optimal Value":  ["200", "20", "sqrt", "2", "1"],
                "What It Controls": [
                    "Number of decision trees in the forest",
                    "Maximum depth each tree is allowed to grow",
                    "Features considered at each split point",
                    "Minimum samples to split a node",
                    "Minimum samples required at a leaf",
                ],
            }),
            use_container_width=True, hide_index=True,
        )
    with s2:
        st.markdown("**Training Configuration**")
        st.dataframe(
            pd.DataFrame({
                "Setting": [
                    "Original dataset", "After cleaning & deduplication",
                    "Training samples (after SMOTE)", "Test samples",
                    "Class balance (training)", "CV folds", "Tuning metric",
                    "GridSearchCV combinations", "Total model fits",
                ],
                "Value": [
                    "7,160 records × 14 columns", "7,014 records",
                    "8,620 (3,009 synthetic Claim=1 added)", "1,403",
                    "50% / 50%", "5", "F1-Score", "216", "1,080",
                ],
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown('<div class="sec-head">📈 Model Performance — Test Set</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Why accuracy is not the primary metric:</strong>
    A naïve classifier that always predicts "No Claim" achieves <strong>76.82% accuracy</strong>
    without learning anything — because 76.82% of buildings never filed a claim.
    We therefore rely on <strong>Precision, Recall, F1-Score, and ROC-AUC</strong>.
    </div>
    """, unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        "Model":     ["Logistic Regression", "Decision Tree", "Tuned Random Forest ✅"],
        "Accuracy":  [0.6636, 0.6536, 0.7106],
        "Precision": [0.3527, 0.2952, 0.3608],
        "Recall":    [0.5415, 0.3569, 0.3231],
        "F1-Score":  [0.4272, 0.3231, 0.3409],
        "ROC-AUC":   [0.6629, 0.5498, 0.6144],
    })
    st.dataframe(
        perf_df.style
            .highlight_max(subset=["Accuracy", "Precision", "F1-Score", "ROC-AUC"], color="#c6efce")
            .highlight_max(subset=["Recall"], color="#c6efce")
            .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                     "Recall": "{:.4f}", "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown("""
    <div class="info-box">
    📌 <strong>Why Random Forest was selected:</strong>
    Highest accuracy (0.7106) and precision (0.3608) on the test set,
    with the most stable cross-validated performance —
    Mean CV F1 = <strong>0.7921 ± 0.1063</strong> across 5 folds.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🎯 Top 10 Feature Importances (Gini Impurity Reduction)</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Feature": [
            "log_building_dimension", "building_dimension", "log_building_age",
            "building_age", "number_of_windows", "transformed_insured_period",
            "building_type_2", "insured_period", "building_painted_V", "residential_1",
        ],
        "Importance": [0.2164, 0.2131, 0.1448, 0.1130, 0.0822,
                       0.0535, 0.0496, 0.0492, 0.0265, 0.0209],
        "Cumulative %": ["21.64%", "42.95%", "57.43%", "68.73%", "75.95%",
                          "82.30%", "87.26%", "92.18%", "94.83%", "96.92%"],
    })
    st.dataframe(
        fi_df.style.background_gradient(subset=["Importance"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )
    st.markdown("""
    - `log_building_dimension` + `building_dimension` = **42.95%** of total importance
    - `log_building_age` + `building_age` = **25.78%** of total importance
    - These two physical characteristics alone explain **68.73%** of the model's decision-making
    - **SHAP confirms:** High building dimension and high building age both push predictions toward Claim (1)
    """)

    st.markdown('<div class="sec-head">🔄 5-Fold Cross-Validation Results (F1-Score)</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Fold":     ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean", "Std Dev"],
            "F1-Score": [0.5993, 0.7538, 0.8651, 0.8731, 0.8694, 0.7921, 0.1063],
            "Note": [
                "Lowest — likely higher proportion of synthetic SMOTE samples in this fold",
                "", "", "", "Highest",
                "Best estimate of true performance on the training distribution",
                "Measure of consistency across folds",
            ],
        }),
        use_container_width=True, hide_index=True,
    )
    st.markdown("""
    <div class="info-box">
    📌 The mean CV F1 of <strong>0.7921</strong> reflects performance on the SMOTE-balanced training distribution.
    The test set F1-Score will be lower because the test set retains the original real-world imbalance (23.18% claims).
    This gap is expected and correct — it reflects honest, real-world performance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">⚙️ Preprocessing Pipeline Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Stage": [
                "Raw dataset loaded", "After data cleaning",
                "After deduplication", "Train-test split",
                "Training set after SMOTE", "Test set (unchanged)",
            ],
            "Samples": ["7,160", "7,014", "7,014",
                        "Train: 5,611 / Test: 1,403", "8,620", "1,403"],
            "Key Detail": [
                "14 columns · 3 missing-value columns · number_of_windows anomalies",
                "Mode imputation on garden, date_of_occupancy, building_dimension",
                "146 exact duplicates removed",
                "stratify=y preserves 76.82% / 23.18% ratio in both sets",
                "3,009 synthetic Claim=1 samples added · 50% / 50% balance",
                "Real-world imbalance preserved — never touched by SMOTE",
            ],
        }),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 About the Project":

    st.markdown("""
    <div class="main-header">
        <h1>📖 About This Project</h1>
        <p>A complete, portfolio-grade data science project — from raw data to a deployed prediction application.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🎯 Project Objective</div>', unsafe_allow_html=True)
    st.markdown("""
To develop a **robust binary classification model** that accurately predicts the probability of a building
filing an insurance claim — providing actionable, data-driven insights to support underwriting decisions,
risk pricing, and resource allocation.

The model is trained on **7,014 real building records** (after cleaning), with **13 engineered features**
covering physical characteristics, policy details, and location context.
    """)

    st.markdown('<div class="sec-head">🔄 Project Workflow</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Step": ["1", "2", "3", "4", "5", "6"],
            "Phase": [
                "Basic Data Exploration", "Exploratory Data Analysis",
                "Feature Engineering", "Data Preprocessing",
                "Model Development", "Business Insights",
            ],
            "Key Actions": [
                "Loaded 7,160 records · cleaned number_of_windows (3 steps) · imputed 3 columns · removed 146 duplicates",
                "Univariate, bivariate, multivariate analysis · confirmed 3.3:1 class imbalance · top predictor identified",
                "Engineered building_age · log1p & reflected log1p transformations · one-hot encoded 4 categorical features",
                "80/20 stratified split → SMOTE (5,611 → 8,620) → StandardScaler (fit on training only)",
                "Trained 3 models · GridSearchCV (216 configs, 1,080 fits) · 5-fold CV · SHAP interpretability",
                "6 actionable recommendations for pricing, inspection, and portfolio management",
            ],
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="sec-head">🔍 Key Findings from EDA</div>', unsafe_allow_html=True)
    findings = [
        ("Class Imbalance (Target Variable)",
         "76.82% No Claim vs 23.18% Claim — a 3.3:1 ratio. A naïve always-predict-0 classifier achieves 76.82% accuracy "
         "by default. Addressed with SMOTE on training data only, applied after the train-test split."),
        ("Building Dimension — Strongest Predictor",
         "Claim buildings have a median size of **1,995 m²** vs **900 m²** for non-claim buildings (2.2× difference). "
         "Accounts for 42.95% of total feature importance combined with its log-transformed version."),
        ("Building Age — Second Most Important",
         "Average age: 49.1 years. Range: 0–316 years. Older buildings claim more frequently. "
         "Accounts for 25.78% of feature importance combined with its log-transformed version."),
        ("Number of Windows — Third Signal",
         "Claim buildings have a median of **3 windows** vs **1 window** — a 3× difference. "
         "Contributes 8.22% of feature importance."),
        ("Urban/Rural Triple Redundancy",
         "`building_fenced`, `garden`, and `settlement` are near-perfectly correlated. "
         "All three show identical claim rates: 25.0% (rural/unfenced/no-garden) vs 21.2% (urban/fenced/garden). "
         "They encode one underlying risk dimension — not three independent signals."),
        ("Building Type Risk Gradient",
         "Type 4 → highest claim rate. Type 1 → lowest. A consistent, monotonic ordering across all analyses."),
    ]
    for title, body in findings:
        with st.expander(f"📌 {title}"):
            st.markdown(body)

    st.markdown('<div class="sec-head">💼 Actionable Business Recommendations</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Recommendation": [
                "1. Risk-Based Premium Pricing",
                "2. Tiered Risk Classification by Building Type",
                "3. Targeted Pre-Renewal Inspections",
                "4. Residential Portfolio Review",
                "5. Maintenance Incentive Programme",
                "6. Early Warning at New Business Application",
            ],
            "Action": [
                "Use building_dimension, building_age, and number_of_windows as primary rating factors.",
                "Type 4 & 3 → high-risk tier (mandatory inspection, higher premiums). Type 1 → preferential pricing.",
                "Flag buildings scoring above 60% claim probability for physical inspection before renewal.",
                "Conduct dedicated actuarial review — residential buildings claim at a disproportionately higher rate.",
                "Offer premium discounts to policyholders who provide verified maintenance records at renewal.",
                "Integrate model into application workflow — refer high-risk applicants for manual underwriter review.",
            ],
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="sec-head">⚠️ Limitations & Next Steps</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Limitation": [
                "Geographic granularity lost",
                "Claims severity not modelled",
                "Temporal drift",
                "Binary target limitation",
                "Scaler statistics hardcoded",
            ],
            "Impact": [
                "geo_code was dropped — location risk (flood zones, fire proximity) unaccounted for.",
                "Predicts whether a claim occurs, not how costly it will be.",
                "Dataset covers 2012–2016 — patterns may have shifted since.",
                "Minor and catastrophic claims both receive claim = 1.",
                "If model is retrained, SCALER_MEANS and SCALER_STDS in app.py must be updated.",
            ],
            "Mitigation": [
                "Enrich with geocoded risk scores and hazard indices.",
                "Build a companion regression model predicting claim amount (for claim = 1 records).",
                "Annual retraining pipeline with KS test / PSI drift monitoring.",
                "Multi-class or severity-weighted target variable.",
                "Persist the scaler object alongside the model and load it in app.py.",
            ],
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="sec-head">👤 Author</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="background:#f7fafc; border:1px solid #e2e8f0; border-radius:10px; padding:1.5rem 2rem; margin-top:0.5rem;">

### Adewale Samson Adeagbo
*Learning deliberately. Teaching authentically.*

| | |
|---|---|
| 🎓 **Education** | B.Sc.(Ed) Computer Science Education — Lagos State University (2023) |
| 📚 **Teaching** | Mathematics, Further Mathematics, Chemistry, Physics — 10+ years, Lagos & Ogun State, Nigeria |
| 🏢 **Organisation** | HMG Concepts (His Marvellous Grace Educational Consult) — Visioner & Data Lead since 2015 |
| 🛠️ **Stack** | Python · Scikit-learn · Pandas · SHAP · SQL · Power BI · Tableau · Streamlit |

[![GitHub](https://img.shields.io/badge/GitHub-cssadewale-181717?style=for-the-badge&logo=github)](https://github.com/cssadewale)
&nbsp;&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adewale%20Samson%20Adeagbo-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/adewalesamsonadeagbo)

</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🏛️ <strong>Building Insurance Claim Prediction</strong> &nbsp;·&nbsp;
    Tuned Random Forest &nbsp;·&nbsp; CV F1 = 0.7921 &nbsp;·&nbsp;
    Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> &nbsp;·&nbsp;
    <a href="https://github.com/cssadewale/insurance-claim-prediction" target="_blank">GitHub</a> &nbsp;·&nbsp;
    <a href="https://linkedin.com/in/adewalesamsonadeagbo" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
