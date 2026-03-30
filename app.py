# ══════════════════════════════════════════════════════════════════════════════
# BUILDING INSURANCE CLAIM PREDICTION — STREAMLIT WEB APP
# Author  : Adewale Samson Adeagbo
# GitHub  : https://github.com/cssadewale
# LinkedIn: https://linkedin.com/in/adewalesamsonadeagbo
# Purpose : Interactive risk assessment tool for building insurance claims
# Model   : Tuned Random Forest Classifier (max_depth=20, n_estimators=200)
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Page configuration — must be the FIRST Streamlit command ──────────────────
st.set_page_config(
    page_title = "Insurance Claim Predictor",
    page_icon  = "🏛️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MODEL LOADING
# The model file exceeds GitHub's 25 MB limit, so it is hosted on Google Drive.
# gdown downloads it once per session and caches it for subsequent calls.
# ══════════════════════════════════════════════════════════════════════════════

# ── Replace this with your actual Google Drive file ID ────────────────────────
# How to get your file ID:
#   1. Upload best_random_forest_model.joblib to Google Drive
#   2. Right-click the file → Share → "Anyone with the link" → Copy link
#   3. The link looks like: https://drive.google.com/file/d/FILE_ID_HERE/view
#   4. Paste just the FILE_ID_HERE part below
MODEL_GDRIVE_FILE_ID = "11UPPIieU8SaHBFTk_Xm-Ex4DTHE8HmjA"
MODEL_LOCAL_PATH     = "best_random_forest_model.joblib"

@st.cache_resource   # cache: downloads once, reuses across all user interactions
def load_model():
    """
    Download the trained model from Google Drive if not already present,
    then load and return it. Uses st.cache_resource so this runs only once
    per Streamlit session — not on every page interaction.
    """
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("⏳ Downloading model from Google Drive (first load only)..."):
            url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_LOCAL_PATH, quiet=False)

    model = joblib.load(MODEL_LOCAL_PATH)
    return model

model = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PREPROCESSING PIPELINE
# Replicates the exact preprocessing steps from the notebook:
#   Step 1 → Compute building_age
#   Step 2 → Apply log/reflected-log transformations
#   Step 3 → One-Hot Encode categorical inputs
#   Step 4 → Apply StandardScaler (using hardcoded training statistics)
#   Step 5 → Arrange columns in exact training order
# ══════════════════════════════════════════════════════════════════════════════

# ── StandardScaler statistics learned from X_train_resampled ──────────────────
# These are the μ (mean) and σ (std) values the scaler fitted on training data.
# They must be hardcoded here so the app transforms new inputs identically
# to how the training data was transformed — without reloading the full dataset.
# Update these values if you retrain the model on new data.
SCALER_MEANS = {
    "insured_period":             0.9083,
    "building_dimension":         1884.8035,
    "number_of_windows":          2.2223,
    "building_age":               49.10,
    "log_building_dimension":     6.9842,
    "transformed_insured_period": 0.1247,
    "log_building_age":           3.4821,
}
SCALER_STDS = {
    "insured_period":             0.2413,
    "building_dimension":         2283.1298,
    "number_of_windows":          2.5233,
    "building_age":               43.85,
    "log_building_dimension":     1.1038,
    "transformed_insured_period": 0.2806,
    "log_building_age":           1.1042,
}

# ── Exact feature column order — must match X_train_resampled.columns ─────────
# From notebook output: ['insured_period', 'residential', 'building_dimension',
# 'building_type', 'number_of_windows', 'building_age', 'log_building_dimension',
# 'transformed_insured_period', 'log_building_age', 'building_painted_V',
# 'building_fenced_V', 'garden_V', 'settlement_U']
FEATURE_COLUMNS = [
    "insured_period",
    "residential",
    "building_dimension",
    "building_type",
    "number_of_windows",
    "building_age",
    "log_building_dimension",
    "transformed_insured_period",
    "log_building_age",
    "building_painted_V",
    "building_fenced_V",
    "garden_V",
    "settlement_U",
]


def preprocess_input(
    insured_period, building_dimension, number_of_windows,
    year_of_observation, date_of_occupancy,
    residential, building_painted, building_fenced,
    garden, settlement, building_type
):
    """
    Transform raw sidebar inputs into the exact 13-feature vector the model expects.
    Mirrors every preprocessing step in the notebook in the same sequence.
    Returns: (preprocessed_df, building_age_int)
    """

    # ── Step 1: Compute building_age ──────────────────────────────────────────
    building_age = max(year_of_observation - date_of_occupancy, 0)

    # ── Step 2: Feature transformations ──────────────────────────────────────
    log_building_dimension      = np.log1p(building_dimension)
    transformed_insured_period  = np.log1p(max(1 - insured_period, 0))
    log_building_age            = np.log1p(building_age)

    # ── Step 3: One-Hot Encode categorical inputs ────────────────────────────
    # Reference categories (dropped in training with drop_first=True):
    #   residential    → reference = 0 (Non-Residential)
    #   building_type  → reference = 1 (Type 1)
    #   building_painted → reference = N (Not Painted)
    #   building_fenced  → reference = N (Not Fenced)
    #   garden           → reference = O (No Garden)
    #   settlement       → reference = R (Rural)
    res_val   = 1 if residential     == "1 (Residential)"  else 0
    btype_val = {"Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4}[building_type]
    painted_V = 1 if building_painted == "V (Painted)"     else 0
    fenced_V  = 1 if building_fenced  == "V (Fenced)"      else 0
    garden_V  = 1 if garden           == "V (Has Garden)"  else 0
    settle_U  = 1 if settlement       == "U (Urban)"       else 0

    # ── Step 4: Assemble raw feature values ───────────────────────────────────
    raw = {
        "insured_period":             insured_period,
        "residential":                res_val,
        "building_dimension":         building_dimension,
        "building_type":              btype_val,
        "number_of_windows":          number_of_windows,
        "building_age":               building_age,
        "log_building_dimension":     log_building_dimension,
        "transformed_insured_period": transformed_insured_period,
        "log_building_age":           log_building_age,
        "building_painted_V":         painted_V,
        "building_fenced_V":          fenced_V,
        "garden_V":                   garden_V,
        "settlement_U":               settle_U,
    }

    # ── Step 5: Apply StandardScaler to numerical columns ─────────────────────
    num_cols = [
        "insured_period", "building_dimension", "number_of_windows",
        "building_age", "log_building_dimension",
        "transformed_insured_period", "log_building_age"
    ]
    for col in num_cols:
        raw[col] = (raw[col] - SCALER_MEANS[col]) / SCALER_STDS[col]

    # ── Step 6: Return DataFrame in exact training column order ───────────────
    input_df = pd.DataFrame([raw])[FEATURE_COLUMNS]
    return input_df, building_age


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_gauge(probability):
    """Semicircular gauge chart: green (low) → orange (medium) → red (high)."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    # Colour segments
    for start, end, color in [
        (0,    0.33, "#2ECC71"),
        (0.33, 0.66, "#F39C12"),
        (0.66, 1.00, "#E74C3C"),
    ]:
        wedge = mpatches.Wedge(
            center=(0.5, 0.2), r=0.45,
            theta1=180 - end * 180, theta2=180 - start * 180,
            width=0.15, facecolor=color, alpha=0.85
        )
        ax.add_patch(wedge)

    # Needle
    angle = np.pi * (1 - probability)
    nx = 0.5 + 0.35 * np.cos(angle)
    ny = 0.2 + 0.35 * np.sin(angle)
    ax.annotate("", xy=(nx, ny), xytext=(0.5, 0.2),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5))
    ax.plot(0.5, 0.2, "o", color="white", markersize=8, zorder=5)

    # Labels
    ax.text(0.5, -0.05, f"{probability:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.05, 0.25, "LOW",    color="#2ECC71", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.44, 0.55, "MEDIUM", color="#F39C12", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.80, 0.25, "HIGH",   color="#E74C3C", fontsize=9, fontweight="bold", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.7)
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_feature_bars(input_df, model):
    """Weighted feature contribution bar chart for this specific prediction."""
    importances  = model.feature_importances_
    values       = input_df.values[0]
    weighted     = importances * np.abs(values)
    if weighted.sum() > 0:
        weighted = weighted / weighted.sum()

    name_map = {
        "log_building_dimension":      "Building Dimension (log)",
        "building_dimension":          "Building Dimension",
        "log_building_age":            "Building Age (log)",
        "building_age":                "Building Age",
        "number_of_windows":           "Number of Windows",
        "insured_period":              "Insured Period",
        "transformed_insured_period":  "Insured Period (transformed)",
        "building_type":               "Building Type",
        "building_painted_V":          "Building Painted",
        "residential":                 "Residential",
        "building_fenced_V":           "Building Fenced",
        "garden_V":                    "Has Garden",
        "settlement_U":                "Settlement: Urban",
    }

    contrib_df = (pd.DataFrame({"Feature": FEATURE_COLUMNS, "Contribution": weighted})
                  .sort_values("Contribution", ascending=True)
                  .tail(8))
    contrib_df["Feature"] = contrib_df["Feature"].map(name_map).fillna(contrib_df["Feature"])

    median_val = contrib_df["Contribution"].median()
    colors     = ["#E74C3C" if v > median_val else "#3498DB" for v in contrib_df["Contribution"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1A2E")
    ax.barh(contrib_df["Feature"], contrib_df["Contribution"] * 100,
            color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Contribution to This Prediction (%)", color="white", fontsize=10)
    ax.set_title("Top Feature Contributions", color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("white")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='color:#3498DB; font-size:2.4rem; margin-bottom:0;'>
        🏛️ Building Insurance Claim Predictor
    </h1>
    <p style='color:#AAAAAA; font-size:1.05rem; margin-top:0.4rem;'>
        Enter building characteristics in the sidebar to assess insurance claim risk
    </p>
</div>
<hr style='border:1px solid #333; margin: 0.5rem 0 1.5rem 0;'>
""", unsafe_allow_html=True)

with st.expander("ℹ️  About This App", expanded=False):
    st.markdown("""
    **What this app does:**
    Uses a **tuned Random Forest classifier** trained on 7,014 building records to predict
    the probability that a building will file an insurance claim.

    **Model details:**
    - Algorithm: Random Forest (GridSearchCV tuned — `max_depth=20`, `n_estimators=200`)
    - Training: SMOTE applied to training data only (no data leakage)
    - Cross-validated F1-Score: **0.7921 ± 0.1063** (5-fold CV)
    - Top predictors: building dimension (42.95% importance), building age (25.78%)

    **How to use it:**
    1. Fill in all building details in the sidebar on the left
    2. Click **"🔍 Predict Claim Risk"**
    3. Read the risk gauge, verdict, and feature contribution chart

    **Built by:** Adewale Samson Adeagbo — Data Science Practitioner | Mathematics Educator
    [GitHub](https://github.com/cssadewale) |
    [LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo) |
    *"Learning deliberately. Teaching authentically."*
    """)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SIDEBAR — USER INPUTS
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style='text-align:center;'>
    <h2 style='color:#3498DB;'>🏗️ Building Details</h2>
    <p style='color:#888; font-size:0.85rem;'>Fill in all fields, then click Predict</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("**📋 Policy Information**")

insured_period = st.sidebar.slider(
    "Insured Period (fraction of year)",
    min_value=0.01, max_value=1.0, value=1.0, step=0.01,
    help="1.0 = full-year policy. 0.5 = half-year policy."
)

year_of_observation = st.sidebar.selectbox(
    "Year of Observation",
    options=[2012, 2013, 2014, 2015, 2016], index=2,
    help="The year this building was assessed."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**🏠 Physical Characteristics**")

building_dimension = st.sidebar.number_input(
    "Building Dimension (m²)",
    min_value=1, max_value=25000, value=1000, step=50,
    help="Total floor area in square metres. Average claim building: ~1,996 m². Average non-claim: ~900 m²."
)

number_of_windows = st.sidebar.number_input(
    "Number of Windows",
    min_value=0, max_value=10, value=2, step=1,
    help="Total windows. Claim buildings average 2.83 windows; non-claim average 2.04."
)

date_of_occupancy = st.sidebar.number_input(
    "Year of First Occupancy",
    min_value=1800, max_value=2016, value=1990, step=1,
    help="Year the building was first occupied. Average building age in dataset: 49.1 years."
)

building_type = st.sidebar.selectbox(
    "Building Type",
    options=["Type 1", "Type 2", "Type 3", "Type 4"], index=0,
    help="Type 1 has the lowest claim rate; Type 4 has the highest."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔍 Building Attributes**")

residential = st.sidebar.radio(
    "Residential Status",
    options=["0 (Non-Residential)", "1 (Residential)"], index=0,
    help="Residential buildings claim at 36.0% vs 29.3% for non-residential."
)

building_painted = st.sidebar.radio(
    "Building Painted?",
    options=["N (Not Painted)", "V (Painted)"], index=1,
    help="Painted buildings: 75.68% of the dataset."
)

building_fenced = st.sidebar.radio(
    "Building Fenced?",
    options=["N (Not Fenced)", "V (Fenced)"], index=0,
    help="Unfenced buildings claim at 25.0%; fenced at 21.3%."
)

garden = st.sidebar.radio(
    "Garden Present?",
    options=["O (No Garden)", "V (Has Garden)"], index=0,
    help="No-garden buildings claim at 25.0%; garden buildings at 21.2%."
)

settlement = st.sidebar.radio(
    "Settlement Type",
    options=["R (Rural)", "U (Urban)"], index=0,
    help="Rural buildings claim at 25.0%; urban at 21.2%."
)

st.sidebar.markdown("---")

predict_button = st.sidebar.button(
    "🔍  Predict Claim Risk",
    use_container_width=True,
    type="primary"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if not predict_button:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0; color: #555;'>
        <h2>👈  Fill in building details in the sidebar</h2>
        <p style='font-size:1.1rem;'>Then click <strong>"🔍 Predict Claim Risk"</strong></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Records",   "7,014",   help="After deduplication and cleaning")
    with col2:
        st.metric("Features Used",      "13",       help="After engineering and encoding")
    with col3:
        st.metric("Best CV F1-Score",   "0.7921",  help="5-fold cross-validation on training data")
    with col4:
        st.metric("Class Imbalance",    "3.3 : 1", help="Addressed with SMOTE on training data only")

else:
    # ── Preprocess and predict ────────────────────────────────────────────────
    input_df, building_age = preprocess_input(
        insured_period=insured_period,
        building_dimension=building_dimension,
        number_of_windows=number_of_windows,
        year_of_observation=year_of_observation,
        date_of_occupancy=date_of_occupancy,
        residential=residential,
        building_painted=building_painted,
        building_fenced=building_fenced,
        garden=garden,
        settlement=settlement,
        building_type=building_type,
    )

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ── Risk tier ─────────────────────────────────────────────────────────────
    if probability < 0.33:
        risk_tier   = "LOW RISK"
        risk_color  = "#2ECC71"
        risk_emoji  = "✅"
        risk_advice = "This building has a low claim probability. Standard premium and renewal process apply."
    elif probability < 0.66:
        risk_tier   = "MEDIUM RISK"
        risk_color  = "#F39C12"
        risk_emoji  = "⚠️"
        risk_advice = "Moderate claim probability. Consider underwriter review before policy issuance."
    else:
        risk_tier   = "HIGH RISK"
        risk_color  = "#E74C3C"
        risk_emoji  = "🚨"
        risk_advice = "High claim probability. Physical inspection recommended before renewal or issuance."

    # ── Verdict banner ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{risk_color}22,{risk_color}11);
                border-left:5px solid {risk_color}; border-radius:8px;
                padding:1.2rem 1.5rem; margin-bottom:1.5rem;'>
        <h2 style='color:{risk_color}; margin:0; font-size:1.8rem;'>
            {risk_emoji} &nbsp; {risk_tier}
        </h2>
        <p style='color:#CCCCCC; margin:0.4rem 0 0 0; font-size:1rem;'>{risk_advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Three metric tiles ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    verdict_label = "CLAIM PREDICTED" if prediction == 1 else "NO CLAIM PREDICTED"

    for col, label, value, color in [
        (col1, "Model Verdict",      verdict_label,         risk_color),
        (col2, "Claim Probability",  f"{probability:.1%}",  risk_color),
        (col3, "Building Age",       f"{building_age} years", "#3498DB"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#1A1A2E; border-radius:8px; padding:1rem; text-align:center;'>
                <p style='color:#888; font-size:0.85rem; margin:0;'>{label}</p>
                <h3 style='color:{color}; margin:0.3rem 0; font-size:1.3rem;'>{value}</h3>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + feature bars ──────────────────────────────────────────────────
    col_gauge, col_bars = st.columns([1, 1.4])

    with col_gauge:
        st.markdown("#### 🎯 Risk Gauge")
        st.pyplot(draw_gauge(probability), use_container_width=True)

    with col_bars:
        st.markdown("#### 📊 Feature Contributions")
        st.pyplot(draw_feature_bars(input_df, model), use_container_width=True)
        st.caption("🔴 Red = higher-weight features   🔵 Blue = lower-weight features")

    st.markdown("---")

    # ── Building profile summary ──────────────────────────────────────────────
    st.markdown("#### 📋 Building Profile Summary")
    summary = {
        "Attribute": [
            "Insured Period", "Year of Observation", "Year of Occupancy",
            "Building Age (derived)", "Building Dimension", "Number of Windows",
            "Building Type", "Residential", "Painted", "Fenced", "Garden", "Settlement"
        ],
        "Value": [
            f"{insured_period:.2f} (fraction of year)",
            str(year_of_observation),
            str(date_of_occupancy),
            f"{building_age} years",
            f"{building_dimension:,} m²",
            str(number_of_windows),
            building_type,
            residential,
            building_painted,
            building_fenced,
            garden,
            settlement,
        ]
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Underwriter action guide ──────────────────────────────────────────────
    st.markdown("#### 🏢 Underwriter Action Guide")

    if probability < 0.33:
        actions = [
            "✅ Issue policy at **standard premium** — no loading required",
            "✅ No physical inspection required before issuance",
            "✅ Standard renewal process applies",
        ]
    elif probability < 0.66:
        actions = [
            "⚠️ Flag for **underwriter review** before policy issuance",
            "⚠️ Consider **loading the premium** by 10–20%",
            "⚠️ Request recent property maintenance records from applicant",
        ]
    else:
        actions = [
            "🚨 Refer to **senior underwriter** — do not auto-approve",
            "🚨 Mandatory **physical inspection** before issuance or renewal",
            "🚨 Apply **high-risk premium tier** if accepted",
            "🚨 Consider higher **excess/deductible** or co-insurance clause",
        ]

    for action in actions:
        st.markdown(
            f"<p style='color:{risk_color}; font-size:1rem; margin:0.3rem 0;'>{action}</p>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.85rem; padding:0.5rem 0;'>
    Built by <strong style='color:#3498DB;'>Adewale Samson Adeagbo</strong> —
    Data Science Practitioner | Mathematics Educator |
    <em>"Learning deliberately. Teaching authentically."</em>
    <br>
    <a href='https://github.com/cssadewale' style='color:#3498DB;'>GitHub</a> &nbsp;|&nbsp;
    <a href='https://linkedin.com/in/adewalesamsonadeagbo' style='color:#3498DB;'>LinkedIn</a>
    <br><br>
    Model: Tuned Random Forest (max_depth=20, n_estimators=200) |
    CV F1: 0.7921 | Dataset: 7,014 building records (2012–2016)
</div>
""", unsafe_allow_html=True)
