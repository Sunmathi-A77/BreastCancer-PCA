import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="centered"
)

# ---------------------- CSS Styling ----------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e3c7ff 0%, #c2e9fb 100%);
        color: #2f004f;
    }
    h1, h2, h3 {
        color: #4b0082 !important;
        text-align: center;
    }
    input {
        background-color: #f8f0ff !important;
        border: 1px solid #a56eff !important;
        border-radius: 5px !important;
        color: #3b0066 !important;
    }
    .stButton>button {
        background-color: #a56eff !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6em 1.2em !important;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #8c5ef3 !important;
        transform: scale(1.05);
    }
    .result-card {
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
    }
    .prob-bar {
        height: 25px;
        border-radius: 10px;
        margin-top: 10px;
        background-color: #e0e0e0;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        text-align: center;
        font-weight: bold;
        color: white;
        line-height: 25px;
    }
    footer {
        color: #4b0082;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Load Preprocessing + Model ----------------------
pt = pickle.load(open("pt.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
svm = pickle.load(open("svm.pkl", "rb"))

# ---------------------- Features & Default Values ----------------------
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

default_values = [
    7.76,24.54,47.92,181.0,0.05263,0.04362,0,0,0.1587,0.05884,
    0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,
    9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039
]

st.title("🩺 Breast Cancer Prediction App")
st.subheader("Enter Feature Values:")

# ---------------------- 2-column Editable Input ----------------------
# Create input_df with float dtype to avoid FutureWarning
input_df = pd.DataFrame([default_values], columns=features, dtype=float)

col1, col2 = st.columns(2)
for i in range(15):
    with col1:
        input_df.at[0, features[i]] = st.number_input(
            f"{features[i]}",
            value=float(input_df.at[0, features[i]]),
            format="%.4f"
        )
for i in range(15, 30):
    with col2:
        input_df.at[0, features[i]] = st.number_input(
            f"{features[i]}",
            value=float(input_df.at[0, features[i]]),
            format="%.4f"
        )

# ---------------------- Prediction ----------------------
if st.button("🔍 Predict"):
    skewed_features = [
        'area_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
        'fractal_dimension_mean', 'radius_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'perimeter_worst', 'area_worst',
        'compactness_worst', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    # Preprocess
    input_df[skewed_features] = pt.transform(input_df[skewed_features])
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)

    # Prediction
    pred = svm.predict(input_pca)[0]
    prob = svm.predict_proba(input_pca)[0][1]

    if pred == 1:
        st.markdown(f'<div class="result-card malignant">⚠️ Malignant (Cancer Detected)</div>', unsafe_allow_html=True)
        fill_color = "#d9534f"
        display_prob = prob*100
    else:
        st.markdown(f'<div class="result-card benign">✅ Benign (No Cancer)</div>', unsafe_allow_html=True)
        fill_color = "#28a745"
        display_prob = (1-prob)*100

    st.markdown(f'''
        <div class="prob-bar">
            <div class="prob-fill" style="width:{display_prob}%; background-color:{fill_color};">
                {display_prob:.2f}%
            </div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed by Sunmathi 🩷 | Breast Cancer PCA-SVM Model")
