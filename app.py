import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('column.pkl')

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #e63946;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">❤️ Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details to assess risk</div>', unsafe_allow_html=True)

st.write("")

# Layout: two columns
col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:
    st.markdown("### 🧍 Personal Info")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

    st.markdown("### 🫀 Heart Metrics")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

# RIGHT COLUMN
with col2:
    st.markdown("### 🧪 Medical Details")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

    st.markdown("### 📉 Advanced Metrics")
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("")

# Predict button (centered)
predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

if predict_btn:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.write("")

    # Result Display (Styled)
    if prediction == 1:
        st.markdown(
            """<div style="background-color:#ff4d4d;padding:20px;border-radius:10px;text-align:center;color:white;font-size:22px;">
            ⚠️ High Risk of Heart Disease
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style="background-color:#2ecc71;padding:20px;border-radius:10px;text-align:center;color:white;font-size:22px;">
            ✅ Low Risk of Heart Disease
            </div>""",
            unsafe_allow_html=True
        )