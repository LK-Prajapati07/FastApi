import streamlit as st
import pandas as pd
import joblib

# ------------------ LOAD MODEL ------------------
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('column.pkl')

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# ------------------ CSS (STATIC CLEAN UI) ------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #1d3557, #457b9d);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #a8dadc;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Button */
div.stButton > button {
    background: #e63946;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 45px;
    border: none;
}

/* Result */
.result-card {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    font-size: 20px;
    font-weight: bold;
}

.high-risk {
    background: rgba(255, 0, 0, 0.2);
}

.low-risk {
    background: rgba(0, 255, 150, 0.2);
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered health risk analysis</div>', unsafe_allow_html=True)

# ------------------ INPUT UI ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧍 Personal Info")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🫀 Heart Metrics")
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Medical Info")
    fasting_bs = st.selectbox("Fasting Sugar >120", [0, 1])
    resting_ecg = st.selectbox("ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Angina", ["Y", "N"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📉 Advanced")
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ BUTTON ------------------
predict_btn = st.button("🚀 Predict Now", use_container_width=True)

# ------------------ PREDICTION ------------------
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

    try:
        prob = model.predict_proba(scaled_input)[0][1]
    except:
        prob = None

    st.write("")

    # ------------------ RESULT ------------------
    if prediction == 1:
        st.markdown(
            '<div class="result-card high-risk">⚠️ High Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-card low-risk">✅ Low Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )

    # ------------------ PROBABILITY ------------------
    if prob is not None:
        st.markdown("### 📊 Risk Probability")
        st.progress(int(prob * 100))
        st.write(f"**Risk Score:** {prob*100:.2f}%")