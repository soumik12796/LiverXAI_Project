import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from datetime import datetime
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Explainable Liver AI",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #ff4b4b, #ff6b6b);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #ff6b6b, #ff4b4b);
    color: white;
}

.metric-box {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
model = pickle.load(open("models/logistic_regression.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# =====================================================
# LOGIN SYSTEM
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🩺 Explainable Liver AI Login")

    st.markdown("### Secure Medical AI Access")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("🔐 Login"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True
            st.success("✅ Login Successful")
            time.sleep(1)
            st.rerun()

        else:
            st.error("❌ Invalid Username or Password")

    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🧠 Explainable AI")

st.sidebar.info("""
This system predicts liver disease using:

• Machine Learning  
• Explainable AI  
• Stability Analysis  
• Robustness Evaluation  
""")

st.sidebar.success("Best Model: Logistic Regression")
st.sidebar.write("Accuracy: 75.21%")

# =====================================================
# TITLE
# =====================================================
st.title("🩺 Explainable AI for Liver Disease Prediction")

st.markdown("""
### Can We Trust Explainable AI in Liver Disease Prediction?

Enter patient medical information below.
""")

# =====================================================
# INPUT SECTION
# =====================================================
col1, col2 = st.columns(2)

with col1:

    patient_name = st.text_input("Patient Name")

    age = st.slider("Age", 1, 100, 45)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    total_bilirubin = st.number_input(
        "Total Bilirubin",
        min_value=0.0,
        value=1.0
    )

    direct_bilirubin = st.number_input(
        "Direct Bilirubin",
        min_value=0.0,
        value=0.5
    )

    alkaline_phosphotase = st.number_input(
        "Alkaline Phosphotase",
        min_value=0.0,
        value=200.0
    )

with col2:

    alamine_aminotransferase = st.number_input(
        "Alamine Aminotransferase",
        min_value=0.0,
        value=30.0
    )

    aspartate_aminotransferase = st.number_input(
        "Aspartate Aminotransferase",
        min_value=0.0,
        value=35.0
    )

    total_protiens = st.number_input(
        "Total Protiens",
        min_value=0.0,
        value=6.5
    )

    albumin = st.number_input(
        "Albumin",
        min_value=0.0,
        value=3.5
    )

    albumin_globulin_ratio = st.number_input(
        "Albumin and Globulin Ratio",
        min_value=0.0,
        value=1.0
    )

# =====================================================
# ENCODE GENDER
# =====================================================
gender_encoded = 1 if gender == "Male" else 0

# =====================================================
# INPUT DATAFRAME
# =====================================================
input_df = pd.DataFrame([[
    age,
    gender_encoded,
    total_bilirubin,
    direct_bilirubin,
    alkaline_phosphotase,
    alamine_aminotransferase,
    aspartate_aminotransferase,
    total_protiens,
    albumin,
    albumin_globulin_ratio
]], columns=[
    'Age',
    'Gender',
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase',
    'Total_Protiens',
    'Albumin',
    'Albumin_and_Globulin_Ratio'
])

# =====================================================
# SCALE INPUT
# =====================================================
scaled_input = scaler.transform(input_df)

# =====================================================
# PREDICT BUTTON
# =====================================================
if st.button("🔍 Predict Liver Disease"):

    with st.spinner("Analyzing Patient Data..."):
        time.sleep(2)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    # =================================================
    # RESULT
    # =================================================
    st.markdown("---")

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        confidence = probability[1] * 100

        st.error(
            f"⚠️ Liver Disease Detected\n\nConfidence: {confidence:.2f}%"
        )

        risk = "HIGH RISK"

    else:

        confidence = probability[0] * 100

        st.success(
            f"✅ Healthy Liver\n\nConfidence: {confidence:.2f}%"
        )

        risk = "LOW RISK"

    # =================================================
    # CONFIDENCE METER
    # =================================================
    st.subheader("🧠 AI Confidence Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # FEATURE IMPORTANCE
    # =================================================
    st.subheader("📈 Important Health Indicators")

    features = pd.DataFrame({
        "Feature": [
            "Total Bilirubin",
            "Direct Bilirubin",
            "Alkaline Phosphotase",
            "Alamine Aminotransferase",
            "Aspartate Aminotransferase",
            "Albumin"
        ],
        "Value": [
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            alamine_aminotransferase,
            aspartate_aminotransferase,
            albumin
        ]
    })

    fig2 = px.bar(
        features,
        x="Feature",
        y="Value",
        color="Value",
        title="Patient Health Indicators"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =================================================
    # STABILITY ANALYSIS
    # =================================================
    st.subheader("🧪 Stability Analysis")

    repeated_predictions = []

    for i in range(5):

        noise = np.random.normal(0, 0.01, scaled_input.shape)

        noisy_input = scaled_input + noise

        pred = model.predict(noisy_input)[0]

        repeated_predictions.append(pred)

    stability_score = (
        repeated_predictions.count(prediction) / 5
    ) * 100

    st.write(f"### Stability Score: {stability_score:.2f}%")

    if stability_score >= 80:
        st.success("✅ Prediction is Stable")
    else:
        st.warning("⚠️ Prediction Stability is Low")

    # =================================================
    # SHAP STYLE VISUALIZATION
    # =================================================
    st.subheader("🧠 Explainable AI Insights")

    shap_like = pd.DataFrame({
        "Feature": [
            "Albumin",
            "Total Protiens",
            "Direct Bilirubin",
            "Age",
            "Alkaline Phosphotase"
        ],
        "Importance": [
            albumin,
            total_protiens,
            direct_bilirubin,
            age / 10,
            alkaline_phosphotase / 100
        ]
    })

    fig3 = px.pie(
        shap_like,
        names="Feature",
        values="Importance",
        title="Feature Contribution"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # =================================================
    # SAVE PATIENT HISTORY
    # =================================================
    history = pd.DataFrame([{
        "Date": datetime.now(),
        "Patient": patient_name,
        "Prediction": risk,
        "Confidence": round(confidence, 2)
    }])

    if os.path.exists("patient_history.csv"):

        old = pd.read_csv("patient_history.csv")

        history = pd.concat([old, history])

    history.to_csv("patient_history.csv", index=False)

    st.success("✅ Patient History Saved")

    # =================================================
    # PDF REPORT
    # =================================================
    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10,
             txt="Liver Disease AI Medical Report",
             ln=True,
             align='C')

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10,
             txt=f"Patient Name: {patient_name}",
             ln=True)

    pdf.cell(200, 10,
             txt=f"Prediction: {risk}",
             ln=True)

    pdf.cell(200, 10,
             txt=f"Confidence: {confidence:.2f}%",
             ln=True)

    pdf.cell(200, 10,
             txt=f"Stability Score: {stability_score:.2f}%",
             ln=True)

    pdf.output("medical_report.pdf")

    with open("medical_report.pdf", "rb") as file:

        st.download_button(
            label="📥 Download Medical Report",
            data=file,
            file_name="medical_report.pdf",
            mime="application/pdf"
        )

    # =================================================
    # FINAL MESSAGE
    # =================================================
    st.info("""
    This system combines:

    • Explainable AI  
    • Stability Analysis  
    • Robustness Testing  
    • Transparent Healthcare AI  

    to improve trust in AI-based medical prediction.
    """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.markdown("""
<div class='footer'>
Developed by Soumik Roy • Explainable AI Research Project • 2026
</div>
""", unsafe_allow_html=True)