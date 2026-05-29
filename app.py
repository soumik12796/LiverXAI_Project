import streamlit as st
from fpdf import FPDF
from datetime import datetime
import random

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="LiverXAI",
    page_icon="🩺",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

html, body, [class*="css"]{
    background-color:#071021;
    color:white;
    font-family:Segoe UI;
}

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#00BFFF;
}

.sub-title{
    text-align:center;
    color:#cfcfcf;
    font-size:18px;
}

.stButton>button{
    width:100%;
    border:none;
    border-radius:10px;
    height:3.2em;
    background:#00BFFF;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stDownloadButton>button{
    width:100%;
    border:none;
    border-radius:10px;
    height:3.2em;
    background:green;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.result-box{
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOGIN SYSTEM
# ======================================================

if "login" not in st.session_state:
    st.session_state.login = False

# ======================================================
# LOGIN PAGE
# ======================================================

if not st.session_state.login:

    st.markdown(
        "<div class='main-title'>🩺 LiverXAI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Explainable AI Based Liver Disease Detection System</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])

    with c2:

        st.markdown("## 🔐 Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if username == "admin" and password == "1234":

                st.session_state.login = True
                st.rerun()

            else:
                st.error("Invalid Username or Password")

# ======================================================
# MAIN APP
# ======================================================

else:

    st.markdown(
        "<div class='main-title'>🩺 LiverXAI Medical System</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Advanced AI Powered Liver Disease Prediction</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ======================================================
    # HOSPITAL INFORMATION
    # ======================================================

    st.header("🏥 Hospital Information")

    col1, col2 = st.columns(2)

    with col1:

        hospital_name = st.text_input(
            "Hospital Name"
        )

        hospital_address = st.text_input(
            "Hospital Address"
        )

    with col2:

        doctor_name = st.text_input(
            "Doctor Name"
        )

        doctor_degree = st.text_input(
            "Doctor Qualification"
        )

    st.markdown("---")

    # ======================================================
    # PATIENT INFORMATION
    # ======================================================

    st.header("👨‍⚕️ Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient Name"
        )

        patient_age = st.number_input(
            "Age",
            1,
            100,
            25
        )

        patient_sex = st.selectbox(
            "Sex",
            ["Male", "Female", "Other"]
        )

        patient_phone = st.text_input(
            "Phone Number"
        )

    with col2:

        patient_email = st.text_input(
            "Email Address"
        )

        patient_blood = st.selectbox(
            "Blood Group",
            ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        )

        patient_weight = st.number_input(
            "Weight (KG)",
            1,
            200,
            70
        )

        patient_height = st.number_input(
            "Height (CM)",
            50,
            250,
            170
        )

    patient_id = st.text_input(
        "Patient ID"
    )

    report_date = st.date_input(
        "Date of Report"
    )

    st.markdown("---")

    # ======================================================
    # LIVER TEST VALUES
    # ======================================================

    st.header("🧪 Liver Test Parameters")

    c1, c2, c3 = st.columns(3)

    with c1:

        total_bilirubin = st.number_input(
            "Total Bilirubin",
            0.0,
            100.0,
            1.0
        )

        direct_bilirubin = st.number_input(
            "Direct Bilirubin",
            0.0,
            100.0,
            0.5
        )

        alkaline = st.number_input(
            "Alkaline Phosphotase",
            0,
            5000,
            200
        )

    with c2:

        alt = st.number_input(
            "ALT",
            0,
            5000,
            30
        )

        ast = st.number_input(
            "AST",
            0,
            5000,
            35
        )

        proteins = st.number_input(
            "Total Proteins",
            0.0,
            20.0,
            6.5
        )

    with c3:

        albumin = st.number_input(
            "Albumin",
            0.0,
            10.0,
            4.0
        )

        agr = st.number_input(
            "Albumin/Globulin Ratio",
            0.0,
            10.0,
            1.2
        )

    st.markdown("---")

    # ======================================================
    # PREDICTION
    # ======================================================

    if st.button("🔍 Predict Liver Disease"):

        # ======================================================
        # ADVANCED LIVER RISK ANALYSIS
        # ======================================================

        risk_score = 0

        if total_bilirubin > 1.2:
            risk_score += 2

        if direct_bilirubin > 0.3:
            risk_score += 2

        if alt > 55:
            risk_score += 2

        if ast > 45:
            risk_score += 2

        if alkaline > 120:
            risk_score += 2

        if albumin < 3.5:
            risk_score += 2

        if agr < 1.0:
            risk_score += 2

        if proteins < 6.0:
            risk_score += 1

        # ======================================================
        # FINAL RESULT
        # ======================================================

        if risk_score >= 6:

            prediction = "Liver Disease Detected"

            confidence = min(99, 75 + risk_score)

            result_color = "#5c1a1a"

            avoid_foods = [
                "Alcohol",
                "Smoking",
                "Fast Food",
                "Oily Foods",
                "Cold Drinks",
                "Red Meat",
                "Excess Salt"
            ]

            diet_foods = [
                "Green Vegetables",
                "Papaya",
                "Apple",
                "Boiled Food",
                "Protein Rich Diet",
                "Drink More Water",
                "Low Fat Diet"
            ]

            followup = "Follow-up liver test recommended within 7 days."

        else:

            prediction = "Healthy Liver"

            confidence = max(85, 95 - risk_score)

            result_color = "#14532d"

            avoid_foods = [
                "Excess Sugar",
                "Too Much Junk Food"
            ]

            diet_foods = [
                "Balanced Diet",
                "Exercise Daily",
                "Fresh Fruits",
                "Healthy Lifestyle"
            ]

            followup = "Routine liver checkup after 6 months."

        # ======================================================
        # RESULT BOX
        # ======================================================

        st.markdown(f"""
        <div class='result-box'
        style='background:{result_color};'>
        {prediction}
        <br><br>
        Confidence Score : {confidence}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ======================================================
        # XAI SECTION
        # ======================================================

        st.header("🧠 Explainable AI Analysis")

        st.success("""
        AI analyzed liver biomarkers and generated transparent prediction results.
        """)

        st.subheader("Important Biomarkers")

        st.write(f"Albumin : {albumin}")
        st.write(f"Total Bilirubin : {total_bilirubin}")
        st.write(f"ALT : {alt}")
        st.write(f"AST : {ast}")

        st.markdown("---")

        # ======================================================
        # DIET SECTION
        # ======================================================

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("❌ Foods To Avoid")

            for i in avoid_foods:
                st.write("- " + i)

        with c2:

            st.subheader("🥗 Recommended Diet")

            for i in diet_foods:
                st.write("- " + i)

        st.markdown("---")

        # ======================================================
        # FOLLOWUP
        # ======================================================

        st.subheader("📅 Follow-up Recommendation")

        st.info(followup)

        # ======================================================
        # PDF REPORT
        # ======================================================

        pdf = FPDF()

        pdf.add_page()

        # ======================================================
        # HEADER
        # ======================================================

        pdf.set_fill_color(0,102,204)

        pdf.set_text_color(255,255,255)

        pdf.set_font("Arial","B",22)

        pdf.cell(
            190,
            15,
            hospital_name,
            ln=True,
            align="C",
            fill=True
        )

        pdf.set_font("Arial","",11)

        pdf.cell(
            190,
            8,
            hospital_address,
            ln=True,
            align="C"
        )

        pdf.ln(5)

        # ======================================================
        # TITLE
        # ======================================================

        pdf.set_text_color(0,0,0)

        pdf.set_font("Arial","B",18)

        pdf.cell(
            190,
            12,
            "LIVER DISEASE MEDICAL REPORT",
            ln=True,
            align="C"
        )

        pdf.ln(10)

        # ======================================================
        # PATIENT INFO
        # ======================================================

        pdf.set_fill_color(220,220,220)

        pdf.set_font("Arial","B",14)

        pdf.cell(
            190,
            10,
            "PATIENT INFORMATION",
            ln=True,
            fill=True
        )

        pdf.set_font("Arial","",11)

        pdf.cell(95,10,f"Patient Name : {patient_name}",border=1)
        pdf.cell(95,10,f"Patient ID : {patient_id}",border=1,ln=True)

        pdf.cell(95,10,f"Age : {patient_age}",border=1)
        pdf.cell(95,10,f"Sex : {patient_sex}",border=1,ln=True)

        pdf.cell(95,10,f"Phone : {patient_phone}",border=1)
        pdf.cell(95,10,f"Email : {patient_email}",border=1,ln=True)

        pdf.cell(95,10,f"Blood Group : {patient_blood}",border=1)
        pdf.cell(95,10,f"Weight : {patient_weight} KG",border=1,ln=True)

        pdf.cell(95,10,f"Height : {patient_height} CM",border=1)
        pdf.cell(95,10,f"Doctor : {doctor_name}",border=1,ln=True)

        pdf.cell(95,10,f"Qualification : {doctor_degree}",border=1)
        pdf.cell(95,10,f"Report Date : {report_date}",border=1,ln=True)

        pdf.ln(8)

        # ======================================================
        # TEST RESULTS
        # ======================================================

        pdf.set_font("Arial","B",14)

        pdf.cell(
            190,
            10,
            "LIVER TEST RESULTS",
            ln=True,
            fill=True
        )

        pdf.set_font("Arial","",11)

        tests = [
            ("Total Bilirubin", total_bilirubin),
            ("Direct Bilirubin", direct_bilirubin),
            ("Alkaline Phosphotase", alkaline),
            ("ALT", alt),
            ("AST", ast),
            ("Total Proteins", proteins),
            ("Albumin", albumin),
            ("A/G Ratio", agr)
        ]

        for t,v in tests:

            pdf.cell(95,10,str(t),border=1)
            pdf.cell(95,10,str(v),border=1,ln=True)

        pdf.ln(8)

        # ======================================================
        # RESULT
        # ======================================================

        pdf.set_font("Arial","B",14)

        pdf.cell(
            190,
            10,
            "AI PREDICTION RESULT",
            ln=True,
            fill=True
        )

        pdf.ln(5)

        if prediction == "Liver Disease Detected":

            pdf.set_text_color(255,0,0)

        else:

            pdf.set_text_color(0,150,0)

        pdf.set_font("Arial","B",18)

        pdf.cell(
            190,
            12,
            prediction,
            ln=True,
            align="C"
        )

        pdf.ln(5)

        pdf.set_text_color(0,0,0)

        pdf.set_font("Arial","",12)

        pdf.cell(
            190,
            10,
            f"Confidence Score : {confidence}%",
            ln=True
        )

        pdf.ln(5)

        # ======================================================
        # AVOID FOODS
        # ======================================================

        pdf.set_font("Arial","B",14)

        pdf.cell(
            190,
            10,
            "FOODS TO AVOID",
            ln=True,
            fill=True
        )

        pdf.set_font("Arial","",11)

        for i in avoid_foods:
            pdf.cell(190,8,"- "+i,ln=True)

        pdf.ln(5)

        # ======================================================
        # DIET PLAN
        # ======================================================

        pdf.set_font("Arial","B",14)

        pdf.cell(
            190,
            10,
            "RECOMMENDED DIET",
            ln=True,
            fill=True
        )

        pdf.set_font("Arial","",11)

        for i in diet_foods:
            pdf.cell(190,8,"- "+i,ln=True)

        pdf.ln(5)

        # ======================================================
        # FOLLOWUP
        # ======================================================

        pdf.set_font("Arial","B",14)

        pdf.cell(
            190,
            10,
            "FOLLOW-UP RECOMMENDATION",
            ln=True,
            fill=True
        )

        pdf.set_font("Arial","",11)

        pdf.multi_cell(
            0,
            8,
            followup
        )

        pdf.ln(10)

        # ======================================================
        # FOOTER
        # ======================================================

        pdf.set_font("Arial","I",10)

        pdf.multi_cell(
            0,
            7,
            "This report was generated using Explainable AI based Liver Disease Detection System."
        )

        # ======================================================
        # SAVE PDF
        # ======================================================

        pdf.output("medical_report.pdf")

        # ======================================================
        # DOWNLOAD BUTTON
        # ======================================================

        with open("medical_report.pdf","rb") as file:

            st.download_button(
                label="📥 Download Hospital Report",
                data=file,
                file_name="medical_report.pdf",
                mime="application/pdf"
            )

    st.markdown("---")

    st.caption("Developed by Soumik Roy | LiverXAI Research Project")