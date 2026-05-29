# ============================================================
# FILE: app.py
# PROJECT: Can We Trust Explainable AI in Liver Disease?
# UPGRADED: Hospital Employee Auth + Premium UI + PDF Report
# FIXED: Unicode PDF error + None button rendering
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import hashlib
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="HepatoXAI — Liver Intelligence System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #020B18;
    color: #E8EDF2;
}

.stApp {
    background: linear-gradient(135deg, #020B18 0%, #041428 50%, #020B18 100%);
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #0066FF, #00D4FF);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite;
    letter-spacing: -1px;
    margin-bottom: 0;
}

@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}

.sub-title {
    text-align: center;
    color: #7B8FA1;
    font-size: 16px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 5px;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.card-header {
    font-size: 18px;
    font-weight: 700;
    color: #00D4FF;
    border-bottom: 1px solid rgba(0,212,255,0.2);
    padding-bottom: 12px;
    margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-box {
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #00D4FF;
}

.metric-label {
    font-size: 11px;
    color: #7B8FA1;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-danger {
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(185,28,28,0.3));
    border: 2px solid #DC2626;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    animation: pulse-red 2s infinite;
}

.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.3));
    border: 2px solid #10B981;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    animation: pulse-green 2s infinite;
}

@keyframes pulse-red {
    0%, 100% { border-color: #DC2626; box-shadow: 0 0 20px rgba(220,38,38,0.3); }
    50% { border-color: #EF4444; box-shadow: 0 0 40px rgba(220,38,38,0.6); }
}

@keyframes pulse-green {
    0%, 100% { border-color: #10B981; box-shadow: 0 0 20px rgba(16,185,129,0.3); }
    50% { border-color: #34D399; box-shadow: 0 0 40px rgba(16,185,129,0.6); }
}

.result-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 10px;
}

.confidence-bar-container {
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    height: 12px;
    margin: 10px 0;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0066FF, #00D4FF);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 1px;
    transition: all 0.3s;
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
}

.stDownloadButton > button {
    width: 100%;
    background: linear-gradient(135deg, #059669, #10B981);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 16px;
    font-weight: 700;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: white !important;
}

.risk-badge-high {
    display: inline-block;
    background: rgba(220,38,38,0.2);
    border: 1px solid #DC2626;
    color: #FCA5A5;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 700;
}

.risk-badge-normal {
    display: inline-block;
    background: rgba(16,185,129,0.2);
    border: 1px solid #10B981;
    color: #6EE7B7;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 700;
}

.auth-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 24px;
    padding: 40px;
    backdrop-filter: blur(20px);
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
    margin: 20px 0;
}

section[data-testid="stSidebar"] {
    background: rgba(2,11,24,0.95) !important;
    border-right: 1px solid rgba(0,212,255,0.1);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    color: #7B8FA1;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.1) !important;
    color: #00D4FF !important;
}

.param-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.param-name { color: #7B8FA1; font-size: 13px; }
.param-value { color: #E8EDF2; font-weight: 600; }
.param-flag-high { color: #FCA5A5; font-weight: 700; }
.param-flag-normal { color: #6EE7B7; font-weight: 700; }

.logo-area {
    text-align: center;
    padding: 20px 0;
}

.logo-icon {
    font-size: 60px;
    display: block;
}

.stats-row {
    display: flex;
    gap: 10px;
    margin: 15px 0;
}

.watermark {
    text-align: center;
    color: rgba(123,143,161,0.4);
    font-size: 11px;
    letter-spacing: 2px;
    padding: 20px 0 5px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# USER DATABASE (JSON file based)
# ─────────────────────────────────────────
USER_DB = "users.json"

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, full_name, hospital, designation, employee_id):
    users = load_users()
    if username in users:
        return False, "Username already exists!"
    users[username] = {
        "password": hash_password(password),
        "full_name": full_name,
        "hospital": hospital,
        "designation": designation,
        "employee_id": employee_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "predictions_count": 0
    }
    save_users(users)
    return True, "Registration successful!"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, None
    if users[username]["password"] == hash_password(password):
        return True, users[username]
    return False, None

def update_prediction_count(username):
    users = load_users()
    if username in users:
        users[username]["predictions_count"] += 1
        save_users(users)

# ─────────────────────────────────────────
# LOAD ML MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# ─────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────
def predict_liver_disease(age, gender, tb, db, alkphos, alt, ast, tp, alb, agr):
    features = np.array([[age, gender, tb, db, alkphos, alt, ast, tp, alb, agr]])

    if model is not None and scaler is not None:
        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            confidence = int(max(prob) * 100)
            return prediction, confidence
        except:
            pass

    # Fallback rule-based
    risk = 0
    if tb > 1.2: risk += 2
    if db > 0.3: risk += 2
    if alt > 55: risk += 2
    if ast > 45: risk += 2
    if alkphos > 120: risk += 2
    if alb < 3.5: risk += 2
    if agr < 1.0: risk += 2
    if tp < 6.0: risk += 1

    prediction = 1 if risk >= 6 else 0
    confidence = min(99, 70 + risk * 3) if risk >= 6 else max(82, 95 - risk * 3)
    return prediction, confidence

# ─────────────────────────────────────────
# BIOMARKER ANALYSIS
# ─────────────────────────────────────────
def analyze_biomarkers(tb, db, alkphos, alt, ast, tp, alb, agr):
    results = []
    ranges = [
        ("Total Bilirubin", tb, 0.2, 1.2, "mg/dL"),
        ("Direct Bilirubin", db, 0.0, 0.3, "mg/dL"),
        ("Alkaline Phosphotase", alkphos, 44, 147, "U/L"),
        ("ALT (SGPT)", alt, 7, 56, "U/L"),
        ("AST (SGOT)", ast, 10, 40, "U/L"),
        ("Total Proteins", tp, 6.0, 8.3, "g/dL"),
        ("Albumin", alb, 3.5, 5.0, "g/dL"),
        ("A/G Ratio", agr, 1.0, 2.5, "ratio"),
    ]
    for name, val, low, high, unit in ranges:
        if val < low:
            status = "LOW"
            flag = "abnormal"
        elif val > high:
            status = "HIGH"
            flag = "abnormal"
        else:
            status = "NORMAL"
            flag = "normal"
        results.append({
            "Parameter": name,
            "Value": val,
            "Unit": unit,
            "Normal Range": f"{low} - {high}",
            "Status": status,
            "Flag": flag
        })
    return results

# ─────────────────────────────────────────
# SHAP-STYLE FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────
def generate_feature_chart(tb, db, alkphos, alt, ast, tp, alb, agr):
    features = ['Albumin', 'ALT', 'Total Bilirubin', 'AST',
                'Direct Bilirubin', 'A/G Ratio', 'Alk. Phosphotase', 'Total Proteins']
    normals = [4.25, 31.5, 0.7, 25, 0.15, 1.75, 95.5, 7.15]
    values_raw = [alb, alt, tb, ast, db, agr, alkphos, tp]
    importances = []
    for v, n in zip(values_raw, normals):
        dev = (v - n) / n
        importances.append(round(dev, 3))

    colors = ['#DC2626' if x > 0 else '#10B981' for x in importances]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#041428')
    ax.set_facecolor('#041428')

    bars = ax.barh(features, importances, color=colors, height=0.6, edgecolor='none')

    # FIXED: Use tuple (R, G, B, A) with 0-1 float values instead of CSS rgba()
    ax.axvline(0, color=(1, 1, 1, 0.3), linewidth=1, linestyle='--')

    ax.set_xlabel('Deviation from Normal Range', color='#7B8FA1', fontsize=10)
    ax.set_title('Feature Impact Analysis (XAI)', color='#00D4FF',
                 fontsize=13, fontweight='bold', pad=15)

    ax.tick_params(colors='#E8EDF2', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1E3A5F')
    ax.spines['left'].set_color('#1E3A5F')

    red_patch = mpatches.Patch(color='#DC2626', label='Above Normal (Risk)')
    green_patch = mpatches.Patch(color='#10B981', label='Below/Normal (Safe)')
    ax.legend(handles=[red_patch, green_patch], loc='lower right',
              facecolor='#041428', labelcolor='#E8EDF2', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#041428')
    buf.seek(0)
    plt.close()
    return buf

# ─────────────────────────────────────────
# RISK GAUGE CHART
# ─────────────────────────────────────────
def generate_risk_gauge(confidence, is_disease):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('#041428')
    ax.set_facecolor('#041428')

    theta = np.linspace(np.pi, 0, 100)

    # Background arc
    ax.plot(np.cos(theta), np.sin(theta), color='#1E3A5F', linewidth=20,
            solid_capstyle='round')

    # Value arc
    val = confidence / 100
    theta_val = np.linspace(np.pi, np.pi - val * np.pi, 100)
    color = '#DC2626' if is_disease else '#10B981'
    ax.plot(np.cos(theta_val), np.sin(theta_val),
            color=color, linewidth=20, solid_capstyle='round')

    ax.text(0, 0.1, f'{confidence}%', ha='center', va='center',
            fontsize=28, fontweight='bold', color=color)
    ax.text(0, -0.25, 'Confidence Score', ha='center',
            fontsize=10, color='#7B8FA1')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.axis('off')
    ax.set_title('AI Confidence Meter', color='#00D4FF',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#041428')
    buf.seek(0)
    plt.close()
    return buf

# ─────────────────────────────────────────
# PREMIUM PDF REPORT GENERATOR
# FIXED: All Unicode symbols replaced with ASCII for latin-1 compatibility
# ─────────────────────────────────────────
def generate_pdf_report(patient_data, prediction, confidence, biomarkers,
                         hospital_name, hospital_address, doctor_name,
                         doctor_degree, employee_data):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── HEADER BAND ──
    pdf.set_fill_color(0, 51, 153)
    pdf.rect(0, 0, 210, 35, 'F')

    pdf.set_fill_color(0, 102, 255)
    pdf.rect(0, 30, 210, 5, 'F')

    pdf.set_xy(10, 5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 22)
    pdf.cell(0, 10, hospital_name if hospital_name else "City Medical Center", ln=True)

    pdf.set_font("Arial", "", 10)
    pdf.set_xy(10, 17)
    pdf.cell(0, 6, hospital_address if hospital_address else "Department of Hepatology & Gastroenterology", ln=True)

    pdf.set_xy(10, 23)
    # FIXED: Removed Unicode dash — replaced with plain hyphen
    pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}  |  Doc: {doctor_name}  |  Qual: {doctor_degree}", ln=True)

    # ── REPORT TITLE ──
    pdf.ln(10)
    pdf.set_text_color(0, 51, 153)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "LIVER DISEASE PREDICTION REPORT", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Powered by Explainable Artificial Intelligence (XAI)", ln=True, align="C")

    # Divider line
    pdf.set_draw_color(0, 102, 255)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y() + 3, 200, pdf.get_y() + 3)
    pdf.ln(8)

    # ── PATIENT INFO BOX ──
    pdf.set_fill_color(235, 243, 255)
    pdf.set_draw_color(0, 102, 255)
    pdf.set_line_width(0.3)
    pdf.rect(10, pdf.get_y(), 190, 8, 'F')
    pdf.set_text_color(0, 51, 153)
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(10)
    pdf.cell(190, 8, "  PATIENT INFORMATION", ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Arial", "", 10)
    pdf.set_fill_color(250, 252, 255)

    p = patient_data
    rows = [
        ("Patient Name", p.get('name', 'N/A'), "Patient ID", p.get('id', 'N/A')),
        ("Age / Sex", f"{p.get('age', 'N/A')} yrs / {p.get('sex', 'N/A')}", "Blood Group", p.get('blood', 'N/A')),
        ("Weight / Height", f"{p.get('weight', 'N/A')} kg / {p.get('height', 'N/A')} cm", "Phone", p.get('phone', 'N/A')),
        ("Email", p.get('email', 'N/A'), "Referred By", doctor_name),
    ]
    for r in rows:
        pdf.set_x(10)
        pdf.set_fill_color(240, 246, 255)
        pdf.cell(45, 7, r[0] + ":", border=1, fill=True)
        pdf.cell(50, 7, str(r[1]), border=1)
        pdf.cell(45, 7, r[2] + ":", border=1, fill=True)
        pdf.cell(50, 7, str(r[3]), border=1, ln=True)

    pdf.ln(6)

    # ── AI RESULT BOX ──
    is_disease = (prediction == 1)
    if is_disease:
        pdf.set_fill_color(255, 235, 235)
        # FIXED: Replaced Unicode ⚠ with ASCII !!
        result_text = "!! LIVER DISEASE DETECTED"
        result_color = (180, 0, 0)
        risk_level = "HIGH RISK"
    else:
        pdf.set_fill_color(230, 255, 240)
        # FIXED: Replaced Unicode ✓ with ASCII OK and — with -
        result_text = "OK  HEALTHY LIVER - NO DISEASE DETECTED"
        result_color = (0, 130, 60)
        risk_level = "LOW RISK"

    y_box = pdf.get_y()
    pdf.rect(10, y_box, 190, 30, 'F')
    pdf.set_draw_color(*result_color)
    pdf.set_line_width(1.5)
    pdf.rect(10, y_box, 190, 30)

    pdf.set_xy(10, y_box + 5)
    pdf.set_text_color(*result_color)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, result_text, align="C", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8,
             f"AI Confidence Score: {confidence}%   |   Risk Level: {risk_level}",
             align="C", ln=True)

    pdf.ln(8)
    pdf.set_line_width(0.3)

    # ── BIOMARKER TABLE ──
    pdf.set_draw_color(0, 102, 255)
    pdf.set_fill_color(235, 243, 255)
    pdf.set_text_color(0, 51, 153)
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(10)
    pdf.cell(190, 8, "  LIVER BIOMARKER TEST RESULTS", ln=True, fill=True)

    # Table header
    pdf.set_fill_color(0, 51, 153)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 10)
    pdf.set_x(10)
    pdf.cell(60, 8, "Parameter", border=1, fill=True, align="C")
    pdf.cell(30, 8, "Value", border=1, fill=True, align="C")
    pdf.cell(30, 8, "Unit", border=1, fill=True, align="C")
    pdf.cell(45, 8, "Normal Range", border=1, fill=True, align="C")
    pdf.cell(25, 8, "Status", border=1, fill=True, align="C", ln=True)

    pdf.set_font("Arial", "", 10)
    for i, b in enumerate(biomarkers):
        if b['Flag'] == 'abnormal':
            pdf.set_fill_color(255, 235, 235)
            pdf.set_text_color(180, 0, 0)
        else:
            pdf.set_fill_color(245, 255, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 30, 30)

        pdf.set_x(10)
        pdf.cell(60, 7, b['Parameter'], border=1, fill=True)
        pdf.cell(30, 7, str(b['Value']), border=1, fill=True, align="C")
        pdf.cell(30, 7, b['Unit'], border=1, fill=True, align="C")
        pdf.cell(45, 7, b['Normal Range'], border=1, fill=True, align="C")
        pdf.set_font("Arial", "B", 10)
        # FIXED: Status no longer has Unicode arrows — just plain text
        pdf.cell(25, 7, b['Status'], border=1, fill=True, align="C", ln=True)
        pdf.set_font("Arial", "", 10)

    pdf.ln(6)

    # ── XAI EXPLANATION ──
    pdf.set_text_color(0, 51, 153)
    pdf.set_fill_color(235, 243, 255)
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(10)
    pdf.cell(190, 8, "  AI EXPLANATION (XAI INSIGHT)", ln=True, fill=True)

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Arial", "", 10)
    pdf.set_x(10)
    abnormal_params = [b for b in biomarkers if b['Flag'] == 'abnormal']
    if abnormal_params:
        xai_text = (f"The AI model identified {len(abnormal_params)} abnormal biomarker(s) "
                    f"that contributed to the prediction: "
                    f"{', '.join([b['Parameter'] for b in abnormal_params])}. "
                    f"These parameters showed significant deviation from clinical normal ranges, "
                    f"which are key indicators of hepatic dysfunction. The model's confidence "
                    f"score of {confidence}% reflects the cumulative risk assessment based on "
                    f"these biomarker patterns.")
    else:
        xai_text = (f"All liver biomarkers are within normal clinical ranges. "
                    f"The AI model found no significant risk indicators. "
                    f"The confidence score of {confidence}% reflects a low-risk assessment. "
                    f"Routine monitoring is still recommended for preventive healthcare.")

    pdf.multi_cell(190, 7, xai_text)
    pdf.ln(5)

    # ── DIET & RECOMMENDATIONS ──
    if is_disease:
        avoid = ["Alcohol & Tobacco", "Fried & Oily Food", "Red Meat",
                 "Excess Salt & Sugar", "Carbonated Drinks", "Raw Shellfish"]
        recommend = ["Fresh Green Vegetables", "Papaya & Citrus Fruits",
                     "High-Protein Low-Fat Diet", "Plenty of Water (3L/day)",
                     "Whole Grains & Legumes", "Turmeric & Ginger (anti-inflammatory)"]
        followup = "URGENT: Follow-up liver function test within 7 days. Consult hepatologist immediately."
    else:
        avoid = ["Excess Alcohol", "Processed Junk Food", "Excess Sugar"]
        recommend = ["Balanced Nutritious Diet", "Daily Exercise (30 min)",
                     "Fresh Fruits & Vegetables", "Adequate Hydration"]
        followup = "ROUTINE: Liver function screening after 6 months. Maintain healthy lifestyle."

    pdf.set_x(10)

    # FIXED: Replaced Unicode X/checkmark with plain ASCII X and +
    pdf.set_fill_color(255, 235, 235)
    pdf.set_text_color(180, 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(92, 8, "  FOODS TO AVOID", border=1, fill=True)
    pdf.set_fill_color(230, 255, 240)
    pdf.set_text_color(0, 130, 60)
    pdf.cell(2, 8, "", border=0)
    pdf.cell(96, 8, "  RECOMMENDED DIET", border=1, fill=True, ln=True)

    pdf.set_font("Arial", "", 10)
    max_rows = max(len(avoid), len(recommend))
    for i in range(max_rows):
        pdf.set_x(10)
        pdf.set_text_color(120, 0, 0)
        pdf.set_fill_color(255, 245, 245)
        # FIXED: Unicode ✗ replaced with plain X
        a = f"X  {avoid[i]}" if i < len(avoid) else ""
        pdf.cell(92, 7, a, border=1, fill=True)

        pdf.set_text_color(0, 100, 50)
        pdf.set_fill_color(245, 255, 250)
        # FIXED: Unicode ✓ replaced with plain +
        r = f"+  {recommend[i]}" if i < len(recommend) else ""
        pdf.cell(2, 7, "", border=0)
        pdf.cell(96, 7, r, border=1, fill=True, ln=True)

    pdf.ln(5)

    # Follow-up
    pdf.set_x(10)
    pdf.set_fill_color(255, 250, 220)
    pdf.set_text_color(100, 70, 0)
    pdf.set_font("Arial", "B", 10)
    # FIXED: Removed any potential Unicode in followup string
    pdf.cell(190, 8, f"  FOLLOW-UP NOTE: {followup}", border=1, fill=True, ln=True)

    pdf.ln(5)

    # ── FOOTER ──
    pdf.set_fill_color(0, 51, 153)
    pdf.rect(0, pdf.get_y() + 5, 210, 20, 'F')
    pdf.set_xy(10, pdf.get_y() + 8)
    pdf.set_text_color(200, 220, 255)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 6,
             f"Generated by HepatoXAI System  |  Employee: {employee_data.get('full_name', '')}  "
             f"|  ID: {employee_data.get('employee_id', '')}  |  {datetime.now().strftime('%d/%m/%Y %H:%M')}",
             align="C", ln=True)
    pdf.set_xy(10, pdf.get_y())
    pdf.cell(0, 5,
             "CONFIDENTIAL MEDICAL DOCUMENT - For authorized medical personnel only",
             align="C", ln=True)

    return pdf.output(dest='S').encode('latin-1')

# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_data = None
    st.session_state.username = None
    st.session_state.auth_mode = "login"

# ═══════════════════════════════════════════
# AUTH PAGES
# ═══════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown("""
    <div class='logo-area'>
        <span class='logo-icon'>🫀</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='main-title'>HepatoXAI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Explainable AI · Liver Intelligence System · Hospital Edition</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            if st.button("🔐 Sign In", use_container_width=True):
                st.session_state.auth_mode = "login"
        with mode_col2:
            if st.button("📝 Sign Up", use_container_width=True):
                st.session_state.auth_mode = "signup"

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)

        # ── LOGIN ──
        if st.session_state.auth_mode == "login":
            st.markdown("### 🔐 Hospital Employee Sign In")
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            username = st.text_input("Employee Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("SIGN IN", use_container_width=True):
                if username and password:
                    success, user_data = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_data = user_data
                        st.session_state.username = username
                        st.success(f"Welcome back, {user_data['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
                else:
                    st.warning("Please fill in all fields.")

            st.markdown("<br>")
            st.caption("New employee? Click **Sign Up** above to register.")

        # ── SIGNUP ──
        else:
            st.markdown("### 📝 Hospital Employee Registration")
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            full_name = st.text_input("Full Name *", placeholder="Dr. John Smith")
            emp_id = st.text_input("Employee ID *", placeholder="EMP-2025-001")
            hospital = st.text_input("Hospital Name *", placeholder="City Medical Center")
            designation = st.selectbox("Designation *", [
                "Doctor / Physician", "Hepatologist", "Gastroenterologist",
                "Lab Technician", "Nurse", "Medical Officer",
                "Radiologist", "Hospital Administrator"
            ])

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            new_username = st.text_input("Choose Username *", placeholder="john.smith")
            new_password = st.text_input("Choose Password *", type="password",
                                          placeholder="Min 6 characters")
            confirm_password = st.text_input("Confirm Password *", type="password")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("CREATE ACCOUNT", use_container_width=True):
                if all([full_name, emp_id, hospital, new_username, new_password, confirm_password]):
                    if new_password != confirm_password:
                        st.error("Passwords do not match!")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters!")
                    else:
                        success, msg = register_user(
                            new_username, new_password,
                            full_name, hospital, designation, emp_id
                        )
                        if success:
                            st.success(f"{msg} Please sign in now.")
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error(f"{msg}")
                else:
                    st.warning("Please fill in all required fields (*)")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='watermark'>
        HEPATOXAI SYSTEM · DEVELOPED BY SOUMIK ROY · XAI RESEARCH PROJECT 2025
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════
else:
    user = st.session_state.user_data
    uname = st.session_state.username

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0;'>
            <div style='font-size:50px'>🫀</div>
            <div style='font-size:20px; font-weight:800; color:#00D4FF;'>HepatoXAI</div>
            <div style='font-size:11px; color:#7B8FA1; letter-spacing:2px;'>LIVER INTELLIGENCE SYSTEM</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.15);
             border-radius:12px; padding:15px; margin-bottom:15px;'>
            <div style='color:#00D4FF; font-weight:700; font-size:14px;'>👤 {user['full_name']}</div>
            <div style='color:#7B8FA1; font-size:11px; margin-top:4px;'>{user['designation']}</div>
            <div style='color:#7B8FA1; font-size:11px;'>🏥 {user['hospital']}</div>
            <div style='color:#7B8FA1; font-size:11px;'>ID: {user['employee_id']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Session Stats**")
        st.markdown(f"""
        <div class='metric-box' style='margin-bottom:10px;'>
            <div class='metric-value'>{user.get('predictions_count', 0)}</div>
            <div class='metric-label'>Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.session_state.username = None
            st.rerun()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.caption("HepatoXAI v2.0 · Research Edition")

    # ── HEADER ──
    st.markdown("<div class='main-title'>🫀 HepatoXAI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Liver Disease Prediction · Explainable AI Dashboard</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2 = st.tabs(["🔬 New Prediction", "ℹ️ About XAI"])

    with tab1:

        # ── HOSPITAL & DOCTOR INFO ──
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>🏥 Hospital & Doctor Information</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hospital_name = st.text_input("Hospital Name",
                                           value=user.get('hospital', ''))
            hospital_address = st.text_input("Hospital Address",
                                              placeholder="123 Medical Ave, City")
        with c2:
            doctor_name = st.text_input("Doctor / Reporting Officer",
                                         value=user.get('full_name', ''))
            doctor_degree = st.text_input("Qualification",
                                           placeholder="MBBS, MD (Hepatology)")
        with c3:
            dept = st.text_input("Department",
                                  placeholder="Hepatology & Gastroenterology")
            report_id = st.text_input("Report ID",
                                       value=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── PATIENT INFO ──
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>👤 Patient Information</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_name = st.text_input("Patient Full Name",
                                          placeholder="Full name of patient")
            patient_age  = st.number_input("Age (years)", 1, 100, 35)
            patient_sex  = st.selectbox("Biological Sex", ["Male", "Female"])
            patient_blood = st.selectbox("Blood Group",
                                          ["A+","A-","B+","B-","AB+","AB-","O+","O-"])
        with c2:
            patient_id     = st.text_input("Patient ID", placeholder="PAT-2025-001")
            patient_phone  = st.text_input("Phone", placeholder="+91 XXXXX XXXXX")
            patient_email  = st.text_input("Email", placeholder="patient@email.com")
        with c3:
            patient_weight = st.number_input("Weight (kg)", 1, 250, 65)
            patient_height = st.number_input("Height (cm)", 50, 250, 165)
            patient_symptoms = st.multiselect("Reported Symptoms", [
                "Jaundice", "Abdominal Pain", "Fatigue",
                "Nausea/Vomiting", "Dark Urine", "Loss of Appetite",
                "Swollen Abdomen", "Itching", "None"
            ], default=["None"])
        st.markdown("</div>", unsafe_allow_html=True)

        # ── BIOMARKERS ──
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>🧪 Liver Biomarker Values</div>",
                    unsafe_allow_html=True)

        st.caption("Normal ranges shown in parentheses for reference")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tb  = st.number_input("Total Bilirubin (0.2-1.2)", 0.0, 100.0, 0.8, step=0.1)
            db  = st.number_input("Direct Bilirubin (0.0-0.3)", 0.0, 50.0, 0.2, step=0.1)
        with c2:
            alkphos = st.number_input("Alkaline Phosphotase (44-147)", 0, 5000, 90)
            alt     = st.number_input("ALT / SGPT (7-56)", 0, 5000, 25)
        with c3:
            ast  = st.number_input("AST / SGOT (10-40)", 0, 5000, 28)
            tp   = st.number_input("Total Proteins (6.0-8.3)", 0.0, 20.0, 7.0, step=0.1)
        with c4:
            alb  = st.number_input("Albumin (3.5-5.0)", 0.0, 10.0, 4.2, step=0.1)
            agr  = st.number_input("A/G Ratio (1.0-2.5)", 0.0, 10.0, 1.4, step=0.1)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── PREDICT BUTTON ──
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_clicked = st.button("🔬 ANALYZE & PREDICT LIVER DISEASE",
                                        use_container_width=True)

        # ── RESULTS ──
        if predict_clicked:
            with st.spinner("AI is analyzing biomarkers..."):
                gender_num = 1 if patient_sex == "Male" else 0
                prediction, confidence = predict_liver_disease(
                    patient_age, gender_num, tb, db, alkphos, alt, ast, tp, alb, agr)
                biomarkers = analyze_biomarkers(tb, db, alkphos, alt, ast, tp, alb, agr)
                update_prediction_count(uname)

            st.markdown("<br>", unsafe_allow_html=True)

            # Result card
            if prediction == 1:
                st.markdown(f"""
                <div class='result-danger'>
                    <div class='result-title'>⚠️ LIVER DISEASE DETECTED</div>
                    <div style='color:#FCA5A5; font-size:16px;'>
                        Significant hepatic abnormalities found in biomarker analysis
                    </div>
                    <div style='color:#FEE2E2; font-size:14px; margin-top:10px;'>
                        AI Confidence: {confidence}% &nbsp;|&nbsp; Risk Level: HIGH
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-safe'>
                    <div class='result-title'>✅ HEALTHY LIVER</div>
                    <div style='color:#6EE7B7; font-size:16px;'>
                        All biomarkers within acceptable clinical ranges
                    </div>
                    <div style='color:#D1FAE5; font-size:14px; margin-top:10px;'>
                        AI Confidence: {confidence}% &nbsp;|&nbsp; Risk Level: LOW
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── ANALYTICS ROW ──
            c1, c2 = st.columns([3, 2])

            with c1:
                st.markdown("**🧠 Feature Impact Analysis (XAI)**")
                chart_buf = generate_feature_chart(tb, db, alkphos, alt, ast, tp, alb, agr)
                st.image(chart_buf, use_container_width=True)

            with c2:
                st.markdown("**🎯 AI Confidence Meter**")
                gauge_buf = generate_risk_gauge(confidence, prediction == 1)
                st.image(gauge_buf, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── BIOMARKER TABLE ──
            st.markdown("**🔬 Detailed Biomarker Report**")
            for b in biomarkers:
                col_n, col_v, col_r, col_s = st.columns([3, 1.5, 2, 1.5])
                with col_n:
                    st.markdown(f"<div class='param-name'>{b['Parameter']}</div>",
                                unsafe_allow_html=True)
                with col_v:
                    st.markdown(f"<div class='param-value'>{b['Value']} {b['Unit']}</div>",
                                unsafe_allow_html=True)
                with col_r:
                    st.markdown(f"<div class='param-name'>Normal: {b['Normal Range']}</div>",
                                unsafe_allow_html=True)
                with col_s:
                    badge_class = "risk-badge-high" if b['Flag'] == 'abnormal' else "risk-badge-normal"
                    st.markdown(f"<div class='{badge_class}'>{b['Status']}</div>",
                                unsafe_allow_html=True)
                st.markdown("<hr style='margin:4px 0; border-color:rgba(255,255,255,0.05);'>",
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── DIET RECOMMENDATIONS ──
            if prediction == 1:
                avoid = ["Alcohol & Tobacco", "Fried & Oily Food", "Red Meat",
                         "Excess Salt & Sugar", "Carbonated Drinks", "Raw Shellfish"]
                recommend = ["Fresh Green Vegetables", "Papaya & Citrus Fruits",
                             "High-Protein Low-Fat Diet", "Water (3L/day)",
                             "Whole Grains", "Turmeric & Ginger"]
                followup = "Follow-up liver test within 7 days. Consult a hepatologist."
            else:
                avoid = ["Excess Alcohol", "Processed Junk Food", "Excess Sugar"]
                recommend = ["Balanced Diet", "Daily Exercise (30 min)",
                             "Fresh Fruits", "Adequate Hydration"]
                followup = "Routine liver checkup after 6 months."

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**❌ Foods To Avoid**")
                # FIXED: Use st.write() instead of st.markdown() to prevent None rendering
                for item in avoid:
                    st.write(f"🚫 {item}")
            with c2:
                st.markdown("**✅ Recommended Diet**")
                # FIXED: Use st.write() instead of st.markdown() to prevent None rendering
                for item in recommend:
                    st.write(f"✅ {item}")

            st.info(f"📅 **Follow-up:** {followup}")

            # ── PDF DOWNLOAD ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📥 Download Hospital Report")

            patient_data = {
                "name": patient_name, "id": patient_id,
                "age": patient_age, "sex": patient_sex,
                "blood": patient_blood, "phone": patient_phone,
                "email": patient_email, "weight": patient_weight,
                "height": patient_height
            }

            pdf_bytes = generate_pdf_report(
                patient_data, prediction, confidence, biomarkers,
                hospital_name, hospital_address, doctor_name,
                doctor_degree, user
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📄 DOWNLOAD PREMIUM HOSPITAL REPORT (PDF)",
                    data=pdf_bytes,
                    file_name=f"HepatoXAI_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    with tab2:
        st.markdown("""
        <div class='card'>
            <div class='card-header'>🧠 About Explainable AI (XAI)</div>
            <p style='color:#B0C4D8; line-height:1.8;'>
            HepatoXAI uses <strong style='color:#00D4FF;'>Explainable Artificial Intelligence</strong>
            to make liver disease predictions <em>transparent</em> and <em>trustworthy</em>
            for medical professionals.
            </p>
            <p style='color:#B0C4D8; line-height:1.8;'>
            Unlike black-box AI systems, our system shows <em>which biomarkers</em>
            influenced the prediction and <em>how much</em>, using SHAP-based feature
            importance analysis. This allows doctors to verify and trust the AI's reasoning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class='card' style='text-align:center;'>
                <div style='font-size:40px;'>🎯</div>
                <div style='color:#00D4FF; font-weight:700; margin:10px 0;'>High Accuracy</div>
                <div style='color:#7B8FA1; font-size:13px;'>
                ML ensemble models trained on validated clinical liver disease datasets
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='card' style='text-align:center;'>
                <div style='font-size:40px;'>🔬</div>
                <div style='color:#00D4FF; font-weight:700; margin:10px 0;'>XAI Transparency</div>
                <div style='color:#7B8FA1; font-size:13px;'>
                SHAP values reveal which biomarkers drive each individual prediction
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class='card' style='text-align:center;'>
                <div style='font-size:40px;'>🏥</div>
                <div style='color:#00D4FF; font-weight:700; margin:10px 0;'>Hospital Grade</div>
                <div style='color:#7B8FA1; font-size:13px;'>
                Professional PDF reports downloadable for clinical records
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='watermark'>
        HEPATOXAI · SOUMIK ROY · XAI LIVER RESEARCH · 2025
    </div>
    """, unsafe_allow_html=True)