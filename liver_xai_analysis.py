# ============================================================
# FILE: liver_xai_analysis.py
# PROJECT: Can We Trust Explainable AI in Liver Disease?
# STEP 5: Explainable AI (SHAP + LIME)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os

import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. CREATE OUTPUT FOLDER
# ─────────────────────────────────────────
os.makedirs("outputs/xai", exist_ok=True)

print("=" * 60)
print("   LIVER DISEASE XAI — STEP 5: SHAP + LIME")
print("=" * 60)

# ─────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────
print("\n📥 Loading Data...")

X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print(f"✅ X_train shape : {X_train.shape}")
print(f"✅ X_test shape  : {X_test.shape}")

# ─────────────────────────────────────────
# 3. LOAD BEST MODEL
# ─────────────────────────────────────────
print("\n🤖 Loading Best Model...")

with open("models/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Logistic Regression model loaded!")

# ─────────────────────────────────────────
# 4. SHAP EXPLAINABILITY
# ─────────────────────────────────────────
print("\n📊 Running SHAP Analysis...")

explainer = shap.Explainer(model, X_train)

shap_values = explainer(X_test)

# ─────────────────────────────────────────
# 5. SHAP SUMMARY PLOT
# ─────────────────────────────────────────
print("📈 Generating SHAP Summary Plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.savefig(
    "outputs/xai/shap_summary_plot.png",
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print("✅ SHAP summary plot saved!")

# ─────────────────────────────────────────
# 6. SHAP BAR PLOT
# ─────────────────────────────────────────
print("📈 Generating SHAP Feature Importance Plot...")

plt.figure()

shap.plots.bar(
    shap_values,
    show=False
)

plt.savefig(
    "outputs/xai/shap_bar_plot.png",
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print("✅ SHAP bar plot saved!")

# ─────────────────────────────────────────
# 7. LOCAL EXPLANATION (Single Patient)
# ─────────────────────────────────────────
print("\n🔍 Generating Local SHAP Explanation...")

sample_index = 5

plt.figure()

shap.plots.waterfall(
    shap_values[sample_index],
    show=False
)

plt.savefig(
    "outputs/xai/shap_waterfall_plot.png",
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print("✅ SHAP waterfall plot saved!")

# ─────────────────────────────────────────
# 8. LIME EXPLANATION
# ─────────────────────────────────────────
print("\n🧠 Running LIME Explanation...")

explainer_lime = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Healthy', 'Disease'],
    mode='classification'
)

lime_exp = explainer_lime.explain_instance(
    data_row=X_test.iloc[sample_index],
    predict_fn=model.predict_proba
)

# Save LIME HTML report
lime_exp.save_to_file(
    "outputs/xai/lime_explanation.html"
)

print("✅ LIME explanation saved!")

# ─────────────────────────────────────────
# 9. FEATURE IMPORTANCE TABLE
# ─────────────────────────────────────────
print("\n📋 Calculating Feature Importance...")

importance = np.abs(shap_values.values).mean(axis=0)

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n" + "=" * 60)
print("📊 GLOBAL FEATURE IMPORTANCE")
print("=" * 60)

print(importance_df.round(4).to_string(index=False))

# Save CSV
importance_df.to_csv(
    "outputs/xai/feature_importance.csv",
    index=False
)

print("\n✅ Feature importance CSV saved!")

# ─────────────────────────────────────────
# 10. TRUST ANALYSIS REPORT
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("🧠 AI TRUST & EXPLAINABILITY REPORT")
print("=" * 60)

top_feature = importance_df.iloc[0]['Feature']

print(f"Most Important Feature : {top_feature}")

print("\nKey Findings:")
print("• SHAP explains global feature importance")
print("• LIME explains single patient prediction")
print("• Model decisions are interpretable")
print("• Transparent AI improves trust in healthcare")
print("• Predictions can be visually explained")

print("=" * 60)

print("\n🎉 XAI ANALYSIS COMPLETE!")
print("📸 Send screenshots for FINAL REPORT step!")
print("=" * 60)