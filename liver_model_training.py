
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. CREATE OUTPUT FOLDER
# ─────────────────────────────────────────
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("     LIVER DISEASE XAI — STEP 4: MODEL TRAINING")
print("=" * 60)

# ─────────────────────────────────────────
# 2. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────
print("\n📥 Loading preprocessed datasets...")

X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print(f"✅ X_train shape : {X_train.shape}")
print(f"✅ X_test  shape : {X_test.shape}")
print(f"✅ y_train shape : {y_train.shape}")
print(f"✅ y_test  shape : {y_test.shape}")

# ─────────────────────────────────────────
# 3. INITIALIZE MODELS
# ─────────────────────────────────────────
print("\n🤖 Initializing Models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
}

print("✅ Models initialized!")

# ─────────────────────────────────────────
# 4. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────
results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"🚀 Training: {name}")
    print("=" * 50)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"✅ Accuracy  : {accuracy:.4f}")
    print(f"✅ Precision : {precision:.4f}")
    print(f"✅ Recall    : {recall:.4f}")
    print(f"✅ F1 Score  : {f1:.4f}")

    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Healthy', 'Disease'],
        yticklabels=['Healthy', 'Disease']
    )

    plt.title(f'Confusion Matrix — {name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"outputs/models/{filename}_cm.png",
        dpi=150,
        bbox_inches='tight'
    )

    plt.show()

    # Save trained model
    with open(f"models/{filename}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved: models/{filename}.pkl")

# ─────────────────────────────────────────
# 5. RESULTS COMPARISON TABLE
# ─────────────────────────────────────────
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("📊 FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

# ─────────────────────────────────────────
# 6. ACCURACY COMPARISON PLOT
# ─────────────────────────────────────────
print("\n📊 Generating Accuracy Comparison Plot...")

plt.figure(figsize=(8,5))

bars = plt.bar(
    results_df["Model"],
    results_df["Accuracy"],
    color=['#3498DB', '#2ECC71', '#E74C3C'],
    edgecolor='black'
)

plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.01,
        f"{yval:.3f}",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

plt.savefig(
    "outputs/models/model_accuracy_comparison.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ─────────────────────────────────────────
# 7. SAVE RESULTS CSV
# ─────────────────────────────────────────
results_df.to_csv(
    "outputs/models/model_results.csv",
    index=False
)

print("✅ Results CSV saved!")

# ─────────────────────────────────────────
# 8. BEST MODEL
# ─────────────────────────────────────────
best_model = results_df.sort_values(
    by="Accuracy",
    ascending=False
).iloc[0]

print("\n" + "=" * 60)
print("🏆 BEST MODEL")
print("=" * 60)

print(f"Model Name : {best_model['Model']}")
print(f"Accuracy   : {best_model['Accuracy']:.4f}")
print(f"F1 Score   : {best_model['F1 Score']:.4f}")

print("=" * 60)

print("\n🎉 MODEL TRAINING COMPLETE!")
print("📸 Send screenshot for STEP 5 — Explainable AI (SHAP + LIME)")
print("=" * 60)