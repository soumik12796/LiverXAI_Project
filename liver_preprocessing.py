# ============================================================
# FILE: liver_preprocessing.py
# PROJECT: Can We Trust Explainable AI in Liver Disease?
# STEP 3: Data Preprocessing + SMOTE Class Balancing
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────
os.makedirs("outputs/preprocessing", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("   LIVER DISEASE XAI — STEP 3: PREPROCESSING")
print("=" * 60)

# ─────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────
df = pd.read_csv("data/liver_encoded.csv")
print(f"\n✅ Dataset Loaded: {df.shape}")
print(f"   Missing Values BEFORE: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────
print("\n📋 Handling Missing Values...")
df['Albumin_and_Globulin_Ratio'].fillna(
    df['Albumin_and_Globulin_Ratio'].median(), inplace=True)

print(f"   Missing Values AFTER : {df.isnull().sum().sum()}")
print("✅ Missing values filled with median!")

# ─────────────────────────────────────────
# 4. ENCODE GENDER
# ─────────────────────────────────────────
print("\n📋 Encoding Gender Column...")
if df['Gender'].dtype == object:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    print("✅ Gender encoded: Male=1, Female=0")
else:
    print("✅ Gender already encoded!")

# ─────────────────────────────────────────
# 5. HANDLE OUTLIERS (IQR Capping)
# ─────────────────────────────────────────
print("\n📋 Handling Outliers using IQR Capping...")

numeric_cols = [
    'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase', 'Total_Protiens',
    'Albumin', 'Albumin_and_Globulin_Ratio'
]

outlier_report = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    outlier_report.append({
        'Feature': col,
        'Outliers Capped': outliers_count,
        'Lower Bound': round(lower, 3),
        'Upper Bound': round(upper, 3)
    })

outlier_df = pd.DataFrame(outlier_report)
print(outlier_df.to_string(index=False))
print("✅ Outliers capped using IQR method!")

# ─────────────────────────────────────────
# 6. SPLIT FEATURES AND TARGET
# ─────────────────────────────────────────
print("\n📋 Splitting Features and Target...")

X = df.drop('Target', axis=1)
y = df['Target']

print(f"   Features (X) shape : {X.shape}")
print(f"   Target   (y) shape : {y.shape}")
print(f"   Feature Names      : {list(X.columns)}")

# ─────────────────────────────────────────
# 7. TRAIN-TEST SPLIT
# ─────────────────────────────────────────
print("\n📋 Splitting into Train/Test Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Training Set  : {X_train.shape[0]} samples")
print(f"   Testing  Set  : {X_test.shape[0]} samples")
print(f"   Train Target  : {y_train.value_counts().to_dict()}")
print(f"   Test  Target  : {y_test.value_counts().to_dict()}")

# ─────────────────────────────────────────
# 8. APPLY SMOTE ON TRAINING DATA ONLY
# ─────────────────────────────────────────
print("\n📋 Applying SMOTE to Balance Training Data...")
print(f"   Before SMOTE: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"   After  SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")
print(f"   Training samples after SMOTE: {X_train_sm.shape[0]}")
print("✅ SMOTE applied successfully!")

# ─────────────────────────────────────────
# 9. FEATURE SCALING
# ─────────────────────────────────────────
print("\n📋 Applying Standard Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X.columns)
X_test_scaled  = pd.DataFrame(
    X_test_scaled,  columns=X.columns)

print(f"   Mean after scaling (first 3): "
      f"{X_train_scaled.mean().values[:3].round(4)}")
print(f"   Std  after scaling (first 3): "
      f"{X_train_scaled.std().values[:3].round(4)}")
print("✅ Standard Scaling applied!")

# ─────────────────────────────────────────
# 10. PLOT 1 — BEFORE vs AFTER SMOTE
# ─────────────────────────────────────────
print("\n📊 Generating Plot 1: SMOTE Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Class Distribution — Before & After SMOTE',
             fontsize=14, fontweight='bold')

# Original full dataset
orig = y.value_counts()
axes[0].bar(['Liver Disease', 'Healthy'],
            [orig[1], orig[0]],
            color=['#E74C3C', '#2ECC71'], edgecolor='black')
axes[0].set_title('Original Dataset')
axes[0].set_ylabel('Count')
for i, v in enumerate([orig[1], orig[0]]):
    axes[0].text(i, v+3, str(v), ha='center', fontweight='bold')

# Before SMOTE (train)
before = y_train.value_counts()
axes[1].bar(['Liver Disease', 'Healthy'],
            [before[1], before[0]],
            color=['#E74C3C', '#2ECC71'], edgecolor='black')
axes[1].set_title('Training Set (Before SMOTE)')
axes[1].set_ylabel('Count')
for i, v in enumerate([before[1], before[0]]):
    axes[1].text(i, v+2, str(v), ha='center', fontweight='bold')

# After SMOTE
after = pd.Series(y_train_sm).value_counts()
axes[2].bar(['Liver Disease', 'Healthy'],
            [after[1], after[0]],
            color=['#E74C3C', '#2ECC71'], edgecolor='black')
axes[2].set_title('Training Set (After SMOTE)')
axes[2].set_ylabel('Count')
for i, v in enumerate([after[1], after[0]]):
    axes[2].text(i, v+2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/preprocessing/01_smote_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 1 saved!")

# ─────────────────────────────────────────
# 11. PLOT 2 — FEATURE SCALING COMPARISON
# ─────────────────────────────────────────
print("📊 Generating Plot 2: Feature Scaling Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Scaling — Before vs After',
             fontsize=14, fontweight='bold')

# Before scaling
X_train_sm_df = pd.DataFrame(X_train_sm, columns=X.columns)
X_train_sm_df[numeric_cols].boxplot(ax=axes[0], rot=45)
axes[0].set_title('Before Scaling')
axes[0].set_ylabel('Feature Value')

# After scaling
X_train_scaled[numeric_cols].boxplot(ax=axes[1], rot=45)
axes[1].set_title('After Standard Scaling')
axes[1].set_ylabel('Scaled Value')

plt.tight_layout()
plt.savefig('outputs/preprocessing/02_scaling_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 2 saved!")

# ─────────────────────────────────────────
# 12. PLOT 3 — CORRELATION AFTER PREPROCESSING
# ─────────────────────────────────────────
print("📊 Generating Plot 3: Post-Preprocessing Correlation...")

fig, ax = plt.subplots(figsize=(12, 9))
corr = X_train_scaled.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax,
            linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Feature Correlation After Preprocessing',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/preprocessing/03_post_correlation.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 3 saved!")

# ─────────────────────────────────────────
# 13. SAVE ALL PREPROCESSED DATA
# ─────────────────────────────────────────
print("\n📋 Saving preprocessed data...")

X_train_scaled.to_csv("data/X_train.csv", index=False)
X_test_scaled.to_csv("data/X_test.csv",   index=False)
pd.Series(y_train_sm).to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",           index=False)

# Save scaler for later use
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ X_train.csv saved!")
print("✅ X_test.csv  saved!")
print("✅ y_train.csv saved!")
print("✅ y_test.csv  saved!")
print("✅ scaler.pkl  saved!")

# ─────────────────────────────────────────
# 14. FINAL PREPROCESSING SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("      📊 PREPROCESSING SUMMARY REPORT")
print("=" * 60)
print(f"  Missing Values Fixed    : 4 (median imputation)")
print(f"  Outliers Capped         : IQR method on 8 features")
print(f"  Gender Encoded          : Male=1, Female=0")
print(f"  Train Size (raw)        : {X_train.shape[0]} samples")
print(f"  Test  Size              : {X_test.shape[0]} samples")
print(f"  Train Size (SMOTE)      : {X_train_sm.shape[0]} samples")
print(f"  Scaling Method          : StandardScaler")
print(f"  Files Saved in          : data/ and models/")
print("=" * 60)
print("\n🎉 PREPROCESSING COMPLETE!")
print("📸 Take a screenshot and send for STEP 4 — Model Training!")
print("=" * 60)