# ============================================================
# FILE: liver_eda_analysis.py
# PROJECT: Can We Trust Explainable AI in Liver Disease?
# STEP 2: Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import urllib.request

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)

print("=" * 60)
print("   LIVER DISEASE XAI PROJECT — EDA ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────
# 2. DOWNLOAD & LOAD DATASET
# ─────────────────────────────────────────
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
       "/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv")

csv_path = "data/liver_disease.csv"

if not os.path.exists(csv_path):
    print("\n📥 Downloading dataset...")
    urllib.request.urlretrieve(url, csv_path)
    print("✅ Downloaded successfully!")
else:
    print("\n✅ Dataset already exists — loading from disk.")

columns = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase', 'Total_Protiens',
    'Albumin', 'Albumin_and_Globulin_Ratio', 'Target'
]

df = pd.read_csv(csv_path, header=None, names=columns)

# Fix target: 1 = Liver Disease, 0 = Healthy
df['Target'] = df['Target'].map({1: 1, 2: 0})

print(f"\n📊 Dataset Shape   : {df.shape}")
print(f"🎯 Target Classes  : {df['Target'].value_counts().to_dict()}")
print(f"   (1 = Liver Disease | 0 = Healthy)")

# ─────────────────────────────────────────
# 3. BASIC INFO
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("📋 FIRST 5 ROWS:")
print("─" * 60)
print(df.head().to_string())

print("\n" + "─" * 60)
print("📋 DATA TYPES:")
print("─" * 60)
print(df.dtypes)

print("\n" + "─" * 60)
print("📋 STATISTICAL SUMMARY:")
print("─" * 60)
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────
# 4. MISSING VALUES
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("🔍 MISSING VALUES CHECK:")
print("─" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])
if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print(f"⚠️  Total missing values: {missing.sum()}")

# ─────────────────────────────────────────
# 5. PLOT 1 — TARGET CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\n📊 Generating Plot 1: Target Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Target Class Distribution', fontsize=15, fontweight='bold')

counts = df['Target'].value_counts()
labels = ['Liver Disease', 'Healthy']
colors = ['#E74C3C', '#2ECC71']

axes[0].bar(labels, [counts[1], counts[0]],
            color=colors, edgecolor='black', width=0.5)
axes[0].set_title('Patient Count per Class')
axes[0].set_ylabel('Number of Patients')
axes[0].set_xlabel('Class')
for i, v in enumerate([counts[1], counts[0]]):
    axes[0].text(i, v + 5, str(v), ha='center',
                 fontweight='bold', fontsize=12)

axes[1].pie([counts[1], counts[0]],
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Proportion (%)')

plt.tight_layout()
plt.savefig('outputs/eda/01_target_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 1 saved!")

# ─────────────────────────────────────────
# 6. PLOT 2 — GENDER DISTRIBUTION
# ─────────────────────────────────────────
print("📊 Generating Plot 2: Gender Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Gender Distribution Analysis', fontsize=15, fontweight='bold')

gender_counts = df['Gender'].value_counts()
axes[0].bar(gender_counts.index, gender_counts.values,
            color=['#3498DB', '#E91E8C'],
            edgecolor='black', width=0.5)
axes[0].set_title('Overall Gender Count')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Gender')
for i, v in enumerate(gender_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

gender_target = df.groupby(['Gender', 'Target']).size().unstack()
gender_target.columns = ['Healthy', 'Liver Disease']
gender_target.plot(kind='bar', ax=axes[1],
                   color=['#2ECC71', '#E74C3C'],
                   edgecolor='black', rot=0)
axes[1].set_title('Gender vs Disease Status')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Gender')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/eda/02_gender_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 2 saved!")

# ─────────────────────────────────────────
# 7. PLOT 3 — AGE DISTRIBUTION
# ─────────────────────────────────────────
print("📊 Generating Plot 3: Age Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Age Distribution Analysis', fontsize=15, fontweight='bold')

axes[0].hist(df[df['Target'] == 1]['Age'], bins=20,
             alpha=0.7, color='#E74C3C',
             label='Liver Disease', edgecolor='black')
axes[0].hist(df[df['Target'] == 0]['Age'], bins=20,
             alpha=0.7, color='#2ECC71',
             label='Healthy', edgecolor='black')
axes[0].set_title('Age Distribution by Class')
axes[0].set_xlabel('Age (Years)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

df.boxplot(column='Age', by='Target', ax=axes[1],
           boxprops=dict(color='#2C3E50'),
           medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Age Boxplot by Target Class')
axes[1].set_xlabel('Target (0 = Healthy, 1 = Disease)')
axes[1].set_ylabel('Age')
plt.suptitle('')

plt.tight_layout()
plt.savefig('outputs/eda/03_age_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 3 saved!")

# ─────────────────────────────────────────
# 8. PLOT 4 — ALL FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────
print("📊 Generating Plot 4: Feature Distributions...")

numeric_cols = [
    'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase', 'Total_Protiens',
    'Albumin', 'Albumin_and_Globulin_Ratio'
]

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Feature Distributions — Disease vs Healthy',
             fontsize=15, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[df['Target'] == 1][col].dropna(),
                 bins=30, alpha=0.6, color='#E74C3C',
                 label='Disease', edgecolor='black')
    axes[i].hist(df[df['Target'] == 0][col].dropna(),
                 bins=30, alpha=0.6, color='#2ECC71',
                 label='Healthy', edgecolor='black')
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.savefig('outputs/eda/04_feature_distributions.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 4 saved!")

# ─────────────────────────────────────────
# 9. PLOT 5 — CORRELATION HEATMAP
# ─────────────────────────────────────────
print("📊 Generating Plot 5: Correlation Heatmap...")

df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0})

fig, ax = plt.subplots(figsize=(13, 10))
corr_matrix = df_encoded.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap',
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/eda/05_correlation_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 5 saved!")

# ─────────────────────────────────────────
# 10. PLOT 6 — BOXPLOTS FOR OUTLIER DETECTION
# ─────────────────────────────────────────
print("📊 Generating Plot 6: Boxplots for Outliers...")

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Boxplots — Outlier Detection per Feature',
             fontsize=15, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    df_encoded.boxplot(
        column=col, by='Target', ax=axes[i],
        boxprops=dict(color='#2C3E50'),
        medianprops=dict(color='red', linewidth=2),
        flierprops=dict(marker='o', color='orange',
                        markersize=4, alpha=0.5))
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].set_xlabel('Target (0=Healthy, 1=Disease)')
    axes[i].set_ylabel('Value')

plt.suptitle('')
plt.tight_layout()
plt.savefig('outputs/eda/06_boxplots_outliers.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 6 saved!")

# ─────────────────────────────────────────
# 11. SAVE ENCODED DATASET FOR NEXT STEP
# ─────────────────────────────────────────
df_encoded.to_csv("data/liver_encoded.csv", index=False)
print("\n✅ Encoded dataset saved: data/liver_encoded.csv")

# ─────────────────────────────────────────
# 12. FINAL EDA SUMMARY REPORT
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("         📊 EDA SUMMARY REPORT")
print("=" * 60)
print(f"  Total Patients         : {len(df)}")
print(f"  Liver Disease Cases    : {(df['Target']==1).sum()} "
      f"({(df['Target']==1).mean()*100:.1f}%)")
print(f"  Healthy Cases          : {(df['Target']==0).sum()} "
      f"({(df['Target']==0).mean()*100:.1f}%)")
print(f"  Male Patients          : {(df['Gender']=='Male').sum()}")
print(f"  Female Patients        : {(df['Gender']=='Female').sum()}")
print(f"  Age Range              : {df['Age'].min()} – "
      f"{df['Age'].max()} years")
print(f"  Average Age            : {df['Age'].mean():.1f} years")
print(f"  Missing Values         : {df.isnull().sum().sum()}")
print(f"  Total Features         : {df.shape[1]-1}")
print(f"  Plots Saved in         : outputs/eda/")
print("=" * 60)
print("\n🎉 EDA COMPLETE! All 6 plots saved successfully!")
print("📸 Take a screenshot and send it for STEP 3!")
print("=" * 60)