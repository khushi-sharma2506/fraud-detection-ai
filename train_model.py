"""
train_model.py
--------------
Downloads the Kaggle Credit Card Fraud dataset (if not present),
trains a Random Forest classifier, saves the model + scaler,
and prints evaluation metrics.

Run once before launching the Streamlit app:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import pickle
import json

# ── 1. Load / Generate Data ──────────────────────────────────────────────────
DATA_PATH = "creditcard.csv"

if os.path.exists(DATA_PATH):
    print("📂 Loading dataset from creditcard.csv ...")
    df = pd.read_csv(DATA_PATH)
else:
    print("⚠️  creditcard.csv not found.")
    print("📥 Generating synthetic dataset for demo purposes...")
    print("   (Download the real dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")

    np.random.seed(42)
    n_legit  = 28000
    n_fraud  = 492

    # Simulate PCA features V1–V28
    legit_data = {
        'Time': np.random.uniform(0, 172800, n_legit),
        'Amount': np.abs(np.random.exponential(88, n_legit)),
        'Class': 0
    }
    fraud_data = {
        'Time': np.random.uniform(0, 172800, n_fraud),
        'Amount': np.abs(np.random.exponential(122, n_fraud)),
        'Class': 1
    }
    for v in range(1, 29):
        legit_data[f'V{v}'] = np.random.normal(0, 1, n_legit)
        fraud_data[f'V{v}'] = np.random.normal(0 if v % 2 == 0 else -2, 1.5, n_fraud)

    df = pd.concat([
        pd.DataFrame(legit_data),
        pd.DataFrame(fraud_data)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ Synthetic dataset created: {len(df):,} transactions\n")

# ── 2. Explore ────────────────────────────────────────────────────────────────
print("=" * 55)
print("📊 DATASET OVERVIEW")
print("=" * 55)
print(f"Total transactions : {len(df):,}")
print(f"Legitimate         : {(df['Class']==0).sum():,}")
print(f"Fraudulent         : {(df['Class']==1).sum():,}")
print(f"Fraud rate         : {df['Class'].mean()*100:.2f}%")
print(f"Features           : {df.shape[1]-1}")
print()

# ── 3. Preprocess ─────────────────────────────────────────────────────────────
features = [c for c in df.columns if c != 'Class']
X = df[features].values
y = df['Class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {len(X_train):,}")
print(f"Test size  : {len(X_test):,}\n")

# ── 4. Train Model ────────────────────────────────────────────────────────────
print("🤖 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Training complete!\n")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc      = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])

print("=" * 55)
print("📈 MODEL PERFORMANCE")
print("=" * 55)
print(f"Accuracy  : {acc*100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")
print()
print("Confusion Matrix:")
print(f"  True Negative  (Legit→Legit)   : {cm[0][0]:,}")
print(f"  False Positive (Legit→Fraud)   : {cm[0][1]:,}")
print(f"  False Negative (Fraud→Legit)   : {cm[1][0]:,}")
print(f"  True Positive  (Fraud→Fraud)   : {cm[1][1]:,}")
print()
print("Classification Report:")
print(report)

# Feature importance
importances = model.feature_importances_
feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 Important Features:")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 200)
    print(f"  {feat:<8} {bar} {imp:.4f}")

# ── 6. Save Artifacts ─────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

metrics = {
    "accuracy"     : round(acc * 100, 2),
    "roc_auc"      : round(roc_auc, 4),
    "total_samples": int(len(df)),
    "fraud_count"  : int((df['Class'] == 1).sum()),
    "legit_count"  : int((df['Class'] == 0).sum()),
    "fraud_rate"   : round(df['Class'].mean() * 100, 2),
    "features"     : features,
    "top_features" : [{"feature": f, "importance": round(i, 4)} for f, i in feat_imp],
    "confusion_matrix": cm.tolist()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print()
print("=" * 55)
print("💾 Saved Files:")
print("   model.pkl    — trained Random Forest model")
print("   scaler.pkl   — fitted StandardScaler")
print("   metrics.json — evaluation metrics")
print("=" * 55)
print()
print("🚀 Now run:  streamlit run app.py")
