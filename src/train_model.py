import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. LOAD DATASET
# ==========================================
DATA_PATH = os.path.join('data', 'autism_screening_data.csv')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

# ==========================================
# 2. DEFINE FEATURES & TARGET
# ==========================================
# Target column: 'Class/ASD' ('YES' -> 1, 'NO' -> 0)
TARGET_COL = 'Class/ASD'
if TARGET_COL not in df.columns:
    # Check alternate naming
    for c in df.columns:
        if 'class' in c.lower() or 'asd' in c.lower():
            TARGET_COL = c
            break

df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper()
y = df[TARGET_COL].map({'YES': 1, 'NO': 0, '1': 1, '0': 0})
if y.isnull().any():
    # Fill any unmapped with 0
    y = y.fillna(0).astype(int)

# 10 AQ-10 screening item scores
AQ10_COLS = [f'A{i}_Score' for i in range(1, 11)]

# Numeric features
NUM_COLS = ['age']

# Categorical features
CAT_COLS = [
    'gender',
    'ethnicity',
    'jundice',
    'austim',
    'contry_of_res',
    'used_app_before',
    'relation'
]

FEATURE_COLS = AQ10_COLS + NUM_COLS + CAT_COLS

X = df[FEATURE_COLS].copy()

# Clean numeric values: convert '?' to NaN
for col in NUM_COLS + AQ10_COLS:
    X[col] = pd.to_numeric(X[col].replace('?', np.nan), errors='coerce')

# Clean categorical values: replace '?' with 'Unknown'
for col in CAT_COLS:
    X[col] = X[col].replace('?', 'Unknown').fillna('Unknown').astype(str).str.strip()

# Cap realistic adult age (e.g. 18 to 100, replace outliers like 383 with median)
median_age = X['age'].median()
X.loc[(X['age'] < 1) | (X['age'] > 100), 'age'] = median_age

print(f"Features: {len(FEATURE_COLS)} total ({len(AQ10_COLS)} AQ-10, {len(NUM_COLS)} numeric, {len(CAT_COLS)} categorical)")

# ==========================================
# 3. TRAIN / TEST SPLIT (STRATIFIED)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)} (ASD: {y_train.sum()}), Test size: {len(X_test)} (ASD: {y_test.sum()})")

# ==========================================
# 4. PREPROCESSING & PIPELINE
# ==========================================
aq_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('aq', aq_transformer, AQ10_COLS),
        ('num', num_transformer, NUM_COLS),
        ('cat', cat_transformer, CAT_COLS)
    ]
)

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_classifier)
])

# ==========================================
# 5. TRAIN MODEL
# ==========================================
pipeline.fit(X_train, y_train)
print("Model training complete.")

# ==========================================
# 6. EVALUATION ON TEST SET
# ==========================================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*40)
print("TEST SET CLASSIFICATION METRICS")
print("="*40)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}  (Screening Sensitivity)")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-ASD (0)', 'ASD (1)']))

# ==========================================
# 7. EXTRACT FEATURE IMPORTANCES
# ==========================================
preproc_fitted = pipeline.named_steps['preprocessor']
cat_encoder = preproc_fitted.named_transformers_['cat'].named_steps['onehot']
cat_encoded_names = list(cat_encoder.get_feature_names_out(CAT_COLS))

all_feature_names = AQ10_COLS + NUM_COLS + cat_encoded_names
importances = pipeline.named_steps['classifier'].feature_importances_

feat_imp_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

feat_imp_path = os.path.join('outputs', 'feature_importance.csv')
feat_imp_df.to_csv(feat_imp_path, index=False)
print(f"\nSaved feature importance to {feat_imp_path}")
print("Top 10 Most Influential Features:")
print(feat_imp_df.head(10))

# ==========================================
# 8. SAVE TRAINED MODEL ARTIFACT
# ==========================================
model_path = os.path.join('models', 'asd_risk_model.joblib')
joblib.dump(pipeline, model_path)
print(f"\nSaved trained Random Forest Classifier pipeline to {model_path}")

# ==========================================
# 9. SAVE BASELINE TEST PREDICTIONS
# ==========================================
# Generate predictions for the full dataset for reference
full_preds = pipeline.predict(X)
full_probs = np.round(pipeline.predict_proba(X)[:, 1], 4)

# Risk Category Mapping:
# Prob < 0.30 -> Low Risk (0h)
# 0.30 <= Prob < 0.70 -> Medium Risk (2h)
# Prob >= 0.70 -> High Risk (5h)
def map_risk_category(p):
    if p < 0.30:
        return 'Low Risk'
    elif p < 0.70:
        return 'Medium Risk'
    else:
        return 'High Risk'

def map_support_demand(r):
    if r == 'High Risk':
        return 5
    elif r == 'Medium Risk':
        return 2
    else:
        return 0

risk_categories = [map_risk_category(p) for p in full_probs]
support_demands = [map_support_demand(r) for r in risk_categories]
risk_scores = np.round(full_probs * 100.0, 2)

ids = [f"IND{i+1:03d}" for i in range(len(df))]

predictions_df = pd.DataFrame({
    'individual_id': ids,
    'asd_probability': full_probs,
    'predicted_class': full_preds,
    'actual_class': y.values,
    'risk_category': risk_categories,
    'risk_score': risk_scores,
    'support_demand_hours': support_demands
})

# Include key raw feature columns for exploration
for c in df.columns:
    if c not in predictions_df.columns:
        predictions_df[c] = df[c]

preds_csv_path = os.path.join('outputs', 'predictions.csv')
predictions_df.to_csv(preds_csv_path, index=False)
print(f"Saved baseline predictions to {preds_csv_path}")

# Save metrics JSON for dashboard display
metrics_dict = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "confusion_matrix": cm.tolist(),
    "n_estimators": 200,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "total_samples": len(df),
    "aq10_features": AQ10_COLS,
    "num_features": NUM_COLS,
    "cat_features": CAT_COLS
}

metrics_json_path = os.path.join('outputs', 'model_metrics.json')
with open(metrics_json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_dict, f, indent=4)
print(f"Saved model metrics to {metrics_json_path}")

# ==========================================
# 10. GENERATE REAL TEST SUBSETS FOR INFERENCE TESTING
# ==========================================
# Requirement 14: Actual rows from official dataset without fabrication
test_25_df = df.iloc[:25].copy()
test_50_df = df.iloc[:50].copy()
test_100_df = df.iloc[:100].copy()

# Subset without target column
test_25_no_target_df = test_25_df.drop(columns=[TARGET_COL] if TARGET_COL in test_25_df.columns else []).copy()

test_25_path = os.path.join('data', 'test_real_25.csv')
test_50_path = os.path.join('data', 'test_real_50.csv')
test_100_path = os.path.join('data', 'test_real_100.csv')
test_25_no_target_path = os.path.join('data', 'test_real_25_no_target.csv')

test_25_df.to_csv(test_25_path, index=False)
test_50_df.to_csv(test_50_path, index=False)
test_100_df.to_csv(test_100_path, index=False)
test_25_no_target_df.to_csv(test_25_no_target_path, index=False)

print(f"Generated real test subsets from official records:")
print(f"  - {test_25_path} (25 real rows with target)")
print(f"  - {test_50_path} (50 real rows with target)")
print(f"  - {test_100_path} (100 real rows with target)")
print(f"  - {test_25_no_target_path} (25 real rows without target)")


