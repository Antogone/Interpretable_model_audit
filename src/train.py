# 1. Load data
# 2. Manual preprocessing (no Pipeline):
#    - Handle missing values
#    - Encode categoricals with pd.get_dummies
#    - Scale numericals
# 3. Train/test split
# 4. Fit XGBClassifier directly on the processed DataFrame
# 5. Evaluate (accuracy, F1, AUC)
# 6. Save model + X_test + y_test with joblib

# ── 0. Packages ───────────────────────────────────────────────────────────────
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

# ── 1. Load Data ──────────────────────────────────────────────────────────────
statlog = fetch_ucirepo(id=144)
X = statlog.data.features
y = statlog.data.targets

# Renaming from the documentation
column_names = {
    'Attribute1':  'checking_status',
    'Attribute2':  'duration',
    'Attribute3':  'credit_history',
    'Attribute4':  'purpose',
    'Attribute5':  'credit_amount',
    'Attribute6':  'savings',
    'Attribute7':  'employment',
    'Attribute8':  'installment_rate',
    'Attribute9':  'personal_status',
    'Attribute10': 'other_debtors',
    'Attribute11': 'residence_since',
    'Attribute12': 'property',
    'Attribute13': 'age',
    'Attribute14': 'other_installments',
    'Attribute15': 'housing',
    'Attribute16': 'existing_credits',
    'Attribute17': 'job',
    'Attribute18': 'dependents',
    'Attribute19': 'telephone',
    'Attribute20': 'foreign_worker',
}

X = X.rename(columns=column_names)

X.describe()

# Inspect — to understand the data and their columns type
print(X.dtypes)
print("\nShape:", X.shape)
print("\nMissing values:\n", X.isnull().sum())
print("\ny unique values:", y.iloc[:, 0].unique())


# ── 2. Auto-detect column types ───────────────────────────────────────────────
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols   = X.select_dtypes(include=["number"]).columns.tolist()


# ── 3. Encode categoricals ────────────────────────────────────────────────────
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X = X.astype(float) # TO get 0,1 and not T/F

# ── 4. Fix target for xgboost ─────────────────────────────────────────────────────────────
y = y.iloc[:, 0].replace({1: 0, 2: 1})  # 0 = good credit, 1 = bad credit


# ── 5. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=66, stratify=y
)

# ── 6. Scale numericals (fit on train only) ───────────────────────────────────

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])


# ── 7. Train XGBoost ───────────────────────────────────

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=66,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ── 10. Evaluate ──────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1       : {f1_score(y_test, y_pred, average='weighted'):.3f}")
print(f"AUC      : {roc_auc_score(y_test, y_proba):.3f}")

# ── 11. Save ──────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

joblib.dump(model,   "outputs/model.joblib")
joblib.dump(X_test,  "outputs/X_test.joblib")
joblib.dump(y_test,  "outputs/y_test.joblib")
joblib.dump(X_train, "outputs/X_train.joblib")  # needed for SHAP background

print("\nSaved to outputs")
