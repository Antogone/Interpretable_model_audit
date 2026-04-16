# ── 1. Packages ───────────────────────────────────────────────────────────────
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
import os

# ── 2. Load data and model ────────────────────────────────────────────────────
model  = joblib.load("outputs/model.joblib")
X_test = joblib.load("outputs/X_test.joblib")
y_test = joblib.load("outputs/y_test.joblib")

os.makedirs("outputs", exist_ok=True)

y_pred = model.predict(X_test)

# ── 3. Reconstruct sensitive features ─────────────────────────────────────────
# Age: split at median into young / old
age_median = X_test["age"].median()
age_group  = (X_test["age"] >= age_median).map({True: "older", False: "younger"})

# Gender — personal_status_A92 = 1 means female
gender_group = X_test["personal_status_A92"].map({1.0: "female", 0.0: "male"})

# Foreign worker — A202 = not foreign worker (A201 was dropped as reference)
foreign_group = X_test["foreign_worker_A202"].map(
    {1.0: "not_foreign", 0.0: "foreign"}
)

# Housing — A151 (renting) was dropped as reference
housing_group = X_test[["housing_A152", "housing_A153"]].max(axis=1).map(
    {1.0: "owns_or_free", 0.0: "renting"}
)

# Employment — A71 (unemployed) was dropped as reference
employment_cols = ["employment_A72", "employment_A73", "employment_A74", "employment_A75"]
employment_group = X_test[employment_cols].max(axis=1).map(
    {1.0: "employed", 0.0: "unemployed"}
)

# ── 4. Fairness metrics ───────────────────────────────────────────────────────
for group_name, sensitive_feature in [
    ("age",            age_group),
    ("gender",         gender_group),
    ("foreign_worker", foreign_group),
    ("housing",        housing_group),
    ("employment",     employment_group),
]:

    dpd = demographic_parity_difference(
        y_test, y_pred, sensitive_features=sensitive_feature
    )
    eod = equalized_odds_difference(
        y_test, y_pred, sensitive_features=sensitive_feature
    )



    print(f"\n── {group_name.upper()} ──────────────────────")
    print(f"Demographic parity difference : {dpd:.3f}")
    print(f"Equalized odds difference     : {eod:.3f}")

    # Per-group accuracy breakdown
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    print(f"Accuracy by group:\n{mf.by_group}")

    # ── 5. Plot per-group accuracy ────────────────────────────────────────────
    mf.by_group.plot(kind="bar", legend=False,
                     color=["#7C3AED", "#0D9488"])
    plt.title(f"Accuracy by {group_name} group")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"outputs/fairness_{group_name}.png", dpi=150)
    plt.close()
    print(f"Saved: outputs/fairness_{group_name}.png")