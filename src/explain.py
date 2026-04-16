# ── 1. Packages ───────────────────────────────────────────────────────────────
import shap
import joblib
import matplotlib.pyplot as plt
import os

# ── 2. Load model and data ────────────────────────────────────────────────────
model   = joblib.load("outputs/model.joblib")
X_train = joblib.load("outputs/X_train.joblib")
X_test  = joblib.load("outputs/X_test.joblib")
y_test  = joblib.load("outputs/y_test.joblib")

os.makedirs("outputs", exist_ok=True)

# ── 3. Create SHAP explainer ──────────────────────────────────────────────────
# TreeExplainer is optimised for tree-based models (XGBoost, RF, etc.)
# X_train is the background dataset — used to compute expected values
explainer   = shap.TreeExplainer(model, X_train)
shap_values = explainer(X_test)

# ── 4. Global — Beeswarm plot ─────────────────────────────────────────────────
print("Generating beeswarm plot...")
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("Global Feature Importance — SHAP Beeswarm")
plt.tight_layout()
plt.savefig("outputs/shap_beeswarm.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: outputs/shap_beeswarm.png")

# ── 5. Local — Waterfall plots for 3 representative cases ────────────────────
# Find 3 interesting cases:
# - one true positive  (actual=1, predicted=1) — correctly flagged risk
# - one false positive (actual=0, predicted=1) — wrongly flagged as risk
# - one false negative (actual=1, predicted=0) — missed risk

y_pred = model.predict(X_test)
y_test_arr = y_test.values

import numpy as np
tp_idx = np.where((y_test_arr == 1) & (y_pred == 1))[0][0]
fp_idx = np.where((y_test_arr == 0) & (y_pred == 1))[0][0]
fn_idx = np.where((y_test_arr == 1) & (y_pred == 0))[0][0]

cases = {
    "true_positive":  tp_idx,
    "false_positive": fp_idx,
    "false_negative": fn_idx,
}

for name, idx in cases.items():
    plt.figure()
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    plt.title(f"Local Explanation — {name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_waterfall_{name}.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: outputs/shap_waterfall_{name}.png")

print("\nAll SHAP plots saved ")