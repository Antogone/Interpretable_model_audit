# Interpretable Model Audit: German Credit Dataset

## Overview

The German Credit dataset contains 1,000 loan applications from a German bank,
each labelled as good or bad credit risk. The goal is to predict creditworthiness
from applicant characteristics: age, employment status, loan amount, loan
duration, savings, credit history, and others.

This project trains an XGBoost classifier on this data and then audits it
rigorously: not just measuring accuracy, but interrogating why the model makes
the decisions it makes, and whether it treats all demographic groups equitably.

The audit surfaces two critical findings: severe performance disparities for
foreign workers (equalized odds difference 0.467) and unemployed applicants
(0.414). The conclusion is that this model should not be deployed without
significant mitigation.

**Model:** XGBoost
**Dataset:** Statlog German Credit (UCI, n=1,000)
**Performance:** AUC 0.745 · Accuracy 0.715

---

## Project Structure

```
project/
├── src/
│   ├── train.py            ← data loading, preprocessing, XGBoost training
│   ├── explain.py          ← SHAP global + local explanations
│   └── fairness_audit.py   ← fairlearn metrics across 5 demographic groups
├── outputs/
│   ├── model.joblib
│   ├── shap_beeswarm.png
│   ├── shap_waterfall_true_positive.png
│   ├── shap_waterfall_false_positive.png
│   ├── shap_waterfall_false_negative.png
│   ├── fairness_age.png
│   ├── fairness_gender.png
│   ├── fairness_foreign_worker.png
│   ├── fairness_housing.png
│   └── fairness_employment.png
├── audit_report.md         ← full written audit with recommendations
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

python src/train.py           # train model, save to outputs/
python src/explain.py         # generate SHAP plots
python src/fairness_audit.py  # generate fairness metrics and plots
```

---

## Data Preprocessing

The UCI Statlog German Credit dataset uses anonymous column names
(Attribute1 through Attribute20). These were mapped to meaningful names
using the official dataset documentation:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

This mapping is documented in train.py with a comment linking to the source,
so anyone reading the code can verify where the names came from.

Preprocessing steps applied before training:

- No missing values were found, so imputation was not required
- Categorical columns encoded with pd.get_dummies(drop_first=True), which
  drops one category per variable to avoid multicollinearity
- Numerical columns scaled with StandardScaler fitted on training data only,
  to prevent data leakage into the test set
- Target remapped: 1 (good credit) → 0, 2 (bad credit) → 1, to produce
  a standard binary classification target

No sklearn Pipeline was used deliberately. SHAP's TreeExplainer requires
direct access to the XGBoost model object and works best when the
preprocessed data retains column names as a pandas DataFrame, which a
Pipeline's numpy output would lose.

---

## SHAP Analysis

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
score for each individual prediction. The scores are grounded in game theory:
they represent the average marginal contribution of each feature across all
possible orderings of features. They sum exactly to the difference between
the model's prediction and the average prediction across the dataset.

Two levels of analysis were performed.

### Global: Beeswarm plot

![Beeswarm](outputs/shap_beeswarm.png)

The beeswarm shows all SHAP values for all test samples. Each row is one
feature. Each dot is one sample. Position on the x-axis is the SHAP value
(positive pushes toward bad credit, negative toward good credit). Colour
encodes the feature value (red = high, blue = low). Features are sorted by
mean absolute SHAP value, so the most influential features appear at the top.

The three most influential features globally:

**checking_status_A14** (no checking account): the dominant signal by a wide
margin. When an applicant has no checking account (feature value = 1, red),
SHAP values are strongly negative, meaning the model associates this with
lower credit risk. This is counterintuitive and worth flagging: having no
checking account at all may not be a reliable proxy for creditworthiness.

**credit_amount and duration**: higher values increase predicted risk. This
is financially intuitive, as larger loans over longer periods carry more
repayment risk.

**age**: older applicants consistently receive lower risk scores, independent
of other features. This directional bias directly motivated the fairness audit
in the next section.

### Local: Waterfall plots

Three representative cases were selected and analysed individually.

**True positive (correctly flagged risk)**

![True Positive](outputs/shap_waterfall_true_positive.png)

The model correctly identified this applicant as a credit risk. The prediction
was driven almost entirely by checking_status_A14 (+0.78). This is a confident,
explainable, and defensible prediction.

**False positive (incorrectly flagged as risk)**

![False Positive](outputs/shap_waterfall_false_positive.png)

This applicant was incorrectly flagged. The final score was 0.094, barely above
the 0.5 decision threshold. The model was pushed toward risk by checking_status_A14
(+0.52) but partially corrected by low credit_amount (-0.47). This is a borderline
case, not a confident wrong prediction. A calibrated threshold would likely
resolve this error without retraining.

**False negative (missed risk)**

![False Negative](outputs/shap_waterfall_false_negative.png)

The most concerning failure. A genuinely risky applicant was missed because
short loan duration (-0.91) overwhelmed multiple risk signals pointing in the
other direction. The model has learned a spurious association: short loan
duration implies safety, even when other indicators suggest otherwise. This
pattern is worth investigating in a production context.

---

## Fairness Audit

### What the metrics mean

Two fairness metrics were computed for each demographic group.

**Demographic parity difference (DPD):** how much more often the model
predicts bad credit for one group vs another, regardless of the true label.
A perfectly fair model has DPD = 0.

**Equalized odds difference (EOD):** how much the error rate differs between
groups. This is the more important metric for high-stakes decisions, because
it captures whether the model makes mistakes at similar rates for everyone.
A model can have DPD = 0 while still systematically misclassifying one group
at a higher rate. EOD catches this.

The threshold commonly used in financial applications is EOD > 0.10,
which flags a meaningful disparity requiring attention.

### Results

| Group | Demographic Parity Diff | Equalized Odds Diff | Accuracy Gap |
|-------|------------------------|---------------------|--------------|
| Age | 0.064 | 0.169 | 0.039 |
| Gender | 0.039 | 0.122 | 0.047 |
| Foreign worker | 0.276 | **0.467** | 0.297 |
| Housing | 0.075 | 0.146 | 0.013 |
| Employment | 0.176 | **0.414** | 0.385 |

### Why these five groups?

Groups were selected based on three criteria: legal relevance (age, gender,
and nationality are protected characteristics under German anti-discrimination
law, AGG §1), data availability (only groups reconstructable from the existing
features were audited), and analytical value (housing was included as a
non-protected control group to contextualise the other numbers).

### Critical findings

**Foreign worker (EOD 0.467)**

![Foreign Worker](outputs/fairness_foreign_worker.png)

The model achieves 100% accuracy on non-foreign applicants but only 70.3%
on foreign applicants, a 29.7% accuracy gap. An equalized odds difference
of 0.467 is severe. Using nationality as a credit scoring signal is
prohibited under German anti-discrimination law (AGG). This is a legal
blocker for deployment, not merely a fairness concern.

**Employment (EOD 0.414)**

![Employment](outputs/fairness_employment.png)

Unemployed applicants receive only 35.7% accuracy, barely above random. The
model has effectively no predictive power for this group. Note that only 7
unemployed applicants appear in the test set, which makes this estimate noisy,
but the direction is unambiguous and warrants investigation with more data.

**Age (EOD 0.169) and gender (EOD 0.122)**

Both exceed the 0.10 threshold for financial applications. The age finding
is consistent with the SHAP analysis, where older age was shown to push
predictions toward lower risk independently of other features. The gender
finding is likely driven by the personal_status feature, which conflates
gender and marital status in a way that introduces proxy discrimination.

**Housing (EOD 0.146)**

The smallest disparity of the five groups and the only one involving a
non-protected characteristic. Included as a control to contextualise the
other numbers.

---

## Verdict

**This model should not be deployed in its current form.**

The foreign worker disparity is a legal blocker under German law. The
employment group failure raises serious ethical concerns. Before any
deployment, the following mitigations are required:

**Blocking issues (must fix first):**
1. Remove the foreign_worker feature entirely (legal requirement under AGG)
2. Investigate whether personal_status (which encodes gender) should be removed
3. Re-audit both groups after retraining

**Non-blocking issues (should fix before deployment):**
4. Apply fairlearn.ExponentiatedGradient with an equalized odds constraint
   to address residual age and gender disparities
5. Collect more data on unemployed and foreign worker applicants before
   drawing firm conclusions about those subgroups
6. Calibrate the decision threshold away from 0.5 to reduce borderline
   false positives (no retraining required)

The full written audit with detailed findings and a retraining checklist
is in audit_report.md.

---

## Skills Demonstrated

- XGBoost training and evaluation on a real-world credit scoring dataset
- SHAP TreeExplainer: global beeswarm plot and local waterfall plots at
  the instance level for true positive, false positive, and false negative cases
- Fairness metrics via fairlearn: demographic parity difference and equalized
  odds difference across five demographic groups
- Responsible AI thinking: identifying legal implications, distinguishing
  blocking from non-blocking issues, writing a concrete remediation plan
- Written communication: translating quantitative model behaviour into plain
  English findings accessible to a non-technical audience
