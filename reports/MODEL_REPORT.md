# Autism Spectrum Disorder (ASD) Risk Screening Model Report

## 1. Executive Summary

This report documents the machine learning classifier developed for the **Autism Spectrum Disorder Risk Screening and Support Resource Allocation** system.

- **Domain:** Autism Spectrum Disorder (ASD) Screening Risk Prioritization
- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Dataset:** Official UCI Machine Learning Repository — Autism Screening Adult Dataset (Dr. Fadi Fayez Thabtah, 704 instances)
- **Target Variable:** `Class/ASD` (Binary: 0 = Non-ASD, 1 = ASD Traits)
- **Objective:** Predict ASD screening risk probability $P(\text{ASD} = 1)$ to prioritize individuals for specialized support intervention and clinical referral.

> [!IMPORTANT]
> **Clinical Governance Notice:** This model is an academic screening prototype designed for risk prioritization. It is **NOT** a medical diagnostic instrument and does not replace formal clinical assessment by licensed healthcare professionals.

---

## 2. Dataset Profile & Preprocessing

### 2.1 Feature Schema (18 Total Predictors)

| Category | Features | Data Type | Preprocessing & Encoding |
|---|---|---|---|
| **AQ-10 Items** | `A1_Score` to `A10_Score` | Binary (0/1) | Most-frequent imputation, binary integer |
| **Demographics** | `age` | Continuous | Median imputation, outlier clipping [1, 100] |
| **Demographics** | `gender`, `ethnicity`, `contry_of_res` | Categorical | Constant imputation ('Unknown') + `OneHotEncoder(handle_unknown='ignore')` |
| **Medical / Clinical History** | `jundice` (born with jaundice), `austim` (family history of ASD) | Categorical (yes/no) | `OneHotEncoder(handle_unknown='ignore')` |
| **Contextual** | `used_app_before`, `relation` (who completed screening) | Categorical | `OneHotEncoder(handle_unknown='ignore')` |

### 2.2 Class Distribution
- **Non-ASD (0):** 515 instances (73.15%)
- **ASD Traits (1):** 189 instances (26.85%)
- **Class Balancing:** `class_weight='balanced'` in Random Forest to prioritize minority class sensitivity.

---

## 3. Training & Evaluation Protocol

- **Split Protocol:** Stratified 80% Train ($N = 563$) / 20% Holdout Test ($N = 141$).
- **Random Seed:** `random_state = 42` (reproducible across runs).
- **Hyperparameters:**
  - `n_estimators = 200`
  - `criterion = 'gini'`
  - `class_weight = 'balanced'`
  - `n_jobs = -1`

---

## 4. Test Set Classification Performance

| Metric | Holdout Test Value | Evaluation Context |
|---|---|---|
| **Accuracy** | **94.33%** | Overall correct classifications across 141 test cases |
| **Precision** | **91.67%** | Proportion of positive predictions that were true ASD traits |
| **Recall (Sensitivity)** | **86.84%** | **Critical screening metric:** proportion of actual ASD cases detected |
| **F1-Score** | **89.19%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9894** | High separability between risk distributions |

### Confusion Matrix (Test Set, $N = 141$)

| Actual \ Predicted | Predicted Non-ASD (0) | Predicted ASD (1) |
|---|---|---|
| **Actual Non-ASD (0)** | **100 (True Negatives)** | **3 (False Positives)** |
| **Actual ASD (1)** | **5 (False Negatives)** | **33 (True Positives)** |

---

## 5. Feature Importance Attribution

Feature attributions extracted from the Random Forest ensemble:

| Rank | Feature | Gini Importance Weight | Description / Interpretation |
|---|---|---|---|
| 1 | `A9_Score` | 0.1661 | AQ-10 item 9 (social communication / detail focus) |
| 2 | `A5_Score` | 0.1130 | AQ-10 item 5 (social interaction / reading intent) |
| 3 | `A6_Score` | 0.0973 | AQ-10 item 6 (social ease / conversation flow) |
| 4 | `A4_Score` | 0.0783 | AQ-10 item 4 (attentional switching / flexibility) |
| 5 | `A10_Score` | 0.0691 | AQ-10 item 10 (facial expression / emotional recognition) |
| 6 | `A3_Score` | 0.0672 | AQ-10 item 3 (sensory / pattern concentration) |
| 7 | `A7_Score` | 0.0413 | AQ-10 item 7 (imagination / theory of mind) |
| 8 | `age` | 0.0395 | Age of individual at screening |
| 9 | `A2_Score` | 0.0367 | AQ-10 item 2 (auditory / background focus) |
| 10 | `A8_Score` | 0.0345 | AQ-10 item 8 (social observation) |

> [!NOTE]
> Feature importance indicates statistical model reliance and **does not establish clinical causation**.

---

## 6. Risk Scoring & Priority Mapping

- $\text{Risk Score} = P(\text{ASD} = 1) \times 100$
- **High Risk ($P \ge 0.70$):** 5 hours/week specialist intervention recommendation.
- **Medium Risk ($0.30 \le P < 0.70$):** 2 hours/week small-group support recommendation.
- **Low Risk ($P < 0.30$):** 0 hours/week (routine standard monitoring).
