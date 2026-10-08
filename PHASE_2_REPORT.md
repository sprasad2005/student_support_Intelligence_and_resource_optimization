# Phase 2 Report: Student Performance Prediction Model

**Project:** Student Learning Outcome Prediction & Resource Allocation  
**Pipeline Stage:** Data Preprocessing → Train/Test Split → Random Forest Regressor → Model Evaluation → Risk Tiering & Support Demand Estimation  
**Status:** Successfully Executed & Validated  

---

## 1. Objective
The objective of Phase 2 is to build a lightweight, fast, and fully reproducible **Random Forest Regression pipeline** using scikit-learn. The model predicts the student's final Mathematics grade ($G3$), computes individual continuous risk scores, classifies each student into an actionable 3-tier risk category (High, Medium, Low Risk), and determines baseline remedial support demands for downstream Operations Research (OR) mathematical optimization.

---

## 2. Dataset Overview
- **Dataset:** UCI Student Performance (`student-mat.csv`)
- **Total Records:** `395` students
- **Total Input Features:** `32` predictor features
  - 17 Categorical features: `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`
  - 15 Numerical features: `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`, `G1`, `G2`
- **Target Variable:** `G3` (Final Mathematics Grade, scale $0–20$)
- **Data Quality:** `0` missing values, `0` duplicate rows.

---

## 3. Data Preprocessing & Pipeline Architecture
The end-to-end preprocessing and model training were assembled inside a unified scikit-learn `Pipeline`:
- **Categorical Columns (17 features):** Encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- **Numerical Columns (15 features):** Passed through directly (`passthrough`) without scaling, preserving the native interpretability of integer/ordinal scores for tree-based splitting.
- **ColumnTransformer:** Dynamically handles heterogeneous feature types and ensures zero data leakage across train and test folds.
- **Train/Test Split:** Standard 80/20 holdout split (`test_size=0.20`, `random_state=42`), producing 316 training samples and 79 validation/test samples.

```text
Input Features (32)
   ├── Categorical (17) ──► OneHotEncoder(handle_unknown='ignore') ──┐
   └── Numerical (15)   ──► Passthrough                           ──┴──► RandomForestRegressor(200 trees)
```

---

## 4. Model Selection & Configuration
- **Model:** `RandomForestRegressor`
  - `n_estimators=200`
  - `random_state=42`
  - `n_jobs=-1`
  - `max_depth=None`
- **Rationale:**
  1. **Instant Training:** Trains in $<1$ second on 395 records.
  2. **Non-linear interactions:** Naturally models interactions between mid-term grades ($G1, G2$), absenteeism, and family support.
  3. **Robustness:** Resilient against extreme skewness in attendance (e.g., $75$ absences).
  4. **Interpretability:** Natively computes Gini-based feature importances across one-hot encoded variables.
  5. **Lightweight Artifact:** Bundled into a standalone `.joblib` file with no deep-learning or GPU dependencies.

---

## 5. Model Evaluation & Performance Metrics
Evaluated on the unseen 20% holdout test set (79 students):

| Metric | Measured Value | Interpretation |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | **`1.1779`** | Predictions deviate by only ~1.18 points on a 20-point scale. |
| **RMSE (Root Mean Squared Error)** | **`2.0014`** | Low penalty for large prediction outliers. |
| **$R^2$ Score (Coefficient of Determination)** | **`0.8047`** | The model explains **~80.5%** of the total variance in final grades. |
| **Actual Mean Grade ($G3$) on Test Set** | `10.7722` | Ground truth test benchmark. |
| **Predicted Mean Grade ($\hat{G3}$) on Test Set** | `10.7207` | Near-zero calibration bias ($-0.05$ difference). |
| **Predicted Min on Test Set** | `0.13` | Accurately identifies severe academic failure risks. |
| **Predicted Max on Test Set** | `18.77` | Captures high-achieving student potential. |

---

## 6. Risk Level & Support Demand Distribution
Using the Phase 1 risk thresholds and demand mapping applied across the entire cohort (395 students):

| Risk Level | Decision Rule | Student Count | Cohort % | Demand / Student | Subtotal Support Hours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High Risk** | $\hat{G3} < 10.0$ | **172** | 43.54% | 5.0 hrs / week | **860 hours** |
| **Medium Risk** | $10.0 \le \hat{G3} \le 13.0$ | **110** | 27.85% | 2.0 hrs / week | **220 hours** |
| **Low Risk** | $\hat{G3} \ge 14.0$ | **113** | 28.61% | 0.0 hrs / week | **0 hours** |
| **Total Cohort** | — | **395** | 100.0% | — | **1,080 hours** |

- **Cohort Predicted Mean Grade:** `10.41`
- **Cohort Predicted Min Grade:** `0.04`
- **Cohort Predicted Max Grade:** `19.36`

---

## 7. Operations Research (OR) Interface Payload
Every student record in [predictions.csv](file:///d:/or%20project/outputs/predictions.csv) provides the essential parameters needed for linear/integer programming resource optimization:

```text
student_id           : STU001, STU002, ..., STU395 (Index variable i)
actual_grade         : Ground truth final score (0 - 20)
predicted_grade      : Expected academic outcome G3_hat (0.00 - 20.00)
risk_level           : Qualitative eligibility tier (High, Medium, Low Risk)
risk_score           : Normalized urgency penalty index (100 - 5 * G3_hat)
support_demand_hours : Base weekly remedial tutoring demand (5, 2, or 0 hrs)
[32 Feature Columns] : Full demographic, attendance, and behavioral context
```

---

## 8. Top 10 Feature Importances

| Rank | Feature | Importance Score | Percentage Contribution | Category |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`G2`** (2nd Period Grade) | `0.7873` | **78.73%** | Academic Milestone |
| **2** | **`absences`** | `0.1141` | **11.41%** | Attendance / Engagement |
| **3** | **`reason_home`** (School chosen for proximity) | `0.0185` | **1.85%** | Background |
| **4** | **`age`** | `0.0099` | **0.99%** | Demographic |
| **5** | **`G1`** (1st Period Grade) | `0.0061` | **0.61%** | Academic Milestone |
| **6** | **`famrel`** (Family Relationship Quality) | `0.0046` | **0.46%** | Well-being |
| **7** | **`health`** (Health Status) | `0.0041` | **0.41%** | Well-being |
| **8** | **`goout`** (Social Outings with Peers) | `0.0039` | **0.39%** | Engagement |
| **9** | **`guardian_mother`** | `0.0034` | **0.34%** | Demographic |
| **10**| **`studytime`** (Weekly Study Time) | `0.0030` | **0.30%** | Academic Commitment |

---

## 9. Automated Validation Verification
All 10 automated assertions passed during execution:
1. `[PASS]` Dataset contains exactly 395 students.
2. `[PASS]` Full prediction cohort count is 395.
3. `[PASS]` Zero missing values in predicted grades.
4. `[PASS]` Zero missing values in risk levels.
5. `[PASS]` All risk levels are strictly within `{'High Risk', 'Medium Risk', 'Low Risk'}`.
6. `[PASS]` All support demand values are strictly within `{0, 2, 5}`.
7. `[PASS]` `predicted_grade` is numeric (float64).
8. `[PASS]` Saved model exists at [student_performance_rf.joblib](file:///d:/or%20project/models/student_performance_rf.joblib).
9. `[PASS]` Predictions file exists at [predictions.csv](file:///d:/or%20project/outputs/predictions.csv).
10. `[PASS]` Feature importances file exists at [feature_importance.csv](file:///d:/or%20project/outputs/feature_importance.csv).

---

## 10. Limitations & Scope Notice
- **Not an Autism Spectrum Disorder (ASD) Dataset:** The UCI Student Performance dataset reflects general secondary education in Portugal. It contains no clinical, neurodevelopmental, or sensory metrics.
- **Prototype Scope:** This model serves as an algorithmic proof-of-concept for the prediction-to-optimization workflow.
- **Sample Size:** 395 records represent a small academic cohort, well-suited for fast prototyping and optimization demonstration.
- **Non-Clinical Application:** Designed for rapid decision-support and resource allocation modeling.
