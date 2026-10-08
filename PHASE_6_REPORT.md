# Phase 6 Report: Fully Dynamic Multi-Dataset Student Support System

**Project:** Student Learning Support & Resource Optimization  
**Pipeline Enhancement:** Dynamic Multi-Dataset Ingestion → Runtime ML Inference → Dynamic OR Optimization → Interactive Multi-Cohort Dashboard  
**Status:** Successfully Implemented, Tested & Validated  

---

## 1. Overview of Changes
In Phase 6, the system was decoupled from static, hardcoded predictions. The Streamlit dashboard ([app.py](file:///d:/or%20project/app.py)) was transformed into a **fully dynamic analytics and decision-support engine** capable of ingesting arbitrary new student cohorts via CSV upload while preserving the baseline UCI benchmark as an instant demo dataset.

---

## 2. Dynamic System Architecture

```text
                           ┌───────────────────────────────┐
                           │      Select Data Source       │
                           └───────────────┬───────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
         [ Demo Dataset (UCI 395) ]                     [ Upload Custom CSV ]
                    │                                             │
                    │                                             ▼
                    │                              ┌─────────────────────────────┐
                    │                              │  Feature Schema Validation  │
                    │                              │    (32 Required Features)   │
                    │                              └──────────────┬──────────────┘
                    │                                             │
                    │ (Precomputed Outputs)                       ▼ (Runtime Inference)
                    │                              ┌─────────────────────────────┐
                    │                              │ Pre-Trained Random Forest   │
                    │                              │  (student_performance_rf)   │
                    │                              └──────────────┬──────────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │      Active Predictions DataFrame           │
                    │ (student_id, predicted_grade, risk_level,   │
                    │  risk_score, support_demand_hours)          │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │  Dynamic Operations Research Optimization   │
                    │  (PuLP 0-1 ILP with Adaptive Capacity)     │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │   Fully Dynamic Streamlit Dashboard Views   │
                    │   (Overview, Risk, Allocation, Explorer)    │
                    └─────────────────────────────────────────────┘
```

---

## 3. Model Compatibility & Feature Schema
The pre-trained model ([models/student_performance_rf.joblib](file:///d:/or%20project/models/student_performance_rf.joblib)) dynamically introspects its input transformer to enforce the exact 32 predictor features:

- **17 Categorical Features:** `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`
- **15 Numerical Features:** `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`, `G1`, `G2`
- **Target `G3` Independence:** The target variable `G3` is strictly **NOT required** in uploaded datasets. If present, it is excluded from model inference to simulate real-world early-warning deployments.

---

## 4. Dynamic Prediction & OR Workflows

1. **Identifier Ingestion:** If the uploaded CSV includes a student identifier column (`student_id`, `id`, etc.), it is preserved; otherwise, sequential IDs (`STU001`, `STU002`, ...) are automatically generated.
2. **Inference Execution:** The pre-trained pipeline runs in $<0.05$ seconds to compute continuous predicted grades ($\hat{G3} \in [0, 20]$).
3. **Risk & Demand Mapping:**
   - **High Risk ($\hat{G3} < 10.0$):** $5\text{ hrs/wk}$ support demand
   - **Medium Risk ($10.0 \le \hat{G3} \le 13.0$):** $2\text{ hrs/wk}$ support demand
   - **Low Risk ($\hat{G3} \ge 14.0$):** $0\text{ hrs/wk}$ support demand
   - **Risk Urgency Score:** $\text{Risk Score} = 100 - (5 \times \hat{G3})$
4. **Adaptive Optimization:**
   - The capacity slider adapts its ceiling to $\sum \text{support\_demand\_hours}$ of the active cohort.
   - The PuLP 0-1 knapsack solver solves the allocation for the active cohort in real-time.

---

## 5. Error Handling & Robustness
The system guards against common data entry issues:
- **Delimiter Detection:** Auto-handles both comma (`,`) and semicolon (`;`) delimited CSVs.
- **Missing Features:** Catches missing columns and lists them with clear diagnostic alerts instead of crashing.
- **Type Coercion:** Coerces numeric strings and fills missing numerical values with zeros.
- **State Isolation:** Uploaded data is processed in memory and never overwrites baseline files.

---

## 6. Test Suite & Validation Results

| Test Scenario | Cohort Source | Records | Total Demand | Capacity Tested | Result / Coverage | Validation Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Test A: Demo Benchmark** | `student-mat.csv` | 395 | 1,080h | 300h | 114 selected (24 HR, 90 MR), 27.78% coverage | `PASSED` (Exact Match) |
| **Test B: Small Cohort** | `data/test_cohort_25.csv` | 25 | 52h | 30h | 9 selected (4 HR, 5 MR), 57.69% coverage | `PASSED` |
| **Test C: Medium Cohort** | `data/test_cohort_50.csv` | 50 | 146h | 50h | 17 selected (5 HR, 12 MR), 33.56% coverage | `PASSED` |
| **Test D: Incompatible Schema** | `data/test_invalid_missing_cols.csv` | 25 | N/A | N/A | Clean diagnostic error: Missing `G1`, `studytime` | `PASSED` (No Crash) |

---

## 7. Interactive Export Functionality
Two export buttons are provided in the sidebar for the active dataset:
1. **Download Predictions CSV:** Export predictions, risk tiers, and demand for the active cohort.
2. **Download Optimized Allocation CSV:** Export the baseline optimal intervention schedule.

---

## 8. Academic Scope & Limitations Notice
- **Non-ASD Dataset:** This system demonstrates performance prediction and resource allocation using secondary school academic datasets. It is **NOT** an autism diagnosis or clinical decision system.
- **Extensible Architecture:** The identical ML + OR pipeline architecture can be adapted to ASD-specific educational datasets upon availability of validated clinical metrics.
