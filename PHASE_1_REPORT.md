# Phase 1 Report: Dataset Understanding & Preparation

**Project:** Student Learning Outcome Prediction & Resource Allocation  
**Pipeline Architecture:** Student Data → ML Prediction → Risk/Support Level → Operations Research → Optimal Support Allocation → Streamlit Dashboard  
**Phase:** 1 — Dataset Understanding, Exploratory Data Analysis & Feature Specification  
**Status:** Completed  

---

## 1. Project Objective
The goal of this project is to build a fast, end-to-end decision-support prototype that:
1. Takes student academic, engagement, and background data.
2. Applies a lightweight Machine Learning (ML) model to predict final academic performance and classify students into actionable **Risk / Support Levels** (High, Medium, Low Risk).
3. Passes these risk scores and support demands into an **Operations Research (OR)** mathematical optimization engine to optimally allocate constrained educational resources (e.g., specialized tutoring hours, mentoring slots, remedial workshops, counselor capacity).
4. Visualizes both predictions and optimal resource assignments in an interactive **Streamlit Dashboard**.

---

## 2. Dataset Name
**UCI Student Performance Dataset (`student-mat.csv`)**  
Course Track: Mathematics (*Secondary Education*).

---

## 3. Dataset Source
- **Repository:** UCI Machine Learning Repository
- **Original Publication:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, March 2008.
- **Data Collection:** Collected from two public secondary schools in the Alentejo region of Portugal (Gabriel Pereira and Mousinho da Silveira) through school reports and questionnaires.

---

## 4. Number of Records
- **Total Rows / Instances:** `395` students.

---

## 5. Number of Features
- **Total Columns:** `33` columns.
  - **Input Attributes:** `32` features (16 categorical / string features, 16 numeric / integer features).
  - **Target Column:** `1` (`G3` — Final Grade).

---

## 6. Important Features & Domain Interpretation

| Feature Name | Type | Range / Values | Domain & Learning Outcome Significance |
| :--- | :--- | :--- | :--- |
| **`G1`** | Numeric | 0 – 20 | First period grade; strongest early baseline indicator of subject mastery. |
| **`G2`** | Numeric | 0 – 20 | Second period grade; proximate indicator directly preceding final assessment. |
| **`failures`** | Numeric | 0 – 3 | Number of past class failures; high values correlate strongly with academic vulnerability. |
| **`studytime`** | Numeric (Ordinal) | 1 (<2h) to 4 (>10h) | Weekly study commitment; proxy for academic effort and learning habit. |
| **`absences`** | Numeric | 0 – 75 | Number of missed classes; proxy for absenteeism, disengagement, or chronic health issues. |
| **`freetime`** | Numeric (Ordinal) | 1 (very low) to 5 (very high) | Free time after school; unguided time availability for self-study vs distractions. |
| **`goout`** | Numeric (Ordinal) | 1 (very low) to 5 (very high) | Frequency of social outing with peers; high values often compete with study time. |
| **`age`** | Numeric | 15 – 22 | Student age; older students in secondary classes often reflect past grade retentions. |
| **`school`** | Categorical | `GP`, `MS` | School institution (`GP`: Gabriel Pereira, `MS`: Mousinho da Silveira). |
| **`sex`** | Categorical | `F`, `M` | Student gender (`F`: Female, `M`: Male). |
| **`address`** | Categorical | `U`, `R` | Home address type (`U`: Urban, `R`: Rural); indicator of geographic and transit accessibility. |
| **`famsize`** | Categorical | `LE3`, `GT3` | Family size (`LE3`: $\le 3$, `GT3`: $> 3$). |
| **`Medu` / `Fedu`**| Numeric (Ordinal) | 0 (none) to 4 (higher ed) | Parental education levels; socioeconomic proxy strongly correlated with home learning support. |
| **`paid`** | Categorical | `yes`, `no` | Extra paid subject tutoring; existing commercial support intervention. |
| **`internet`** | Categorical | `yes`, `no` | Home internet access; essential for digital learning material access. |
| **`health`** | Numeric (Ordinal) | 1 (very bad) to 5 (very good) | Self-reported health status; affects stamina and consistent learning participation. |
| **`higher`** | Categorical | `yes`, `no` | Aspiration to pursue higher education; high correlation with academic motivation. |
| **`schoolsup`** | Categorical | `yes`, `no` | Extra educational support provided by the school. |
| **`famsup`** | Categorical | `yes`, `no` | Family educational support. |

---

## 7. Target Variable Analysis (`G3`)
- **Name:** `G3` (Final Course Grade in Mathematics).
- **Scale:** Continuous / Discrete scale from `0` to `20` (Portuguese national grading scale, where passing is $\ge 10$).
- **Distribution Summary Statistics:**
  - **Count:** 395
  - **Mean:** `10.42`
  - **Median:** `11.00`
  - **Std Dev:** `4.58`
  - **Min:** `0`
  - **Max:** `20`
  - **Unique Values:** 18 distinct values.
- **Zero-Grade Observation:** Exactly 38 students (9.62%) have $G3 = 0$. This represents students who dropped out, were disqualified due to excessive absences, or did not sit for the final exam.

---

## 8. Data Quality Findings
1. **Missing Values:** `0` missing values across all 33 columns (clean dataset).
2. **Duplicate Rows:** `0` duplicate rows.
3. **Invalid Values:** All features adhere strictly to their defined valid ranges in documentation.
4. **Outliers:** 
   - Feature `absences` has a long right tail (max 75, with values like 54, 56, 75). These are legitimate non-synthetic records reflecting chronic absenteeism.
   - Non-linear tree-based models (such as Random Forest) naturally handle these extreme attendance values without skewing linear weights.
5. **Class / Distribution Balance:**
   - **High Risk ($G3 < 10$):** 130 students (32.91%) — *Require urgent / high remedial allocation.*
   - **Medium Risk ($10 \le G3 \le 13$):** 165 students (41.77%) — *Require moderate maintenance support.*
   - **Low Risk ($G3 \ge 14$):** 100 students (25.32%) — *Low risk / self-sufficient.*
   - All 3 risk tiers have substantial sample support with no severe class collapse.

---

## 9. ML Problem Type Formulation
We evaluate two formulation options:

1. **Option A: Regression ($G3 \in [0, 20]$)**  
   *Predict the exact numerical final score $\hat{G3}$.*
   - *Pros:* Retains full granular detail; allows computing a continuous Priority/Risk score ($Risk = 100 - 5 \times \hat{G3}$) and estimating expected score deficits for linear programming objectives in OR.
   - *Downstream conversion:* Easily binned into discrete support categories (`High`, `Medium`, `Low`).

2. **Option B: Multiclass Classification (3 Risk Categories)**  
   *Predict categorical class directly: High Risk vs Medium Risk vs Low Risk.*
   - *Pros:* Direct category mapping.
   - *Cons:* Loses intra-category ranking (e.g., student at 4/20 vs student at 9/20 both labeled "High Risk", making fair allocation harder).

### Recommended Approach for Prototype:
**Regression with Downstream Tier Binning** (or direct 3-Tier Classification with predicted probabilities).  
Predicting $\hat{G3} \in [0, 20]$ gives the Operations Research model both:
- **A ranking priority metric** (who needs help most urgently).
- **A discrete risk tier assignment** (High / Medium / Low).

---

## 10. Recommended ML Model
- **Primary Recommendation:** **Random Forest Regressor** (from `scikit-learn`).
- **Rationale for 3–4 Hour Prototype:**
  1. **Instant Training:** Trains in $<1$ second on 395 records.
  2. **Mixed Data Handling:** Handles combined numeric and one-hot/ordinal encoded categorical features seamlessly.
  3. **Robustness:** Immune to monotonic feature scaling issues and resilient against extreme outlier values in `absences`.
  4. **Interpretability:** Provides native Gini / feature importance scores to explain *why* a student is flagged at risk.
  5. **No Deep Learning Overhead:** Avoids heavy CNN-LSTM architectures, PyTorch GPU configurations, or slow convergence cycles.

---

## 11. Proposed Feature Set & Data Leakage Assessment

### Selected Feature Categories:
- **Academic Performance:** `G1`, `G2`, `failures`, `studytime`, `paid`
- **Attendance & Engagement:** `absences`, `freetime`, `goout`, `activities`
- **Background & Demographics:** `age`, `school`, `sex`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`
- **Support & Well-being:** `schoolsup`, `famsup`, `higher`, `internet`, `romantic`, `famrel`, `Dalc`, `Walc`, `health`

### Data Leakage Analysis on $G1$ and $G2$:
- $G1$ (Period 1) and $G2$ (Period 2) correlate strongly with $G3$ ($r = 0.80$ and $r = 0.90$).
- **Is this acceptable?** **Yes.** In educational decision systems, this represents a **mid-term early warning system** where mid-period grades are genuinely available before the final exam to allocate remedial resources before the term ends.
- **Protocol:** $G1$ and $G2$ are strictly recognized as historical sequential milestone assessments, not post-outcome leakage.

---

## 12. Output Passed from ML to Operations Research (OR)
The ML model will generate a structured table for each student $i$ containing:

| Output Field | Data Type | Description | OR Usage |
| :--- | :--- | :--- | :--- |
| `student_id` | String/Int | Unique student identifier (`S001`, `S002`, ...) | Decision variable indexing $x_{i, j}$ |
| `predicted_grade` ($\hat{G3}$) | Float | Predicted final score ($0.0 - 20.0$) | Objective function weight / deficit quantification |
| `risk_level` | Categorical | `High Risk`, `Medium Risk`, `Low Risk` | Eligibility filter for high-intensity resources |
| `risk_score` | Float | Normalized urgency metric: $100 - (5 \times \hat{G3})$ | Priority penalty / reward coefficient in OR objective |
| `support_demand_hours` | Float | Estimated weekly remedial hours needed | Demand constraint: $\sum_j y_{i, j} \ge \text{demand}_i$ |

### Example ML → OR Payload:
- **Student 1 (`S001`):** $\hat{G3} = 16.4$ → `Low Risk` (Risk Score: 18.0) → Support Demand: 0–1 hr/wk
- **Student 2 (`S002`):** $\hat{G3} = 11.2$ → `Medium Risk` (Risk Score: 44.0) → Support Demand: 2–3 hrs/wk
- **Student 3 (`S003`):** $\hat{G3} = 5.8$ → `High Risk` (Risk Score: 71.0) → Support Demand: 5–6 hrs/wk + 1-on-1 Tutor

---

## 13. Important Limitations of the Dataset

> [!IMPORTANT]
> **NOT AN AUTISM SPECTRUM DISORDER (ASD) DATASET:**
> The UCI Student Performance dataset reflects general secondary school students in Portugal. It does **NOT** contain medical, diagnostic, cognitive, or behavioral indicators specific to Autism Spectrum Disorder (ASD).
>
> **Prototype Scope:**
> This dataset is utilized strictly as a proof-of-concept for the algorithmic framework: **Student Data → ML Prediction → Risk Tiering → OR Resource Allocation Optimization → Streamlit Visualization**.
>
> Actual deployment for neurodiverse or ASD-specific learning support would require clinically validated datasets incorporating individualized education plans (IEPs), sensory accommodations, behavioral therapies, and ASD diagnostic metrics.
