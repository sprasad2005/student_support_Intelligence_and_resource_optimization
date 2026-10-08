# Phase 4 Report: Streamlit Dashboard Implementation

**Project:** Student Learning Outcome Prediction & Resource Allocation  
**Pipeline Stage:** Web Visualization & Interaction Layer (Streamlit Dashboard)  
**Status:** Successfully Implemented, Tested & Validated  

---

## 1. Dashboard Objective
The objective of Phase 4 is to build an interactive, professional, and lightweight **Streamlit web application** (`app.py`). The dashboard integrates the entire decision-support pipeline into a unified interface for academic demonstrations:
$$\text{Student Data} \longrightarrow \text{ML Prediction} \longrightarrow \text{Risk Assessment} \longrightarrow \text{OR Optimization} \longrightarrow \text{Support Allocation}$$

---

## 2. Dashboard Architecture & Pages

The application is structured into five distinct functional modules via sidebar navigation:

1. **Executive Overview:**
   - Real-time KPI summary metrics (Total Students: 395, High Risk: 172, Medium Risk: 110, Low Risk: 113, Total Demand: 1,080 hrs/wk, Baseline 300h Allocation: 300 hrs/wk).
   - Cohort risk distribution bar chart and unconstrained support demand breakdown.
   - End-to-end architectural workflow diagram and domain explanation.

2. **Student Risk Analysis:**
   - Multi-criteria filtering (by Risk Level multi-select and Predicted Grade slider range $0–20$).
   - Dynamic cohort risk distribution charts and predicted grade distribution histogram.
   - Searchable student risk table displaying student ID, grades, risk scores, and support demands.

3. **Operations Research Resource Allocation:**
   - Real-time **Capacity Slider** ($0–1080$ hours/week in 10-hour steps).
   - Real-time invocation of the PuLP/CBC linear programming solver upon slider adjustment.
   - Live KPI cards: Available Hours, Allocated Hours, Unused Capacity, Resource Utilization (%), Demand Coverage (%), Students Selected, and Total Risk Score Addressed.
   - Visualizations of capacity utilization and risk tier coverage progress bars.
   - Searchable table of selected students sorted by descending risk score.
   - Precomputed Capacity Sensitivity Analysis charts ($100\text{h}–1000\text{h}$).

4. **Individual Student Explorer:**
   - Dropdown student selector (`STU001` to `STU395`).
   - Detailed metric badges for Predicted Grade, Ground Truth Grade, Risk Tier, Urgency Risk Score, and Support Demand.
   - Contextual intervention recommendations (High Priority 5h/wk, Moderate 2h/wk, or Standard Learning Pace) with waitlist/selection status.
   - Granular background attributes (absences, study time, prior failures, health, family support).

5. **Model Insights & Governance:**
   - Phase 2 holdout performance metrics (MAE: `1.1779`, RMSE: `2.0014`, $R^2$: `0.8047`).
   - Top 10 Feature Importance horizontal bar chart (`G2`, `absences`, `reason_home`, `age`, `G1`, `famrel`, etc.).
   - Model architecture specifications (RandomForestRegressor, 200 trees, 80/20 train-test split, `random_state=42`).
   - Explicit project scope and non-clinical limitations disclaimer.

---

## 3. Data Sources & Integration

All modules consume existing, validated outputs from Phases 2 and 3 without retraining models or modifying original datasets:
- [outputs/predictions.csv](file:///d:/or%20project/outputs/predictions.csv) (395 records with predictions, risk tiers, and demand)
- [outputs/feature_importance.csv](file:///d:/or%20project/outputs/feature_importance.csv) (Trained Random Forest Gini importances)
- [outputs/optimized_allocation.csv](file:///d:/or%20project/outputs/optimized_allocation.csv) (Baseline 300h integer linear program allocation)
- [outputs/capacity_analysis.csv](file:///d:/or%20project/outputs/capacity_analysis.csv) (Precomputed sensitivity analysis across 6 capacity tiers)
- [src/optimize_resources.py](file:///d:/or%20project/src/optimize_resources.py) (Reusable `solve_resource_allocation` solver module)

---

## 4. Interactive Features
- **Dynamic In-Memory Optimization:** The dashboard dynamically invokes the PuLP 0-1 binary knapsack solver when the capacity slider is adjusted, caching results with `@st.cache_data` for instant responsiveness.
- **Bi-directional Filtering:** Users can slice the student cohort simultaneously by risk categories and numerical score intervals.
- **Individual Drill-down:** Educators can inspect individual student profiles and see exactly how mathematical optimization impacts their intervention priority.

---

## 5. Live Demonstration Results

Testing of dynamic capacity adjustments yielded the following verified mathematical allocation results:

| Demonstration Metric | Capacity = 300h (Baseline) | Capacity = 500h (Expanded) | Capacity = 700h (High Capacity) |
| :--- | :--- | :--- | :--- |
| **Solver Status** | `Optimal` | `Optimal` | `Optimal` |
| **Allocated Hours** | `300` hrs/week | `500` hrs/week | `700` hrs/week |
| **Unused Hours** | `0` hrs/week | `0` hrs/week | `0` hrs/week |
| **Resource Utilization** | `100.00%` | `100.00%` | `100.00%` |
| **Total Students Selected** | **`114` students** | **`166` students** | **`206` students** |
| **High Risk Students (of 172)** | `24` (13.95% coverage) | `56` (32.56% coverage) | `96` (55.81% coverage) |
| **Medium Risk Students (of 110)**| `90` (81.82% coverage) | `110` (100.00% coverage)| `110` (100.00% coverage)|
| **Demand Coverage** | `27.78%` | `46.30%` | `64.81%` |
| **Total Risk Score Addressed**| `6,385.60` pts | `9,648.00` pts | `12,155.60` pts |

---

## 6. Execution & Validation
- Application launch command: `streamlit run app.py`
- All 5 pages render without unhandled exceptions or missing-file errors.
- Dynamic slider adjustments recalculate optimal allocations in $< 0.1$ seconds.
- Exact match confirmed with Phase 3 offline optimization results at 300h, 500h, and 700h.

---

## 7. Scope & Limitations Notice
1. **Non-ASD Dataset:** The UCI Student Performance dataset is from general secondary education in Portugal and contains no Autism Spectrum Disorder (ASD) or neurodevelopmental diagnostic data.
2. **Academic Proof-of-Concept:** Designed strictly as a prototype demonstrating ML prediction coupled with Operations Research optimization.
3. **Synthetic Demand Heuristics:** Support demands (5h/2h/0h) and priority risk scores are modeling proxies.
4. **No Grade Improvement Guarantee:** Optimization guarantees mathematical optimality in risk coverage, not pedagogical outcome improvements.
