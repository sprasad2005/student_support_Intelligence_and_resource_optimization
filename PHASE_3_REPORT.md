# Phase 3 Report: Operations Research Resource Allocation

**Project:** Student Learning Outcome Prediction & Resource Allocation  
**Pipeline Stage:** ML Predictions → Benefit/Demand Formulation → 0-1 Integer Linear Program (PuLP) → Optimal Support Allocation → Capacity Sensitivity Analysis  
**Status:** Successfully Executed & Validated  

---

## 1. Objective
The objective of Phase 3 is to develop a mathematical **Operations Research (OR) optimization model** that resolves the resource allocation problem:
> *"Given a strictly limited budget of teacher/specialist support hours per week, which students should receive intervention to maximize total academic support impact?"*

While Phase 2 identifies student learning outcomes and classifies 282 students into support-demanding tiers (totaling 1,080 unconstrained hours), real-world academic institutions operate under tight resource ceilings. This OR engine determines the optimal subset of students to support within any specified capacity constraint.

---

## 2. Decision Variables
For each student $i \in \{1, 2, \dots, N\}$ in the candidate pool ($\text{support\_demand\_hours}_i > 0$):

$$x_i \in \{0, 1\}$$

- $x_i = 1$: Student $i$ is selected to receive targeted remedial support.
- $x_i = 0$: Student $i$ is not selected for support in the current allocation cycle.

*Note:* Low Risk students ($\text{support\_demand\_hours}_i = 0$) are excluded from the decision candidate pool ($x_i = 0$).

---

## 3. Objective Function
The objective maximizes the aggregate addressed risk score across all selected students:

$$\text{Maximize } Z = \sum_{i=1}^{N} (\text{risk\_score}_i \times x_i)$$

Where:
- $\text{risk\_score}_i = 100 - (5 \times \hat{G3}_i)$ represents the intervention urgency / academic deficiency penalty for student $i$.

---

## 4. Resource Capacity Constraint
The total weekly support hours allocated across all selected students cannot exceed the total available teacher/specialist capacity:

$$\sum_{i=1}^{N} (\text{support\_demand\_hours}_i \times x_i) \le \text{TOTAL\_SUPPORT\_HOURS}$$

Where support demand per student follows the Phase 1 & 2 protocol:
- **High Risk ($\hat{G3} < 10$):** $5.0$ hours/week
- **Medium Risk ($10 \le \hat{G3} \le 13$):** $2.0$ hours/week
- **Low Risk ($\hat{G3} \ge 14$):** $0.0$ hours/week

---

## 5. Baseline Resource Capacity
- **Demonstration Baseline Budget:** `300` hours/week.
- **Unconstrained Cohort Demand:** `1,080` hours/week (860h High Risk + 220h Medium Risk).
- **Resource Scarcity Ratio:** $300 / 1080 = 27.78\%$ of total demand can be satisfied.

---

## 6. Baseline Optimization Results (300 Hours Capacity)
The problem was solved using the open-source **CBC (Coin-or Branch and Cut)** solver via **PuLP**:

| Metric | Result Value | Notes |
| :--- | :--- | :--- |
| **Solver Status** | **`Optimal`** | Global mathematical optimum found ($<0.1$ sec) |
| **Total Students in Cohort** | `395` | Full dataset |
| **Students Requiring Support** | `282` | 172 High Risk + 110 Medium Risk |
| **Available Capacity** | `300` hours/week | Configured ceiling |
| **Allocated Support Hours** | **`300` hours/week** | Exact budget match |
| **Unused Capacity** | `0` hours/week | 100.00% capacity utilization |
| **Total Students Selected** | **`114` students** | (24 High Risk + 90 Medium Risk) |
| **High Risk Students Selected** | `24` / 172 (13.95%) | 24 students × 5 hrs = 120 hrs |
| **Medium Risk Students Selected** | `90` / 110 (81.82%) | 90 students × 2 hrs = 180 hrs |
| **Low Risk Students Selected** | `0` / 113 (0.00%) | 0 hours demand |
| **Total Risk Score Addressed** | **`6,385.60` points** | Maximized objective value |
| **Resource Utilization** | **`100.00%`** | $300 / 300 \times 100$ |
| **Demand Coverage** | **`27.78%`** | $300 / 1080 \times 100$ |

---

## 7. Sensitivity Analysis Across Capacity Levels
To evaluate how resource scaling impacts student coverage and risk alleviation, the optimization was executed across multiple capacity thresholds:

| Capacity | Allocated Hours | Unused Hours | Total Selected | High Risk Selected (of 172) | Medium Risk Selected (of 110) | Risk Score Addressed | Demand Coverage | Resource Utilization |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 hrs** | 100 hrs | 0 hrs | 50 | 0 | 50 | 2,354.70 | 9.26% | 100.00% |
| **200 hrs** | 200 hrs | 0 hrs | 88 | 8 | 80 | 4,415.80 | 18.52% | 100.00% |
| **300 hrs** | 300 hrs | 0 hrs | 114 | 24 | 90 | 6,385.60 | 27.78% | 100.00% |
| **500 hrs** | 500 hrs | 0 hrs | 166 | 56 | 110 | 9,648.00 | 46.30% | 100.00% |
| **700 hrs** | 700 hrs | 0 hrs | 206 | 96 | 110 | 12,155.60 | 64.81% | 100.00% |
| **1000 hrs**| 1000 hrs | 0 hrs | 266 | 156 | 110 | 15,461.60 | 92.59% | 100.00% |

All sensitivity results are preserved in [outputs/capacity_analysis.csv](file:///d:/or%20project/outputs/capacity_analysis.csv).

---

## 8. Pedagogical & Decision Interpretation
> **Pipeline Synergy:**
> - The **Machine Learning model** (Phase 2) objectively predicts which students are at risk and quantifies their expected academic deficiency ($\hat{G3}$ and `risk_score`).
> - The **Operations Research model** (Phase 3) translates these predictive insights into mathematically optimal intervention schedules under finite operational constraints.
>
> In knapsack optimization terms, students with smaller support footprints ($2\text{ hrs}$) and moderate risk scores often yield high marginal benefit-to-cost ratios ($\approx 22\text{ pts/hr}$), while high-risk students requiring intensive $5\text{ hr}$ interventions are prioritized when capacity expands or when individual risk scores are extraordinarily high.

---

## 9. Automated Validation Verification
All 10 assertions passed without error:
1. `[PASS]` Input/output cohort contains exactly 395 students.
2. `[PASS]` Every selected student has `support_demand_hours > 0`.
3. `[PASS]` Unselected students strictly have `selected_for_support = 0`.
4. `[PASS]` Allocated hours ($300\text{h}$) strictly respect capacity ($300\text{h}$).
5. `[PASS]` Selection flags are strictly binary $\{0, 1\}$.
6. `[PASS]` All selected students exist in `predictions.csv`.
7. `[PASS]` Zero duplicate student IDs in allocation output.
8. `[PASS]` Solver status confirmed as `Optimal`.
9. `[PASS]` Total allocated hours match individual student demand sum.
10. `[PASS]` Student risk scores strictly match Phase 2 predictions.

---

## 10. Limitations & Assumptions
- **Support-Hour Assumptions:** The 5h (High) / 2h (Medium) / 0h (Low) allocations are synthetic domain heuristics defined for prototype demonstration.
- **Linear Benefit Proxy:** The `risk_score` metric assumes linear urgency and does not incorporate diminishing marginal returns or dynamic multi-week progress tracking.
- **No Causal Guarantee:** The optimizer solves the allocation problem based on predicted risk, but does not simulate or guarantee post-intervention grade improvement.
- **Non-ASD Dataset:** The underlying dataset remains the UCI secondary student dataset; clinical or specialized neurodivergent intervention constraints would require specialized IEP datasets.
