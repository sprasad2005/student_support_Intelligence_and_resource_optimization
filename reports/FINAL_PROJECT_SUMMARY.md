# Autism Spectrum Disorder (ASD) Risk Screening & Support Resource Allocation

## 1. Project Overview

**NeuroRisk** is an end-to-end artificial intelligence and operations research system that transforms behavioral screening questionnaire data into actionable, prioritized specialist support resource allocations.

### Pipeline Architecture

```text
┌──────────────────────────────┐
│ ASD Screening Cohort CSV     │ (AQ-10 Items, Demographics, Clinical History)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Robust Preprocessing         │ (ColumnTransformer: OneHotEncoder + SimpleImputer)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Random Forest Classifier     │ (200 Trees, Balanced Class Weights)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Screening Risk Probability   │ P(ASD = 1) ∈ [0.0, 1.0]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Risk Tier & Support Demand   │ High (5h), Medium (2h), Low (0h)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Operations Research (PuLP)   │ 0-1 Integer Linear Program (Knapsack Optimization)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Streamlit SaaS Dashboard     │ Dark Theme, Analytics, History, Case Explorer
└──────────────────────────────┘
```

---

## 2. Key Technical Components

| Component | Technical Implementation |
|---|---|
| **Domain** | Autism Spectrum Disorder (ASD) Screening Risk Prioritization |
| **Dataset** | Official UCI Machine Learning Repository — Autism Screening Adult Dataset ($N = 704$) |
| **Model** | `RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')` |
| **Evaluation Metrics** | Accuracy: **94.33%** · Precision: **91.67%** · Recall: **86.84%** · F1: **89.19%** · ROC-AUC: **0.9894** |
| **Optimization Engine** | PuLP 0-1 Integer Linear Programming (`CBC` solver) |
| **User Interface** | Streamlit Dark SaaS Dashboard with Altair visualizations |
| **State & Persistence** | Fresh startup state + persistent history in `outputs/analysis_history.json` |

---

## 3. Screening vs. Clinical Diagnosis Framework

> [!IMPORTANT]
> **Ethical & Clinical Boundary:**
> - **Screening / Prioritization (Our Scope):** Estimates statistical likelihood to prioritize limited specialist assessment hours.
> - **Clinical Diagnosis (Out of Scope):** Comprehensive medical evaluation by licensed developmental pediatricians, psychiatrists, and clinical psychologists.

---

## 4. How to Run the Application

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the Random Forest Classifier
python src/train_model.py

# 3. Launch the Streamlit Web Application
streamlit run app.py
```
Application URL: `http://localhost:8501`

---

## 5. Viva / Defense Quick Reference & Critical Rules

### 4 Golden Rules for Viva Defense:

1. **Terminology (`Class/ASD = 1`):**
   - Say: *"ASD Traits Observed"* or *"Screening Classification: Trait Present"*.
   - Never say: *"ASD diagnosis"* or *"diagnosed with autism"*. The model performs behavioural screening triage, not medical diagnosis.

2. **Accuracy & Metric Claims:**
   - Say: *"On our held-out test set of 141 samples, the Random Forest achieved 94.33% accuracy and 86.84% recall."*
   - Never say: *"The system is 94.33% accurate for diagnosing autism."* (Diagnoses require clinical multi-disciplinary tools like ADOS-2 / ADI-R).

3. **Resource Sizing Heuristic (5h / 2h / 0h):**
   - Say: *"The 5h / 2h / 0h values represent a project-level operational heuristic to demonstrate Operations Research (0-1 Knapsack) resource allocation under constrained specialist capacity."*
   - Never claim this is a medically prescribed therapy dosage.

4. **Probability Representation:**
   - Say: *"Predicted ASD probability"* (or *"Tree ensemble voting probability"* from `predict_proba()`).
   - Do NOT say: *"Calibrated ASD probability"* unless isotonic regression / Platt scaling (`CalibratedClassifierCV`) was explicitly fitted.

---

### Key Viva Questions & Answers:

1. **Why Random Forest over Deep Learning?**
   - Tabular clinical screening data with categorical/binary survey items achieves high ROC-AUC (0.9894) with full explainability (Gini importance), sub-millisecond inference, and reproducibility without GPU overhead.

2. **Why is Recall prioritized over Accuracy?**
   - In clinical screening triage, false negatives (missing an individual with significant traits who needs support) carry severe consequences compared to false positives (who are filtered out during secondary clinical review).

3. **Why PuLP 0-1 Knapsack for Resource Allocation?**
   - Heuristic allocation (e.g. first-come-first-served) fails to maximize aggregate screening priority and underutilizes clinical specialist hours under strict weekly capacity constraints.

