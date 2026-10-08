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

## 5. Viva / Defense Quick Reference

1. **Why Random Forest over Deep Learning?**
   - Tabular clinical screening data with categorical/binary survey items achieves state-of-the-art ROC-AUC (0.9894) with full explainability, fast inference, and zero risk of GPU out-of-memory errors on standard hardware.
2. **Why is Recall prioritized over Accuracy?**
   - In clinical screening, false negatives (missing an individual who requires early intervention) carry much higher downstream cost than false positives (which undergo secondary clinical review).
3. **Why PuLP 0-1 Knapsack for Resource Allocation?**
   - Heuristic allocation (e.g. first-come-first-served) leaves specialist hours unutilized and fails to maximize aggregate screening priority under strict weekly capacity constraints.
