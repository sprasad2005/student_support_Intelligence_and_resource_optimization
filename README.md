# NeuroRisk: Autism Spectrum Disorder Risk Screening & Support Resource Allocation

An end-to-end Machine Learning and Operations Research prototype for Autism Spectrum Disorder (ASD) screening risk estimation and optimal specialist support resource allocation.

---

## ◈ System Architecture

```text
ASD Screening Data (AQ-10 + Demographics)
  │
  ▼
Scikit-Learn Preprocessing (ColumnTransformer + OneHotEncoder)
  │
  ▼
Random Forest Classifier (200 Trees, Balanced Class Weights)
  │
  ▼
Predicted ASD Risk Probability & Priority Tiers (High / Medium / Low)
  │
  ▼
Operations Research Optimization (PuLP 0-1 Integer Linear Program)
  │
  ▼
Interactive Dark SaaS Streamlit Dashboard
```

---

## ◈ Key Features

1. **Machine Learning Classifier:**
   - Pre-trained on the official **UCI Autism Screening Adult Dataset** ($N = 704$).
   - Features 10 Autism Spectrum Quotient (AQ-10) screening items + contextual demographic/clinical variables.
   - Classification Performance: **94.33% Accuracy**, **91.67% Precision**, **86.84% Recall / Sensitivity**, **0.9894 ROC-AUC**.

2. **Screening Risk Tier Prioritization:**
   - **High Risk ($P \ge 0.70$):** 5 hours / week specialist support demand.
   - **Medium Risk ($0.30 \le P < 0.70$):** 2 hours / week support demand.
   - **Low Risk ($P < 0.30$):** 0 hours / week (routine standard monitoring).
   - Priority Score: $P \times 100$.

3. **Operations Research Resource Optimization:**
   - Formulates and solves a **0-1 Binary Knapsack Problem** via `pulp` to maximize total screening priority addressed under constrained specialist capacity.

4. **Dynamic Streamlit Web Dashboard:**
   - **Fresh Startup:** Starts with a clean empty state (no preloaded stale data).
   - **Dynamic Upload:** Ingest custom ASD screening CSV cohorts on the fly.
   - **Case Explorer:** Inspect individual AQ-10 item responses and family history.
   - **Model Governance:** Real-time confusion matrix, feature attributions, and clinical disclaimers.
   - **Persistent History:** Automatically logs processed analyses with one-click restore and deletion.

---

## ◈ Installation & Quickstart

### Prerequisites
- Python 3.9+

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/sprasad2005/student_support_Intelligence_and_resource_optimization.git
cd student_support_Intelligence_and_resource_optimization
pip install -r requirements.txt
```

### 2. Train the Model (Optional / Pre-trained Included)
```bash
python src/train_model.py
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## ◈ Test Sample Datasets

Sample cohorts are included in `data/` for immediate testing:
- `data/asd_test_sample_25.csv`: Sample screening cohort of 25 individuals.
- `data/asd_test_high_risk_30.csv`: High-priority screening cohort of 30 individuals.
- `data/asd_test_balanced_50.csv`: Balanced cohort of 50 individuals.
- `data/test_invalid_missing_cols.csv`: Schema validation test file with missing attributes.

---

## ◈ Ethical & Clinical Disclaimer

This software is an academic research prototype developed for screening risk prioritization and resource optimization. **It is not a medical or diagnostic device and does not diagnose autism spectrum disorder.** All outputs must be evaluated by licensed medical and psychiatric healthcare professionals.
