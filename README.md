# Student Learning Support & Resource Optimization

An end-to-end prototype combining **Machine Learning (Random Forest Regression)** and **Operations Research (PuLP Integer Linear Programming)** with an interactive **Streamlit Dashboard** for student outcome prediction and resource allocation.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Architecture

```text
├── app.py                     # Interactive multi-dataset Streamlit dashboard
├── src/
│   ├── train_model.py         # Phase 2: ML training and evaluation pipeline
│   └── optimize_resources.py  # Phase 3: OR resource allocation solver
├── models/
│   └── student_performance_rf.joblib  # Saved pre-trained Random Forest pipeline
├── outputs/
│   ├── predictions.csv        # Phase 2 baseline predictions
│   ├── feature_importance.csv # Phase 2 feature importances
│   ├── optimized_allocation.csv # Phase 3 baseline 300h allocation
│   └── capacity_analysis.csv  # Phase 3 sensitivity analysis across capacities
├── data/
│   ├── test_cohort_25.csv     # Sample 25-student test dataset (without G3)
│   └── test_cohort_50.csv     # Sample 50-student test dataset (without G3)
├── PHASE_1_REPORT.md          # Dataset exploration report
├── PHASE_2_REPORT.md          # ML model performance report
├── PHASE_3_REPORT.md          # OR resource allocation report
├── PHASE_4_REPORT.md          # Dashboard implementation report
├── PHASE_6_REPORT.md          # Dynamic multi-dataset integration report
└── requirements.txt           # Project dependencies
```

---

## 🔄 Using Your Own Dataset

You can upload and evaluate any new student cohort without retraining the model:

1. Launch the dashboard: `streamlit run app.py`.
2. In the left sidebar under **📁 Data Source**, select **Upload CSV**.
3. Upload your `.csv` student file (you can try `data/test_cohort_25.csv` or `data/test_cohort_50.csv`).
4. The system validates the schema and runs the pre-trained Random Forest model in real time.
5. Navigate to **Resource Allocation**, adjust the **Available Support Hours** slider, and observe the dynamic PuLP optimization re-allocating resources.
6. Export the predictions and optimal allocation using the sidebar download buttons.

### Required Feature Schema (32 Features):
- **Academic:** `G1`, `G2`, `failures`, `studytime`, `paid`
- **Attendance & Engagement:** `absences`, `freetime`, `goout`, `activities`
- **Background:** `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `guardian`
- **Support & Well-being:** `schoolsup`, `famsup`, `higher`, `internet`, `romantic`, `famrel`, `Dalc`, `Walc`, `health`
- *Note:* The target `G3` is **not required** in uploaded datasets.

---

## ⚠️ Academic Scope & Ethical Disclaimer
This prototype utilizes the UCI Student Performance dataset and is **NOT an Autism Spectrum Disorder (ASD) or clinical diagnosis tool**. It demonstrates the algorithmic integration of predictive ML with Operations Research optimization for educational resource management.
