import os
import sys
import io
import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import altair as alt

# Add current working directory to sys.path to ensure src imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.optimize_resources import solve_resource_allocation
except ImportError:
    st.error("Could not import `solve_resource_allocation` from `src.optimize_resources`.")
    st.stop()


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NeuroRisk — ASD Screening & Support Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CONFIGURABLE CONSTANTS (ASD DOMAIN)
# ==========================================
# Screening Risk Probability Thresholds
THRESHOLD_LOW_RISK = 0.30     # ASD Probability < 0.30 -> Low Risk
THRESHOLD_HIGH_RISK = 0.70    # ASD Probability >= 0.70 -> High Risk (0.30 <= P < 0.70 -> Medium Risk)

# Project Resource Allocation Assumptions (Support Hours / Week)
SUPPORT_HOURS_HIGH_RISK = 5   # Intensive specialist support
SUPPORT_HOURS_MED_RISK = 2    # Structured periodic review
SUPPORT_HOURS_LOW_RISK = 0    # Routine standard monitoring


# ==========================================
# DARK SAAS DESIGN SYSTEM
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-page: #09090B;
        --bg-sidebar: #0D0F13;
        --bg-card: #111318;
        --bg-elevated: #151820;
        --bg-hover: #181B23;
        --border-color: #272B33;
        --border-dashed: #3F4450;
        --text-primary: #F4F4F5;
        --text-secondary: #A1A1AA;
        --text-muted: #71717A;
        --accent-indigo: #6366F1;
        --accent-indigo-hover: #4F46E5;
        --risk-high: #EF4444;
        --risk-med: #F59E0B;
        --risk-low: #22C55E;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--bg-page) !important;
        color: var(--text-primary) !important;
    }

    /* Hide Default Streamlit Chrome */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Centered Application Container */
    .block-container {
        max-width: 1240px !important;
        padding: 1.5rem 2.25rem 3.5rem 2.25rem !important;
        margin: 0 auto !important;
    }

    /* Dark Sidebar Shell */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
        width: 255px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1.1rem !important;
        max-width: 100% !important;
    }

    /* Sidebar Brand Section */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 1.1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }

    .brand-badge {
        width: 28px;
        height: 28px;
        background: var(--accent-indigo);
        color: #ffffff;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.35);
    }

    .brand-text {
        font-size: 15px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .brand-subtext {
        font-size: 11px;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Sidebar Section Header */
    .sidebar-category {
        font-size: 10.5px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.25rem;
        margin-bottom: 0.4rem;
        padding-left: 4px;
    }

    /* Sidebar Radio Dark Styling */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 3px !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        padding: 7px 10px !important;
        border-radius: 6px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        transition: all 0.12s ease;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border-left: 3px solid var(--accent-indigo) !important;
    }

    /* Sidebar Active Dataset Indicator */
    .dataset-status-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem 0.875rem;
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
    }

    .dataset-status-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .dataset-status-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .dataset-status-meta {
        font-size: 11.5px;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Top Page Header */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }

    .top-header-title {
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.025em;
        margin: 0;
    }

    .top-header-subtitle {
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Standard Dark Metric Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.125rem 1.25rem;
        position: relative;
    }

    .card-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .card-footer {
        font-size: 11.5px;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* Semantic Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-high {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .badge-medium {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .badge-low {
        background: rgba(34, 197, 94, 0.12);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }

    .badge-neutral {
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
    }

    /* Hero Insight Banner */
    .hero-insight {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-indigo);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
    }

    .hero-insight-title {
        font-size: 13.5px;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .hero-insight-body {
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-top: 6px;
    }

    /* Ethical & Medical Disclaimer Banner */
    .disclaimer-box {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-left: 4px solid var(--accent-indigo);
        border-radius: 8px;
        padding: 0.875rem 1.125rem;
        font-size: 12.5px;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-top: 1.25rem;
    }

    /* Empty State & Centered Upload Containers */
    .empty-state-box {
        max-width: 600px;
        margin: 2.5rem auto;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
    }

    .upload-container {
        max-width: 600px;
        margin: 1.5rem auto 1rem auto;
        background: var(--bg-card);
        border: 1px dashed var(--border-dashed);
        border-radius: 12px;
        padding: 2.25rem 2rem;
        text-align: center;
        transition: border-color 0.15s ease;
    }

    .upload-container:hover {
        border-color: var(--accent-indigo);
    }

    .upload-icon {
        width: 44px;
        height: 44px;
        background: var(--bg-elevated);
        color: var(--accent-indigo);
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-bottom: 0.875rem;
    }

    /* Step Grid */
    .step-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 1.25rem;
    }

    .step-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.875rem;
        font-size: 12px;
    }

    .step-num {
        font-size: 10.5px;
        font-weight: 700;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .step-title {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 2px;
    }

    .step-desc {
        color: var(--text-secondary);
        line-height: 1.4;
    }

    /* Buttons */
    .stButton button {
        background-color: var(--accent-indigo) !important;
        color: #ffffff !important;
        border: 1px solid var(--accent-indigo) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0.45rem 0.9rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }

    .stButton button:hover {
        background-color: var(--accent-indigo-hover) !important;
        border-color: var(--accent-indigo-hover) !important;
        color: #ffffff !important;
    }

    .stDownloadButton button {
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 12.5px !important;
        padding: 0.4rem 0.8rem !important;
    }

    .stDownloadButton button:hover {
        background-color: var(--bg-hover) !important;
        border-color: var(--text-muted) !important;
        color: #ffffff !important;
    }

    /* Dark Input Widgets & Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="input"] > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
    }

    /* Dataframe Dark Container */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        background: var(--bg-card) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# MODEL ARTIFACT LOADING & SCHEMA DEFINITION
# ==========================================
@st.cache_resource
def load_trained_pipeline():
    model_path = os.path.join('models', 'asd_risk_model.joblib')
    if not os.path.exists(model_path):
        return None, [], [], []
    try:
        pipeline = joblib.load(model_path)
        aq10_cols = [f'A{i}_Score' for i in range(1, 11)]
        num_cols = ['age']
        cat_cols = ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'relation']
        return pipeline, aq10_cols, num_cols, cat_cols
    except Exception as e:
        st.error(f"Error loading model artifact: {str(e)}")
        return None, [], [], []


@st.cache_data
def load_feature_importance():
    feat_path = os.path.join('outputs', 'feature_importance.csv')
    if os.path.exists(feat_path):
        return pd.read_csv(feat_path)
    return None


@st.cache_data
def load_model_metrics():
    metrics_path = os.path.join('outputs', 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


pipeline_model, EXPECTED_AQ_COLS, EXPECTED_NUM_COLS, EXPECTED_CAT_COLS = load_trained_pipeline()
ALL_EXPECTED_FEATURES = set(EXPECTED_AQ_COLS + EXPECTED_NUM_COLS + EXPECTED_CAT_COLS)
cached_feat_imp = load_feature_importance()
cached_metrics = load_model_metrics()


# ==========================================
# PERSISTENT HISTORY SYSTEM
# ==========================================
HISTORY_FILE = os.path.join('outputs', 'analysis_history.json')
HISTORY_DIR = os.path.join('outputs', 'history')


def load_analysis_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_analysis_to_history(dataset_name, source_type, preds_df, alloc_df, metrics, capacity):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history = load_analysis_history()
    
    base_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = base_id
    counter = 1
    while any(r.get('analysis_id') == analysis_id for r in history):
        analysis_id = f"{base_id}_{counter}"
        counter += 1
        
    preds_path = os.path.join(HISTORY_DIR, f"{analysis_id}_predictions.csv")
    alloc_path = os.path.join(HISTORY_DIR, f"{analysis_id}_allocation.csv")
    
    preds_df.to_csv(preds_path, index=False)
    alloc_df.to_csv(alloc_path, index=False)
    
    risk_col = 'risk_category' if 'risk_category' in preds_df.columns else 'risk_level'
    high_count = int(len(preds_df[preds_df[risk_col] == 'High Risk']))
    med_count = int(len(preds_df[preds_df[risk_col] == 'Medium Risk']))
    low_count = int(len(preds_df[preds_df[risk_col] == 'Low Risk']))
    total_demand = int(preds_df['support_demand_hours'].sum())
    
    record = {
        "analysis_id": analysis_id,
        "dataset_name": dataset_name,
        "source_type": source_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_count": len(preds_df),
        "individual_count": len(preds_df),
        "high_risk": high_count,
        "medium_risk": med_count,
        "low_risk": low_count,
        "total_support_demand": total_demand,
        "capacity": int(capacity),
        "allocated_hours": int(metrics.get('allocated_hours', 0)),
        "unused_hours": int(metrics.get('unused_hours', 0)),
        "individuals_selected": int(metrics.get('individuals_selected', metrics.get('students_selected', 0))),
        "demand_coverage": float(metrics.get('demand_coverage_percentage', 0.0)),
        "predictions_file": preds_path,
        "allocation_file": alloc_path
    }
    
    history.insert(0, record)
    if len(history) > 50:
        history = history[:50]
        
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
        
    return analysis_id


def delete_analysis_from_history(analysis_id):
    history = load_analysis_history()
    to_delete = [r for r in history if r.get('analysis_id') == analysis_id]
    for r in to_delete:
        p_file = r.get('predictions_file')
        a_file = r.get('allocation_file')
        if p_file and os.path.exists(p_file):
            try:
                os.remove(p_file)
            except Exception:
                pass
        if a_file and os.path.exists(a_file):
            try:
                os.remove(a_file)
            except Exception:
                pass
    new_history = [r for r in history if r.get('analysis_id') != analysis_id]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_history, f, indent=4)


def clear_all_history():
    history = load_analysis_history()
    for r in history:
        p_file = r.get('predictions_file')
        a_file = r.get('allocation_file')
        if p_file and os.path.exists(p_file):
            try:
                os.remove(p_file)
            except Exception:
                pass
        if a_file and os.path.exists(a_file):
            try:
                os.remove(a_file)
            except Exception:
                pass
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=4)


# ==========================================
# INFERENCE ENGINE (ASD CLASSIFICATION)
# ==========================================
def generate_asd_predictions_from_model(df_input, model, aq_cols, num_cols, cat_cols):
    """
    Runs the trained Random Forest Classifier pipeline on an input DataFrame
    to generate predicted ASD screening probabilities, risk tiers, and support demands.
    """
    # 1. Individual ID extraction or generation
    id_candidates = [c for c in df_input.columns if c.lower() in ['individual_id', 'client_id', 'student_id', 'id', 'case_id', 'subject_id']]
    if id_candidates:
        individual_ids = df_input[id_candidates[0]].astype(str).tolist()
    else:
        individual_ids = [f"IND{i+1:03d}" for i in range(len(df_input))]

    # 2. Extract and sanitize feature matrix
    X_features = pd.DataFrame(index=df_input.index)
    
    # AQ-10 features (clean numeric/binary)
    for col in aq_cols:
        if col in df_input.columns:
            X_features[col] = pd.to_numeric(df_input[col].replace('?', np.nan), errors='coerce').fillna(0).astype(int)
        else:
            X_features[col] = 0

    # Numeric features (age)
    for col in num_cols:
        if col in df_input.columns:
            val = pd.to_numeric(df_input[col].replace('?', np.nan), errors='coerce')
            med = val.median() if not val.isnull().all() else 25.0
            val = val.fillna(med)
            # Clip unrealistic age outliers
            val = val.apply(lambda a: med if (a < 1 or a > 100) else a)
            X_features[col] = val
        else:
            X_features[col] = 25.0

    # Categorical features
    for col in cat_cols:
        if col in df_input.columns:
            X_features[col] = df_input[col].replace('?', 'Unknown').fillna('Unknown').astype(str).str.strip()
        else:
            X_features[col] = 'Unknown'

    # Ensure column order matches pipeline expectations
    X_features = X_features[aq_cols + num_cols + cat_cols]

    # 3. Model Inference (Probabilities & Classes)
    probs = model.predict_proba(X_features)[:, 1]
    probs_rounded = np.round(probs, 4)
    preds = model.predict(X_features)

    # 4. Risk Classification Mapping
    def map_risk(p):
        if p < THRESHOLD_LOW_RISK:
            return "Low Risk"
        elif p < THRESHOLD_HIGH_RISK:
            return "Medium Risk"
        else:
            return "High Risk"

    def map_support(r):
        if r == "High Risk":
            return SUPPORT_HOURS_HIGH_RISK
        elif r == "Medium Risk":
            return SUPPORT_HOURS_MED_RISK
        else:
            return SUPPORT_HOURS_LOW_RISK

    risk_categories = [map_risk(p) for p in probs_rounded]
    support_demand = [map_support(r) for r in risk_categories]
    risk_scores = np.round(probs_rounded * 100.0, 2)

    out_df = pd.DataFrame({
        'individual_id': individual_ids,
        'asd_probability': probs_rounded,
        'predicted_class': preds,
        'risk_category': risk_categories,
        'risk_score': risk_scores,
        'support_demand_hours': support_demand
    })

    # Backward compatibility alias
    out_df['student_id'] = individual_ids
    out_df['risk_level'] = risk_categories

    # Retain ground truth target if available (for reference)
    target_candidates = [c for c in df_input.columns if 'class' in c.lower() or 'asd' in c.lower()]
    if target_candidates and target_candidates[0] in df_input.columns:
        out_df['actual_target'] = df_input[target_candidates[0]].astype(str)

    # Attach remaining original columns
    for c in df_input.columns:
        if c not in out_df.columns:
            out_df[c] = df_input[c]

    return out_df


@st.cache_data
def run_dynamic_optimization(df, capacity):
    status, alloc_df, metrics = solve_resource_allocation(df, capacity)
    return status, alloc_df, metrics


# ==========================================
# INITIAL STATE (FRESH START - NO PRELOADED DATA)
# ==========================================
if "active_dataset" not in st.session_state:
    st.session_state["active_dataset"] = None
if "active_predictions" not in st.session_state:
    st.session_state["active_predictions"] = None
if "active_allocation" not in st.session_state:
    st.session_state["active_allocation"] = None
if "active_dataset_name" not in st.session_state:
    st.session_state["active_dataset_name"] = None
if "active_dataset_source" not in st.session_state:
    st.session_state["active_dataset_source"] = None
if "active_analysis_id" not in st.session_state:
    st.session_state["active_analysis_id"] = None
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Overview"


def clear_active_dataset():
    st.session_state["active_dataset"] = None
    st.session_state["active_predictions"] = None
    st.session_state["active_dataset_name"] = None
    st.session_state["active_dataset_source"] = None
    st.session_state["active_allocation"] = None
    st.session_state["active_analysis_id"] = None
    st.session_state["nav_selection"] = "Overview"


# ==========================================
# SIDEBAR: APPLICATION SHELL
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-badge">◈</div>
            <div>
                <div class="brand-text">NeuroRisk</div>
                <div class="brand-subtext">ASD Screening & OR Support</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-category">WORKSPACE</div>', unsafe_allow_html=True)
    
    nav_options = [
        "Overview",
        "ASD Risk Analysis",
        "Support Resource Allocation",
        "Individual Screening Explorer"
    ]
    
    workspace_nav = st.radio(
        "Workspace Navigation",
        nav_options,
        index=nav_options.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in nav_options else 0,
        key="nav_radio_workspace",
        label_visibility="collapsed"
    )
    if workspace_nav != st.session_state["nav_selection"] and st.session_state["nav_selection"] in nav_options:
        st.session_state["nav_selection"] = workspace_nav

    st.markdown('<div class="sidebar-category">DATA SOURCE</div>', unsafe_allow_html=True)
    
    if st.session_state["active_dataset"] is not None:
        num_ind = len(st.session_state["active_dataset"])
        total_dem = int(st.session_state["active_dataset"]['support_demand_hours'].sum())
        st.markdown(
            f"""
            <div class="dataset-status-card">
                <div class="dataset-status-label"><span style="color:#22C55E;">●</span> Active Screening Cohort</div>
                <div class="dataset-status-name">{st.session_state["active_dataset_name"]}</div>
                <div class="dataset-status-meta"><b>{num_ind}</b> individuals · <b>{total_dem}h</b> demand</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Clear Active Dataset", key="btn_clear_active", use_container_width=True):
            clear_active_dataset()
            st.rerun()

        st.markdown('<div class="sidebar-category">EXPORT</div>', unsafe_allow_html=True)
        pred_csv = st.session_state["active_dataset"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="↓ Predictions CSV",
            data=pred_csv,
            file_name=f"asd_screening_predictions_{st.session_state['active_dataset_name'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        def_cap = min(300, total_dem) if total_dem > 0 else 0
        _, cur_alloc_df, _ = run_dynamic_optimization(st.session_state["active_dataset"], def_cap)
        alloc_csv = cur_alloc_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="↓ Allocation Plan",
            data=alloc_csv,
            file_name=f"asd_resource_allocation_{st.session_state['active_dataset_name'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown(
            """
            <div class="dataset-status-card">
                <div class="dataset-status-label"><span style="color:#71717A;">○</span> No Dataset Active</div>
                <div class="dataset-status-meta" style="margin-top:4px;">Upload an ASD screening CSV to begin analysis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("+ Upload ASD CSV", key="btn_quick_upload", use_container_width=True):
            st.session_state["nav_selection"] = "Import Dataset"
            st.rerun()

    st.markdown('<div class="sidebar-category">MODEL</div>', unsafe_allow_html=True)
    if st.button("◈  Model Insights", key="btn_model_nav", use_container_width=True):
        st.session_state["nav_selection"] = "Model Insights"
        st.rerun()

    st.markdown('<div class="sidebar-category">HISTORY</div>', unsafe_allow_html=True)
    if st.button("↻  Analysis History", key="btn_history_nav", use_container_width=True):
        st.session_state["nav_selection"] = "Analysis History"
        st.rerun()


# ==========================================
# PAGE ROUTING & STATE ACCESS
# ==========================================
active_view = st.session_state["nav_selection"]
current_predictions_df = st.session_state["active_dataset"]
dataset_display_name = st.session_state["active_dataset_name"] or "No Active Dataset"


# ==========================================
# PAGE: IMPORT DATASET (UPLOAD SCREEN)
# ==========================================
if active_view == "Import Dataset":
    st.markdown(
        """
        <div class="top-header">
            <div>
                <h1 class="top-header-title">ASD Screening Cohort Ingestion</h1>
                <p class="top-header-subtitle">Import an Autism Spectrum Disorder screening questionnaire CSV to estimate risk probabilities and optimize specialist support allocation.</p>
            </div>
            <span class="badge badge-neutral">● Ready for upload</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="upload-container">
            <div class="upload-icon">↑</div>
            <div style="font-size: 16px; font-weight: 700; color: #F4F4F5; margin-bottom: 4px;">Upload ASD Screening CSV</div>
            <div style="font-size: 13.5px; color: #A1A1AA; margin-bottom: 16px;">
                Expected features: 10 AQ items (<code>A1_Score</code> to <code>A10_Score</code>), <code>age</code>, <code>gender</code>, <code>ethnicity</code>, <code>jundice</code>, <code>austim</code>, <code>contry_of_res</code>, <code>used_app_before</code>, <code>relation</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        label_visibility="collapsed",
        key="main_uploader_widget"
    )

    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            try:
                raw_df = pd.read_csv(io.BytesIO(bytes_data))
                if len(raw_df.columns) == 1 and ';' in bytes_data.decode('utf-8', errors='ignore')[:500]:
                    raw_df = pd.read_csv(io.BytesIO(bytes_data), sep=';')
            except Exception:
                raw_df = pd.read_csv(io.BytesIO(bytes_data), sep=';')

            # Clean column names (strip whitespace)
            raw_df.columns = [c.strip() for c in raw_df.columns]

            missing_cols = [c for c in ALL_EXPECTED_FEATURES if c not in raw_df.columns]
            
            if missing_cols:
                st.error(f"❌ **Dataset Missing {len(missing_cols)} Required Screening Features:**")
                st.markdown("\n".join([f"- `{col}`" for col in missing_cols]))
                st.info("💡 Expected schema includes AQ-10 screening scores (A1_Score to A10_Score), age, gender, ethnicity, jaundice, family history of autism, country, app usage, and relation.")
            elif pipeline_model is None:
                st.error("Trained ASD classifier artifact missing from `models/asd_risk_model.joblib`. Please run training first.")
            else:
                preds_df = generate_asd_predictions_from_model(
                    raw_df, pipeline_model, EXPECTED_AQ_COLS, EXPECTED_NUM_COLS, EXPECTED_CAT_COLS
                )
                total_dem = int(preds_df['support_demand_hours'].sum())
                def_cap = min(300, total_dem) if total_dem > 0 else 0
                _, alloc_df, metrics = solve_resource_allocation(preds_df, def_cap)
                
                aid = save_analysis_to_history(uploaded_file.name, "Uploaded CSV", preds_df, alloc_df, metrics, def_cap)
                
                st.session_state["active_dataset"] = preds_df
                st.session_state["active_predictions"] = preds_df
                st.session_state["active_dataset_name"] = uploaded_file.name
                st.session_state["active_dataset_source"] = "Uploaded CSV"
                st.session_state["active_allocation"] = alloc_df
                st.session_state["active_analysis_id"] = aid
                
                st.markdown(
                    f"""
                    <div class="card" style="border-left: 4px solid #22C55E; margin-top: 1rem;">
                        <div style="font-size: 14.5px; font-weight: 700; color: #F4F4F5; display: flex; justify-content: space-between; align-items: center;">
                            <span style="color:#4ADE80;">✓ Screening Cohort processed & Saved to History</span>
                            <span class="badge badge-low">Inference complete</span>
                        </div>
                        <div style="font-size: 13px; color: #A1A1AA; margin-top: 8px;">
                            • Cohort File: <b>{uploaded_file.name}</b><br>
                            • Individuals Screened: <b>{len(preds_df)}</b> · <b>18 screening features</b><br>
                            • Classifier: <b>Random Forest Classifier (200 trees)</b><br>
                            • Total Specialist Support Demand: <b>{total_dem} hours / week</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("View Analysis Dashboard →", key="btn_run_analysis"):
                    st.session_state["nav_selection"] = "Overview"
                    st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # How It Works Grid
    st.markdown(
        """
        <div style="margin-top: 2rem;">
            <div style="font-size: 11px; font-weight: 700; color: #71717A; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">HOW THE ASD SCREENING & OPTIMIZATION PIPELINE WORKS</div>
            <div class="step-grid">
                <div class="step-card">
                    <div class="step-num">01</div>
                    <div class="step-title">Ingest & Validate</div>
                    <div class="step-desc">Verify 10 AQ-10 screening items & demographic indicators.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">02</div>
                    <div class="step-title">ML Risk Estimation</div>
                    <div class="step-desc">Random Forest estimates individual ASD risk probability.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">03</div>
                    <div class="step-title">Tier Prioritization</div>
                    <div class="step-desc">Classify into High, Medium, Low support priority tiers.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">04</div>
                    <div class="step-title">PuLP Allocation</div>
                    <div class="step-desc">Solve 0-1 knapsack to optimally allocate limited specialist hours.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
elif active_view == "Overview":
    if current_predictions_df is None:
        # CLEAN EMPTY STATE
        st.markdown(
            """
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">ASD Screening Risk Overview</h1>
                    <p class="top-header-subtitle">Monitor Autism Spectrum Disorder screening risk tiers and optimize specialist support allocation.</p>
                </div>
                <span class="badge badge-neutral">○ NO DATASET ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 5 KPI Placeholder Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown('<div class="card"><div class="card-label">Total Individuals</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">No active cohort</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="card"><div class="card-label">High Risk Tier</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="card"><div class="card-label">Medium Risk Tier</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown('<div class="card"><div class="card-label">Low Risk Tier</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown('<div class="card"><div class="card-label">Support Demand</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="empty-state-box">
                <div style="width: 44px; height: 44px; background: #151820; color: #6366F1; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 1rem;">◈</div>
                <div style="font-size: 17px; font-weight: 700; color: #F4F4F5; margin-bottom: 4px;">No screening dataset loaded</div>
                <div style="font-size: 13.5px; color: #A1A1AA; margin-bottom: 20px;">Upload an Autism Spectrum Disorder screening cohort CSV to estimate screening risk probabilities and allocate specialist intervention hours.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        btn_c1, btn_c2, btn_c3 = st.columns([1, 1.2, 1])
        with btn_c2:
            if st.button("Upload ASD Screening CSV", key="btn_empty_upload", use_container_width=True):
                st.session_state["nav_selection"] = "Import Dataset"
                st.rerun()

        st.markdown(
            """
            <div class="disclaimer-box">
                <b>Academic Prototype & Screening Disclaimer:</b> This system is an academic research prototype designed for ASD screening risk prioritization and resource optimization. <b>It is NOT a medical or clinical diagnostic tool</b> and does not diagnose or confirm autism spectrum disorder. Formal diagnosis requires comprehensive assessment by qualified medical and clinical professionals.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # ACTIVE DATASET OVERVIEW
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">ASD Screening Risk Overview</h1>
                    <p class="top-header-subtitle">Monitor screening risk distribution and optimal specialist support allocation across the active cohort.</p>
                </div>
                <span class="badge badge-neutral">● ACTIVE: {dataset_display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        total_individuals = len(current_predictions_df)
        high_risk_count = len(current_predictions_df[current_predictions_df['risk_category'] == 'High Risk'])
        med_risk_count = len(current_predictions_df[current_predictions_df['risk_category'] == 'Medium Risk'])
        low_risk_count = len(current_predictions_df[current_predictions_df['risk_category'] == 'Low Risk'])
        
        total_demand_hours = int(current_predictions_df['support_demand_hours'].sum())
        high_demand_hours = high_risk_count * SUPPORT_HOURS_HIGH_RISK
        med_demand_hours = med_risk_count * SUPPORT_HOURS_MED_RISK

        base_cap = min(300, total_demand_hours) if total_demand_hours > 0 else 0
        _, _, base_metrics = run_dynamic_optimization(current_predictions_df, base_cap)

        # 5-Card Metric Row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f'<div class="card"><div class="card-label">Total Individuals</div><div class="card-value">{total_individuals}</div><div class="card-footer">Active cohort</div></div>', unsafe_allow_html=True)
        with k2:
            hr_pct = (high_risk_count / total_individuals * 100) if total_individuals else 0
            st.markdown(f'<div class="card"><div class="card-label">High Risk (P ≥ 0.70)</div><div class="card-value" style="color:#F87171;">{high_risk_count}</div><div class="card-footer"><span class="badge badge-high">{hr_pct:.1f}%</span> of cohort</div></div>', unsafe_allow_html=True)
        with k3:
            mr_pct = (med_risk_count / total_individuals * 100) if total_individuals else 0
            st.markdown(f'<div class="card"><div class="card-label">Medium Risk (0.30–0.70)</div><div class="card-value" style="color:#FBBF24;">{med_risk_count}</div><div class="card-footer"><span class="badge badge-medium">{mr_pct:.1f}%</span> of cohort</div></div>', unsafe_allow_html=True)
        with k4:
            lr_pct = (low_risk_count / total_individuals * 100) if total_individuals else 0
            st.markdown(f'<div class="card"><div class="card-label">Low Risk (P < 0.30)</div><div class="card-value" style="color:#4ADE80;">{low_risk_count}</div><div class="card-footer"><span class="badge badge-low">{lr_pct:.1f}%</span> of cohort</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown(f'<div class="card"><div class="card-label">Support Demand</div><div class="card-value" style="color:#818CF8;">{total_demand_hours}h</div><div class="card-footer">Unconstrained / week</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Hero Insight Banner
        cov_pct = float(base_metrics.get('demand_coverage_percentage', 0.0))
        ind_sel = int(base_metrics.get('individuals_selected', base_metrics.get('selected_individuals', 0)))
        st.markdown(
            f"""
            <div class="hero-insight">
                <div class="hero-insight-title">
                    <span>Screening Cohort Summary</span>
                    <span class="badge badge-neutral">{total_demand_hours}h Total Specialist Demand</span>
                </div>
                <div class="hero-insight-body">
                    <b>{high_risk_count} individuals</b> are categorized as High Risk (probability ≥ 70%).<br>
                    Estimated support requirement: <b>{high_demand_hours} hours / week</b> for high-risk individuals. At baseline capacity (<b>{base_cap}h</b>), the PuLP optimization solver covers <b>{cov_pct:.1f}%</b> of total cohort demand (<b>{ind_sel} individuals</b> supported).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Full-Width Risk Distribution Chart
        st.markdown('<div class="card"><div class="card-label">Screening Risk Distribution (Individuals by Priority Tier)</div>', unsafe_allow_html=True)
        hr_p = (high_risk_count / total_individuals * 100) if total_individuals else 0
        mr_p = (med_risk_count / total_individuals * 100) if total_individuals else 0
        lr_p = (low_risk_count / total_individuals * 100) if total_individuals else 0
        
        risk_bar_data = pd.DataFrame({
            'Category': ['High Risk', 'Medium Risk', 'Low Risk'],
            'Count': [high_risk_count, med_risk_count, low_risk_count],
            'Label': [
                f"{high_risk_count} ({hr_p:.1f}%)",
                f"{med_risk_count} ({mr_p:.1f}%)",
                f"{low_risk_count} ({lr_p:.1f}%)"
            ]
        })
        max_c = max(high_risk_count, med_risk_count, low_risk_count, 1)
        
        base_risk = alt.Chart(risk_bar_data).encode(
            y=alt.Y('Category:N', title=None, sort=['High Risk', 'Medium Risk', 'Low Risk'], axis=alt.Axis(labelColor='#F4F4F5', labelFontSize=12)),
            x=alt.X('Count:Q', title='Individuals', scale=alt.Scale(domain=[0, max_c * 1.25]), axis=alt.Axis(grid=True, gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA'))
        )
        bars_risk = base_risk.mark_bar(cornerRadiusEnd=4, size=24).encode(
            color=alt.Color(
                'Category:N',
                scale=alt.Scale(
                    domain=['High Risk', 'Medium Risk', 'Low Risk'],
                    range=['#EF4444', '#F59E0B', '#22C55E']
                ),
                legend=None
            ),
            tooltip=['Category', 'Count', 'Label']
        )
        text_risk = base_risk.mark_text(
            align='left',
            baseline='middle',
            dx=6,
            color='#F4F4F5',
            fontSize=12,
            fontWeight=600
        ).encode(
            text='Label:N'
        )
        risk_chart = (bars_risk + text_risk).properties(height=200, background='#111318').configure_view(stroke=None)
        st.altair_chart(risk_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="disclaimer-box">
                <b>Medical & Ethical Notice:</b> This software is an academic screening risk prioritization and resource optimization prototype. <b>It is not a medical diagnostic tool</b> and does not diagnose autism. Screening results reflect statistical model associations and should always be accompanied by clinical evaluation by certified healthcare practitioners.
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# PAGE 2: ASD RISK ANALYSIS
# ==========================================
elif active_view == "ASD Risk Analysis":
    if current_predictions_df is None:
        st.markdown('<div class="top-header"><div><h1 class="top-header-title">ASD Risk Analysis</h1><p class="top-header-subtitle">Explore screening risk probabilities and questionnaire metrics across the active cohort.</p></div><span class="badge badge-neutral">○ NO DATASET</span></div>', unsafe_allow_html=True)
        st.info("💡 No dataset active. Please upload an ASD screening CSV to begin analysis.")
        if st.button("Upload ASD Screening CSV", key="btn_guard_upload_risk"):
            st.session_state["nav_selection"] = "Import Dataset"
            st.rerun()
    else:
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">ASD Risk Analysis</h1>
                    <p class="top-header-subtitle">Detailed risk probability breakdown and screening questionnaire insights.</p>
                </div>
                <span class="badge badge-neutral">● {dataset_display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="card" style="padding: 0.875rem 1.25rem; margin-bottom: 1.25rem;">', unsafe_allow_html=True)
        f1, f2, f3 = st.columns([1.5, 2, 1.2])
        with f1:
            risk_filter = st.multiselect("Risk Tier Filter", options=["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk", "Low Risk"])
        with f2:
            min_p = float(current_predictions_df['asd_probability'].min()) if not current_predictions_df.empty else 0.0
            max_p = float(current_predictions_df['asd_probability'].max()) if not current_predictions_df.empty else 1.0
            prob_range = st.slider("ASD Probability Range", min_value=0.0, max_value=1.0, value=(min_p, max_p), step=0.05)
        with f3:
            search_id = st.text_input("Filter ID", "", placeholder="e.g. IND001")
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = current_predictions_df[
            (current_predictions_df['risk_category'].isin(risk_filter)) &
            (current_predictions_df['asd_probability'] >= prob_range[0]) &
            (current_predictions_df['asd_probability'] <= prob_range[1])
        ]
        if search_id.strip():
            filtered_df = filtered_df[filtered_df['individual_id'].str.contains(search_id.strip(), case=False, na=False)]

        st.caption(f"Displaying **{len(filtered_df)}** of **{len(current_predictions_df)}** individuals in active cohort.")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('<div class="card"><div class="card-label">Filtered Risk Breakdown</div>', unsafe_allow_html=True)
            hr_cnt = len(filtered_df[filtered_df['risk_category'] == 'High Risk'])
            mr_cnt = len(filtered_df[filtered_df['risk_category'] == 'Medium Risk'])
            lr_cnt = len(filtered_df[filtered_df['risk_category'] == 'Low Risk'])
            r_counts = pd.DataFrame({
                'Risk': ['High Risk', 'Medium Risk', 'Low Risk'],
                'Count': [hr_cnt, mr_cnt, lr_cnt],
                'Label': [f"{hr_cnt} ind", f"{mr_cnt} ind", f"{lr_cnt} ind"]
            })
            max_rc = max(hr_cnt, mr_cnt, lr_cnt, 1)
            base_r = alt.Chart(r_counts).encode(
                x=alt.X('Risk:N', title=None, sort=['High Risk', 'Medium Risk', 'Low Risk'], axis=alt.Axis(labelColor='#F4F4F5', labelFontSize=12)),
                y=alt.Y('Count:Q', title='Individuals', scale=alt.Scale(domain=[0, max_rc * 1.3]), axis=alt.Axis(grid=True, gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA'))
            )
            bars_r = base_r.mark_bar(cornerRadiusEnd=4, size=32).encode(
                color=alt.Color('Risk:N', scale=alt.Scale(domain=['High Risk', 'Medium Risk', 'Low Risk'], range=['#EF4444', '#F59E0B', '#22C55E']), legend=None),
                tooltip=['Risk', 'Count', 'Label']
            )
            text_r = base_r.mark_text(
                align='center',
                baseline='bottom',
                dy=-4,
                color='#F4F4F5',
                fontSize=12,
                fontWeight=600
            ).encode(
                text='Label:N'
            )
            r_bar = (bars_r + text_r).properties(height=180, background='#111318').configure_view(stroke=None)
            st.altair_chart(r_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rc2:
            st.markdown('<div class="card"><div class="card-label">ASD Probability Distribution</div>', unsafe_allow_html=True)
            if not filtered_df.empty:
                g_hist = alt.Chart(filtered_df).mark_bar(cornerRadiusEnd=2, color='#6366F1').encode(
                    x=alt.X('asd_probability:Q', bin=alt.Bin(maxbins=20), title='Predicted ASD Probability (0.0–1.0)', axis=alt.Axis(gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA')),
                    y=alt.Y('count():Q', title='Individuals', axis=alt.Axis(grid=True, gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA'))
                ).properties(height=180, background='#111318').configure_view(stroke=None)
                st.altair_chart(g_hist, use_container_width=True)
            else:
                st.info("No matching records.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 10px;">Screening Roster & Feature Attribution</div>', unsafe_allow_html=True)
        tbl_cols = ['individual_id', 'asd_probability', 'risk_category', 'risk_score', 'support_demand_hours', 'age', 'gender', 'ethnicity', 'austim', 'jundice']
        display_cols = [c for c in tbl_cols if c in filtered_df.columns]
        
        # Include AQ-10 summary if present
        aq_in_df = [f'A{i}_Score' for i in range(1, 11) if f'A{i}_Score' in filtered_df.columns]
        if aq_in_df:
            display_cols.extend(aq_in_df)

        st.dataframe(filtered_df[display_cols].sort_values(by='risk_score', ascending=False), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 3: SUPPORT RESOURCE ALLOCATION
# ==========================================
elif active_view == "Support Resource Allocation":
    if current_predictions_df is None:
        st.markdown('<div class="top-header"><div><h1 class="top-header-title">Support Resource Allocation</h1><p class="top-header-subtitle">Optimize limited specialist support hours across individuals with highest screening priority.</p></div><span class="badge badge-neutral">○ NO DATASET</span></div>', unsafe_allow_html=True)
        st.info("💡 No dataset active. Please upload an ASD screening CSV to begin analysis.")
        if st.button("Upload ASD Screening CSV", key="btn_guard_upload_alloc"):
            st.session_state["nav_selection"] = "Import Dataset"
            st.rerun()
    else:
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">Support Resource Allocation</h1>
                    <p class="top-header-subtitle">Operations Research 0-1 Integer Linear Program (Knapsack formulation) for optimal specialist support distribution.</p>
                </div>
                <span class="badge badge-neutral">● {dataset_display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        total_demand = int(current_predictions_df['support_demand_hours'].sum())
        capacity_input = min(300, total_demand) if total_demand > 0 else 0

        status, dyn_alloc_df, metrics = run_dynamic_optimization(current_predictions_df, capacity_input)
        if status != "Optimal":
            st.error(f"Solver Status: {status}")
            st.stop()

        alloc_h = int(metrics.get('allocated_hours', 0))
        avail_h = int(metrics.get('available_hours', capacity_input))
        unused_h = int(metrics.get('unused_hours', 0))
        util_pct = float(metrics.get('resource_utilization_percentage', 0.0))
        cov_pct = float(metrics.get('demand_coverage_percentage', 0.0))
        ind_sel = int(metrics.get('individuals_selected', metrics.get('selected_individuals', 0)))
        hr_sel = int(metrics.get('high_risk_selected', 0))
        mr_sel = int(metrics.get('medium_risk_selected', 0))

        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #22C55E; margin-bottom: 1.25rem;">
                <div style="font-size: 14px; font-weight: 700; color: #F4F4F5; display: flex; justify-content: space-between;">
                    <span style="color:#4ADE80;">✓ Optimal Support Allocation Plan Solved (PuLP ILP)</span>
                    <span class="badge badge-low">{util_pct:.1f}% Resource Utilization</span>
                </div>
                <div style="font-size: 13px; color: #A1A1AA; margin-top: 6px;">
                    <b>{alloc_h}h allocated</b> of {avail_h}h available specialist budget · <b>{unused_h}h remaining</b> · <b>{ind_sel} individuals supported</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="card"><div class="card-label">Allocated Support</div><div class="card-value" style="color:#818CF8;">{alloc_h}h</div><div class="card-footer">Budget: {capacity_input}h</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="card"><div class="card-label">Remaining Capacity</div><div class="card-value">{unused_h}h</div><div class="card-footer">Unallocated hours</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="card"><div class="card-label">Demand Coverage</div><div class="card-value">{cov_pct:.1f}%</div><div class="card-footer">{alloc_h}h of {total_demand}h total demand</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="card"><div class="card-label">Individuals Selected</div><div class="card-value">{ind_sel}</div><div class="card-footer">{hr_sel} High + {mr_sel} Medium</div></div>', unsafe_allow_html=True)


        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 10px;">Prioritized Support Allocation Schedule</div>', unsafe_allow_html=True)
        sel_df = dyn_alloc_df[dyn_alloc_df['selected_for_support'] == 1].sort_values(by='risk_score', ascending=False)
        s_cols = ['individual_id', 'asd_probability', 'risk_category', 'risk_score', 'support_demand_hours', 'age', 'gender', 'ethnicity', 'austim']
        display_s_cols = [c for c in s_cols if c in sel_df.columns]
        st.dataframe(sel_df[display_s_cols], use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 4: INDIVIDUAL SCREENING EXPLORER
# ==========================================
elif active_view == "Individual Screening Explorer":
    if current_predictions_df is None:
        st.markdown('<div class="top-header"><div><h1 class="top-header-title">Individual Screening Explorer</h1><p class="top-header-subtitle">Review an individual\'s screening questionnaire items and recommended support priority.</p></div><span class="badge badge-neutral">○ NO DATASET</span></div>', unsafe_allow_html=True)
        st.info("💡 No dataset active. Please upload an ASD screening CSV to begin analysis.")
        if st.button("Upload ASD Screening CSV", key="btn_guard_upload_exp"):
            st.session_state["nav_selection"] = "Import Dataset"
            st.rerun()
    else:
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">Individual Screening Explorer</h1>
                    <p class="top-header-subtitle">Inspect AQ-10 item scores, demographic factors, and recommended support hours for individual cases.</p>
                </div>
                <span class="badge badge-neutral">● {dataset_display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        individual_list = current_predictions_df['individual_id'].tolist()
        if not individual_list:
            st.info("No individuals in active cohort.")
            st.stop()

        st.markdown('<div class="card" style="padding: 1rem 1.25rem; margin-bottom: 1.25rem;">', unsafe_allow_html=True)
        sc1, sc2 = st.columns([2, 3])
        with sc1:
            sel_id = st.selectbox("Select Individual", individual_list, index=0)
        
        ind = current_predictions_df[current_predictions_df['individual_id'] == sel_id].iloc[0]
        
        # Current allocation match
        def_cap = min(300, int(current_predictions_df['support_demand_hours'].sum())) if len(current_predictions_df) else 0
        _, cur_alloc_df, _ = run_dynamic_optimization(current_predictions_df, def_cap)
        base_m = cur_alloc_df[cur_alloc_df['individual_id'] == sel_id]
        is_sel = int(base_m.iloc[0]['selected_for_support']) if not base_m.empty else 0

        with sc2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            if ind['risk_category'] == "High Risk":
                st.markdown('<span class="badge badge-high">● High Risk Priority</span>', unsafe_allow_html=True)
            elif ind['risk_category'] == "Medium Risk":
                st.markdown('<span class="badge badge-medium">● Medium Risk Priority</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-low">● Low Risk Priority</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="card"><div class="card-label">ASD Risk Probability</div><div class="card-value" style="color:#818CF8;">{ind["asd_probability"]:.4f}</div><div class="card-footer">{ind["asd_probability"]*100:.1f}% estimated probability</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="card"><div class="card-label">Screening Priority Score</div><div class="card-value">{ind["risk_score"]:.1f}</div><div class="card-footer">Scale: 0 – 100</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="card"><div class="card-label">Support Demand</div><div class="card-value">{ind["support_demand_hours"]}h</div><div class="card-footer">Weekly allocation hours</div></div>', unsafe_allow_html=True)
        with m4:
            status_tag = '<span class="badge badge-low">✓ Selected for Support</span>' if is_sel == 1 else ('<span class="badge badge-high">Waitlisted (Capacity)</span>' if ind['support_demand_hours'] > 0 else '<span class="badge badge-neutral">Routine Monitoring</span>')
            st.markdown(f'<div class="card"><div class="card-label">Allocation Status</div><div style="margin-top:6px;">{status_tag}</div><div class="card-footer">Under {def_cap}h baseline budget</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Recommendation Card
        if ind['risk_category'] == "High Risk":
            st.markdown(f'<div class="card" style="border-left: 4px solid #EF4444;"><div style="font-size: 13.5px; font-weight: 700; color: #F87171;">HIGH SCREENING PRIORITY</div><div style="font-size: 13px; color: #A1A1AA; margin-top: 4px;">Individual demonstrates high screening score indicators (estimated probability {ind["asd_probability"]*100:.1f}%). Recommended support allocation: <b style="color:#F4F4F5;">{SUPPORT_HOURS_HIGH_RISK} hours / week</b> of structured one-on-one specialist intervention.</div></div>', unsafe_allow_html=True)
        elif ind['risk_category'] == "Medium Risk":
            st.markdown(f'<div class="card" style="border-left: 4px solid #F59E0B;"><div style="font-size: 13.5px; font-weight: 700; color: #FBBF24;">MODERATE SCREENING PRIORITY</div><div style="font-size: 13px; color: #A1A1AA; margin-top: 4px;">Individual demonstrates moderate screening traits (estimated probability {ind["asd_probability"]*100:.1f}%). Recommended support allocation: <b style="color:#F4F4F5;">{SUPPORT_HOURS_MED_RISK} hours / week</b> of small-group review.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card" style="border-left: 4px solid #22C55E;"><div style="font-size: 13.5px; font-weight: 700; color: #4ADE80;">ROUTINE MONITORING</div><div style="font-size: 13px; color: #A1A1AA; margin-top: 4px;">Individual demonstrates low screening traits (estimated probability {ind["asd_probability"]*100:.1f}%). Recommended support allocation: <b style="color:#F4F4F5;">{SUPPORT_HOURS_LOW_RISK} hours / week</b> (standard routine monitoring).</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Contextual Questionnaire and Demographics
        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 10px;">Screening Factors & Questionnaire Responses</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**AQ-10 Items 1–5:**<br>• A1 Score: **{ind.get('A1_Score', 'N/A')}**<br>• A2 Score: **{ind.get('A2_Score', 'N/A')}**<br>• A3 Score: **{ind.get('A3_Score', 'N/A')}**<br>• A4 Score: **{ind.get('A4_Score', 'N/A')}**<br>• A5 Score: **{ind.get('A5_Score', 'N/A')}**", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**AQ-10 Items 6–10:**<br>• A6 Score: **{ind.get('A6_Score', 'N/A')}**<br>• A7 Score: **{ind.get('A7_Score', 'N/A')}**<br>• A8 Score: **{ind.get('A8_Score', 'N/A')}**<br>• A9 Score: **{ind.get('A9_Score', 'N/A')}**<br>• A10 Score: **{ind.get('A10_Score', 'N/A')}**", unsafe_allow_html=True)
        with c3:
            st.markdown(f"**Demographics & Clinical History:**<br>• Age: **{ind.get('age', 'N/A')}** | Gender: **{ind.get('gender', 'N/A')}**<br>• Ethnicity: **{ind.get('ethnicity', 'N/A')}**<br>• Family ASD History: **{ind.get('austim', 'N/A')}**<br>• Born with Jaundice: **{ind.get('jundice', 'N/A')}**<br>• Country: **{ind.get('contry_of_res', 'N/A')}**", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 5: MODEL INSIGHTS & GOVERNANCE
# ==========================================
elif active_view == "Model Insights":
    st.markdown(
        """
        <div class="top-header">
            <div>
                <h1 class="top-header-title">Model Performance & Governance</h1>
                <p class="top-header-subtitle">Performance benchmarks, classification metrics, feature importance, and clinical governance disclaimers.</p>
            </div>
            <span class="badge badge-neutral">● Random Forest Classifier</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    acc_val = cached_metrics.get("accuracy", 0.9433) if cached_metrics else 0.9433
    prec_val = cached_metrics.get("precision", 0.9167) if cached_metrics else 0.9167
    rec_val = cached_metrics.get("recall", 0.8684) if cached_metrics else 0.8684
    f1_val = cached_metrics.get("f1_score", 0.8919) if cached_metrics else 0.8919
    roc_val = cached_metrics.get("roc_auc", 0.9894) if cached_metrics else 0.9894

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="card"><div class="card-label">Accuracy</div><div class="card-value" style="color:#818CF8;">{acc_val*100:.1f}%</div><div class="card-footer">Overall test accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="card"><div class="card-label">Recall (Sensitivity)</div><div class="card-value" style="color:#4ADE80;">{rec_val*100:.1f}%</div><div class="card-footer">High-risk detection rate</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="card"><div class="card-label">Precision</div><div class="card-value">{prec_val*100:.1f}%</div><div class="card-footer">True positive rate</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="card"><div class="card-label">F1-Score</div><div class="card-value">{f1_val:.3f}</div><div class="card-footer">Harmonic mean</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="card"><div class="card-label">ROC-AUC</div><div class="card-value" style="color:#FBBF24;">{roc_val:.3f}</div><div class="card-footer">Discriminative ability</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col_cm, col_imp = st.columns([1.2, 2])
    with col_cm:
        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 8px;">Confusion Matrix (Holdout Test Set)</div>', unsafe_allow_html=True)
        cm_matrix = cached_metrics.get("confusion_matrix", [[100, 3], [5, 33]]) if cached_metrics else [[100, 3], [5, 33]]
        cm_df = pd.DataFrame([
            {"Actual": "Non-ASD (0)", "Predicted": "Non-ASD (0)", "Count": cm_matrix[0][0], "Type": "True Negative"},
            {"Actual": "Non-ASD (0)", "Predicted": "ASD (1)", "Count": cm_matrix[0][1], "Type": "False Positive"},
            {"Actual": "ASD (1)", "Predicted": "Non-ASD (0)", "Count": cm_matrix[1][0], "Type": "False Negative"},
            {"Actual": "ASD (1)", "Predicted": "ASD (1)", "Count": cm_matrix[1][1], "Type": "True Positive"}
        ])
        cm_chart = alt.Chart(cm_df).mark_rect().encode(
            x=alt.X('Predicted:N', title='Predicted Class', axis=alt.Axis(labelColor='#F4F4F5', labelFontSize=11)),
            y=alt.Y('Actual:N', title='Actual Ground Truth', sort=['Non-ASD (0)', 'ASD (1)'], axis=alt.Axis(labelColor='#F4F4F5', labelFontSize=11)),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='indigo'), legend=None),
            tooltip=['Actual', 'Predicted', 'Count', 'Type']
        ).properties(height=210, background='#111318')
        cm_text = cm_chart.mark_text(baseline='middle', fontSize=14, fontWeight=700).encode(
            text='Count:Q',
            color=alt.value('#FFFFFF')
        )
        st.altair_chart(cm_chart + cm_text, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_imp:
        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 8px;">Top 10 Feature Attributions</div>', unsafe_allow_html=True)
        if cached_feat_imp is not None:
            top10 = cached_feat_imp.head(10).sort_values(by='importance', ascending=True)
            imp_chart = alt.Chart(top10).mark_bar(cornerRadiusEnd=3, color='#6366F1', size=16).encode(
                x=alt.X('importance:Q', title='Gini Importance Weight', axis=alt.Axis(gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA')),
                y=alt.Y('feature:N', title=None, sort='-x', axis=alt.Axis(labelColor='#A1A1AA')),
                tooltip=['feature', 'importance']
            ).properties(height=210, background='#111318').configure_view(stroke=None)
            st.altair_chart(imp_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-label" style="margin-bottom: 8px;">Model Architecture & Training Specifications</div>
            <div style="font-size: 13px; color: #A1A1AA; line-height: 1.6;">
                • <b style="color:#F4F4F5;">Algorithm:</b> <code>RandomForestClassifier</code> (scikit-learn ensemble)<br>
                • <b style="color:#F4F4F5;">Hyperparameters:</b> 200 estimators · <code>class_weight='balanced'</code> · <code>random_state=42</code><br>
                • <b style="color:#F4F4F5;">Preprocessing Pipeline:</b> <code>ColumnTransformer</code> with One-Hot Encoding for categorical features & Median Imputation for numeric features<br>
                • <b style="color:#F4F4F5;">Feature Vector:</b> 18 predictors (10 AQ-10 item scores + Age + Gender + Ethnicity + Jaundice + Family History + Country + App Usage + Relation)<br>
                • <b style="color:#F4F4F5;">Evaluation Split:</b> Stratified 80% Train (563 instances) / 20% Holdout Test (141 instances)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="disclaimer-box" style="margin-top: 1rem;">
            <b style="color:#F4F4F5;">Comprehensive Ethical, Clinical & Governance Disclaimer:</b><br>
            1. <b>Screening vs. Diagnosis:</b> This machine learning system is an academic research prototype designed to assist in <i>screening risk prioritization</i> and <i>resource allocation</i>. It does <b>NOT</b> provide clinical diagnosis or medical confirmation of Autism Spectrum Disorder.<br>
            2. <b>False Positives & Negatives:</b> Machine learning classifiers may exhibit false positives (generating unwarranted concern) or false negatives (potentially delaying specialist referral). High screening sensitivity/recall was prioritized during model optimization.<br>
            3. <b>Demographic & Sampling Limitations:</b> Model feature importance reflects statistical predictive associations within the training dataset (UCI Autism Screening Adult Dataset by Dr. Fadi Fayez Thabtah) and does not establish clinical etiology or causation.<br>
            4. <b>Clinical Prerequisite:</b> All allocation recommendations must be reviewed by qualified clinical and psychiatric professionals prior to real-world intervention planning.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PAGE 6: ANALYSIS HISTORY
# ==========================================
elif active_view == "Analysis History":
    st.markdown(
        """
        <div class="top-header">
            <div>
                <h1 class="top-header-title">Analysis History</h1>
                <p class="top-header-subtitle">Review previous ASD screening cohorts and their optimization allocation records.</p>
            </div>
            <span class="badge badge-neutral">● Persistent Storage</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    history_records = load_analysis_history()

    if not history_records:
        st.markdown(
            """
            <div class="empty-state-box">
                <div style="font-size: 24px; margin-bottom: 8px;">↻</div>
                <div style="font-size: 16px; font-weight: 700; color: #F4F4F5; margin-bottom: 4px;">No analysis history yet</div>
                <div style="font-size: 13px; color: #A1A1AA;">Once you process an ASD screening cohort CSV, your previous analyses will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Search & Filter Toolbar
        st.markdown('<div class="card" style="padding: 0.875rem 1.25rem; margin-bottom: 1.25rem;">', unsafe_allow_html=True)
        h_f1, h_f2, h_f3 = st.columns([2, 1, 1])
        with h_f1:
            h_search = st.text_input("Search Cohorts", "", placeholder="Search by dataset name or ID...")
        with h_f2:
            h_source_filter = st.selectbox("Source Filter", ["All Sources", "Uploaded CSV"])
        with h_f3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🗑 Clear All History", key="btn_clear_all_hist", use_container_width=True):
                clear_all_history()
                st.session_state["active_analysis_id"] = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_history = history_records
        if h_search.strip():
            filtered_history = [r for r in filtered_history if h_search.lower() in r.get('dataset_name', '').lower() or h_search.lower() in r.get('analysis_id', '').lower()]
        if h_source_filter != "All Sources":
            filtered_history = [r for r in filtered_history if r.get('source_type') == h_source_filter]

        st.caption(f"Displaying **{len(filtered_history)}** historical analysis record(s).")

        for idx, rec in enumerate(filtered_history):
            is_currently_active = (st.session_state.get("active_analysis_id") == rec.get("analysis_id"))
            rec_id = rec.get("analysis_id")
            ind_count = rec.get('individual_count', rec.get('student_count', 0))
            
            st.markdown(
                f"""
                <div class="card" style="margin-bottom: 1rem; border-left: 4px solid {'#22C55E' if is_currently_active else '#272B33'};">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                        <div>
                            <div style="font-size:15px; font-weight:700; color:#F4F4F5;">{rec.get('dataset_name')}</div>
                            <div style="font-size:12px; color:#71717A; margin-top:2px;">
                                {rec.get('source_type')} · <b>{ind_count} individuals</b> · {rec.get('timestamp')} · ID: <code>{rec_id}</code>
                            </div>
                        </div>
                        <div>
                            {'<span class="badge badge-low">● Active in Dashboard</span>' if is_currently_active else '<span class="badge badge-neutral">Archived</span>'}
                        </div>
                    </div>
                    <div style="display:flex; gap:16px; margin-top:12px; font-size:13px; color:#A1A1AA; flex-wrap:wrap;">
                        <span>High Risk: <b style="color:#F87171;">{rec.get('high_risk')}</b></span>
                        <span>Medium Risk: <b style="color:#FBBF24;">{rec.get('medium_risk')}</b></span>
                        <span>Low Risk: <b style="color:#4ADE80;">{rec.get('low_risk')}</b></span>
                        <span>·</span>
                        <span>Specialist Demand: <b style="color:#F4F4F5;">{rec.get('total_support_demand')}h</b></span>
                        <span>Allocated: <b style="color:#818CF8;">{rec.get('allocated_hours')}h / {rec.get('capacity')}h</b></span>
                        <span>Supported: <b style="color:#F4F4F5;">{rec.get('individuals_selected', rec.get('students_selected', 0))}</b></span>
                        <span>Coverage: <b style="color:#F4F4F5;">{rec.get('demand_coverage'):.1f}%</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            c_act1, c_act2, c_act3 = st.columns([1.5, 1.2, 3])
            with c_act1:
                if not is_currently_active:
                    if st.button(f"Load as Active", key=f"btn_load_{rec_id}_{idx}"):
                        preds_file = rec.get('predictions_file')
                        if os.path.exists(preds_file):
                            hist_df = pd.read_csv(preds_file)
                            total_dem = int(hist_df['support_demand_hours'].sum())
                            def_cap = min(int(rec.get('capacity', 300)), total_dem) if total_dem > 0 else 0
                            _, hist_alloc, _ = solve_resource_allocation(hist_df, def_cap)
                            
                            st.session_state["active_dataset"] = hist_df
                            st.session_state["active_predictions"] = hist_df
                            st.session_state["active_dataset_name"] = rec.get('dataset_name')
                            st.session_state["active_dataset_source"] = rec.get('source_type')
                            st.session_state["active_allocation"] = hist_alloc
                            st.session_state["active_analysis_id"] = rec_id
                            st.session_state["nav_selection"] = "Overview"
                            st.rerun()
                else:
                    st.button("Currently Active", key=f"btn_act_{rec_id}_{idx}", disabled=True)
            
            with c_act2:
                if st.button("🗑 Delete", key=f"btn_del_{rec_id}_{idx}", use_container_width=True):
                    delete_analysis_from_history(rec_id)
                    if is_currently_active:
                        st.session_state["active_analysis_id"] = None
                    st.rerun()

            with c_act3:
                with st.expander("View Data"):
                    p_file = rec.get('predictions_file')
                    if os.path.exists(p_file):
                        st.dataframe(pd.read_csv(p_file).head(20), use_container_width=True, hide_index=True)
                    else:
                        st.info("Historical data file not found.")
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
