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
    page_title="EduRisk — Student Support Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
        width: 250px !important;
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
        background-color: transparent !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: var(--bg-hover) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-left: 2px solid var(--accent-indigo) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Active Dataset Status Box */
    .dataset-status-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 12px;
    }

    .dataset-status-label {
        font-size: 10px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .dataset-status-name {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .dataset-status-meta {
        font-size: 11.5px;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Compact Top Header */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 1.1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }

    .top-header-title {
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0;
    }

    .top-header-subtitle {
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 3px;
        margin-bottom: 0;
    }

    /* Dark SaaS Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.2);
    }

    .card-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.15;
    }

    .card-footer {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* Semantic Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        line-height: 1.2;
    }

    .badge-high {
        background-color: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .badge-medium {
        background-color: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .badge-low {
        background-color: rgba(34, 197, 94, 0.12);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }

    .badge-neutral {
        background-color: var(--bg-elevated);
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

    /* Empty State & Centered Upload Containers */
    .empty-state-box {
        max-width: 580px;
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

    div[data-baseweb="slider"] {
        color: var(--accent-indigo) !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        background: var(--bg-card) !important;
    }

    .scope-box {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-size: 12.5px;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# MODEL & PIPELINE LOADER
# ==========================================
@st.cache_resource
def load_trained_pipeline():
    model_path = os.path.join('models', 'student_performance_rf.joblib')
    if not os.path.exists(model_path):
        return None, [], []
    
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps.get('preprocessor')
    cat_cols = []
    num_cols = []
    if preprocessor is not None:
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'cat':
                cat_cols = list(cols)
            elif name == 'num':
                num_cols = list(cols)
                
    return pipeline, cat_cols, num_cols


@st.cache_data
def load_feature_importance():
    feat_path = os.path.join('outputs', 'feature_importance.csv')
    if os.path.exists(feat_path):
        return pd.read_csv(feat_path)
    return None


pipeline_model, EXPECTED_CAT_COLS, EXPECTED_NUM_COLS = load_trained_pipeline()
ALL_EXPECTED_FEATURES = set(EXPECTED_CAT_COLS + EXPECTED_NUM_COLS)
cached_feat_imp = load_feature_importance()


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
    
    high_count = int(len(preds_df[preds_df['risk_level'] == 'High Risk']))
    med_count = int(len(preds_df[preds_df['risk_level'] == 'Medium Risk']))
    low_count = int(len(preds_df[preds_df['risk_level'] == 'Low Risk']))
    total_demand = int(preds_df['support_demand_hours'].sum())
    
    record = {
        "analysis_id": analysis_id,
        "dataset_name": dataset_name,
        "source_type": source_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_count": len(preds_df),
        "high_risk": high_count,
        "medium_risk": med_count,
        "low_risk": low_count,
        "total_support_demand": total_demand,
        "capacity": int(capacity),
        "allocated_hours": int(metrics.get('allocated_hours', 0)),
        "unused_hours": int(metrics.get('unused_hours', 0)),
        "students_selected": int(metrics.get('students_selected', 0)),
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
# INFERENCE & OPTIMIZATION ENGINES
# ==========================================
def generate_predictions_from_model(df_input, model, cat_cols, num_cols):
    id_candidates = [c for c in df_input.columns if c.lower() in ['student_id', 'studentid', 'id', 'student_code']]
    if id_candidates:
        student_ids = df_input[id_candidates[0]].astype(str).tolist()
    else:
        student_ids = [f"STU{i+1:03d}" for i in range(len(df_input))]

    X_features = df_input[cat_cols + num_cols].copy()
    for nc in num_cols:
        X_features[nc] = pd.to_numeric(X_features[nc], errors='coerce').fillna(0)

    preds = model.predict(X_features)
    preds_rounded = np.round(preds, 2)

    def map_risk(p):
        if p < 10.0:
            return "High Risk"
        elif p <= 13.0:
            return "Medium Risk"
        else:
            return "Low Risk"

    risk_levels = [map_risk(p) for p in preds_rounded]
    risk_scores = np.round(100.0 - (5.0 * preds_rounded), 2)

    demand_map = {"High Risk": 5, "Medium Risk": 2, "Low Risk": 0}
    support_demand = [demand_map[r] for r in risk_levels]

    out_df = pd.DataFrame({
        'student_id': student_ids,
        'predicted_grade': preds_rounded,
        'risk_level': risk_levels,
        'risk_score': risk_scores,
        'support_demand_hours': support_demand
    })

    if 'G3' in df_input.columns:
        out_df['actual_grade'] = pd.to_numeric(df_input['G3'], errors='coerce')
    else:
        out_df['actual_grade'] = np.nan

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


def load_demo_dataset():
    demo_path = os.path.join('student+performance', 'student', 'student-mat.csv')
    if not os.path.exists(demo_path):
        demo_path = 'student-mat.csv'
    raw_df = pd.read_csv(demo_path, sep=';')
    preds_df = generate_predictions_from_model(raw_df, pipeline_model, EXPECTED_CAT_COLS, EXPECTED_NUM_COLS)
    default_cap = 300
    _, alloc_df, metrics = solve_resource_allocation(preds_df, default_cap)
    
    aid = save_analysis_to_history("UCI Student Performance (Math)", "Demo Dataset", preds_df, alloc_df, metrics, default_cap)
    
    st.session_state["active_dataset"] = preds_df
    st.session_state["active_predictions"] = preds_df
    st.session_state["active_dataset_name"] = "UCI Student Performance (Math)"
    st.session_state["active_dataset_source"] = "Demo Dataset"
    st.session_state["active_allocation"] = alloc_df
    st.session_state["active_analysis_id"] = aid
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
                <div class="brand-text">EduRisk</div>
                <div class="brand-subtext">Student Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-category">WORKSPACE</div>', unsafe_allow_html=True)
    
    nav_options = [
        "Overview",
        "Risk Analysis",
        "Resource Allocation",
        "Student Explorer"
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
        num_stu = len(st.session_state["active_dataset"])
        total_dem = int(st.session_state["active_dataset"]['support_demand_hours'].sum())
        st.markdown(
            f"""
            <div class="dataset-status-card">
                <div class="dataset-status-label"><span style="color:#22C55E;">●</span> Active Dataset</div>
                <div class="dataset-status-name">{st.session_state["active_dataset_name"]}</div>
                <div class="dataset-status-meta"><b>{num_stu}</b> students · <b>{total_dem}h</b> demand</div>
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
            file_name=f"predictions_{st.session_state['active_dataset_name'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        def_cap = min(300, total_dem) if total_dem > 0 else 0
        _, cur_alloc_df, _ = run_dynamic_optimization(st.session_state["active_dataset"], def_cap)
        alloc_csv = cur_alloc_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="↓ Allocation Plan",
            data=alloc_csv,
            file_name=f"allocation_{st.session_state['active_dataset_name'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown(
            """
            <div class="dataset-status-card">
                <div class="dataset-status-label"><span style="color:#71717A;">○</span> No Dataset Active</div>
                <div class="dataset-status-meta" style="margin-top:4px;">Upload a CSV to begin analysis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("+ Upload CSV", key="btn_quick_upload", use_container_width=True):
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
# PAGE ROUTING & GUARDS
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
                <h1 class="top-header-title">Dataset Ingestion</h1>
                <p class="top-header-subtitle">Import a student cohort CSV to generate predictions and optimize academic support.</p>
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
            <div style="font-size: 16px; font-weight: 700; color: #F4F4F5; margin-bottom: 4px;">Import student data</div>
            <div style="font-size: 13.5px; color: #A1A1AA; margin-bottom: 16px;">Upload a CSV containing your student cohort.<br>32 required features · G3 optional</div>
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

            missing_cols = [c for c in ALL_EXPECTED_FEATURES if c not in raw_df.columns]
            
            if missing_cols:
                st.error(f"❌ **Dataset Missing {len(missing_cols)} Required Features:**")
                st.markdown("\n".join([f"- `{col}`" for col in missing_cols]))
                st.info("💡 Expected features include demographic, attendance, family, and prior milestone grades (G1, G2).")
            elif pipeline_model is None:
                st.error("Trained model artifact missing from `models/`.")
            else:
                preds_df = generate_predictions_from_model(
                    raw_df, pipeline_model, EXPECTED_CAT_COLS, EXPECTED_NUM_COLS
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
                            <span style="color:#4ADE80;">✓ Dataset ready & Saved to History</span>
                            <span class="badge badge-low">Prediction ready</span>
                        </div>
                        <div style="font-size: 13px; color: #A1A1AA; margin-top: 8px;">
                            • File: <b>{uploaded_file.name}</b><br>
                            • Size: <b>{len(preds_df)} students</b> · <b>32 features</b><br>
                            • Model Inference: <b>Random Forest Regressor</b><br>
                            • Support Demand: <b>{total_dem} hours / week</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("Run Analysis →", key="btn_run_analysis"):
                    st.session_state["nav_selection"] = "Overview"
                    st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # How It Works Grid
    st.markdown(
        """
        <div style="margin-top: 2rem;">
            <div style="font-size: 11px; font-weight: 700; color: #71717A; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">HOW IT WORKS</div>
            <div class="step-grid">
                <div class="step-card">
                    <div class="step-num">01</div>
                    <div class="step-title">Validate</div>
                    <div class="step-desc">Verify dataset schema against 32 features.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">02</div>
                    <div class="step-title">Predict</div>
                    <div class="step-desc">Generate expected grades on 0–20 scale.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">03</div>
                    <div class="step-title">Prioritize</div>
                    <div class="step-desc">Identify students requiring support.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">04</div>
                    <div class="step-title">Allocate</div>
                    <div class="step-desc">Optimize limited intervention hours.</div>
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
                    <h1 class="top-header-title">Student Risk Overview</h1>
                    <p class="top-header-subtitle">Monitor academic risk and understand where support resources are needed.</p>
                </div>
                <span class="badge badge-neutral">○ NO DATASET</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 5 KPI Placeholder Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown('<div class="card"><div class="card-label">Total Students</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">No active cohort</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="card"><div class="card-label">High Risk</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="card"><div class="card-label">Medium Risk</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown('<div class="card"><div class="card-label">Low Risk</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown('<div class="card"><div class="card-label">Support Demand</div><div class="card-value" style="color:#71717A;">—</div><div class="card-footer">—</div></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="empty-state-box">
                <div style="width: 44px; height: 44px; background: #151820; color: #6366F1; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 1rem;">◈</div>
                <div style="font-size: 17px; font-weight: 700; color: #F4F4F5; margin-bottom: 4px;">No dataset loaded</div>
                <div style="font-size: 13.5px; color: #A1A1AA; margin-bottom: 20px;">Upload a student cohort to begin analysis. Predictions, risk analysis and resource optimization will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        btn_c1, btn_c2, btn_c3 = st.columns([1, 1.2, 1])
        with btn_c2:
            if st.button("Upload CSV Cohort", key="btn_empty_upload", use_container_width=True):
                st.session_state["nav_selection"] = "Import Dataset"
                st.rerun()
    else:
        # ACTIVE DATASET OVERVIEW
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">Student Risk Overview</h1>
                    <p class="top-header-subtitle">Monitor academic risk and understand where support resources are needed.</p>
                </div>
                <span class="badge badge-neutral">● ACTIVE: {dataset_display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        total_students = len(current_predictions_df)
        high_risk_count = len(current_predictions_df[current_predictions_df['risk_level'] == 'High Risk'])
        med_risk_count = len(current_predictions_df[current_predictions_df['risk_level'] == 'Medium Risk'])
        low_risk_count = len(current_predictions_df[current_predictions_df['risk_level'] == 'Low Risk'])
        
        total_demand_hours = int(current_predictions_df['support_demand_hours'].sum())
        high_demand_hours = high_risk_count * 5
        med_demand_hours = med_risk_count * 2

        base_cap = min(300, total_demand_hours) if total_demand_hours > 0 else 0
        _, _, base_metrics = run_dynamic_optimization(current_predictions_df, base_cap)

        # 5-Card Metric Row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f'<div class="card"><div class="card-label">Total Students</div><div class="card-value">{total_students}</div><div class="card-footer">Active cohort</div></div>', unsafe_allow_html=True)
        with k2:
            hr_pct = (high_risk_count / total_students * 100) if total_students else 0
            st.markdown(f'<div class="card"><div class="card-label">High Risk</div><div class="card-value" style="color:#F87171;">{high_risk_count}</div><div class="card-footer"><span class="badge badge-high">{hr_pct:.1f}%</span> of cohort</div></div>', unsafe_allow_html=True)
        with k3:
            mr_pct = (med_risk_count / total_students * 100) if total_students else 0
            st.markdown(f'<div class="card"><div class="card-label">Medium Risk</div><div class="card-value" style="color:#FBBF24;">{med_risk_count}</div><div class="card-footer"><span class="badge badge-medium">{mr_pct:.1f}%</span> of cohort</div></div>', unsafe_allow_html=True)
        with k4:
            lr_pct = (low_risk_count / total_students * 100) if total_students else 0
            st.markdown(f'<div class="card"><div class="card-label">Low Risk</div><div class="card-value" style="color:#4ADE80;">{low_risk_count}</div><div class="card-footer"><span class="badge badge-low">{lr_pct:.1f}%</span> of cohort</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown(f'<div class="card"><div class="card-label">Support Demand</div><div class="card-value" style="color:#818CF8;">{total_demand_hours}h</div><div class="card-footer">Unconstrained</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Hero Insight
        st.markdown(
            f"""
            <div class="hero-insight">
                <div class="hero-insight-title">
                    <span>Cohort Requires Attention</span>
                    <span class="badge badge-neutral">{total_demand_hours}h Total Demand</span>
                </div>
                <div class="hero-insight-body">
                    <b>{high_risk_count} students</b> are currently classified as High Risk.<br>
                    Estimated support requirement: <b>{high_demand_hours} hours / week</b> for high-risk students. At baseline capacity (<b>{base_cap}h</b>), the optimizer covers <b>{base_metrics['demand_coverage_percentage']:.1f}%</b> of total cohort demand.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk Distribution Dark Chart with Direct Text Labels
        st.markdown('<div class="card"><div class="card-label">Risk Distribution (Students by Tier)</div>', unsafe_allow_html=True)
        hr_p = (high_risk_count / total_students * 100) if total_students else 0
        mr_p = (med_risk_count / total_students * 100) if total_students else 0
        lr_p = (low_risk_count / total_students * 100) if total_students else 0
        
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
            x=alt.X('Count:Q', title='Students', scale=alt.Scale(domain=[0, max_c * 1.25]), axis=alt.Axis(grid=True, gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA'))
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
        risk_chart = (bars_risk + text_risk).properties(height=210, background='#111318').configure_view(stroke=None)
        st.altair_chart(risk_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 2: STUDENT RISK ANALYSIS
# ==========================================
elif active_view == "Risk Analysis":
    if current_predictions_df is None:
        st.markdown('<div class="top-header"><div><h1 class="top-header-title">Student Risk Analysis</h1><p class="top-header-subtitle">Explore predicted academic risk across the active cohort.</p></div><span class="badge badge-neutral">○ NO DATASET</span></div>', unsafe_allow_html=True)
        st.info("💡 No dataset active. Please upload a student cohort CSV to begin analysis.")
        if st.button("Upload Cohort CSV", key="btn_guard_upload_risk"):
            st.session_state["nav_selection"] = "Import Dataset"
            st.rerun()
    else:
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">Student Risk Analysis</h1>
                    <p class="top-header-subtitle">Explore predicted academic risk across the active cohort.</p>
                </div>
                <span class="badge badge-neutral">● {dataset_display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="card" style="padding: 0.875rem 1.25rem; margin-bottom: 1.25rem;">', unsafe_allow_html=True)
        f1, f2, f3 = st.columns([1.5, 2, 1.2])
        with f1:
            risk_filter = st.multiselect("Risk Tier", options=["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk", "Low Risk"])
        with f2:
            min_p = float(current_predictions_df['predicted_grade'].min()) if not current_predictions_df.empty else 0.0
            max_p = float(current_predictions_df['predicted_grade'].max()) if not current_predictions_df.empty else 20.0
            grade_range = st.slider("Predicted Grade Range", min_value=0.0, max_value=20.0, value=(min_p, max_p), step=0.5)
        with f3:
            search_id = st.text_input("Filter ID", "", placeholder="e.g. STU001")
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = current_predictions_df[
            (current_predictions_df['risk_level'].isin(risk_filter)) &
            (current_predictions_df['predicted_grade'] >= grade_range[0]) &
            (current_predictions_df['predicted_grade'] <= grade_range[1])
        ]
        if search_id.strip():
            filtered_df = filtered_df[filtered_df['student_id'].str.contains(search_id.strip(), case=False, na=False)]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(current_predictions_df)}** students.")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('<div class="card"><div class="card-label">Filtered Risk Breakdown</div>', unsafe_allow_html=True)
            hr_cnt = len(filtered_df[filtered_df['risk_level'] == 'High Risk'])
            mr_cnt = len(filtered_df[filtered_df['risk_level'] == 'Medium Risk'])
            lr_cnt = len(filtered_df[filtered_df['risk_level'] == 'Low Risk'])
            r_counts = pd.DataFrame({
                'Risk': ['High Risk', 'Medium Risk', 'Low Risk'],
                'Count': [hr_cnt, mr_cnt, lr_cnt],
                'Label': [f"{hr_cnt} stu", f"{mr_cnt} stu", f"{lr_cnt} stu"]
            })
            max_rc = max(hr_cnt, mr_cnt, lr_cnt, 1)
            base_r = alt.Chart(r_counts).encode(
                x=alt.X('Risk:N', title=None, sort=['High Risk', 'Medium Risk', 'Low Risk'], axis=alt.Axis(labelColor='#F4F4F5', labelFontSize=12)),
                y=alt.Y('Count:Q', title='Students', scale=alt.Scale(domain=[0, max_rc * 1.3]), axis=alt.Axis(grid=True, gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA'))
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
            st.markdown('<div class="card"><div class="card-label">Predicted Grade Distribution</div>', unsafe_allow_html=True)
            if not filtered_df.empty:
                g_hist = alt.Chart(filtered_df).mark_bar(cornerRadiusEnd=2, color='#6366F1').encode(
                    x=alt.X('predicted_grade:Q', bin=alt.Bin(maxbins=20), title='Predicted Grade (0–20)', axis=alt.Axis(gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA')),
                    y=alt.Y('count():Q', title='Students', axis=alt.Axis(grid=True, gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA'))
                ).properties(height=180, background='#111318').configure_view(stroke=None)
                st.altair_chart(g_hist, use_container_width=True)
            else:
                st.info("No matching records.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 10px;">Student Roster</div>', unsafe_allow_html=True)
        tbl_cols = ['student_id', 'predicted_grade', 'risk_level', 'risk_score', 'support_demand_hours']
        if 'actual_grade' in filtered_df.columns and not filtered_df['actual_grade'].isnull().all():
            tbl_cols.insert(2, 'actual_grade')
        for extra in ['absences', 'failures', 'studytime', 'G1', 'G2']:
            if extra in filtered_df.columns:
                tbl_cols.append(extra)

        st.dataframe(filtered_df[tbl_cols].sort_values(by='risk_score', ascending=False), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 3: RESOURCE ALLOCATION
# ==========================================
elif active_view == "Resource Allocation":
    if current_predictions_df is None:
        st.markdown('<div class="top-header"><div><h1 class="top-header-title">Resource Allocation</h1><p class="top-header-subtitle">Optimize limited support hours across students with highest predicted need.</p></div><span class="badge badge-neutral">○ NO DATASET</span></div>', unsafe_allow_html=True)
        st.info("💡 No dataset active. Please upload a student cohort CSV to begin analysis.")
        if st.button("Upload Cohort CSV", key="btn_guard_upload_alloc"):
            st.session_state["nav_selection"] = "Import Dataset"
            st.rerun()
    else:
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">Resource Allocation</h1>
                    <p class="top-header-subtitle">Optimize limited support hours across students with highest predicted need.</p>
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
            st.error(f"Solver Error: {status}")
            st.stop()

        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #22C55E; margin-bottom: 1.25rem;">
                <div style="font-size: 14px; font-weight: 700; color: #F4F4F5; display: flex; justify-content: space-between;">
                    <span style="color:#4ADE80;">✓ Optimal allocation found</span>
                    <span class="badge badge-low">{metrics['resource_utilization_percentage']:.1f}% Utilization</span>
                </div>
                <div style="font-size: 13px; color: #A1A1AA; margin-top: 6px;">
                    <b>{metrics['allocated_hours']}h allocated</b> of {metrics['available_hours']}h capacity · <b>{metrics['unused_hours']}h remaining</b> · <b>{metrics['students_selected']} students supported</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="card"><div class="card-label">Allocated Hours</div><div class="card-value" style="color:#818CF8;">{metrics["allocated_hours"]}h</div><div class="card-footer">Budget: {capacity_input}h</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="card"><div class="card-label">Remaining Capacity</div><div class="card-value">{metrics["unused_hours"]}h</div><div class="card-footer">Unused hours</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="card"><div class="card-label">Demand Coverage</div><div class="card-value">{metrics["demand_coverage_percentage"]:.1f}%</div><div class="card-footer">{metrics["allocated_hours"]} of {total_demand}h</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="card"><div class="card-label">Students Selected</div><div class="card-value">{metrics["students_selected"]}</div><div class="card-footer">{metrics["high_risk_selected"]} High + {metrics["medium_risk_selected"]} Med</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 10px;">Intervention Schedule (Prioritized Students)</div>', unsafe_allow_html=True)
        sel_df = dyn_alloc_df[dyn_alloc_df['selected_for_support'] == 1].sort_values(by='risk_score', ascending=False)
        s_cols = ['student_id', 'predicted_grade', 'risk_level', 'risk_score', 'support_demand_hours']
        for extra in ['absences', 'failures', 'studytime', 'G1', 'G2']:
            if extra in sel_df.columns:
                s_cols.append(extra)
        st.dataframe(sel_df[s_cols], use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 4: STUDENT EXPLORER
# ==========================================
elif active_view == "Student Explorer":
    if current_predictions_df is None:
        st.markdown('<div class="top-header"><div><h1 class="top-header-title">Student Explorer</h1><p class="top-header-subtitle">Review an individual student\'s predicted risk and recommended intervention level.</p></div><span class="badge badge-neutral">○ NO DATASET</span></div>', unsafe_allow_html=True)
        st.info("💡 No dataset active. Please upload a student cohort CSV to begin analysis.")
        if st.button("Upload Cohort CSV", key="btn_guard_upload_exp"):
            st.session_state["nav_selection"] = "Import Dataset"
            st.rerun()
    else:
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1 class="top-header-title">Student Explorer</h1>
                    <p class="top-header-subtitle">Review an individual student's predicted risk and recommended intervention level.</p>
                </div>
                <span class="badge badge-neutral">● {dataset_display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        student_list = current_predictions_df['student_id'].tolist()
        if not student_list:
            st.info("No students in active cohort.")
            st.stop()

        st.markdown('<div class="card" style="padding: 1rem 1.25rem; margin-bottom: 1.25rem;">', unsafe_allow_html=True)
        sc1, sc2 = st.columns([2, 3])
        with sc1:
            sel_id = st.selectbox("Select Student", student_list, index=0)
        
        stu = current_predictions_df[current_predictions_df['student_id'] == sel_id].iloc[0]
        
        # Current allocation match
        def_cap = min(300, int(current_predictions_df['support_demand_hours'].sum())) if len(current_predictions_df) else 0
        _, cur_alloc_df, _ = run_dynamic_optimization(current_predictions_df, def_cap)
        base_m = cur_alloc_df[cur_alloc_df['student_id'] == sel_id]
        is_sel = int(base_m.iloc[0]['selected_for_support']) if not base_m.empty else 0

        with sc2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            if stu['risk_level'] == "High Risk":
                st.markdown('<span class="badge badge-high">● High Risk</span>', unsafe_allow_html=True)
            elif stu['risk_level'] == "Medium Risk":
                st.markdown('<span class="badge badge-medium">● Medium Risk</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-low">● Low Risk</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="card"><div class="card-label">Predicted Grade</div><div class="card-value">{stu["predicted_grade"]:.2f} <span style="font-size:14px; color:#A1A1AA;">/ 20</span></div><div class="card-footer">{"Actual: " + str(stu["actual_grade"]) if pd.notnull(stu.get("actual_grade")) else "Unseen G3"}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="card"><div class="card-label">Risk Score</div><div class="card-value">{stu["risk_score"]:.1f}</div><div class="card-footer">Urgency index</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="card"><div class="card-label">Support Demand</div><div class="card-value">{stu["support_demand_hours"]}h</div><div class="card-footer">Weekly allocation</div></div>', unsafe_allow_html=True)
        with m4:
            status_tag = '<span class="badge badge-low">✓ Selected</span>' if is_sel == 1 else ('<span class="badge badge-high">Waitlisted</span>' if stu['support_demand_hours'] > 0 else '<span class="badge badge-neutral">Standard Pace</span>')
            st.markdown(f'<div class="card"><div class="card-label">Allocation Status</div><div style="margin-top:6px;">{status_tag}</div><div class="card-footer">Under {def_cap}h baseline capacity</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        if stu['risk_level'] == "High Risk":
            st.markdown('<div class="card" style="border-left: 4px solid #EF4444;"><div style="font-size: 13.5px; font-weight: 700; color: #F87171;">PRIORITY INTERVENTION</div><div style="font-size: 13px; color: #A1A1AA; margin-top: 4px;">This student is classified as High Risk. Recommended support allocation: <b style="color:#F4F4F5;">5 hours / week</b> of structured 1-on-1 tutoring.</div></div>', unsafe_allow_html=True)
        elif stu['risk_level'] == "Medium Risk":
            st.markdown('<div class="card" style="border-left: 4px solid #F59E0B;"><div style="font-size: 13.5px; font-weight: 700; color: #FBBF24;">MODERATE SUPPORT</div><div style="font-size: 13px; color: #A1A1AA; margin-top: 4px;">This student is classified as Medium Risk. Recommended support allocation: <b style="color:#F4F4F5;">2 hours / week</b> of small-group review.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card" style="border-left: 4px solid #22C55E;"><div style="font-size: 13.5px; font-weight: 700; color: #4ADE80;">STANDARD PROGRESSION</div><div style="font-size: 13px; color: #A1A1AA; margin-top: 4px;">This student is classified as Low Risk. No additional support allocation required under current policy.</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 10px;">Contextual Student Indicators</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Academic Milestones:**<br>• G1: **{stu.get('G1', 'N/A')}** / 20<br>• G2: **{stu.get('G2', 'N/A')}** / 20<br>• Past Failures: **{stu.get('failures', 'N/A')}**<br>• Study Time: Level **{stu.get('studytime', 'N/A')}**", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Attendance & Well-being:**<br>• Absences: **{stu.get('absences', 'N/A')} days**<br>• Health: **{stu.get('health', 'N/A')}** / 5<br>• Free Time: **{stu.get('freetime', 'N/A')}** / 5<br>• Peer Outings: **{stu.get('goout', 'N/A')}** / 5", unsafe_allow_html=True)
        with c3:
            st.markdown(f"**Background & Goals:**<br>• School: **{stu.get('school', 'N/A')}** | Age: **{stu.get('age', 'N/A')}**<br>• Mother Education: Level **{stu.get('Medu', 'N/A')}**<br>• Father Education: Level **{stu.get('Fedu', 'N/A')}**<br>• Higher Ed: **{stu.get('higher', 'N/A')}**", unsafe_allow_html=True)
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
                <p class="top-header-subtitle">Performance benchmarks, feature attribution, and governance metadata for the Random Forest model.</p>
            </div>
            <span class="badge badge-neutral">● Random Forest Regressor</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="card"><div class="card-label">R² Score</div><div class="card-value" style="color:#818CF8;">0.805</div><div class="card-footer">80.5% variance explained</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="card"><div class="card-label">MAE</div><div class="card-value">1.178</div><div class="card-footer">On a 20-point scale</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="card"><div class="card-label">RMSE</div><div class="card-value">2.001</div><div class="card-footer">Holdout test evaluation</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="card"><div class="card-label">Ensemble Size</div><div class="card-value">200</div><div class="card-footer">Decision trees</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label" style="margin-bottom: 10px;">Top 10 Feature Attribution</div>', unsafe_allow_html=True)
    if cached_feat_imp is not None:
        top10 = cached_feat_imp.head(10).sort_values(by='importance', ascending=True)
        imp_chart = alt.Chart(top10).mark_bar(cornerRadiusEnd=3, color='#6366F1', size=16).encode(
            x=alt.X('importance:Q', title='Feature Importance Weight', axis=alt.Axis(gridColor='#272B33', labelColor='#A1A1AA', titleColor='#A1A1AA')),
            y=alt.Y('feature:N', title=None, sort='-x', axis=alt.Axis(labelColor='#A1A1AA')),
            tooltip=['feature', 'importance']
        ).properties(height=240, background='#111318').configure_view(stroke=None)
        st.altair_chart(imp_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-label" style="margin-bottom: 8px;">Model Architecture</div>
            <div style="font-size: 13px; color: #A1A1AA; line-height: 1.6;">
                • <b style="color:#F4F4F5;">Algorithm:</b> <code>RandomForestRegressor</code> (scikit-learn)<br>
                • <b style="color:#F4F4F5;">Estimators:</b> 200 trees · <b>Random State:</b> 42<br>
                • <b style="color:#F4F4F5;">Features:</b> 32 predictors (17 categorical one-hot encoded, 15 numeric pass-through)<br>
                • <b style="color:#F4F4F5;">Training / Holdout Split:</b> 80% (316) / 20% (79)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="scope-box">
            <b style="color:#F4F4F5;">Scope & Limitations:</b> This prototype predicts student academic performance. It is <b>not a clinical or diagnostic system and is not validated for ASD-specific prediction</b>. The underlying architecture can be adapted to validated clinical educational datasets.
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
                <p class="top-header-subtitle">Review previous student cohorts and their resource allocation results.</p>
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
                <div style="font-size: 13px; color: #A1A1AA;">Once you process a student cohort, your previous analyses will appear here.</div>
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
            if st.button("🗑 Clear All", key="btn_clear_all_hist", use_container_width=True):
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
            
            st.markdown(
                f"""
                <div class="card" style="margin-bottom: 1rem; border-left: 4px solid {'#22C55E' if is_currently_active else '#272B33'};">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                        <div>
                            <div style="font-size:15px; font-weight:700; color:#F4F4F5;">{rec.get('dataset_name')}</div>
                            <div style="font-size:12px; color:#71717A; margin-top:2px;">
                                {rec.get('source_type')} · <b>{rec.get('student_count')} students</b> · {rec.get('timestamp')} · ID: <code>{rec_id}</code>
                            </div>
                        </div>
                        <div>
                            {'<span class="badge badge-low">● Active in Dashboard</span>' if is_currently_active else '<span class="badge badge-neutral">Archived</span>'}
                        </div>
                    </div>
                    <div style="display:flex; gap:16px; margin-top:12px; font-size:13px; color:#A1A1AA; flex-wrap:wrap;">
                        <span>High: <b style="color:#F87171;">{rec.get('high_risk')}</b></span>
                        <span>Med: <b style="color:#FBBF24;">{rec.get('medium_risk')}</b></span>
                        <span>Low: <b style="color:#4ADE80;">{rec.get('low_risk')}</b></span>
                        <span>·</span>
                        <span>Demand: <b style="color:#F4F4F5;">{rec.get('total_support_demand')}h</b></span>
                        <span>Allocated: <b style="color:#818CF8;">{rec.get('allocated_hours')}h / {rec.get('capacity')}h</b></span>
                        <span>Selected: <b style="color:#F4F4F5;">{rec.get('students_selected')}</b></span>
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
