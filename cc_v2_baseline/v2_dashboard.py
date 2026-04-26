import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FraudGuard V2.0 | Security Console", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION (For Live ROI Tracking) ---
if 'processed_txns' not in st.session_state:
    st.session_state.processed_txns = {}
if 'current_txn_id' not in st.session_state:
    st.session_state.current_txn_id = None
if 'txn_type' not in st.session_state:
    st.session_state.txn_type = None

# --- ADVANCED FUTURISTIC CSS (V2.0 Cyan/Emerald Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(rgba(10, 15, 30, 0.90), rgba(10, 15, 30, 0.98)), 
                    url('https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?q=80&w=2070&auto=format&fit=crop'); /* Updated Matrix/Hacker background */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #F8FAFC;
    }

    h1, h2, h3, .hero-title { font-family: 'Orbitron', sans-serif; letter-spacing: 2px; }
    p, label, .stMarkdown, li { font-family: 'Inter', sans-serif; color: #cbd5e1; }

    /* V2.0 Hero Section */
    .hero-container { text-align: center; padding: 40px 0 40px 0; }
    .hero-title {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(180deg, #FFFFFF 0%, #06B6D4 100%); /* Cyan Gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        filter: drop-shadow(0 0 25px rgba(6, 182, 212, 0.5));
    }
    
    .v2-badge {
        background: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        padding: 5px 15px;
        border-radius: 20px;
        border: 1px solid rgba(6, 182, 212, 0.5);
        font-family: 'Orbitron';
        font-size: 0.9rem;
        letter-spacing: 1px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.98) !important;
        border-right: 1px solid rgba(6, 182, 212, 0.3);
    }
    [data-testid="stSidebarNav"] { display: none; }
    
    /* Expander Glassmorphism */
    [data-testid="stExpander"] details {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(6, 182, 212, 0.2) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stExpander"] details:hover {
        border-color: rgba(6, 182, 212, 0.6) !important;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.15);
    }
    [data-testid="stExpander"] summary span {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        color: #f1f5f9;
        text-transform: uppercase;
        font-size: 0.9rem;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        border: 1px solid rgba(6, 182, 212, 0.5);
        background: rgba(15, 23, 42, 0.8);
        color: white;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        border-color: #22D3EE;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
        color: #22D3EE;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] { background: rgba(15, 23, 42, 0.5); border-radius: 8px; }
    
    /* Metric Value Styling */
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: white !important; font-family: 'Orbitron', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model, scaler, df, error = None, None, None, None

    if os.path.exists('cc_fraud2.0.joblib'):
        model = joblib.load('cc_fraud2.0.joblib')
    elif os.path.exists('v2_xgboost_baseline_model.joblib'):
        model = joblib.load('v2_xgboost_baseline_model.joblib')
    else:
        error = "❌ Model file not found. Please ensure 'cc_fraud2.0.joblib' is present."

    if os.path.exists('v2_scaler.joblib'):
        scaler = joblib.load('v2_scaler.joblib')

    if os.path.exists('test_dataset.csv'):
        df = pd.read_csv('test_dataset.csv')
    else:
        error = "❌ 'test_dataset.csv' not found. Please run the training script."
        
    return model, scaler, df, error

model, scaler, dataset, error = load_resources()

# --- MLOps DATA LOADER ---
def get_mlops_metrics():
    """Reads the automated experiment tracker for live telemetry."""
    if os.path.exists('experiment_tracking.csv'):
        try:
            tracker = pd.read_csv('experiment_tracking.csv')
            return tracker.iloc[-1] # Get the most recent training run
        except Exception:
            return None
    return None

mlops_data = get_mlops_metrics()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9528/9528169.png", width=70)
    st.markdown("<h3 style='margin-top: 10px; margin-bottom: 30px; letter-spacing: 2px;'>FRAUDGUARD <span style='color:#06B6D4;'>V2.0</span></h3>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #94A3B8; margin-bottom: 10px;'>INTERFACE SELECT</p>", unsafe_allow_html=True)
    page = st.radio("", ["🏠 OVERVIEW", "📡 LIVE MONITOR"], label_visibility="collapsed")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #94A3B8; margin-bottom: 10px;'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem;'><span style='color:#10B981; font-weight:bold;'>🟢</span> XGBoost Engine Active</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem;'><span style='color:#06B6D4; font-weight:bold;'>🔵</span> MLOps Guardrails: ON</p>", unsafe_allow_html=True)

    if page == "📡 LIVE MONITOR":
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #94A3B8; margin-bottom: 10px;'>SENSITIVITY OVERRIDE</p>", unsafe_allow_html=True)
        threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)

    # Add MLOps Telemetry to Sidebar
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #94A3B8; margin-bottom: 10px;'>📡 MLOPS TELEMETRY</p>", unsafe_allow_html=True)
    if mlops_data is not None:
        run_date = mlops_data.get('Date', mlops_data.get('Timestamp', 'N/A'))
        model_name = mlops_data.get('Model', 'XGBoost V2.0')
        st.caption(f"**Last Run:** {run_date}")
        st.caption(f"**Engine:** {model_name}")
    else:
        st.caption("Telemetry offline (No tracking log found).")


# --- PAGE 1: PROJECT OVERVIEW ---
if page == "🏠 OVERVIEW":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">FRAUDGUARD AI</div>
            <p style="color: #94A3B8; font-size: 1.2rem; margin-top:10px;"><span class="v2-badge">VERSION 2.0 ARCHITECTURE</span></p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📖 THE EVOLUTION (V1 vs V2)", expanded=True):
            st.markdown("""
            **Why V2.0 was necessary:**
            FraudGuard V1 analyzed 28 features but struggled with the 0.17% imbalance paradox. It dropped time data completely.
            
            **V2.0** acts as a true investigator. It implements **Cyclical Time Engineering** (catching night-time attacks) and **Log-Normalization** to penalize suspicious financial behaviors 577x harder than normal traffic.
            """)
        with st.expander("📊 DATA STREAM"):
            st.markdown("""
            **Ingestion Node:**
            - Base: European Transactions
            - Engineered Features: `Log_Amount`, `Is_Night`, `Hour_of_Day`
            - Latent Vectors: V1-V28 (PCA Anonymized)
            """)

    with col2:
        with st.expander("⚙️ THE XGBOOST ENGINE", expanded=True):
            st.markdown("""
            **Algorithmic Shift:**
            We abandoned synthetic data (SMOTE) and legacy Random Forests. 
            
            V2.0 is powered by **Extreme Gradient Boosting (XGBoost)**. It uses a mathematical `scale_pos_weight` penalty, forcing the AI to focus aggressively on the minority fraud class using strictly real-world data.
            """)
        with st.expander("🔐 PCA ANONYMITY"):
            st.markdown("""
            Principal Component Analysis (PCA) ensures the AI learns behavioral and spatial patterns without ever seeing private personal information, making it GDPR compliant.
            """)

    with col3:
        with st.expander("🛠️ INVESTIGATOR MODE", expanded=True):
            st.markdown("""
            **How to operate the Live Monitor:**
            1. Trigger **Normal** or **Attack** vectors.
            2. View the **Raw Data Stream** payload.
            3. Analyze the **AI Decision Logic** to see exactly *which* features caused the alarm.
            """)
        with st.expander("🚀 MLOPS PIPELINE"):
            st.markdown("""
            - Automated Experiment Tracking (CSV)
            - AUPRC > 0.80 Guardrails
            - Joblib Artifact Serialization
            """)

# --- PAGE 2: LIVE MONITOR ---
elif page == "📡 LIVE MONITOR":
    if error:
        st.error(error)
        st.stop()

    st.markdown("### 📡 LIVE TRANSACTION STREAM & EXPLAINABILITY")
    st.markdown("---")

    # Placeholder for top metrics so we can calculate Live ROI first
    metrics_placeholder = st.empty()
    impact = 0.0 # Default value for delta

    col_ctrl, col_viz = st.columns([1, 1.5])

    with col_ctrl:
        st.markdown("##### 📥 1. INGEST TRAFFIC")
        if dataset is not None:
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("SIMULATE NORMAL", use_container_width=True):
                    st.session_state['txn'] = dataset[dataset['Class'] == 0].sample(1)
                    st.session_state['current_txn_id'] = np.random.randint(1000000, 9999999)
                    st.session_state['txn_type'] = 'normal'
            with c_btn2:
                if st.button("SIMULATE ATTACK", use_container_width=True):
                    st.session_state['txn'] = dataset[dataset['Class'] == 1].sample(1)
                    st.session_state['current_txn_id'] = np.random.randint(1000000, 9999999)
                    st.session_state['txn_type'] = 'fraud'
            
            # --- NEW: MANUAL INJECTION WIDGET ---
            with st.expander("🛠️ INJECT SPECIFIC ROW", expanded=False):
                tab_idx, tab_raw = st.tabs(["By Dataset Index", "Paste Values"])
                
                with tab_idx:
                    st.caption("Note: Pandas uses a **0-based index**. Index 0 is the 1st data row (Excel Row 2).")
                    max_idx = len(dataset) - 1 if dataset is not None and len(dataset) > 0 else 0
                    manual_idx = st.number_input(f"Enter Row Index (0 to {max_idx})", min_value=0, max_value=max_idx, step=1, value=0)
                    
                    if st.button("LOAD BY INDEX", use_container_width=True):
                        try:
                            # Use .iloc for strict 0-based positional indexing
                            row = dataset.iloc[[manual_idx]]
                            st.session_state['txn'] = row
                            st.session_state['current_txn_id'] = f"IDX-{manual_idx}"
                            st.session_state['txn_type'] = 'fraud' if row['Class'].values[0] == 1 else 'normal'
                        except IndexError:
                            st.error(f"Row {manual_idx} is out of bounds for the dataset.")
                            
                with tab_raw:
                    st.caption("Paste comma-separated values (Must be exactly all features, excluding 'Class').")
                    raw_input = st.text_area("Paste Data:", height=100)
                    if st.button("LOAD RAW VALUES", use_container_width=True):
                        try:
                            # Clean the input, split by comma, and convert to floats
                            vals = [float(x.strip()) for x in raw_input.split(',') if x.strip() != '']
                            expected_cols = [c for c in dataset.columns if c not in ['Class', 'Unnamed: 0']]
                            
                            if len(vals) == len(expected_cols):
                                injected_df = pd.DataFrame([vals], columns=expected_cols)
                                injected_df['Class'] = -1 # Unknown class for raw injection
                                st.session_state['txn'] = injected_df
                                st.session_state['current_txn_id'] = f"RAW-{np.random.randint(1000, 9999)}"
                                st.session_state['txn_type'] = 'unknown' # Flags the UI to handle this gracefully
                            else:
                                st.error(f"Expected {len(expected_cols)} features, but you pasted {len(vals)} values.")
                        except Exception as e:
                            st.error(f"Parsing error: Make sure you pasted only numbers separated by commas. Error: {e}")
        
        if 'txn' in st.session_state:
            row = st.session_state['txn']
            X_input = row.drop(['Class', 'Unnamed: 0'], axis=1, errors='ignore')
            
            # Predict
            try:
                if scaler:
                    X_ready = pd.DataFrame(scaler.transform(X_input), columns=X_input.columns)
                else:
                    X_ready = X_input
                    
                prob_fraud = model.predict_proba(X_ready)[0][1]
                is_fraud = prob_fraud > threshold

                # --- DYNAMIC LIVE ROI CALCULATION ---
                txn_id = st.session_state['current_txn_id']
                actual_type = st.session_state['txn_type']
                
                # Safely extract the original dollar amount (reversing the log1p transform)
                log_col = 'Log_Amount' if 'Log_Amount' in X_input.columns else ('log_amount' if 'log_amount' in X_input.columns else None)
                actual_amount = np.expm1(X_input[log_col].values[0]) if log_col else 0.0
                
                impact = 0.0
                is_fp = 0
                
                if is_fraud and actual_type == 'fraud':
                    impact = actual_amount  # Caught Fraud -> Saved Money!
                elif is_fraud and actual_type == 'normal':
                    impact = -5.0           # False Alarm -> Customer Friction Cost
                    is_fp = 1
                elif not is_fraud and actual_type == 'fraud':
                    impact = -actual_amount # Missed Fraud -> Hacker Stole Money!
                    
                # Save it to the memory log (Allows threshold slider to dynamically recalculate!)
                st.session_state.processed_txns[txn_id] = {'impact': impact, 'is_fp': is_fp}
                # ------------------------------------

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 🧠 2. AI VERDICT")
                
                if is_fraud:
                    if actual_type == 'fraud':
                        sub_text = "✅ <b>True Positive:</b> Transaction intercepted successfully. Hackers stopped!"
                        bg_color = "rgba(16, 185, 129, 0.1)"
                        border_color = "rgba(16, 185, 129, 0.4)"
                    elif actual_type == 'normal':
                        sub_text = "⚠️ <b>FALSE ALARM (False Positive):</b> This was a normal transaction. Incurred $5.00 friction cost."
                        bg_color = "rgba(245, 158, 11, 0.1)"
                        border_color = "rgba(245, 158, 11, 0.4)"
                    else:
                        sub_text = "🔍 <b>MANUAL DATA ALERT:</b> Unknown ground truth. Model suspects fraud."
                        bg_color = "rgba(6, 182, 212, 0.1)"
                        border_color = "rgba(6, 182, 212, 0.4)"
                        
                    st.markdown(f"""
                    <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.4); padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 0 20px rgba(244,63,94,0.1);">
                        <h2 style="color: #F43F5E; margin-bottom: 5px; text-shadow: 0 0 15px rgba(244,63,94,0.5);">🚨 FRAUD DETECTED</h2>
                        <p style="font-size: 1.2rem; color: white;">Risk Score: <b style="color: #F43F5E;">{prob_fraud:.1%}</b></p>
                        <div style="background: {bg_color}; border: 1px solid {border_color}; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 0.9rem; color: #f8fafc;">
                            {sub_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if actual_type == 'normal':
                        sub_text = "✅ <b>True Negative:</b> Traffic approved. Normal transaction successfully processed."
                        bg_color = "rgba(16, 185, 129, 0.1)" 
                        border_color = "rgba(16, 185, 129, 0.4)"
                    elif actual_type == 'fraud':
                        sub_text = f"❌ <b>MISSED FRAUD (False Negative):</b> This was an attack! The model's risk score ({prob_fraud:.1%}) fell below your threshold of {threshold:.1%}. Try lowering the Anomaly Threshold slider."
                        bg_color = "rgba(244, 63, 94, 0.2)"
                        border_color = "rgba(244, 63, 94, 0.5)"
                    else:
                        sub_text = "✅ <b>MANUAL DATA SAFE:</b> Unknown ground truth. Model approves transaction."
                        bg_color = "rgba(6, 182, 212, 0.1)"
                        border_color = "rgba(6, 182, 212, 0.4)"

                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.4); padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 0 20px rgba(16,185,129,0.1);">
                        <h2 style="color: #10B981; margin-bottom: 5px; text-shadow: 0 0 15px rgba(16,185,129,0.5);">✅ TRANSACTION SAFE</h2>
                        <p style="font-size: 1.2rem; color: white;">Safety Score: <b style="color: #10B981;">{(1-prob_fraud):.1%}</b></p>
                        <div style="background: {bg_color}; border: 1px solid {border_color}; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 0.9rem; color: #f8fafc;">
                            {sub_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # --- RENDER TOP METRICS NOW THAT ROI IS CALCULATED ---
    with metrics_placeholder.container():
        if mlops_data is not None:
            m1, m2, m3, m4 = st.columns(4)
            roi = mlops_data.get('Net_ROI_USD', mlops_data.get('ROI', 0))
            auprc = mlops_data.get('AUPRC', 0)
            
            live_roi = sum(item['impact'] for item in st.session_state.processed_txns.values())
            live_fp = sum(item['is_fp'] for item in st.session_state.processed_txns.values())
            
            # Formatting fix to ensure Streamlit parses negatives correctly
            delta_str = f"-${abs(impact):,.2f}" if impact < 0 else f"+${impact:,.2f}"
            
            m1.metric("💰 Training ROI (Offline)", f"${roi:,.2f}")
            m2.metric("🟢 Live Session ROI", f"${live_roi:,.2f}", delta=delta_str if 'txn' in st.session_state and impact != 0 else None)
            m3.metric("🛑 Live False Alarms", str(live_fp))
            m4.metric("🎯 AUPRC Score", f"{auprc:.4f}")
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("MLOps Telemetry offline. Run the training pipeline to generate 'experiment_tracking.csv'.")
            st.markdown("<br>", unsafe_allow_html=True)


    with col_viz:
        if 'txn' in st.session_state:
            # --- NEW: RAW DATA STREAM ---
            # --- NEW: TRUE EXPLAINABLE AI (Model-Weighted Threat Matrix) ---
            st.markdown("##### 🔍 AI DECISION LOGIC (Threat Matrix)")
            st.caption("Analyzes the true impact using XGBoost internal feature weights.")
            
            # 1. Get the scaled values for this specific transaction
            feat_values = X_ready.iloc[0]
            
            # 2. Get the global feature importances from the trained XGBoost model
            # (If using Random Forest fallback, it also has feature_importances_)
            if hasattr(model, 'feature_importances_'):
                xgb_weights = model.feature_importances_
            else:
                xgb_weights = np.ones(len(feat_values)) # Fallback if model lacks weights
                
            # 3. Calculate TRUE LOCAL IMPACT: |Z-Score| * XGBoost Weight
            # This prevents loud but useless features (like a noisy V8) from dominating the chart.
            # It only highlights features that the model actually cares about.
            impact_scores = abs(feat_values) * xgb_weights
            
            # 4. Get the names of the top 6 most IMPACTFUL features for this specific row
            top_6_features = impact_scores.sort_values(ascending=False).head(6).index.tolist()
            
            # 5. Extract their actual Z-scores for the visual matrix
            final_impact = [(feat, feat_values[feat]) for feat in top_6_features]
            
            # Build Custom HTML Threat Matrix
            matrix_html = '<div style="display: flex; flex-direction: column; gap: 10px;">'
            for feat, val in final_impact:
                abs_val = abs(val)
                
                # Determine Severity and Colors
                if abs_val > 3.0:
                    level = "CRITICAL"
                    text_color = "#f43f5e" # Rose
                    bar_bg = "#f43f5e"
                    badge_bg = "rgba(244, 63, 94, 0.1)"
                    border_color = "rgba(244, 63, 94, 0.3)"
                elif abs_val > 1.5:
                    level = "ELEVATED"
                    text_color = "#fbbf24" # Amber
                    bar_bg = "#f59e0b"
                    badge_bg = "rgba(245, 158, 11, 0.1)"
                    border_color = "rgba(245, 158, 11, 0.3)"
                else:
                    level = "NORMAL"
                    text_color = "#34d399" # Emerald
                    bar_bg = "#10b981"
                    badge_bg = "rgba(16, 185, 129, 0.1)"
                    border_color = "rgba(16, 185, 129, 0.3)"
                
                # Calculate bar width (cap at 6 standard deviations for the visual scale)
                bar_width = min(100, (abs_val / 6.0) * 100)
                val_str = f"{val:+.2f}σ"
                
                # STRICTLY NO INDENTATION below this line to prevent Markdown code blocks!
                matrix_html += f"""
<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 12px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
    <div style="width: 30%;">
        <div style="color: #06B6D4; font-family: 'Orbitron', sans-serif; font-weight: bold; font-size: 0.9rem; text-transform: uppercase;">{feat}</div>
        <div style="color: #94a3b8; font-size: 0.75rem; font-family: monospace; margin-top: 4px;">Dev: {val_str}</div>
    </div>
    <div style="width: 40%; padding: 0 15px;">
        <div style="height: 6px; width: 100%; background: rgba(30, 41, 59, 0.8); border-radius: 10px; overflow: hidden;">
            <div style="height: 100%; width: {bar_width}%; background: {bar_bg};"></div>
        </div>
    </div>
    <div style="width: 30%; text-align: right;">
        <span style="display: inline-block; padding: 4px 8px; background: {badge_bg}; color: {text_color}; border: 1px solid {border_color}; border-radius: 4px; font-size: 0.7rem; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">
            {level}
        </span>
    </div>
</div>
"""
            matrix_html += '</div>'
            
            st.markdown(matrix_html, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height: 500px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px dashed rgba(6, 182, 212, 0.3); border-radius: 12px; background: rgba(15,23,42,0.5);">
                <i class="fa-solid fa-satellite-dish" style="font-size: 3rem; color: #334155; margin-bottom: 20px;"></i>
                <p style="color: #64748B; font-family: 'Orbitron', sans-serif;">AWAITING DATA INJECTION...</p>
            </div>
            """, unsafe_allow_html=True)