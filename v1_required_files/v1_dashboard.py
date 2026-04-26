import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FraudGuard AI | Security Console", page_icon="🛡️", layout="wide")

# --- ADVANCED FUTURISTIC CSS ---
st.markdown("""
<style>
    /* Global Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(rgba(10, 15, 30, 0.9), rgba(10, 15, 30, 0.9)), 
                    url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=2070');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #F8FAFC;
    }

    /* Typography */
    h1, h2, h3, .hero-title { font-family: 'Orbitron', sans-serif; letter-spacing: 2px; }
    p, label, .stMarkdown { font-family: 'Inter', sans-serif; }

    /* Centered Hero */
    .hero-container {
        text-align: center;
        padding: 60px 0;
        margin-bottom: 40px;
    }
    .hero-title {
        font-size: 4rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.5));
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .glass-panel:hover {
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
    }

    /* Metric Styling */
    .metric-label { color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #FFFFFF; font-family: 'Orbitron', sans-serif; }
    .metric-trend { color: #10B981; font-size: 0.8rem; }

    /* Status Indicators */
    .status-alert {
        color: #F43F5E;
        font-weight: 700;
        font-size: 1.8rem;
        text-shadow: 0 0 10px rgba(244, 63, 94, 0.5);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model = None
    scaler = None
    df = None
    is_raw_data = False
    error = None

    if os.path.exists('fraud_model.joblib'):
        model = joblib.load('fraud_model.joblib')
    else:
        error = "❌ Model file 'fraud_model.joblib' not found."

    if os.path.exists('scaler.joblibv1'):
        scaler = joblib.load('scaler.joblibv1')

    # Load Simulation Data
    for name in ['Test_Datasetv1.csv', 'test_dataset.csv']:
        if os.path.exists(name):
            df = pd.read_csv(name)
            break
    
    if df is not None and 'Amount' in df.columns:
        if df['Amount'].max() > 100:
            is_raw_data = True
        
    return model, scaler, df, is_raw_data, error

model, scaler, dataset, is_raw_data, error = load_resources()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9528/9528169.png", width=80)
    st.title("FRAUDGUARD AI")
    st.markdown("---")
    page = st.radio("INTERFACE SELECT", ["🏠 OVERVIEW", "📡 LIVE MONITOR"])
    st.markdown("---")
    
    if page == "📡 LIVE MONITOR":
        st.subheader("SENSITIVITY")
        threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)
        st.caption("Lowering threshold increases recall (catches more fraud but more false alarms).")
    
    st.markdown("### SYSTEM STATUS")
    st.caption("🟢 Neural Network Active")
    st.caption("🔵 Node: EU-West-1 (Simulated)")

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "🏠 OVERVIEW":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">FRAUDGUARD AI</div>
            <p style="color: #94A3B8; font-size: 1.2rem;">Next-Generation Financial Security Infrastructure</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Information Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📖 THE MISSION", expanded=True):
            st.markdown("""
            Building a defensive layer that identifies patterns invisible to the human eye. 
            FraudGuard V1 analyzes 28 anonymized latent features to protect assets in real-time.
            """)
        with st.expander("📊 DATASET SOURCE"):
            st.markdown("""
            **European Transactions Dataset**
            - Total: 284,807 transactions
            - Fraud Ratio: 0.17% (Extreme Imbalance)
            - Features: V1-V28 (PCA transformed)
            """)

    with col2:
        with st.expander("📉 WHY MISSED CASES?", expanded=True):
            st.markdown("""
            **The Precision-Recall Trade-off**
            Catching 100% of fraud would mean blocking thousands of innocent users. We prioritize a 97% Precision rate to ensure 
            legitimate users are not harassed by false declines.
            """)
        with st.expander("🔐 PCA ANONYMITY"):
            st.markdown("""
            **Why no names or addresses?**
            Principal Component Analysis (PCA) turns sensitive data into mathematical vectors. This ensures the AI 
            learns behavioral patterns without ever seeing private personal information.
            """)

    with col3:
        with st.expander("🛠️ HOW TO OPERATE", expanded=True):
            st.markdown("""
            1. Switch to **Live Monitor**.
            2. Trigger **Normal** or **Attack** vectors.
            3. Observe the **Anomaly Fingerprint**.
            4. Adjust sensitivity to see threshold shifts.
            """)
        with st.expander("🚀 VERSION 2.0 ROADMAP"):
            st.markdown("""
            - XGBoost Gradient Boosting
            - Automated Threshold Tuning
            - Synthetic Data Augmentation
            """)

# --- PAGE 2: LIVE MONITOR ---
elif page == "📡 LIVE MONITOR":
    if error:
        st.error(error)
        st.stop()

    # Top Metrics Row (Global Model Stats)
    m1, m2, m3 = st.columns(3)
    with m1:
        # Precision: Fixed value from your training (97%). It represents the model's global trustworthiness.
        st.markdown('<div class="glass-panel"><p class="metric-label">Model Precision</p><p class="metric-value">97.2%</p><p class="metric-trend">Global Performance Metric</p></div>', unsafe_allow_html=True)
    with m2:
        # Recall: Fixed value from your training (~75%). It represents the model's global catch rate.
        st.markdown('<div class="glass-panel"><p class="metric-label">Fraud Recall</p><p class="metric-value">~75.0%</p><p class="metric-trend">Global Performance Metric</p></div>', unsafe_allow_html=True)
    with m3:
        # Active Node: Simulated "Flavor Text" to make the dashboard look like a real server console.
        st.markdown('<div class="glass-panel"><p class="metric-label">System Status</p><p class="metric-value">ONLINE</p><p class="metric-trend">● Latency: 12ms (Simulated)</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_ctrl, col_viz = st.columns([1, 1.5])

    with col_ctrl:
        st.markdown("### 📥 INGEST TRAFFIC")
        if dataset is not None:
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("SIMULATE NORMAL", use_container_width=True):
                    st.session_state['txn'] = dataset[dataset['Class'] == 0].sample(1)
            with c_btn2:
                if st.button("SIMULATE ATTACK", type="primary", use_container_width=True):
                    st.session_state['txn'] = dataset[dataset['Class'] == 1].sample(1)
        
        if 'txn' in st.session_state:
            row = st.session_state['txn']
            X_input = row.drop(['Class', 'Unnamed: 0'], axis=1, errors='ignore')
            if 'Time' in X_input.columns: X_input = X_input.drop('Time', axis=1)
            
            # Preprocessing
            if is_raw_data and scaler:
                X_ready = pd.DataFrame(scaler.transform(X_input), columns=X_input.columns)
            else:
                X_ready = X_input
            
            # Prediction
            prob_fraud = model.predict_proba(X_ready)[0][1]
            is_fraud = prob_fraud > threshold

            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.caption("ANALYSIS RESULT")
            if is_fraud:
                st.markdown('<p class="status-alert">🚨 FRAUD DETECTED</p>', unsafe_allow_html=True)
                # FIXED: Formatting percentage correctly
                st.markdown(f"**Risk Score:** <span style='color:#F43F5E; font-size:1.5rem;'>{prob_fraud:.1%}</span>", unsafe_allow_html=True)
                st.error("Action Required: Transaction Blocked. Flagged for manual review.")
            else:
                st.markdown('<p class="status-safe" style="color:#10B981; font-size:1.8rem; font-weight:700;">✅ TRANSACTION SAFE</p>', unsafe_allow_html=True)
                # FIXED: Formatting percentage correctly
                st.markdown(f"**Safety Score:** <span style='color:#10B981; font-size:1.5rem;'>{(1-prob_fraud):.1%}</span>", unsafe_allow_html=True)
                st.success("Action: Transaction Approved.")
            
            # Ground Truth Check & Diagnostics
            actual_val = row['Class'].values[0]
            if (is_fraud and actual_val == 1) or (not is_fraud and actual_val == 0):
                st.caption(f"Verified Match (Historical: {'Fraud' if actual_val == 1 else 'Normal'})")
            else:
                st.warning(f"Prediction Mismatch (Historical: {'Fraud' if actual_val == 1 else 'Normal'})")
                
                # --- NEW DIAGNOSIS BLOCK ---
                st.markdown('<div style="background-color: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B; padding: 15px; border-radius: 4px; margin-top: 10px;">', unsafe_allow_html=True)
                st.markdown("#### 🧠 AI DIAGNOSIS: WHY WAS THIS WRONG?")
                
                if actual_val == 1: # False Negative (Missed Fraud)
                    st.markdown(f"""
                    **Type:** False Negative (Missed Detection)
                    - **The Reason:** The AI assigned a risk score of **{prob_fraud:.1%}**.
                    - **The Barrier:** Your alert threshold is set to **{threshold:.1%}**.
                    - **Conclusion:** This transaction looked suspicious, but not suspicious *enough* to cross your current safety line.
                    - **Fix:** Try lowering the **Anomaly Threshold** slider to catch subtle fraud like this.
                    """)
                else: # False Positive (False Alarm)
                    st.markdown(f"""
                    **Type:** False Positive (False Alarm)
                    - **The Reason:** The AI assigned a risk score of **{prob_fraud:.1%}**.
                    - **The Barrier:** Your alert threshold is set to **{threshold:.1%}**.
                    - **Conclusion:** This normal transaction had unusual features (outliers) that triggered a high risk score.
                    - **Fix:** Try raising the **Anomaly Threshold** slider to filter out noise.
                    """)
                st.markdown('</div>', unsafe_allow_html=True)
                # ---------------------------

            st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        if 'txn' in st.session_state:
            st.markdown("### 📊 ANOMALY FINGERPRINT")
            
            # Radar Chart for Feature Inclination
            # We take the top 8 features by absolute value to show "inclination"
            data_dict = X_input.iloc[0].to_dict()
            sorted_feats = sorted(data_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            labels = [x[0] for x in sorted_feats]
            values = [abs(x[1]) for x in sorted_feats]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name='Transaction Profile',
                line_color='#22D3EE',
                fillcolor='rgba(34, 211, 238, 0.3)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, max(values) * 1.1]),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Orbitron'),
                margin=dict(l=40, r=40, t=20, b=20),
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.markdown('<div style="text-align:center; color:#94A3B8; font-size:0.8rem;">Feature vectors shifted from normal distribution are highlighted in the polygon.</div>', unsafe_allow_html=True)

            with st.expander("🔍 RAW DATA STREAM"):
                st.dataframe(X_input)
        else:
            st.markdown("""
            <div style="height: 400px; display: flex; align-items: center; justify-content: center; border: 1px dashed rgba(59, 130, 246, 0.3); border-radius: 12px;">
                <p style="color: #64748B;">WAITING FOR INPUT STREAM...</p>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown('<div style="text-align:center; color:#64748B; font-size:0.8rem;">FRAUDGUARD SECURE CONSOLE V1.0 | ENCRYPTED LINK ACTIVE</div>', unsafe_allow_html=True)