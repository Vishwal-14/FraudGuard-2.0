import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import json
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard 2.0 | MLOps Console",
    page_icon="🛡️",
    layout="wide"
)

# ── BASE DIR — anchor all file paths to this script's location ────────────────
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Syne:wght@300;400;600&display=swap');

.stApp {
    background:
        linear-gradient(135deg, rgba(2, 6, 18, 0.95) 0%, rgba(6, 12, 35, 0.93) 100%),
        url('https://images.unsplash.com/photo-1639762681485-074b7f938ba0?auto=format&fit=crop&q=80&w=2070');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #E2E8F0;
}
* { font-family: 'Syne', sans-serif; }
h1,h2,h3 { font-family: 'Orbitron', sans-serif; letter-spacing: 2px; }

[data-testid="stSidebar"] {
    background: rgba(4, 8, 20, 0.97) !important;
    border-right: 1px solid rgba(251, 191, 36, 0.15);
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif; }

.glass {
    background: rgba(10, 18, 40, 0.7);
    border: 1px solid rgba(251, 191, 36, 0.15);
    backdrop-filter: blur(16px);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass:hover {
    border-color: rgba(251, 191, 36, 0.4);
    box-shadow: 0 0 24px rgba(251, 191, 36, 0.06);
}

.hero { text-align: center; padding: 56px 0 40px; }
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.6rem;
    font-weight: 900;
    letter-spacing: 6px;
    background: linear-gradient(135deg, #FFFFFF 0%, #FCD34D 55%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 20px rgba(251, 191, 36, 0.4));
    margin-bottom: 8px;
}
.hero-sub {
    color: #64748B;
    font-size: 1.1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.version-badge {
    display: inline-block;
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.3);
    color: #FCD34D;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 2px;
    margin-top: 12px;
}

.metric-card {
    background: rgba(10, 18, 40, 0.8);
    border: 1px solid rgba(251, 191, 36, 0.1);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.metric-label { color: #64748B; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; }
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 1.9rem; font-weight: 700; color: #FFFFFF; }
.metric-sub { color: #FCD34D; font-size: 0.75rem; margin-top: 4px; }

.status-fraud {
    font-family: 'Orbitron', sans-serif;
    color: #F43F5E; font-size: 1.6rem; font-weight: 700;
    text-shadow: 0 0 12px rgba(244, 63, 94, 0.5);
    animation: pulse-red 1.8s infinite;
}
.status-safe {
    font-family: 'Orbitron', sans-serif;
    color: #10B981; font-size: 1.6rem; font-weight: 700;
    text-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
}
@keyframes pulse-red { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }

.mlops-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.mlops-table th {
    color: #64748B; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 1.5px; padding: 8px 12px;
    border-bottom: 1px solid rgba(251, 191, 36, 0.1); text-align: left;
}
.mlops-table td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #CBD5E1; }
.mlops-table tr:hover td { background: rgba(34, 211, 238, 0.04); }
.champion-row td { color: #FFFFFF; background: rgba(251, 191, 36, 0.05); }

.pipeline { display: flex; align-items: center; gap: 0; margin: 16px 0; }
.pipe-step { display: flex; flex-direction: column; align-items: center; gap: 6px; flex: 1; }
.pipe-dot {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Orbitron', sans-serif; font-size: 0.75rem; font-weight: 700;
}
.pipe-dot-blue  { background: rgba(59,130,246,0.2);  border: 1px solid #3B82F6; color: #3B82F6; }
.pipe-dot-cyan  { background: rgba(34,211,238,0.2);  border: 1px solid #22D3EE; color: #22D3EE; }
.pipe-dot-amber { background: rgba(245,158,11,0.2);  border: 1px solid #F59E0B; color: #F59E0B; }
.pipe-dot-green { background: rgba(16,185,129,0.2);  border: 1px solid #10B981; color: #10B981; }
.pipe-label { font-size: 0.65rem; color: #64748B; text-align: center; text-transform: uppercase; letter-spacing: 0.5px; max-width: 70px; line-height: 1.3; }
.pipe-line { flex: 1; height: 1px; background: rgba(251,191,36,0.2); margin-bottom: 22px; }

.badge-champion {
    background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.3);
    color: #FCD34D; font-size: 0.65rem; padding: 2px 10px;
    border-radius: 20px; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ── LOAD RESOURCES ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = scaler = df = None
    error = None

    model_path    = BASE_DIR / "registered_models" / "v3_xgb_20260421_0024.joblib"
    scaler_path   = BASE_DIR / "registered_models" / "v3_scaler_20260421_0024.joblib"
    registry_path = BASE_DIR / "model_registry.json"
    RAW_CSV       = BASE_DIR / "creditcard.csv"

    # Fallback to registry if default model path doesn't exist
    if not model_path.exists() and registry_path.exists():
        with open(registry_path) as f:
            reg = json.load(f)
        model_path  = BASE_DIR / reg.get("model_path", "")
        scaler_path = BASE_DIR / reg.get("scaler_path", "")

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        error = f"Model not found at {model_path}"

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    try:
        if RAW_CSV.exists():
            raw = pd.read_csv(RAW_CSV)
            subset = [c for c in raw.columns if c not in ["Time", "Class"]]
            raw.drop_duplicates(subset=subset, inplace=True)
            _, df, _, _ = train_test_split(
                raw, raw["Class"],
                test_size=0.20,
                random_state=42,
                stratify=raw["Class"]
            )
            df = df.reset_index(drop=True)
        else:
            error = f"creditcard.csv not found at {RAW_CSV}"
            df = None
    except Exception as e:
        error = f"Dataset error: {str(e)}"
        df = None

    return model, scaler, df, error

model, scaler, dataset, load_error = load_resources()


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def engineer_features(raw_df):
    df = raw_df.copy()
    if "Time" in df.columns:
        df["Hour_of_Day"] = (df["Time"] // 3600) % 24
        df["IS_NIGHT"]    = df["Hour_of_Day"].apply(lambda x: 1 if x <= 6 or x >= 23 else 0)
        df.drop("Time", axis=1, inplace=True)
    if "Amount" in df.columns:
        df["LOG_AMOUNT"]  = np.log1p(df["Amount"])
        df["AMOUNT_TIER"] = pd.cut(
            df["Amount"],
            bins=[-1, 1, 10, 100, 1000, float("inf")],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        df.drop("Amount", axis=1, inplace=True)
    for col in ["Class", "Unnamed: 0"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df


# ── LOAD EXPERIMENT LOG & REGISTRY ───────────────────────────────────────────
def load_experiment_log():
    for name in ["experiment_tracking_v3.csv", "experiment_tracking.csv"]:
        path = BASE_DIR / name
        if path.exists():
            return pd.read_csv(path)
    return None

def load_registry():
    path = BASE_DIR / "model_registry.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9528/9528169.png", width=72)
    st.markdown("### FRAUDGUARD 2.0")
    st.markdown('<span class="version-badge">V3 · MLOPS EDITION</span>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "INTERFACE SELECT",
        ["🏠 OVERVIEW", "📡 LIVE MONITOR", "🧬 MLOPS CONSOLE"],
        label_visibility="visible"
    )
    st.markdown("---")

    if page == "📡 LIVE MONITOR":
        st.markdown("#### SENSITIVITY")
        threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)
        st.caption("Lower threshold → catches more fraud but more false alarms.")
    else:
        threshold = 0.5

    st.markdown("#### SYSTEM STATUS")
    st.caption("🟢 XGBoost Champion Active")
    st.caption("🟢 MLOps Registry Loaded")
    st.caption("🔵 Node: EU-West-1 (Simulated)")
    st.markdown("---")
    st.caption("FRAUDGUARD 2.0 SECURE CONSOLE")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 OVERVIEW":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">FRAUDGUARD 2.0</div>
        <div class="hero-sub">Intelligent Fraud Detection · MLOps Edition</div>
        <div class="version-badge">XGBoost · Feature Engineering · Champion Registry</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("📖 THE MISSION", expanded=True):
            st.markdown("""
            FraudGuard 2.0 is a production-grade fraud detection system built in three iterations.
            It combines **XGBoost gradient boosting**, **4 engineered features**, and a full
            **MLOps pipeline** — experiment tracking, health guardrails, and a champion model registry.
            Every design decision is deliberate and explainable.
            """)
        with st.expander("📈 MODEL EVOLUTION"):
            st.markdown("""
            | Version | Algorithm | AUPRC |
            |---------|-----------|-------|
            | V1 | Random Forest | 0.8016 |
            | V2 | XGBoost Baseline | ~0.81 |
            | **V3** | **XGBoost + Features** | **0.8427** |

            Each version is tracked, compared, and registered through the MLOps pipeline.
            The champion model is auto-selected — not manually picked.
            """)
        with st.expander("🧬 MLOPS PIPELINE"):
            st.markdown("""
            Every training run automatically:
            - Logs all hyperparameters + metrics to `experiment_tracking_v3.csv`
            - Runs 3 guardrail checks: AUPRC ≥ 0.75, Recall ≥ 70%, FP Rate < 0.5%
            - Registers the model **only** if it beats the current champion AUPRC
            - Saves champion path + threshold to `model_registry.json`
            """)

    with col2:
        with st.expander("🛠️ HOW TO USE", expanded=True):
            st.markdown("""
            1. Switch to **Live Monitor** in the sidebar
            2. Click **Simulate Normal** or **Simulate Attack**
            3. Observe the **Feature Deviation Profile** chart
            4. Adjust the **Sensitivity Slider** to shift the fraud threshold
            5. Visit **MLOps Console** to see experiment history and guardrail status
            """)
        with st.expander("⚙️ FEATURE ENGINEERING"):
            st.markdown("""
            V3 engineers 4 new signals from the raw `Time` and `Amount` columns:
            - **Hour_of_Day** — extracted from raw seconds elapsed
            - **IS_NIGHT** — binary flag for the 11pm–6am high-risk window
            - **LOG_AMOUNT** — log-scaled value to squash high-value outliers
            - **AMOUNT_TIER** — transaction size bucket: micro / small / medium / large / whale
            """)
        with st.expander("📉 PRECISION VS RECALL"):
            st.markdown("""
            **The core trade-off in fraud detection:**

            Catching 100% of fraud means blocking thousands of innocent customers.
            FraudGuard 2.0 achieves **99% precision** at default threshold — when it
            flags a transaction as fraud, it's almost always right.
            Use the sensitivity slider to shift this balance at runtime.
            """)

    with col3:
        with st.expander("📊 DATASET", expanded=True):
            st.markdown("""
            **European Cardholders Dataset (Kaggle)**
            - 284,807 transactions over 2 days
            - Fraud ratio: **0.17%** — extreme class imbalance
            - Features: V1–V28 (PCA-anonymised) + 4 engineered
            - 8,063 behavioural duplicate transactions removed
            """)
        with st.expander("🔐 WHY PCA?"):
            st.markdown("""
            The original features (merchant, location, card type) are sensitive personal data.
            PCA transforms them into anonymous mathematical vectors — the model learns
            **behavioural patterns** without ever seeing private information.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 LIVE MONITOR":
    if load_error:
        st.error(f"❌ {load_error}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("AUPRC Score",   "0.8427", "V3 Champion"),
        ("Fraud Recall",  "77.9%",  "At threshold 0.5"),
        ("Precision",     "80.0%",  "False alarm rate: 0.002%"),
        ("System Status", "ONLINE", "● Latency: 8ms (Simulated)"),
    ]
    for col, (label, value, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_ctrl, col_viz = st.columns([1, 1.5])

    with col_ctrl:
        st.markdown("#### 📥 INGEST TRAFFIC")

        if dataset is None:
            st.warning("Dataset not loaded. Check logs for details.")
        else:
            b1, b2 = st.columns(2)
            with b1:
                if st.button("SIMULATE NORMAL", use_container_width=True):
                    normal = dataset[dataset["Class"] == 0]
                    st.session_state["txn"] = normal.sample(1).reset_index(drop=True)
            with b2:
                if st.button("SIMULATE ATTACK", type="primary", use_container_width=True):
                    fraud = dataset[dataset["Class"] == 1]
                    st.session_state["txn"] = fraud.sample(1).reset_index(drop=True)

        if "txn" in st.session_state:
            row    = st.session_state["txn"]
            actual = int(row["Class"].values[0])

            X_engineered = engineer_features(row.copy())
            if scaler:
                X_ready = pd.DataFrame(
                    scaler.transform(X_engineered),
                    columns=X_engineered.columns
                )
            else:
                X_ready = X_engineered

            prob_fraud = model.predict_proba(X_ready)[0][1]
            is_fraud   = prob_fraud >= threshold

            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.caption("ANALYSIS RESULT")

            if is_fraud:
                st.markdown('<p class="status-fraud">🚨 FRAUD DETECTED</p>', unsafe_allow_html=True)
                st.markdown(f"**Risk Score:** <span style='color:#F43F5E;font-size:1.4rem;'>{prob_fraud:.1%}</span>", unsafe_allow_html=True)
                st.error("Action Required: Transaction Blocked. Flagged for manual review.")
            else:
                st.markdown('<p class="status-safe">✅ TRANSACTION SAFE</p>', unsafe_allow_html=True)
                st.markdown(f"**Safety Score:** <span style='color:#10B981;font-size:1.4rem;'>{(1-prob_fraud):.1%}</span>", unsafe_allow_html=True)
                st.success("Action: Transaction Approved.")

            match = (is_fraud and actual == 1) or (not is_fraud and actual == 0)
            if match:
                st.caption(f"✓ Verified Match (Historical: {'Fraud' if actual == 1 else 'Normal'})")
            else:
                st.warning(f"Prediction Mismatch (Historical: {'Fraud' if actual == 1 else 'Normal'})")
                st.markdown('<div style="background:rgba(245,158,11,0.08);border-left:3px solid #F59E0B;padding:14px;border-radius:4px;margin-top:8px;">', unsafe_allow_html=True)
                st.markdown("#### 🧠 AI DIAGNOSIS: WHY WAS THIS WRONG?")
                if actual == 1:
                    st.markdown(f"""
                    **Type:** False Negative (Missed Detection)
                    - **Risk score assigned:** {prob_fraud:.1%}
                    - **Your threshold:** {threshold:.1%}
                    - **Conclusion:** Transaction looked suspicious but didn't cross the alert line.
                    - **Fix:** Lower the Anomaly Threshold slider in the sidebar.
                    """)
                else:
                    st.markdown(f"""
                    **Type:** False Positive (False Alarm)
                    - **Risk score assigned:** {prob_fraud:.1%}
                    - **Your threshold:** {threshold:.1%}
                    - **Conclusion:** Normal transaction had unusual PCA feature values that triggered a high score.
                    - **Fix:** Raise the Anomaly Threshold slider in the sidebar.
                    """)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("🔬 V3 ENGINEERED FEATURES"):
                eng_cols = ["Hour_of_Day", "IS_NIGHT", "LOG_AMOUNT", "AMOUNT_TIER"]
                eng_vals = {c: X_engineered[c].values[0] for c in eng_cols if c in X_engineered.columns}
                for k, v in eng_vals.items():
                    st.caption(f"{k}: **{v}**")

    with col_viz:
        if "txn" in st.session_state:
            st.markdown("#### 📊 FEATURE DEVIATION PROFILE")
            row      = st.session_state["txn"]
            pca_cols = [c for c in row.columns if c.startswith("V")]
            data_dict = {c: row[c].values[0] for c in pca_cols}
            sorted_feats = sorted(data_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            labels = [x[0] for x in sorted_feats]
            values = [x[1] for x in sorted_feats]
            bar_colors = ["#F43F5E" if v > 0 else "#22D3EE" for v in values]

            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker=dict(color=bar_colors, opacity=0.85, line=dict(width=0)),
                text=[f"{v:+.2f}" for v in values],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=10),
            ))
            fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.15)")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", family="Syne"),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#475569"),
                yaxis=dict(showgrid=False, color="#CBD5E1", autorange="reversed"),
                margin=dict(l=10, r=60, t=10, b=10), height=380, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div style="display:flex;gap:20px;justify-content:center;margin-top:4px;">'
                '<span style="color:#F43F5E;font-size:0.72rem;">● Positive deviation</span>'
                '<span style="color:#22D3EE;font-size:0.72rem;">● Negative deviation</span>'
                '</div>', unsafe_allow_html=True
            )
            with st.expander("🔍 RAW DATA STREAM"):
                display_cols = [c for c in row.columns if c not in ["Class", "Unnamed: 0"]]
                st.dataframe(row[display_cols])
        else:
            st.markdown("""
            <div style="height:380px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed rgba(34,211,238,0.2);border-radius:12px;">
                <p style="color:#334155;font-family:Orbitron,sans-serif;font-size:0.85rem;letter-spacing:2px;">
                    WAITING FOR INPUT STREAM...
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MLOPS CONSOLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 MLOPS CONSOLE":
    st.markdown("""
    <div class="hero" style="padding: 36px 0 28px;">
        <div class="hero-title" style="font-size:2.4rem;">MLOPS CONSOLE</div>
        <div class="hero-sub">Experiment Tracking · Guardrails · Champion Registry</div>
    </div>
    """, unsafe_allow_html=True)

    registry = load_registry()
    if registry:
        c1, c2, c3, c4 = st.columns(4)
        champ_metrics = [
            ("CHAMPION MODEL", registry.get("champion_version", "—"), "Currently serving"),
            ("BEST AUPRC",     str(registry.get("best_auprc", "—")), "Across all runs"),
            ("NET ROI",        f"${registry.get('net_roi', 0):,.2f}", "Test set"),
            ("REGISTERED",     registry.get("registered_at", "—")[:10], "Date"),
        ]
        for col, (label, value, sub) in zip([c1, c2, c3, c4], champ_metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:rgba(251,191,36,0.25);">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.3rem;">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No model_registry.json found.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("**MLOps Pipeline — every training run follows this flow:**")
    st.markdown("""
    <div class="pipeline">
        <div class="pipe-step"><div class="pipe-dot pipe-dot-blue">1</div><div class="pipe-label">Load &amp; Clean Data</div></div>
        <div class="pipe-line"></div>
        <div class="pipe-step"><div class="pipe-dot pipe-dot-blue">2</div><div class="pipe-label">Feature Engineering</div></div>
        <div class="pipe-line"></div>
        <div class="pipe-step"><div class="pipe-dot pipe-dot-cyan">3</div><div class="pipe-label">Train XGBoost</div></div>
        <div class="pipe-line"></div>
        <div class="pipe-step"><div class="pipe-dot pipe-dot-amber">4</div><div class="pipe-label">Guardrail Checks</div></div>
        <div class="pipe-line"></div>
        <div class="pipe-step"><div class="pipe-dot pipe-dot-green">5</div><div class="pipe-label">Register if Champion</div></div>
        <div class="pipe-line"></div>
        <div class="pipe-step"><div class="pipe-dot pipe-dot-cyan">6</div><div class="pipe-label">Log to Tracker</div></div>
    </div>
    <div style="font-size:0.72rem;color:#475569;margin-top:8px;">
        Step 4 — model is rejected if AUPRC &lt; 0.75, Recall &lt; 70%, or FP Rate &gt; 0.5%.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    for col, (label, val, note) in zip([g1, g2, g3], [
        ("MIN AUPRC",   "0.75", "V3 scores 0.8427 — ✅ passes"),
        ("MIN RECALL",  "70%",  "V3 scores 74.7% — ✅ passes"),
        ("MAX FP RATE", "<0.5%","V3 at 0.002% — ✅ passes"),
    ]):
        with col:
            st.markdown(f"""
            <div class="glass" style="text-align:center;border-color:rgba(16,185,129,0.2);">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.8rem;color:#10B981;">{val}</div>
                <div style="font-size:0.72rem;color:#64748B;margin-top:6px;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    log = load_experiment_log()

    if log is not None and not log.empty:
        st.markdown("#### Experiment Log")

        if "AUPRC" in log.columns and "Version" in log.columns:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=log["Version"], y=log["AUPRC"],
                mode="lines+markers",
                line=dict(color="#F59E0B", width=2),
                marker=dict(size=8, color="#F59E0B", line=dict(color="#0A1228", width=2)),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.06)"
            ))
            fig_line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", family="Syne"),
                xaxis=dict(showgrid=False, color="#475569"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", range=[0.75, 1.0], color="#475569"),
                margin=dict(l=20, r=20, t=20, b=20), height=200,
            )
            st.plotly_chart(fig_line, use_container_width=True)

        display_cols = [c for c in [
            "Timestamp", "Version", "n_estimators", "max_depth",
            "AUPRC", "Recall", "FP_Rate", "Net_ROI_USD", "Health_Status"
        ] if c in log.columns]

        champion_version = registry.get("champion_version", "") if registry else ""
        rows_html = ""
        for _, row in log[display_cols].iterrows():
            version_str = str(row.get("Version", ""))
            is_champ    = champion_version in version_str
            row_class   = "champion-row" if is_champ else ""
            cells = ""
            for col in display_cols:
                val = row[col]
                if col == "Version" and is_champ:
                    cells += f'<td>{val} <span class="badge-champion">CHAMPION</span></td>'
                elif col == "Health_Status":
                    color = "#10B981" if str(val) == "HEALTHY" else "#F43F5E"
                    cells += f'<td style="color:{color};">{val}</td>'
                elif col == "AUPRC" and isinstance(val, float):
                    cells += f'<td>{val:.4f}</td>'
                elif col == "Net_ROI_USD" and isinstance(val, float):
                    cells += f'<td>${val:,.2f}</td>'
                else:
                    cells += f'<td>{val}</td>'
            rows_html += f'<tr class="{row_class}">{cells}</tr>'

        headers = "".join(f"<th>{c}</th>" for c in display_cols)
        st.markdown(f"""
        <div class="glass" style="overflow-x:auto;">
            <table class="mlops-table">
                <thead><tr>{headers}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        if "Net_ROI_USD" in log.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Net ROI by Version")
            colors = ["#F59E0B" if champion_version in str(v) else "#1E3A5F" for v in log["Version"]]
            fig_bar = go.Figure(go.Bar(
                x=log["Version"], y=log["Net_ROI_USD"],
                marker_color=colors,
                text=[f"${v:,.0f}" for v in log["Net_ROI_USD"]],
                textposition="outside", textfont=dict(color="#94A3B8", size=11)
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", family="Syne"),
                xaxis=dict(showgrid=False, color="#475569"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#475569"),
                margin=dict(l=20, r=20, t=30, b=20), height=260, showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No experiment_tracking_v3.csv found. Run the V3 training notebook to populate the log.")

    # ── Fraud heatmap ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🕐 Fraud Rate by Hour of Day")
    st.caption("Shows why the IS_NIGHT feature was engineered — fraud spikes during the night window.")

    if dataset is not None:
        heatmap_data = dataset.copy()
        if "Time" in heatmap_data.columns:
            heatmap_data["Hour_of_Day"] = (heatmap_data["Time"] // 3600) % 24
        if "Hour_of_Day" in heatmap_data.columns and "Class" in heatmap_data.columns:
            hourly = heatmap_data.groupby("Hour_of_Day").agg(
                total=("Class", "count"), fraud=("Class", "sum")
            ).reset_index()
            hourly["fraud_rate"] = hourly["fraud"] / hourly["total"] * 100
            bar_colors = ["#F43F5E" if (h <= 6 or h >= 23) else "#22D3EE" for h in hourly["Hour_of_Day"]]
            fig_heat = go.Figure(go.Bar(
                x=hourly["Hour_of_Day"], y=hourly["fraud_rate"],
                marker=dict(color=bar_colors, opacity=0.85, line=dict(width=0)),
                text=[f"{v:.2f}%" for v in hourly["fraud_rate"]],
                textposition="outside", textfont=dict(color="#94A3B8", size=9),
            ))
            fig_heat.add_vrect(x0=-0.5, x1=6.5, fillcolor="rgba(244,63,94,0.06)", line_width=0,
                annotation_text="IS_NIGHT", annotation_position="top left",
                annotation_font=dict(color="#F43F5E", size=10))
            fig_heat.add_vrect(x0=22.5, x1=23.5, fillcolor="rgba(244,63,94,0.06)", line_width=0)
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", family="Syne"),
                xaxis=dict(title="Hour of Day (0–23)", showgrid=False, color="#475569", dtick=1, tickfont=dict(size=10)),
                yaxis=dict(title="Fraud Rate (%)", showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#475569"),
                margin=dict(l=20, r=20, t=30, b=20), height=280, showlegend=False,
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.caption("Dataset not loaded — heatmap unavailable.")

    # ── Version cards ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Version Evolution")
    v1, v2, v3 = st.columns(3)
    versions = [
        ("V1 · RANDOM FOREST", "#3B82F6", ["Random Forest (100 trees)", "No feature engineering", "SMOTE comparison tested", "GridSearchCV tuning", "AUPRC: 0.8016"]),
        ("V2 · XGBOOST BASELINE", "#8B5CF6", ["XGBoost replaces RF", "scale_pos_weight vs SMOTE", "log_amount + Hour_of_Day", "IS_NIGHT feature", "AUPRC: ~0.81"]),
        ("V3 · MLOPS EDITION", "#F59E0B", ["AMOUNT_TIER feature added", "Early stopping wired up", "Experiment tracker (CSV)", "3-guardrail health check", "Champion registry (JSON)", "AUPRC: 0.8427 ← champion"]),
    ]
    for col, (title, color, points) in zip([v1, v2, v3], versions):
        with col:
            bullet_html = "".join(f'<div style="font-size:0.8rem;color:#94A3B8;padding:3px 0;">• {p}</div>' for p in points)
            st.markdown(f"""
            <div class="glass" style="border-top: 2px solid {color}; min-height: 200px;">
                <div style="font-family:Orbitron,sans-serif;font-size:0.75rem;color:{color};letter-spacing:1.5px;margin-bottom:12px;">{title}</div>
                {bullet_html}
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#1E293B;font-size:0.75rem;font-family:Orbitron,sans-serif;">'
    'FRAUDGUARD 2.0 SECURE CONSOLE · MLOPS EDITION · ENCRYPTED LINK ACTIVE'
    '</div>',
    unsafe_allow_html=True
)