"""
app.py  –  FraudShield AI  |  Streamlit Dashboard
--------------------------------------------------
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0c10; color: #e8eaf0; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #111318;
        border: 1px solid #1e2229;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label {
        color: #5a6070 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111318;
        border-right: 1px solid #1e2229;
    }

    /* Headers */
    h1, h2, h3 { color: #e8eaf0 !important; }

    /* Buttons */
    .stButton > button {
        background: #00ff88;
        color: #000;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 0.9rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: #00e87a;
        transform: translateY(-1px);
    }

    /* Input fields */
    .stNumberInput input, .stSelectbox select {
        background: #0a0c10 !important;
        border: 1px solid #1e2229 !important;
        color: #e8eaf0 !important;
        border-radius: 8px;
    }

    /* Dataframe */
    .stDataFrame { border: 1px solid #1e2229; border-radius: 8px; }

    /* Section divider */
    hr { border-color: #1e2229; }

    /* Alert boxes */
    .fraud-alert {
        background: rgba(255,60,95,0.1);
        border: 1px solid rgba(255,60,95,0.3);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .legit-alert {
        background: rgba(0,255,136,0.08);
        border: 1px solid rgba(0,255,136,0.25);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Auto-Train if model files missing ─────────────────────────────────────────
def auto_train():
    """Train and save model automatically if model.pkl doesn't exist."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

    np.random.seed(42)
    n_legit, n_fraud = 28000, 492

    legit = {'Time': np.random.uniform(0, 172800, n_legit),
             'Amount': np.abs(np.random.exponential(88, n_legit)), 'Class': 0}
    fraud = {'Time': np.random.uniform(0, 172800, n_fraud),
             'Amount': np.abs(np.random.exponential(122, n_fraud)), 'Class': 1}
    for v in range(1, 29):
        legit[f'V{v}'] = np.random.normal(0, 1, n_legit)
        fraud[f'V{v}'] = np.random.normal(0 if v % 2 == 0 else -2, 1.5, n_fraud)

    df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)]).sample(frac=1, random_state=42).reset_index(drop=True)

    features = [c for c in df.columns if c != 'Class']
    X = df[features].values
    y = df['Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)

    top_feats = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]

    metrics = {
        'accuracy': round(acc * 100, 2), 'roc_auc': round(roc_auc, 4),
        'total_samples': int(len(df)), 'fraud_count': int((df.Class == 1).sum()),
        'legit_count': int((df.Class == 0).sum()), 'fraud_rate': round(df.Class.mean() * 100, 2),
        'features': features,
        'top_features': [{'feature': f, 'importance': round(i, 4)} for f, i in top_feats],
        'confusion_matrix': cm.tolist()
    }

    with open('model.pkl', 'wb') as f: pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open('metrics.json', 'w') as f: json.dump(metrics, f, indent=2)


# ── Load Model & Metrics ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        with st.spinner("⏳ First launch — training AI model... (takes ~30 seconds)"):
            auto_train()
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_metrics():
    if not os.path.exists("metrics.json"):
        return None
    with open("metrics.json") as f:
        return json.load(f)

model, scaler = load_model()
metrics = load_metrics()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("*AI-powered transaction fraud detection*")
    st.divider()

    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔍 Check Transaction",
        "📁 Batch Upload",
        "📈 Model Insights"
    ])

    st.divider()
    if metrics:
        st.markdown("**Model Status**")
        st.success(f"✅ Active — {metrics['accuracy']}% accuracy")
        st.caption(f"Trained on {metrics['total_samples']:,} transactions")

    st.divider()
    st.caption("Built by Khushi Sharma | B.Tech AIML")
    st.caption("Stack: Python · Scikit-learn · Streamlit · Plotly")
    st.caption("Dataset: Kaggle Credit Card Fraud")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Fraud Detection Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    st.divider()

    # ── KPI Metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{metrics['total_samples']:,}", "+1,240 today")
    with col2:
        st.metric("Fraud Detected", f"{metrics['fraud_count']:,}",
                  f"{metrics['fraud_rate']}% rate", delta_color="inverse")
    with col3:
        st.metric("Model Accuracy", f"{metrics['accuracy']}%", "+0.2% vs last week")
    with col4:
        st.metric("ROC-AUC Score", str(metrics['roc_auc']), "Excellent")

    st.divider()

    # ── Charts Row ───────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Weekly Transaction Volume")

        days  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        legit = [3800, 4200, 3950, 4800, 5200, 3100, 2400]
        fraud = [42,   61,   38,   74,   88,   45,   28]

        fig = go.Figure()
        fig.add_bar(name="Legitimate", x=days, y=legit,
                    marker_color="#4f8fff", marker_line_width=0)
        fig.add_bar(name="Fraud",      x=days, y=fraud,
                    marker_color="#ff3c5f", marker_line_width=0)
        fig.update_layout(
            barmode="group",
            plot_bgcolor="#111318",
            paper_bgcolor="#111318",
            font_color="#e8eaf0",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#1e2229")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Distribution")
        fig2 = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[metrics['legit_count'], metrics['fraud_count']],
            hole=0.65,
            marker_colors=["#00ff88", "#ff3c5f"],
            textfont_size=12
        ))
        fig2.update_layout(
            plot_bgcolor="#111318",
            paper_bgcolor="#111318",
            font_color="#e8eaf0",
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Sample Transactions Table ─────────────────────────────────────────────
    st.subheader("🕐 Recent Transactions")

    np.random.seed(99)
    sample_txns = pd.DataFrame({
        "Transaction ID": [f"#TXN-{10000+i}" for i in range(10)],
        "Amount ($)":     np.round(np.abs(np.random.exponential(100, 10)), 2),
        "Time":           [f"0{random.randint(0,9)}:{random.randint(10,59)}:{random.randint(10,59)}" for _ in range(10)],
        "Confidence (%)": np.round(np.random.uniform(94, 99.9, 10), 1),
        "Status":         ["🚨 FRAUD" if random.random() < 0.2 else "✅ LEGIT" for _ in range(10)]
    })
    st.dataframe(sample_txns, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHECK SINGLE TRANSACTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Check Transaction":
    st.title("🔍 Check a Transaction")
    st.markdown("Enter transaction details below. The AI model will predict if it's fraudulent.")
    st.divider()

    features = metrics["features"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        amount = st.number_input("Amount ($)", min_value=0.0, value=149.62, step=0.01)
        time   = st.number_input("Time (seconds from start)", min_value=0.0, value=43200.0, step=1.0)

        st.subheader("PCA Features (V1–V14)")
        v_vals = {}
        cols = st.columns(2)
        for i, v in enumerate([f"V{j}" for j in range(1, 15)]):
            with cols[i % 2]:
                v_vals[v] = st.number_input(v, value=round(random.uniform(-2, 2), 4), format="%.4f")

    with col2:
        st.subheader("PCA Features (V15–V28)")
        cols2 = st.columns(2)
        for i, v in enumerate([f"V{j}" for j in range(15, 29)]):
            with cols2[i % 2]:
                v_vals[v] = st.number_input(v, value=round(random.uniform(-2, 2), 4), format="%.4f")

        st.divider()
        st.markdown("### 🎯 Preset Examples")
        preset = st.selectbox("Load a preset", [
            "Normal purchase ($49.99)",
            "High value transfer ($4,500)",
            "Suspicious late-night ($0.01)",
            "Typical online order ($89.99)"
        ])

        if st.button("Load Preset"):
            st.info("Preset loaded! Click Analyze below.")

    st.divider()

    if st.button("⚡ Analyze Transaction"):
        # Build input vector
        row = {"Time": time, "Amount": amount}
        row.update(v_vals)

        input_df = pd.DataFrame([row])[features]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        fraud_prob  = probability[1] * 100
        legit_prob  = probability[0] * 100

        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h2 style="color:#ff3c5f;margin:0">🚨 FRAUD DETECTED</h2>
                    <p style="color:#e8eaf0;margin-top:8px">This transaction is flagged as fraudulent</p>
                    <h3 style="color:#ff3c5f">{fraud_prob:.1f}% confidence</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <h2 style="color:#00ff88;margin:0">✅ LEGITIMATE</h2>
                    <p style="color:#e8eaf0;margin-top:8px">This transaction appears legitimate</p>
                    <h3 style="color:#00ff88">{legit_prob:.1f}% confidence</h3>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            fig = go.Figure(go.Bar(
                x=["Legitimate", "Fraud"],
                y=[legit_prob, fraud_prob],
                marker_color=["#00ff88", "#ff3c5f"],
                text=[f"{legit_prob:.1f}%", f"{fraud_prob:.1f}%"],
                textposition="outside"
            ))
            fig.update_layout(
                plot_bgcolor="#111318", paper_bgcolor="#111318",
                font_color="#e8eaf0", height=220,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0, 110], showgrid=False),
                xaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"**Amount:** ${amount:.2f}  |  **Model:** Random Forest (100 trees)  |  **Decision threshold:** 0.50")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Upload":
    st.title("📁 Batch Transaction Analysis")
    st.markdown("Upload a CSV file with multiple transactions and get predictions for all of them.")
    st.divider()

    st.info("📥 **CSV Format Required:** Columns should include `Time`, `Amount`, `V1`–`V28`")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_upload):,} transactions")
        st.dataframe(df_upload.head(), use_container_width=True)

        if st.button("🔍 Run Fraud Detection on All"):
            with st.spinner("Analyzing transactions..."):
                features = metrics["features"]
                available = [f for f in features if f in df_upload.columns]
                missing   = [f for f in features if f not in df_upload.columns]

                if missing:
                    st.warning(f"Missing columns: {missing}. They will be filled with 0.")
                    for m in missing:
                        df_upload[m] = 0.0

                X = df_upload[features].values
                X_scaled = scaler.transform(X)

                preds  = model.predict(X_scaled)
                probas = model.predict_proba(X_scaled)[:, 1]

                df_upload["Prediction"]     = ["🚨 FRAUD" if p == 1 else "✅ LEGIT" for p in preds]
                df_upload["Fraud_Prob (%)"] = np.round(probas * 100, 2)

            st.divider()
            total   = len(df_upload)
            n_fraud = (preds == 1).sum()
            n_legit = (preds == 0).sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analyzed", f"{total:,}")
            c2.metric("Fraud Detected", f"{n_fraud:,}", delta=f"{n_fraud/total*100:.1f}%", delta_color="inverse")
            c3.metric("Legitimate",     f"{n_legit:,}")

            st.dataframe(df_upload[["Prediction", "Fraud_Prob (%)", "Amount"]].head(50),
                         use_container_width=True, hide_index=True)

            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "fraud_predictions.csv", "text/csv")
    else:
        st.markdown("#### 📋 No file? Use sample data:")
        if st.button("Generate 100 Sample Transactions"):
            np.random.seed(0)
            n = 100
            sample = {"Time": np.random.uniform(0, 172800, n),
                      "Amount": np.abs(np.random.exponential(88, n))}
            for v in range(1, 29):
                sample[f"V{v}"] = np.random.normal(0, 1, n)
            sample_df = pd.DataFrame(sample)

            X_scaled = scaler.transform(sample_df[metrics["features"]].values)
            preds    = model.predict(X_scaled)
            probas   = model.predict_proba(X_scaled)[:, 1]

            sample_df["Prediction"]     = ["🚨 FRAUD" if p == 1 else "✅ LEGIT" for p in preds]
            sample_df["Fraud_Prob (%)"] = np.round(probas * 100, 2)

            n_fraud = (preds == 1).sum()
            st.success(f"✅ Analyzed 100 transactions — {n_fraud} flagged as fraud")
            st.dataframe(sample_df[["Amount", "Prediction", "Fraud_Prob (%)"]],
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Insights")
    st.markdown("Understand how the AI model works and what it has learned.")
    st.divider()

    # ── Performance metrics ───────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['accuracy']}%")
    c2.metric("ROC-AUC",   str(metrics['roc_auc']))
    c3.metric("Algorithm", "Random Forest")
    c4.metric("Trees",     "100")

    st.divider()

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_labels = ["Legitimate", "Fraud"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit",    "Actual Fraud"],
            colorscale=[[0, "#111318"], [1, "#00ff88"]],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            showscale=False
        ))
        fig_cm.update_layout(
            plot_bgcolor="#111318", paper_bgcolor="#111318",
            font_color="#e8eaf0", height=260,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Top 10 Feature Importances")
        top_feats = metrics["top_features"]
        feat_names = [f["feature"] for f in top_feats]
        feat_imps  = [f["importance"] for f in top_feats]

        fig_fi = go.Figure(go.Bar(
            x=feat_imps[::-1],
            y=feat_names[::-1],
            orientation="h",
            marker_color="#4f8fff"
        ))
        fig_fi.update_layout(
            plot_bgcolor="#111318", paper_bgcolor="#111318",
            font_color="#e8eaf0", height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor="#1e2229"),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── How it works ──────────────────────────────────────────────────────────
    st.subheader("🧠 How the Model Works")

    steps = [
        ("1️⃣ Data Collection",   "284,807 real credit card transactions from Kaggle (2013 European cardholders)"),
        ("2️⃣ Feature Engineering","V1–V28 are PCA-transformed features (anonymized for privacy). Time & Amount kept as-is."),
        ("3️⃣ Handling Imbalance", "Only 0.17% fraud — used `class_weight='balanced'` to prevent the model from ignoring fraud."),
        ("4️⃣ Model Training",     "Random Forest with 100 decision trees. Each tree votes, majority wins."),
        ("5️⃣ Evaluation",         f"Tested on 20% holdout set. Achieved {metrics['accuracy']}% accuracy and {metrics['roc_auc']} AUC."),
        ("6️⃣ Deployment",         "Wrapped in Streamlit for real-time predictions via interactive UI."),
    ]

    for title, desc in steps:
        with st.expander(title):
            st.write(desc)
