"""
Personal Finance Prediction System — Streamlit App
Trains a Random Forest model on the local CSV and serves a premium UI.
"""

import os
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Prediction System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 2rem 3rem !important; max-width: 1300px; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 0 2.5rem;
}
.app-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #64748b;
    font-size: 1rem;
    margin: 0;
}

/* ── Cards ── */
.card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.75rem;
    backdrop-filter: blur(16px);
    height: 100%;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-bottom: 1.25rem;
}

/* ── Inputs ── */
.stNumberInput input, .stTextInput input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.2) !important;
}
label, .stNumberInput label, .stTextInput label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6366f1, #38bdf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.03em !important;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    border: 1.5px solid;
    transition: all 0.3s ease;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; line-height: 1; }
.result-label { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
.result-sub { color: #64748b; font-size: 0.82rem; margin-bottom: 1.2rem; }
.conf-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.conf-fill { height: 100%; border-radius: 99px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }
.conf-pct { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem; }
.result-meta { color: #475569; font-size: 0.78rem; }
.placeholder-card {
    border: 1.5px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 3rem 1.5rem;
    text-align: center;
    color: #334155;
    font-size: 0.9rem;
}

/* ── History table ── */
.history-empty {
    color: #334155;
    text-align: center;
    padding: 2.5rem 1rem;
    font-size: 0.88rem;
}
.hist-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    color: #cbd5e1;
}
.hist-table th {
    text-align: left;
    padding: 0.5rem 0.75rem;
    color: #475569;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.hist-table td { padding: 0.6rem 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.04); }
.badge {
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 0.5rem 0 !important; }

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.4rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ── Train model (cached — runs only once) ────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training model on your data…")
def load_and_train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "Personal_Finance_Dataset.csv"))
    df = df.dropna()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Amount"] = np.log1p(df["Amount"])
    df["Type"] = df["Type"].map({"Income": 1, "Expense": 0}).astype(int)
    df = df.dropna(subset=["Type"])

    X = df[["Amount", "Month", "Day"]]
    y = df["Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    acc = round((model.predict(X_test) == y_test).mean() * 100, 1)
    return model, acc, len(df)


# ── Session state for history ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Load model ────────────────────────────────────────────────────────────────
model, accuracy, total_rows = load_and_train()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
  <h1>💹 Personal Finance Prediction System</h1>
  <p>Predict whether a transaction is <strong>Income</strong> or <strong>Expense</strong> using a trained Random Forest model</p>
</div>
""", unsafe_allow_html=True)

# ── Top stats ──────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("🎯 Model Accuracy", f"{accuracy}%")
m2.metric("📊 Training Records", f"{total_rows:,}")
m3.metric("🌲 Algorithm", "Random Forest")
m4.metric("📅 Predictions Made", len(st.session_state.history))

st.markdown("<br>", unsafe_allow_html=True)

# ── Main layout: LEFT inputs | RIGHT result + history ─────────────────────────
left, right = st.columns([1, 2], gap="large")

# ── LEFT: Input form ──────────────────────────────────────────────────────────
with left:
    st.markdown("<div class='card-title'>📋 Transaction Details</div>", unsafe_allow_html=True)

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1,
        max_value=10_000_000,
        value=5000,
        step=100,
    )

    date_str = st.text_input(
        "Transaction Date (DD-MM-YYYY)",
        value=datetime.date.today().strftime("%d-%m-%Y"),
        placeholder="e.g. 15-05-2026",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  Predict Transaction Type", use_container_width=True)

# ── RIGHT: Result (top) + History (bottom) ────────────────────────────────────
with right:
    res_col, hist_col = st.columns([1, 1], gap="medium")

    # ── Result panel ──────────────────────────────────────────────────────────
    with res_col:
        st.markdown("<div class='card-title'>🔮 Prediction Result</div>", unsafe_allow_html=True)

        if predict_clicked:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%d-%m-%Y")
                month, day = date_obj.month, date_obj.day
                log_amount = np.log1p(amount)

                pred = model.predict([[log_amount, month, day]])[0]
                proba = model.predict_proba([[log_amount, month, day]])[0]
                confidence = round(float(max(proba)) * 100, 1)
                label = "Income" if pred == 1 else "Expense"

                if label == "Income":
                    color, bg, icon = "#10b981", "rgba(16,185,129,0.12)", "💰"
                else:
                    color, bg, icon = "#f43f5e", "rgba(244,63,94,0.12)", "💸"

                st.markdown(f"""
                <div class='result-card' style='border-color:{color}; background:{bg};'>
                  <div class='result-icon'>{icon}</div>
                  <div class='result-label' style='color:{color};'>{label}</div>
                  <div class='result-sub'>Model confidence</div>
                  <div class='conf-wrap'>
                    <div class='conf-fill' style='width:{confidence}%; background:{color};'></div>
                  </div>
                  <div class='conf-pct' style='color:{color};'>{confidence}%</div>
                  <div class='result-meta'>₹{amount:,.0f} &nbsp;·&nbsp; {date_str}</div>
                </div>
                """, unsafe_allow_html=True)

                # Save to history
                st.session_state.history.insert(0, {
                    "Date": date_str,
                    "Amount": f"₹{amount:,.0f}",
                    "Type": label,
                    "Confidence": f"{confidence}%",
                    "_color": color,
                })
                st.session_state.history = st.session_state.history[:10]

            except ValueError:
                st.error("⚠️ Invalid date format. Please use DD-MM-YYYY (e.g. 15-05-2026)")
        else:
            st.markdown("""
            <div class='placeholder-card'>
              <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🔮</div>
              <div>Enter details on the left<br>and click <strong>Predict</strong></div>
            </div>
            """, unsafe_allow_html=True)

    # ── History panel ──────────────────────────────────────────────────────────
    with hist_col:
        st.markdown("<div class='card-title'>🕒 Prediction History</div>", unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown("<div class='history-empty'>No predictions yet.<br>Run one to start logging!</div>", unsafe_allow_html=True)
        else:
            rows_html = ""
            for h in st.session_state.history:
                badge = f"<span class='badge' style='background:{h['_color']}22; color:{h['_color']}; border:1px solid {h['_color']}44'>{h['Type']}</span>"
                rows_html += f"<tr><td>{h['Date']}</td><td>{h['Amount']}</td><td>{badge}</td><td>{h['Confidence']}</td></tr>"

            st.markdown(f"""
            <table class='hist-table'>
              <thead>
                <tr><th>Date</th><th>Amount</th><th>Type</th><th>Conf.</th></tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#1e293b; font-size:0.78rem; padding-bottom:1rem;'>
  Built with Random Forest · Scikit-learn · Streamlit &nbsp;|&nbsp; Personal Finance Prediction System
</div>
""", unsafe_allow_html=True)
