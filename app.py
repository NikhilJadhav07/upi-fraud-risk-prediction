"""UPIShield — Cost-Aware UPI Fraud Detection Streamlit UI.

UI-only integration for the existing notebook artifacts. Place this file beside:
- upi_transactions_india.csv
- upi_fraud_model.pkl
- feature_names.pkl
- threshold_info.pkl
- scaler.pkl (only required when the saved model is Logistic Regression)

Run: streamlit run app.py
"""
from pathlib import Path
from datetime import datetime, date, time
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------- App setup ----------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "upi_transactions_india.csv"
MODEL_PATH = ROOT / "upi_fraud_model.pkl"
FEATURES_PATH = ROOT / "feature_names.pkl"
THRESHOLD_PATH = ROOT / "threshold_info.pkl"
SCALER_PATH = ROOT / "scaler.pkl"

st.set_page_config(
    page_title="UPIShield | Transaction Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette: midnight navy, electric cyan, mint, amber and coral.
BG = "#07111F"
PANEL = "#101E30"
PANEL_2 = "#13253A"
TEXT = "#EAF2FF"
MUTED = "#9BB0C9"
CYAN = "#42D6E8"
MINT = "#5CE1B9"
AMBER = "#FFCA70"
CORAL = "#FF6F7D"
PURPLE = "#9B8CFF"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root {{ color-scheme: dark; }}
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{ background: radial-gradient(ellipse at 10% 0%, #12304a 0%, {BG} 42%, #050b14 100%); color: {TEXT}; }}
    [data-testid="stHeader"] {{ background: rgba(7,17,31,.78); }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0b1929 0%, #08121f 100%); border-right: 1px solid rgba(66,214,232,.16); }}
    [data-testid="stSidebar"] * {{ color: {TEXT}; }}
    h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.025em; color: {TEXT}; }}
    h1 {{ font-size: clamp(2rem, 3.5vw, 3.15rem) !important; }}
    p, label, [data-testid="stMarkdownContainer"] {{ color: {TEXT}; }}
    .hero {{ padding: 1.5rem 1.65rem; border: 1px solid rgba(66,214,232,.24); border-radius: 24px;
      background: linear-gradient(120deg, rgba(20,48,72,.96), rgba(12,27,44,.90) 58%, rgba(17,43,55,.84));
      box-shadow: 0 18px 55px rgba(0,0,0,.22); position: relative; overflow: hidden; margin-bottom: 1.1rem; }}
    .hero:after {{ content: '↗'; position:absolute; right:2rem; top:.3rem; font-size:8rem; font-family:'Space Grotesk'; color:rgba(66,214,232,.08); transform:rotate(-12deg); }}
    .eyebrow {{ color:{CYAN}; text-transform:uppercase; font-size:.72rem; letter-spacing:.18em; font-weight:700; margin-bottom:.4rem; }}
    .hero-title {{ font-family:'Space Grotesk',sans-serif; font-size:2.25rem; font-weight:700; line-height:1.08; margin:0 0 .55rem; }}
    .hero-sub {{ color:{MUTED}; max-width:760px; font-size:.98rem; line-height:1.6; }}
    .pill {{ display:inline-block; border:1px solid rgba(92,225,185,.3); color:{MINT}; background:rgba(92,225,185,.08); padding:.28rem .65rem; border-radius:999px; font-size:.75rem; font-weight:600; margin-top:.8rem; }}
    .metric-card {{ background:linear-gradient(145deg, rgba(19,37,58,.96), rgba(13,28,45,.96)); border:1px solid rgba(151,178,210,.14); border-radius:18px; padding:1rem 1.05rem; min-height:122px; box-shadow:0 12px 35px rgba(0,0,0,.12); }}
    .metric-label {{ color:{MUTED}; font-size:.78rem; font-weight:600; letter-spacing:.02em; }}
    .metric-value {{ color:{TEXT}; font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; margin:.28rem 0 .15rem; }}
    .metric-foot {{ color:{MUTED}; font-size:.72rem; }}
    .section-head {{ display:flex; align-items:center; gap:.6rem; margin:.8rem 0 .2rem; font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700; }}
    .section-caption {{ color:{MUTED}; font-size:.82rem; margin-bottom:.8rem; }}
    .insight {{ background:rgba(66,214,232,.065); border:1px solid rgba(66,214,232,.18); border-left:3px solid {CYAN}; border-radius:12px; padding:.8rem .95rem; color:#D7E8F8; font-size:.86rem; line-height:1.55; }}
    .risk-box {{ padding:1.1rem 1.2rem; border-radius:18px; border:1px solid rgba(255,255,255,.12); background:linear-gradient(145deg, rgba(19,37,58,.98), rgba(8,20,34,.98)); }}
    .risk-number {{ font-family:'Space Grotesk',sans-serif; font-size:3rem; font-weight:700; line-height:1.1; }}
    .small-muted {{ color:{MUTED}; font-size:.8rem; }}
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{ border-radius:12px; border:1px solid rgba(66,214,232,.35); background:linear-gradient(90deg,#1d9caf,#42d6e8); color:#04111d; font-weight:800; padding:.68rem 1.05rem; min-height:2.7rem; }}
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{ border-color:{MINT}; color:#03111a; filter:brightness(1.08); }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {{ background:#0b1a2a; border-color:#2a425b; border-radius:10px; }}
    [data-testid="stDataFrame"] {{ border:1px solid rgba(151,178,210,.16); border-radius:14px; overflow:hidden; }}
    hr {{ border-color:rgba(151,178,210,.15); }}
    .footer {{ color:{MUTED}; font-size:.73rem; text-align:center; padding:1.3rem 0 .4rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.35rem; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=TEXT, size=12),
    margin=dict(l=18, r=18, t=35, b=18),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def hero(kicker: str, title: str, subtitle: str, badge: str = "COST-AWARE RISK INTELLIGENCE") -> None:
    st.markdown(
        f'''<div class="hero"><div class="eyebrow">{kicker}</div>
        <div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div>
        <span class="pill">● {badge}</span></div>''',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot: str, accent: str = CYAN) -> None:
    st.markdown(
        f'''<div class="metric-card"><div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{accent}">{value}</div>
        <div class="metric-foot">{foot}</div></div>''',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-head">{title}</div><div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def money(value: float) -> str:
    return f"₹{value:,.0f}"


@st.cache_data(show_spinner=False)
def load_transactions(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "timestamp" in data:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if "fraud_flag" in data:
        data["fraud_flag"] = pd.to_numeric(data["fraud_flag"], errors="coerce").fillna(0).astype(int)
    return data


@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [p.name for p in [MODEL_PATH, FEATURES_PATH, THRESHOLD_PATH] if not p.exists()]
    if missing:
        return None, None, None, None, missing
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    threshold_info = joblib.load(THRESHOLD_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, features, threshold_info, scaler, []


def build_model_row(transaction: dict, feature_names: list[str]) -> pd.DataFrame:
    """Match the raw-to-dummy feature logic used in the existing notebook."""
    t = dict(transaction)
    ts = pd.to_datetime(t["timestamp"])
    t["hour"] = int(ts.hour)
    t["day_of_week"] = int(ts.weekday())
    t["month"] = int(ts.month)
    t["is_weekend"] = int(ts.weekday() >= 5)
    t["is_night"] = int(ts.hour <= 5)
    avg_amount = max(float(t["avg_txn_amount_30d"]), 1.0)
    t["amount_to_avg_ratio"] = float(t["amount_inr"]) / avg_amount
    t["is_round_amount"] = int(float(t["amount_inr"]) % 100 == 0 and float(t["amount_inr"]) >= 1000)

    numeric_inputs = [
        "amount_inr", "is_new_device", "is_new_payee", "sender_account_age_days",
        "txn_count_24h", "failed_attempts_24h", "avg_txn_amount_30d", "hour",
        "day_of_week", "month", "is_weekend", "is_night", "amount_to_avg_ratio", "is_round_amount",
    ]
    categorical_inputs = [
        "transaction_type", "merchant_category", "sender_state", "sender_bank",
        "receiver_bank", "sender_age_group", "device_type", "network_type",
    ]
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for col in numeric_inputs:
        if col in row.columns:
            row.at[0, col] = float(t[col])
    for col in categorical_inputs:
        dummy = f"{col}_{t[col]}"
        if dummy in row.columns:
            row.at[0, dummy] = 1.0
    return row


def score_transaction(transaction: dict, model, feature_names, threshold_info, scaler):
    row = build_model_row(transaction, feature_names)
    threshold = float(threshold_info.get("threshold", 0.5)) if isinstance(threshold_info, dict) else 0.5
    # Notebook scales only Logistic Regression; model name is read internally, never displayed.
    saved_name = threshold_info.get("model_name", "") if isinstance(threshold_info, dict) else ""
    if str(saved_name).lower() == "logistic regression":
        if scaler is None:
            raise FileNotFoundError("scaler.pkl is required for this saved model. Re-run notebook Step 9 to create it.")
        model_input = scaler.transform(row)
    else:
        model_input = row
    probability = float(model.predict_proba(model_input)[0, 1])
    if probability >= threshold:
        level = "HIGH"
    elif probability >= threshold / 2:
        level = "MEDIUM"
    else:
        level = "LOW"
    return probability, level, threshold


def select_or_default(options, default):
    return default if default in options else options[0]


# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family:Space Grotesk;font-size:1.45rem;font-weight:700'>🛡️ UPI<span style='color:#42D6E8'>Shield</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>TRANSACTION INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview Dashboard", "Transaction Prediction"], index=0, label_visibility="visible")
    st.markdown("---")
    st.markdown("**Project scope**")
    st.caption("A cost-aware fraud-risk prototype for UPI-style transactions.")
    st.markdown("**Data status**")
    st.caption("Synthetic demonstration data · not live banking data")
    st.markdown("---")
    if st.button("↻ Refresh dashboard data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ------------------------------- Load data ----------------------------------
if not DATA_PATH.exists():
    st.error(f"Dataset not found: `{DATA_PATH.name}`. Keep `upi_transactions_india.csv` in the same folder as app.py.")
    st.stop()

try:
    df = load_transactions(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not read the transaction dataset: {exc}")
    st.stop()

required = {"fraud_flag", "amount_inr", "timestamp", "transaction_type"}
if not required.issubset(df.columns):
    st.error("The CSV is missing one or more required columns: fraud_flag, amount_inr, timestamp, transaction_type.")
    st.stop()

# ----------------------------- Overview page -------------------------------
if page == "Overview Dashboard":
    hero(
        "UPI PAYMENT SECURITY / OVERVIEW",
        "See the risk. Understand the pattern.",
        "An executive view of transaction activity, suspicious patterns and potential financial exposure — designed for a clear project demonstration.",
    )

    # Filters change the dashboard visuals and KPIs together.
    with st.container(border=True):
        f1, f2, f3 = st.columns([1.15, 1.15, 1.3])
        type_options = ["All transaction types"] + sorted(df["transaction_type"].dropna().astype(str).unique().tolist())
        selected_type = f1.selectbox("Transaction type", type_options)
        min_date = df["timestamp"].min().date() if df["timestamp"].notna().any() else date(2024, 1, 1)
        max_date = df["timestamp"].max().date() if df["timestamp"].notna().any() else date.today()
        selected_dates = f2.date_input("Transaction date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        risk_view = f3.selectbox("Analysis view", ["All transactions", "Genuine only", "Fraud only"], index=0)

    view = df.copy()
    if selected_type != "All transaction types":
        view = view[view["transaction_type"].astype(str) == selected_type]
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2 and "timestamp" in view:
        start_date, end_date = selected_dates
        view = view[view["timestamp"].dt.date.between(start_date, end_date)]
    if risk_view == "Genuine only":
        view = view[view["fraud_flag"] == 0]
    elif risk_view == "Fraud only":
        view = view[view["fraud_flag"] == 1]

    total = len(view)
    fraud_count = int(view["fraud_flag"].sum()) if total else 0
    genuine_count = total - fraud_count
    fraud_rate = (fraud_count / total * 100) if total else 0.0
    txn_value = float(view["amount_inr"].fillna(0).sum()) if total else 0.0
    fraud_value = float(view.loc[view["fraud_flag"] == 1, "amount_inr"].fillna(0).sum()) if total else 0.0

    m = st.columns(4, gap="medium")
    with m[0]: metric_card("TRANSACTIONS ANALYSED", f"{total:,}", "Rows in current filter", CYAN)
    with m[1]: metric_card("LABELLED FRAUD CASES", f"{fraud_count:,}", "Synthetic dataset labels", CORAL)
    with m[2]: metric_card("OBSERVED FRAUD RATE", f"{fraud_rate:.2f}%", "Share of filtered transactions", AMBER)
    with m[3]: metric_card("TRANSACTION VALUE", money(txn_value), f"Flagged-label value: {money(fraud_value)}", MINT)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.02, 1.38], gap="large")
    with left:
        section("Transaction mix", "How the selected activity splits between genuine and fraud-labelled records.")
        if total:
            donut = pd.DataFrame({"Status": ["Genuine", "Fraud-labelled"], "Transactions": [genuine_count, fraud_count]})
            fig = px.pie(donut, names="Status", values="Transactions", hole=.72, color="Status",
                         color_discrete_map={"Genuine": MINT, "Fraud-labelled": CORAL})
            fig.update_traces(textposition="outside", texttemplate="%{label}<br>%{percent:.1%}",
                              marker=dict(line=dict(color=BG, width=4)), hovertemplate="%{label}: %{value:,}<extra></extra>")
            fig.update_layout(**PLOT_LAYOUT, height=330, showlegend=False, annotations=[dict(text=f"{total:,}<br><span style='font-size:11px'>transactions</span>", x=.5, y=.5, showarrow=False, font=dict(size=22, color=TEXT))])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No transactions match these filters.")
    with right:
        section("Fraud rate by transaction type", "Rate is more useful than raw count when transaction volumes differ.")
        if total:
            by_type = view.groupby("transaction_type", dropna=False).agg(transactions=("fraud_flag", "size"), fraud_rate=("fraud_flag", "mean")).reset_index()
            by_type["fraud_rate_pct"] = by_type["fraud_rate"] * 100
            by_type = by_type.sort_values("fraud_rate_pct", ascending=True)
            fig = px.bar(by_type, x="fraud_rate_pct", y="transaction_type", orientation="h", text=by_type["fraud_rate_pct"].map(lambda x: f"{x:.2f}%"),
                         color="fraud_rate_pct", color_continuous_scale=[[0, "#237F94"], [.55, CYAN], [1, CORAL]],
                         hover_data={"transactions": True, "fraud_rate_pct": ":.2f", "fraud_rate": False})
            fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
            fig.update_layout(**PLOT_LAYOUT, height=330, xaxis_title="Fraud-labelled transactions (%)", yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No transactions match these filters.")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        section("When does labelled fraud appear?", "Hourly pattern across the selected transactions. This is descriptive, not causal.")
        if total:
            timed = view.copy()
            timed["hour"] = timed["timestamp"].dt.hour
            hourly = timed.groupby("hour").agg(transactions=("fraud_flag", "size"), fraud_rate=("fraud_flag", "mean")).reset_index()
            hourly["fraud_rate_pct"] = hourly["fraud_rate"] * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hourly["hour"], y=hourly["transactions"], name="Transaction volume", marker_color="rgba(66,214,232,.22)", yaxis="y", hovertemplate="Hour %{x}:00<br>%{y:,} transactions<extra></extra>"))
            fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["fraud_rate_pct"], name="Fraud rate", mode="lines+markers", line=dict(color=AMBER, width=3), marker=dict(size=7), yaxis="y2", hovertemplate="Hour %{x}:00<br>Fraud rate %{y:.2f}%<extra></extra>"))
            # PLOT_LAYOUT already defines the horizontal legend; do not pass
            # a second legend keyword to update_layout().
            fig.update_layout(**PLOT_LAYOUT, height=340, xaxis=dict(title="Hour of day", dtick=2, gridcolor="rgba(151,178,210,.08)"),
                              yaxis=dict(title="Transaction volume", gridcolor="rgba(151,178,210,.08)"),
                              yaxis2=dict(title="Fraud rate (%)", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No transactions match these filters.")
    with right:
        section("Behavioural risk signals", "Compare the labelled fraud rate for common transaction signals.")
        signal_rows = []
        for col, label in [("is_new_device", "New device"), ("is_new_payee", "New payee")]:
            if col in view.columns:
                for value, group in view.groupby(col):
                    signal_rows.append({"Signal": label, "Group": "Yes" if int(value) == 1 else "No", "Fraud rate (%)": group["fraud_flag"].mean() * 100, "Transactions": len(group)})
        if signal_rows:
            sig = pd.DataFrame(signal_rows)
            fig = px.bar(sig, x="Signal", y="Fraud rate (%)", color="Group", barmode="group", text=sig["Fraud rate (%)"].map(lambda x: f"{x:.2f}%"),
                         color_discrete_map={"No": "#4A718C", "Yes": CORAL}, hover_data={"Transactions": True})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(**PLOT_LAYOUT, height=340, yaxis_title="Fraud-labelled rate (%)", xaxis_title="", legend_title="Signal value")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Signal columns are not available in this dataset.")

    # Data-led narrative to help explain charts in a faculty presentation.
    st.markdown("### What this view tells us")
    if total:
        top_type = view.groupby("transaction_type")["fraud_flag"].mean().sort_values(ascending=False)
        top_type_name = str(top_type.index[0]) if len(top_type) else "N/A"
        st.markdown(
            f'<div class="insight"><b>Reading the dashboard:</b> The current filters show <b>{total:,}</b> transactions, of which <b>{fraud_count:,}</b> carry the synthetic fraud label. '
            f'The highest observed labelled-fraud rate by transaction type in this filtered view is <b>{top_type_name}</b>. '
            'Use the charts to compare patterns, not to assume that a particular transaction type causes fraud.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div class='footer'>UPIShield · Academic demonstration · Synthetic UPI-style transaction data · Not a live fraud-monitoring system</div>", unsafe_allow_html=True)

# ---------------------------- Prediction page ------------------------------
else:
    hero(
        "UPI PAYMENT SECURITY / LIVE DEMO",
        "Transaction risk assessment",
        "Enter a transaction profile and review its estimated fraud risk. The result is a decision-support signal, not a bank authorization or a guarantee of fraud.",
        "SINGLE-TRANSACTION SCORING",
    )
    model, feature_names, threshold_info, scaler, missing_artifacts = load_artifacts()
    if missing_artifacts:
        st.error("Saved prediction artifacts were not found: " + ", ".join(missing_artifacts))
        st.markdown("Run all cells in your existing notebook through Step 9 first. Keep the generated `.pkl` files in the same folder as `app.py`.")
        st.stop()

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    with st.container(border=True):
        st.markdown("#### Transaction details")
        st.caption("Use realistic values. The current notebook expects these fields for a single-transaction prediction.")
        with st.form("prediction_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                amount = st.number_input("Transaction amount (₹)", min_value=1.0, max_value=1_000_000.0, value=2500.0, step=100.0, help="Amount of the proposed UPI transaction.")
                txn_type = st.selectbox("Transaction type", ["P2P", "P2M", "Bill Payment", "Recharge", "Collect Request"])
                category_options = sorted(df["merchant_category"].dropna().astype(str).unique().tolist()) if "merchant_category" in df else ["Peer Transfer", "Grocery", "Food"]
                suggested_category = "Peer Transfer" if txn_type in ["P2P", "Collect Request"] else ("Utilities" if txn_type == "Bill Payment" else ("Mobile Recharge" if txn_type == "Recharge" else "Grocery"))
                category = st.selectbox("Merchant / payment category", category_options, index=category_options.index(suggested_category) if suggested_category in category_options else 0)
                tx_date = st.date_input("Transaction date", value=date.today())
                tx_time = st.time_input("Transaction time", value=time(20, 10))
            with c2:
                is_new_device = st.toggle("New device", value=False, help="Has the sender's device been seen before?")
                is_new_payee = st.toggle("New payee", value=True, help="Is this the first payment to this recipient?")
                device_options = sorted(df["device_type"].dropna().astype(str).unique().tolist()) if "device_type" in df else ["Android", "iOS", "Web"]
                device = st.selectbox("Device type", device_options, index=device_options.index("Android") if "Android" in device_options else 0)
                network_options = sorted(df["network_type"].dropna().astype(str).unique().tolist()) if "network_type" in df else ["4G", "5G", "WiFi"]
                network = st.selectbox("Network type", network_options, index=network_options.index("4G") if "4G" in network_options else 0)
            with c3:
                account_age = st.number_input("Account age (days)", min_value=1, max_value=15000, value=720, step=30)
                txn_count = st.number_input("Transactions in last 24h", min_value=0, max_value=100, value=3, step=1)
                failed_attempts = st.number_input("Failed attempts in last 24h", min_value=0, max_value=30, value=0, step=1)
                avg_amount = st.number_input("Average transaction amount (last 30 days, ₹)", min_value=1.0, max_value=1_000_000.0, value=1800.0, step=100.0)
            st.markdown("#### Sender profile")
            p1, p2, p3, p4 = st.columns(4)
            state_options = sorted(df["sender_state"].dropna().astype(str).unique().tolist()) if "sender_state" in df else ["Maharashtra", "Karnataka"]
            sender_state = p1.selectbox("State", state_options, index=state_options.index("Maharashtra") if "Maharashtra" in state_options else 0)
            sender_bank_options = sorted(df["sender_bank"].dropna().astype(str).unique().tolist()) if "sender_bank" in df else ["HDFC Bank", "State Bank of India"]
            sender_bank = p2.selectbox("Sender bank", sender_bank_options, index=sender_bank_options.index("HDFC Bank") if "HDFC Bank" in sender_bank_options else 0)
            receiver_bank_options = sorted(df["receiver_bank"].dropna().astype(str).unique().tolist()) if "receiver_bank" in df else ["State Bank of India", "ICICI Bank"]
            receiver_bank = p3.selectbox("Receiver bank", receiver_bank_options, index=receiver_bank_options.index("State Bank of India") if "State Bank of India" in receiver_bank_options else 0)
            age_options = sorted(df["sender_age_group"].dropna().astype(str).unique().tolist()) if "sender_age_group" in df else ["18-25", "26-35", "36-45", "46-60", "60+"]
            age_group = p4.selectbox("Sender age group", age_options, index=age_options.index("26-35") if "26-35" in age_options else 0)
            submitted = st.form_submit_button("🛡️ Predict transaction risk", use_container_width=True)

    if submitted:
        transaction = {
            "timestamp": datetime.combine(tx_date, tx_time).isoformat(sep=" "),
            "transaction_type": txn_type,
            "merchant_category": category,
            "amount_inr": float(amount),
            "sender_state": sender_state,
            "sender_bank": sender_bank,
            "receiver_bank": receiver_bank,
            "sender_age_group": age_group,
            "device_type": device,
            "network_type": network,
            "is_new_device": int(is_new_device),
            "is_new_payee": int(is_new_payee),
            "sender_account_age_days": float(account_age),
            "txn_count_24h": int(txn_count),
            "failed_attempts_24h": int(failed_attempts),
            "avg_txn_amount_30d": float(avg_amount),
        }
        try:
            probability, level, threshold = score_transaction(transaction, model, feature_names, threshold_info, scaler)
            st.session_state.last_prediction = {"transaction": transaction, "probability": probability, "level": level, "threshold": threshold}
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

    result = st.session_state.last_prediction
    if result:
        probability = result["probability"]
        level = result["level"]
        threshold = result["threshold"]
        accent = CORAL if level == "HIGH" else AMBER if level == "MEDIUM" else MINT
        st.markdown("### Assessment result")
        r1, r2 = st.columns([1.05, 1.2], gap="large")
        with r1:
            st.markdown(
                f'''<div class="risk-box"><div class="eyebrow">ESTIMATED FRAUD PROBABILITY</div>
                <div class="risk-number" style="color:{accent}">{probability:.1%}</div>
                <div style="margin:.45rem 0 .8rem"><span style="background:{accent}20;color:{accent};border:1px solid {accent}66;border-radius:999px;padding:.35rem .75rem;font-weight:800;letter-spacing:.08em">{level} RISK</span></div>
                <div class="small-muted">Decision threshold: {threshold:.2%}. Risk band follows the threshold policy saved by your notebook.</div></div>''',
                unsafe_allow_html=True,
            )
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, number={"suffix": "%", "font": {"size": 26, "color": TEXT}},
                gauge={"axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                       "bar": {"color": accent, "thickness": .28}, "bgcolor": PANEL,
                       "steps": [{"range": [0, 10], "color": "#173C3B"}, {"range": [10, 35], "color": "#4C4229"}, {"range": [35, 100], "color": "#4B2735"}],
                       "threshold": {"line": {"color": CYAN, "width": 3}, "thickness": .8, "value": threshold * 100}}))
            gauge.update_layout(**PLOT_LAYOUT, height=245, margin=dict(l=25, r=25, t=20, b=10))
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        with r2:
            t = result["transaction"]
            section("Transaction summary", "The inputs used for this assessment.")
            summary = pd.DataFrame([
                ["Amount", money(t["amount_inr"])], ["Payment type", t["transaction_type"]],
                ["Payment category", t["merchant_category"]], ["New device", "Yes" if t["is_new_device"] else "No"],
                ["New payee", "Yes" if t["is_new_payee"] else "No"], ["Failed attempts (24h)", str(t["failed_attempts_24h"])],
                ["Account age", f'{t["sender_account_age_days"]:,.0f} days'], ["Transactions (24h)", str(t["txn_count_24h"])],
                ["Amount vs. 30-day average", f'{t["amount_inr"] / max(t["avg_txn_amount_30d"], 1):.1f}×'],
            ], columns=["Input", "Value"])
            st.dataframe(summary, hide_index=True, use_container_width=True, height=345)
            if level == "HIGH":
                message = "The score is above the saved high-risk threshold. In a real system, this could trigger additional verification or manual review."
            elif level == "MEDIUM":
                message = "The score is near the decision boundary. A real payment system could request an additional verification step, based on its policy."
            else:
                message = "The score is below the current alert band. Low risk does not mean zero risk; normal monitoring should still apply."
            st.markdown(f'<div class="insight"><b>How to interpret this:</b> {message}</div>', unsafe_allow_html=True)
        with st.expander("How the risk band is assigned"):
            st.write("The app converts the saved model's probability into a risk band using the threshold saved from the existing notebook. The model and training workflow have not been changed by this UI file.")
            st.write("The risk probability is an estimate from a model trained on synthetic demonstration data. It should not be used to approve, block or investigate real payments.")

    st.markdown("<div class='footer'>UPIShield · Prediction demonstration · Model artifact is loaded locally · Synthetic-data prototype only</div>", unsafe_allow_html=True)
