import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, precision_recall_curve, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DefCone  |  Fraud Intelligence Platform",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL THEME / CSS
# ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
COLOR_LEGIT  = "#06d6a0"
COLOR_FRAUD  = "#ef476f"
COLOR_WARN   = "#ffd166"
COLOR_INFO   = "#118ab2"
COLOR_DARK   = "#073b4c"
PALETTE = [COLOR_LEGIT, COLOR_FRAUD, COLOR_WARN, COLOR_INFO, COLOR_DARK, "#8338ec", "#ff6b6b"]

st.markdown("""
<style>
/* ── sidebar ── */
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#073b4c 0%,#0a2540 100%);}
section[data-testid="stSidebar"] * {color:#e8e8e8 !important;}
section[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.12);}
section[data-testid="stSidebar"] .stRadio label:hover {background:rgba(255,255,255,.06);border-radius:6px;}

/* ── metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#ffffff 0%,#f7f9fc 100%);
    border: 1px solid #e2e8f0; border-radius:12px;
    padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,.06);
}
div[data-testid="stMetric"] label {font-size:.78rem!important; text-transform:uppercase; letter-spacing:.04em; color:#64748b!important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {font-size:1.6rem!important; font-weight:700!important; color:#0f172a!important;}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {font-size:.82rem!important;}

/* ── headings ── */
h1,h2,h3 {font-family:'Inter',system-ui,sans-serif!important;}

/* ── tabs ── */
button[data-baseweb="tab"] {font-weight:600!important; font-size:.88rem!important;}

/* ── expander ── */
details[data-testid="stExpander"] {border:1px solid #e2e8f0; border-radius:10px; overflow:hidden;}

/* ── general ── */
.block-container {padding-top:1.2rem; padding-bottom:1rem;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw():
    txn = pd.read_csv("Data/Fraud/transactions_raw.csv", parse_dates=["created_at"])
    users = pd.read_csv("Data/Fraud/users.csv")
    devices = pd.read_csv("Data/Fraud/devices.csv")
    ips = pd.read_csv("Data/Fraud/ips.csv")
    merchants = pd.read_csv("Data/Fraud/merchants.csv")
    cb = pd.read_csv("Data/Fraud/chargebacks.csv", parse_dates=["created_at", "chargeback_at"])
    return txn, users, devices, ips, merchants, cb


@st.cache_data(show_spinner=False)
def build_master():
    txn, users, devices, ips, merchants, cb = load_raw()
    df = txn.copy()
    df = df.merge(users, on="user_id", how="left", suffixes=("", "_u"))
    df = df.merge(devices, on="device_id", how="left")
    df = df.merge(ips, on="ip_id", how="left", suffixes=("", "_ip"))
    df = df.merge(merchants, on="merchant_id", how="left")
    cb_flag = cb[["transaction_id", "chargeback_reason"]].copy()
    cb_flag["has_chargeback"] = 1
    df = df.merge(cb_flag, on="transaction_id", how="left")
    df["has_chargeback"] = df["has_chargeback"].fillna(0).astype(int)
    # derived features
    df["country_mismatch"] = (df["country"] != df["card_country"]).astype(int)
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["day_of_month"] = df["created_at"].dt.day
    df["date"] = df["created_at"].dt.date
    return df


def _fmt(n, prefix="$"):
    if n >= 1_000_000:
        return f"{prefix}{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{prefix}{n/1_000:.1f}K"
    return f"{prefix}{n:,.0f}"

def _pct(n):
    return f"{n:.2f}%"


def _chart(fig, height=370):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


# ──────────────────────────────────────────────────────────────
# 1  EXECUTIVE OVERVIEW
# ──────────────────────────────────────────────────────────────
def page_overview():
    df = build_master()
    txn, users, devices, ips, merchants, cb = load_raw()

    st.markdown("## Executive Summary")
    st.caption("High-level fraud intelligence snapshot — all datasets merged")

    # ── row 1: core KPIs ──
    total_txn = len(df)
    total_amt = df["amount"].sum()
    fraud_mask = df["fraud_label"] == 1
    fraud_n = fraud_mask.sum()
    fraud_amt = df.loc[fraud_mask, "amount"].sum()
    fraud_rate = fraud_n / total_txn * 100
    cb_n = int(df["has_chargeback"].sum())
    cb_rate = cb_n / total_txn * 100
    approved_mask = df["auth_status"] == "Approved"
    approval_rate = approved_mask.sum() / total_txn * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Transactions", f"{total_txn:,}")
    c2.metric("Gross Volume", _fmt(total_amt))
    c3.metric("Fraud Cases", f"{fraud_n:,}", _pct(fraud_rate))
    c4.metric("Fraud Exposure", _fmt(fraud_amt), _pct(fraud_amt / total_amt * 100))
    c5.metric("Chargebacks", f"{cb_n:,}", _pct(cb_rate))
    c6.metric("Approval Rate", _pct(approval_rate))

    st.markdown("")

    # ── row 2: gauges + fraud trend ──
    left, right = st.columns([1, 2])

    with left:
        # Gauge: Fraud rate
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_rate,
            number={"suffix": "%", "font": {"size": 44}},
            delta={"reference": 15, "suffix": "%", "relative": False},
            title={"text": "Fraud Rate", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 30], "ticksuffix": "%"},
                "bar": {"color": COLOR_FRAUD},
                "steps": [
                    {"range": [0, 10], "color": "#d4edda"},
                    {"range": [10, 20], "color": "#fff3cd"},
                    {"range": [20, 30], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.8, "value": fraud_rate},
            },
        ))
        st.plotly_chart(_chart(gauge, 280), use_container_width=True)

        # Fraud vs legit donut
        donut = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[total_txn - fraud_n, fraud_n],
            hole=0.6,
            marker_colors=[COLOR_LEGIT, COLOR_FRAUD],
            textinfo="percent+label",
            textfont_size=12,
        ))
        donut.update_layout(showlegend=False, title_text="Fraud Share", title_x=0.5)
        st.plotly_chart(_chart(donut, 260), use_container_width=True)

    with right:
        # Daily fraud trend
        daily = df.groupby("date").agg(
            total=("fraud_label", "count"),
            fraud=("fraud_label", "sum"),
            volume=("amount", "sum"),
            fraud_volume=("amount", lambda x: x[df.loc[x.index, "fraud_label"] == 1].sum()),
        ).reset_index()
        daily["fraud_rate"] = daily["fraud"] / daily["total"] * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["total"], name="All Txns",
                                 line=dict(color=COLOR_INFO, width=2), fill="tozeroy",
                                 fillcolor="rgba(17,138,178,.08)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["fraud"], name="Fraud Txns",
                                 line=dict(color=COLOR_FRAUD, width=2.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["fraud_rate"], name="Fraud Rate %",
                                 line=dict(color=COLOR_WARN, width=2, dash="dot")), secondary_y=True)
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate %", secondary_y=True, range=[0, daily["fraud_rate"].max() * 1.3])
        fig.update_layout(title="Daily Fraud Trend & Rate", hovermode="x unified")
        st.plotly_chart(_chart(fig, 400), use_container_width=True)

    # ── row 3: top risk tables ──
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🏪 Top Risk Merchants")
        merch = df.groupby("merchant_id").agg(
            txns=("fraud_label", "count"),
            fraud=("fraud_label", "sum"),
            amount=("amount", "sum"),
        ).reset_index()
        merch["fraud_rate"] = (merch["fraud"] / merch["txns"] * 100).round(2)
        merch = merch[merch["txns"] >= 50].sort_values("fraud_rate", ascending=False).head(10)
        merch.columns = ["Merchant ID", "Txns", "Fraud", "Volume ($)", "Fraud Rate %"]
        st.dataframe(merch, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("##### 🌍 Top Risk Countries")
        ctry = df.groupby("country").agg(
            txns=("fraud_label", "count"),
            fraud=("fraud_label", "sum"),
            amount=("amount", "sum"),
        ).reset_index()
        ctry["fraud_rate"] = (ctry["fraud"] / ctry["txns"] * 100).round(2)
        ctry = ctry[ctry["txns"] >= 100].sort_values("fraud_rate", ascending=False).head(10)
        ctry.columns = ["Country", "Txns", "Fraud", "Volume ($)", "Fraud Rate %"]
        st.dataframe(ctry, hide_index=True, use_container_width=True)

    with col3:
        st.markdown("##### 🔁 Chargeback Reasons")
        cbr = df[df["has_chargeback"] == 1]["chargeback_reason"].value_counts().reset_index()
        cbr.columns = ["Reason", "Count"]
        cbr["Share %"] = (cbr["Count"] / cbr["Count"].sum() * 100).round(1)
        st.dataframe(cbr, hide_index=True, use_container_width=True)

    # ── row 4: channel / payment / PSP breakdown ──
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        ch = df.groupby("channel").agg(total=("fraud_label","count"), fraud=("fraud_label","sum")).reset_index()
        ch["Fraud Rate %"] = (ch["fraud"]/ch["total"]*100).round(2)
        fig = px.bar(ch, x="channel", y=["total","fraud"], barmode="group",
                     color_discrete_sequence=[COLOR_INFO, COLOR_FRAUD],
                     title="Channel Breakdown", labels={"value":"Count","variable":"","channel":"Channel"})
        st.plotly_chart(_chart(fig, 320), use_container_width=True)

    with c2:
        pm = df.groupby("payment_method").agg(total=("fraud_label","count"), fraud=("fraud_label","sum")).reset_index()
        pm["fraud_rate"] = (pm["fraud"]/pm["total"]*100).round(2)
        fig = px.bar(pm.sort_values("fraud_rate",ascending=False), x="payment_method", y="fraud_rate",
                     color="fraud_rate", color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                     title="Fraud Rate by Payment Method", labels={"fraud_rate":"Fraud Rate %","payment_method":"Method"})
        st.plotly_chart(_chart(fig, 320), use_container_width=True)

    with c3:
        psp = df.groupby("psp_name").agg(total=("fraud_label","count"), fraud=("fraud_label","sum")).reset_index()
        psp["fraud_rate"] = (psp["fraud"]/psp["total"]*100).round(2)
        fig = px.bar(psp.sort_values("fraud_rate",ascending=False), x="psp_name", y="fraud_rate",
                     color="fraud_rate", color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                     title="Fraud Rate by PSP", labels={"fraud_rate":"Fraud Rate %","psp_name":"PSP"})
        st.plotly_chart(_chart(fig, 320), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# 2  EDA  (tabbed, with global filters)
# ──────────────────────────────────────────────────────────────
def page_eda():
    df = build_master()
    txn, users, devices, ips, merchants, cb = load_raw()

    st.markdown("## Exploratory Data Analysis")
    st.caption("Deep-dive into fraud patterns — use filters to narrow scope")

    # ── global filters ──
    with st.expander("🔎  **Global Filters**  — click to expand", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            amt_range = st.slider("Amount range ($)", 0.0, float(df["amount"].max()), (0.0, float(df["amount"].max())), step=10.0)
        with fc2:
            countries = st.multiselect("Countries", sorted(df["country"].unique()), default=[])
        with fc3:
            channels = st.multiselect("Channels", sorted(df["channel"].unique()), default=[])
        with fc4:
            methods = st.multiselect("Payment Methods", sorted(df["payment_method"].unique()), default=[])

    filt = df.copy()
    filt = filt[(filt["amount"] >= amt_range[0]) & (filt["amount"] <= amt_range[1])]
    if countries:
        filt = filt[filt["country"].isin(countries)]
    if channels:
        filt = filt[filt["channel"].isin(channels)]
    if methods:
        filt = filt[filt["payment_method"].isin(methods)]

    st.info(f"Showing **{len(filt):,}** of {len(df):,} transactions after filters")

    # ── tabs ──
    tab_data, tab_fraud, tab_time, tab_geo, tab_pay, tab_risk = st.tabs([
        "📋 Data Profile", "🚨 Fraud Patterns", "📅 Temporal",
        "🌍 Geographic", "💳 Payments", "⚠️ Risk Scores",
    ])

    # ── TAB: Data Profile ──
    with tab_data:
        dataset_choice = st.selectbox("Dataset", ["Merged (master)", "Transactions", "Users", "Devices", "IPs", "Merchants", "Chargebacks"], key="ds")
        dmap = {"Merged (master)": filt, "Transactions": txn, "Users": users, "Devices": devices, "IPs": ips, "Merchants": merchants, "Chargebacks": cb}
        sel = dmap[dataset_choice]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(sel):,}")
        m2.metric("Columns", len(sel.columns))
        m3.metric("Missing Cells", f"{sel.isnull().sum().sum():,}")
        m4.metric("Duplicate Rows", f"{sel.duplicated().sum():,}")

        st.dataframe(sel.head(15), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Column Types**")
            dtypes = sel.dtypes.value_counts().reset_index()
            dtypes.columns = ["Type", "Count"]
            st.dataframe(dtypes, hide_index=True, use_container_width=True)
        with right:
            st.markdown("**Descriptive Statistics**")
            st.dataframe(sel.describe().T.round(3), use_container_width=True)

        miss = sel.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=True)
        if len(miss):
            fig = px.bar(x=miss.values, y=miss.index, orientation="h", title="Missing Values",
                         labels={"x":"Count","y":"Column"}, color_discrete_sequence=[COLOR_WARN])
            st.plotly_chart(_chart(fig, 300), use_container_width=True)

    # ── TAB: Fraud Patterns ──
    with tab_fraud:
        c1, c2 = st.columns(2)
        with c1:
            grp = filt.groupby(["auth_status","fraud_label"]).size().reset_index(name="count")
            fig = px.bar(grp, x="auth_status", y="count", color="fraud_label",
                         barmode="group", color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="Fraud by Auth Status", labels={"fraud_label":"Is Fraud","count":"Txns","auth_status":"Status"})
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            grp = filt.groupby(["channel","fraud_label"]).size().reset_index(name="count")
            fig = px.bar(grp, x="channel", y="count", color="fraud_label",
                         barmode="group", color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="Fraud by Channel", labels={"fraud_label":"Is Fraud","count":"Txns"})
            st.plotly_chart(_chart(fig), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            pm_f = filt[filt["fraud_label"]==1].groupby("payment_method").size().reset_index(name="count")
            fig = px.pie(pm_f, values="count", names="payment_method", title="Fraud by Payment Method",
                         color_discrete_sequence=PALETTE, hole=0.45)
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            tds = filt.groupby(["is_3ds","fraud_label"]).size().reset_index(name="count")
            tds["is_3ds"] = tds["is_3ds"].map({0:"No 3DS",1:"3DS Enabled"})
            fig = px.bar(tds, x="is_3ds", y="count", color="fraud_label", barmode="group",
                         color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="3DS Impact on Fraud", labels={"is_3ds":"3DS","count":"Txns","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)

        # Chargeback
        c1, c2 = st.columns(2)
        with c1:
            cb_sum = filt.groupby("fraud_label")["has_chargeback"].sum().reset_index()
            cb_sum["label"] = cb_sum["fraud_label"].map({0:"Legitimate",1:"Fraud"})
            fig = px.bar(cb_sum, x="label", y="has_chargeback", color="label",
                         color_discrete_map={"Legitimate":COLOR_LEGIT,"Fraud":COLOR_FRAUD},
                         title="Chargebacks by Txn Type", labels={"has_chargeback":"Chargebacks","label":"Type"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            cbr = filt[filt["has_chargeback"]==1]["chargeback_reason"].value_counts().reset_index()
            cbr.columns = ["reason","count"]
            if len(cbr):
                fig = px.pie(cbr, values="count", names="reason", title="Chargeback Reasons", hole=0.4,
                             color_discrete_sequence=PALETTE)
                st.plotly_chart(_chart(fig), use_container_width=True)

    # ── TAB: Temporal ──
    with tab_time:
        c1, c2 = st.columns(2)
        with c1:
            hr = filt.groupby(["hour","fraud_label"]).size().reset_index(name="count")
            fig = px.line(hr, x="hour", y="count", color="fraud_label",
                          color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                          title="Hourly Volume", labels={"hour":"Hour","count":"Txns","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            dn = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            dw = filt.groupby(["day_of_week","fraud_label"]).size().reset_index(name="count")
            dw["day"] = dw["day_of_week"].map(lambda x: dn[x])
            fig = px.bar(dw, x="day", y="count", color="fraud_label", barmode="group",
                         color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="Day-of-Week Volume", labels={"day":"Day","count":"Txns","fraud_label":"Is Fraud"},
                         category_orders={"day": dn})
            st.plotly_chart(_chart(fig), use_container_width=True)

        # Heatmap – hour vs day
        heat = filt.groupby(["day_of_week","hour"]).agg(fraud_rate=("fraud_label","mean")).reset_index()
        heat["fraud_rate"] = heat["fraud_rate"] * 100
        heat_pivot = heat.pivot(index="day_of_week", columns="hour", values="fraud_rate")
        heat_pivot.index = [dn[i] for i in heat_pivot.index]
        fig = px.imshow(heat_pivot, title="Fraud Rate Heatmap (Hour × Day)", aspect="auto",
                        color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                        labels={"x":"Hour","y":"Day","color":"Fraud %"})
        st.plotly_chart(_chart(fig, 340), use_container_width=True)

        # Daily area
        daily = filt.groupby(["date","fraud_label"]).size().reset_index(name="count")
        fig = px.area(daily, x="date", y="count", color="fraud_label",
                      color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                      title="Daily Transaction Volume", labels={"date":"Date","count":"Txns","fraud_label":"Is Fraud"})
        st.plotly_chart(_chart(fig, 320), use_container_width=True)

    # ── TAB: Geographic ──
    with tab_geo:
        c1, c2 = st.columns(2)
        with c1:
            top = filt["country"].value_counts().head(15).index
            g = filt[filt["country"].isin(top)].groupby(["country","fraud_label"]).size().reset_index(name="count")
            fig = px.bar(g, x="country", y="count", color="fraud_label", barmode="stack",
                         color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="Top 15 Countries by Volume", labels={"count":"Txns","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            cs = filt.groupby("country").agg(txns=("fraud_label","count"), fraud=("fraud_label","sum")).reset_index()
            cs["rate"] = (cs["fraud"]/cs["txns"]*100).round(2)
            cs = cs[cs["txns"]>=100].sort_values("rate",ascending=False).head(15)
            fig = px.bar(cs, x="country", y="rate", color="rate",
                         color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                         title="Fraud Rate by Country (≥100 txns)", labels={"rate":"Fraud %"})
            st.plotly_chart(_chart(fig), use_container_width=True)

        # Country mismatch
        mm = filt.groupby(["country_mismatch","fraud_label"]).size().reset_index(name="count")
        mm["label"] = mm["country_mismatch"].map({0:"Country Match",1:"Country Mismatch"})
        fig = px.bar(mm, x="label", y="count", color="fraud_label", barmode="group",
                     color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                     title="Country vs Card-Country Mismatch", labels={"label":"","count":"Txns","fraud_label":"Is Fraud"})
        st.plotly_chart(_chart(fig, 320), use_container_width=True)

    # ── TAB: Payments ──
    with tab_pay:
        c1, c2 = st.columns(2)
        with c1:
            pp = filt.groupby(["psp_name","fraud_label"]).size().reset_index(name="count")
            fig = px.bar(pp, x="psp_name", y="count", color="fraud_label", barmode="group",
                         color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="PSP Distribution", labels={"psp_name":"PSP","count":"Txns","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            dec = filt[filt["decline_code"].notna()]["decline_code"].value_counts().head(10)
            if len(dec):
                fig = px.bar(x=dec.index.astype(str), y=dec.values, title="Top 10 Decline Codes",
                             labels={"x":"Code","y":"Count"}, color_discrete_sequence=[COLOR_INFO])
                st.plotly_chart(_chart(fig), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            rt = filt.groupby(["retry_count","fraud_label"]).size().reset_index(name="count")
            fig = px.bar(rt, x="retry_count", y="count", color="fraud_label", barmode="group",
                         color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="Retry Count Analysis", labels={"retry_count":"Retries","count":"Txns","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            fig = px.box(filt, x="payment_method", y="amount", color="fraud_label",
                         color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="Amount Distribution by Method", labels={"amount":"$","fraud_label":"Is Fraud"})
            fig.update_yaxes(range=[0, filt["amount"].quantile(0.95)])
            st.plotly_chart(_chart(fig), use_container_width=True)

    # ── TAB: Risk Scores ──
    with tab_risk:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(filt, x="bin_risk", color="fraud_label", nbins=50, barmode="overlay",
                               color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                               title="BIN Risk Distribution", labels={"bin_risk":"BIN Risk","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            fig = px.histogram(filt, x="ip_risk_level", color="fraud_label", nbins=50, barmode="overlay",
                               color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                               title="IP Risk Distribution", labels={"ip_risk_level":"IP Risk","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "merchant_risk" in filt.columns:
                fig = px.histogram(filt, x="merchant_risk", color="fraud_label", nbins=50, barmode="overlay",
                                   color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                                   title="Merchant Risk", labels={"merchant_risk":"Risk","fraud_label":"Is Fraud"})
                st.plotly_chart(_chart(fig), use_container_width=True)
        with c2:
            er = filt.groupby(["email_risk","fraud_label"]).size().reset_index(name="count")
            er["email_risk"] = er["email_risk"].map({0:"Low Risk",1:"High Risk"})
            fig = px.bar(er, x="email_risk", y="count", color="fraud_label", barmode="group",
                         color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                         title="Email Risk", labels={"email_risk":"","count":"Txns","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig), use_container_width=True)

        # User age
        fig = px.histogram(filt, x="user_age_days", color="fraud_label", nbins=50, barmode="overlay",
                           color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                           title="User Account Age", labels={"user_age_days":"Age (days)","fraud_label":"Is Fraud"})
        st.plotly_chart(_chart(fig, 300), use_container_width=True)

        # Correlation
        risk_cols = [c for c in ["bin_risk","ip_risk_level","email_risk","user_age_days",
                                  "merchant_risk","retry_count","amount","is_3ds",
                                  "country_mismatch","is_fraud_device_hint","is_psp_outage_window",
                                  "fraud_label"] if c in filt.columns]
        corr = filt[risk_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Feature Correlation Matrix", aspect="auto")
        st.plotly_chart(_chart(fig, 500), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# 3  FRAUD INVESTIGATION
# ──────────────────────────────────────────────────────────────
def page_investigation():
    df = build_master()

    st.markdown("## Fraud Investigation Console")
    st.caption("Search, filter & drill into individual transactions and user profiles")

    # ── filters ──
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            fraud_only = st.selectbox("Fraud filter", ["All","Fraud only","Legitimate only"], key="inv_fraud")
        with c2:
            amt_min = st.number_input("Min amount ($)", 0.0, value=0.0, step=10.0)
        with c3:
            amt_max = st.number_input("Max amount ($)", 0.0, value=float(df["amount"].max()), step=10.0)
        with c4:
            search_user = st.text_input("User ID", placeholder="e.g. 12345")
        with c5:
            search_txn = st.text_input("Transaction ID", placeholder="e.g. 100")

    view = df.copy()
    if fraud_only == "Fraud only":
        view = view[view["fraud_label"]==1]
    elif fraud_only == "Legitimate only":
        view = view[view["fraud_label"]==0]
    view = view[(view["amount"]>=amt_min)&(view["amount"]<=amt_max)]
    if search_user:
        try:
            view = view[view["user_id"]==int(search_user)]
        except ValueError:
            st.warning("Enter a valid numeric User ID")
    if search_txn:
        try:
            view = view[view["transaction_id"]==int(search_txn)]
        except ValueError:
            st.warning("Enter a valid numeric Transaction ID")

    st.markdown(f"**{len(view):,}** transactions match your criteria")

    display_cols = ["transaction_id","created_at","user_id","amount","country","card_country",
                    "channel","psp_name","payment_method","auth_status","fraud_label",
                    "bin_risk","ip_risk_level","has_chargeback","chargeback_reason"]
    display_cols = [c for c in display_cols if c in view.columns]
    st.dataframe(view[display_cols].head(200).sort_values("created_at", ascending=False),
                 use_container_width=True, hide_index=True)

    # ── download button ──
    csv = view[display_cols].to_csv(index=False)
    st.download_button("⬇️  Download filtered data (CSV)", csv, "fraud_investigation.csv", "text/csv")

    st.markdown("---")

    # ── user profiler ──
    st.markdown("### 👤 User Risk Profiler")
    uid = st.number_input("Enter User ID to profile", min_value=1, value=1, step=1)
    user_txns = df[df["user_id"]==uid]

    if len(user_txns) == 0:
        st.warning("No transactions found for this user.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Transactions", len(user_txns))
        m2.metric("Fraud Txns", int(user_txns["fraud_label"].sum()))
        m3.metric("Total Volume", _fmt(user_txns["amount"].sum()))
        m4.metric("Chargebacks", int(user_txns["has_chargeback"].sum()))
        m5.metric("Avg Amount", _fmt(user_txns["amount"].mean()))

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(user_txns, x="created_at", y="amount", color="fraud_label",
                             color_discrete_map={0:COLOR_LEGIT,1:COLOR_FRAUD},
                             title=f"User {uid} — Transaction Timeline",
                             labels={"created_at":"Date","amount":"$","fraud_label":"Is Fraud"})
            st.plotly_chart(_chart(fig, 300), use_container_width=True)
        with c2:
            cntry = user_txns["country"].value_counts().reset_index()
            cntry.columns = ["Country","Count"]
            fig = px.pie(cntry, values="Count", names="Country", title=f"User {uid} — Countries",
                         hole=0.45, color_discrete_sequence=PALETTE)
            st.plotly_chart(_chart(fig, 300), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# 4  ML MODELING  (enhanced)
# ──────────────────────────────────────────────────────────────
def page_ml():
    df = build_master()

    st.markdown("## ML Modeling & Prediction")
    st.caption("Train, compare & evaluate fraud classifiers")

    # ── Feature engineering (cached) ──
    @st.cache_data(show_spinner=False)
    def prepare_features(_df):
        m = _df.copy()
        cat_cols = ["channel","psp_name","payment_method","auth_status","country","card_country"]
        num_cols = ["amount","bin_risk","ip_risk_level","user_age_days","email_risk",
                    "is_3ds","retry_count","is_psp_outage_window","hour","day_of_week",
                    "day_of_month","country_mismatch"]
        if "merchant_risk" in m.columns: num_cols.append("merchant_risk")
        if "is_fraud_device_hint" in m.columns: num_cols.append("is_fraud_device_hint")
        le = LabelEncoder()
        for c in cat_cols:
            if c in m.columns:
                m[f"{c}_enc"] = le.fit_transform(m[c].astype(str))
                num_cols.append(f"{c}_enc")
        num_cols = [c for c in num_cols if c in m.columns]
        X = m[num_cols].fillna(0)
        y = m["fraud_label"]
        return X, y, num_cols

    X, y, feature_names = prepare_features(df)

    with st.expander("🔧  Feature Engineering Pipeline", expanded=False):
        st.markdown(f"""
        | Step | Description |
        |---|---|
        | DateTime | Extracted `hour`, `day_of_week`, `day_of_month` |
        | Interaction | `country_mismatch` flag |
        | Encoding | Label-encoded 6 categorical fields |
        | Imputation | Filled NaN → 0 |
        | **Total Features** | **{len(feature_names)}** |
        """)
        st.dataframe(pd.DataFrame({"Feature": feature_names, "#": range(1, len(feature_names)+1)}),
                     hide_index=True, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", f"{len(X):,}")
    m2.metric("Features", len(feature_names))
    m3.metric("Fraud Rate", _pct(y.mean()*100))
    m4.metric("Class 0 / Class 1", f"{(y==0).sum():,} / {(y==1).sum():,}")

    st.markdown("---")

    # ── config ──
    st.markdown("### ⚙️ Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test split", 0.10, 0.40, 0.25, 0.05)
        seed = st.number_input("Random seed", 0, 999, 42)
    with c2:
        model_names = st.multiselect("Models",
            ["Logistic Regression","Decision Tree","Random Forest","Gradient Boosting","AdaBoost"],
            default=["Random Forest","Gradient Boosting"])
    with c3:
        do_scale = st.toggle("Scale features", value=True)
        do_cv = st.toggle("Cross-validation (5-fold)", value=False)

    run = st.button("🚀  Train Models", type="primary", use_container_width=True)

    if run:
        if not model_names:
            st.warning("Select at least one model."); return

        progress = st.progress(0, text="Preparing data …")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        if do_scale:
            sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

        registry = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
            "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=seed),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=seed),
            "AdaBoost": AdaBoostClassifier(n_estimators=120, random_state=seed),
        }

        results, preds, fitted = [], {}, {}
        for i, name in enumerate(model_names):
            progress.progress((i+1)/len(model_names), text=f"Training **{name}** …")
            mdl = registry[name]
            mdl.fit(X_tr, y_tr)
            fitted[name] = mdl
            yp = mdl.predict(X_te)
            ypp = mdl.predict_proba(X_te)[:,1] if hasattr(mdl,"predict_proba") else yp.astype(float)
            preds[name] = {"y_pred": yp, "y_proba": ypp}

            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_te, yp),
                "Precision": precision_score(y_te, yp, zero_division=0),
                "Recall": recall_score(y_te, yp, zero_division=0),
                "F1": f1_score(y_te, yp, zero_division=0),
                "ROC-AUC": roc_auc_score(y_te, ypp),
            }

            if do_cv:
                cv_scores = cross_val_score(registry[name], X_tr, y_tr, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1)
                row["CV AUC (mean±std)"] = f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"

            results.append(row)

        progress.empty()
        st.success("All models trained successfully!")

        res_df = pd.DataFrame(results)

        # ── leaderboard ──
        st.markdown("### 🏆 Leaderboard")
        num_metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        styled = res_df.style.format({m:"{:.4f}" for m in num_metrics})\
                    .background_gradient(subset=num_metrics, cmap="RdYlGn", vmin=0.5, vmax=1.0)\
                    .highlight_max(subset=num_metrics, color="#bbf7d0")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── visuals ──
        st.markdown("### 📈 Performance Charts")
        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            colors_bar = [COLOR_INFO, COLOR_FRAUD, COLOR_WARN, COLOR_LEGIT, COLOR_DARK]
            for idx, metric in enumerate(num_metrics):
                fig.add_trace(go.Bar(name=metric, x=res_df["Model"], y=res_df[metric],
                                     marker_color=colors_bar[idx % len(colors_bar)],
                                     text=res_df[metric].round(3), textposition="outside"))
            fig.update_layout(barmode="group", title="Metrics Comparison", yaxis_range=[0,1.08])
            st.plotly_chart(_chart(fig, 400), use_container_width=True)

        with c2:
            fig = go.Figure()
            for name in preds:
                fpr, tpr, _ = roc_curve(y_te, preds[name]["y_proba"])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                    name=f'{name} (AUC {auc(fpr,tpr):.3f})'))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines",
                line=dict(dash="dash",color="grey"), name="Random"))
            fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(_chart(fig, 400), use_container_width=True)

        # ── detailed drill-down ──
        st.markdown("---")
        st.markdown("### 🔬 Model Deep Dive")
        pick = st.selectbox("Select model", list(preds.keys()))

        yp = preds[pick]["y_pred"]
        ypp = preds[pick]["y_proba"]

        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(y_te, yp)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Legit","Fraud"], y=["Legit","Fraud"],
                            labels={"x":"Predicted","y":"Actual","color":"Count"},
                            title=f"{pick} — Confusion Matrix")
            st.plotly_chart(_chart(fig, 360), use_container_width=True)
        with c2:
            rpt = classification_report(y_te, yp, output_dict=True)
            rpt_df = pd.DataFrame(rpt).T
            st.markdown(f"**Classification Report — {pick}**")
            st.dataframe(rpt_df.style.format("{:.4f}"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            prec_vals, rec_vals, _ = precision_recall_curve(y_te, ypp)
            fig = go.Figure(go.Scatter(x=rec_vals, y=prec_vals, fill="tozeroy",
                            line=dict(color=COLOR_INFO)))
            fig.update_layout(title=f"{pick} — Precision-Recall Curve",
                              xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(_chart(fig, 360), use_container_width=True)

        with c2:
            # Threshold analysis
            fpr_t, tpr_t, thresholds = roc_curve(y_te, ypp)
            if len(thresholds) > 2:
                j_scores = tpr_t - fpr_t
                best_idx = np.argmax(j_scores)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=thresholds, y=tpr_t[:-1] if len(tpr_t)>len(thresholds) else tpr_t[:len(thresholds)],
                                         name="TPR", line=dict(color=COLOR_LEGIT)))
                fig.add_trace(go.Scatter(x=thresholds, y=fpr_t[:-1] if len(fpr_t)>len(thresholds) else fpr_t[:len(thresholds)],
                                         name="FPR", line=dict(color=COLOR_FRAUD)))
                fig.add_vline(x=thresholds[best_idx], line_dash="dash", line_color=COLOR_WARN,
                              annotation_text=f"Optimal: {thresholds[best_idx]:.3f}")
                fig.update_layout(title=f"{pick} — Threshold Analysis (Youden's J)",
                                  xaxis_title="Threshold", yaxis_title="Rate")
                st.plotly_chart(_chart(fig, 360), use_container_width=True)

        # Feature importance
        mdl_obj = fitted[pick]
        if hasattr(mdl_obj, "feature_importances_"):
            imp = pd.DataFrame({"Feature": feature_names, "Importance": mdl_obj.feature_importances_})
            imp = imp.sort_values("Importance", ascending=True).tail(20)
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis",
                         title=f"{pick} — Top 20 Feature Importances")
            st.plotly_chart(_chart(fig, 500), use_container_width=True)
        elif hasattr(mdl_obj, "coef_"):
            coef = pd.DataFrame({"Feature": feature_names, "Coefficient": mdl_obj.coef_[0]})
            coef["abs"] = coef["Coefficient"].abs()
            coef = coef.sort_values("abs", ascending=True).tail(20)
            fig = px.bar(coef, x="Coefficient", y="Feature", orientation="h",
                         color="Coefficient", color_continuous_scale="RdBu_r",
                         title=f"{pick} — Top 20 Feature Coefficients")
            st.plotly_chart(_chart(fig, 500), use_container_width=True)

    # ── Prediction Simulator ──
    st.markdown("---")
    st.markdown("### 🎯 Quick-Score Simulator")
    st.caption("Manually enter a transaction profile to see estimated fraud probability (uses a pre-trained Random Forest)")

    with st.form("sim_form"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            sim_amt = st.number_input("Amount ($)", 0.0, 10000.0, 120.0)
            sim_bin = st.slider("BIN Risk", 0.0, 1.0, 0.15)
        with sc2:
            sim_ip = st.slider("IP Risk", 0.0, 1.0, 0.10)
            sim_merch = st.slider("Merchant Risk", 0.0, 1.0, 0.05)
        with sc3:
            sim_email = st.selectbox("Email Risk", [0,1], index=0)
            sim_3ds = st.selectbox("3DS Enabled", [0,1], index=0)
        with sc4:
            sim_retry = st.number_input("Retries", 0, 10, 0)
            sim_age = st.number_input("User Age (days)", 0, 1000, 90)
        submitted = st.form_submit_button("Score Transaction", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Scoring …"):
            # Quick RF on numeric subset
            quick_feats = ["amount","bin_risk","ip_risk_level","email_risk","is_3ds",
                           "retry_count","user_age_days"]
            if "merchant_risk" in df.columns: quick_feats.append("merchant_risk")
            Xq = df[quick_feats].fillna(0)
            yq = df["fraud_label"]
            rfq = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rfq.fit(Xq, yq)
            inp = [sim_amt, sim_bin, sim_ip, sim_email, sim_3ds, sim_retry, sim_age]
            if "merchant_risk" in df.columns: inp.append(sim_merch)
            prob = rfq.predict_proba(np.array(inp).reshape(1,-1))[0][1]

        risk_label = "🟢 LOW" if prob < 0.3 else ("🟡 MEDIUM" if prob < 0.6 else "🔴 HIGH")
        col_a, col_b = st.columns([1,2])
        with col_a:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob*100,
                number={"suffix":"%","font":{"size":50}},
                title={"text":"Fraud Probability"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":COLOR_FRAUD if prob>=0.5 else COLOR_WARN if prob>=0.3 else COLOR_LEGIT},
                       "steps":[{"range":[0,30],"color":"#d4edda"},{"range":[30,60],"color":"#fff3cd"},{"range":[60,100],"color":"#f8d7da"}]}))
            st.plotly_chart(_chart(gauge, 280), use_container_width=True)
        with col_b:
            st.markdown(f"### Risk Assessment: {risk_label}")
            st.markdown(f"""
            | Factor | Value | Signal |
            |---|---|---|
            | Amount | ${sim_amt:,.2f} | {'⚠️' if sim_amt > 200 else '✅'} |
            | BIN Risk | {sim_bin:.2f} | {'🔴' if sim_bin > 0.3 else '✅'} |
            | IP Risk | {sim_ip:.2f} | {'🔴' if sim_ip > 0.3 else '✅'} |
            | Merchant Risk | {sim_merch:.2f} | {'🔴' if sim_merch > 0.3 else '✅'} |
            | Email Risk | {'High' if sim_email else 'Low'} | {'🔴' if sim_email else '✅'} |
            | 3DS | {'Yes' if sim_3ds else 'No'} | {'✅' if sim_3ds else '⚠️'} |
            | Retries | {sim_retry} | {'🔴' if sim_retry >= 3 else '✅'} |
            | Account Age | {sim_age}d | {'⚠️' if sim_age < 30 else '✅'} |
            """)


# ──────────────────────────────────────────────────────────────
# 5  DATA EXPLORER
# ──────────────────────────────────────────────────────────────
def page_data():
    df = build_master()
    txn, users, devices, ips, merchants, cb = load_raw()

    st.markdown("## Data Explorer")
    st.caption("Inspect raw and merged datasets, column stats, and correlations")

    choice = st.selectbox("Dataset", ["Master (merged)","transactions_raw","users","devices","ips","merchants","chargebacks"])
    dmap = {"Master (merged)":df, "transactions_raw":txn, "users":users, "devices":devices,
            "ips":ips, "merchants":merchants, "chargebacks":cb}
    sel = dmap[choice]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Rows", f"{len(sel):,}")
    m2.metric("Columns", len(sel.columns))
    m3.metric("Memory (MB)", f"{sel.memory_usage(deep=True).sum()/1e6:.1f}")
    m4.metric("Missing %", f"{sel.isnull().sum().sum()/(sel.shape[0]*sel.shape[1])*100:.2f}%")

    # column-level profiling
    with st.expander("📊 Column Profiles", expanded=False):
        profile = []
        for col in sel.columns:
            s = sel[col]
            profile.append({
                "Column": col,
                "Type": str(s.dtype),
                "Non-Null": s.count(),
                "Null %": f"{s.isnull().mean()*100:.1f}%",
                "Unique": s.nunique(),
                "Sample": str(s.dropna().iloc[0]) if s.count() > 0 else "—",
            })
        st.dataframe(pd.DataFrame(profile), hide_index=True, use_container_width=True)

    st.dataframe(sel.head(50), use_container_width=True)

    csv = sel.to_csv(index=False)
    st.download_button("⬇️  Download full dataset (CSV)", csv, f"{choice}.csv", "text/csv")


# ──────────────────────────────────────────────────────────────
# MAIN — SIDEBAR NAV
# ──────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("## 🛡️ DefCone")
        st.caption("Fraud Intelligence Platform")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            [
                "📊  Executive Overview",
                "🔍  Exploratory Analysis",
                "🕵️  Fraud Investigation",
                "🤖  ML Modeling",
                "🗂️  Data Explorer",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:.75rem;opacity:.6;line-height:1.5;'>"
            "DefCone v2.0<br>Senior Lender Report Dashboard<br>"
            "© 2026 — Confidential</div>",
            unsafe_allow_html=True,
        )

    label = page.split("  ", 1)[1] if "  " in page else page

    if "Executive" in label:
        page_overview()
    elif "Exploratory" in label:
        page_eda()
    elif "Investigation" in label:
        page_investigation()
    elif "ML" in label:
        page_ml()
    elif "Data" in label:
        page_data()


if __name__ == "__main__":
    main()