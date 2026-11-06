"""RetailPulse 360 - churn-risk dashboard for a retention team.

Run locally from the repository root:
    streamlit run app/streamlit_app.py
Uses only committed files (model + pre-scored CSVs), so the raw data is not needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets joblib find src.retailpulse classes if ever needed

APP_DATA = ROOT / "app" / "data"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
RESULTS_PATH = ROOT / "reports" / "model_results.json"
FIGURES = ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
BAND_ORDER = ["Low", "Medium", "High", "Very high"]
BAND_COLORS = ["#86b6ef", "#3987e5", "#eb6834", "#a8302f"]
COHORTS = {"Latest: 1 Dec 2011 (next 90 days unknown)": "2011-12-01",
           "Test: 1 Sep 2011 (actual outcome known)": "2011-09-01"}
FEATURE_HELP = {
    "recency_days": "Days since the customer's last order",
    "frequency": "Orders in the last 365 days",
    "monetary": "Spend (£) in the last 365 days",
    "avg_order_value": "Average order value (£)",
    "total_quantity": "Units bought in the last 365 days",
    "n_products": "Distinct products bought",
    "active_months": "Distinct months with a purchase (last 365 days)",
    "tenure_days": "Days since first ever order",
    "avg_days_between_orders": "Average days between orders (blank if only one order)",
    "frequency_recent": "Orders in the last 90 days",
    "monetary_recent": "Spend (£) in the last 90 days",
    "recent_share_monetary": "Share of yearly spend made in the last 90 days",
    "return_rate": "Share of order lines later cancelled/returned",
}

st.set_page_config(page_title="RetailPulse 360 – Churn Risk", page_icon="📉", layout="wide")


@st.cache_resource
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_scores(cutoff: str) -> pd.DataFrame:
    df = pd.read_csv(APP_DATA / f"scores_{cutoff}.csv", dtype={"Customer ID": str})
    df["risk_band"] = pd.Categorical(df["risk_band"], BAND_ORDER, ordered=True)
    return df


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gbp(v: float) -> str:
    return f"£{v:,.0f}"


bundle = load_model()
results = load_json(RESULTS_PATH)
ref = load_json(APP_DATA / "feature_reference.json")
FEATURES = bundle["features"]
NUMERIC = [f for f in FEATURES if f != "country_group"]
THRESHOLD = bundle["threshold"]

# ---------------------------------------------------------------- sidebar
st.sidebar.title("RetailPulse 360")
st.sidebar.caption("Customer churn risk · UCI Online Retail II")
cohort_label = st.sidebar.radio("Customer snapshot", list(COHORTS))
cutoff = COHORTS[cohort_label]
scores = load_scores(cutoff)
countries = ["All"] + sorted(scores["country"].dropna().unique())
country = st.sidebar.selectbox("Country", countries)
bands = st.sidebar.multiselect("Risk band", BAND_ORDER, default=BAND_ORDER)
min_spend = st.sidebar.number_input("Min. spend last 365 days (£)", 0, step=100, value=0)
view = scores[scores["risk_band"].isin(bands) & (scores["monetary"] >= min_spend)]
if country != "All":
    view = view[view["country"] == country]
st.sidebar.divider()
st.sidebar.markdown(
    f"**Model:** {bundle['model_name']}  \n**Decision threshold:** {THRESHOLD}  \n"
    f"**Test ROC-AUC:** {results['test'][bundle['model_name']]['roc_auc']:.3f}  \n"
    "**Churn** = no purchase in the next 90 days.")

st.title("Customer churn risk")
st.caption(f"Snapshot at **{cutoff}** · features use only purchases before this date · "
           f"{len(view):,} of {len(scores):,} customers match the filters")

tab_overview, tab_customer, tab_whatif, tab_batch, tab_model = st.tabs(
    ["Overview", "Customer lookup", "What-if", "Batch scoring", "Model performance"])

# ---------------------------------------------------------------- overview
with tab_overview:
    flagged = view[view["predicted_churn"] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(view):,}")
    c2.metric("Predicted to churn", f"{len(flagged):,}",
              f"{len(flagged) / max(len(view), 1):.0%} of customers", delta_color="off")
    c3.metric("Past-year spend of predicted churners", gbp(flagged["monetary"].sum()))
    c4.metric("Revenue at risk (Σ p × spend)", gbp(view["revenue_at_risk"].sum()),
              help="Sum over customers of churn probability × spend in the previous 365 days. "
                   "A planning estimate, not a forecast.")
    if "churn_90d" in view:
        acc = (view["predicted_churn"] == view["churn_90d"]).mean() if len(view) else float("nan")
        st.info(f"Actual outcome is known for this snapshot: **{view['churn_90d'].mean():.1%}** of these "
                f"customers really churned; the model's yes/no prediction was correct for **{acc:.1%}** of them.")

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Customers by risk band")
        band = (view.groupby("risk_band", observed=False)
                .agg(customers=("Customer ID", "size"), revenue_at_risk=("revenue_at_risk", "sum"))
                .reset_index())
        st.altair_chart(
            alt.Chart(band).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("customers:Q", title="Customers"),
                y=alt.Y("risk_band:N", sort=BAND_ORDER, title=None),
                color=alt.Color("risk_band:N", scale=alt.Scale(domain=BAND_ORDER, range=BAND_COLORS),
                                legend=None),
                tooltip=["risk_band", "customers", alt.Tooltip("revenue_at_risk:Q", format=",.0f")]
            ).properties(height=220), width="stretch")
        st.caption("Bands: Low < 0.4 ≤ Medium < 0.6 ≤ High < 0.8 ≤ Very high")
    with right:
        st.subheader("Risk vs. value")
        sample = view.sample(min(len(view), 2500), random_state=1) if len(view) else view
        st.altair_chart(
            alt.Chart(sample).mark_circle(size=28, opacity=0.55).encode(
                x=alt.X("churn_probability:Q", title="Churn probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("monetary:Q", title="Spend last 365 days (£, log)", scale=alt.Scale(type="log")),
                color=alt.Color("risk_band:N", scale=alt.Scale(domain=BAND_ORDER, range=BAND_COLORS),
                                legend=alt.Legend(title="Risk band", orient="bottom")),
                tooltip=["Customer ID", "country", alt.Tooltip("churn_probability:Q", format=".2f"),
                         alt.Tooltip("monetary:Q", format=",.0f"), "recency_days", "frequency"]
            ).properties(height=260), width="stretch")
        st.caption("Top-right = valuable customers at high risk: the first people to contact.")

    st.subheader("Priority contact list")
    st.caption("Sorted by revenue at risk (churn probability × past-year spend).")
    n = st.slider("Show top", 10, 200, 25, step=5)
    cols = ["Customer ID", "country", "churn_probability", "risk_band", "monetary", "revenue_at_risk",
            "recency_days", "frequency", "active_months"] + (["churn_90d"] if "churn_90d" in view else [])
    top = view.sort_values("revenue_at_risk", ascending=False).head(n)[cols]
    st.dataframe(top, hide_index=True, width="stretch", column_config={
        "churn_probability": st.column_config.ProgressColumn("Churn prob.", min_value=0, max_value=1, format="%.2f"),
        "monetary": st.column_config.NumberColumn("Spend 365d (£)", format="%.0f"),
        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk (£)", format="%.0f"),
        "recency_days": st.column_config.NumberColumn("Days since last order", format="%.0f"),
        "churn_90d": st.column_config.NumberColumn("Actually churned", format="%d"),
    })
    st.download_button("Download filtered list (CSV)", view.to_csv(index=False).encode(),
                       file_name=f"churn_risk_{cutoff}.csv", mime="text/csv")

# ---------------------------------------------------------------- customer lookup
with tab_customer:
    ids = scores.sort_values("revenue_at_risk", ascending=False)["Customer ID"].tolist()
    cid = st.selectbox("Customer ID (sorted by revenue at risk)", ids)
    row = scores[scores["Customer ID"] == cid].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{row['churn_probability']:.0%}", f"{row['risk_band']} risk", delta_color="off")
    b.metric("Spend last 365 days", gbp(row["monetary"]))
    c.metric("Days since last order", f"{row['recency_days']:.0f}")
    if "churn_90d" in row:
        st.write("Actual outcome: **" + ("churned" if row["churn_90d"] == 1 else "purchased again") + "**")
    comp = pd.DataFrame({"feature": NUMERIC,
                         "description": [FEATURE_HELP[f] for f in NUMERIC],
                         "this customer": [row[f] for f in NUMERIC],
                         "typical customer (median)": [ref[f]["median"] for f in NUMERIC]})
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={"this customer": st.column_config.NumberColumn(format="%.2f"),
                                "typical customer (median)": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Strongest drivers in the model: active months, recency and frequency (see Model performance).")

# ---------------------------------------------------------------- what-if
with tab_whatif:
    st.write("Change a customer's behaviour and see how the predicted churn risk responds.")
    base = scores[scores["Customer ID"] == cid].iloc[0] if cid else None
    st.caption(f"Starting values come from customer **{cid}** (selected in *Customer lookup*).")
    cols3 = st.columns(3)
    inputs = {}
    for i, f in enumerate(NUMERIC):
        lo, hi = 0.0, float(max(ref[f]["p99"] * 1.5, (base[f] if base is not None and pd.notna(base[f]) else 0) * 1.2, 1))
        val = float(base[f]) if base is not None and pd.notna(base[f]) else float(ref[f]["median"])
        inputs[f] = cols3[i % 3].number_input(f, min_value=lo, max_value=hi, value=min(max(val, lo), hi),
                                              help=FEATURE_HELP[f], key=f"wi_{f}")
    groups = ["United Kingdom", "EIRE", "Germany", "France", "Netherlands", "Other"]
    cg = str(base["country"]) if base is not None else "United Kingdom"
    inputs["country_group"] = st.selectbox("country_group", groups,
                                           index=groups.index(cg) if cg in groups else len(groups) - 1)
    p = float(bundle["pipeline"].predict_proba(pd.DataFrame([inputs])[FEATURES])[:, 1][0])
    st.metric("Predicted churn probability", f"{p:.1%}",
              "above threshold → likely to churn" if p >= THRESHOLD else "below threshold → likely to stay",
              delta_color="inverse" if p >= THRESHOLD else "normal")

# ---------------------------------------------------------------- batch scoring
with tab_batch:
    st.write("Upload a CSV with one row per customer and these columns (extra columns are kept):")
    st.code(", ".join(FEATURES), language=None)
    template = scores[["Customer ID"] + FEATURES[:-1]].head(5).assign(country_group="United Kingdom")
    st.download_button("Download a template CSV", template.to_csv(index=False).encode(),
                       file_name="churn_scoring_template.csv", mime="text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            data["churn_probability"] = bundle["pipeline"].predict_proba(data[FEATURES])[:, 1].round(4)
            data["predicted_churn"] = (data["churn_probability"] >= THRESHOLD).astype(int)
            st.success(f"Scored {len(data):,} rows.")
            st.dataframe(data, hide_index=True, width="stretch")
            st.download_button("Download scores", data.to_csv(index=False).encode(),
                               file_name="churn_scores.csv", mime="text/csv")

# ---------------------------------------------------------------- model performance
with tab_model:
    st.write(f"Selected model: **{results['selected_model']}**, chosen on the validation month "
             f"({results['validation_cutoff']}) and evaluated once on the test month ({results['test_cutoff']}).")
    metrics = (pd.DataFrame(results["test"]).T[["roc_auc", "pr_auc", "macro_f1", "precision_churn",
                                                "recall_churn", "accuracy"]]
               .sort_values("roc_auc", ascending=False))
    st.dataframe(metrics.style.format("{:.3f}"), width="stretch")
    g1, g2 = st.columns(2)
    for col, name, cap in [(g1, "fig13_test_roc_pr", "ROC and precision-recall curves (test month)"),
                           (g2, "fig15_permutation_importance", "What drives the predictions"),
                           (g1, "fig14_confusion_matrix", "Confusion matrix (test month)"),
                           (g2, "fig16_cumulative_gains", "Cumulative gains")]:
        path = FIGURES / f"{name}.png"
        if path.exists():
            col.image(str(path), caption=cap, width="stretch")
    st.caption("Limitations: historical data (2009–2011) from one UK retailer; churn is a behavioural proxy; "
               "the test month is seasonally different from training. See the project report.")
