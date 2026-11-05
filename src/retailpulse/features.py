"""Customer snapshots: features from the past, churn label from the future.

For a cutoff date C (midnight):
  * eligible customers: >= 1 valid purchase in [C - 365 days, C)
  * features: computed ONLY from transactions with InvoiceDate < C
  * label churn_90d: 1 if no valid purchase in [C, C + 90 days), else 0

Because features and label use disjoint time ranges, a snapshot never
contains information from its own future (no target leakage).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 365
HORIZON_DAYS = 90
RECENT_DAYS = 90

NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "total_quantity",
    "n_products", "active_months", "tenure_days", "avg_days_between_orders",
    "frequency_recent", "monetary_recent", "recent_share_monetary", "return_rate",
]
CATEGORICAL_FEATURES = ["country_group"]
TARGET = "churn_90d"
MAIN_COUNTRIES = ["United Kingdom", "EIRE", "Germany", "France", "Netherlands"]


def build_snapshot(tx: pd.DataFrame, cutoff, lookback_days: int = LOOKBACK_DAYS,
                   horizon_days: int = HORIZON_DAYS, with_label: bool = True) -> pd.DataFrame:
    """One row per eligible customer at `cutoff`. `tx` = cleaned transactions."""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=lookback_days)
    recent_start = cutoff - pd.Timedelta(days=RECENT_DAYS)

    known = tx[tx["has_customer"]]
    past = known[known["InvoiceDate"] < cutoff]            # nothing at/after cutoff
    valid_past = past[past["is_valid_purchase"]]
    win = valid_past[valid_past["InvoiceDate"] >= start]
    if win.empty:
        return pd.DataFrame()

    g = win.groupby("Customer ID")
    orders = (win.groupby(["Customer ID", "Invoice"])
              .agg(date=("InvoiceDate", "min"), value=("Revenue", "sum")).reset_index())
    og = orders.groupby("Customer ID")
    first_order, last_order = og["date"].min(), og["date"].max()
    frequency = og.size()

    snap = pd.DataFrame({
        "recency_days": (cutoff - last_order).dt.total_seconds() / 86400,
        "frequency": frequency,
        "monetary": g["Revenue"].sum(),
        "total_quantity": g["Quantity"].sum(),
        "n_products": g["StockCode"].nunique(),
        "active_months": g["InvoiceDate"].apply(lambda s: s.dt.to_period("M").nunique()),
    })
    snap["avg_order_value"] = snap["monetary"] / snap["frequency"]
    span_days = (last_order - first_order).dt.total_seconds() / 86400
    # undefined for single-order customers -> NaN, imputed inside the model pipeline
    snap["avg_days_between_orders"] = (span_days / (frequency - 1)).where(frequency > 1)
    # tenure uses all history before the cutoff (data starts 2009-12-01, so it is right-censored)
    snap["tenure_days"] = ((cutoff - valid_past.groupby("Customer ID")["InvoiceDate"].min())
                           .dt.total_seconds() / 86400).reindex(snap.index)

    rec = orders[orders["date"] >= recent_start].groupby("Customer ID")
    snap["frequency_recent"] = rec.size().reindex(snap.index, fill_value=0)
    snap["monetary_recent"] = rec["value"].sum().reindex(snap.index, fill_value=0.0)
    snap["recent_share_monetary"] = np.where(snap["monetary"] > 0,
                                             snap["monetary_recent"] / snap["monetary"], 0.0)

    # returns: cancellation lines in the window vs all product lines in the window
    win_all = past[(past["InvoiceDate"] >= start) & ~past["is_non_product"]]
    cancels = win_all[win_all["is_cancellation"]].groupby("Customer ID").size()
    lines = g.size()
    snap["return_rate"] = (cancels.reindex(snap.index, fill_value=0)
                           / (lines + cancels.reindex(snap.index, fill_value=0)))

    country = win.groupby("Customer ID")["Country"].agg(lambda s: s.mode().iat[0])
    snap["country"] = country
    snap["country_group"] = country.where(country.isin(MAIN_COUNTRIES), "Other")

    if with_label:
        end = cutoff + pd.Timedelta(days=horizon_days)
        fut = known[(known["InvoiceDate"] >= cutoff) & (known["InvoiceDate"] < end)
                    & known["is_valid_purchase"]]
        buyers = set(fut["Customer ID"])
        snap[TARGET] = (~snap.index.isin(buyers)).astype(int)
        snap["future_revenue_90d"] = (fut.groupby("Customer ID")["Revenue"].sum()
                                      .reindex(snap.index, fill_value=0.0))

    snap.insert(0, "cutoff", cutoff)
    return snap.reset_index().rename(columns={"index": "Customer ID"})


def build_snapshots(tx: pd.DataFrame, cutoffs: Iterable, **kw) -> pd.DataFrame:
    return pd.concat([build_snapshot(tx, c, **kw) for c in cutoffs], ignore_index=True)
