import pandas as pd

from src.retailpulse.features import build_snapshot


def _tx(rows):
    df = pd.DataFrame(rows, columns=["Customer ID", "Invoice", "InvoiceDate", "Revenue"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"], df["Quantity"], df["Country"] = "85123A", 1, "United Kingdom"
    df["has_customer"] = True
    df["is_cancellation"] = df["Invoice"].str.startswith("C")
    df["is_non_product"] = False
    df["is_valid_purchase"] = ~df["is_cancellation"]
    return df


ROWS = [
    ("A", "1", "2011-01-10", 10.0),
    ("A", "2", "2011-02-20", 30.0),
    ("A", "3", "2011-04-15", 50.0),   # after cutoff, inside 90-day window -> A retained
    ("B", "4", "2010-12-01", 20.0),   # B never buys again -> churn
    ("C", "5", "2010-01-01", 99.0),   # older than 365 days -> not eligible
]


def test_label_and_eligibility():
    snap = build_snapshot(_tx(ROWS), "2011-03-01").set_index("Customer ID")
    assert set(snap.index) == {"A", "B"}
    assert snap.loc["A", "churn_90d"] == 0
    assert snap.loc["B", "churn_90d"] == 1
    assert snap.loc["A", "frequency"] == 2
    assert snap.loc["A", "monetary"] == 40.0
    assert snap.loc["A", "recency_days"] == 9.0  # 20 Feb -> 1 Mar
    assert pd.isna(snap.loc["B", "avg_days_between_orders"])  # single order


def test_features_ignore_future_rows():
    """Adding a huge purchase after the cutoff must not change any feature."""
    base = build_snapshot(_tx(ROWS), "2011-03-01", with_label=False)
    more = build_snapshot(_tx(ROWS + [("A", "9", "2011-03-02", 1e6)]), "2011-03-01",
                          with_label=False)
    pd.testing.assert_frame_equal(base, more)
