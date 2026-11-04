"""Phase 2: clean the raw transactions and flag (not silently drop) problem rows.

Run from the repository root:  python -m src.retailpulse.cleaning
Outputs: data/processed/transactions_clean.parquet, reports/cleaning_log.json

Design choice: rows are *flagged* rather than deleted wherever an analysis may
still need them (e.g. cancellations feed a customer "return rate" feature,
rows without Customer ID still count for transaction-level revenue EDA).
Only exact duplicates and immediately-reversed order/cancellation pairs are
removed, because they are not real business events.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import RAW_PARQUET, REPORTS, SOURCE_COLUMNS
from .data import load_raw

CLEAN_PARQUET = RAW_PARQUET.with_name("transactions_clean.parquet")

# Service/accounting codes that are not merchandise (checked against Description
# in the Phase 2 profiling: postage, carriage, manual, discount, samples, fees,
# gift vouchers, adjustments, test items, charity commission, bad-debt).
NON_PRODUCT_PATTERN = (r"^(POST|DOT|M|C\d|D|S|BANK CHARGES|AMAZONFEE|PADS|CRUK|B|"
                       r"GIFT(_\d+_\d+)?|ADJUST\d*|TEST\d*)$")
REVERSAL_WINDOW = pd.Timedelta(days=1)


def _reversed_pairs(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Flag purchase lines cancelled by an exactly matching line within 1 day.

    Match key: same customer, stock code, absolute quantity and price, with the
    cancellation at or after the purchase. Matching is one-to-one, so each
    cancellation offsets at most one purchase. Returns (purchase_flag, cancel_flag).
    """
    buy_flag = pd.Series(False, index=df.index)
    can_flag = pd.Series(False, index=df.index)
    known = df["Customer ID"].notna()
    buys = df[known & ~df["is_cancellation"] & (df["Quantity"] > 0)]
    cans = df[known & df["is_cancellation"] & (df["Quantity"] < 0)]
    if buys.empty or cans.empty:
        return buy_flag, can_flag
    key = ["Customer ID", "StockCode", "absq", "Price"]
    b = buys.assign(absq=buys["Quantity"].abs()).reset_index(names="row_b")
    c = cans.assign(absq=cans["Quantity"].abs()).reset_index(names="row_c")
    m = b[key + ["row_b", "InvoiceDate"]].merge(
        c[key + ["row_c", "InvoiceDate"]], on=key, suffixes=("", "_c"))
    lag = m["InvoiceDate_c"] - m["InvoiceDate"]
    m = (m[(lag >= pd.Timedelta(0)) & (lag <= REVERSAL_WINDOW)]
         .sort_values(["InvoiceDate_c", "InvoiceDate"])
         .drop_duplicates("row_c").drop_duplicates("row_b"))
    buy_flag.loc[m["row_b"].to_numpy()] = True
    can_flag.loc[m["row_c"].to_numpy()] = True
    return buy_flag, can_flag


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"raw_rows": int(len(raw))}
    df = raw.copy()

    # 1. Exact duplicates (includes the 1-9 Dec 2010 overlap between the two sheets)
    dup = df.duplicated(subset=SOURCE_COLUMNS, keep="first")
    log["dropped_exact_duplicates"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)

    # 2. Data types and text normalisation
    df["Invoice"] = df["Invoice"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    log["filled_missing_description"] = int(df["Description"].isna().sum())
    df["Description"] = df["Description"].fillna("UNKNOWN")
    df["Customer ID"] = (pd.to_numeric(df["Customer ID"], errors="coerce")
                         .astype("Int64").astype("string"))
    df["Country"] = df["Country"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = df["Quantity"].astype("int64")
    df["Price"] = df["Price"].astype("float64")

    # 3. Row-level flags
    df["is_cancellation"] = df["Invoice"].str.startswith("C")
    df["is_adjustment"] = df["Invoice"].str.startswith("A")
    df["is_non_product"] = df["StockCode"].str.match(NON_PRODUCT_PATTERN)
    df["has_customer"] = df["Customer ID"].notna()
    df["Revenue"] = df["Quantity"] * df["Price"]

    # 4. Orders reversed within a day (e.g. 80,995 and 74,215 unit data-entry errors)
    reversed_buy, reversed_can = _reversed_pairs(df)
    log["dropped_reversed_purchase_lines"] = int(reversed_buy.sum())
    log["dropped_reversing_cancellation_lines"] = int(reversed_can.sum())
    df = df.loc[~(reversed_buy | reversed_can)].reset_index(drop=True)

    # 5. Definition of a valid purchase line (used for labels and most features)
    df["is_valid_purchase"] = (~df["is_cancellation"] & ~df["is_adjustment"]
                               & ~df["is_non_product"]
                               & (df["Quantity"] > 0) & (df["Price"] > 0))

    log.update({
        "clean_rows": int(len(df)),
        "rows_missing_customer_id": int((~df["has_customer"]).sum()),
        "cancellation_rows": int(df["is_cancellation"].sum()),
        "adjustment_rows": int(df["is_adjustment"].sum()),
        "non_product_rows": int(df["is_non_product"].sum()),
        "nonpositive_price_rows": int((df["Price"] <= 0).sum()),
        "valid_purchase_rows": int(df["is_valid_purchase"].sum()),
        "valid_purchase_rows_with_customer": int((df["is_valid_purchase"] & df["has_customer"]).sum()),
        "customers_with_valid_purchase": int(df.loc[df["is_valid_purchase"], "Customer ID"].nunique()),
        "valid_revenue_gbp": round(float(df.loc[df["is_valid_purchase"], "Revenue"].sum()), 2),
        "valid_revenue_with_customer_gbp": round(float(
            df.loc[df["is_valid_purchase"] & df["has_customer"], "Revenue"].sum()), 2),
    })
    log["outliers"] = outlier_summary(df.loc[df["is_valid_purchase"]])
    return df, log


def outlier_summary(valid: pd.DataFrame) -> dict:
    """IQR (Tukey 1.5×IQR) outlier counts for line-level Quantity, Price, Revenue."""
    out = {}
    for col in ["Quantity", "Price", "Revenue"]:
        s = valid[col]
        q1, q3 = s.quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        out[col] = {"q1": float(q1), "q3": float(q3), "upper_fence": float(upper),
                    "n_above_fence": int((s > upper).sum()),
                    "pct_above_fence": round(float((s > upper).mean() * 100), 2),
                    "p99": float(s.quantile(0.99)), "max": float(s.max())}
    return out


def load_clean(refresh: bool = False) -> pd.DataFrame:
    if CLEAN_PARQUET.exists() and not refresh:
        return pd.read_parquet(CLEAN_PARQUET)
    df, _ = clean_transactions(load_raw())
    df.to_parquet(CLEAN_PARQUET, index=False)
    return df


def main() -> None:
    df, log = clean_transactions(load_raw())
    df.to_parquet(CLEAN_PARQUET, index=False)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cleaning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    print(f"Saved {len(df):,} rows to {CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
