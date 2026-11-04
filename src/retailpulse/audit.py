"""Phase 1: audit the raw source data without altering the original workbook.

Run from the repository root:  python -m src.retailpulse.audit
Output: reports/data_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import RAW_XLSX, REPORTS, SOURCE_COLUMNS
from .data import load_raw


def audit_sheet(df: pd.DataFrame, sheet_name: str) -> dict:
    """Data-quality counts for one block of raw rows."""
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    invoice = df["Invoice"].astype("string")
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["Price"], errors="coerce")
    cols = [c for c in SOURCE_COLUMNS if c in df.columns]
    return {
        "sheet": sheet_name,
        "rows": int(len(df)),
        "columns": cols,
        "duplicate_rows": int(df[cols].duplicated().sum()),
        "missing_by_column": {str(k): int(v) for k, v in df[cols].isna().sum().items()},
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "cancellation_rows": int(invoice.str.upper().str.startswith("C", na=False).sum()),
        "negative_quantity_rows": int((quantity < 0).sum()),
        "nonpositive_price_rows": int((price <= 0).sum()),
        "distinct_invoices": int(invoice.nunique(dropna=True)),
        "distinct_customers_nonmissing": int(df["Customer ID"].nunique(dropna=True)),
        "distinct_stockcodes": int(df["StockCode"].nunique(dropna=True)),
        "distinct_countries": int(df["Country"].nunique(dropna=True)),
    }


def sheet_overlap(df: pd.DataFrame) -> dict:
    """Check whether the two yearly sheets share the same invoices (double counting risk)."""
    sheets = list(df["source_sheet"].unique())
    if len(sheets) != 2:
        return {}
    a, b = (df[df["source_sheet"] == s] for s in sheets)
    shared_invoices = set(a["Invoice"].dropna()) & set(b["Invoice"].dropna())
    shared_rows_in_second = int(b["Invoice"].isin(shared_invoices).sum())
    dates_b = pd.to_datetime(b.loc[b["Invoice"].isin(shared_invoices), "InvoiceDate"])
    return {
        "sheets": sheets,
        "shared_invoice_count": len(shared_invoices),
        "rows_in_second_sheet_with_shared_invoice": shared_rows_in_second,
        "shared_date_min": str(dates_b.min()) if len(dates_b) else None,
        "shared_date_max": str(dates_b.max()) if len(dates_b) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=RAW_XLSX)
    parser.add_argument("--output", type=Path, default=REPORTS / "data_audit.json")
    parser.add_argument("--refresh", action="store_true", help="re-read Excel instead of Parquet cache")
    args = parser.parse_args()

    df = load_raw(args.file, refresh=args.refresh)
    results = [audit_sheet(g, s) for s, g in df.groupby("source_sheet", sort=False)]
    combined = audit_sheet(df, "combined")
    payload = {
        "source": "UCI Online Retail II (dataset 502), https://doi.org/10.24432/C5CG6D",
        "total_rows": int(len(df)),
        "sheets": results,
        "combined": combined,
        "sheet_overlap": sheet_overlap(df),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
