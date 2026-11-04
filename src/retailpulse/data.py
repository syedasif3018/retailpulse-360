"""Load the UCI Online Retail II workbook once and cache it as Parquet.

The workbook has two sheets ("Year 2009-2010", "Year 2010-2011"). Reading ~1M
rows from Excel takes minutes, so we convert once to a columnar Parquet file
(seconds to reload). The original .xlsx is never modified.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_PARQUET, RAW_XLSX, SOURCE_COLUMNS

RENAME = {"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"}


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one sheet, keeping identifier columns as text (e.g. 'C489449', '85123A')."""
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl",
                       dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str})
    df = df.rename(columns=RENAME)
    absent = set(SOURCE_COLUMNS) - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns in {sheet_name}: {sorted(absent)}")
    df = df[SOURCE_COLUMNS].copy()
    df["source_sheet"] = sheet_name
    return df


def load_raw(xlsx: Path = RAW_XLSX, cache: Path = RAW_PARQUET, refresh: bool = False) -> pd.DataFrame:
    """Return all raw rows from both sheets (no cleaning), using the Parquet cache if present."""
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    if not xlsx.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx}. Download it from UCI into data/raw/.")
    sheets = pd.ExcelFile(xlsx, engine="openpyxl").sheet_names
    frames = []
    for s in sheets:
        print(f"Reading sheet {s!r} ...", flush=True)
        frames.append(read_sheet(xlsx, s))
    df = pd.concat(frames, ignore_index=True)
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype("string")
    df["Invoice"] = df["Invoice"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"Cached {len(df):,} rows to {cache}")
    return df
