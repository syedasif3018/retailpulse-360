"""Central project paths so every script/notebook uses the same locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_XLSX = ROOT / "data" / "raw" / "online_retail_II.xlsx"
RAW_PARQUET = ROOT / "data" / "processed" / "transactions_raw.parquet"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

SOURCE_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity",
                  "InvoiceDate", "Price", "Customer ID", "Country"]
RANDOM_STATE = 42
