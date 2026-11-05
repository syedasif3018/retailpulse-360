# RetailPulse 360

**End-to-end customer churn prediction and retail analytics | CSE570 Machine Learning with Python**

## Problem statement

Given a customer's purchase history up to an observation cutoff, predict whether the customer will have **no qualifying purchase in the following 90 days**. This is a **customer-level binary classification** problem. We will define eligibility and qualifying purchases carefully before modelling. The broader solution will include segmentation, an analyst dashboard, an API and optional Azure deployment.

## Data source

Daqing Chen, *Online Retail II*, UCI Machine Learning Repository, https://doi.org/10.24432/C5CG6D (CC BY 4.0). https://archive.ics.uci.edu/dataset/502/online+retail+ii

Real UK online retail transaction history, 01-Dec-2009 through 09-Dec-2011, 1,067,371 rows. Eight source fields (note: original Excel sheet headers may be `Invoice`, `Price`, and `Customer ID`):

| Field | Meaning |
|---|---|
| Invoice / InvoiceNo | Invoice identifier; codes beginning C denote cancellations |
| StockCode | Product identifier |
| Description | Product description |
| Quantity | Item quantity; may be negative for returns/cancellations |
| InvoiceDate | Transaction timestamp |
| Price / UnitPrice | Price in GBP |
| Customer ID / CustomerID | Customer identifier; some missing |
| Country | Customer country |

Source workbook is not committed to GitHub. Download it yourself and preserve the original file.

## Setup on Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Tested with Python 3.14.5 on Windows 10. The virtual environment can live outside the repo (e.g. `C:\Users\<you>\.venvs\retailpulse-360`) if the project drive is short of space.

Download https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip, extract it, and save the workbook as `data/raw/online_retail_II.xlsx`.

Run the source-data audit from the repository root (the first run reads Excel in about 1.5 min and caches `data/processed/transactions_raw.parquet`):

```powershell
python -m src.retailpulse.audit
python -m pytest -q
```

The audit writes `reports/data_audit.json` and does not modify original data. It loads each Excel sheet separately to lower peak memory use. It may still take time on low-memory laptops.

## Phase roadmap

1. Data understanding, data dictionary, audit and problem framing. Done: [problem statement](docs/01_problem_statement.md), [dataset & data dictionary](docs/02_dataset.md), [audit output](reports/data_audit.json).
2. Cleaning, EDA and statistical tests. Done: [cleaning module](src/retailpulse/cleaning.py), [cleaning log](reports/cleaning_log.json), [EDA notebook](notebooks/02_eda_statistics.ipynb), [figures](reports/figures/).
3. Churn modelling. Done: [modelling module](src/retailpulse/modeling.py), [notebook](notebooks/03_churn_modelling.ipynb), [results](reports/model_results.json). Gradient Boosting: test ROC-AUC 0.774, PR-AUC 0.746, macro-F1 0.693.
4. Segmentation/PCA, FastAPI, Streamlit, optional Power BI.
5. Optional cloud deployment, documentation, report and viva preparation.

## Methodology safeguards

- Use separate customer observation cutoffs and future 90-day outcome windows.
- Keep all future outcomes out of features; time-based evaluation and no test-set tuning.
- Define valid orders and handle cancellation/returns, missing customer IDs, and class imbalance transparently.
- Never claim that the historic data predicts current real-world retail churn without external validation.

## Academic assignment deliverables

Data dictionary; cleaning and EDA; one suitable statistical test; baseline + at least three ML models; cross-validation, tuning, metrics, interpretation; report, viva, GitHub evidence, LinkedIn post.
