# Phase 1 checklist (data understanding)

- [x] Azure for Students account activated (no resources created yet; deployment only after the local app works).
- [x] Python 3.14.5, Git installed; virtual environment created and `requirements.txt` installed.
- [x] UCI Online Retail II workbook downloaded to `data/raw/` (SHA-256 recorded in `docs/02_dataset.md`).
- [x] Audit run on the real workbook; results in `reports/data_audit.json`.
- [x] Data dictionary for Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.
- [x] Counts recorded: missing values, duplicates (incl. 1–9 Dec 2010 sheet overlap), cancellations, negative quantities, non-positive prices.
- [x] Git initialised; foundation commit on `main`; `feature/data-understanding` branch.
- [x] Target definition, monthly cutoffs, 365-day lookback and 90-day horizon justified (gap analysis: 80.7% of repeat gaps ≤ 90 days).
- [ ] Push to GitHub (account `syedasif3018`) and open/merge PR #1.
