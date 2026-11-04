# 2. Dataset Description

## 2.1 Source and citation
- **Name:** Online Retail II (UCI Machine Learning Repository, dataset ID 502)
- **Link:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **Citation:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0). Reuse is allowed with attribution.
- **File used:** `online_retail_II.xlsx` (45.6 MB, two sheets), downloaded 4 Nov 2026.
  SHA-256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.
  The raw file is not committed to Git. Reproduce the download with the steps in the README.

All counts below come from `python -m src.retailpulse.audit` (saved in `reports/data_audit.json`).

## 2.2 Size
| Sheet | Rows | Date range |
|---|---:|---|
| Year 2009-2010 | 525,461 | 2009-12-01 07:45 → 2010-12-09 20:01 |
| Year 2010-2011 | 541,910 | 2010-12-01 08:26 → 2011-12-09 12:50 |
| **Total** | **1,067,371** | 2009-12-01 → 2011-12-09 (about 2 years) |

8 original columns. 53,628 distinct invoices, 5,942 distinct known customers, 5,305 stock codes, 43 countries.

## 2.3 Data dictionary
| Column | Type in file | Feature type | Meaning | Notes |
|---|---|---|---|---|
| `Invoice` | text | Identifier | 6-digit invoice number, unique per order | Prefix `C` = cancellation (19,494 rows); prefix `A` = bad-debt adjustment (6 rows) |
| `StockCode` | text | Identifier / categorical | 5-digit product code (with an optional letter suffix) | Non-product codes exist: `POST` (postage), `DOT`, `M` (manual), `BANK CHARGES`, `AMAZONFEE`, `ADJUST`, `TEST001`, … |
| `Description` | text | Text | Product name | 4,382 missing |
| `Quantity` | integer | Numeric (discrete) | Units on the invoice line | Negative for returns/cancellations (22,950 rows) |
| `InvoiceDate` | datetime | Datetime | Date and time of the invoice | Used to build time windows |
| `Price` | float | Numeric (continuous) | Unit price in GBP (£) | 6,207 rows with price ≤ 0 (adjustments, free items, bad debt) |
| `Customer ID` | float (5-digit ID) | Identifier | Customer number | 243,007 missing (22.8%); stored as float in the file, converted to integer text |
| `Country` | text | Categorical (nominal) | Customer's country | 43 values; the UK accounts for about 83% of revenue |

**Derived during cleaning:** `Revenue = Quantity × Price`, `IsCancellation`, `IsValidPurchase`.
**Derived at customer level:** the engineered features and the target `churn_90d` (see the problem statement).

## 2.4 Data-quality summary (raw)
| Issue | Rows | Planned treatment |
|---|---:|---|
| Missing `Customer ID` | 243,007 (22.8%) | Kept for transaction-level EDA, excluded from customer-level modelling (cannot follow these buyers over time) |
| Missing `Description` | 4,382 (0.4%) | Not needed for modelling. Mostly zero-price stock adjustments |
| Exact duplicate rows (combined) | 34,335 | Removed (keep first) |
| … of which from the **sheet overlap** | ~22,200 | The two sheets both contain 1–9 Dec 2010 (1,088 shared invoices, 22,523 rows). Stacking them without deduplication would double-count that week |
| Cancellation invoices (`C…`) | 19,494 | Not counted as purchases. Used for a customer "return rate" feature |
| Negative quantity | 22,950 | As above |
| Price ≤ 0 | 6,207 | Excluded from purchases |

After exact de-duplication, **1,033,036** rows remain. Of these, **779,425** are valid purchase lines (non-cancelled, Quantity > 0, Price > 0) with a known customer: **5,878 customers, 36,969 invoices, about £17.37 M revenue**.

## 2.5 Target distribution (preview)
`churn_90d` is engineered, not provided. For each monthly cutoff, eligible customers are those with a valid purchase in the previous 365 days:

| Cutoff | Eligible customers | Churn rate |
|---|---:|---:|
| 2010-12-01 | 4,266 | 66.9% |
| 2011-01-01 | 4,231 | 67.1% |
| 2011-02-01 | 4,237 | 65.5% |
| 2011-03-01 | 4,276 | 63.2% |
| 2011-04-01 | 4,332 | 62.3% |
| 2011-05-01 | 4,332 | 61.3% |
| 2011-06-01 | 4,338 | 63.4% |
| 2011-07-01 | 4,326 | 61.4% |
| 2011-08-01 | 4,337 | 56.8% |
| 2011-09-01 | 4,335 | 50.5% |

The classes are moderately imbalanced, with **churn as the majority class**. The rate falls for cutoffs whose 90-day window covers the Sep–Nov pre-Christmas peak, a seasonal shift that the time-based test split will expose. 28.7% of known customers purchased on only one day.

## 2.6 Why this dataset fits the problem
1. **Real transactions** with customer IDs and timestamps, which is exactly what is needed to measure whether, and when, a customer buys again.
2. **Two years of history** allow several historical cutoffs, each with a 365-day feature window and a 90-day outcome window, plus a time-ordered train/validation/test split.
3. **Enough volume:** about 4,300 eligible customers per cutoff, so roughly 40k labelled snapshots across cutoffs.
4. **Realistic messiness** (cancellations, missing IDs, duplicates, non-product codes) exercises the cleaning steps the course requires.
5. **Openly licensed (CC BY 4.0)** and widely cited, so results can be reproduced.

**Limitations:** one retailer, mostly UK wholesale customers, 2009–2011. There is no customer demographic or marketing data. "Churn" is a behavioural proxy, not a cancelled contract.
