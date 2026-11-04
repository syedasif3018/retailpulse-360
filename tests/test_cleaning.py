import pandas as pd

from src.retailpulse.cleaning import clean_transactions


def _raw():
    rows = [
        # Invoice, StockCode, Desc, Qty, Date, Price, Customer, Country
        ("100", "85123A", "HEART", 6, "2010-01-04 10:00", 2.55, 12346.0, "United Kingdom"),
        ("100", "85123A", "HEART", 6, "2010-01-04 10:00", 2.55, 12346.0, "United Kingdom"),  # exact dup
        ("101", "POST", "POSTAGE", 1, "2010-01-04 11:00", 18.0, 12347.0, "France"),          # non-product
        ("102", "23166", "JAR", 74215, "2011-01-18 10:01", 1.04, 12348.0, "United Kingdom"),  # reversed
        ("C103", "23166", "JAR", -74215, "2011-01-18 10:17", 1.04, 12348.0, "United Kingdom"),
        ("C104", "85123A", "HEART", -2, "2010-02-01 09:00", 2.55, 12346.0, "United Kingdom"),  # real return
        ("105", "22423", None, 1, "2010-03-01 09:00", 0.0, None, "United Kingdom"),          # zero price
        ("A106", "B", "Adjust bad debt", 1, "2010-04-01 09:00", -100.0, None, "United Kingdom"),
    ]
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["source_sheet"] = "test"
    return df


def test_cleaning_rules():
    clean, log = clean_transactions(_raw())
    assert log["dropped_exact_duplicates"] == 1
    assert log["dropped_reversed_purchase_lines"] == 1
    assert log["dropped_reversing_cancellation_lines"] == 1
    assert len(clean) == 5
    # only the original HEART purchase is a valid purchase
    valid = clean[clean["is_valid_purchase"]]
    assert valid["Invoice"].tolist() == ["100"]
    assert valid["Customer ID"].tolist() == ["12346"]  # float ID -> clean text
    # the genuine return is kept (needed for the return-rate feature)
    assert clean["is_cancellation"].sum() == 1
    assert clean.loc[clean["StockCode"] == "POST", "is_non_product"].all()
    assert clean.loc[clean["Invoice"] == "A106", "is_adjustment"].all()
    assert (clean["Description"] == "UNKNOWN").sum() == 1
