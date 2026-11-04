import pandas as pd

from src.retailpulse.audit import audit_sheet, sheet_overlap


def _sample():
    return pd.DataFrame({
        "Invoice": ["100", "C101", "102", "102"],
        "StockCode": ["A", "B", "A", "A"],
        "Description": ["Item", "Item2", "Item", "Item"],
        "Quantity": [2, -1, 1, 1],
        "InvoiceDate": ["2010-01-01", "2010-01-02", "2010-12-05", "2010-12-05"],
        "Price": [2.5, 0, 1.0, 1.0],
        "Customer ID": [10, None, 11, 11],
        "Country": ["UK", "UK", "France", "France"],
        "source_sheet": ["Y1", "Y1", "Y1", "Y2"],
    })


def test_audit_counts():
    result = audit_sheet(_sample(), "sample")
    assert result["rows"] == 4
    assert result["cancellation_rows"] == 1
    assert result["negative_quantity_rows"] == 1
    assert result["nonpositive_price_rows"] == 1
    assert result["missing_by_column"]["Customer ID"] == 1
    assert result["duplicate_rows"] == 1  # row 102 appears in both sheets
    assert result["distinct_customers_nonmissing"] == 2


def test_sheet_overlap_detects_shared_invoice():
    overlap = sheet_overlap(_sample())
    assert overlap["shared_invoice_count"] == 1
    assert overlap["rows_in_second_sheet_with_shared_invoice"] == 1
