"""Tests for data_cleaner.py v2.

Run with: pytest tests/ -v
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Add parent dir to path so we can import data_cleaner
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_cleaner import DataCleaner, ColumnTypeDetector

FIXTURES = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------
# Column type detector tests
# ----------------------------------------------------------------------
class TestColumnTypeDetector:
    def test_detects_id_column_by_name(self):
        s = pd.Series(["C001", "C002", "C003"])
        assert ColumnTypeDetector.detect("client_id", s) == "id"

    def test_detects_phone_column(self):
        s = pd.Series(["0612345678", "+212661234567", "05225123456"])
        assert ColumnTypeDetector.detect("telephone", s) == "phone"

    def test_detects_cin_column(self):
        s = pd.Series(["AB123456", "CD789012", "EF345678"])
        assert ColumnTypeDetector.detect("cin", s) == "cin"

    def test_detects_email_column(self):
        s = pd.Series(["a@b.com", "c@d.com", "e@f.com"])
        assert ColumnTypeDetector.detect("email", s) == "email"

    def test_detects_numeric_column(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert ColumnTypeDetector.detect("age", s) == "numeric"

    def test_phone_with_leading_zero_not_numeric(self):
        """Critical: 0612345678 must NOT be treated as numeric (loses zero)."""
        s = pd.Series(["0612345678", "0612345679", "0612345680"])
        detected = ColumnTypeDetector.detect("contact", s)
        assert detected != "numeric", \
            "Phone numbers with leading zeros must not be detected as numeric"

    def test_detects_datetime_string(self):
        s = pd.Series(["03/04/1990", "15/08/1992", "22/11/1988"])
        assert ColumnTypeDetector.detect("date_naissance", s) == "datetime"


# ----------------------------------------------------------------------
# Phone / ID preservation tests (the v1 fatal bug)
# ----------------------------------------------------------------------
class TestIDPreservation:
    def test_phone_keeps_leading_zero(self):
        """The critical bug: 0612345678 must stay '0612345678', not become 612345678.0."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "telephone": ["0612345678", "+212661234567", "05225123456"],
            "value": [1, 2, 3],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.handle_missing(df, audits)

        assert df["telephone"].iloc[0] == "0612345678", \
            f"Leading zero lost! Got: {df['telephone'].iloc[0]!r}"
        assert "." not in str(df["telephone"].iloc[0]), \
            "Phone was converted to float!"

    def test_phone_missing_not_imputed(self):
        """Missing phone numbers must NOT be filled with median."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "telephone": ["0612345678", None, "0612345679"],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.handle_missing(df, audits)

        assert df["telephone"].iloc[1] == "", \
            f"Phone imputed with median! Got: {df['telephone'].iloc[1]!r}"
        assert "." not in str(df["telephone"].iloc[1]), \
            "Phone imputed as float!"

    def test_cin_preserved(self):
        """CIN must stay as string with letters + digits intact."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "cin": ["AB123456", "CD789012", None],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.handle_missing(df, audits)

        assert df["cin"].iloc[0] == "AB123456"
        assert df["cin"].iloc[2] == ""  # not imputed


# ----------------------------------------------------------------------
# Casing tests (the v1 title() bug)
# ----------------------------------------------------------------------
class TestCasing:
    def test_email_lowercased_not_titled(self):
        """Email must be lowercased, NOT title-cased to 'Ahmed.Benali@Example.Com'."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "email": ["AHMED.BENALI@Example.COM", "fatima.alami@EXAMPLE.com"],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)

        assert df["email"].iloc[0] == "ahmed.benali@example.com", \
            f"Email not lowercased! Got: {df['email'].iloc[0]!r}"

    def test_mcdonald_not_mcdonald(self):
        """McDonald must stay 'McDonald', not become 'Mcdonald'."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "nom": ["MCDONALD", "O'Brien"],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)

        assert df["nom"].iloc[0] == "McDonald", \
            f"McDonald broken! Got: {df['nom'].iloc[0]!r}"
        assert df["nom"].iloc[1] == "O'Brien", \
            f"O'Brien broken! Got: {df['nom'].iloc[1]!r}"

    def test_formula_not_titled(self):
        """Cells starting with = must NOT be title-cased (=Sum instead of =SUM)."""
        cleaner = DataCleaner(sanitize_injection=False)
        df = pd.DataFrame({
            "valeur": ["=SUM(A1:A9)", "=AVG(B1:B9)"],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)

        assert df["valeur"].iloc[0] == "=SUM(A1:A9)", \
            f"Formula title-cased! Got: {df['valeur'].iloc[0]!r}"


# ----------------------------------------------------------------------
# Duplicate tests (the v1 ordering bug)
# ----------------------------------------------------------------------
class TestDuplicateOrdering:
    def test_duplicates_after_casing_normalization(self):
        """Ahmed / ahmed / AHMED with same ID must all collapse to 1 row."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "client_id": ["C001", "C001", "C002"],
            "nom": ["  benali  ", "BENALI", "Alami"],
            "prenom": ["ahmed", "AHMED", "Fatima"],
            "email": ["ahmed@x.com", "AHMED@x.com", "f@x.com"],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.remove_duplicates(df, audits)

        assert len(df) == 2, \
            f"Dedup before cleaning missed variants! Rows: {len(df)}"


# ----------------------------------------------------------------------
# Datetime tests
# ----------------------------------------------------------------------
class TestDatetime:
    def test_dd_mm_yyyy_parsed_correctly(self):
        """03/04/2024 must be 3 April 2024, NOT 4 March 2024."""
        cleaner = DataCleaner(date_format="dd/mm/yyyy")
        df = pd.DataFrame({"date_naissance": ["03/04/1990", "15/08/1992"]})
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)

        assert df["date_naissance"].iloc[0].day == 3, \
            f"Day wrong! Got: {df['date_naissance'].iloc[0]}"
        assert df["date_naissance"].iloc[0].month == 4, \
            f"Month wrong! Got: {df['date_naissance'].iloc[0]}"

    def test_missing_date_not_ffilled(self):
        """Missing dates must NOT be filled with previous row's date."""
        cleaner = DataCleaner(date_format="dd/mm/yyyy", date_fill=pd.NaT)
        df = pd.DataFrame({"date_naissance": ["03/04/1990", None, "15/08/1992"]})
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.handle_missing(df, audits)

        assert pd.isna(df["date_naissance"].iloc[1]), \
            f"Date was forward-filled! Got: {df['date_naissance'].iloc[1]}"


# ----------------------------------------------------------------------
# Integer preservation tests
# ----------------------------------------------------------------------
class TestIntegerPreservation:
    def test_age_stays_integer_after_imputation(self):
        """Age 34 must stay 34, not become 34.0."""
        cleaner = DataCleaner()
        df = pd.DataFrame({"age": [34, 28, None, 45]})
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.handle_missing(df, audits)

        # After imputation, age should be Int64 (nullable), values should be integers
        assert str(df["age"].dtype) in ("Int64", "int64"), \
            f"Age dtype is {df['age'].dtype}, should be Int64"
        # Check that no value has a decimal point
        for val in df["age"]:
            assert float(val).is_integer(), \
                f"Age value {val} is not an integer!"


# ----------------------------------------------------------------------
# CSV injection tests
# ----------------------------------------------------------------------
class TestCSVInjection:
    def test_formula_sanitized(self):
        """=SUM(A1:A9) must become '=SUM(A1:A9) in CSV output."""
        cleaner = DataCleaner()
        df = pd.DataFrame({"valeur": ["=SUM(A1:A9)", "normal"]})
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.sanitize_csv_injection(df, audits)

        assert df["valeur"].iloc[0].startswith("'"), \
            f"Formula not sanitized! Got: {df['valeur'].iloc[0]!r}"


# ----------------------------------------------------------------------
# Encoding tests
# ----------------------------------------------------------------------
class TestEncoding:
    def test_arabic_preserved(self):
        """Arabic text must be preserved, not turned into mojibake."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "nom": ["بيناني", "أحمد"],
            "ville": ["فاس", "مكناس"],
        })
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)

        assert df["nom"].iloc[0] == "بيناني", \
            f"Arabic corrupted! Got: {df['nom'].iloc[0]!r}"

    def test_windows1256_csv_read_correctly(self):
        """A Windows-1256-encoded CSV (Arabic Windows) must be read correctly."""
        fixture = FIXTURES / "messy_arabic_win1256.csv"
        if not fixture.exists():
            pytest.skip("Windows-1256 fixture not generated")

        cleaner = DataCleaner()
        df, report = cleaner.clean(fixture, report_only=True)

        # Arabic must be preserved (not mojibake)
        assert "بناني" in df["nom"].values, \
            f"Arabic not preserved! Got: {df['nom'].tolist()}"
        assert "Ã" not in str(df["nom"].iloc[0]), \
            "Mojibake detected in Arabic column"

    def test_phone_not_treated_as_csv_injection(self):
        """+212 (phone prefix) must NOT be sanitized as CSV injection."""
        cleaner = DataCleaner()
        df = pd.DataFrame({"telephone": ["+212661234567", "0612345678"]})
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.sanitize_csv_injection(df, audits)

        # +212 should be preserved (not prefixed with ')
        assert df["telephone"].iloc[0] == "+212661234567", \
            f"Phone + prefix wrongly sanitized! Got: {df['telephone'].iloc[0]!r}"


# ----------------------------------------------------------------------
# No-imputation (ML workflow) tests
# ----------------------------------------------------------------------
class TestNoImputation:
    def test_no_imputation_keeps_nan(self):
        """--no-imputation must leave NaN in place (for ML pipelines)."""
        cleaner = DataCleaner(no_imputation=True)
        df = pd.DataFrame({"age": [34, 28, None, 45]})
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.handle_missing(df, audits)

        assert pd.isna(df["age"].iloc[2]), \
            "no_imputation=True but NaN was filled!"

    def test_mean_imputation_non_integer_stays_float(self):
        """If mean is non-integer (e.g. 68714.28), column must become float, not crash."""
        cleaner = DataCleaner(numeric_strategy="mean")
        df = pd.DataFrame({"salaire": [65000, 72000, None, 81000]})
        audits = cleaner.detect_types(df)
        df = cleaner.clean_strings(df, audits)
        df = cleaner.handle_missing(df, audits)

        # Should not crash, should be float
        assert str(df["salaire"].dtype).startswith("float"), \
            f"Non-integer mean should leave column as float, got {df['salaire'].dtype}"
        assert not pd.isna(df["salaire"].iloc[2]), "NaN not filled"


# ----------------------------------------------------------------------
# Full integration test on the messy Moroccan fixture
# ----------------------------------------------------------------------
class TestEndToEnd:
    def test_messy_moroccan_csv(self, tmp_path):
        """Run the full pipeline on the messy Moroccan fixture."""
        fixture = FIXTURES / "messy_moroccan.csv"
        if not fixture.exists():
            pytest.skip("Fixture not generated")

        # Use duplicate_subset to exclude client_id (different IDs but same person)
        cleaner = DataCleaner(
            duplicate_subset=["nom", "prenom", "cin", "telephone", "email"]
        )
        output = tmp_path / "cleaned.csv"
        df, report = cleaner.clean(fixture, output_path=str(output))

        # 1. Duplicates removed (C001 and C005 should collapse)
        assert report.duplicates_removed >= 1

        # 2. Phone numbers preserved as strings (no .0)
        for val in df["telephone"].dropna():
            assert "." not in str(val), \
                f"Phone corrupted to float: {val!r}"

        # 3. CIN preserved as string
        for val in df["cin"].dropna():
            assert val != "" and not str(val).replace(".", "").isdigit() or val == "", \
                f"CIN corrupted: {val!r}"

        # 4. Email lowercased
        for val in df["email"].dropna():
            if val and "@" in val:
                assert val == val.lower(), \
                    f"Email not lowercased: {val!r}"

        # 5. Age is integer (not 34.0)
        assert str(df["age"].dtype) in ("Int64", "int64"), \
            f"Age dtype should be Int64, got {df['age'].dtype}"

        # 6. McDonald preserved (not Mcdonald)
        if "McDonald" in df["nom"].values:
            assert "Mcdonald" not in df["nom"].values, \
                "McDonald was title-cased to Mcdonald"

        # 7. Audit log written
        audit_log = output.with_suffix(".audit.json")
        assert audit_log.exists(), "Audit log not written"

        # 8. Output file is valid CSV
        assert output.exists()
        df_back = pd.read_csv(output)
        assert len(df_back) == len(df)


if __name__ == "__main__":
    # Run with: python tests/test_data_cleaner.py
    pytest.main([__file__, "-v", "--tb=short"])
