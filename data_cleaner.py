#!/usr/bin/env python3
"""
data_cleaner.py
===============

A robust, production-ready data cleaning utility built on Pandas.

Features
--------
1. Load any messy CSV or Excel (.xlsx / .xls) file automatically.
2. Detect and remove duplicate rows (with a configurable subset strategy).
3. Clean string columns:
       - Strip leading / trailing whitespace
       - Collapse multiple internal spaces
       - Normalize casing (title-case by default, configurable)
4. Handle missing values:
       - Numeric columns  -> median (default) or mean imputation
       - Categorical/text -> filled with 'Unknown' (configurable)
       - Datetime columns -> forward-fill then backward-fill
5. Generate a clean output file (CSV or Excel) with a timestamped name.
6. Print a detailed, human-readable summary report of every fix applied.

Usage (CLI mini-tool)
---------------------
    python data_cleaner.py input.csv
    python data_cleaner.py input.xlsx --output cleaned.xlsx
    python data_cleaner.py input.csv --numeric-strategy mean --text-fill "N/A"
    python data_cleaner.py input.csv --casing lower --report-only

Author: Super Z
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# Data class to hold a detailed cleaning report
# ----------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Container that records every action performed during cleaning."""

    input_file: str = ""
    output_file: str = ""
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    duplicates_removed: int = 0
    string_columns_cleaned: list = field(default_factory=list)
    numeric_missing_before: dict = field(default_factory=dict)
    numeric_missing_after: dict = field(default_factory=dict)
    numeric_strategy: str = ""
    text_missing_filled: dict = field(default_factory=dict)
    text_fill_value: str = "Unknown"
    datetime_columns_fixed: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def pretty_print(self) -> str:
        """Render the report as a human-readable multi-line string."""
        line = "=" * 70
        sections = [line, "DATA CLEANING SUMMARY REPORT".center(70), line]

        sections.append(f"Input file            : {self.input_file}")
        sections.append(f"Output file           : {self.output_file or '(not written)'}")
        sections.append(f"Elapsed time          : {self.elapsed_seconds:.3f} sec")
        sections.append("")

        sections.append("-- Shape Changes " + "-" * 53)
        sections.append(f"Rows    : {self.rows_before}  ->  {self.rows_after}"
                        f"   (removed {self.rows_before - self.rows_after})")
        sections.append(f"Columns : {self.cols_before}  ->  {self.cols_after}")
        sections.append("")

        sections.append("-- Duplicate Removal " + "-" * 49)
        sections.append(f"Duplicates removed    : {self.duplicates_removed}")
        sections.append("")

        sections.append("-- String Column Cleaning " + "-" * 44)
        if self.string_columns_cleaned:
            for col in self.string_columns_cleaned:
                sections.append(f"  - {col}")
        else:
            sections.append("  (no string columns to clean)")
        sections.append("")

        sections.append("-- Missing Numeric Imputation " + "-" * 40)
        sections.append(f"Strategy              : {self.numeric_strategy}")
        if self.numeric_missing_before:
            for col, n_before in self.numeric_missing_before.items():
                n_after = self.numeric_missing_after.get(col, 0)
                sections.append(f"  - {col:<30} missing: {n_before} -> {n_after}")
        else:
            sections.append("  (no numeric missing values found)")
        sections.append("")

        sections.append("-- Missing Text Fill " + "-" * 49)
        sections.append(f"Fill value            : '{self.text_fill_value}'")
        if self.text_missing_filled:
            for col, n in self.text_missing_filled.items():
                sections.append(f"  - {col:<30} filled: {n}")
        else:
            sections.append("  (no text missing values found)")
        sections.append("")

        if self.datetime_columns_fixed:
            sections.append("-- Datetime Coercion " + "-" * 49)
            for col in self.datetime_columns_fixed:
                sections.append(f"  - {col}")
            sections.append("")

        if self.warnings:
            sections.append("-- Warnings " + "-" * 58)
            for w in self.warnings:
                sections.append(f"  ! {w}")
            sections.append("")

        sections.append(line)
        return "\n".join(sections)


# ----------------------------------------------------------------------
# Main DataCleaner class
# ----------------------------------------------------------------------
class DataCleaner:
    """
    A reusable data-cleaning pipeline.

    Parameters
    ----------
    numeric_strategy : str
        'median' (default) or 'mean' -- how to impute missing numerics.
    text_fill : str
        Value used to fill missing categorical / text values.
    casing : str
        'title' (default), 'lower', 'upper', or 'none'.
    duplicate_subset : list[str] | None
        Column subset to consider for duplicate detection. None = all columns.
    """

    VALID_STRATEGIES = {"median", "mean"}
    VALID_CASINGS = {"title", "lower", "upper", "none"}

    def __init__(
        self,
        numeric_strategy: str = "median",
        text_fill: str = "Unknown",
        casing: str = "title",
        duplicate_subset: Optional[list] = None,
    ) -> None:
        if numeric_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"numeric_strategy must be one of {self.VALID_STRATEGIES}, "
                f"got '{numeric_strategy}'"
            )
        if casing not in self.VALID_CASINGS:
            raise ValueError(
                f"casing must be one of {self.VALID_CASINGS}, got '{casing}'"
            )

        self.numeric_strategy = numeric_strategy
        self.text_fill = text_fill
        self.casing = casing
        self.duplicate_subset = duplicate_subset
        self.report = CleaningReport(
            numeric_strategy=numeric_strategy,
            text_fill_value=text_fill,
        )

    # ------------------------------------------------------------------
    # Step 1 - Load
    # ------------------------------------------------------------------
    def load(self, file_path: str | Path) -> pd.DataFrame:
        """Load a CSV or Excel file into a DataFrame with error handling."""
        path = Path(file_path)
        self.report.input_file = str(path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                # Try UTF-8 first; fall back to latin-1 for messy encodings
                try:
                    df = pd.read_csv(path, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(path, encoding="latin-1")
                    self.report.warnings.append(
                        "CSV was not valid UTF-8; re-read with latin-1 encoding."
                    )
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                raise ValueError(
                    f"Unsupported file extension '{ext}'. "
                    "Supported: .csv, .xlsx, .xls"
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to read '{path}': {exc}") from exc

        self.report.rows_before = len(df)
        self.report.cols_before = df.shape[1]
        return df

    # ------------------------------------------------------------------
    # Step 2 - Remove duplicates
    # ------------------------------------------------------------------
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate rows and record how many were removed."""
        before = len(df)
        # Use subset if provided; otherwise consider all columns
        df = df.drop_duplicates(subset=self.duplicate_subset, keep="first")
        self.report.duplicates_removed = before - len(df)
        return df

    # ------------------------------------------------------------------
    # Step 3 - Clean string columns
    # ------------------------------------------------------------------
    def clean_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace, collapse internal spaces, normalize casing."""
        # Identify object / string columns (avoid categorical dtype for now)
        string_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in string_cols:
            # Convert to string safely, preserving NaN as NaN (not the literal "nan")
            df[col] = df[col].astype("object")

            # Define a vectorized cleaning function
            def _clean_cell(value):
                if pd.isna(value):
                    return np.nan
                if not isinstance(value, str):
                    value = str(value)
                # Strip leading/trailing whitespace
                value = value.strip()
                # Collapse multiple internal whitespace characters into one
                value = " ".join(value.split())
                # Apply casing
                if self.casing == "title":
                    value = value.title()
                elif self.casing == "lower":
                    value = value.lower()
                elif self.casing == "upper":
                    value = value.upper()
                # 'none' -> leave as-is
                # If the cell becomes empty after stripping, treat as missing
                if value == "":
                    return np.nan
                return value

            df[col] = df[col].map(_clean_cell)
            self.report.string_columns_cleaned.append(col)

        return df

    # ------------------------------------------------------------------
    # Step 4 - Handle missing values
    # ------------------------------------------------------------------
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing numerics and fill missing text categories.

        Order of operations (important):
          1. Detect & convert datetime columns FIRST (before text fill),
             so the 'Unknown' placeholder does not break date parsing.
          2. Impute numeric columns (median / mean).
          3. Fill text / categorical columns with the configured fill value.
          4. Forward-fill then backward-fill any remaining datetime gaps.
        """
        # --- 4a. Datetime detection (before text fill!) ---
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                # Take a non-null sample to decide whether the column looks like dates
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    continue
                # Require *every* non-null value to parse as a date
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        converted = pd.to_datetime(
                            non_null, errors="raise", format="mixed"
                        )
                except (ValueError, TypeError):
                    continue  # Not a datetime column -- skip silently

                # Re-attach NaN positions and assign back
                full = pd.Series(index=df.index, dtype="datetime64[ns]")
                full.loc[non_null.index] = converted
                df[col] = full
                self.report.datetime_columns_fixed.append(col)

        # --- 4b. Numeric columns ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            missing_mask = df[col].isna()
            n_missing = int(missing_mask.sum())
            self.report.numeric_missing_before[col] = n_missing
            if n_missing == 0:
                self.report.numeric_missing_after[col] = 0
                continue

            if self.numeric_strategy == "median":
                fill_value = df[col].median()
            else:  # mean
                fill_value = df[col].mean()

            # If the entire column is NaN, mean/median returns NaN -- warn & use 0
            if pd.isna(fill_value):
                fill_value = 0
                self.report.warnings.append(
                    f"Column '{col}' is entirely NaN; filled with 0."
                )

            df[col] = df[col].fillna(fill_value)
            self.report.numeric_missing_after[col] = int(df[col].isna().sum())

        # --- 4c. Text / categorical columns ---
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in text_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                df[col] = df[col].fillna(self.text_fill)
                self.report.text_missing_filled[col] = n_missing

        # --- 4d. Datetime gaps (forward-fill then backward-fill) ---
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        for col in datetime_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                df[col] = df[col].ffill().bfill()
                self.report.warnings.append(
                    f"Datetime column '{col}' had {n_missing} missing value(s); "
                    "filled via forward/backward fill."
                )

        return df

    # ------------------------------------------------------------------
    # Step 5 - Save output
    # ------------------------------------------------------------------
    def save(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Save the cleaned DataFrame to disk. Auto-name if not provided."""
        if output_path is None:
            stem = Path(self.report.input_file).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/home/z/my-project/download/{stem}_cleaned_{timestamp}.csv"

        output_path = str(output_path)
        ext = Path(output_path).suffix.lower()

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            if ext == ".csv":
                df.to_csv(output_path, index=False, encoding="utf-8")
            elif ext in (".xlsx", ".xls"):
                df.to_excel(output_path, index=False)
            else:
                # Default to CSV
                output_path = output_path + ".csv"
                df.to_csv(output_path, index=False, encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to write output file: {exc}") from exc

        self.report.output_file = output_path
        self.report.rows_after = len(df)
        self.report.cols_after = df.shape[1]
        return output_path

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def clean(
        self,
        input_path: str | Path,
        output_path: Optional[str] = None,
        report_only: bool = False,
    ) -> tuple[pd.DataFrame, CleaningReport]:
        """Run the full cleaning pipeline end-to-end."""
        start = datetime.now()

        # 1. Load
        df = self.load(input_path)

        # 2. Remove duplicates
        df = self.remove_duplicates(df)

        # 3. Clean strings
        df = self.clean_strings(df)

        # 4. Handle missing values
        df = self.handle_missing(df)

        # 5. Save (unless report-only mode)
        if not report_only:
            self.save(df, output_path)
        else:
            self.report.rows_after = len(df)
            self.report.cols_after = df.shape[1]

        self.report.elapsed_seconds = (datetime.now() - start).total_seconds()
        return df, self.report


# ----------------------------------------------------------------------
# CLI mini-tool
# ----------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="data_cleaner",
        description="Clean messy CSV / Excel files with Pandas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python data_cleaner.py data.csv
  python data_cleaner.py data.xlsx --output cleaned.xlsx
  python data_cleaner.py data.csv --numeric-strategy mean --text-fill "N/A"
  python data_cleaner.py data.csv --casing lower --report-only
""",
    )
    parser.add_argument("input", help="Path to the input CSV or Excel file")
    parser.add_argument(
        "-o", "--output",
        help="Path to the cleaned output file (default: auto-named in /download)",
    )
    parser.add_argument(
        "--numeric-strategy",
        choices=["median", "mean"],
        default="median",
        help="Imputation strategy for numeric columns (default: median)",
    )
    parser.add_argument(
        "--text-fill",
        default="Unknown",
        help="Fill value for missing text categories (default: 'Unknown')",
    )
    parser.add_argument(
        "--casing",
        choices=["title", "lower", "upper", "none"],
        default="title",
        help="Casing normalization for string columns (default: title)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the report only; do not write an output file",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    try:
        cleaner = DataCleaner(
            numeric_strategy=args.numeric_strategy,
            text_fill=args.text_fill,
            casing=args.casing,
        )
        _, report = cleaner.clean(
            input_path=args.input,
            output_path=args.output,
            report_only=args.report_only,
        )
        print(report.pretty_print())
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
