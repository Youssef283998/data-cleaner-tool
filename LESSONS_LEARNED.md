# Lessons Learned: v1 → v2

A documented post-mortem of every fatal bug in v1 of this tool, why each
happened, and how v2 fixes it. This document exists so future contributors
don't repeat the same mistakes.

## Bug 1: Hardcoded sandbox path

**v1 behavior:** Without `--output`, the file was written to
`/home/z/my-project/download/` — a path that exists in the developer's
sandbox but not on a customer's machine. The tool would either fail or
create strange directories.

**v2 fix:** Default output goes next to the input file with a `_cleaned`
suffix. No sandbox paths anywhere.

## Bug 2: Phone numbers converted to float

**v1 behavior:** `0612345678` became `612345678.0` — the leading zero was
lost AND the value became a float. The root cause was treating any
digit-only column as numeric.

**v2 fix:** The `ColumnTypeDetector` checks for:
- Leading zeros (a strong signal that it's an ID/phone, not a measurement)
- Fixed-length digit strings (also a signal)
- Column name patterns (`phone`, `tel`, `cin`, `id`, etc.)
- Content patterns (Moroccan CIN format `AB123456`)

Phone/ID/CIN columns are preserved as strings. Test:
`tests/test_data_cleaner.py::TestIDPreservation::test_phone_keeps_leading_zero`

## Bug 3: Missing phones imputed with median

**v1 behavior:** A missing phone number was filled with `633950616.5` —
the median of the other phone numbers. This invented a non-existent
phone number.

**v2 fix:** Identifiers (phone, CIN, ID) are NEVER statistically
imputed. Missing values are filled with an empty string (configurable).
Test: `test_phone_missing_not_imputed`

## Bug 4: `title()` on everything broke emails, names, and formulas

**v1 behavior:**
- `AHMED.BENALI@Example.COM` → `Ahmed.Benali@Example.Com` (broken email)
- `McDonald` → `Mcdonald` (broken name)
- `=SUM(A1:A9)` → `=Sum(A1:A9)` (broken formula)

**v2 fix:**
- Emails are auto-detected and lowercased (never title-cased)
- `_safe_title()` handles `Mc`, `Mac`, `O'Brien` cases
- Formulas (cells starting with `= + - @`) are never title-cased
- "auto" casing only title-cases name-like columns (nom, prenom, ville)
- Default is "auto"; user can override per-column via YAML config
- Tests: `test_email_lowercased_not_titled`, `test_mcdonald_not_mcdonald`,
  `test_formula_not_titled`

## Bug 5: Datetime detection failed on pandas 3

**v1 behavior:** On pandas 3+, string columns have dtype `str` (not
`object`). The check `pd.api.types.is_object_dtype()` returned False,
so datetime detection was skipped. Dates stayed as strings, and the
`Unknown` placeholder got inserted into date columns.

**v2 fix:** Use `pd.api.types.is_string_dtype()` which works for both
`str` and `object` dtypes. Also, datetime detection is done via content
sampling (try to parse a sample, check success ratio).

## Bug 6: Dates parsed as mm/dd instead of dd/mm

**v1 behavior:** `03/04/2024` became `2024-03-04` (March 4) instead of
`2024-04-03` (April 3). Morocco and most of the world uses dd/mm/yyyy.

**v2 fix:** `dayfirst=True` by default. Configurable via `--date-format`.
Test: `test_dd_mm_yyyy_parsed_correctly`

## Bug 7: Missing dates forward-filled from previous row

**v1 behavior:** A missing date was filled with the previous row's date
via `ffill()`. This invented data — a customer who didn't provide a
birth date was assigned the previous customer's birth date.

**v2 fix:** Missing dates are left as `NaT` (or filled with a
configurable value via `--date-fill`, default empty). NEVER forward-filled.
Test: `test_missing_date_not_ffilled`

## Bug 8: Arabic text silently corrupted

**v1 behavior:** An Excel file saved with Windows-1256 encoding was read
with `latin-1` (which never fails on any byte sequence). The result was
mojibake: `Ãíãï Èäúáí` instead of `أحمد بناني`. Only one warning line in
the report, easy to miss.

**v2 fix:**
- Use `charset-normalizer` for automatic encoding detection
- Fall back to a list of common encodings (UTF-8, Windows-1256, ISO-8859-6)
- Sanity check: detect mojibake (replacement char `�` or `Ãï` patterns)
- Warn prominently in the report
- Write output as UTF-8 with BOM by default (so Excel opens Arabic correctly)

## Bug 9: Dedup before cleaning missed variants

**v1 behavior:** `Ahmed`, `ahmed`, and `AHMED` (same person, same ID)
were treated as three different rows because dedup ran before casing
normalization.

**v2 fix:** Clean strings (strip + casing) FIRST, then dedup. Now
variants collapse to one row.
Test: `test_duplicates_after_casing_normalization`

## Bug 10: Imputation before train/test split = data leakage

**v1 behavior:** The median was computed on the entire dataset (including
the test set) before any train/test split. This is textbook data leakage.

**v2 fix:**
- `--no-imputation` flag for ML workflows (leaves NaN in place)
- Warning in audit log when imputation is used
- README documents that for ML, you should use `sklearn.pipeline.Pipeline`
  with `SimpleImputer` fitted only on the training fold
- Test: `test_no_imputation_keeps_nan`

## Bug 11: Integers became floats after imputation

**v1 behavior:** Age `34` became `34.0` after median imputation, because
`fillna()` with a float median converts the column to float.

**v2 fix:**
- Use nullable `Int64` dtype (preserves NaN without float conversion)
- After imputation, downcast back to Int64 ONLY if all values (including
  the imputation value) are integers
- If mean is non-integer (e.g., 68714.28), leave as float with a note
- Tests: `test_age_stays_integer_after_imputation`,
  `test_mean_imputation_non_integer_stays_float`

## Bug 12: CSV injection unsanitized

**v1 behavior:** A cell containing `=SUM(A1:A9)` was written as-is to
the output CSV. When opened in Excel, the formula would execute — a
classic CSV injection attack vector.

**v2 fix:**
- Cells starting with `=`, `+`, `-`, `@` are prefixed with `'`
- Exception: phone/id/cin columns only sanitize `=` and `@` (since `+`
  is a legitimate phone prefix like `+212`)
- Test: `test_formula_sanitized`

## Bug 13: UTF-8 without BOM broke Arabic in Excel

**v1 behavior:** Output was written as plain UTF-8. Excel on Windows
(often) misinterprets this as ANSI, breaking Arabic characters.

**v2 fix:** Write UTF-8 with BOM by default (`utf-8-sig`). Disable with
`--no-bom` if needed.

## Bug 14: `except Exception` hid tracebacks

**v1 behavior:** A bare `except Exception` printed only the error
message, hiding the traceback. Debugging was impossible.

**v2 fix:**
- Full traceback logged to `data_cleaner.log`
- Clean error message to stderr with a pointer to the log file
- Specific exceptions (`FileNotFoundError`, `ValueError`, `RuntimeError`)
  have their own handlers with appropriate exit codes

## Bug 15: No tests, no README, no license

**v1 behavior:** The script claimed to be "production-ready" but had
zero tests, no README, no requirements.txt, no license.

**v2 fix:**
- 22 pytest tests covering every bug above
- Full README with usage, CLI options, column config format, and this
  lessons-learned document
- `requirements.txt` with pinned dependencies
- MIT `LICENSE`
- `column_config.example.yaml` showing all per-column options

---

## Business lessons

Beyond the technical bugs, the v1 approach had business problems:

1. **"Production-ready" was an overstatement.** A tool that destroys
   phone numbers and invents dates is not production-ready. Be honest
   about maturity.

2. **Generic CSV cleaners are a commodity.** pandas, OpenRefine, Power
   Query, and pyjanitor all do this for free. To be sellable, the tool
   must specialize (e.g., Moroccan/Arabic data with phone normalization,
   CIN validation, Arabic text normalization) AND be a complete product
   (UI, per-column config, audit log, undo, support).

3. **AI-generated code has IP concerns.** Some marketplaces require
   disclosure of AI-generated code. Check the platform's terms before
   listing.

4. **Data leakage in ML preprocessing is subtle but fatal.** Always
   use `sklearn.pipeline.Pipeline` with imputers fitted on training
   data only. Document this prominently.
