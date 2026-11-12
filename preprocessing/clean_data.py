"""Create validated discount features for the analysis notebooks.

This step intentionally preserves every transaction row. Potential returns,
zero-sales records, and extreme quantities are flagged for analysis rather than
silently removed.
"""

from pathlib import Path
import argparse

import pandas as pd

try:  # Support both ``python preprocessing/clean_data.py`` and package imports.
    from preprocessing.feature_engineering import (
        create_discount_features,
        validate_discount_features,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from feature_engineering import (
        create_discount_features,
        validate_discount_features,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "transactions_enriched.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "transactions_cleaned.csv"
REQUIRED_COLUMNS = {
    "SALES_VALUE",
    "RETAIL_DISC",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
}


def clean_transaction_data(transactions):
    """Add positive discount amounts and a guarded discount percentage.

    The function does not drop rows or replace original source columns.
    """

    missing = sorted(REQUIRED_COLUMNS - set(transactions.columns))
    if missing:
        raise ValueError(
            "Cannot calculate discount features; missing columns: "
            + ", ".join(missing)
        )

    cleaned = create_discount_features(transactions)
    cleaned["discount_percentage"] = cleaned["discount_percentage"].round(6)
    return cleaned


def process_file(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, chunksize=250_000):
    """Process an enriched transaction CSV in chunks and report validation."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input not found: {input_path}. Run preprocessing/merge_datasets.py first."
        )
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    rows_written = 0
    invalid_pct_count = 0
    negative_discount_count = 0
    non_positive_original_count = 0

    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(input_path, chunksize=chunksize, low_memory=False)
        ):
            cleaned = clean_transaction_data(chunk)
            validation = validate_discount_features(cleaned)
            rows_written += len(cleaned)
            invalid_pct_count += validation["missing_discount_percentages"]
            negative_discount_count += validation["negative_total_discounts"]
            non_positive_original_count += validation[
                "non_positive_original_values"
            ]

            cleaned.to_csv(
                partial_path,
                index=False,
                mode="w" if chunk_number == 0 else "a",
                header=chunk_number == 0,
            )

        if rows_written == 0:
            raise ValueError(f"Input contains no transaction rows: {input_path}")

        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    report = {
        "rows_written": rows_written,
        "missing_discount_percentages": invalid_pct_count,
        "negative_total_discounts": negative_discount_count,
        "non_positive_original_values": non_positive_original_count,
    }
    print(f"Saved {rows_written:,} rows to {output_path}")
    print("Validation summary:")
    for name, value in report.items():
        print(f"  {name}: {value:,}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    process_file(args.input, args.output, args.chunksize)


if __name__ == "__main__":
    main()
