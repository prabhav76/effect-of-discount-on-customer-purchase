
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROMOTION_KEYS = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]


# --------------------------------------------------
# Merge transactions with promotional information
# --------------------------------------------------

def merge_promotional_data(transactions, causal):
    """
    Match relevant promotional records to transactions.

    Multiple promotional records for the same product, store
    and week are aggregated before merging.
    """

    # Identify product-store-week combinations in transactions
    transaction_keys = transactions[
        PROMOTION_KEYS
    ].drop_duplicates()

    # Keep only promotional records relevant to transactions
    causal_relevant = causal.merge(
        transaction_keys,
        on=PROMOTION_KEYS,
        how="inner"
    )

    # Aggregate promotional information
    causal_agg = (
        causal_relevant.groupby(PROMOTION_KEYS)
        .agg(
            display_values=(
                "display",
                lambda x: ",".join(sorted(x.unique()))
            ),
            mailer_values=(
                "mailer",
                lambda x: ",".join(sorted(x.unique()))
            ),
            promotion_records=("display", "size")
        )
        .reset_index()
    )

    # Validate aggregation
    assert (
        causal_agg["promotion_records"].sum()
        == len(causal_relevant)
    ), "Some promotional records were lost during aggregation."

    assert not causal_agg.duplicated(
        PROMOTION_KEYS
    ).any(), "Duplicate promotional keys found."

    # Merge promotional information with transactions
    transactions_promo = transactions.merge(
        causal_agg,
        on=PROMOTION_KEYS,
        how="left",
        validate="many_to_one"
    )

    # Ensure all transaction rows are preserved
    assert len(transactions_promo) == len(transactions), (
        "Transaction row count changed after promotional merge."
    )

    return transactions_promo


# --------------------------------------------------
# Merge product information with transactions
# --------------------------------------------------

def merge_product_information(transactions_promo, product):
    """Add commodity, sub-commodity and brand information."""

    product_info = product[
        [
            "PRODUCT_ID",
            "COMMODITY_DESC",
            "SUB_COMMODITY_DESC",
            "BRAND"
        ]
    ]

    transactions_enriched = transactions_promo.merge(
        product_info,
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one"
    )

    # Ensure all transaction rows are preserved
    assert len(transactions_enriched) == len(transactions_promo), (
        "Transaction row count changed after product merge."
    )

    return transactions_enriched


# --------------------------------------------------
# Save processed datasets
# --------------------------------------------------

def save_processed_dataset(df, filename):
    """Save a processed dataset in data/processed."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / filename

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return output_path


# --------------------------------------------------
# Complete merging workflow
# --------------------------------------------------

def build_processed_datasets():
    """Create promotional and product-enriched transaction datasets."""

    transactions = pd.read_csv(
        RAW_DATA_DIR / "transaction_data.csv"
    )

    causal = pd.read_csv(
        RAW_DATA_DIR / "causal_data.csv"
    )

    product = pd.read_csv(
        RAW_DATA_DIR / "product.csv"
    )

    # Merge promotional information
    transactions_promo = merge_promotional_data(
        transactions,
        causal
    )

    print("Original transaction rows:", len(transactions))
    print("After promotional merge:", len(transactions_promo))

    matched = transactions_promo["promotion_records"].notna().sum()
    unmatched = transactions_promo["promotion_records"].isna().sum()

    print("Transactions with promotional records:", matched)
    print("Transactions without promotional records:", unmatched)
    print(
        "Promotion coverage (%):",
        round(matched / len(transactions_promo) * 100, 2)
    )

    save_processed_dataset(
        transactions_promo,
        "transactions_promo.csv"
    )

    # Add product information
    transactions_enriched = merge_product_information(
        transactions_promo,
        product
    )

    print("After product merge:", len(transactions_enriched))
    print(
        "Missing commodity matches:",
        transactions_enriched["COMMODITY_DESC"].isna().sum()
    )

    save_processed_dataset(
        transactions_enriched,
        "transactions_enriched.csv"
    )

    return transactions_promo, transactions_enriched
