
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths and dataset inventory
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_FILES = [
    "transaction_data.csv",
    "product.csv",
    "causal_data.csv",
    "hh_demographic.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "campaign_table.csv",
    "campaign_desc.csv",
]


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

def load_dataset(filename, data_dir=RAW_DATA_DIR):
    """Load one raw CSV dataset."""
    path = Path(data_dir) / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def load_all_datasets(data_dir=RAW_DATA_DIR):
    """Load all eight raw datasets into a dictionary."""
    datasets = {}

    for filename in DATA_FILES:
        name = filename.replace(".csv", "")
        datasets[name] = load_dataset(filename, data_dir)

        print(f"{name}: {datasets[name].shape}")

    return datasets


# --------------------------------------------------
# General dataset inspection
# --------------------------------------------------

def inspect_dataset(df, name):
    """Inspect dimensions, duplicates, missing values,
    first records and data types."""

    print("\n" + "=" * 50)
    print(name.upper())
    print("=" * 50)

    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\nFirst five records:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)


def inspect_all_datasets(datasets):
    """Run general inspection on every loaded dataset."""

    for name, df in datasets.items():
        inspect_dataset(df, name)


# --------------------------------------------------
# Transaction structure and numerical statistics
# --------------------------------------------------

def inspect_transaction_structure(df):
    """Inspect unique transaction entities and
    numerical transaction statistics."""

    print("Unique households:", df["household_key"].nunique())
    print("Unique baskets:", df["BASKET_ID"].nunique())
    print("Unique products:", df["PRODUCT_ID"].nunique())
    print("Unique stores:", df["STORE_ID"].nunique())
    print("Unique weeks:", df["WEEK_NO"].nunique())

    numerical_columns = [
        "QUANTITY",
        "SALES_VALUE",
        "RETAIL_DISC",
        "COUPON_DISC",
        "COUPON_MATCH_DISC",
    ]

    print("\nNumerical statistics:")
    print(df[numerical_columns].describe().T)


def inspect_basket_structure(df):
    """Inspect basket-product combinations and
    products and records per basket."""

    print("Total transaction rows:", len(df))
    print("Unique basket IDs:", df["BASKET_ID"].nunique())

    print(
        "Duplicate basket-product combinations:",
        df.duplicated(["BASKET_ID", "PRODUCT_ID"]).sum()
    )

    products_per_basket = (
        df.groupby("BASKET_ID")["PRODUCT_ID"].nunique()
    )

    transactions_per_basket = df.groupby("BASKET_ID").size()

    print("\nProducts per basket:")
    print(products_per_basket.describe())

    print("\nTransactions per basket:")
    print(transactions_per_basket.describe())


# --------------------------------------------------
# Dataset relationship checks
# --------------------------------------------------

def inspect_product_household_coverage(
    transactions,
    products,
    demographics
):
    """Check unmatched product IDs and household IDs."""

    transaction_products = set(transactions["PRODUCT_ID"].unique())
    product_ids = set(products["PRODUCT_ID"].unique())

    transaction_households = set(
        transactions["household_key"].unique()
    )
    demographic_households = set(
        demographics["household_key"].unique()
    )

    print("TRANSACTION -> PRODUCT")
    print("Transaction product IDs:", len(transaction_products))
    print("Product table IDs:", len(product_ids))
    print(
        "Unmatched product IDs:",
        len(transaction_products - product_ids)
    )

    print("\nTRANSACTION -> HOUSEHOLD DEMOGRAPHICS")
    print("Transaction households:", len(transaction_households))
    print("Demographic households:", len(demographic_households))
    print(
        "Unmatched households:",
        len(transaction_households - demographic_households)
    )


# --------------------------------------------------
# Unusual transaction values
# --------------------------------------------------

def inspect_transaction_values(df):
    """Inspect quantity, sales and discount values
    without removing or changing any records."""

    print("Quantity value counts:")
    print(df["QUANTITY"].value_counts().head(20))

    print("\nQuantity percentiles:")
    print(
        df["QUANTITY"].quantile(
            [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1]
        )
    )

    print("\nRecords with quantity above 100:")
    print(
        df[df["QUANTITY"] > 100]
        .sort_values("QUANTITY", ascending=False)
        .head(15)
    )

    print("\nZero sales records:", (df["SALES_VALUE"] == 0).sum())
    print("Negative sales records:", (df["SALES_VALUE"] < 0).sum())

    print("\nSales value distribution:")
    print(
        df["SALES_VALUE"].describe(
            percentiles=[0.90, 0.95, 0.99, 0.999]
        )
    )

    print("\nPositive retail discount records:")
    positive_retail = df[df["RETAIL_DISC"] > 0]

    print("Total positive records:", len(positive_retail))
    print(
        "Meaningful positive records:",
        (positive_retail["RETAIL_DISC"] > 0.001).sum()
    )

    discount_columns = [
        "RETAIL_DISC",
        "COUPON_DISC",
        "COUPON_MATCH_DISC",
    ]

    for column in discount_columns:
        print(f"\n{column}")
        print("Negative:", (df[column] < 0).sum())
        print("Zero:", (df[column] == 0).sum())
        print("Positive:", (df[column] > 0).sum())


# --------------------------------------------------
# Coupon mapping checks
# --------------------------------------------------

def inspect_coupon_duplicates(coupon):
    """Inspect exact duplicates and repeated coupon mappings."""

    print("Total rows:", len(coupon))
    print("Exact duplicate rows:", coupon.duplicated().sum())

    print("Unique coupon UPCs:", coupon["COUPON_UPC"].nunique())
    print("Unique products:", coupon["PRODUCT_ID"].nunique())
    print("Unique campaigns:", coupon["CAMPAIGN"].nunique())

    print(
        "Duplicate coupon-product-campaign combinations:",
        coupon.duplicated(
            subset=["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"]
        ).sum()
    )

    print(
        "Duplicate product-campaign combinations:",
        coupon.duplicated(
            subset=["PRODUCT_ID", "CAMPAIGN"]
        ).sum()
    )

    print(
        "Duplicate coupon UPCs:",
        coupon.duplicated(subset=["COUPON_UPC"]).sum()
    )


# --------------------------------------------------
# Causal data key validation
# --------------------------------------------------

def inspect_causal_keys(causal):
    """Check uniqueness of product-store-week keys."""

    keys = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]

    print("Total causal rows:", len(causal))
    print(
        "Unique key combinations:",
        causal[keys].drop_duplicates().shape[0]
    )
    print(
        "Duplicate key combinations:",
        causal.duplicated(keys).sum()
    )
