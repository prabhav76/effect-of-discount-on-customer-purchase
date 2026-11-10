
import pandas as pd


# --------------------------------------------------
# General numerical summary
# --------------------------------------------------

def numerical_summary(df, columns=None):
    """Return descriptive statistics for numerical columns."""

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    return df[columns].describe().T


# --------------------------------------------------
# Dataset size and unique entity counts
# --------------------------------------------------

def transaction_statistics(df):
    """Return the main transaction dataset statistics."""

    return pd.DataFrame({
        "Metric": [
            "Transaction rows",
            "Unique households",
            "Unique baskets",
            "Unique products",
            "Unique stores",
            "Unique weeks",
        ],
        "Value": [
            len(df),
            df["household_key"].nunique(),
            df["BASKET_ID"].nunique(),
            df["PRODUCT_ID"].nunique(),
            df["STORE_ID"].nunique(),
            df["WEEK_NO"].nunique(),
        ],
    })


# --------------------------------------------------
# Quantity and sales distributions
# --------------------------------------------------

def quantity_statistics(df):
    """Return quantity distribution and selected percentiles."""

    return pd.DataFrame({
        "Quantity": df["QUANTITY"].describe(
            percentiles=[
                0.25, 0.50, 0.75,
                0.90, 0.95, 0.99, 0.999
            ]
        )
    })


def sales_statistics(df):
    """Return sales distribution and unusual-value counts."""

    summary = df["SALES_VALUE"].describe(
        percentiles=[0.90, 0.95, 0.99, 0.999]
    )

    return {
        "distribution": summary,
        "zero_sales_records": int((df["SALES_VALUE"] == 0).sum()),
        "negative_sales_records": int((df["SALES_VALUE"] < 0).sum()),
    }


# --------------------------------------------------
# Discount field statistics
# --------------------------------------------------

DISCOUNT_COLUMNS = [
    "RETAIL_DISC",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
]


def discount_field_statistics(df):
    """Summarize negative, zero and positive discount values."""

    results = []

    for column in DISCOUNT_COLUMNS:
        results.append({
            "Discount_Field": column,
            "Negative_Count": int((df[column] < 0).sum()),
            "Zero_Count": int((df[column] == 0).sum()),
            "Positive_Count": int((df[column] > 0).sum()),
            "Mean": df[column].mean(),
            "Median": df[column].median(),
            "Minimum": df[column].min(),
            "Maximum": df[column].max(),
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Basket statistics
# --------------------------------------------------

def basket_statistics(df):
    """Return products and transaction records per basket."""

    products_per_basket = (
        df.groupby("BASKET_ID")["PRODUCT_ID"].nunique()
    )

    transactions_per_basket = df.groupby("BASKET_ID").size()

    return {
        "products_per_basket": products_per_basket.describe(),
        "transactions_per_basket": transactions_per_basket.describe(),
        "duplicate_basket_product_combinations": int(
            df.duplicated(["BASKET_ID", "PRODUCT_ID"]).sum()
        ),
    }
