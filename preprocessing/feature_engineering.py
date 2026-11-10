
import numpy as np
import pandas as pd


def create_discount_features(df):
    """
    Create discount-related features from the original
    transaction discount columns.

    Original discount values are negative, so they are
    converted into positive discount amounts.
    """

    df = df.copy()

    # Convert recorded discounts into positive amounts.
    # Round monetary values to two decimal places and
    # remove negative values.
    df["retail_discount_amount"] = (
        -df["RETAIL_DISC"].round(2)
    ).clip(lower=0)

    df["coupon_discount_amount"] = (
        -df["COUPON_DISC"].round(2)
    ).clip(lower=0)

    df["coupon_match_discount_amount"] = (
        -df["COUPON_MATCH_DISC"].round(2)
    ).clip(lower=0)

    discount_cols = [
        "retail_discount_amount",
        "coupon_discount_amount",
        "coupon_match_discount_amount"
    ]

    # Calculate total discount
    df["total_discount"] = (
        df[discount_cols].sum(axis=1)
    )

    # Estimate original value before discounts
    df["estimated_original_value"] = (
        df["SALES_VALUE"].round(2)
        + df["total_discount"]
    )

    # Calculate discount percentage only for
    # transactions with positive original value
    df["discount_percentage"] = np.nan

    valid = df["estimated_original_value"] > 0

    df.loc[valid, "discount_percentage"] = (
        df.loc[valid, "total_discount"]
        / df.loc[valid, "estimated_original_value"]
    ) * 100

    return df


def validate_discount_features(df):
    """Return validation statistics for discount features."""

    return {
        "negative_total_discounts": int(
            (df["total_discount"] < 0).sum()
        ),
        "non_positive_original_values": int(
            (df["estimated_original_value"] <= 0).sum()
        ),
        "missing_discount_percentages": int(
            df["discount_percentage"].isna().sum()
        ),
        "infinite_discount_percentages": int(
            np.isinf(
                df["discount_percentage"].dropna()
            ).sum()
        ),
        "discount_percentage_summary": (
            df["discount_percentage"].describe()
        )
    }
