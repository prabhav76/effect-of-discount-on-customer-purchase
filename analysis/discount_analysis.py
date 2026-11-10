
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


PROMOTION_COLUMNS = [
    "display_values",
    "mailer_values",
    "promotion_records"
]

CHUNK_SIZE = 250_000


# --------------------------------------------------
# Promotion metadata classification
# --------------------------------------------------

def has_promotion_metadata(df):
    """Identify records with any recorded promotion metadata."""

    return df[PROMOTION_COLUMNS].notna().any(axis=1)


def classify_promotion_type(df):
    """Classify records by display and mailer metadata."""

    has_display = df["display_values"].notna()
    has_mailer = df["mailer_values"].notna()

    conditions = [
        has_display & has_mailer,
        has_display & ~has_mailer,
        ~has_display & has_mailer
    ]

    choices = [
        "Display and mailer recorded",
        "Display only",
        "Mailer only"
    ]

    return np.select(
        conditions,
        choices,
        default="Neither recorded"
    )


# --------------------------------------------------
# Promotion exposure comparison
# --------------------------------------------------

def analyse_promotion_exposure(file_path, chunksize=CHUNK_SIZE):
    """Compare transaction measures with and without recorded metadata."""

    groups = {
        "With recorded promotion metadata": {
            "records": 0,
            "sales": 0.0,
            "quantity": 0.0,
            "retail_discount": 0.0,
            "coupon_discount": 0.0,
            "coupon_match_discount": 0.0,
            "total_discount": 0.0,
            "discount_pct_sum": 0.0,
            "discount_pct_count": 0
        },
        "Without recorded promotion metadata": {
            "records": 0,
            "sales": 0.0,
            "quantity": 0.0,
            "retail_discount": 0.0,
            "coupon_discount": 0.0,
            "coupon_match_discount": 0.0,
            "total_discount": 0.0,
            "discount_pct_sum": 0.0,
            "discount_pct_count": 0
        }
    }

    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False
    ):
        total_rows += len(chunk)

        metadata = has_promotion_metadata(chunk)

        chunk["promotion_group"] = metadata.map({
            True: "With recorded promotion metadata",
            False: "Without recorded promotion metadata"
        })

        columns = {
            "SALES_VALUE": "sales",
            "QUANTITY": "quantity",
            "retail_discount_amount": "retail_discount",
            "coupon_discount_amount": "coupon_discount",
            "coupon_match_discount_amount": "coupon_match_discount",
            "total_discount": "total_discount"
        }

        for group_name, group in chunk.groupby("promotion_group"):
            result = groups[group_name]

            result["records"] += len(group)

            for column, key in columns.items():
                result[key] += group[column].sum()

            valid_pct = group["discount_percentage"].dropna()
            result["discount_pct_sum"] += valid_pct.sum()
            result["discount_pct_count"] += valid_pct.count()

    results = []

    for group_name, values in groups.items():
        n = values["records"]

        results.append({
            "Promotion_Group": group_name,
            "Records": n,
            "Record_Share_Percent": n / total_rows * 100,
            "Mean_Sales_Per_Record": values["sales"] / n,
            "Mean_Quantity_Per_Record": values["quantity"] / n,
            "Mean_Retail_Discount_Per_Record": (
                values["retail_discount"] / n
            ),
            "Mean_Coupon_Discount_Per_Record": (
                values["coupon_discount"] / n
            ),
            "Mean_Coupon_Match_Discount_Per_Record": (
                values["coupon_match_discount"] / n
            ),
            "Mean_Total_Discount_Per_Record": (
                values["total_discount"] / n
            ),
            "Mean_Discount_Percentage": (
                values["discount_pct_sum"]
                / values["discount_pct_count"]
                if values["discount_pct_count"] else np.nan
            ),
            "Valid_Discount_Percentage_Records": (
                values["discount_pct_count"]
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Display and mailer comparison
# --------------------------------------------------

def analyse_promotion_types(file_path, chunksize=CHUNK_SIZE):
    """Compare display-only, mailer-only and combined metadata groups."""

    groups = {
        name: {
            "records": 0,
            "sales": 0.0,
            "discount": 0.0,
            "pct_sum": 0.0,
            "pct_count": 0
        }
        for name in [
            "Display and mailer recorded",
            "Display only",
            "Mailer only",
            "Neither recorded"
        ]
    }

    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False
    ):
        total_rows += len(chunk)
        chunk["promotion_group"] = classify_promotion_type(chunk)

        for group_name, group in chunk.groupby("promotion_group"):
            result = groups[group_name]

            result["records"] += len(group)
            result["sales"] += group["SALES_VALUE"].sum()
            result["discount"] += group["total_discount"].sum()

            valid_pct = group["discount_percentage"].dropna()
            result["pct_sum"] += valid_pct.sum()
            result["pct_count"] += valid_pct.count()

    results = []

    for group_name, values in groups.items():
        n = values["records"]

        results.append({
            "Promotion_Type": group_name,
            "Records": n,
            "Record_Share_Percent": n / total_rows * 100,
            "Mean_Sales_Per_Record": (
                values["sales"] / n if n else np.nan
            ),
            "Mean_Total_Discount_Per_Record": (
                values["discount"] / n if n else np.nan
            ),
            "Mean_Discount_Percentage": (
                values["pct_sum"] / values["pct_count"]
                if values["pct_count"] else np.nan
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Coupon activity
# --------------------------------------------------

def analyse_coupon_activity(file_path, chunksize=CHUNK_SIZE):
    """Count coupon and coupon-match discount records."""

    totals = {
        "records": 0,
        "coupon_discount_records": 0,
        "coupon_match_records": 0,
        "either_coupon_records": 0,
        "coupon_discount_sum": 0.0,
        "coupon_match_sum": 0.0,
        "total_discount_sum": 0.0
    }

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False
    ):
        coupon = chunk["COUPON_DISC"].fillna(0)
        match = chunk["COUPON_MATCH_DISC"].fillna(0)

        totals["records"] += len(chunk)
        totals["coupon_discount_records"] += (coupon != 0).sum()
        totals["coupon_match_records"] += (match != 0).sum()
        totals["either_coupon_records"] += (
            (coupon != 0) | (match != 0)
        ).sum()

        totals["coupon_discount_sum"] += coupon.sum()
        totals["coupon_match_sum"] += match.sum()
        totals["total_discount_sum"] += (
            chunk["total_discount"].fillna(0).sum()
        )

    results = []

    for label, key in [
        ("Records with coupon discount", "coupon_discount_records"),
        ("Records with coupon-match discount", "coupon_match_records"),
        ("Records with either coupon discount", "either_coupon_records")
    ]:
        count = totals[key]

        results.append({
            "Metric": label,
            "Count": count,
            "Record_Share_Percent": (
                count / totals["records"] * 100
            )
        })

    return (
        pd.DataFrame(results),
        totals
    )


# --------------------------------------------------
# Coupon comparison at household level
# --------------------------------------------------

def analyse_coupon_households(file_path, chunksize=CHUNK_SIZE):
    """Compare households with and without recorded coupon discounts."""

    household_stats = {}

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False
    ):
        chunk["has_coupon"] = (
            chunk["COUPON_DISC"].fillna(0) != 0
        )

        for household, group in chunk.groupby("household_key"):
            if household not in household_stats:
                household_stats[household] = {
                    "records": 0,
                    "sales": 0.0,
                    "discount_pct_sum": 0.0,
                    "discount_pct_count": 0,
                    "coupon_records": 0
                }

            result = household_stats[household]

            result["records"] += len(group)
            result["sales"] += group["SALES_VALUE"].sum()
            result["coupon_records"] += group["has_coupon"].sum()

            valid_pct = group["discount_percentage"].dropna()
            result["discount_pct_sum"] += valid_pct.sum()
            result["discount_pct_count"] += valid_pct.count()

    rows = []

    for household, values in household_stats.items():
        rows.append({
            "household_key": household,
            "Coupon_Group": (
                "Coupon user"
                if values["coupon_records"] > 0
                else "Non-coupon user"
            ),
            "Mean_Sales_Per_Record": (
                values["sales"] / values["records"]
            ),
            "Mean_Discount_Percentage": (
                values["discount_pct_sum"]
                / values["discount_pct_count"]
                if values["discount_pct_count"] else np.nan
            )
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Promotion comparison at household level
# --------------------------------------------------

def analyse_promotion_households(file_path, chunksize=CHUNK_SIZE):
    """Classify households by recorded promotion metadata exposure."""

    household_stats = {}

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False
    ):
        chunk["has_metadata"] = has_promotion_metadata(chunk)

        for household, group in chunk.groupby("household_key"):
            if household not in household_stats:
                household_stats[household] = {
                    "records": 0,
                    "sales": 0.0,
                    "discount_pct_sum": 0.0,
                    "discount_pct_count": 0,
                    "promotion_records": 0
                }

            result = household_stats[household]

            result["records"] += len(group)
            result["sales"] += group["SALES_VALUE"].sum()
            result["promotion_records"] += group["has_metadata"].sum()

            valid_pct = group["discount_percentage"].dropna()
            result["discount_pct_sum"] += valid_pct.sum()
            result["discount_pct_count"] += valid_pct.count()

    rows = []

    for household, values in household_stats.items():
        rows.append({
            "household_key": household,
            "Promotion_Group": (
                "Promotion metadata recorded"
                if values["promotion_records"] > 0
                else "No promotion metadata recorded"
            ),
            "Mean_Sales_Per_Record": (
                values["sales"] / values["records"]
            ),
            "Mean_Discount_Percentage": (
                values["discount_pct_sum"]
                / values["discount_pct_count"]
                if values["discount_pct_count"] else np.nan
            )
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Promotion comparison at basket level
# --------------------------------------------------

def analyse_promotion_baskets(file_path, chunksize=CHUNK_SIZE):
    """Compare baskets with and without recorded promotion metadata."""

    basket_stats = {}

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False
    ):
        chunk["has_metadata"] = has_promotion_metadata(chunk)

        for basket, group in chunk.groupby("BASKET_ID"):
            if basket not in basket_stats:
                basket_stats[basket] = {
                    "sales": 0.0,
                    "discount_pct_sum": 0.0,
                    "discount_pct_count": 0,
                    "has_promotion": False
                }

            result = basket_stats[basket]

            result["sales"] += group["SALES_VALUE"].sum()
            result["has_promotion"] = (
                result["has_promotion"]
                or group["has_metadata"].any()
            )

            valid_pct = group["discount_percentage"].dropna()
            result["discount_pct_sum"] += valid_pct.sum()
            result["discount_pct_count"] += valid_pct.count()

    rows = []

    for basket, values in basket_stats.items():
        rows.append({
            "BASKET_ID": basket,
            "Promotion_Group": (
                "Promotion metadata recorded"
                if values["has_promotion"]
                else "No promotion metadata recorded"
            ),
            "Basket_Sales": values["sales"],
            "Mean_Discount_Percentage": (
                values["discount_pct_sum"]
                / values["discount_pct_count"]
                if values["discount_pct_count"] else np.nan
            )
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Statistical testing and effect sizes
# --------------------------------------------------

def mann_whitney_comparison(
    values1,
    values2,
    analysis,
    metric,
    group1,
    group2
):
    """Run Mann-Whitney U test and calculate rank-biserial correlation."""

    values1 = pd.Series(values1).dropna()
    values2 = pd.Series(values2).dropna()

    if len(values1) == 0 or len(values2) == 0:
        raise ValueError("Both groups must contain valid observations.")

    u_stat, p_value = mannwhitneyu(
        values1,
        values2,
        alternative="two-sided"
    )

    rank_biserial = (
        2 * u_stat / (len(values1) * len(values2))
    ) - 1

    return {
        "Analysis": analysis,
        "Metric": metric,
        "Group_1": group1,
        "Group_2": group2,
        "N_Group_1": len(values1),
        "N_Group_2": len(values2),
        "U_Statistic": u_stat,
        "P_Value": p_value,
        "Rank_Biserial_Correlation": rank_biserial
    }


def run_discount_statistical_tests(
    household_coupon_df,
    household_promotion_df,
    basket_promotion_df
):
    """Run the six household- and basket-level comparisons."""

    results = []

    comparisons = [
        (
            household_coupon_df,
            "Coupon_Group",
            "Coupon user",
            "Non-coupon user",
            ["Mean_Sales_Per_Record", "Mean_Discount_Percentage"],
            "Household coupon comparison"
        ),
        (
            household_promotion_df,
            "Promotion_Group",
            "Promotion metadata recorded",
            "No promotion metadata recorded",
            ["Mean_Sales_Per_Record", "Mean_Discount_Percentage"],
            "Household promotion comparison"
        ),
        (
            basket_promotion_df,
            "Promotion_Group",
            "Promotion metadata recorded",
            "No promotion metadata recorded",
            ["Basket_Sales", "Mean_Discount_Percentage"],
            "Basket-level promotion comparison"
        )
    ]

    for (
        data, group_column, group1, group2,
        metrics, analysis
    ) in comparisons:

        for metric in metrics:
            result = mann_whitney_comparison(
                data.loc[data[group_column] == group1, metric],
                data.loc[data[group_column] == group2, metric],
                analysis,
                metric,
                group1,
                group2
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    results_df["Holm_Adjusted_P_Value"] = multipletests(
        results_df["P_Value"],
        method="holm"
    )[1]

    results_df["Significant_After_Correction"] = (
        results_df["Holm_Adjusted_P_Value"] < 0.05
    )

    return results_df


# --------------------------------------------------
# Basket-level summary
# --------------------------------------------------

def summarize_promotion_baskets(basket_promotion_df):
    """Summarize basket sales and discount percentage by promotion group."""

    return (
        basket_promotion_df.groupby("Promotion_Group")
        .agg(
            Baskets=("BASKET_ID", "count"),
            Mean_Basket_Sales=("Basket_Sales", "mean"),
            Median_Basket_Sales=("Basket_Sales", "median"),
            Mean_Discount_Percentage=("Mean_Discount_Percentage", "mean")
        )
        .reset_index()
    )
