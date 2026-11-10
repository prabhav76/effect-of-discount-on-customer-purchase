
import matplotlib.pyplot as plt
import numpy as np


def plot_discount_band_rates(household_counts):
    summary = (
        household_counts.groupby(
            "discount_band",
            observed=True
        )
        .agg(
            multi_unit=("Multi-Unit", "sum"),
            total=("total_purchases", "sum")
        )
    )

    summary["rate"] = (
        summary["multi_unit"] / summary["total"] * 100
    )

    plt.figure(figsize=(9, 5))
    plt.bar(summary.index.astype(str), summary["rate"])

    plt.xlabel("Discount Band")
    plt.ylabel("Multi-Unit Purchases (%)")
    plt.title("Multi-Unit Purchase Rate Across Discount Bands")

    plt.tight_layout()
    plt.show()


def plot_brand_odds_ratios(brand_results):
    data = brand_results.copy()
    positions = np.arange(len(data))

    errors = np.vstack([
        data["Odds_Ratio"] - data["CI_Lower"],
        data["CI_Upper"] - data["Odds_Ratio"]
    ])

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        data["Odds_Ratio"],
        positions,
        xerr=errors,
        fmt="o",
        capsize=5
    )

    plt.axvline(1, linestyle="--")
    plt.yticks(positions, data["Brand"])
    plt.xlabel("Odds Ratio (95% Confidence Interval)")
    plt.title("Discount Association by Brand Type")

    plt.tight_layout()
    plt.show()


def plot_category_odds_ratios(category_results, top_n=15):
    data = (
        category_results
        .sort_values("p_value")
        .head(top_n)
        .sort_values("odds_ratio")
        .copy()
    )

    positions = np.arange(len(data))

    errors = np.vstack([
        data["odds_ratio"] - data["ci_lower"],
        data["ci_upper"] - data["odds_ratio"]
    ])

    plt.figure(figsize=(11, 8))

    plt.errorbar(
        data["odds_ratio"],
        positions,
        xerr=errors,
        fmt="o",
        capsize=4
    )

    plt.axvline(1, linestyle="--")
    plt.yticks(positions, data["COMMODITY_DESC"])
    plt.xlabel("Odds Ratio (95% Confidence Interval)")
    plt.ylabel("Commodity Category")
    plt.title("Discount Association Across Commodity Categories")

    plt.tight_layout()
    plt.show()


def plot_category_significance(category_results):
    if "holm_p_value" not in category_results.columns:
        raise ValueError("Holm-adjusted p-values are required.")

    data = category_results.sort_values("holm_p_value")

    plt.figure(figsize=(11, 7))

    plt.barh(
        data["COMMODITY_DESC"],
        -np.log10(data["holm_p_value"].clip(lower=1e-300))
    )

    plt.axvline(
        -np.log10(0.05),
        linestyle="--",
        label="Adjusted p = 0.05"
    )

    plt.xlabel("-log10(Holm-adjusted p-value)")
    plt.ylabel("Commodity Category")
    plt.title("Statistical Significance Across Categories")
    plt.legend()

    plt.tight_layout()
    plt.show()
