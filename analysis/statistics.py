
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import norm
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------
# Prepare statistical analysis variables
# --------------------------------------------------

def prepare_statistical_data(transactions):
    """Create discount-status and purchase-type variables."""

    data = transactions.copy()

    data["discount_status"] = (
        data["total_discount"]
        .gt(0)
        .map({
            True: "Discounted",
            False: "Non-Discounted"
        })
    )

    data["purchase_type"] = np.where(
        data["QUANTITY"] > 1,
        "Multi-Unit",
        "Single-Unit"
    )

    return data


# --------------------------------------------------
# RQ1: Overall discount association
# --------------------------------------------------

def calculate_overall_purchase_rates(data):
    """Calculate multi-unit purchase rates by discount status."""

    summary = pd.crosstab(
        data["discount_status"],
        data["purchase_type"]
    )

    discounted_rate = (
        summary.loc["Discounted", "Multi-Unit"]
        / summary.loc["Discounted"].sum()
    )

    non_discounted_rate = (
        summary.loc["Non-Discounted", "Multi-Unit"]
        / summary.loc["Non-Discounted"].sum()
    )

    difference = (
        discounted_rate - non_discounted_rate
    ) * 100

    return summary, {
        "discounted_rate": discounted_rate,
        "non_discounted_rate": non_discounted_rate,
        "difference_percentage_points": difference
    }


def bootstrap_household_difference(
    data,
    n_bootstrap=2000,
    random_state=42
):
    """Estimate the multi-unit rate difference using household bootstrap."""

    household_counts = (
        data.groupby(
            ["household_key", "discount_status", "purchase_type"]
        )
        .size()
        .unstack(
            ["discount_status", "purchase_type"],
            fill_value=0
        )
    )

    required_columns = [
        ("Discounted", "Multi-Unit"),
        ("Discounted", "Single-Unit"),
        ("Non-Discounted", "Multi-Unit"),
        ("Non-Discounted", "Single-Unit")
    ]

    counts = household_counts.reindex(
        columns=pd.MultiIndex.from_tuples(required_columns),
        fill_value=0
    ).to_numpy(dtype=float)

    rng = np.random.default_rng(random_state)
    n_households = len(counts)

    differences = []

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(
            0, n_households, n_households
        )

        sampled_counts = counts[sampled_indices].sum(axis=0)

        discounted_total = sampled_counts[0] + sampled_counts[1]
        non_discounted_total = sampled_counts[2] + sampled_counts[3]

        if discounted_total == 0 or non_discounted_total == 0:
            continue

        discounted_rate = sampled_counts[0] / discounted_total
        non_discounted_rate = sampled_counts[2] / non_discounted_total

        differences.append(
            (discounted_rate - non_discounted_rate) * 100
        )

    confidence_interval = np.percentile(
        differences, [2.5, 97.5]
    )

    return {
        "bootstrap_differences": differences,
        "confidence_interval": confidence_interval
    }


def run_cmh_test(data):
    """Apply the Cochran-Mantel-Haenszel test across households."""

    household_counts = (
        data.groupby(
            ["household_key", "discount_status", "purchase_type"]
        )
        .size()
        .unstack(
            ["discount_status", "purchase_type"],
            fill_value=0
        )
    )

    required_columns = [
        ("Discounted", "Multi-Unit"),
        ("Discounted", "Single-Unit"),
        ("Non-Discounted", "Multi-Unit"),
        ("Non-Discounted", "Single-Unit")
    ]

    household_counts = household_counts.reindex(
        columns=pd.MultiIndex.from_tuples(required_columns),
        fill_value=0
    )

    household_tables = []

    for _, counts in household_counts.iterrows():
        table = [
            [
                counts[("Discounted", "Multi-Unit")],
                counts[("Discounted", "Single-Unit")]
            ],
            [
                counts[("Non-Discounted", "Multi-Unit")],
                counts[("Non-Discounted", "Single-Unit")]
            ]
        ]

        household_tables.append(table)

    cmh = StratifiedTable(household_tables)
    test = cmh.test_null_odds()

    return {
        "statistic": test.statistic,
        "p_value": test.pvalue,
        "common_odds_ratio": cmh.oddsratio_pooled,
        "confidence_interval": cmh.oddsratio_pooled_confint()
    }


# --------------------------------------------------
# RQ2: Discount bands
# --------------------------------------------------

DISCOUNT_BINS = [-0.01, 0, 10, 25, 50, 100]

DISCOUNT_LABELS = [
    "No Discount",
    "Low",
    "Moderate",
    "High",
    "Very High"
]


def create_discount_bands(data):
    """Assign transactions to the notebook's discount bands."""

    result = data.copy()

    result["discount_band"] = pd.cut(
        result["discount_percentage"],
        bins=DISCOUNT_BINS,
        labels=DISCOUNT_LABELS
    )

    return result


def aggregate_discount_band_purchases(data):
    """Aggregate single- and multi-unit counts by household and band."""

    result = (
        data.groupby(
            ["household_key", "discount_band", "purchase_type"],
            observed=True
        )
        .size()
        .unstack("purchase_type", fill_value=0)
        .reset_index()
    )

    result["total_purchases"] = (
        result["Multi-Unit"] + result["Single-Unit"]
    )

    result["multi_unit_rate"] = (
        result["Multi-Unit"] / result["total_purchases"]
    )

    return result


def fit_discount_band_gee(household_counts):
    """Fit the independence-structure household-clustered binomial GEE."""

    model = smf.gee(
        "multi_unit_rate ~ C(discount_band)",
        groups="household_key",
        data=household_counts,
        weights=household_counts["total_purchases"],
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence()
    )

    return model.fit()


def test_discount_band_effect(result):
    """Test whether all discount-band coefficients are jointly zero."""

    terms = [
        f"C(discount_band)[T.{band}] = 0"
        for band in DISCOUNT_LABELS[1:]
    ]

    wald_test = result.wald_test(", ".join(terms))

    return {
        "statistic": wald_test.statistic,
        "p_value": wald_test.pvalue
    }


def compare_discount_bands(result):
    """Calculate pairwise odds ratios and Holm-adjusted p-values."""

    parameters = result.params
    covariance = result.cov_params()

    bands = DISCOUNT_LABELS
    comparisons = []

    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            band1 = bands[i]
            band2 = bands[j]

            term1 = (
                f"C(discount_band)[T.{band1}]"
                if i > 0 else None
            )
            term2 = (
                f"C(discount_band)[T.{band2}]"
                if j > 0 else None
            )

            coefficient = (
                parameters[term2] if term2 else 0
            ) - (
                parameters[term1] if term1 else 0
            )

            vector = pd.Series(
                0.0, index=parameters.index
            )

            if term1:
                vector[term1] = -1

            if term2:
                vector[term2] = 1

            standard_error = np.sqrt(
                vector @ covariance @ vector
            )

            z_score = coefficient / standard_error
            p_value = 2 * norm.sf(abs(z_score))

            comparisons.append({
                "Band_1": band1,
                "Band_2": band2,
                "Odds_Ratio": np.exp(coefficient),
                "CI_Lower": np.exp(
                    coefficient - 1.96 * standard_error
                ),
                "CI_Upper": np.exp(
                    coefficient + 1.96 * standard_error
                ),
                "P_Value": p_value
            })

    results = pd.DataFrame(comparisons)

    results["Holm_Adjusted_P_Value"] = multipletests(
        results["P_Value"],
        method="holm"
    )[1]

    return results


# --------------------------------------------------
# RQ3: National and Private brand comparison
# --------------------------------------------------

def prepare_brand_analysis(data):
    """Prepare the notebook's brand-comparison dataset."""

    result = data[
        ~data["COMMODITY_DESC"].isin(
            ["FUEL", "COUPON/MISC ITEMS"]
        )
    ].copy()

    return result


def aggregate_brand_purchases(data):
    """Aggregate purchases by household, brand and discount status."""

    result = (
        data.groupby(
            [
                "household_key",
                "BRAND",
                "discount_status",
                "purchase_type"
            ]
        )
        .size()
        .unstack("purchase_type", fill_value=0)
        .reset_index()
    )

    result["total_purchases"] = (
        result["Multi-Unit"] + result["Single-Unit"]
    )

    result["multi_unit_rate"] = (
        result["Multi-Unit"] / result["total_purchases"]
    )

    result["discount_status"] = pd.Categorical(
        result["discount_status"],
        categories=["Non-Discounted", "Discounted"]
    )

    result["BRAND"] = pd.Categorical(
        result["BRAND"],
        categories=["National", "Private"]
    )

    return result


def fit_brand_gee(household_counts):
    """Fit household-clustered GEE with discount-brand interaction."""

    model = smf.gee(
        "multi_unit_rate ~ C(discount_status) * C(BRAND)",
        groups="household_key",
        data=household_counts,
        weights=household_counts["total_purchases"],
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence()
    )

    return model.fit()


def calculate_brand_odds_ratios(result):
    """Calculate discount odds ratios for National and Private brands."""

    params = result.params
    covariance = result.cov_params()

    discount_term = "C(discount_status)[T.Discounted]"
    interaction_term = (
        "C(discount_status)[T.Discounted]:C(BRAND)[T.Private]"
    )

    national_coef = params[discount_term]
    private_coef = national_coef + params[interaction_term]

    national_se = np.sqrt(
        covariance.loc[discount_term, discount_term]
    )

    private_vector = np.zeros(len(params))
    private_vector[params.index.get_loc(discount_term)] = 1
    private_vector[params.index.get_loc(interaction_term)] = 1

    private_se = np.sqrt(
        private_vector @ covariance.to_numpy() @ private_vector
    )

    interaction_coef = params[interaction_term]
    interaction_se = np.sqrt(
        covariance.loc[interaction_term, interaction_term]
    )

    estimates = [
        ("National", national_coef, national_se),
        ("Private", private_coef, private_se),
        ("Private vs National interaction",
         interaction_coef, interaction_se)
    ]

    return pd.DataFrame([
        {
            "Brand": name,
            "Odds_Ratio": np.exp(coef),
            "CI_Lower": np.exp(coef - 1.96 * se),
            "CI_Upper": np.exp(coef + 1.96 * se)
        }
        for name, coef, se in estimates
    ])


def test_brand_interaction(result):
    """Test whether the discount association differs by brand type."""

    interaction_term = (
        "C(discount_status)[T.Discounted]:C(BRAND)[T.Private]"
    )

    test = result.wald_test(
        f"{interaction_term} = 0"
    )

    return {
        "statistic": test.statistic,
        "p_value": test.pvalue
    }


# --------------------------------------------------
# RQ4: Commodity category comparison
# --------------------------------------------------

def prepare_category_analysis(data):
    """Exclude the categories excluded in the notebook."""

    return data[
        ~data["COMMODITY_DESC"].isin(
            ["FUEL", "COUPON/MISC ITEMS"]
        )
    ].copy()


def aggregate_category_purchases(data):
    """Aggregate multi-unit and single-unit purchases by household."""

    result = (
        data.groupby(
            [
                "household_key",
                "COMMODITY_DESC",
                "discount_status",
                "purchase_type"
            ]
        )
        .size()
        .unstack("purchase_type", fill_value=0)
        .reset_index()
    )

    result["total_purchases"] = (
        result["Multi-Unit"] + result["Single-Unit"]
    )

    result["multi_unit_rate"] = (
        result["Multi-Unit"] / result["total_purchases"]
    )

    return result


def fit_category_gee(category_data):
    """Fit the household-clustered GEE for one commodity category."""

    y = category_data["multi_unit_rate"].to_numpy()

    discount = (
        category_data["discount_status"] == "Discounted"
    ).astype(int).to_numpy()

    design = np.column_stack([
        np.ones(len(category_data)),
        discount
    ])

    model = sm.GEE(
        endog=y,
        exog=design,
        groups=category_data["household_key"].to_numpy(),
        weights=category_data["total_purchases"].to_numpy(),
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence()
    )

    return model.fit(maxiter=100)


def analyse_categories(category_data, minimum_households=30):
    """Fit category-level GEE models and apply primary-analysis criteria."""

    results = []

    for category, group in category_data.groupby("COMMODITY_DESC"):

        try:
            model_result = fit_category_gee(group)

            params = np.asarray(model_result.params)
            standard_errors = np.asarray(model_result.bse)
            p_values = np.asarray(model_result.pvalues)

            coefficient = params[1]
            standard_error = standard_errors[1]

            results.append({
                "COMMODITY_DESC": category,
                "households": group["household_key"].nunique(),
                "coefficient": coefficient,
                "standard_error": standard_error,
                "odds_ratio": np.exp(coefficient),
                "ci_lower": np.exp(
                    coefficient - 1.96 * standard_error
                ),
                "ci_upper": np.exp(
                    coefficient + 1.96 * standard_error
                ),
                "p_value": p_values[1],
                "converged": model_result.converged
            })

        except Exception as error:
            results.append({
                "COMMODITY_DESC": category,
                "households": group["household_key"].nunique(),
                "coefficient": np.nan,
                "standard_error": np.nan,
                "odds_ratio": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "converged": False,
                "error": str(error)
            })

    results = pd.DataFrame(results)

    numeric_columns = [
        "coefficient",
        "standard_error",
        "odds_ratio",
        "ci_lower",
        "ci_upper",
        "p_value"
    ]

    results["valid_estimate"] = (
        np.isfinite(results[numeric_columns]).all(axis=1)
        & (results["standard_error"] > 0)
        & results["p_value"].between(0, 1)
        & (results["ci_lower"] > 0)
        & (results["ci_upper"] > 0)
    )

    # Require non-zero counts in all four outcome groups
    outcome_counts = (
        category_data.groupby(
            ["COMMODITY_DESC", "discount_status", "purchase_type"]
        )
        .size()
        .unstack(
            ["discount_status", "purchase_type"],
            fill_value=0
        )
    )

    required_groups = [
        ("Discounted", "Multi-Unit"),
        ("Discounted", "Single-Unit"),
        ("Non-Discounted", "Multi-Unit"),
        ("Non-Discounted", "Single-Unit")
    ]

    outcome_counts = outcome_counts.reindex(
        columns=pd.MultiIndex.from_tuples(required_groups),
        fill_value=0
    )

    outcome_supported = (
        outcome_counts > 0
    ).all(axis=1)

    results["outcome_supported"] = (
        results["COMMODITY_DESC"].map(outcome_supported)
    )

    # Primary analysis: minimum household clusters and valid outcomes
    primary = results[
        (results["households"] >= minimum_households)
        & results["valid_estimate"]
        & results["outcome_supported"]
    ].copy()

    # Sensitivity analysis: no minimum household threshold
    sensitivity = results[
        results["valid_estimate"]
        & results["outcome_supported"]
    ].copy()

    for result_df in [primary, sensitivity]:
        if not result_df.empty:
            result_df["holm_p_value"] = multipletests(
                result_df["p_value"],
                method="holm"
            )[1]

            result_df["significant_after_holm"] = (
                result_df["holm_p_value"] < 0.05
            )

    return results, primary, sensitivity


# --------------------------------------------------
# Overall statistical workflow
# --------------------------------------------------

def run_statistical_analysis(transactions):
    """Run the reusable statistical analysis for RQ1 to RQ4."""

    data = prepare_statistical_data(transactions)

    # RQ1
    rq1_table, rq1_rates = calculate_overall_purchase_rates(data)
    rq1_bootstrap = bootstrap_household_difference(data)
    rq1_cmh = run_cmh_test(data)

    # RQ2
    band_data = create_discount_bands(data)
    band_counts = aggregate_discount_band_purchases(band_data)
    rq2_model = fit_discount_band_gee(band_counts)
    rq2_overall_test = test_discount_band_effect(rq2_model)
    rq2_pairwise = compare_discount_bands(rq2_model)

    # RQ3
    brand_data = prepare_brand_analysis(data)
    brand_counts = aggregate_brand_purchases(brand_data)
    rq3_model = fit_brand_gee(brand_counts)
    rq3_odds_ratios = calculate_brand_odds_ratios(rq3_model)
    rq3_interaction = test_brand_interaction(rq3_model)

    # RQ4
    category_data = prepare_category_analysis(data)
    category_counts = aggregate_category_purchases(category_data)

    (
        rq4_all,
        rq4_primary,
        rq4_sensitivity
    ) = analyse_categories(category_counts)

    return {
        "prepared_data": data,
        "rq1_contingency_table": rq1_table,
        "rq1_rates": rq1_rates,
        "rq1_bootstrap": rq1_bootstrap,
        "rq1_cmh": rq1_cmh,
        "rq2_household_counts": band_counts,
        "rq2_model": rq2_model,
        "rq2_overall_test": rq2_overall_test,
        "rq2_pairwise_comparisons": rq2_pairwise,
        "rq3_household_counts": brand_counts,
        "rq3_model": rq3_model,
        "rq3_odds_ratios": rq3_odds_ratios,
        "rq3_interaction_test": rq3_interaction,
        "rq4_category_counts": category_counts,
        "rq4_all_results": rq4_all,
        "rq4_primary_results": rq4_primary,
        "rq4_sensitivity_results": rq4_sensitivity
    }
