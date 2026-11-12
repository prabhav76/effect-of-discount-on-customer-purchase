"""Build the small JSON data file consumed by the React dashboard.

The dashboard reads derived analysis outputs rather than loading the large
transaction table in a browser. Rerun the relevant notebooks after changing
source data, then run this exporter (npm's dev/build hooks do this for you).
This module uses only the Python standard library.
"""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "public" / "data" / "dashboard.json"

INPUTS = {
    "transactions": PROCESSED_DIR / "transactions_enriched.csv",
    "rq1": PROCESSED_DIR / "statistical_analysis" / "rq1_overall_discount_analysis.csv",
    "rq2": PROCESSED_DIR / "statistical_analysis" / "rq2_discount_band_analysis.csv",
    "rq3": PROCESSED_DIR / "statistical_analysis" / "rq3_brand_comparison.csv",
    "rq4": PROCESSED_DIR / "statistical_analysis" / "rq4_category_primary_results.csv",
    "statistical_summary": PROCESSED_DIR / "statistical_analysis" / "statistical_analysis_summary.csv",
    "promotion_findings": PROCESSED_DIR / "discount_effectiveness" / "discount_effectiveness_findings.csv",
    "promotion_tests": PROCESSED_DIR / "discount_effectiveness" / "statistical_test_results.csv",
    "promotion_summary": PROCESSED_DIR / "discount_effectiveness" / "promotion_analysis.csv",
    "coupon_summary": PROCESSED_DIR / "discount_effectiveness" / "coupon_analysis.csv",
    "basket_summary": PROCESSED_DIR / "discount_effectiveness" / "basket_discount_analysis.csv",
    "cluster_profiles": PROCESSED_DIR / "customer_clustering" / "cluster_profiles.csv",
    "cluster_assignments": PROCESSED_DIR / "customer_clustering" / "household_cluster_assignments.csv",
    "cluster_comparison": PROCESSED_DIR / "customer_clustering" / "clustering_comparison.csv",
    "rules": PROCESSED_DIR / "association_rule_mining" / "association_rules_apriori.csv",
    "fpgrowth_rules": PROCESSED_DIR / "association_rule_mining" / "association_rules_fpgrowth.csv",
    "interpreted_rules": PROCESSED_DIR / "association_rule_mining" / "high_confidence_interpretable_rules.csv",
    "itemset_comparison": PROCESSED_DIR / "association_rule_mining" / "itemset_comparison.csv",
    "rule_comparison": PROCESSED_DIR / "association_rule_mining" / "rule_comparison.csv",
    "itemset_findings": PROCESSED_DIR / "association_rule_mining" / "key_findings.csv",
}


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def int_value(value, default=0):
    parsed = number(value)
    return int(parsed) if parsed is not None else default


def median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def measure_map(path):
    return {
        row["Measure"]: number(row.get("Value"))
        for row in read_rows(path)
        if row.get("Measure")
    }


def calculate_discount_band_rates(path):
    """Count observed multi-unit rates from the enriched transaction file."""

    bins = [
        ("No Discount", -0.01, 0),
        ("Low", 0, 10),
        ("Moderate", 10, 25),
        ("High", 25, 50),
        ("Very High", 50, 100),
    ]
    totals = {label: [0, 0] for label, _, _ in bins}

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"QUANTITY", "discount_percentage"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            discount = number(row.get("discount_percentage"))
            quantity = number(row.get("QUANTITY"))
            if discount is None or quantity is None:
                continue
            for label, lower, upper in bins:
                # Match pandas.cut with the project's default right-closed bins.
                if lower < discount <= upper:
                    totals[label][1] += 1
                    if quantity > 1:
                        totals[label][0] += 1
                    break

    result = []
    for label, _, _ in bins:
        multi_unit, records = totals[label]
        result.append({
            "name": label,
            "rate": (multi_unit / records * 100) if records else None,
            "multi_unit_records": multi_unit,
            "records": records,
        })
    return result


def build_payload(previous_payload=None):
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in INPUTS.values() if not path.is_file()]
    if missing:
        return {
            "available": False,
            "message": "Analysis outputs are missing. Add the processed project data and analysis results, then reload the dashboard.",
            "missing_files": missing,
        }

    transaction_time = INPUTS["transactions"].stat().st_mtime_ns
    stale_outputs = [
        str(path.relative_to(PROJECT_ROOT))
        for name, path in INPUTS.items()
        if name != "transactions" and path.stat().st_mtime_ns < transaction_time
    ]
    if stale_outputs:
        return {
            "available": False,
            "message": (
                "The enriched transaction file is newer than some analysis outputs. "
                "Rerun the affected analysis notebooks, then refresh the dashboard."
            ),
            "stale_files": stale_outputs,
        }

    rq1 = measure_map(INPUTS["rq1"])
    rq2 = read_rows(INPUTS["rq2"])
    rq3 = read_rows(INPUTS["rq3"])
    rq4 = read_rows(INPUTS["rq4"])
    statistical_summary = read_rows(INPUTS["statistical_summary"])
    promotion_findings = read_rows(INPUTS["promotion_findings"])
    promotion_tests = read_rows(INPUTS["promotion_tests"])
    promotion_summary = read_rows(INPUTS["promotion_summary"])
    coupon_summary = read_rows(INPUTS["coupon_summary"])
    basket_summary = read_rows(INPUTS["basket_summary"])
    cluster_profiles = read_rows(INPUTS["cluster_profiles"])
    assignments = read_rows(INPUTS["cluster_assignments"])
    cluster_comparison = read_rows(INPUTS["cluster_comparison"])
    rules = read_rows(INPUTS["rules"])
    fpgrowth_rules = read_rows(INPUTS["fpgrowth_rules"])
    interpreted_rules = read_rows(INPUTS["interpreted_rules"])
    itemset_comparison = read_rows(INPUTS["itemset_comparison"])
    rule_comparison = read_rows(INPUTS["rule_comparison"])
    itemset_findings = {
        row["Metric"]: number(row.get("Value"))
        for row in read_rows(INPUTS["itemset_findings"])
        if row.get("Metric")
    }

    promotion_by_type = {row["Promotion_Type"]: row for row in promotion_summary}
    promotion_coverage = number(
        promotion_by_type.get("Display and mailer recorded", {}).get("Record_Share_Percent")
    )
    no_metadata = promotion_by_type.get("Neither recorded", {})
    with_metadata = promotion_by_type.get("Display and mailer recorded", {})

    coupon_by_group = {row["Coupon_Group"]: row for row in coupon_summary}
    coupon_users = coupon_by_group.get("Coupon discount recorded", {})
    non_coupon_users = coupon_by_group.get("No coupon discount recorded", {})
    basket_by_group = {row["Promotion_Group"]: row for row in basket_summary}
    basket_exposed = basket_by_group.get("Promotion metadata recorded", {})
    basket_unrecorded = basket_by_group.get("No promotion metadata recorded", {})

    tests_by_key = {
        (row.get("Analysis"), row.get("Metric")): row
        for row in promotion_tests
    }
    coupon_discount_test = tests_by_key.get((
        "Household coupon comparison", "Mean_Discount_Percentage"
    ), {})
    coupon_sales_test = tests_by_key.get((
        "Household coupon comparison", "Mean_Sales_Per_Record"
    ), {})
    promotion_household_tests = [
        row for (analysis, _), row in tests_by_key.items()
        if analysis == "Household promotion comparison"
    ]
    basket_sales_test = tests_by_key.get((
        "Basket-level promotion comparison", "Basket_Sales"
    ), {})
    basket_discount_test = tests_by_key.get((
        "Basket-level promotion comparison", "Mean_Discount_Percentage"
    ), {})

    rq2_terms = {row.get("Coefficient"): row for row in rq2}
    rq3_terms = {row.get("Coefficient"): row for row in rq3}
    low_term = rq2_terms.get("C(discount_band)[T.Low]", {})
    very_high_term = rq2_terms.get("C(discount_band)[T.Very High]", {})
    national_discount = rq3_terms.get("C(discount_status)[T.Discounted]", {})
    brand_interaction = rq3_terms.get(
        "C(discount_status)[T.Discounted]:C(BRAND)[T.Private]", {}
    )
    private_or = number(national_discount.get("Estimate"), 0) + number(
        brand_interaction.get("Estimate"), 0
    )
    private_or = math.exp(private_or)

    significant_categories = [
        row for row in rq4
        if str(row.get("significant_after_holm", "")).strip().lower() == "true"
    ]
    positive_categories = sum(
        1 for row in significant_categories if number(row.get("odds_ratio"), 0) > 1
    )
    negative_categories = sum(
        1 for row in significant_categories if number(row.get("odds_ratio"), 0) < 1
    )

    summary_by_question = {
        row.get("Research_Question"): row for row in statistical_summary
    }
    rq2_key_result = summary_by_question.get("RQ2", {}).get("Key_Result", "")
    try:
        rq2_overall_p_value = float(rq2_key_result.split(":", 1)[-1].strip())
    except ValueError:
        rq2_overall_p_value = None

    selected_gmm = next((
        row for row in cluster_comparison
        if row.get("Algorithm") == "GMM" and row.get("Representation") == "PCA"
    ), {})

    cluster_counts = {}
    for row in assignments:
        cluster = str(row.get("cluster", ""))
        if cluster:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    cluster_profiles.sort(key=lambda row: int_value(row.get("cluster")))
    profile_rows = []
    for row in cluster_profiles:
        cluster_id = str(row.get("cluster", ""))
        profile_rows.append({
            "id": cluster_id,
            "households": cluster_counts.get(cluster_id, 0),
            "median_baskets": number(row.get("unique_baskets_median")),
            "median_basket_value": number(row.get("average_basket_value_median")),
            "median_discount_percentage": number(row.get("discount_percentage_median")),
            "median_multi_unit_share": number(row.get("multi_unit_share_median")),
        })
    ranked_profiles = sorted(profile_rows, key=lambda row: row["median_baskets"] or 0)
    if ranked_profiles:
        ranked_profiles[0]["activity"] = "Lower shopping activity"
        ranked_profiles[-1]["activity"] = "Higher shopping activity"
    for profile in profile_rows:
        profile.setdefault("activity", "Shopping activity")
        total_households = len(assignments)
        profile["share"] = profile["households"] / total_households * 100 if total_households else 0

    interpreted_rules.sort(key=lambda row: number(row.get("lift"), 0), reverse=True)
    rule_rows = [{
        "antecedent": row.get("antecedent_description", ""),
        "consequent": row.get("consequent_description", ""),
        "support": number(row.get("support")),
        "confidence": number(row.get("confidence")),
        "lift": number(row.get("lift")),
    } for row in interpreted_rules[:4]]

    basket_rule_count = len(rules)
    high_lift_count = sum(1 for row in rules if number(row.get("lift"), 0) > 1)
    household_count = len(assignments)
    matched_rule_count = sum(1 for row in rule_comparison if row.get("_merge") == "both")
    matched_itemset_count = sum(1 for row in itemset_comparison if row.get("_merge") == "both")
    rule_metric_differences = []
    for row in rule_comparison:
        for column, value in row.items():
            if column.endswith("_apriori"):
                paired_column = column.removesuffix("_apriori") + "_fpgrowth"
                left, right = number(value), number(row.get(paired_column))
                if left is not None and right is not None:
                    rule_metric_differences.append(abs(left - right))

    findings = [
        {
            "area": "Discount status",
            "finding": (
                f"{rq1.get('Discounted multi-unit rate', 0) * 100:.2f}% of discounted "
                f"records were multi-unit purchases vs "
                f"{rq1.get('Non-discounted multi-unit rate', 0) * 100:.2f}% without a discount"
            ),
            "method": (
                f"Household bootstrap 95% CI: {rq1.get('Bootstrap CI lower', 0):.2f}–"
                f"{rq1.get('Bootstrap CI upper', 0):.2f} percentage points"
            ),
        },
        {
            "area": "Discount bands",
            "finding": "Multi-unit purchase rates and odds vary across discount bands",
            "method": "Household-clustered binomial GEE; pairwise Holm correction",
        },
        {
            "area": "Promotion effectiveness",
            "finding": f"{promotion_coverage or 0:.2f}% of records had recorded promotion metadata",
            "method": "Household- and basket-level comparisons with Holm correction",
        },
        {
            "area": "Basket associations",
            "finding": f"Apriori produced {basket_rule_count} association rules",
            "method": "Cross-checked with FP-Growth",
        },
    ]

    significant_promotion_tests = sum(
        1 for row in promotion_tests
        if str(row.get("Significant_After_Correction", "")).strip().lower() == "true"
    )

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": {
            "households": household_count,
            "discounted_multi_unit_rate": rq1.get("Discounted multi-unit rate"),
            "non_discounted_multi_unit_rate": rq1.get("Non-discounted multi-unit rate"),
            "difference_pp": rq1.get("Difference (percentage points)"),
            "bootstrap_ci_lower": rq1.get("Bootstrap CI lower"),
            "bootstrap_ci_upper": rq1.get("Bootstrap CI upper"),
            "cmh_odds_ratio": rq1.get("Common odds ratio"),
            "cmh_ci_lower": rq1.get("Common OR CI lower"),
            "cmh_ci_upper": rq1.get("Common OR CI upper"),
            "cmh_p_value": rq1.get("CMH p-value"),
        },
        "discount": {
            "bands": (
                previous_payload.get("discount", {}).get("bands")
                if previous_payload else None
            ) or calculate_discount_band_rates(INPUTS["transactions"]),
            "low_vs_no_discount_or": number(low_term.get("Odds_Ratio")),
            "very_high_vs_no_discount_or": number(very_high_term.get("Odds_Ratio")),
            "national_or": math.exp(number(national_discount.get("Estimate"), 0)),
            "private_or": private_or,
            "brand_interaction_p_value": number(brand_interaction.get("P_Value")),
            "overall_band_p_value": rq2_overall_p_value,
            "category_count": len(rq4),
            "significant_category_count": len(significant_categories),
            "positive_category_count": positive_categories,
            "negative_category_count": negative_categories,
            "category_median_or": median([
                number(row.get("odds_ratio")) for row in rq4
            ]),
        },
        "promotion": {
            "metadata_share_pct": promotion_coverage,
            "significant_test_count": significant_promotion_tests,
            "test_count": len(promotion_tests),
            "mean_sales_with_metadata": number(basket_exposed.get("Mean_Basket_Sales")),
            "median_sales_with_metadata": number(basket_exposed.get("Median_Basket_Sales")),
            "mean_sales_without_metadata": number(basket_unrecorded.get("Mean_Basket_Sales")),
            "median_sales_without_metadata": number(basket_unrecorded.get("Median_Basket_Sales")),
            "mean_discount_with_metadata": number(basket_exposed.get("Mean_Discount_Percentage")),
            "mean_discount_without_metadata": number(basket_unrecorded.get("Mean_Discount_Percentage")),
            "coupon_sales_significant": str(coupon_sales_test.get("Significant_After_Correction", "")).lower() == "true",
            "coupon_discount_significant": str(coupon_discount_test.get("Significant_After_Correction", "")).lower() == "true",
            "coupon_mean_discount_with_coupon": number(coupon_users.get("Mean_Discount_Percentage")),
            "coupon_mean_discount_without_coupon": number(non_coupon_users.get("Mean_Discount_Percentage")),
            "coupon_mean_sales_with_coupon": number(coupon_users.get("Mean_Sales_Per_Record")),
            "coupon_mean_sales_without_coupon": number(non_coupon_users.get("Mean_Sales_Per_Record")),
            "basket_sales_significant": str(basket_sales_test.get("Significant_After_Correction", "")).lower() == "true",
            "basket_discount_significant": str(basket_discount_test.get("Significant_After_Correction", "")).lower() == "true",
            "promotion_household_significant": any(
                str(row.get("Significant_After_Correction", "")).lower() == "true"
                for row in promotion_household_tests
            ),
            "findings": promotion_findings,
            "coupon_groups": coupon_summary,
            "promotion_groups": promotion_summary,
        },
        "clusters": {
            "households": household_count,
            "model": f"{selected_gmm.get('Algorithm', 'GMM')} · {selected_gmm.get('Representation', 'PCA')} · {selected_gmm.get('Configuration', '2 components')}",
            "profiles": profile_rows,
        },
        "basket": {
            "rule_count": basket_rule_count,
            "fpgrowth_rule_count": len(fpgrowth_rules),
            "matched_rule_count": matched_rule_count,
            "itemset_count": int_value(itemset_findings.get("Apriori frequent itemsets")),
            "fpgrowth_itemset_count": int_value(itemset_findings.get("FP-Growth frequent itemsets")),
            "matched_itemset_count": matched_itemset_count,
            "algorithms_agree": (
                basket_rule_count == len(fpgrowth_rules)
                and matched_rule_count == basket_rule_count
                and matched_itemset_count == int_value(itemset_findings.get("Apriori frequent itemsets"))
                and (max(rule_metric_differences) if rule_metric_differences else 0) < 1e-12
            ),
            "high_lift_count": high_lift_count,
            "max_lift": max((number(row.get("lift"), 0) for row in rules), default=0),
            "rules": rule_rows,
        },
        "findings": findings,
        "sources": {
            name: str(path.relative_to(PROJECT_ROOT))
            for name, path in INPUTS.items()
        },
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous_payload = None
    if OUTPUT_PATH.is_file():
        try:
            previous_payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_payload = None

    signature = {}
    for name, path in INPUTS.items():
        if path.is_file():
            stat = path.stat()
            signature[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        else:
            signature[name] = None

    if previous_payload and previous_payload.get("_input_signature") == signature:
        print(f"Dashboard data is current: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
        return

    previous_signature = (previous_payload or {}).get("_input_signature", {})
    unchanged_transactions = previous_signature.get("transactions") == signature.get("transactions")
    payload = build_payload(previous_payload if unchanged_transactions else None)
    payload["_input_signature"] = signature
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["available"]:
        print(f"Exported dashboard data to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    else:
        print(payload["message"])
        for filename in payload["missing_files"]:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
