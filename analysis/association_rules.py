
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import (
    apriori,
    fpgrowth,
    association_rules
)


# --------------------------------------------------
# Prepare basket transactions
# --------------------------------------------------

def prepare_baskets(transactions):
    """Remove coupon/miscellaneous records and construct baskets."""

    transactions_for_mining = transactions[
        transactions["COMMODITY_DESC"] != "COUPON/MISC ITEMS"
    ].copy()

    baskets = (
        transactions_for_mining.groupby("BASKET_ID")["PRODUCT_ID"]
        .apply(lambda x: sorted(set(x)))
    )

    product_frequency = (
        transactions_for_mining.groupby("PRODUCT_ID")["BASKET_ID"]
        .nunique()
        .sort_values(ascending=False)
    )

    return baskets, product_frequency, transactions_for_mining


# --------------------------------------------------
# Select candidate products
# --------------------------------------------------

def select_candidate_products(
    product_frequency,
    total_baskets,
    min_frequency=0.005
):
    """Select products appearing in the required proportion of baskets."""

    min_product_frequency = int(total_baskets * min_frequency)

    candidate_products = product_frequency[
        product_frequency >= min_product_frequency
    ]

    return candidate_products


# --------------------------------------------------
# Filter baskets for mining
# --------------------------------------------------

def filter_mining_baskets(baskets, candidate_products):
    """Keep only candidate products in each basket."""

    candidate_product_ids = set(candidate_products.index)

    mining_baskets = baskets.apply(
        lambda products: [
            product for product in products
            if product in candidate_product_ids
        ]
    )

    # Preserve the original notebook's basket population,
    # including baskets that become empty after filtering.
    return mining_baskets


# --------------------------------------------------
# Convert baskets into a transaction matrix
# --------------------------------------------------

def create_basket_matrix(mining_baskets):
    """Create a sparse transaction matrix for MLxtend."""

    encoder = TransactionEncoder()

    basket_matrix = encoder.fit(
        mining_baskets
    ).transform(
        mining_baskets,
        sparse=True
    )

    basket_df = pd.DataFrame.sparse.from_spmatrix(
        basket_matrix,
        columns=encoder.columns_
    )

    # Convert product IDs to strings for MLxtend compatibility
    basket_df.columns = basket_df.columns.astype(str)

    return basket_df


# --------------------------------------------------
# Apriori frequent itemsets
# --------------------------------------------------

def mine_apriori(
    basket_df,
    min_support=0.005,
    max_len=3
):
    """Generate frequent itemsets using Apriori."""

    return apriori(
        basket_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
        low_memory=True
    )


# --------------------------------------------------
# FP-Growth frequent itemsets
# --------------------------------------------------

def mine_fpgrowth(
    basket_df,
    min_support=0.005,
    max_len=3
):
    """Generate frequent itemsets using FP-Growth."""

    return fpgrowth(
        basket_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )


# --------------------------------------------------
# Generate association rules
# --------------------------------------------------

def generate_rules(
    frequent_itemsets,
    min_confidence=0.1
):
    """Generate association rules from frequent itemsets."""

    if frequent_itemsets.empty:
        return pd.DataFrame()

    return association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )


# --------------------------------------------------
# Interpret product IDs
# --------------------------------------------------

def create_product_lookup(transactions):
    """Create a product ID to product description mapping."""

    product_details = (
        transactions[
            [
                "PRODUCT_ID",
                "COMMODITY_DESC",
                "SUB_COMMODITY_DESC"
            ]
        ]
        .drop_duplicates("PRODUCT_ID")
        .copy()
    )

    product_details["PRODUCT_ID"] = (
        product_details["PRODUCT_ID"].astype(str)
    )

    return (
        product_details.set_index("PRODUCT_ID")
        .apply(
            lambda row: (
                f"{row['SUB_COMMODITY_DESC']} "
                f"({row['COMMODITY_DESC']})"
            ),
            axis=1
        )
        .to_dict()
    )


def interpret_rules(rules, product_lookup):
    """Add readable product descriptions to association rules."""

    if rules.empty:
        return rules.copy()

    interpreted = rules.copy()

    def describe_product_set(product_set):
        return [
            product_lookup.get(str(product_id), str(product_id))
            for product_id in product_set
        ]

    interpreted["antecedents"] = (
        interpreted["antecedents"].apply(describe_product_set)
    )

    interpreted["consequents"] = (
        interpreted["consequents"].apply(describe_product_set)
    )

    return interpreted


# --------------------------------------------------
# Compare Apriori and FP-Growth
# --------------------------------------------------

def compare_itemsets(apriori_itemsets, fpgrowth_itemsets):
    """Compare frequent itemsets produced by both algorithms."""

    apriori_comparison = apriori_itemsets.copy()
    fpgrowth_comparison = fpgrowth_itemsets.copy()

    apriori_comparison["itemsets"] = (
        apriori_comparison["itemsets"].apply(frozenset)
    )

    fpgrowth_comparison["itemsets"] = (
        fpgrowth_comparison["itemsets"].apply(frozenset)
    )

    comparison = apriori_comparison.merge(
        fpgrowth_comparison,
        on="itemsets",
        how="outer",
        suffixes=("_apriori", "_fpgrowth"),
        indicator=True
    )

    common = comparison[
        comparison["_merge"] == "both"
    ].copy()

    common["support_difference"] = (
        common["support_apriori"]
        - common["support_fpgrowth"]
    ).abs()

    return comparison, common


def compare_rules(apriori_rules, fpgrowth_rules):
    """Compare rules and their metrics across both algorithms."""

    apriori_comparison = apriori_rules.copy()
    fpgrowth_comparison = fpgrowth_rules.copy()

    for df in [apriori_comparison, fpgrowth_comparison]:
        df["antecedents"] = df["antecedents"].apply(frozenset)
        df["consequents"] = df["consequents"].apply(frozenset)

    comparison = apriori_comparison.merge(
        fpgrowth_comparison,
        on=["antecedents", "consequents"],
        how="outer",
        suffixes=("_apriori", "_fpgrowth"),
        indicator=True
    )

    common = comparison[
        comparison["_merge"] == "both"
    ].copy()

    for metric in ["support", "confidence", "lift"]:
        common[f"{metric}_difference"] = (
            common[f"{metric}_apriori"]
            - common[f"{metric}_fpgrowth"]
        ).abs()

    return comparison, common


# --------------------------------------------------
# Select high-confidence rules
# --------------------------------------------------

def select_high_confidence_rules(rules, threshold=0.30):
    """Select rules meeting the notebook's confidence threshold."""

    selected = rules[
        rules["confidence"] >= threshold
    ].copy()

    return selected.sort_values(
        "lift",
        ascending=False
    )


# --------------------------------------------------
# Complete association rule mining workflow
# --------------------------------------------------

def run_association_rule_mining(
    transactions,
    min_support=0.005,
    max_len=3,
    min_confidence=0.1,
    high_confidence_threshold=0.30
):
    """Run the complete association rule mining workflow."""

    baskets, product_frequency, transactions_for_mining = (
        prepare_baskets(transactions)
    )

    candidate_products = select_candidate_products(
        product_frequency,
        len(baskets),
        min_frequency=min_support
    )

    mining_baskets = filter_mining_baskets(
        baskets,
        candidate_products
    )

    basket_df = create_basket_matrix(mining_baskets)

    # Frequent itemsets
    apriori_itemsets = mine_apriori(
        basket_df,
        min_support=min_support,
        max_len=max_len
    )

    fpgrowth_itemsets = mine_fpgrowth(
        basket_df,
        min_support=min_support,
        max_len=max_len
    )

    # Association rules
    apriori_rules = generate_rules(
        apriori_itemsets,
        min_confidence
    )

    fpgrowth_rules = generate_rules(
        fpgrowth_itemsets,
        min_confidence
    )

    # Product descriptions
    product_lookup = create_product_lookup(transactions)

    interpreted_rules = interpret_rules(
        apriori_rules,
        product_lookup
    )

    high_confidence_rules = select_high_confidence_rules(
        apriori_rules,
        high_confidence_threshold
    )

    high_confidence_interpretable = interpret_rules(
        high_confidence_rules,
        product_lookup
    )

    # Algorithm comparisons
    itemset_comparison, common_itemsets = compare_itemsets(
        apriori_itemsets,
        fpgrowth_itemsets
    )

    rule_comparison, common_rules = compare_rules(
        apriori_rules,
        fpgrowth_rules
    )

    return {
        "baskets": baskets,
        "candidate_products": candidate_products,
        "mining_baskets": mining_baskets,
        "basket_matrix": basket_df,
        "apriori_itemsets": apriori_itemsets,
        "fpgrowth_itemsets": fpgrowth_itemsets,
        "apriori_rules": apriori_rules,
        "fpgrowth_rules": fpgrowth_rules,
        "interpreted_rules": interpreted_rules,
        "high_confidence_rules": high_confidence_interpretable,
        "itemset_comparison": itemset_comparison,
        "common_itemsets": common_itemsets,
        "rule_comparison": rule_comparison,
        "common_rules": common_rules,
        "transactions_for_mining": transactions_for_mining
    }
