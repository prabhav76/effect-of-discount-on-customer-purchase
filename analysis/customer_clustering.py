
import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    MeanShift,
    estimate_bandwidth
)
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

from sklearn_extra.cluster import KMedoids


# --------------------------------------------------
# Load household feature representations
# --------------------------------------------------

def load_household_features(feature_dir):
    """Load standardized and PCA-transformed household features."""

    feature_dir = str(feature_dir)

    original = pd.read_csv(
        f"{feature_dir}/household_features_scaled.csv",
        index_col=0
    )

    pca = pd.read_csv(
        f"{feature_dir}/household_pca_features.csv"
    ).set_index("household_key")

    assert original.index.equals(pca.index), (
        "Household IDs or their ordering differ."
    )

    return original, pca


def get_feature_representations(original, pca):
    return {
        "Original": original,
        "PCA": pca
    }


# --------------------------------------------------
# Common clustering evaluation
# --------------------------------------------------

def evaluate_clustering(features, labels):
    """Calculate clustering quality and noise statistics."""

    labels = np.asarray(labels)

    noise_mask = labels == -1
    valid_mask = ~noise_mask

    valid_labels = labels[valid_mask]
    n_clusters = len(np.unique(valid_labels))

    result = {
        "Clusters": n_clusters,
        "Noise Points": int(noise_mask.sum()),
        "Noise Fraction": float(noise_mask.mean()),
        "Silhouette": np.nan,
        "Davies-Bouldin": np.nan,
        "Calinski-Harabasz": np.nan
    }

    if n_clusters >= 2 and valid_mask.sum() > n_clusters:
        valid_features = features.iloc[
            np.flatnonzero(valid_mask)
        ]

        result["Silhouette"] = silhouette_score(
            valid_features, valid_labels
        )
        result["Davies-Bouldin"] = davies_bouldin_score(
            valid_features, valid_labels
        )
        result["Calinski-Harabasz"] = calinski_harabasz_score(
            valid_features, valid_labels
        )

    return result


# --------------------------------------------------
# K-Means tuning
# --------------------------------------------------

def tune_kmeans(representations):
    results = []

    for name, features in representations.items():
        for k in range(2, 11):
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=42
            )

            labels = model.fit_predict(features)

            results.append({
                "Representation": name,
                "K": k,
                "Inertia": model.inertia_,
                "Silhouette": silhouette_score(features, labels)
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# K-Medoids and Agglomerative clustering
# --------------------------------------------------

def evaluate_kmedoids(representations):
    results = []

    for name, features in representations.items():
        for k in [2, 4, 5, 6]:
            model = KMedoids(
                n_clusters=k,
                metric="euclidean",
                method="alternate",
                random_state=42
            )

            labels = model.fit_predict(features)
            result = evaluate_clustering(features, labels)

            results.append({
                "Representation": name,
                "K": k,
                "Inertia": model.inertia_,
                **result
            })

    return pd.DataFrame(results)


def evaluate_agglomerative(representations):
    results = []

    for name, features in representations.items():
        for k in [2, 4, 5, 6]:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward"
            )

            labels = model.fit_predict(features)

            results.append({
                "Representation": name,
                "K": k,
                **evaluate_clustering(features, labels)
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# GMM evaluation and tuning
# --------------------------------------------------

def evaluate_gmm(representations):
    results = []

    for name, features in representations.items():
        for k in [2, 4, 5, 6]:
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=5,
                random_state=42
            )

            model.fit(features)
            labels = model.predict(features)

            results.append({
                "Representation": name,
                "Components": k,
                **evaluate_clustering(features, labels),
                "AIC": model.aic(features),
                "BIC": model.bic(features)
            })

    return pd.DataFrame(results)


def tune_gmm(representations):
    results = []

    for name, features in representations.items():
        for k in [2, 3, 4, 5, 6]:
            for covariance in ["full", "tied", "diag", "spherical"]:

                model = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance,
                    n_init=5,
                    random_state=42
                )

                model.fit(features)
                labels = model.predict(features)

                results.append({
                    "Representation": name,
                    "Components": k,
                    "Covariance": covariance,
                    **evaluate_clustering(features, labels),
                    "AIC": model.aic(features),
                    "BIC": model.bic(features)
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# DBSCAN tuning
# --------------------------------------------------

def tune_dbscan(representations):
    results = []

    parameter_ranges = {
        "Original": {
            "eps": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "min_samples": [5, 10, 15]
        },
        "PCA": {
            "eps": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "min_samples": [5, 10, 15]
        }
    }

    for name, features in representations.items():
        for eps in parameter_ranges[name]["eps"]:
            for min_samples in parameter_ranges[name]["min_samples"]:

                model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric="euclidean"
                )

                labels = model.fit_predict(features)

                results.append({
                    "Representation": name,
                    "eps": eps,
                    "min_samples": min_samples,
                    **evaluate_clustering(features, labels)
                })

    return pd.DataFrame(results)


def refine_dbscan(representations):
    results = []

    parameters = {
        "Original": {
            "eps": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "min_samples": [5, 10]
        },
        "PCA": {
            "eps": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "min_samples": [5, 10]
        }
    }

    for name, features in representations.items():
        for eps in parameters[name]["eps"]:
            for min_samples in parameters[name]["min_samples"]:

                labels = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric="euclidean"
                ).fit_predict(features)

                results.append({
                    "Representation": name,
                    "eps": eps,
                    "min_samples": min_samples,
                    **evaluate_clustering(features, labels)
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# OPTICS evaluation and tuning
# --------------------------------------------------

def evaluate_optics(representations):
    results = []

    for name, features in representations.items():
        model = OPTICS(
            min_samples=5,
            metric="euclidean",
            cluster_method="xi"
        )

        labels = model.fit_predict(features)

        results.append({
            "Representation": name,
            **evaluate_clustering(features, labels)
        })

    return pd.DataFrame(results)


def tune_optics(representations):
    results = []

    for name, features in representations.items():
        for min_samples in [5, 10, 15]:
            for xi in [0.03, 0.05, 0.07, 0.10]:

                model = OPTICS(
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_method="xi",
                    xi=xi
                )

                labels = model.fit_predict(features)

                results.append({
                    "Representation": name,
                    "min_samples": min_samples,
                    "xi": xi,
                    **evaluate_clustering(features, labels)
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# Mean Shift evaluation and tuning
# --------------------------------------------------

def evaluate_meanshift(representations):
    results = []

    for name, features in representations.items():
        bandwidth = estimate_bandwidth(
            features,
            quantile=0.2,
            n_samples=1000,
            random_state=42
        )

        model = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True
        )

        labels = model.fit_predict(features)

        results.append({
            "Representation": name,
            "Bandwidth": bandwidth,
            **evaluate_clustering(features, labels)
        })

    return pd.DataFrame(results)


def tune_meanshift(representations):
    results = []

    for name, features in representations.items():
        for quantile in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:

            bandwidth = estimate_bandwidth(
                features,
                quantile=quantile,
                n_samples=1000,
                random_state=42
            )

            model = MeanShift(
                bandwidth=bandwidth,
                bin_seeding=True
            )

            labels = model.fit_predict(features)

            results.append({
                "Representation": name,
                "Quantile": quantile,
                "Bandwidth": bandwidth,
                **evaluate_clustering(features, labels)
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# Clustering stability
# --------------------------------------------------

def evaluate_initialization_stability(features, n_runs=10):
    results = []

    for algorithm in ["K-Means", "GMM"]:
        repeated_labels = []

        for seed in range(n_runs):
            if algorithm == "K-Means":
                model = KMeans(
                    n_clusters=2,
                    init="k-means++",
                    n_init=1,
                    random_state=seed
                )
            else:
                model = GaussianMixture(
                    n_components=2,
                    covariance_type="spherical",
                    n_init=1,
                    random_state=seed
                )

            repeated_labels.append(model.fit_predict(features))

        ari_scores = [
            adjusted_rand_score(
                repeated_labels[i],
                repeated_labels[j]
            )
            for i in range(n_runs)
            for j in range(i + 1, n_runs)
        ]

        results.append({
            "Algorithm": algorithm,
            "Mean ARI": np.mean(ari_scores),
            "Minimum ARI": np.min(ari_scores),
            "Maximum ARI": np.max(ari_scores)
        })

    return pd.DataFrame(results)


def evaluate_perturbation_stability(features):
    results = []
    X = features.to_numpy()

    for algorithm in ["K-Means", "GMM"]:

        if algorithm == "K-Means":
            baseline_model = KMeans(
                n_clusters=2,
                init="k-means++",
                n_init=10,
                random_state=42
            )
        else:
            baseline_model = GaussianMixture(
                n_components=2,
                covariance_type="spherical",
                n_init=5,
                random_state=42
            )

        baseline_labels = baseline_model.fit_predict(X)

        for noise_level in [0.01, 0.05, 0.10]:
            scores = []

            for seed in range(5):
                rng = np.random.default_rng(seed)

                perturbed = X + rng.normal(
                    loc=0,
                    scale=noise_level,
                    size=X.shape
                )

                if algorithm == "K-Means":
                    model = KMeans(
                        n_clusters=2,
                        init="k-means++",
                        n_init=10,
                        random_state=42
                    )
                else:
                    model = GaussianMixture(
                        n_components=2,
                        covariance_type="spherical",
                        n_init=5,
                        random_state=42
                    )

                labels = model.fit_predict(perturbed)

                scores.append(
                    adjusted_rand_score(baseline_labels, labels)
                )

            results.append({
                "Algorithm": algorithm,
                "Noise Level": noise_level,
                "Mean ARI": np.mean(scores),
                "Minimum ARI": np.min(scores)
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# Final GMM model
# --------------------------------------------------

def fit_final_gmm(pca_features):
    """Fit the selected two-component spherical GMM."""

    model = GaussianMixture(
        n_components=2,
        covariance_type="spherical",
        n_init=5,
        random_state=42
    )

    labels = model.fit_predict(pca_features)

    assignments = pd.DataFrame({
        "household_key": pca_features.index,
        "cluster": labels
    })

    return model, assignments


# --------------------------------------------------
# Household-level cluster profiling
# --------------------------------------------------

PROFILE_FEATURES = [
    "transaction_records",
    "unique_baskets",
    "unique_products",
    "unique_categories",
    "total_quantity",
    "total_sales",
    "average_basket_value",
    "average_line_sales",
    "average_quantity",
    "multi_unit_share",
    "total_discount",
    "discounted_record_share",
    "private_brand_share",
    "discount_percentage"
]


def create_household_profiles(transactions, assignments):
    """Recreate household-level features using Notebook 05 definitions."""

    basket_summary = (
        transactions
        .groupby(["household_key", "BASKET_ID"])
        .agg(
            basket_value=("SALES_VALUE", "sum"),
            basket_quantity=("QUANTITY", "sum")
        )
        .reset_index()
    )

    household_features = (
        transactions.groupby("household_key")
        .agg(
            transaction_records=("BASKET_ID", "count"),
            unique_baskets=("BASKET_ID", "nunique"),
            unique_products=("PRODUCT_ID", "nunique"),
            unique_categories=("COMMODITY_DESC", "nunique"),
            total_quantity=("QUANTITY", "sum"),
            total_sales=("SALES_VALUE", "sum"),
            average_line_sales=("SALES_VALUE", "mean"),
            average_quantity=("QUANTITY", "mean"),
            multi_unit_share=("QUANTITY", lambda x: (x > 1).mean()),
            total_discount=("total_discount", "sum"),
            discounted_record_share=(
                "total_discount", lambda x: (x > 0).mean()
            ),
            private_brand_share=("BRAND", lambda x: (x == "Private").mean())
        )
    )

    average_basket_value = (
        basket_summary.groupby("household_key")["basket_value"]
        .mean()
        .rename("average_basket_value")
    )

    household_features = household_features.join(
        average_basket_value
    )

    original_value = (
        transactions.groupby("household_key")[
            "estimated_original_value"
        ].sum()
    )

    household_features["discount_percentage"] = (
        household_features["total_discount"]
        / original_value.replace(0, np.nan)
        * 100
    )

    cluster_profile_data = household_features.join(
        assignments.set_index("household_key")["cluster"]
    )

    return cluster_profile_data


def compare_cluster_profiles(cluster_profile_data):
    """Calculate mean and median of household features by cluster."""

    return (
        cluster_profile_data
        .groupby("cluster")[PROFILE_FEATURES]
        .agg(["mean", "median"])
        .round(3)
    )


# --------------------------------------------------
# Category and discount behaviour by cluster
# --------------------------------------------------

def analyse_category_preferences(transactions, assignments):
    """Compare category sales and quantities across clusters."""

    data = transactions.merge(
        assignments,
        on="household_key",
        how="inner"
    )

    category_profile = (
        data.groupby(["cluster", "COMMODITY_DESC"])
        .agg(
            total_sales=("SALES_VALUE", "sum"),
            total_quantity=("QUANTITY", "sum"),
            transaction_records=("BASKET_ID", "count")
        )
        .reset_index()
    )

    category_profile["sales_share"] = (
        category_profile["total_sales"]
        / category_profile.groupby("cluster")[
            "total_sales"
        ].transform("sum")
        * 100
    )

    return (
        category_profile
        .sort_values(
            ["cluster", "total_sales"],
            ascending=[True, False]
        )
        .groupby("cluster")
        .head(10)
        .reset_index(drop=True)
    )


def analyse_discount_behaviour(transactions, assignments):
    """Compare discount behaviour by cluster and product category."""

    data = transactions.merge(
        assignments,
        on="household_key",
        how="inner"
    )

    profile = (
        data.groupby(["cluster", "COMMODITY_DESC"])
        .agg(
            total_sales=("SALES_VALUE", "sum"),
            total_discount=("total_discount", "sum"),
            transaction_records=("BASKET_ID", "count"),
            discounted_records=(
                "total_discount", lambda x: (x > 0).sum()
            )
        )
        .reset_index()
    )

    profile["discounted_record_share"] = (
        profile["discounted_records"]
        / profile["transaction_records"]
        * 100
    )

    profile["discount_to_sales_percentage"] = (
        profile["total_discount"]
        / (
            profile["total_sales"] + profile["total_discount"]
        ).replace(0, np.nan)
        * 100
    )

    return profile


def compare_discount_categories(discount_category_profile):
    """Compare discount measures for leading categories across clusters."""

    common_categories = set()

    for cluster in [0, 1]:
        top_categories = (
            discount_category_profile[
                discount_category_profile["cluster"] == cluster
            ]
            .nlargest(10, "total_discount")["COMMODITY_DESC"]
        )

        common_categories.update(top_categories)

    comparison = (
        discount_category_profile[
            discount_category_profile["COMMODITY_DESC"].isin(
                common_categories
            )
        ]
        .pivot(
            index="COMMODITY_DESC",
            columns="cluster",
            values=[
                "discounted_record_share",
                "discount_to_sales_percentage"
            ]
        )
    )

    comparison.columns = [
        f"{metric}_cluster_{cluster}"
        for metric, cluster in comparison.columns
    ]

    comparison = comparison.reset_index()

    comparison["discounted_share_difference"] = (
        comparison["discounted_record_share_cluster_1"]
        - comparison["discounted_record_share_cluster_0"]
    )

    comparison["discount_percentage_difference"] = (
        comparison["discount_to_sales_percentage_cluster_1"]
        - comparison["discount_to_sales_percentage_cluster_0"]
    )

    return comparison.sort_values(
        "discount_percentage_difference",
        ascending=False
    )
