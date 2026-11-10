
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


LOG_FEATURES = [
    "unique_baskets",
    "total_quantity",
    "total_sales",
    "average_line_sales",
    "average_quantity",
    "total_discount"
]

UNCHANGED_FEATURES = [
    "transaction_records",
    "unique_products",
    "unique_categories",
    "multi_unit_share",
    "discounted_record_share",
    "private_brand_share",
    "average_basket_value",
    "discount_percentage"
]


# --------------------------------------------------
# Create household-level features
# --------------------------------------------------

def create_household_features(transactions):
    """Aggregate transaction records into household-level features."""

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
        transactions
        .groupby("household_key")
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
            private_brand_share=(
                "BRAND", lambda x: (x == "Private").mean()
            )
        )
    )

    # Average basket value per household
    average_basket_value = (
        basket_summary
        .groupby("household_key")["basket_value"]
        .mean()
        .rename("average_basket_value")
    )

    household_features = household_features.join(
        average_basket_value
    )

    # Weighted household discount percentage
    household_original_value = (
        transactions
        .groupby("household_key")["estimated_original_value"]
        .sum()
    )

    household_features["discount_percentage"] = (
        household_features["total_discount"]
        / household_original_value.replace(0, np.nan)
        * 100
    )

    return household_features


# --------------------------------------------------
# Transform skewed features
# --------------------------------------------------

def transform_household_features(household_features):
    """Apply log1p to the selected features."""

    transformed = household_features.copy()

    transformed[LOG_FEATURES] = np.log1p(
        transformed[LOG_FEATURES]
    )

    return transformed


# --------------------------------------------------
# Standardize household features
# --------------------------------------------------

def standardize_household_features(transformed_features):
    """Standardize all household features using StandardScaler."""

    if transformed_features.isna().any().any():
        missing = transformed_features.isna().sum()
        missing = missing[missing > 0]

        raise ValueError(
            "Missing values found in household features. "
            "Check these columns before standardization:\n"
            f"{missing}"
        )

    if not np.isfinite(transformed_features.to_numpy()).all():
        raise ValueError(
            "Infinite values found in household features."
        )

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(
        transformed_features
    )

    scaled_features = pd.DataFrame(
        scaled_array,
        columns=transformed_features.columns,
        index=transformed_features.index
    )

    return scaled_features, scaler


# --------------------------------------------------
# Apply PCA
# --------------------------------------------------

def run_pca(scaled_features, n_components=6):
    """Fit PCA and retain the requested number of components."""

    if n_components > min(scaled_features.shape):
        raise ValueError(
            "Number of PCA components exceeds the allowed limit."
        )

    pca = PCA()

    pca_all = pca.fit_transform(scaled_features)

    # Explained variance
    pca_variance = pd.DataFrame({
        "Component": [
            f"PC{i + 1}"
            for i in range(len(pca.explained_variance_ratio_))
        ],
        "Explained Variance Ratio": pca.explained_variance_ratio_,
        "Cumulative Explained Variance": np.cumsum(
            pca.explained_variance_ratio_
        )
    })

    # Retain the first six components
    reduced_features = pd.DataFrame(
        pca_all[:, :n_components],
        columns=[
            f"PC{i + 1}" for i in range(n_components)
        ],
        index=scaled_features.index
    )

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=scaled_features.columns,
        columns=[
            f"PC{i + 1}" for i in range(n_components)
        ]
    )

    return pca, reduced_features, pca_variance, loadings


# --------------------------------------------------
# Calculate reconstruction errors
# --------------------------------------------------

def calculate_reconstruction_errors(
    scaled_features,
    reduced_features,
    pca
):
    """Calculate overall and feature-wise reconstruction errors."""

    reconstructed = (
        reduced_features.to_numpy()
        @ pca.components_[:reduced_features.shape[1]]
        + pca.mean_
    )

    original = scaled_features.to_numpy()

    overall_error = np.mean(
        (original - reconstructed) ** 2
    )

    feature_errors = pd.DataFrame({
        "Feature": scaled_features.columns,
        "Reconstruction Error": np.mean(
            (original - reconstructed) ** 2,
            axis=0
        )
    })

    feature_errors = feature_errors.sort_values(
        "Reconstruction Error",
        ascending=False
    )

    return overall_error, feature_errors


# --------------------------------------------------
# Apply t-SNE
# --------------------------------------------------

def run_tsne(scaled_features):
    """Generate a two-dimensional t-SNE representation."""

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    coordinates = tsne.fit_transform(
        scaled_features
    )

    return pd.DataFrame(
        coordinates,
        columns=["TSNE1", "TSNE2"],
        index=scaled_features.index
    )


# --------------------------------------------------
# Save PCA and t-SNE outputs
# --------------------------------------------------

def save_pca_outputs(
    output_dir,
    scaled_features,
    reduced_features,
    variance,
    loadings,
    reconstruction_errors,
    tsne_features,
    transformed_features
):
    """Save the outputs generated by the dimensionality-reduction workflow."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaled_features.to_csv(
        output_dir / "household_features_scaled.csv"
    )

    reduced_features.to_csv(
        output_dir / "household_pca_features.csv"
    )

    variance.to_csv(
        output_dir / "pca_explained_variance.csv",
        index=False
    )

    loadings.to_csv(
        output_dir / "pca_component_loadings.csv"
    )

    reconstruction_errors.to_csv(
        output_dir / "pca_reconstruction_errors.csv",
        index=False
    )

    tsne_features.to_csv(
        output_dir / "household_tsne_features.csv"
    )

    transformation_decisions = pd.DataFrame({
        "Feature": LOG_FEATURES + UNCHANGED_FEATURES,
        "Decision": (
            ["Apply log1p"] * len(LOG_FEATURES)
            + ["Keep unchanged"] * len(UNCHANGED_FEATURES)
        )
    })

    transformation_decisions.to_csv(
        output_dir / "pca_transformation_decisions.csv",
        index=False
    )

    print("PCA and t-SNE outputs saved to:", output_dir)


# --------------------------------------------------
# Complete dimensionality-reduction workflow
# --------------------------------------------------

def run_dimensionality_reduction(
    transactions,
    output_dir,
    n_components=6
):
    """Run household feature engineering, PCA and t-SNE."""

    # Household-level feature engineering
    household_features = create_household_features(
        transactions
    )

    # Log transformation
    transformed_features = transform_household_features(
        household_features
    )

    # Standardization
    scaled_features, scaler = standardize_household_features(
        transformed_features
    )

    # PCA
    (
        pca,
        reduced_features,
        variance,
        loadings
    ) = run_pca(
        scaled_features,
        n_components=n_components
    )

    # Reconstruction errors
    overall_error, feature_errors = (
        calculate_reconstruction_errors(
            scaled_features,
            reduced_features,
            pca
        )
    )

    # t-SNE
    tsne_features = run_tsne(
        scaled_features
    )

    # Save outputs
    save_pca_outputs(
        output_dir,
        scaled_features,
        reduced_features,
        variance,
        loadings,
        feature_errors,
        tsne_features,
        transformed_features
    )

    print(
        "Retained PCA variance:",
        round(
            pca.explained_variance_ratio_[:n_components].sum() * 100,
            2
        ),
        "%"
    )

    print(
        "Mean squared reconstruction error:",
        round(overall_error, 4)
    )

    return {
        "household_features": household_features,
        "transformed_features": transformed_features,
        "scaled_features": scaled_features,
        "scaler": scaler,
        "pca": pca,
        "pca_features": reduced_features,
        "pca_variance": variance,
        "pca_loadings": loadings,
        "reconstruction_error": overall_error,
        "feature_reconstruction_errors": feature_errors,
        "tsne_features": tsne_features
    }
