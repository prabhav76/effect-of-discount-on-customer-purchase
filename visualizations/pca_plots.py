
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


# --------------------------------------------------
# Household feature correlation
# --------------------------------------------------

def plot_feature_correlations(household_features):
    """Plot correlations between household features."""

    feature_correlation = household_features.corr()

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        feature_correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title("Correlation Between Household Features")
    plt.tight_layout()
    plt.show()

    return feature_correlation


# --------------------------------------------------
# Household feature distributions
# --------------------------------------------------

SKEWED_FEATURES = [
    "transaction_records",
    "unique_baskets",
    "total_quantity",
    "total_sales",
    "average_quantity",
    "total_discount",
    "average_basket_value"
]


def plot_feature_distributions(household_features):
    """Plot distributions of selected household features."""

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, column in enumerate(SKEWED_FEATURES):
        sns.histplot(
            household_features[column],
            bins=40,
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(f"Distribution of {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Number of Households")

    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# PCA explained variance
# --------------------------------------------------

def plot_explained_variance(pca):
    """Plot individual and cumulative PCA explained variance."""

    n_components = len(pca.explained_variance_ratio_)
    component_numbers = range(1, n_components + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scree plot
    axes[0].plot(
        component_numbers,
        pca.explained_variance_ratio_,
        marker="o"
    )

    axes[0].set_title("Scree Plot")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_xticks(list(component_numbers))
    axes[0].grid(True)

    # Cumulative explained variance
    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    axes[1].plot(
        component_numbers,
        cumulative_variance,
        marker="o"
    )

    axes[1].axhline(
        y=0.90,
        linestyle="--",
        label="90% variance"
    )

    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance")
    axes[1].set_xticks(list(component_numbers))
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# PCA component loadings
# --------------------------------------------------

def plot_pca_loadings(pca_loadings):
    """Plot the PCA component loading heatmap."""

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        pca_loadings,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )

    plt.title("PCA Component Loadings")
    plt.xlabel("Principal Components")
    plt.ylabel("Original Features")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Household distribution in PCA space
# --------------------------------------------------

def plot_pca_scatter(
    pca_features,
    household_features
):
    """Plot households in the first two principal components."""

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        pca_features["PC1"],
        pca_features["PC2"],
        c=household_features["discount_percentage"],
        cmap="viridis",
        alpha=0.7,
        s=25
    )

    plt.colorbar(
        scatter,
        label="Household Discount Percentage"
    )

    plt.title("Households in 2D PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Household distribution in t-SNE space
# --------------------------------------------------

def plot_tsne_scatter(
    tsne_features,
    household_features
):
    """Plot households in t-SNE space."""

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        tsne_features["TSNE1"],
        tsne_features["TSNE2"],
        c=household_features["discount_percentage"],
        cmap="viridis",
        alpha=0.7,
        s=25
    )

    plt.colorbar(
        scatter,
        label="Household Discount Percentage"
    )

    plt.title("Households in t-SNE Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.tight_layout()
    plt.show()
