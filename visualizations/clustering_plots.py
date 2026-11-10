
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kmeans_tuning(kmeans_tuning_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for representation in ["Original", "PCA"]:
        results = kmeans_tuning_results[
            kmeans_tuning_results["Representation"] == representation
        ]

        axes[0].plot(
            results["K"],
            results["Inertia"],
            marker="o",
            label=representation
        )

        axes[1].plot(
            results["K"],
            results["Silhouette"],
            marker="o",
            label=representation
        )

    axes[0].set_title("K-Means Elbow Method")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_gmm_clusters(pca_features, cluster_assignments):
    if len(pca_features) != len(cluster_assignments):
        raise ValueError(
            "PCA features and cluster assignments do not match."
        )

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x=pca_features.iloc[:, 0],
        y=pca_features.iloc[:, 1],
        hue=cluster_assignments["cluster"].values,
        palette="Set1",
        alpha=0.65,
        s=35
    )

    plt.title("Customer Clusters — GMM on PCA Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def save_gmm_clusters(pca_features, cluster_assignments, output_path):
    if len(pca_features) != len(cluster_assignments):
        raise ValueError(
            "PCA features and cluster assignments do not match."
        )

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x=pca_features.iloc[:, 0],
        y=pca_features.iloc[:, 1],
        hue=cluster_assignments["cluster"].values,
        palette="Set1",
        alpha=0.65,
        s=35
    )

    plt.title("Customer Clusters — GMM on PCA Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Cluster visualisation saved successfully.")
