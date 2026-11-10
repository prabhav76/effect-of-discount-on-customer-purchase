
import matplotlib.pyplot as plt


# --------------------------------------------------
# Discount distributions
# --------------------------------------------------

def plot_discount_histograms(transactions, analysis_cols):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(analysis_cols):
        axes[i].hist(transactions[col].dropna(), bins=50)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for i in range(len(analysis_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_central_99_distributions(transactions, analysis_cols):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(analysis_cols):
        data = transactions[col].dropna()
        upper_limit = data.quantile(0.99)

        axes[i].hist(data[data <= upper_limit], bins=50)
        axes[i].set_title(f"{col} (Central 99%)")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for i in range(len(analysis_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_discount_boxplots(transactions, analysis_cols):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(analysis_cols):
        axes[i].boxplot(transactions[col].dropna())
        axes[i].set_title(col)
        axes[i].set_ylabel("Value")

    for i in range(len(analysis_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Discount vs purchasing behaviour
# --------------------------------------------------

def plot_discount_vs_quantity(transactions):
    data = transactions[
        ["discount_percentage", "QUANTITY"]
    ].dropna()

    sample = data.sample(
        n=min(20000, len(data)),
        random_state=42
    )

    plt.figure(figsize=(9, 5))
    plt.scatter(
        sample["discount_percentage"],
        sample["QUANTITY"],
        alpha=0.3
    )

    plt.title("Discount Percentage vs Quantity Purchased")
    plt.xlabel("Discount Percentage")
    plt.ylabel("Quantity Purchased")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_discount_vs_sales(transactions):
    data = transactions[
        ["discount_percentage", "SALES_VALUE"]
    ].dropna()

    sample = data.sample(
        n=min(20000, len(data)),
        random_state=42
    )

    plt.figure(figsize=(9, 5))
    plt.scatter(
        sample["discount_percentage"],
        sample["SALES_VALUE"],
        alpha=0.3
    )

    plt.title("Discount Percentage vs Sales Value")
    plt.xlabel("Discount Percentage")
    plt.ylabel("Sales Value")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Commodity and brand comparisons
# --------------------------------------------------

def plot_commodity_comparison(reliable_commodity_comparison):
    data = (
        reliable_commodity_comparison
        .sort_values("difference", ascending=False)
        .head(15)
        .sort_values("difference")
    )

    plt.figure(figsize=(10, 7))
    plt.barh(data.index, data["difference"])

    plt.xlabel(
        "Difference in Multi-Unit Purchases (Percentage Points)"
    )
    plt.ylabel("Commodity")
    plt.title("Discounted vs Non-Discounted Purchases by Commodity")

    plt.tight_layout()
    plt.show()


def plot_brand_comparison(brand_comparison):
    brand_plot = brand_comparison.pivot(
        index="BRAND",
        columns="discount_status",
        values="pct_qty_above_1"
    )

    brand_plot.plot(kind="bar", figsize=(8, 5))

    plt.ylabel("Records with Quantity > 1 (%)")
    plt.xlabel("Brand Type")
    plt.title("Multi-Unit Purchases by Brand and Discount Status")
    plt.xticks(rotation=0)
    plt.legend(title="Discount Status")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Weekly trends
# --------------------------------------------------

def plot_weekly_transaction_activity(weekly_records):
    plt.figure(figsize=(12, 5))

    plt.plot(
        weekly_records["WEEK_NO"],
        weekly_records["total_records"],
        marker="o",
        markersize=3
    )

    plt.xlabel("Week Number")
    plt.ylabel("Number of Transaction Records")
    plt.title("Weekly Transaction Activity")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_weekly_discount_percentage(weekly_discount):
    plt.figure(figsize=(12, 5))

    plt.plot(
        weekly_discount["WEEK_NO"],
        weekly_discount["discount_percentage"],
        marker="o",
        markersize=3
    )

    plt.xlabel("Week Number")
    plt.ylabel("Discounted Records (%)")
    plt.title("Weekly Discounted Transaction Percentage")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_weekly_multiunit_percentage(weekly_multiunit):
    weekly_plot = weekly_multiunit.pivot(
        index="WEEK_NO",
        columns="discount_status",
        values="multiunit_percentage"
    )

    plt.figure(figsize=(12, 5))

    for status in ["Discounted", "Non-Discounted"]:
        if status in weekly_plot.columns:
            plt.plot(
                weekly_plot.index,
                weekly_plot[status],
                label=status,
                marker="o",
                markersize=3
            )

    plt.xlabel("Week Number")
    plt.ylabel("Records with Quantity > 1 (%)")
    plt.title("Weekly Multi-Unit Purchase Percentage")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Outliers and transformations
# --------------------------------------------------

def plot_outlier_boxplots(transactions, outlier_cols):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, col in zip(axes.flatten(), outlier_cols):
        ax.boxplot(transactions[col].dropna(), vert=False)
        ax.set_title(col)
        ax.set_xlabel("Value")

    for i in range(len(outlier_cols), len(axes.flatten())):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()


def plot_sales_log_transformation(transactions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(transactions["SALES_VALUE"], bins=50)
    axes[0].set_title("Original Sales Value")
    axes[0].set_xlabel("Sales Value")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(transactions["SALES_VALUE_log"], bins=50)
    axes[1].set_title("Log-Transformed Sales Value")
    axes[1].set_xlabel("Log(1 + Sales Value)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_discount_log_transformation(transactions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(transactions["total_discount"], bins=50)
    axes[0].set_title("Original Total Discount")
    axes[0].set_xlabel("Total Discount")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(transactions["total_discount_log"], bins=50)
    axes[1].set_title("Log-Transformed Total Discount")
    axes[1].set_xlabel("Log(1 + Total Discount)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_discount_percentage_distribution(transactions):
    values = transactions["discount_percentage"].dropna()

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=50)

    plt.title("Discount Percentage Distribution")
    plt.xlabel("Discount Percentage")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print("Skewness:", values.skew())


def plot_yeojohnson_distributions(transactions, variables):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, col in zip(axes.flatten(), variables):
        transformed_col = col + "_yeojohnson"

        ax.hist(transactions[transformed_col], bins=50)
        ax.set_title(col + " - Yeo-Johnson")
        ax.set_xlabel("Transformed Value")
        ax.set_ylabel("Frequency")

    for i in range(len(variables), len(axes.flatten())):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()
