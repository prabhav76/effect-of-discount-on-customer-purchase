
import matplotlib.pyplot as plt


def plot_top_association_rules(interpretable_rules, top_n=15):
    if interpretable_rules.empty:
        print("No association rules available.")
        return

    top_rules = (
        interpretable_rules
        .sort_values("lift", ascending=False)
        .head(top_n)
        .copy()
    )

    top_rules["rule"] = top_rules.apply(
        lambda row: (
            f"{', '.join(row['antecedents'])} "
            f"→ {', '.join(row['consequents'])}"
        ),
        axis=1
    )

    top_rules = top_rules.sort_values("lift")

    plt.figure(figsize=(12, 8))
    plt.barh(top_rules["rule"], top_rules["lift"])

    plt.xlabel("Lift")
    plt.ylabel("Association Rule")
    plt.title(f"Top {top_n} Product Association Rules by Lift")

    plt.tight_layout()
    plt.show()


def plot_support_confidence_lift(rules):
    if rules.empty:
        print("No association rules available.")
        return

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        s=80,
        alpha=0.8,
        edgecolors="black"
    )

    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules: Support vs Confidence")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_support_confidence_lift(rules, output_path):
    if rules.empty:
        print("No association rules available.")
        return

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        s=80,
        alpha=0.8,
        edgecolors="black"
    )

    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules: Support vs Confidence")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Visualisation saved to:", output_path)
