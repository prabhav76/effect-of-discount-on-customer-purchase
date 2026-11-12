# Effect of Discounts on Customer Purchase Behavior

This project studies how recorded discounts and promotions are **associated** with grocery purchase behavior. It combines transaction data with product and promotion records, then analyzes purchase quantity, discount levels, brand/category differences, household profiles, and products bought together.

The work is observational. Results describe relationships in the available data; they do not prove that discounts caused a change in purchasing.

## Research questions

1. Are discounted transaction records more likely to contain a multi-unit purchase (`QUANTITY > 1`) than non-discounted records, after accounting for repeated observations from the same household?
2. Does the multi-unit purchase rate vary across discount percentage bands?
3. Does the discount association differ between National and Private brands?
4. Does the association differ across product categories?

The project also studies recorded promotion exposure and coupon use, groups households with PCA and Gaussian mixture clustering, and mines market-basket association rules.

## Main findings recorded in the notebooks

| Analysis | Reported result |
| --- | --- |
| Discount status | Multi-unit purchase rate was 24.28% for discounted records and 16.45% for non-discounted records, a 7.83 percentage-point difference (household-bootstrap 95% CI: 7.48–8.20). The household-stratified common odds ratio was 1.611 (95% CI: 1.601–1.621). |
| Discount bands | The low-discount band had the highest multi-unit rate (29.80%); no-discount records had the lowest (16.58%). The association was not a simple “more discount, more units” pattern. |
| Brand | Discount-associated odds were 1.48 for National brands and 1.55 for Private brands; the interaction test was significant (p = 0.032). |
| Product categories | After complete-outcome checks and Holm correction, 124 of 258 categories were significant: 110 positive and 14 negative associations. |
| Promotion and coupons | Promotion metadata was recorded for 21.72% of records. Basket sales and discount percentage comparisons remained significant after correction; household-level sales differences for coupon users and promotion-exposed households did not. |
| Basket rules | Apriori and FP-Growth agreed on 31 rules. The strongest reported lift was 14.15 for hot dog buns → hamburger buns. |

These are the saved notebook results, not a fresh run against data in this repository. The original data and output CSVs are local-only and are not present here.

## Repository layout

- `preprocessing/` joins source tables and derives discount features.
- `analysis/` contains reusable statistical, clustering, PCA, and association-rule functions.
- `notebooks/` contains the exploratory analysis and the full course-project workflow, in order from 01 to 08.
- `visualizations/` contains plots used by the notebooks.
- `models/` contains saved scaler, PCA, and Gaussian mixture model artifacts.
- `frontend/` contains the React and Tailwind research dashboard.
- `data/raw/` and `data/processed/` are local-only folders; the datasets and generated outputs are excluded from Git.

## Data setup

The raw data is not included in this repository. Put the source CSV files in `data/raw/`. The preprocessing script requires:

- `transaction_data.csv`
- `causal_data.csv`
- `product.csv`

The transaction table is expected to include the household, basket, product, store, week, sales, quantity, and discount fields used in the notebooks. The product table supplies commodity, sub-commodity, and brand fields. The causal table supplies display and mailer promotion fields.

## Python workflow

Use Python 3.10 or 3.11 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Build the joined transaction tables and derive validated discount features:

```bash
python preprocessing/merge_datasets.py
python preprocessing/clean_data.py
```

These commands write `transactions_promo.csv`, `transactions_enriched.csv`, and `transactions_cleaned.csv` under `data/processed/`. The cleaning step is optional for the notebook workflow: it creates an additional validated feature file, while the analysis notebooks use `transactions_enriched.csv` as their main input. It processes rows in chunks, preserves all rows, and reports invalid or missing derived discount values.

### Rebuilding all processed analysis outputs

Keep all eight source CSVs in `data/raw/`. From the repository root, start Jupyter with the notebooks directory as its working directory so the notebooks' `../data/...` paths resolve correctly:

```bash
cd notebooks
jupyter lab
```

Then open and run every cell in order in these notebooks:

1. `01_dataset_exploration.ipynb` reads all eight raw CSVs and creates `data/processed/transactions_promo.csv`.
2. `02_data_preprocessing.ipynb` reads that file and `product.csv`, then creates `transactions_enriched.csv`.
3. `03_exploratory_data_analysis.ipynb` reads the generated transaction tables and runs the exploratory analysis.
4. `04_statistical_analysis.ipynb` writes the statistical analysis tables.
5. `05_dimensionality_reduction.ipynb` writes household features used by clustering.
6. `06_association_rule_mining.ipynb` writes frequent itemsets, rules, and findings.
7. `07_customer_clustering.ipynb` uses outputs from notebook 05 and writes clustering results.
8. `08_discount_effectiveness.ipynb` writes discount and promotion effectiveness results.

The notebooks create the analysis folders under `data/processed/` as they run. Run notebook 05 before 07 because clustering depends on its household feature files. Alternatively, `python preprocessing/merge_datasets.py` can create the two base transaction tables from `transaction_data.csv`, `causal_data.csv`, and `product.csv`; if you use that shortcut, continue with notebooks 03–08. The optional `python preprocessing/clean_data.py` creates `transactions_cleaned.csv`; it does not recreate the analysis outputs.

## Methods at a glance

- Discount amounts are converted from the source's negative discount values to positive amounts; discount percentage is calculated only when the estimated pre-discount value is positive.
- Multi-unit purchases are transaction records with quantity greater than one. Purchase-rate comparisons use a household bootstrap and a Cochran–Mantel–Haenszel test to account for repeated household observations.
- Discount bands, brand interactions, and category-level associations use household-clustered binomial GEE models. Pairwise and category tests use Holm correction for multiple comparisons.
- Promotion and coupon analyses use household- or basket-level Mann–Whitney comparisons and Holm correction. Missing promotion metadata means “not recorded,” not “known not promoted.”
- Household features are scaled and reduced with PCA before clustering. Association rules are cross-checked between Apriori and FP-Growth.

## Dashboard

```bash
cd frontend
npm install
npm run dev
```

The React frontend reads a generated JSON summary built from the current processed-analysis CSVs. `npm run dev` and `npm run build` refresh this summary automatically. If you change source transactions, rerun the relevant analysis notebooks first; then refresh or restart the frontend so the updated results appear. The browser does not load the multi-hundred-megabyte transaction files directly.

## Important interpretation notes

- A statistically significant association is not automatically a practically large effect or a causal effect.
- Quantity and discount data contain skew and unusual records; inspect the notebook diagnostics before interpreting averages.
- Promotion metadata can be missing. Treat that group as “no metadata recorded,” not as a confirmed no-promotion control group.
- Household clusters are numeric model outputs. Give them behavioral names only after reviewing the saved cluster profiles.
- Re-run the notebooks when the source data changes so derived results remain in sync with the analysis.
