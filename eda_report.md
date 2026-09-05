# Phase 3 Exploratory Data Analysis

## Scope

This report analyzes `data/processed/cardekho_cleaned.csv` with 15,244 listings and 12 columns. Phase 3 does not impute, split, engineer model features, or train models.

## Main findings

1. **Prices are strongly right-skewed.** The median is ₹5.6 L, the mean is ₹7.7 L, and skewness is 10.11. The maximum of ₹4.0 Cr belongs to the luxury tail, so medians and log-scale charts are more representative than raw averages.
2. **The market is concentrated below ₹10 lakh.** 43.0% of listings are at or below ₹5 lakh and 40.7% are between ₹5 lakh and ₹10 lakh. Only 5.1% exceed ₹20 lakh.
3. **Vehicle age has a strong monotonic relationship with price.** Spearman correlation is -0.478. The cross-sectional median falls from ₹8.2 L at age two to ₹3.4 L at age ten (-59.1%). This is not a true depreciation rate because the vehicle mix changes across ages.
4. **Power and engine size are the strongest positive numerical signals.** Maximum power has Pearson correlation 0.751 and Spearman correlation 0.721 with price. Engine size has Spearman correlation 0.643.
5. **Category medians reflect different vehicle mixes.** Automatic vehicles have a median of ₹10.5 L, compared with ₹5.0 L for manual vehicles. Diesel median price is ₹7.0 L, compared with ₹4.6 L for petrol. These gaps are not causal estimates.
6. **Dealer listings remain somewhat higher even in a basic matched comparison.** Across 258 brand-model-age groups with at least five Dealer and five Individual listings, the median Dealer-minus-Individual difference is ₹28K; Dealer medians are higher in 76.0% of groups. Condition, trim, location, and other unobserved factors may still explain part of this difference.
7. **Coverage is uneven.** The data includes 32 brands and 120 model labels, but 16 brands and 65 models have fewer than 50 examples. Predictions for rare or unseen groups will need wider intervals and lower confidence.

## Figures

### Price distribution

![Raw and log price distributions](figures/01_price_distribution.png)

### Brand and model comparisons

![Median price by common brand](figures/02_brand_price_comparison.png)

![Median price by well-represented model](figures/03_model_price_comparison.png)

### Category comparisons

![Fuel, transmission, and seller medians](figures/04_category_price_comparison.png)

### Numerical relationships

![Vehicle age and price](figures/05_vehicle_age_price_trend.png)

![Kilometres driven and price](figures/06_km_price_relationship.png)

![Engine and power relationships](figures/07_specifications_price_relationship.png)

![Spearman correlation heatmap](figures/08_spearman_correlation_heatmap.png)

### Market composition and coverage

![Price-band distribution](figures/09_price_band_distribution.png)

![Category coverage](figures/10_category_coverage.png)

## Data-quality context

- The two missing `seats` values remain unimputed.
- All 12 records above 500,000 km remain in the analysis and are displayed on log scales where applicable.
- Electric has only 4 rows and LPG has only 44 rows. Their median prices are descriptive, not reliable market estimates.
- Group comparisons are observational and confounded by model, trim, condition, and other omitted variables.
- The dataset lacks listing dates, locations, service history, accident history, ownership count, and confirmed transaction prices.

## Phase 3 conclusion

The EDA supports log-target experiments, strong treatment of categorical identity, explicit age and usage features, and segment-level evaluation. It does not choose a model or justify deleting legitimate luxury prices. Feature engineering must be evaluated with cross-validation in Phase 4.
