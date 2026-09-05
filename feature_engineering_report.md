# Phase 4 Feature Engineering

## Scope and leakage protection

Phase 4 evaluates six deterministic candidate features. A fixed 15% test holdout was reserved first using price-band stratification and random seed 42. The 2,287 test rows were **not** scored, inspected for model performance, or used for feature selection. Five-fold cross-validation used only the remaining 12,957 development rows.

All imputation was fitted inside each training fold. The diagnostic estimator trained on `log1p(selling_price)`, and MAE/RMSE were calculated after predictions were converted back to Indian rupees.

## Candidate definitions

- `km_per_year = km_driven / max(vehicle_age, 1)`
- `log_km_driven = log1p(km_driven)`
- `power_per_cc = max_power / engine`
- `engine_per_seat = engine / seats`
- `is_automatic = 1` for automatic transmission, otherwise `0`
- `is_luxury_brand = 1` for the documented mapping in `src/features.py`

Undefined ratios remain missing and are imputed from each training fold. The luxury mapping is an explicit heuristic, not a target-derived rule.

## Cross-validation results

| Experiment | Added features | Mean MAE | Improvement vs baseline | Better folds | Decision |
|---|---|---:|---:|---:|---|
| baseline | none | ₹97,821 | +0 | 0/5 | Do not prioritize |
| add_km_per_year | km_per_year | ₹98,302 | -481 | 0/5 | Do not prioritize |
| add_log_km_driven | log_km_driven | ₹97,821 | +0 | 0/5 | Do not prioritize |
| add_power_per_cc | power_per_cc | ₹97,941 | -120 | 2/5 | Do not prioritize |
| add_engine_per_seat | engine_per_seat | ₹97,710 | +110 | 3/5 | Do not prioritize |
| add_is_automatic | is_automatic | ₹97,821 | +0 | 0/5 | Do not prioritize |
| add_is_luxury_brand | is_luxury_brand | ₹97,493 | +327 | 3/5 | Do not prioritize |
| all_candidates | km_per_year, log_km_driven, power_per_cc, engine_per_seat, is_automatic, is_luxury_brand | ₹98,528 | -707 | 2/5 | Do not prioritize |

The baseline diagnostic MAE is ₹97,821. The lowest mean MAE is ₹97,493 from `add_is_luxury_brand`, an improvement of ₹327 (0.33%). The individual features meeting the transparent carry-forward rule are none of the individual candidates.

![Cross-validated feature comparison](figures/11_feature_engineering_mae_comparison.png)

## Interpretation

- This experiment tests incremental value beyond the existing raw columns; it does not rank general feature importance.
- `is_automatic` duplicates information already present in `transmission_type`, and `is_luxury_brand` summarizes information already present in `brand`. They should only be retained if cross-validation shows consistent incremental value.
- Ratio features can help express vehicle usage or specification intensity, but they can also add noise. Their value must be judged from validation evidence rather than intuition.
- The diagnostic histogram boosting model is used only to compare feature sets efficiently on an 8 GB computer. It is not one of the final candidate models and is not saved.
- Small differences should not be treated as guaranteed improvements. Phase 5 must apply the fixed split and compare the dummy baseline before advanced-model training begins.

## Phase 4 conclusion

Carry the supported candidates into Phase 5 as experiments, not permanent assumptions. Keep the test assignment untouched until final model evaluation.
