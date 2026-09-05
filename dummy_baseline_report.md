# Phase 5 Data Split and Dummy Baseline

## Fixed split

The cleaned dataset is divided once using fixed price-band stratification and random seed 42:

- Training: 10,670 rows (70.0%)
- Validation: 2,287 rows (15.0%)
- Test: 2,287 rows (15.0%)

The Phase 4 test holdout is preserved exactly. No test predictions or test metrics were generated. Every advanced model must use this same split for a fair comparison.

![Fixed split distribution](figures/12_split_distribution.png)

## Median dummy baseline

`DummyRegressor(strategy="median")` learned one number from the training targets: **₹560,000**. It predicts that amount for every listing, regardless of brand, age, kilometres, or specifications.

Validation results on the original rupee scale:

- MAE: **₹396,881**
- RMSE: **₹834,402**
- R²: **-0.0657**
- Median absolute error: **₹210,000**
- RMSLE: **0.6946**

R² is expected to be near zero or negative for a constant median predictor. The baseline exists to prove that advanced models learn useful relationships; it is not a practical recommendation engine.

## Validation performance by price band

| Price band | Rows | Actual median | MAE | RMSE | Mean error |
|---|---:|---:|---:|---:|---:|
| Up to ₹5 lakh | 983 | ₹350,000 | ₹216,749 | ₹240,928 | +216,749 |
| ₹5–10 lakh | 931 | ₹675,000 | ₹152,615 | ₹198,380 | -144,313 |
| ₹10–20 lakh | 256 | ₹1,327,500 | ₹821,555 | ₹865,255 | -821,555 |
| Above ₹20 lakh | 117 | ₹2,925,000 | ₹2,924,786 | ₹3,342,185 | -2,924,786 |

A positive mean error means overprediction; a negative value means underprediction. The single median prediction is expected to overprice inexpensive cars and severely underprice expensive cars.

![Dummy baseline validation](figures/13_dummy_baseline_validation.png)

## Phase 5 conclusion

The baseline establishes the minimum benchmark that CatBoost, LightGBM, and XGBoost must beat. Phase 6 should prioritize lower validation MAE in rupees while checking RMSE, R², RMSLE, and segment stability. The test set remains unavailable.
