# Phase 7 Validation Model Evaluation

## Scope and leakage protection

Phase 7 reloaded the three saved untuned model artifacts and evaluated only the fixed 2,287 validation rows. It did not retrain a model, tune hyperparameters, transform test features, generate test predictions, or calculate test metrics.

Reloaded predictions reproduced the Phase 6 audit table with a maximum absolute difference of ₹0.00000000.

## Overall validation results

Residual means `actual price − predicted price`; a positive mean residual indicates average underprediction.

| Model | MAE | RMSE | R² | Median AE | RMSLE | Within ±20% | Mean residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy baseline | ₹396,881 | ₹834,402 | -0.0657 | ₹210,000 | 0.6946 | 29.1% | +207,175 |
| CatBoost | ₹89,656 | ₹167,077 | 0.9573 | ₹52,049 | 0.1674 | 82.6% | +7,891 |
| LightGBM | ₹96,809 | ₹262,283 | 0.8947 | ₹51,223 | 0.1691 | 81.7% | -14 |
| XGBoost | ₹99,047 | ₹411,492 | 0.7408 | ₹50,533 | 0.1728 | 82.4% | -1,237 |

CatBoost remains the strongest untuned candidate. It has the lowest MAE, RMSE, and RMSLE, the highest R², and 82.6% of its validation predictions fall within ±20% of the listing price.

![Actual versus predicted](figures/16_actual_vs_predicted.png)

## Residual behavior

![Residual distribution](figures/17_residual_distribution.png)

The median residual is close to zero for every advanced model, but the error spread grows substantially for expensive cars. LightGBM and especially XGBoost have severe high-price overpredictions that inflate RMSE. CatBoost's average residual is mildly positive, so it underpredicts by about ₹7,891 on average.

![Residuals versus predicted](figures/18_residuals_vs_predicted.png)

## Price-band stability

| Price band | Lowest-MAE model | MAE |
|---|---|---:|
| Up to ₹5 lakh | CatBoost | ₹45,168 |
| ₹5–10 lakh | XGBoost | ₹73,388 |
| ₹10–20 lakh | CatBoost | ₹165,434 |
| Above ₹20 lakh | CatBoost | ₹423,261 |

CatBoost has the lowest MAE in three of the four price bands. XGBoost is narrowly best in the ₹5–10 lakh band, but its errors are much less stable above ₹20 lakh.

![MAE by price band](figures/19_model_mae_by_price_band.png)

## Brand behavior

The major-brand chart includes brands with at least 30 validation rows. CatBoost has the lowest MAE for 8 of 13 major brands, LightGBM for 3, and XGBoost for 2. Brand results reflect different vehicle and price mixes and are not causal brand-quality comparisons.

![MAE by major brand](figures/20_model_mae_by_major_brand.png)

## Fuel and transmission behavior

Diesel and automatic listings have larger rupee MAE because they include more expensive vehicles. Rare-fuel metrics must not be treated as reliable: Electric (n=1), LPG (n=8). The single electric validation listing is especially insufficient for any model claim.

![MAE by fuel and transmission](figures/21_model_mae_by_fuel_and_transmission.png)

## Largest misses

For CatBoost, the largest underprediction is row 13852, Audi Q7: actual ₹6,350,000, predicted ₹4,098,612. The largest overprediction is row 11447, Land Rover Rover: actual ₹4,500,000, predicted ₹5,939,586.

The full largest-errors table contains the ten largest underpredictions and overpredictions for every model. These examples are diagnostic listings, not proof that one feature caused an error.

## Phase 7 conclusion

Carry **CatBoost** into Phase 8 as the primary tuning candidate and keep **LightGBM** as a challenger. XGBoost remains documented but is lower priority because its luxury-segment extremes make RMSE and R² much less stable. This is not final model selection, and the test set remains sealed.
