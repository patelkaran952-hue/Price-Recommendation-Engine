# Phase 6 Untuned Model Training

## Scope

CatBoost, LightGBM, and XGBoost were trained on the fixed 10,670 training rows and compared on the fixed 2,287 validation rows. All three trained on `log1p(selling_price)` and were evaluated after predictions were converted back to rupees.

The 2,287 test rows were not transformed, predicted, or evaluated. Hyperparameter tuning was not performed.

## Validation comparison

| Model | MAE | RMSE | R² | Median AE | RMSLE | MAE gain vs dummy | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy baseline | ₹396,881 | ₹834,402 | -0.0657 | ₹210,000 | 0.6946 | 0.0% | 0.0s |
| CatBoost | ₹89,656 | ₹167,077 | 0.9573 | ₹52,049 | 0.1674 | 77.4% | 7.0s |
| LightGBM | ₹96,809 | ₹262,283 | 0.8947 | ₹51,223 | 0.1691 | 75.6% | 0.8s |
| XGBoost | ₹99,047 | ₹411,492 | 0.7408 | ₹50,533 | 0.1728 | 75.0% | 2.8s |

The dummy baseline MAE is ₹396,881. The preliminary best untuned model is **CatBoost**, with MAE ₹89,656—a 77.4% reduction. This is a validation-only comparison, not final model selection.

![Overall validation comparison](figures/14_model_validation_comparison.png)

## Preprocessing

- **CatBoost:** categorical columns were passed explicitly after conversion to stable strings. Numeric missing values were filled using training-only medians.
- **LightGBM and XGBoost:** training-only median/mode imputation and `OneHotEncoder(handle_unknown="ignore")`; no `StandardScaler`.
- **All models:** early stopping used validation log-RMSE, random seed 42, and at most 4 CPU threads.

## Price-band behavior

![Validation MAE by price band](figures/15_model_mae_by_price_band.png)

The price-band table is saved separately because overall MAE can hide weak performance on expensive cars. Phase 7 will expand this into residual, brand, fuel, and transmission diagnostics.

## Phase 6 conclusion

All three advanced models must be carried into Phase 7 for detailed validation analysis. The preliminary winner should not be tuned or declared final solely from this table. The test set remains unavailable.
