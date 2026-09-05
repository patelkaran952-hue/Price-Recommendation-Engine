# Phase 8 Hyperparameter Tuning

## Scope and guardrails

Phase 8 tuned only the Phase 7 primary candidate, CatBoost, and its challenger, LightGBM. The randomized searches used the fixed 10,670 training rows only. The fixed 2,287 validation rows were used only after the searches, for early stopping during the final refit and one post-search comparison.

The 2,287 test rows were not selected, transformed, predicted, or evaluated.

## Search design

- **Objective:** minimize mean MAE in original rupees after reversing the log target.
- **Cross-validation:** three shuffled folds stratified by the fixed price bands, random seed 42.
- **Leakage control:** numeric imputation and categorical processing were fit separately inside every training fold.
- **Compute limit:** four CPU threads, 8 CatBoost trials and 6 LightGBM trials.
- **Early stopping:** 75 rounds inside CV and 100 rounds during the final validation-assisted refit.

The complete search spaces, every sampled configuration, fold metrics, best iteration, and runtime are recorded in the JSON summary and CSV audit tables.

![Cross-validation search results](figures/22_tuning_cv_results.png)

## Best cross-validation configurations

### CatBoost

- Best trial: 1 of 8
- Mean CV MAE: ₹101,263 ± ₹7,870
- Search time: 112.5 seconds
- Parameters: `{"bagging_temperature": 3.0, "border_count": 128, "depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.08, "random_strength": 0.5}`

### LightGBM

- Best trial: 2 of 6
- Mean CV MAE: ₹102,438 ± ₹7,105
- Search time: 16.9 seconds
- Parameters: `{"colsample_bytree": 1.0, "learning_rate": 0.03, "max_depth": 6, "min_child_samples": 10, "num_leaves": 31, "reg_alpha": 0.1, "reg_lambda": 5.0, "subsample": 1.0}`

## Fixed-validation comparison

| Model | MAE | RMSE | R² | RMSLE | Within ±20% |
|---|---:|---:|---:|---:|---:|
| Dummy baseline | ₹396,881 | ₹834,402 | -0.0657 | 0.6946 | 29.1% |
| CatBoost untuned | ₹89,656 | ₹167,077 | 0.9573 | 0.1674 | 82.6% |
| CatBoost tuned | ₹88,384 | ₹164,818 | 0.9584 | 0.1666 | 82.6% |
| LightGBM untuned | ₹96,809 | ₹262,283 | 0.8947 | 0.1691 | 81.7% |
| LightGBM tuned | ₹97,847 | ₹344,378 | 0.8185 | 0.1702 | 81.5% |
| XGBoost untuned | ₹99,047 | ₹411,492 | 0.7408 | 0.1728 | 82.4% |

![Tuned validation comparison](figures/23_tuned_validation_comparison.png)

CatBoost tuning reduced validation MAE by ₹1,272 (1.42%). LightGBM tuning increased it by ₹1,038 (1.07%).

![Validation MAE by price band](figures/24_tuned_validation_by_price_band.png)

CatBoost's overall gain is concentrated in the `Above ₹20 lakh` band, where MAE falls from about ₹4.23 lakh to ₹3.97 lakh. It is almost unchanged in the two middle bands. LightGBM's tuned luxury-band errors worsen, which explains its higher overall MAE and RMSE.

## Phase 8 conclusion

**CatBoost tuned** has the lowest fixed-validation MAE at ₹88,384 and is the Phase 9 explanation-and-recommendation candidate. This remains a validation-based development decision, not final test confirmation. The test set stays sealed until the project plan explicitly authorizes one final evaluation.
