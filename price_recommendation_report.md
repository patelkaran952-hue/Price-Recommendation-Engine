# Phase 9 Explanation and Price Recommendation

## Scope

Phase 9 reloaded the saved tuned CatBoost model from Phase 8. It did not retrain or modify the model. The fixed 2,287 test rows remain sealed: they were not transformed, explained, predicted, or evaluated.

## What drives the CatBoost prediction?

CatBoost's built-in exact TreeSHAP implementation explains all 2,287 validation predictions. SHAP values are additive on the model's `log1p(price)` target. Their maximum reconstruction error is 1.07e-14, confirming that the feature contributions reproduce the model output numerically.

The five strongest global features by mean absolute SHAP are: **vehicle_age, max_power, engine, model, transmission_type**.

![Global feature importance](figures/25_global_feature_importance.png)

![TreeSHAP summary](figures/26_shap_summary.png)

SHAP describes what the model associated with higher or lower prices. It does not prove that changing a feature would causally change a car's value.

## Data-driven 80% price range

The fixed validation rows were deterministically divided into 1,143 calibration rows and 1,144 separate interval-evaluation rows, stratified by the established actual-price bands.

The interval is not an arbitrary ±10%. Absolute errors in log-price space were calibrated separately for inference-safe predicted-price bands. For a new car, the point prediction selects its band; the actual price is never needed. The calibrated log error expands below and above the recommendation multiplicatively.

On the separate interval-evaluation half, the nominal 80% range covered **83.8%** of actual prices. The median interval width was **₹2.07 L**.

![Interval coverage](figures/28_interval_coverage.png)

Coverage above 80% is conservative overall, but segment coverage varies. In particular, rare cars remain harder to price reliably. This is a development-stage interval check because the validation split supported earlier model development; final confirmation still requires the untouched test set when the project plan authorizes it.

## Confidence labels

- **High:** common brand-model support, inputs inside common training ranges, and a narrow calibrated interval.
- **Medium:** adequate support but an unusual input or moderate interval width.
- **Low:** unseen or rare brand-model support, a numeric input outside the training range, or a very wide interval.

Among the 1,144 interval-evaluation rows, 359 are High confidence, 667 are Medium, and 118 are Low.

Low-confidence cases achieve only 74.6% empirical coverage, versus 82.5% for High-confidence cases. That shortfall is useful: the label identifies the segment where the range should not be treated as dependable without more comparable listings or human review.

![Individual prediction explanations](figures/27_individual_shap_explanations.png)

The individual explanation table turns the strongest positive and negative SHAP contributions into plain language. Percentages describe approximate multiplicative associations on the log-price scale, not causal effects.

## Phase 9 conclusion

Tuned CatBoost now produces four connected outputs: a point recommendation, a calibrated 80% price range, a support-aware confidence label, and local TreeSHAP drivers. It remains the Phase 10 candidate rather than a finally confirmed production model. The test set is still untouched.
