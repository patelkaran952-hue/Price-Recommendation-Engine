# Phase 10 — Production artifacts and final test

## Outcome

Tuned CatBoost is now the frozen production model. The exact native model was
copied before the untouched test rows were selected. The test split was then
evaluated once, the one-time access was recorded, and a versioned inference
bundle was reloaded in a fresh Python process.

| Metric | Validation | Final test |
|---|---:|---:|
| MAE | ₹88,384 | ₹97,826 |
| RMSE | ₹1,64,818 | ₹2,41,826 |
| R² | 0.9584 | 0.9187 |
| Median absolute error | ₹50,722 | ₹52,316 |
| RMSLE | 0.1666 | 0.1616 |
| Within ±20% | 82.6% | 82.3% |

The final-test dummy median baseline MAE is
₹4,00,255. CatBoost reduces that error by
75.6%.
Validation-to-test MAE changed by
10.7% and all predefined
selection checks passed: **True**.

## Final-test performance by actual price band

| Price band | Rows | MAE | RMSE | R² | Within ±20% |
|---|---:|---:|---:|---:|---:|
| Up to ₹5 lakh | 984 | ₹45,560 | ₹62,985 | 0.6495 | 77.5% |
| ₹5–10 lakh | 931 | ₹72,105 | ₹94,105 | 0.5059 | 88.0% |
| ₹10–20 lakh | 256 | ₹1,57,573 | ₹2,04,151 | 0.4474 | 84.0% |
| Above ₹20 lakh | 116 | ₹6,15,759 | ₹9,77,884 | 0.6812 | 73.3% |

Rupee error remains largest above ₹20 lakh. That segment and rare brand-model
rows should be interpreted with the confidence label and range, not only the
point estimate.

## Calibrated interval check

| Segment | Rows | Coverage | Median range width |
|---|---:|---:|---:|
| All final-test rows | 2,287 | 81.4% | ₹2,05,348 |
| Up to ₹5 lakh | 950 | 82.4% | ₹1,69,080 |
| ₹5–10 lakh | 951 | 80.0% | ₹2,15,378 |
| ₹10–20 lakh | 267 | 88.0% | ₹5,39,817 |
| Above ₹20 lakh | 119 | 69.7% | ₹12,50,467 |
| High | 707 | 81.3% | ₹2,27,151 |
| Medium | 1,339 | 83.3% | ₹1,97,619 |
| Low | 241 | 71.4% | ₹1,07,532 |

The target is 80% empirical coverage. Segment results are diagnostic; small
luxury and Low-confidence groups have more sampling uncertainty. In particular,
coverage is only **69.7%**
above ₹20 lakh and **71.4%**
for Low-confidence cases. The app must not present those ranges as equally
reliable as the overall 81.4% result.

## Risk and error audit

- Luxury-brand MAE: ₹4,94,727
  across 162 rows, versus
  ₹67,568 for
  mainstream brands.
- Rare or unseen brand-model MAE:
  ₹4,25,073 across
  31 rows.
- Largest absolute error: Porsche
  Cayenne,
  ₹50,36,323.

These findings do not reverse final model selection because the frozen model
beats the baseline and passes the predefined overall stability checks. They do
define the warnings that Phase 11 must expose.

![Validation and final-test comparison](figures/29_final_validation_test_comparison.png)

![Final-test interval coverage](figures/30_final_test_interval_coverage.png)

## Saved production contract

`models/production/` contains the native CatBoost model, ordered schema,
allowed categorical inputs and brand-model mapping, preprocessing values,
interval calibration, confidence support profile, final metrics, metadata,
model card, verification result, and hash manifest. The portable archive is
`models/used_car_price_recommender_v1.zip`.

For normal inference, import `PriceRecommendationEngine` from `src.predict`.
For integrity checks after this one-time test, run:

```bash
python -m src.package_production --verify-only
```

## Final caution

This is an offline historical listing-price model. It does not incorporate live
market changes or vehicle condition details absent from the dataset. Phase 11
should expose every warning and confidence reason in the app.
