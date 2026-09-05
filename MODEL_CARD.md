# Used Car Price Recommender v1.0.0

## Intended use

Estimate an Indian used-car listing price from 11 vehicle and listing fields.
The output includes a point recommendation, an empirical 80% range, a
support-aware confidence label, warnings, and associative TreeSHAP drivers.

## Final model

- CatBoostRegressor trained on `log1p(selling_price)`
- Frozen native artifact: `catboost_price_recommender_v1.cbm`
- Training rows: 10,670
- Random seed: 42
- Final-test rows: 2,287, evaluated exactly once
- Final-test MAE: ₹97,826
- Final-test RMSE: ₹2,41,826
- Final-test R²: 0.9187
- Final-test RMSLE: 0.1616
- Final-test interval coverage: 81.4% for an 80% target

## Segment warnings from the final test

- Luxury-brand MAE is ₹4,94,727
  across 162 rows, versus
  ₹67,568 for
  mainstream brands.
- Rare or unseen brand-model MAE is
  ₹4,25,073 across
  only 31 rows.
- The range covers 69.7%
  of predictions above ₹20 lakh and
  71.4% of Low-confidence
  cases, both below the 80% target.

## Input and validation contract

All fields listed in `feature_schema.json` are required in exact order when
converted to a model table. Negative numeric values are rejected. Zero values
for engine, maximum power, and seats are rejected. Inputs outside training
ranges produce warnings rather than artificial hard limits. Known but invalid
brand-model combinations are rejected. Unseen categories remain scoreable but
force Low confidence.

## Important limitations

- The source is historical listing data, not completed transactions; the model
  estimates an asking-price pattern, not a guaranteed sale price.
- It does not know live market conditions, location, trim, accident history,
  service history, ownership count, tyre condition, or cosmetic condition.
- Rare luxury, electric, LPG, and unusual brand-model cases have limited data.
- Error in rupees increases for expensive cars.
- The interval is empirically calibrated and does not guarantee coverage for an
  individual vehicle or under distribution shift.
- SHAP describes learned associations, not causal effects.

## Safe use

Treat the result as one reference point alongside a physical inspection,
service records, regional listings, and professional valuation. Do not use this
model alone for lending, insurance, or legal decisions.
