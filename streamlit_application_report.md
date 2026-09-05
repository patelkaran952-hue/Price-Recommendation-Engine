# Phase 11 — Streamlit Application Report

## Outcome

Phase 11 is complete. The project now has a validated Streamlit application at
`app.py` that calls the frozen Phase 10 `PriceRecommendationEngine`. It does not
train a model, read the cleaned modeling table, load split assignments, or score
the held-out test set.

## Application contract

The application prepares exactly one ordered record with the 11 frozen features:

| App control | Production feature | UI protection |
|---|---|---|
| Brand | `brand` | Saved allowed values only |
| Model | `model` | Filtered by selected brand |
| Vehicle age | `vehicle_age` | Non-negative; training-range help |
| Kilometres driven | `km_driven` | Non-negative; training-range help |
| Seller type | `seller_type` | Saved allowed values only |
| Fuel type | `fuel_type` | Saved allowed values only |
| Transmission | `transmission_type` | Saved allowed values only |
| Mileage | `mileage` | Non-negative; training-range help |
| Engine size | `engine` | Must be greater than zero |
| Maximum power | `max_power` | Must be greater than zero |
| Seats | `seats` | Must be greater than zero |

`src/app_logic.py` validates the UI contract against the saved artifact schema,
constructs the dictionary in the exact feature order, formats SHAP drivers, and
creates targeted risk notices. The production engine performs the final input
validation again before inference. A missing, extra, invalid, or impossible
brand-model input therefore cannot silently reach CatBoost.

Numeric controls start at training medians. Their help text shows both common
and observed training ranges and explicitly states that these are references,
not universal vehicle limits. There is no artificial maximum: an unusual but
physically valid value can be entered, then the production engine warns that it
is outside the training distribution and lowers confidence when appropriate.

## Result design

Each successful recommendation shows:

- Indian-rupee fair-price estimate.
- Data-driven nominal 80% calibrated range.
- Matching brand-model training support.
- High, Medium, or Low confidence and its reason.
- Distribution warnings returned by the production engine.
- Five strongest local TreeSHAP drivers with direction and approximate signal.
- A permanent explanation that SHAP describes model associations, not causes.
- A clear valuation and missing-data disclaimer.

The coverage warning is always visible with a result. Additional prominent
cautions appear for Low-confidence, rare, luxury, and above-₹20-lakh cases. They
preserve the Phase 10 findings: 71.4% Low-confidence range coverage, 69.7%
above-₹20-lakh coverage, and substantially larger errors for luxury and rare
brand-model segments.

The last submitted record is stored with its recommendation. If any widget is
changed, the old estimate is hidden and the user is asked to click the button
again. This prevents a stale result from appearing to belong to new inputs.

## Runtime choices

The application pins Streamlit 1.63.0, which is compatible with the project's
Python 3.12 environment. `st.cache_resource` loads and hash-verifies the model
once per server process, avoiding repeated disk loading during widget reruns.
The current official references are:

- <https://pypi.org/project/streamlit/>
- <https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource>
- <https://docs.streamlit.io/develop/api-reference/app-testing>

The theme and privacy-safe local defaults are stored in
`.streamlit/config.toml`. No secret values are required.

## Verification evidence

| Check | Result |
|---|---:|
| Phase 11-specific tests | 13 passed |
| Complete project suite | 116 passed |
| Live Streamlit health endpoint | HTTP 200 |
| Headless prediction | Passed |
| Dependent brand-model control | Passed |
| Stale-result protection | Passed |
| Notebook code cells | 4 executed successfully |
| Production verify-only test rows accessed | 0 |
| Fresh-process prediction difference | 0.0 |
| Production bundle integrity | Passed |
| Final-test access hash changed | No |

Protected Phase 10 hashes remain:

| Artifact | SHA-256 |
|---|---|
| Final-test access record | `66ea4e00be10ad7ff2234235fef845002bdd2087d6f9e61c0ce9397c74ebe049` |
| Production manifest | `5ae4f4e88defa2bfa1588712f200f0a4b6df200d053e8adac2ffdcdbee523dc1` |
| Production bundle | `6d0da17092ba61d1edb881176f69dfccb45b1c07e2b8baa71fee9aa0261c9a8d` |
| Tuned and production CatBoost model | `3b08c06c643129fb0995d7e9ccf4609d8303adf14722e488f54a2fd7ed938b1f` |

## Run locally

From the project root after installing `requirements.txt`:

```bash
python -m streamlit run app.py
```

Streamlit normally opens `http://localhost:8501`. The application is suitable
for the documented Windows, Anaconda, VS Code, and 8 GB RAM workflow because it
loads one compact model, uses one-row inference, avoids retraining, and does not
copy the full dataset.

## Remaining limitations

The interface cannot correct limitations in the source data. It estimates
historical listing prices rather than confirmed transactions and lacks listing
date, location, accident and service history, owner count, insurance, cosmetic
condition, modifications, complete trim detail, and current local demand.
Electric and several luxury or rare categories have too little support for
strong reliability claims. The range and confidence label reduce the risk of
overinterpretation but do not remove it.

## Conclusion

Phase 11 satisfies the application requirements in `CLAUDE.md`: clean dependent
inputs, validated preparation, fair-price and range output, confidence and
coverage warnings, local explanations, and an honest disclaimer. Phase 12 can
now focus on final documentation, screenshots, repository review, and portfolio
delivery without changing the frozen model or final-test result.
