Used Car Price Recommendation Engine

An end-to-end machine-learning application that estimates a fair listing price
for used cars in India. The system returns an Indian-rupee point estimate, a
data-driven 80% recommendation range, a confidence label, and the main factors
influencing each prediction.







Project overview

Used-car prices depend on much more than age and kilometres driven. Brand,
model, engine size, power, fuel type, transmission, seller type, mileage, and
seating capacity all influence the expected price. This project combines those
signals in a tuned CatBoost regression model and presents the result through a
validated Streamlit interface.

The application provides:

Estimated fair price in Indian rupees.

Data-driven recommended lower and upper prices.

High, Medium, or Low confidence based on training support, input
distribution, and interval width.

Five local TreeSHAP drivers showing which inputs moved the model's price
signal upward or downward.

Explicit warnings for rare combinations, unusual inputs, luxury cars,
Low-confidence cases, and predictions above ₹20 lakh.

Important: This is a decision-support estimate based on historical
listings. It is not a valuation certificate, guaranteed sale price, or offer
to buy or sell.

Table of contents

Dataset

Workflow

Model development

Final test results

Recommendation range and confidence

Streamlit application

Project structure

Installation

Running the project

Testing and reproducibility

Known limitations

Dataset

The project uses the
CarDekho Used Car Dataset.

Property

Value

Raw rows

15,411

Raw columns

14

Cleaned rows

15,244

Brands

32

Model labels

120

Target

selling_price

Target unit

Indian rupees

The original CSV remains unchanged. Cleaning removes the exported index,
removes 167 duplicate listings, drops the redundant car_name field, and
converts two impossible zero-seat values to missing. Suspicious high-odometer
records are retained and documented instead of being silently altered.

Model inputs

Category

Features

Vehicle identity

brand, model

Usage

vehicle_age, km_driven

Listing details

seller_type

Powertrain

fuel_type, transmission_type

Specifications

mileage, engine, max_power, seats

Workflow

flowchart TD
    A["Raw CarDekho data"] --> B["Validate and clean"]
    B --> C["Fixed 70 / 15 / 15 split"]
    C --> D["Train and compare boosted models"]
    D --> E["Tune and freeze CatBoost"]
    E --> F["Calibrate price range and confidence"]
    F --> G["Package and serve with Streamlit"]

Key safeguards:

Fixed random seed of 42.

Duplicate removal before splitting.

Training-only fitting of imputation and preprocessing.

Same split used for direct model comparisons.

Hyperparameter search restricted to training data.

Validation data used for model selection and interval calibration.

Held-out test set evaluated exactly once after the final model was frozen.

Production artifact hashes checked before inference.

Model development

CatBoost, LightGBM, and XGBoost were trained on
log1p(selling_price). Predictions were converted back to rupees before
business metrics were calculated.

Validation comparison

Model

MAE

RMSE

R²

RMSLE

Median dummy baseline

₹3,96,881

₹8,34,402

-0.0657

0.6946

CatBoost, untuned

₹89,656

₹1,67,077

0.9573

0.1674

LightGBM, untuned

₹96,809

₹2,62,283

0.8947

0.1691

XGBoost, untuned

₹99,047

₹4,11,492

0.7408

0.1728

CatBoost, tuned

₹88,384

₹1,64,818

0.9584

0.1666

The tuned CatBoost model uses depth 6, learning rate 0.08, L2 regularization
5, random strength 0.5, bagging temperature 3, and 128 borders. It stopped at
iteration 958 during the final training-only/validation refit.

CatBoost was selected because it achieved the best validation MAE, RMSE, R²,
and RMSLE while handling categorical variables natively.

Final test results

The tuned CatBoost candidate was frozen before the 2,287-row held-out test set
was accessed. The test set was evaluated once and cannot be reopened by the
production verification command.

Metric

Final test result

MAE

₹97,826

RMSE

₹2,41,826

R²

0.9187

Median absolute error

₹52,316

RMSLE

0.1616

Predictions within ±10%

51.8%

Predictions within ±20%

82.3%

80% interval coverage

81.4%

MAE improvement over dummy baseline

75.6%

Test MAE was 10.7% higher than validation MAE, within the predefined 25%
stability limit. All five predeclared production-selection checks passed.

Recommendation range and confidence

The recommendation range is not an arbitrary percentage around the prediction.
Absolute residuals on the log-price scale were calibrated separately across
four predicted-price bands:

Up to ₹5 lakh.

₹5–10 lakh.

₹10–20 lakh.

Above ₹20 lakh.

The nominal 80% range achieved 81.4% overall coverage on the frozen test set.
Coverage varies by segment, so the application displays targeted cautions
instead of presenting the interval as a guarantee.

Confidence rules

Label

Meaning

High

Common brand-model combination, inputs inside common training ranges, and a narrow interval

Medium

Adequate support with moderately unusual inputs or a wider interval

Low

Rare or unseen category, out-of-distribution numeric value, or very wide interval

Known risk segments are preserved in both the reports and application:

Above-₹20-lakh interval coverage: 69.7%.

Low-confidence interval coverage: 71.4%.

Luxury-brand MAE: ₹4,94,727, compared with ₹67,568 for mainstream brands.

Rare or unseen brand-model MAE: ₹4,25,073 across 31 test cases.

TreeSHAP explanations describe model associations, not causal effects.

Streamlit application

The Streamlit app loads the frozen CatBoost model through
PriceRecommendationEngine and verifies all production artifact hashes.
st.cache_resource keeps one loaded model available across normal interface
reruns.

Application features:

Brand-dependent model dropdown.

All 11 inputs in the exact training feature order.

Training medians as practical numeric defaults.

Common and observed training ranges shown as reference-only help.

Physical-value and schema validation.

Indian currency formatting.

Point estimate, calibrated range, confidence reason, and training support.

Five local SHAP drivers.

Protection against showing a stale result after inputs change.

Permanent limitations and valuation disclaimer.

Start the application:

python -m streamlit run app.py

The local address is normally http://localhost:8501.

Project structure

price-recommendation-engine/
├── app.py
├── CLAUDE.md
├── README.md
├── requirements.txt
├── environment_versions.json
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── production/
│       ├── catboost_price_recommender_v1.cbm
│       ├── production_manifest.json
│       ├── feature_schema.json
│       ├── allowed_inputs.json
│       ├── preprocessing.json
│       ├── interval_calibration.json
│       ├── confidence_profile.json
│       ├── final_test_metrics.json
│       └── MODEL_CARD.md
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── ...
│   └── 11_streamlit_application.ipynb
├── reports/
│   ├── metrics/
│   ├── tables/
│   └── figures/
├── src/
│   ├── app_logic.py
│   ├── predict.py
│   ├── package_production.py
│   └── ...
└── tests/
    ├── test_production_artifacts.py
    ├── test_streamlit_app.py
    └── ...

Detailed technical reports:

Exploratory data analysis

Feature engineering

Model training

Model evaluation

Hyperparameter tuning

Price recommendation and SHAP

Production artifact audit

Streamlit application validation

Production model card

Installation

The project was validated with Python 3.12. A separate Conda environment is
recommended:

conda create -n car-price python=3.12 -y
conda activate car-price
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

If Streamlit reports ModuleNotFoundError: No module named 'catboost', install
the exact model-library version in the active environment:

python -m pip install catboost==1.2.10

Verify that Python and CatBoost come from the same environment:

python -c "import sys, catboost; print(sys.executable); print(catboost.__version__)"

Expected CatBoost version: 1.2.10.

Running the project

Launch the recommendation app

python -m streamlit run app.py

Run automated tests

python -m unittest discover -s tests -v

Verify the frozen production bundle

python -m src.package_production --verify-only

Do not run python -m src.package_production again. The default command is
the guarded one-time final-test evaluator. Only --verify-only is appropriate
after Phase 10.

Reproduce an earlier development phase

Each notebook corresponds to a project phase and imports reusable functions
from src/. Commands for data cleaning, EDA, training, and tuning are
documented in the relevant notebook and report.

The raw dataset is not modified in place.

GitHub and Streamlit Cloud notes

The deployed app needs the files inside models/production/, but it does not
need the raw or cleaned CSV files. Before deployment, confirm that the following
are visible in the GitHub repository:

models/production/catboost_price_recommender_v1.cbm
models/production/production_manifest.json
models/production/feature_schema.json
models/production/allowed_inputs.json
models/production/preprocessing.json
models/production/interval_calibration.json
models/production/confidence_profile.json
models/production/model_metadata.json
models/production/final_test_metrics.json
models/production/MODEL_CARD.md

If the repository's .gitignore excludes models/, explicitly allow or
force-add models/production/. Do not upload the raw dataset merely to make the
app run.

Use app.py as the Streamlit Cloud entry point and select Python 3.12.

Testing and reproducibility

The complete suite contains 116 passing tests covering:

Raw and cleaned data contracts.

Duplicate and invalid-seat handling.

Leak-safe fixed splits.

Feature engineering.

Model loading and prediction.

Unknown-category handling.

Exact inference feature order.

Non-negative finite predictions.

Recommendation-bound ordering.

Production manifest hashes.

Fresh-process prediction parity.

Streamlit controls and dependent dropdowns.

Stale-result protection.

Test-set access guard.

The production model reproduces the saved reference prediction with a maximum
numeric difference of 0.0.

Known limitations

Prices are historical listing values, not confirmed transaction prices.

Listing dates are absent, so inflation and changing market conditions cannot
be learned.

Location is absent, so regional price differences are not modeled.

Variant and trim information may be incomplete.

Service history, accident history, physical condition, insurance, ownership
count, and modifications are unavailable.

Electric vehicles and several luxury or rare models have very limited
representation.

Mileage units differ by fuel type in the source, but the original unit label
is not retained in the numeric field.

A random split estimates performance on similar historical listings, not
necessarily future market conditions.

Always compare the recommendation with a physical inspection and current local
market listings.

Author

Karan Patel

GitHub: patelkaran952-hue

Focus: Data Analytics, Data Science, and Machine Learning

If this project helped you, consider starring the repository.
