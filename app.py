"""Phase 11 Streamlit application for the used-car price recommender."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.app_logic import (
    EXPECTED_FEATURE_ORDER,
    build_input_record,
    driver_rows,
    models_for_brand,
    numeric_widget_spec,
    risk_notices,
    same_input_record,
    validate_app_contract,
)
from src.predict import PriceRecommendationEngine


st.set_page_config(
    page_title="Used Car Fair-Price Recommender",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero-kicker {color: #0f766e; font-size: .82rem; font-weight: 700;
                  letter-spacing: .08em; text-transform: uppercase;}
    .hero-copy {color: #475569; font-size: 1.05rem; max-width: 780px;}
    .result-shell {border: 1px solid #dbe5e1; border-radius: 18px;
                   padding: 1.25rem; background: #f8fbfa; margin-top: .75rem;}
    .confidence-high, .confidence-medium, .confidence-low {
        display: inline-block; border-radius: 999px; padding: .3rem .75rem;
        font-weight: 700; margin-bottom: .35rem;
    }
    .confidence-high {background: #dcfce7; color: #166534;}
    .confidence-medium {background: #fef3c7; color: #92400e;}
    .confidence-low {background: #fee2e2; color: #991b1b;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #dbe5e1;
                                 padding: 1rem; border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> PriceRecommendationEngine:
    """Load and hash-check the frozen model once per Streamlit server."""

    engine = PriceRecommendationEngine()
    validate_app_contract(engine.feature_order, engine.allowed_inputs)
    return engine


def _number_input(
    engine: PriceRecommendationEngine,
    name: str,
    label: str,
    step: float,
    display_format: str,
) -> float:
    spec = numeric_widget_spec(engine.schema["numeric_features"][name])
    return float(
        st.number_input(
            label,
            min_value=float(spec["min_value"]),
            value=float(spec["value"]),
            step=step,
            format=display_format,
            help=str(spec["help"]),
            key=name,
        )
    )


def _show_notice(notice: dict[str, str]) -> None:
    body = f"**{notice['title']}** — {notice['message']}"
    renderer = getattr(st, notice["severity"], st.warning)
    renderer(body)


def _render_recommendation(
    recommendation: dict[str, Any], submitted_input: dict[str, Any]
) -> None:
    st.subheader("Your price recommendation")
    st.markdown('<div class="result-shell">', unsafe_allow_html=True)
    point_column, range_column, support_column = st.columns(3)
    point_column.metric(
        "Estimated fair price", recommendation["recommended_price_display"]
    )
    range_column.metric(
        "Recommended range",
        (
            f"{recommendation['range_lower_display']} – "
            f"{recommendation['range_upper_display']}"
        ),
    )
    support_column.metric(
        "Matching training listings",
        f"{recommendation['brand_model_training_rows']:,}",
    )

    confidence = recommendation["confidence"]
    st.markdown(
        f'<span class="confidence-{confidence.lower()}">{confidence} confidence</span>',
        unsafe_allow_html=True,
    )
    st.caption(recommendation["confidence_reason"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning(
        "**Coverage note** — The recommended range targets 80% empirical coverage "
        "and covered 81.4% of the frozen test cases overall. It is not a guarantee, "
        "and coverage varies across car segments."
    )

    for warning in recommendation["warnings"]:
        st.warning(f"**Input/distribution warning** — {warning}")
    for notice in risk_notices(recommendation, str(submitted_input["brand"])):
        _show_notice(notice)

    st.subheader("Main price drivers")
    st.dataframe(
        pd.DataFrame(driver_rows(recommendation["top_shap_drivers"])),
        hide_index=True,
        width="stretch",
    )
    st.caption(recommendation["explanation_warning"])


try:
    engine = load_engine()
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(
        "The verified production model could not be loaded safely. "
        f"Details: {error}"
    )
    st.stop()

st.markdown('<div class="hero-kicker">India · CatBoost v1.0.0</div>', unsafe_allow_html=True)
st.title("Used Car Fair-Price Recommender")
st.markdown(
    '<p class="hero-copy">Enter a car’s listing specifications to receive a '
    "historical-data estimate, a calibrated range, and the model signals behind "
    "the result.</p>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("1. Choose the car")
    brands = tuple(engine.allowed_inputs["categorical_values"]["brand"])
    default_brand_index = brands.index("Maruti") if "Maruti" in brands else 0
    brand_column, model_column = st.columns(2)
    with brand_column:
        brand = st.selectbox("Brand", brands, index=default_brand_index, key="brand")
    available_models = models_for_brand(engine.allowed_inputs, str(brand))
    with model_column:
        model = st.selectbox("Model", available_models, key="model")
    st.caption(
        "The model list is filtered to combinations observed with the selected brand."
    )

    st.subheader("2. Add usage and specifications")
    age_column, distance_column = st.columns(2)
    with age_column:
        vehicle_age = _number_input(engine, "vehicle_age", "Vehicle age (years)", 1.0, "%.0f")
    with distance_column:
        km_driven = _number_input(engine, "km_driven", "Kilometres driven", 1_000.0, "%.0f")

    seller_column, fuel_column, transmission_column = st.columns(3)
    categories = engine.allowed_inputs["categorical_values"]
    with seller_column:
        seller_type = st.selectbox("Seller type", categories["seller_type"], key="seller_type")
    with fuel_column:
        fuel_type = st.selectbox("Fuel type", categories["fuel_type"], key="fuel_type")
    with transmission_column:
        transmission_type = st.selectbox(
            "Transmission", categories["transmission_type"], key="transmission_type"
        )

    mileage_column, engine_column, power_column, seats_column = st.columns(4)
    with mileage_column:
        mileage = _number_input(
            engine, "mileage", "Mileage (km/l or km/kg)", 0.1, "%.1f"
        )
    with engine_column:
        engine_size = _number_input(engine, "engine", "Engine size (cc)", 10.0, "%.0f")
    with power_column:
        max_power = _number_input(engine, "max_power", "Maximum power (bhp)", 0.5, "%.1f")
    with seats_column:
        seats = _number_input(engine, "seats", "Seats", 1.0, "%.0f")

    current_input = build_input_record(
        {
            "brand": brand,
            "model": model,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": engine_size,
            "max_power": max_power,
            "seats": seats,
        }
    )
    if tuple(current_input) != EXPECTED_FEATURE_ORDER:
        st.error("The input order does not match the frozen model contract.")
        st.stop()

    predict_clicked = st.button(
        "Recommend fair price",
        type="primary",
        width="stretch",
        help="Runs the frozen CatBoost model; it does not retrain or access test rows.",
    )

if predict_clicked:
    try:
        with st.spinner("Checking the inputs and calculating the estimate…"):
            st.session_state["last_recommendation"] = engine.recommend(current_input)
            st.session_state["last_submitted_input"] = dict(current_input)
    except (TypeError, ValueError) as error:
        st.error(f"Please correct the input: {error}")
    except RuntimeError:
        st.error("The model could not produce a safe estimate. Please review the inputs.")

last_recommendation = st.session_state.get("last_recommendation")
last_submitted_input = st.session_state.get("last_submitted_input")
if last_recommendation and last_submitted_input:
    if same_input_record(current_input, last_submitted_input):
        _render_recommendation(last_recommendation, last_submitted_input)
    else:
        st.info("Inputs changed. Click **Recommend fair price** to calculate a new result.")

with st.expander("Model performance and limitations"):
    metric_columns = st.columns(3)
    metric_columns[0].metric("Frozen test MAE", "₹97,826")
    metric_columns[1].metric("Frozen test R²", "0.919")
    metric_columns[2].metric("80% range coverage", "81.4%")
    st.markdown(
        "The estimate is based on historical listing prices, not confirmed sale prices. "
        "The data has no listing date or location and does not capture service history, "
        "accidents, ownership count, cosmetic condition, insurance, modifications, full "
        "variant details, or current local demand. Electric and some luxury cars have "
        "limited representation."
    )

st.divider()
st.caption(
    "Disclaimer: This is a decision-support estimate from historical CarDekho listings, "
    "not a valuation certificate, guaranteed market price, or offer to buy or sell. "
    "Compare the result with inspections and current local listings before deciding."
)
