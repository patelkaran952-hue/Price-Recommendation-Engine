"""Phase 11 tests for UI input preparation and headless Streamlit behavior."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.app_logic import (
    EXPECTED_FEATURE_ORDER,
    LUXURY_BRANDS,
    build_input_record,
    driver_rows,
    models_for_brand,
    numeric_widget_spec,
    risk_notices,
    same_input_record,
    validate_app_contract,
)
from src.config import FINAL_TEST_ACCESS_PATH, PROJECT_ROOT
from src.predict import PriceRecommendationEngine


APP_PATH = PROJECT_ROOT / "app.py"
APP_REPORT_PATH = PROJECT_ROOT / "reports" / "streamlit_application_report.md"
APP_SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "streamlit_application_summary.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class StreamlitAppTests(unittest.TestCase):
    """Exercise the app without a browser or any model retraining."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = PriceRecommendationEngine()
        cls.access_hash_before = _sha256(FINAL_TEST_ACCESS_PATH)

    def _new_app(self) -> AppTest:
        return AppTest.from_file(str(APP_PATH), default_timeout=120).run(timeout=120)

    def test_app_contract_matches_frozen_feature_order(self) -> None:
        validate_app_contract(self.engine.feature_order, self.engine.allowed_inputs)
        self.assertEqual(tuple(self.engine.feature_order), EXPECTED_FEATURE_ORDER)

    def test_brand_filters_models_and_rejects_missing_mapping(self) -> None:
        maruti_models = models_for_brand(self.engine.allowed_inputs, "Maruti")
        self.assertIn("Swift", maruti_models)
        self.assertNotIn("City", maruti_models)
        with self.assertRaisesRegex(ValueError, "No model mapping"):
            models_for_brand(self.engine.allowed_inputs, "Unknown Brand")

    def test_numeric_defaults_are_training_medians_with_reference_help(self) -> None:
        for name, rule in self.engine.schema["numeric_features"].items():
            spec = numeric_widget_spec(rule)
            self.assertGreaterEqual(float(spec["value"]), float(spec["min_value"]))
            self.assertIn("Training-data reference only", str(spec["help"]))
            self.assertIn("not universal vehicle limits", str(spec["help"]))
            if not rule["minimum_inclusive"]:
                self.assertGreater(float(spec["min_value"]), 0)

    def test_ui_record_preserves_order_and_rejects_schema_drift(self) -> None:
        sample = {
            "brand": "Maruti",
            "model": "Swift",
            "vehicle_age": 6.0,
            "km_driven": 50_000.0,
            "seller_type": "Individual",
            "fuel_type": "Petrol",
            "transmission_type": "Manual",
            "mileage": 19.6,
            "engine": 1_248.0,
            "max_power": 88.5,
            "seats": 5.0,
        }
        prepared = build_input_record(dict(reversed(list(sample.items()))))
        self.assertEqual(tuple(prepared), EXPECTED_FEATURE_ORDER)
        normalized, _, _ = self.engine.validate_input(prepared)
        self.assertEqual(tuple(normalized), EXPECTED_FEATURE_ORDER)
        with self.assertRaisesRegex(ValueError, "Missing"):
            build_input_record({key: value for key, value in sample.items() if key != "seats"})
        with self.assertRaisesRegex(ValueError, "extra"):
            build_input_record(dict(sample, selling_price=500_000))

    def test_driver_table_and_risk_notices_are_user_facing(self) -> None:
        rows = driver_rows(
            [
                {
                    "feature": "km_driven",
                    "value": 50_000,
                    "direction": "Downward",
                    "approximate_price_signal_percentage": 12.345,
                }
            ]
        )
        self.assertEqual(rows[0]["Driver"], "Kilometres driven")
        self.assertEqual(rows[0]["Entered value"], "50,000 km")
        self.assertEqual(rows[0]["Approx. signal"], "12.3%")

        recommendation = {
            "confidence": "Low",
            "brand_model_training_rows": 3,
            "predicted_price_band": "Above ₹20 lakh",
        }
        notices = risk_notices(recommendation, "BMW")
        titles = {notice["title"] for notice in notices}
        self.assertEqual(
            titles,
            {
                "Low-confidence estimate",
                "Rare brand-model combination",
                "Luxury-car caution",
                "High-price coverage caution",
            },
        )
        self.assertIn("BMW", LUXURY_BRANDS)

    def test_headless_app_loads_all_controls_and_disclaimer(self) -> None:
        app = self._new_app()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Used Car Fair-Price Recommender")
        self.assertEqual(len(app.selectbox), 5)
        self.assertEqual(len(app.number_input), 6)
        self.assertEqual(app.button[0].label, "Recommend fair price")
        captions = " ".join(str(item.value) for item in app.caption)
        self.assertIn("not a valuation certificate", captions)

    def test_headless_brand_change_updates_model_options(self) -> None:
        app = self._new_app()
        app.selectbox[0].select("Honda").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, "Honda")
        self.assertIn("City", app.selectbox[1].options)
        self.assertNotIn("Swift", app.selectbox[1].options)

    def test_headless_prediction_shows_price_range_confidence_and_drivers(self) -> None:
        app = self._new_app()
        app.button[0].click().run(timeout=120)
        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertTrue(str(metrics["Estimated fair price"]).startswith("₹"))
        self.assertIn("₹", str(metrics["Recommended range"]))
        self.assertIn("Matching training listings", metrics)
        self.assertEqual(len(app.dataframe), 1)
        warnings = " ".join(str(item.value) for item in app.warning)
        self.assertIn("Coverage note", warnings)

    def test_changed_inputs_do_not_keep_a_stale_recommendation_visible(self) -> None:
        app = self._new_app()
        app.button[0].click().run(timeout=120)
        app.number_input[0].set_value(7.0).run(timeout=120)
        labels = [metric.label for metric in app.metric]
        self.assertNotIn("Estimated fair price", labels)
        info_messages = " ".join(str(item.value) for item in app.info)
        self.assertIn("Inputs changed", info_messages)

    def test_app_uses_cached_inference_without_training_or_test_sources(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("@st.cache_resource", source)
        self.assertIn("PriceRecommendationEngine", source)
        for forbidden in (
            "src.train_models",
            "src.tune_models",
            "src.package_production",
            "cardekho_cleaned.csv",
            "phase5_split_assignment.csv",
            "phase10_final_test_predictions.csv",
        ):
            self.assertNotIn(forbidden, source)

    def test_final_test_access_record_is_unchanged(self) -> None:
        self.assertEqual(_sha256(FINAL_TEST_ACCESS_PATH), self.access_hash_before)
        self.assertEqual(
            self.access_hash_before,
            "66ea4e00be10ad7ff2234235fef845002bdd2087d6f9e61c0ce9397c74ebe049",
        )

    def test_record_comparison_covers_every_feature(self) -> None:
        original = {name: position for position, name in enumerate(EXPECTED_FEATURE_ORDER)}
        self.assertTrue(same_input_record(original, dict(original)))
        changed = dict(original, model="different")
        self.assertFalse(same_input_record(original, changed))

    def test_phase11_outputs_and_summary_are_complete(self) -> None:
        for path in (
            APP_PATH,
            PROJECT_ROOT / "src" / "app_logic.py",
            PROJECT_ROOT / ".streamlit" / "config.toml",
            PROJECT_ROOT / "notebooks" / "11_streamlit_application.ipynb",
            APP_REPORT_PATH,
            APP_SUMMARY_PATH,
        ):
            self.assertTrue(path.exists(), path)
        summary = json.loads(APP_SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["input_contract"]["feature_count"], 11)
        self.assertEqual(summary["application"]["sha256"], _sha256(APP_PATH))
        self.assertFalse(summary["application"]["test_rows_loaded_by_app"])


if __name__ == "__main__":
    unittest.main()
