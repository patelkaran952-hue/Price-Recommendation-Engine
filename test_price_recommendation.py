"""Automated checks for Phase 9 explanations and calibrated price ranges."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.config import (
    CATBOOST_TUNED_MODEL_PATH,
    FIGURES_DIR,
    PRICE_RECOMMENDATION_METADATA_PATH,
    PRICE_RECOMMENDATION_REPORT_PATH,
    PRICE_RECOMMENDATION_SUMMARY_PATH,
    PROCESSED_DATA_PATH,
    SPLIT_ASSIGNMENT_PATH,
    TABLES_DIR,
)
from src.explain_recommend import (
    GLOBAL_IMPORTANCE_FIGURE_FILENAME,
    GLOBAL_IMPORTANCE_FILENAME,
    INDIVIDUAL_EXPLANATIONS_FIGURE_FILENAME,
    INDIVIDUAL_EXPLANATIONS_FILENAME,
    INTERVAL_ASSIGNMENT_FILENAME,
    INTERVAL_CALIBRATION_FILENAME,
    INTERVAL_COVERAGE_FIGURE_FILENAME,
    INTERVAL_COVERAGE_FILENAME,
    NOMINAL_COVERAGE,
    RECOMMENDATIONS_FILENAME,
    SHAP_SUMMARY_FIGURE_FILENAME,
    SHAP_VALUES_FILENAME,
    recommend_price,
    split_interval_roles,
)
from src.features import BASE_FEATURE_COLUMNS, PRICE_BAND_LABELS, load_cleaned_features
from src.train_models import load_fixed_split, split_training_and_validation
from src.validate_data import file_sha256


EXPECTED_CLEANED_SHA256 = (
    "7a9742b34ebf4017cbd97dec129f0040fc661e582ab749e4f04365b7612b42a0"
)
EXPECTED_SPLIT_SHA256 = (
    "601f9ab1ec8eb30c8d30fe1c1efa6f78d6cc88df85721f716dc9963cfb1739f7"
)
EXPECTED_TUNED_CATBOOST_SHA256 = (
    "3b08c06c643129fb0995d7e9ccf4609d8303adf14722e488f54a2fd7ed938b1f"
)


class PriceRecommendationTests(unittest.TestCase):
    """Verify TreeSHAP, interval calibration, confidence, and split protection."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_cleaned_features(PROCESSED_DATA_PATH)
        cls.assignment = load_fixed_split(SPLIT_ASSIGNMENT_PATH)
        (
            cls.x_train,
            cls.x_validation,
            _,
            cls.y_validation,
            _,
            cls.validation_indices,
        ) = split_training_and_validation(cls.dataframe, cls.assignment)
        cls.summary = json.loads(
            PRICE_RECOMMENDATION_SUMMARY_PATH.read_text(encoding="utf-8")
        )
        cls.metadata = json.loads(
            PRICE_RECOMMENDATION_METADATA_PATH.read_text(encoding="utf-8")
        )
        cls.importance = pd.read_csv(TABLES_DIR / GLOBAL_IMPORTANCE_FILENAME)
        cls.shap_values = pd.read_csv(TABLES_DIR / SHAP_VALUES_FILENAME)
        cls.roles = pd.read_csv(TABLES_DIR / INTERVAL_ASSIGNMENT_FILENAME)
        cls.calibration = pd.read_csv(TABLES_DIR / INTERVAL_CALIBRATION_FILENAME)
        cls.coverage = pd.read_csv(TABLES_DIR / INTERVAL_COVERAGE_FILENAME)
        cls.recommendations = pd.read_csv(TABLES_DIR / RECOMMENDATIONS_FILENAME)
        cls.explanations = pd.read_csv(
            TABLES_DIR / INDIVIDUAL_EXPLANATIONS_FILENAME
        )

    def test_source_split_and_tuned_model_are_unchanged(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(file_sha256(SPLIT_ASSIGNMENT_PATH), EXPECTED_SPLIT_SHA256)
        self.assertEqual(
            file_sha256(CATBOOST_TUNED_MODEL_PATH), EXPECTED_TUNED_CATBOOST_SHA256
        )
        self.assertEqual(
            self.summary["model"]["sha256"], EXPECTED_TUNED_CATBOOST_SHA256
        )
        self.assertFalse(self.summary["model"]["retrained_in_phase9"])

    def test_interval_roles_are_reproducible_validation_only(self) -> None:
        calibration_positions, evaluation_positions, reproduced = (
            split_interval_roles(self.assignment, self.validation_indices)
        )
        self.assertEqual(len(calibration_positions), 1_143)
        self.assertEqual(len(evaluation_positions), 1_144)
        pd.testing.assert_frame_equal(reproduced, self.roles, check_dtype=False)
        validation_indices = set(self.validation_indices.astype(int))
        test_indices = set(
            self.assignment.loc[
                self.assignment["split"].eq("test"), "row_index"
            ].astype(int)
        )
        output_indices = set(self.roles["row_index"].astype(int))
        self.assertEqual(output_indices, validation_indices)
        self.assertTrue(output_indices.isdisjoint(test_indices))

    def test_interval_is_data_driven_and_inference_safe(self) -> None:
        self.assertEqual(self.metadata["interval"]["nominal_coverage"], 0.8)
        self.assertIn("log-residual", self.metadata["interval"]["method"])
        self.assertEqual(
            set(self.metadata["interval"]["band_log_quantiles"]),
            set(PRICE_BAND_LABELS),
        )
        self.assertEqual(len(self.calibration), 5)
        self.assertTrue(self.calibration["absolute_log_residual_quantile"].gt(0).all())
        self.assertTrue(self.calibration["lower_price_multiplier"].lt(1).all())
        self.assertTrue(self.calibration["upper_price_multiplier"].gt(1).all())
        self.assertEqual(
            self.summary["interval"]["band_assignment_at_inference"],
            "Point prediction only; actual price is not used",
        )

    def test_empirical_interval_coverage_is_close_to_nominal(self) -> None:
        overall = self.coverage.loc[self.coverage["group_type"].eq("Overall")].iloc[0]
        self.assertEqual(int(overall["rows"]), 1_144)
        self.assertEqual(float(overall["nominal_coverage"]), NOMINAL_COVERAGE)
        self.assertAlmostEqual(
            float(overall["empirical_coverage"]), 0.8382867133, places=9
        )
        self.assertLessEqual(
            abs(float(overall["empirical_coverage"]) - NOMINAL_COVERAGE), 0.05
        )
        bands = self.coverage.loc[
            self.coverage["group_type"].eq("Predicted price band")
        ]
        self.assertEqual(bands["rows"].astype(int).tolist(), [488, 474, 123, 59])

    def test_recommendations_are_ordered_and_validation_only(self) -> None:
        self.assertEqual(len(self.recommendations), 2_287)
        self.assertEqual(
            self.recommendations["row_index"].astype(int).tolist(),
            self.validation_indices.astype(int).tolist(),
        )
        self.assertTrue(
            self.recommendations["range_lower"]
            .le(self.recommendations["recommended_price"])
            .all()
        )
        self.assertTrue(
            self.recommendations["recommended_price"]
            .le(self.recommendations["range_upper"])
            .all()
        )
        self.assertTrue(self.recommendations["range_lower"].ge(0).all())
        self.assertEqual(
            set(self.recommendations["confidence"]), {"High", "Medium", "Low"}
        )

    def test_confidence_counts_and_low_confidence_warning_are_honest(self) -> None:
        self.assertEqual(
            self.summary["confidence"]["evaluation_counts"],
            {"High": 359, "Medium": 667, "Low": 118},
        )
        confidence_coverage = self.summary["confidence"]["evaluation_coverage"]
        self.assertAlmostEqual(confidence_coverage["High"], 0.824512535, places=9)
        self.assertAlmostEqual(confidence_coverage["Low"], 0.745762712, places=9)
        self.assertLess(confidence_coverage["Low"], NOMINAL_COVERAGE)
        report = PRICE_RECOMMENDATION_REPORT_PATH.read_text(encoding="utf-8")
        self.assertIn("Low-confidence cases achieve only 74.6%", report)

    def test_tree_shap_is_additive_and_top_features_are_stable(self) -> None:
        self.assertEqual(len(self.shap_values), 2_287)
        shap_columns = [
            column for column in self.shap_values if column.startswith("shap_")
        ]
        self.assertEqual(len(shap_columns), len(BASE_FEATURE_COLUMNS))
        reconstructed = self.shap_values["expected_log_prediction"] + self.shap_values[
            shap_columns
        ].sum(axis=1)
        np.testing.assert_allclose(
            reconstructed,
            self.shap_values["model_log_prediction"],
            rtol=0,
            atol=1e-10,
        )
        self.assertLess(
            self.summary["explanation"]["maximum_additivity_error"], 1e-8
        )
        self.assertEqual(
            self.importance.head(5)["feature"].tolist(),
            ["vehicle_age", "max_power", "engine", "model", "transmission_type"],
        )

    def test_individual_explanations_cover_four_price_bands(self) -> None:
        self.assertEqual(len(self.explanations), 4 * 6)
        self.assertEqual(
            self.explanations.groupby("case").size().unique().tolist(), [6]
        )
        self.assertEqual(
            set(self.explanations["predicted_price_band"]), set(PRICE_BAND_LABELS)
        )
        self.assertTrue(
            self.explanations["case_summary"]
            .str.contains("not causal effects", case=False)
            .all()
        )

    def test_recommend_price_handles_unseen_categories(self) -> None:
        sample = self.x_validation.iloc[0].to_dict()
        sample["brand"] = "Unknown Brand"
        sample["model"] = "Unknown Model"
        recommendation = recommend_price(sample)
        self.assertGreaterEqual(recommendation["recommended_price"], 0)
        self.assertLessEqual(
            recommendation["range_lower"], recommendation["recommended_price"]
        )
        self.assertGreaterEqual(
            recommendation["range_upper"], recommendation["recommended_price"]
        )
        self.assertEqual(recommendation["confidence"], "Low")
        self.assertIn("unseen category", recommendation["confidence_reason"])
        self.assertEqual(len(recommendation["top_shap_drivers"]), 5)

    def test_test_set_remains_completely_sealed(self) -> None:
        test = self.summary["test_set"]
        self.assertEqual(self.summary["split"]["test_rows_used"], 0)
        self.assertFalse(test["evaluated"])
        self.assertFalse(test["transformed"])
        self.assertFalse(test["explained"])
        self.assertFalse(test["predictions_generated"])
        self.assertFalse(test["metrics_computed"])
        self.assertFalse(self.metadata["test_set_evaluated"])
        self.assertFalse(
            self.summary["phase10_recommendation"]["final_model_selected"]
        )

    def test_all_phase9_outputs_exist(self) -> None:
        expected_paths = (
            PRICE_RECOMMENDATION_SUMMARY_PATH,
            PRICE_RECOMMENDATION_REPORT_PATH,
            PRICE_RECOMMENDATION_METADATA_PATH,
            TABLES_DIR / GLOBAL_IMPORTANCE_FILENAME,
            TABLES_DIR / SHAP_VALUES_FILENAME,
            TABLES_DIR / INTERVAL_ASSIGNMENT_FILENAME,
            TABLES_DIR / INTERVAL_CALIBRATION_FILENAME,
            TABLES_DIR / INTERVAL_COVERAGE_FILENAME,
            TABLES_DIR / RECOMMENDATIONS_FILENAME,
            TABLES_DIR / INDIVIDUAL_EXPLANATIONS_FILENAME,
            FIGURES_DIR / GLOBAL_IMPORTANCE_FIGURE_FILENAME,
            FIGURES_DIR / SHAP_SUMMARY_FIGURE_FILENAME,
            FIGURES_DIR / INDIVIDUAL_EXPLANATIONS_FIGURE_FILENAME,
            FIGURES_DIR / INTERVAL_COVERAGE_FIGURE_FILENAME,
        )
        for path in expected_paths:
            self.assertTrue(path.exists(), path.name)
            self.assertGreater(path.stat().st_size, 10, path.name)


if __name__ == "__main__":
    unittest.main()
