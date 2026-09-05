"""Automated checks for leakage-safe Phase 4 feature engineering."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_REPORT_PATH,
    FEATURE_SUMMARY_PATH,
    FIGURES_DIR,
    HOLDOUT_ASSIGNMENT_PATH,
    PROCESSED_DATA_PATH,
    TABLES_DIR,
)
from src.features import (
    CANDIDATE_FEATURES,
    CV_RESULTS_FILENAME,
    FEATURE_FIGURE_FILENAME,
    FOLD_RESULTS_FILENAME,
    LUXURY_BRANDS,
    create_holdout_assignment,
    engineer_features,
    load_cleaned_features,
)
from src.validate_data import file_sha256


EXPECTED_CLEANED_SHA256 = (
    "7a9742b34ebf4017cbd97dec129f0040fc661e582ab749e4f04365b7612b42a0"
)


class FeatureEngineeringTests(unittest.TestCase):
    """Verify feature formulas, split protection, and saved experiment results."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_cleaned_features(PROCESSED_DATA_PATH)
        cls.summary = json.loads(FEATURE_SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.saved_assignment = pd.read_csv(HOLDOUT_ASSIGNMENT_PATH)
        cls.cv_results = pd.read_csv(TABLES_DIR / CV_RESULTS_FILENAME)
        cls.fold_results = pd.read_csv(TABLES_DIR / FOLD_RESULTS_FILENAME)

    def test_cleaned_source_is_unchanged(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(self.summary["source"]["sha256"], EXPECTED_CLEANED_SHA256)

    def test_candidate_feature_formulas(self) -> None:
        example = pd.DataFrame(
            {
                "brand": ["Audi", "Maruti"],
                "transmission_type": ["Automatic", "Manual"],
                "vehicle_age": [0, 5],
                "km_driven": [10_000, 50_000],
                "engine": [1_000, 1_200],
                "max_power": [100.0, 90.0],
                "seats": [5.0, np.nan],
            }
        )
        engineered = engineer_features(example)

        self.assertEqual(engineered.loc[0, "km_per_year"], 10_000)
        self.assertAlmostEqual(
            engineered.loc[0, "log_km_driven"], np.log1p(10_000)
        )
        self.assertEqual(engineered.loc[0, "power_per_cc"], 0.1)
        self.assertEqual(engineered.loc[0, "engine_per_seat"], 200)
        self.assertEqual(engineered.loc[0, "is_automatic"], 1)
        self.assertEqual(engineered.loc[0, "is_luxury_brand"], 1)
        self.assertEqual(engineered.loc[1, "km_per_year"], 10_000)
        self.assertTrue(pd.isna(engineered.loc[1, "engine_per_seat"]))
        self.assertEqual(engineered.loc[1, "is_automatic"], 0)
        self.assertEqual(engineered.loc[1, "is_luxury_brand"], 0)

    def test_invalid_feature_requests_and_negative_log_input_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            engineer_features(self.dataframe.head(), ["unknown_feature"])
        negative = self.dataframe.head().copy()
        negative.loc[negative.index[0], "km_driven"] = -1
        with self.assertRaises(ValueError):
            engineer_features(negative, ["log_km_driven"])

    def test_holdout_assignment_is_reproducible_and_complete(self) -> None:
        regenerated = create_holdout_assignment(self.dataframe)
        pd.testing.assert_frame_equal(regenerated, self.saved_assignment)
        self.assertEqual(len(regenerated), 15_244)
        self.assertEqual(regenerated["row_index"].nunique(), 15_244)
        self.assertEqual(int(regenerated["split"].eq("development").sum()), 12_957)
        self.assertEqual(int(regenerated["split"].eq("test").sum()), 2_287)

        band_split = pd.crosstab(regenerated["price_band"], regenerated["split"])
        test_shares = band_split["test"] / band_split.sum(axis=1)
        self.assertTrue(test_shares.between(0.145, 0.155).all())

    def test_test_holdout_was_not_evaluated(self) -> None:
        holdout = self.summary["holdout"]
        self.assertFalse(holdout["test_target_metrics_computed"])
        self.assertFalse(self.summary["test_set_evaluated"])
        self.assertFalse(self.summary["production_model_trained_or_saved"])
        self.assertEqual(holdout["test_rows"], 2_287)

    def test_cross_validation_output_is_complete(self) -> None:
        self.assertEqual(len(self.cv_results), 8)
        self.assertEqual(len(self.fold_results), 8 * 5)
        self.assertEqual(set(self.fold_results["fold"]), {1, 2, 3, 4, 5})
        self.assertEqual(
            set(self.cv_results["experiment"]),
            {
                "baseline",
                *(f"add_{feature}" for feature in CANDIDATE_FEATURES),
                "all_candidates",
            },
        )

    def test_result_interpretation_is_conservative(self) -> None:
        baseline = self.cv_results.loc[
            self.cv_results["experiment"].eq("baseline")
        ].iloc[0]
        luxury = self.cv_results.loc[
            self.cv_results["experiment"].eq("add_is_luxury_brand")
        ].iloc[0]
        self.assertAlmostEqual(float(baseline["mean_mae_inr"]), 97_820.72, places=2)
        self.assertAlmostEqual(float(luxury["mean_mae_inr"]), 97_493.24, places=2)
        self.assertAlmostEqual(
            float(luxury["mae_improvement_percentage"]), 0.3348, places=4
        )
        self.assertFalse(bool(luxury["recommended_for_phase5"]))
        self.assertEqual(self.summary["recommended_experiments_for_phase5"], [])

    def test_mapping_and_generated_outputs_are_documented(self) -> None:
        self.assertIn("Audi", LUXURY_BRANDS)
        self.assertIn("Rolls-Royce", LUXURY_BRANDS)
        self.assertNotIn("Maruti", LUXURY_BRANDS)
        for path in (
            FEATURE_SUMMARY_PATH,
            FEATURE_REPORT_PATH,
            HOLDOUT_ASSIGNMENT_PATH,
            TABLES_DIR / CV_RESULTS_FILENAME,
            TABLES_DIR / FOLD_RESULTS_FILENAME,
            FIGURES_DIR / FEATURE_FIGURE_FILENAME,
        ):
            self.assertTrue(path.exists(), path.name)
            self.assertGreater(path.stat().st_size, 10, path.name)


if __name__ == "__main__":
    unittest.main()
