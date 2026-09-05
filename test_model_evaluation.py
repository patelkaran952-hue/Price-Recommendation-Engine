"""Automated checks for the Phase 7 validation-only model evaluation."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.config import (
    CATBOOST_MODEL_PATH,
    FIGURES_DIR,
    LIGHTGBM_MODEL_PATH,
    MODEL_EVALUATION_REPORT_PATH,
    MODEL_EVALUATION_SUMMARY_PATH,
    PROCESSED_DATA_PATH,
    SPLIT_ASSIGNMENT_PATH,
    TABLES_DIR,
    XGBOOST_MODEL_PATH,
)
from src.evaluate_models import (
    ACTUAL_PREDICTED_FIGURE_FILENAME,
    BRAND_FIGURE_FILENAME,
    BRAND_METRICS_FILENAME,
    FUEL_METRICS_FILENAME,
    FUEL_TRANSMISSION_FIGURE_FILENAME,
    LARGEST_ERRORS_FILENAME,
    OVERALL_METRICS_FILENAME,
    PRICE_BAND_FIGURE_FILENAME,
    PRICE_BAND_METRICS_FILENAME,
    RESIDUAL_DISTRIBUTION_FIGURE_FILENAME,
    RESIDUAL_PREDICTIONS_FILENAME,
    RESIDUAL_SCATTER_FIGURE_FILENAME,
    TRANSMISSION_METRICS_FILENAME,
    reproduce_validation_predictions,
    verify_phase6_prediction_reproduction,
)
from src.features import load_cleaned_features
from src.train_models import load_fixed_split, split_training_and_validation
from src.validate_data import file_sha256


EXPECTED_CLEANED_SHA256 = (
    "7a9742b34ebf4017cbd97dec129f0040fc661e582ab749e4f04365b7612b42a0"
)
EXPECTED_SPLIT_SHA256 = (
    "601f9ab1ec8eb30c8d30fe1c1efa6f78d6cc88df85721f716dc9963cfb1739f7"
)
EXPECTED_MODEL_HASHES = {
    "CatBoost": "1048494a0f8ac5306339a06918c6d08942e5f8116286d46af5d1215f5509c047",
    "LightGBM": "ce832e9ff159018e212c3397f84e3b19bb891c8a26bfbefb879c4c839a5d9cc9",
    "XGBoost": "e82f4caa0ce642710f4946dc365a399541efc4f6a6b4ec3c33889fedd88433f9",
}


class ModelEvaluationTests(unittest.TestCase):
    """Verify Phase 7 metrics, diagnostics, and test-set protection."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            MODEL_EVALUATION_SUMMARY_PATH.read_text(encoding="utf-8")
        )
        cls.overall = pd.read_csv(TABLES_DIR / OVERALL_METRICS_FILENAME)
        cls.bands = pd.read_csv(TABLES_DIR / PRICE_BAND_METRICS_FILENAME)
        cls.brands = pd.read_csv(TABLES_DIR / BRAND_METRICS_FILENAME)
        cls.fuels = pd.read_csv(TABLES_DIR / FUEL_METRICS_FILENAME)
        cls.transmissions = pd.read_csv(
            TABLES_DIR / TRANSMISSION_METRICS_FILENAME
        )
        cls.errors = pd.read_csv(TABLES_DIR / LARGEST_ERRORS_FILENAME)
        cls.predictions = pd.read_csv(TABLES_DIR / RESIDUAL_PREDICTIONS_FILENAME)

    def test_source_split_and_model_artifacts_are_unchanged(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(file_sha256(SPLIT_ASSIGNMENT_PATH), EXPECTED_SPLIT_SHA256)
        self.assertEqual(file_sha256(CATBOOST_MODEL_PATH), EXPECTED_MODEL_HASHES["CatBoost"])
        self.assertEqual(file_sha256(LIGHTGBM_MODEL_PATH), EXPECTED_MODEL_HASHES["LightGBM"])
        self.assertEqual(file_sha256(XGBOOST_MODEL_PATH), EXPECTED_MODEL_HASHES["XGBoost"])
        self.assertFalse(
            self.summary["phase6_artifact_verification"]["models_retrained"]
        )

    def test_saved_predictions_are_exactly_reproduced(self) -> None:
        dataframe = load_cleaned_features(PROCESSED_DATA_PATH)
        assignment = load_fixed_split(SPLIT_ASSIGNMENT_PATH)
        _, x_validation, _, _, _, validation_indices = split_training_and_validation(
            dataframe, assignment
        )
        metadata = json.loads(
            (CATBOOST_MODEL_PATH.parent / "phase6_model_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        reproduced = reproduce_validation_predictions(x_validation, metadata)
        maximum_difference = verify_phase6_prediction_reproduction(
            validation_indices, reproduced
        )
        self.assertLessEqual(maximum_difference, 1e-5)
        self.assertTrue(
            self.summary["phase6_artifact_verification"]["predictions_reproduced"]
        )

    def test_overall_metrics_match_phase6_and_required_schema(self) -> None:
        self.assertEqual(
            self.overall["model"].tolist(),
            ["Dummy baseline", "CatBoost", "LightGBM", "XGBoost"],
        )
        for column in (
            "mae_inr",
            "rmse_inr",
            "r2",
            "median_absolute_error_inr",
            "rmsle",
        ):
            self.assertIn(column, self.overall.columns)
        expected = {
            "CatBoost": (89_655.98, 167_076.60, 0.957272),
            "LightGBM": (96_809.49, 262_283.02, 0.894701),
            "XGBoost": (99_046.99, 411_491.93, 0.740817),
        }
        for model_name, (mae, rmse, r2) in expected.items():
            row = self.overall.loc[self.overall["model"].eq(model_name)].iloc[0]
            self.assertAlmostEqual(float(row["mae_inr"]), mae, places=2)
            self.assertAlmostEqual(float(row["rmse_inr"]), rmse, places=2)
            self.assertAlmostEqual(float(row["r2"]), r2, places=6)

    def test_practical_error_rates_and_residual_definition(self) -> None:
        catboost = self.overall.loc[self.overall["model"].eq("CatBoost")].iloc[0]
        self.assertAlmostEqual(
            float(catboost["within_20_percent_rate"]), 0.825536, places=6
        )
        self.assertAlmostEqual(
            float(catboost["mean_residual_inr"]), 7_891.49, places=2
        )
        self.assertEqual(
            self.summary["residual_definition"],
            "actual_price - predicted_price; positive means underprediction",
        )
        example = self.predictions.loc[self.predictions["row_index"].eq(13_852)].iloc[0]
        self.assertGreater(float(example["catboost_residual"]), 0)
        self.assertAlmostEqual(
            float(example["catboost_residual"]),
            float(example["actual_price"] - example["catboost_prediction"]),
            places=6,
        )

    def test_price_band_results_are_complete_and_stable(self) -> None:
        self.assertEqual(len(self.bands), 4 * 4)
        for model_name in ("Dummy baseline", "CatBoost", "LightGBM", "XGBoost"):
            rows = self.bands.loc[self.bands["model"].eq(model_name), "rows"]
            self.assertEqual(rows.astype(int).tolist(), [983, 931, 256, 117])
        winner_models = [
            row["model"] for row in self.summary["price_band_winners"]
        ]
        self.assertEqual(
            winner_models, ["CatBoost", "XGBoost", "CatBoost", "CatBoost"]
        )

    def test_major_brand_threshold_and_wins_are_recorded(self) -> None:
        major = self.brands[self.brands["major_brand"].astype(bool)]
        self.assertEqual(major["brand"].nunique(), 13)
        self.assertTrue(major["rows"].ge(30).all())
        self.assertEqual(
            self.summary["major_brand_mae_wins"],
            {"CatBoost": 8, "LightGBM": 3, "XGBoost": 2},
        )

    def test_rare_fuel_categories_are_flagged_not_hidden(self) -> None:
        fuel_counts = self.fuels.groupby("fuel_type")["rows"].first().astype(int)
        self.assertEqual(fuel_counts.loc["Electric"], 1)
        self.assertEqual(fuel_counts.loc["LPG"], 8)
        rare = self.summary["rare_fuel_warning"]["categories"]
        self.assertEqual(
            rare,
            [{"fuel_type": "Electric", "rows": 1}, {"fuel_type": "LPG", "rows": 8}],
        )
        self.assertEqual(set(self.transmissions["transmission_type"]), {"Manual", "Automatic"})

    def test_largest_errors_cover_both_directions_for_every_model(self) -> None:
        self.assertEqual(len(self.errors), 3 * 2 * 10)
        counts = self.errors.groupby(["evaluated_model", "direction"]).size()
        self.assertTrue(counts.eq(10).all())
        cat_under = self.errors.loc[
            self.errors["evaluated_model"].eq("CatBoost")
            & self.errors["direction"].eq("Underprediction")
            & self.errors["rank"].eq(1)
        ].iloc[0]
        self.assertEqual(int(cat_under["row_index"]), 13_852)
        self.assertAlmostEqual(float(cat_under["residual_inr"]), 2_251_388.19, places=2)

    def test_evaluation_table_contains_validation_rows_only(self) -> None:
        assignment = pd.read_csv(SPLIT_ASSIGNMENT_PATH)
        validation_indices = set(
            assignment.loc[assignment["split"].eq("validation"), "row_index"]
        )
        test_indices = set(
            assignment.loc[assignment["split"].eq("test"), "row_index"]
        )
        output_indices = set(self.predictions["row_index"].astype(int))
        self.assertEqual(len(output_indices), 2_287)
        self.assertEqual(output_indices, validation_indices)
        self.assertTrue(output_indices.isdisjoint(test_indices))

    def test_phase8_recommendation_is_not_final_selection(self) -> None:
        recommendation = self.summary["phase8_recommendation"]
        self.assertEqual(recommendation["primary_tuning_candidate"], "CatBoost")
        self.assertEqual(recommendation["challenger"], "LightGBM")
        self.assertFalse(recommendation["final_model_selected"])
        self.assertFalse(self.summary["test_set"]["evaluated"])
        self.assertFalse(self.summary["test_set"]["predictions_generated"])
        self.assertFalse(self.summary["test_set"]["metrics_computed"])

    def test_all_phase7_outputs_exist(self) -> None:
        for path in (
            MODEL_EVALUATION_SUMMARY_PATH,
            MODEL_EVALUATION_REPORT_PATH,
            TABLES_DIR / OVERALL_METRICS_FILENAME,
            TABLES_DIR / PRICE_BAND_METRICS_FILENAME,
            TABLES_DIR / BRAND_METRICS_FILENAME,
            TABLES_DIR / FUEL_METRICS_FILENAME,
            TABLES_DIR / TRANSMISSION_METRICS_FILENAME,
            TABLES_DIR / LARGEST_ERRORS_FILENAME,
            TABLES_DIR / RESIDUAL_PREDICTIONS_FILENAME,
            FIGURES_DIR / ACTUAL_PREDICTED_FIGURE_FILENAME,
            FIGURES_DIR / RESIDUAL_DISTRIBUTION_FIGURE_FILENAME,
            FIGURES_DIR / RESIDUAL_SCATTER_FIGURE_FILENAME,
            FIGURES_DIR / PRICE_BAND_FIGURE_FILENAME,
            FIGURES_DIR / BRAND_FIGURE_FILENAME,
            FIGURES_DIR / FUEL_TRANSMISSION_FIGURE_FILENAME,
        ):
            self.assertTrue(path.exists(), path.name)
            self.assertGreater(path.stat().st_size, 10, path.name)


if __name__ == "__main__":
    unittest.main()
