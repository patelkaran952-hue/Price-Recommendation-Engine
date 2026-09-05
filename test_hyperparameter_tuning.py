"""Automated checks for controlled, leak-safe Phase 8 tuning."""

from __future__ import annotations

import json
import unittest

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.config import (
    CATBOOST_MODEL_PATH,
    CATBOOST_TUNED_MODEL_PATH,
    FIGURES_DIR,
    HYPERPARAMETER_TUNING_METADATA_PATH,
    HYPERPARAMETER_TUNING_REPORT_PATH,
    HYPERPARAMETER_TUNING_SUMMARY_PATH,
    LIGHTGBM_MODEL_PATH,
    LIGHTGBM_TUNED_MODEL_PATH,
    PROCESSED_DATA_PATH,
    SPLIT_ASSIGNMENT_PATH,
    TABLES_DIR,
    XGBOOST_MODEL_PATH,
)
from src.features import BASE_FEATURE_COLUMNS, PRICE_BAND_LABELS, load_cleaned_features
from src.train_models import (
    _validated_predictions,
    load_fixed_split,
    prepare_catboost_features,
    split_training_and_validation,
)
from src.tune_models import (
    CATBOOST_TRIALS,
    CV_FOLDS,
    CV_RESULTS_FIGURE_FILENAME,
    FOLDS_TABLE_FILENAME,
    LIGHTGBM_TRIALS,
    PREDICTIONS_FILENAME,
    PRICE_BAND_FIGURE_FILENAME,
    PRICE_BAND_FILENAME,
    TRIALS_TABLE_FILENAME,
    VALIDATION_COMPARISON_FIGURE_FILENAME,
    VALIDATION_COMPARISON_FILENAME,
    build_training_cv_splits,
)
from src.validate_data import file_sha256


EXPECTED_CLEANED_SHA256 = (
    "7a9742b34ebf4017cbd97dec129f0040fc661e582ab749e4f04365b7612b42a0"
)
EXPECTED_SPLIT_SHA256 = (
    "601f9ab1ec8eb30c8d30fe1c1efa6f78d6cc88df85721f716dc9963cfb1739f7"
)
EXPECTED_UNTUNED_HASHES = {
    CATBOOST_MODEL_PATH: "1048494a0f8ac5306339a06918c6d08942e5f8116286d46af5d1215f5509c047",
    LIGHTGBM_MODEL_PATH: "ce832e9ff159018e212c3397f84e3b19bb891c8a26bfbefb879c4c839a5d9cc9",
    XGBOOST_MODEL_PATH: "e82f4caa0ce642710f4946dc365a399541efc4f6a6b4ec3c33889fedd88433f9",
}


class HyperparameterTuningTests(unittest.TestCase):
    """Verify search scope, metrics, artifacts, and test-set protection."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_cleaned_features(PROCESSED_DATA_PATH)
        cls.assignment = load_fixed_split(SPLIT_ASSIGNMENT_PATH)
        (
            cls.x_train,
            cls.x_validation,
            cls.y_train,
            cls.y_validation,
            cls.training_indices,
            cls.validation_indices,
        ) = split_training_and_validation(cls.dataframe, cls.assignment)
        cls.summary = json.loads(
            HYPERPARAMETER_TUNING_SUMMARY_PATH.read_text(encoding="utf-8")
        )
        cls.metadata = json.loads(
            HYPERPARAMETER_TUNING_METADATA_PATH.read_text(encoding="utf-8")
        )
        cls.trials = pd.read_csv(TABLES_DIR / TRIALS_TABLE_FILENAME)
        cls.folds = pd.read_csv(TABLES_DIR / FOLDS_TABLE_FILENAME)
        cls.comparison = pd.read_csv(TABLES_DIR / VALIDATION_COMPARISON_FILENAME)
        cls.bands = pd.read_csv(TABLES_DIR / PRICE_BAND_FILENAME)
        cls.predictions = pd.read_csv(TABLES_DIR / PREDICTIONS_FILENAME)

    def test_sources_split_and_untuned_artifacts_are_unchanged(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(file_sha256(SPLIT_ASSIGNMENT_PATH), EXPECTED_SPLIT_SHA256)
        for path, expected_hash in EXPECTED_UNTUNED_HASHES.items():
            self.assertEqual(file_sha256(path), expected_hash)

    def test_cv_contains_training_rows_only_and_is_price_stratified(self) -> None:
        strata, folds = build_training_cv_splits(
            self.assignment, self.training_indices
        )
        self.assertEqual(len(strata), 10_670)
        self.assertEqual(len(folds), CV_FOLDS)
        test_indices = set(
            self.assignment.loc[
                self.assignment["split"].eq("test"), "row_index"
            ].astype(int)
        )
        for fit_positions, score_positions in folds:
            self.assertTrue(set(fit_positions).isdisjoint(set(score_positions)))
            fold_original_indices = set(self.training_indices[score_positions])
            self.assertTrue(fold_original_indices.isdisjoint(test_indices))
            self.assertEqual(set(strata[score_positions]), set(PRICE_BAND_LABELS))

    def test_search_counts_rankings_and_fold_audit_are_complete(self) -> None:
        self.assertEqual(len(self.trials), CATBOOST_TRIALS + LIGHTGBM_TRIALS)
        self.assertEqual(
            len(self.folds), (CATBOOST_TRIALS + LIGHTGBM_TRIALS) * CV_FOLDS
        )
        self.assertEqual(
            self.trials.groupby("model").size().to_dict(),
            {"CatBoost": 8, "LightGBM": 6},
        )
        self.assertEqual(
            self.folds.groupby(["model", "trial"]).size().unique().tolist(), [3]
        )
        for _, group in self.trials.groupby("model"):
            self.assertEqual(sorted(group["rank"].astype(int)), list(range(1, len(group) + 1)))
        self.assertEqual(self.summary["cv_audit"]["fold_result_rows"], 42)

    def test_search_objective_and_leakage_contract_are_recorded(self) -> None:
        contract = self.summary["search_contract"]
        self.assertIn("original INR", contract["objective"])
        self.assertEqual(contract["folds"], 3)
        self.assertEqual(contract["random_seed"], 42)
        self.assertEqual(contract["cpu_threads"], 4)
        self.assertTrue(contract["preprocessing_fit_inside_each_fold"])
        self.assertFalse(contract["validation_used_during_search"])

    def test_best_parameters_and_cross_validation_results_are_recorded(self) -> None:
        self.assertEqual(
            self.summary["models"]["CatBoost"]["best_parameters"],
            {
                "bagging_temperature": 3.0,
                "border_count": 128,
                "depth": 6,
                "l2_leaf_reg": 5.0,
                "learning_rate": 0.08,
                "random_strength": 0.5,
            },
        )
        self.assertEqual(
            self.summary["models"]["LightGBM"]["best_parameters"],
            {
                "colsample_bytree": 1.0,
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_samples": 10,
                "num_leaves": 31,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "subsample": 1.0,
            },
        )
        self.assertGreater(self.summary["models"]["CatBoost"]["search_seconds"], 0)
        self.assertGreater(self.summary["models"]["LightGBM"]["search_seconds"], 0)

    def test_tuned_catboost_improves_and_is_phase9_candidate(self) -> None:
        values = self.comparison.set_index("model")["mae_inr"]
        self.assertAlmostEqual(values["CatBoost untuned"], 89_655.98, places=2)
        self.assertAlmostEqual(values["CatBoost tuned"], 88_383.72, places=2)
        self.assertLess(values["CatBoost tuned"], values["CatBoost untuned"])
        self.assertEqual(self.summary["phase9_candidate"]["model"], "CatBoost tuned")
        self.assertFalse(self.summary["phase9_candidate"]["final_model_selected"])
        self.assertAlmostEqual(
            self.summary["models"]["CatBoost"]["validation_mae_change_vs_untuned_percentage"],
            1.419056,
            places=6,
        )

    def test_lightgbm_non_improvement_is_retained_not_promoted(self) -> None:
        values = self.comparison.set_index("model")["mae_inr"]
        self.assertAlmostEqual(values["LightGBM untuned"], 96_809.49, places=2)
        self.assertAlmostEqual(values["LightGBM tuned"], 97_847.50, places=2)
        self.assertGreater(values["LightGBM tuned"], values["LightGBM untuned"])
        self.assertFalse(self.summary["models"]["LightGBM"]["validation_improved"])

    def test_validation_prediction_and_price_band_tables_are_complete(self) -> None:
        validation_indices = set(self.validation_indices.astype(int))
        test_indices = set(
            self.assignment.loc[
                self.assignment["split"].eq("test"), "row_index"
            ].astype(int)
        )
        output_indices = set(self.predictions["row_index"].astype(int))
        self.assertEqual(len(self.predictions), 2_287)
        self.assertEqual(output_indices, validation_indices)
        self.assertTrue(output_indices.isdisjoint(test_indices))
        self.assertEqual(len(self.bands), 4 * 4)
        for model_name in (
            "CatBoost untuned", "CatBoost tuned", "LightGBM untuned", "LightGBM tuned"
        ):
            rows = self.bands.loc[self.bands["model"].eq(model_name), "rows"]
            self.assertEqual(rows.astype(int).tolist(), [983, 931, 256, 117])

    def test_tuned_artifacts_reload_and_support_unknown_categories(self) -> None:
        sample = self.x_validation.head(1).copy()
        sample.loc[:, "brand"] = "Unknown Brand"
        sample.loc[:, "model"] = "Unknown Model"

        catboost_model = CatBoostRegressor()
        catboost_model.load_model(CATBOOST_TUNED_MODEL_PATH)
        catboost_input = prepare_catboost_features(
            sample, self.metadata["catboost_numeric_fill_values"]
        )
        catboost_prediction = _validated_predictions(
            catboost_model.predict(catboost_input)
        )

        lightgbm_bundle = joblib.load(LIGHTGBM_TUNED_MODEL_PATH)
        self.assertEqual(
            lightgbm_bundle["feature_columns"], list(BASE_FEATURE_COLUMNS)
        )
        transformed = lightgbm_bundle["preprocessor"].transform(sample)
        lightgbm_prediction = _validated_predictions(
            lightgbm_bundle["model"].predict(transformed)
        )
        for prediction in (catboost_prediction, lightgbm_prediction):
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue((prediction >= 0).all())

    def test_test_set_is_still_sealed(self) -> None:
        self.assertEqual(self.summary["split"]["test_rows_used"], 0)
        self.assertFalse(self.summary["test_set"]["evaluated"])
        self.assertFalse(self.summary["test_set"]["transformed"])
        self.assertFalse(self.summary["test_set"]["predictions_generated"])
        self.assertFalse(self.summary["test_set"]["metrics_computed"])
        self.assertFalse(self.metadata["test_set_evaluated"])

    def test_all_phase8_outputs_exist_and_artifact_hashes_match(self) -> None:
        expected_paths = (
            HYPERPARAMETER_TUNING_SUMMARY_PATH,
            HYPERPARAMETER_TUNING_REPORT_PATH,
            HYPERPARAMETER_TUNING_METADATA_PATH,
            CATBOOST_TUNED_MODEL_PATH,
            LIGHTGBM_TUNED_MODEL_PATH,
            TABLES_DIR / TRIALS_TABLE_FILENAME,
            TABLES_DIR / FOLDS_TABLE_FILENAME,
            TABLES_DIR / VALIDATION_COMPARISON_FILENAME,
            TABLES_DIR / PRICE_BAND_FILENAME,
            TABLES_DIR / PREDICTIONS_FILENAME,
            FIGURES_DIR / CV_RESULTS_FIGURE_FILENAME,
            FIGURES_DIR / VALIDATION_COMPARISON_FIGURE_FILENAME,
            FIGURES_DIR / PRICE_BAND_FIGURE_FILENAME,
        )
        for path in expected_paths:
            self.assertTrue(path.exists(), path.name)
            self.assertGreater(path.stat().st_size, 10, path.name)
        self.assertEqual(
            file_sha256(CATBOOST_TUNED_MODEL_PATH),
            self.summary["artifacts"]["CatBoost tuned"]["sha256"],
        )
        self.assertEqual(
            file_sha256(LIGHTGBM_TUNED_MODEL_PATH),
            self.summary["artifacts"]["LightGBM tuned"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
