"""Automated checks for leak-safe Phase 6 boosting model training."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.config import (
    CATBOOST_MODEL_PATH,
    FIGURES_DIR,
    LIGHTGBM_MODEL_PATH,
    MODEL_TRAINING_METADATA_PATH,
    MODEL_TRAINING_REPORT_PATH,
    MODEL_TRAINING_SUMMARY_PATH,
    PROCESSED_DATA_PATH,
    SPLIT_ASSIGNMENT_PATH,
    TABLES_DIR,
    XGBOOST_MODEL_PATH,
)
from src.features import BASE_FEATURE_COLUMNS, load_cleaned_features
from src.train_models import (
    COMPARISON_FIGURE_FILENAME,
    COMPARISON_TABLE_FILENAME,
    MODEL_ORDER,
    PREDICTIONS_TABLE_FILENAME,
    PRICE_BAND_FIGURE_FILENAME,
    PRICE_BAND_TABLE_FILENAME,
    _validated_predictions,
    load_fixed_split,
    prepare_catboost_features,
    split_training_and_validation,
)
from src.validate_data import file_sha256


EXPECTED_CLEANED_SHA256 = (
    "7a9742b34ebf4017cbd97dec129f0040fc661e582ab749e4f04365b7612b42a0"
)
EXPECTED_SPLIT_SHA256 = (
    "601f9ab1ec8eb30c8d30fe1c1efa6f78d6cc88df85721f716dc9963cfb1739f7"
)


class ModelTrainingTests(unittest.TestCase):
    """Verify Phase 6 metrics, split protection, and saved model artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_cleaned_features(PROCESSED_DATA_PATH)
        cls.assignment = load_fixed_split(SPLIT_ASSIGNMENT_PATH)
        (
            cls.x_train,
            cls.x_validation,
            cls.y_train,
            cls.y_validation,
            _,
            cls.validation_indices,
        ) = split_training_and_validation(cls.dataframe, cls.assignment)
        cls.summary = json.loads(
            MODEL_TRAINING_SUMMARY_PATH.read_text(encoding="utf-8")
        )
        cls.metadata = json.loads(
            MODEL_TRAINING_METADATA_PATH.read_text(encoding="utf-8")
        )
        cls.comparison = pd.read_csv(TABLES_DIR / COMPARISON_TABLE_FILENAME)
        cls.band_metrics = pd.read_csv(TABLES_DIR / PRICE_BAND_TABLE_FILENAME)
        cls.predictions = pd.read_csv(TABLES_DIR / PREDICTIONS_TABLE_FILENAME)

    def test_sources_and_fixed_split_are_unchanged(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(file_sha256(SPLIT_ASSIGNMENT_PATH), EXPECTED_SPLIT_SHA256)
        self.assertEqual(self.summary["source"]["sha256"], EXPECTED_CLEANED_SHA256)
        self.assertEqual(
            self.summary["split"]["assignment_sha256"], EXPECTED_SPLIT_SHA256
        )

    def test_only_training_and_validation_rows_are_returned(self) -> None:
        self.assertEqual(len(self.x_train), 10_670)
        self.assertEqual(len(self.x_validation), 2_287)
        self.assertEqual(len(self.y_train), 10_670)
        self.assertEqual(len(self.y_validation), 2_287)
        test_indices = set(
            self.assignment.loc[
                self.assignment["split"].eq("test"), "row_index"
            ].astype(int)
        )
        self.assertTrue(test_indices.isdisjoint(set(self.validation_indices)))
        self.assertEqual(self.summary["split"]["test_rows_used"], 0)

    def test_phase6_library_versions_are_recorded(self) -> None:
        self.assertEqual(
            self.summary["library_versions"],
            {
                "catboost": "1.2.10",
                "lightgbm": "4.7.0",
                "xgboost": "3.4.1",
                "scikit_learn": "1.8.0",
                "joblib": "1.5.3",
            },
        )

    def test_all_advanced_models_beat_the_dummy_baseline(self) -> None:
        self.assertEqual(self.comparison["model"].tolist(), list(MODEL_ORDER))
        baseline_mae = float(self.comparison.iloc[0]["mae_inr"])
        self.assertAlmostEqual(baseline_mae, 396_881.07, places=2)
        advanced = self.comparison.iloc[1:]
        self.assertTrue(advanced["mae_inr"].lt(baseline_mae).all())
        self.assertTrue(
            advanced["mae_improvement_vs_dummy_percentage"].gt(75).all()
        )

    def test_recorded_validation_metrics_and_winner(self) -> None:
        expected = {
            "CatBoost": (89_655.98, 167_076.60, 0.957272),
            "LightGBM": (96_809.49, 262_283.02, 0.894701),
            "XGBoost": (99_046.99, 411_491.93, 0.740817),
        }
        for model_name, (mae, rmse, r2) in expected.items():
            row = self.comparison.loc[self.comparison["model"].eq(model_name)].iloc[0]
            self.assertAlmostEqual(float(row["mae_inr"]), mae, places=2)
            self.assertAlmostEqual(float(row["rmse_inr"]), rmse, places=2)
            self.assertAlmostEqual(float(row["r2"]), r2, places=6)
        self.assertEqual(self.summary["preliminary_best_untuned_model"], "CatBoost")
        self.assertFalse(
            self.summary["training_contract"]["hyperparameter_tuning_performed"]
        )

    def test_validation_outputs_are_complete_and_validation_only(self) -> None:
        self.assertEqual(len(self.predictions), 2_287)
        self.assertEqual(set(self.predictions["row_index"]), set(self.validation_indices))
        self.assertEqual(len(self.band_metrics), 4 * 4)
        for model_name in MODEL_ORDER:
            rows = self.band_metrics.loc[
                self.band_metrics["model"].eq(model_name), "rows"
            ]
            self.assertEqual(int(rows.sum()), 2_287)
            self.assertEqual(rows.astype(int).tolist(), [983, 931, 256, 117])

    def test_log_inverse_is_finite_and_non_negative(self) -> None:
        predictions = _validated_predictions(np.array([-1_000.0, 0.0, 1.0]))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        self.assertEqual(predictions[0], 0.0)
        self.assertAlmostEqual(predictions[1], 0.0)
        self.assertAlmostEqual(predictions[2], np.e - 1)

    def test_unknown_categories_do_not_break_saved_models(self) -> None:
        sample = self.x_validation.head(1).copy()
        sample.loc[:, "brand"] = "Unknown Brand"
        sample.loc[:, "model"] = "Unknown Model"

        for path in (LIGHTGBM_MODEL_PATH, XGBOOST_MODEL_PATH):
            bundle = joblib.load(path)
            self.assertEqual(bundle["feature_columns"], list(BASE_FEATURE_COLUMNS))
            transformed = bundle["preprocessor"].transform(sample)
            prediction = _validated_predictions(bundle["model"].predict(transformed))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue((prediction >= 0).all())

        catboost_model = CatBoostRegressor()
        catboost_model.load_model(CATBOOST_MODEL_PATH)
        prepared = prepare_catboost_features(
            sample, self.metadata["catboost_numeric_fill_values"]
        )
        catboost_prediction = _validated_predictions(catboost_model.predict(prepared))
        self.assertTrue(np.isfinite(catboost_prediction).all())
        self.assertTrue((catboost_prediction >= 0).all())

    def test_test_set_remains_unevaluated(self) -> None:
        test = self.summary["test_set"]
        self.assertFalse(test["evaluated"])
        self.assertFalse(test["predictions_generated"])
        self.assertFalse(test["metrics_computed"])
        self.assertFalse(self.metadata["test_set_evaluated"])

    def test_saved_models_load_and_predict_in_a_fresh_process(self) -> None:
        code = f"""
import json
import joblib
import numpy as np
from catboost import CatBoostRegressor
from src.config import PROCESSED_DATA_PATH
from src.features import load_cleaned_features
from src.train_models import prepare_catboost_features

x = load_cleaned_features(PROCESSED_DATA_PATH).iloc[[0]].drop(columns='selling_price')
metadata = json.loads(open(r'{MODEL_TRAINING_METADATA_PATH}', encoding='utf-8').read())
cat = CatBoostRegressor()
cat.load_model(r'{CATBOOST_MODEL_PATH}')
cat_x = prepare_catboost_features(x, metadata['catboost_numeric_fill_values'])
outputs = [np.expm1(cat.predict(cat_x))[0]]
for path in (r'{LIGHTGBM_MODEL_PATH}', r'{XGBOOST_MODEL_PATH}'):
    bundle = joblib.load(path)
    transformed = bundle['preprocessor'].transform(x)
    outputs.append(np.expm1(bundle['model'].predict(transformed))[0])
assert np.isfinite(outputs).all()
assert (np.asarray(outputs) >= 0).all()
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROCESSED_DATA_PATH.parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_all_phase6_outputs_exist(self) -> None:
        for path in (
            MODEL_TRAINING_SUMMARY_PATH,
            MODEL_TRAINING_REPORT_PATH,
            MODEL_TRAINING_METADATA_PATH,
            CATBOOST_MODEL_PATH,
            LIGHTGBM_MODEL_PATH,
            XGBOOST_MODEL_PATH,
            TABLES_DIR / COMPARISON_TABLE_FILENAME,
            TABLES_DIR / PRICE_BAND_TABLE_FILENAME,
            TABLES_DIR / PREDICTIONS_TABLE_FILENAME,
            FIGURES_DIR / COMPARISON_FIGURE_FILENAME,
            FIGURES_DIR / PRICE_BAND_FIGURE_FILENAME,
        ):
            self.assertTrue(path.exists(), path.name)
            self.assertGreater(path.stat().st_size, 10, path.name)


if __name__ == "__main__":
    unittest.main()
