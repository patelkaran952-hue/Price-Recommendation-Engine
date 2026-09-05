"""Automated checks for the fixed Phase 5 split and dummy baseline."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

from src.baseline import (
    BASELINE_FIGURE_FILENAME,
    EXPECTED_SPLIT_COUNTS,
    PRICE_BAND_TABLE_FILENAME,
    SPLIT_FIGURE_FILENAME,
    create_final_split_assignment,
    load_phase4_holdout,
)
from src.config import (
    BASELINE_METADATA_PATH,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    BASELINE_SUMMARY_PATH,
    FIGURES_DIR,
    HOLDOUT_ASSIGNMENT_PATH,
    PROCESSED_DATA_PATH,
    SPLIT_ASSIGNMENT_PATH,
    TABLES_DIR,
)
from src.features import BASE_FEATURE_COLUMNS, load_cleaned_features
from src.validate_data import file_sha256


EXPECTED_CLEANED_SHA256 = (
    "7a9742b34ebf4017cbd97dec129f0040fc661e582ab749e4f04365b7612b42a0"
)


class BaselineTests(unittest.TestCase):
    """Verify split integrity, baseline metrics, and saved artifact loading."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_cleaned_features(PROCESSED_DATA_PATH)
        cls.holdout = load_phase4_holdout(HOLDOUT_ASSIGNMENT_PATH)
        cls.assignment = pd.read_csv(SPLIT_ASSIGNMENT_PATH)
        cls.summary = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.band_metrics = pd.read_csv(TABLES_DIR / PRICE_BAND_TABLE_FILENAME)
        cls.model = joblib.load(BASELINE_MODEL_PATH)

    def test_cleaned_source_and_phase4_holdout_are_unchanged(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(self.summary["source"]["sha256"], EXPECTED_CLEANED_SHA256)
        self.assertEqual(
            file_sha256(HOLDOUT_ASSIGNMENT_PATH),
            self.summary["split"]["phase4_holdout_sha256"],
        )

    def test_final_split_is_reproducible_and_complete(self) -> None:
        regenerated = create_final_split_assignment(self.holdout)
        pd.testing.assert_frame_equal(regenerated, self.assignment)
        self.assertEqual(len(self.assignment), 15_244)
        self.assertEqual(self.assignment["row_index"].nunique(), 15_244)
        self.assertEqual(
            self.assignment["split"].value_counts().to_dict(),
            EXPECTED_SPLIT_COUNTS,
        )

    def test_phase4_test_rows_are_preserved_exactly(self) -> None:
        phase4_test = set(
            self.holdout.loc[self.holdout["split"].eq("test"), "row_index"]
        )
        phase5_test = set(
            self.assignment.loc[self.assignment["split"].eq("test"), "row_index"]
        )
        self.assertEqual(phase4_test, phase5_test)
        self.assertTrue(self.summary["split"]["test_rows_preserved_exactly"])

    def test_price_band_stratification_is_balanced(self) -> None:
        split_band = pd.crosstab(
            self.assignment["price_band"], self.assignment["split"], normalize="columns"
        )
        max_gap = float(split_band.max(axis=1).sub(split_band.min(axis=1)).max())
        self.assertLess(max_gap, 0.002)

    def test_dummy_model_contract_and_prediction(self) -> None:
        self.assertEqual(self.model.strategy, "median")
        self.assertEqual(int(self.model.n_features_in_), len(BASE_FEATURE_COLUMNS))
        sample = self.dataframe.loc[:, list(BASE_FEATURE_COLUMNS)].head(3)
        predictions = self.model.predict(sample)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        np.testing.assert_allclose(predictions, [560_000.0] * 3)
        self.assertEqual(
            self.summary["feature_contract"]["feature_columns"],
            list(BASE_FEATURE_COLUMNS),
        )

    def test_validation_metrics_match_the_recorded_baseline(self) -> None:
        metrics = self.summary["validation_metrics"]
        self.assertAlmostEqual(metrics["mae_inr"], 396_881.07, places=2)
        self.assertAlmostEqual(metrics["rmse_inr"], 834_402.15, places=2)
        self.assertAlmostEqual(metrics["r2"], -0.065699, places=6)
        self.assertEqual(metrics["median_absolute_error_inr"], 210_000.0)
        self.assertAlmostEqual(metrics["rmsle"], 0.694584, places=6)

    def test_validation_price_band_metrics_are_complete(self) -> None:
        self.assertEqual(self.band_metrics["rows"].astype(int).tolist(), [983, 931, 256, 117])
        self.assertEqual(int(self.band_metrics["rows"].sum()), 2_287)
        luxury_mae = float(self.band_metrics.iloc[-1]["mae_inr"])
        self.assertAlmostEqual(luxury_mae, 2_924_786.32, places=2)

    def test_test_set_remains_unevaluated(self) -> None:
        test = self.summary["test_set"]
        self.assertFalse(test["evaluated"])
        self.assertFalse(test["predictions_generated"])
        self.assertFalse(test["metrics_computed"])
        self.assertFalse(self.summary["advanced_model_trained"])

    def test_saved_model_loads_in_a_fresh_process(self) -> None:
        code = (
            "import joblib, numpy as np; "
            f"model=joblib.load(r'{BASELINE_MODEL_PATH}'); "
            "prediction=model.predict(np.zeros((1, 11))); "
            "assert prediction.tolist() == [560000.0]"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_all_phase5_outputs_exist(self) -> None:
        for path in (
            SPLIT_ASSIGNMENT_PATH,
            BASELINE_SUMMARY_PATH,
            BASELINE_REPORT_PATH,
            TABLES_DIR / PRICE_BAND_TABLE_FILENAME,
            FIGURES_DIR / SPLIT_FIGURE_FILENAME,
            FIGURES_DIR / BASELINE_FIGURE_FILENAME,
            BASELINE_MODEL_PATH,
            BASELINE_METADATA_PATH,
        ):
            self.assertTrue(path.exists(), path.name)
            self.assertGreater(path.stat().st_size, 10, path.name)


if __name__ == "__main__":
    unittest.main()
