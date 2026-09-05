"""Automated checks for the frozen Phase 10 production bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.config import (
    CATBOOST_TUNED_MODEL_PATH,
    FINAL_TEST_ACCESS_PATH,
    FINAL_TEST_EVALUATION_PATH,
    PROCESSED_DATA_PATH,
    PRODUCTION_ALLOWED_INPUTS_PATH,
    PRODUCTION_BUNDLE_PATH,
    PRODUCTION_CONFIDENCE_PATH,
    PRODUCTION_FEATURE_SCHEMA_PATH,
    PRODUCTION_INTERVAL_PATH,
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_METADATA_PATH,
    PRODUCTION_METRICS_PATH,
    PRODUCTION_MODEL_CARD_PATH,
    PRODUCTION_MODEL_PATH,
    PRODUCTION_PREPROCESSING_PATH,
    PRODUCTION_REPORT_PATH,
    PRODUCTION_SUMMARY_PATH,
    PRODUCTION_VERIFICATION_PATH,
    SPLIT_ASSIGNMENT_PATH,
    TABLES_DIR,
)
from src.features import BASE_FEATURE_COLUMNS, PRICE_BAND_LABELS, load_cleaned_features
from src.package_production import (
    EXPECTED_CLEANED_SHA256,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_TUNED_MODEL_SHA256,
    FINAL_BAND_TABLE_FILENAME,
    FINAL_COVERAGE_TABLE_FILENAME,
    FINAL_LARGEST_ERRORS_FILENAME,
    FINAL_METRICS_TABLE_FILENAME,
    FINAL_PREDICTIONS_TABLE_FILENAME,
    FINAL_SEGMENT_TABLE_FILENAME,
    verify_only,
)
from src.predict import PriceRecommendationEngine, format_indian_rupees
from src.train_models import load_fixed_split
from src.validate_data import file_sha256


class ProductionArtifactTests(unittest.TestCase):
    """Verify final selection, inference safety, and artifact portability."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(PRODUCTION_SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.access = json.loads(FINAL_TEST_ACCESS_PATH.read_text(encoding="utf-8"))
        cls.evaluation = json.loads(
            FINAL_TEST_EVALUATION_PATH.read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(PRODUCTION_MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.schema = json.loads(PRODUCTION_FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.allowed = json.loads(PRODUCTION_ALLOWED_INPUTS_PATH.read_text(encoding="utf-8"))
        cls.predictions = pd.read_csv(TABLES_DIR / FINAL_PREDICTIONS_TABLE_FILENAME)
        cls.bands = pd.read_csv(TABLES_DIR / FINAL_BAND_TABLE_FILENAME)
        cls.segments = pd.read_csv(TABLES_DIR / FINAL_SEGMENT_TABLE_FILENAME)
        cls.coverage = pd.read_csv(TABLES_DIR / FINAL_COVERAGE_TABLE_FILENAME)
        cls.engine = PriceRecommendationEngine()

    def test_protected_sources_and_frozen_model_hashes_match(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(file_sha256(SPLIT_ASSIGNMENT_PATH), EXPECTED_SPLIT_SHA256)
        self.assertEqual(file_sha256(CATBOOST_TUNED_MODEL_PATH), EXPECTED_TUNED_MODEL_SHA256)
        self.assertEqual(file_sha256(PRODUCTION_MODEL_PATH), EXPECTED_TUNED_MODEL_SHA256)
        self.assertFalse(self.summary["candidate_frozen_before_test"]["retrained_in_phase10"])

    def test_final_test_access_is_recorded_exactly_once(self) -> None:
        self.assertEqual(self.access["status"], "completed")
        self.assertFalse(self.access["repeat_evaluation_allowed"])
        self.assertEqual(self.access["test_rows_evaluated"], 2_287)
        self.assertEqual(self.summary["split"]["test_evaluations"], 1)
        self.assertTrue(self.evaluation["catboost_metrics"])

    def test_final_predictions_match_fixed_test_rows_and_metrics(self) -> None:
        assignment = load_fixed_split(SPLIT_ASSIGNMENT_PATH)
        expected_indices = assignment.loc[
            assignment["split"].eq("test"), "row_index"
        ].astype(int)
        self.assertEqual(len(self.predictions), 2_287)
        self.assertEqual(
            self.predictions["row_index"].astype(int).tolist(),
            expected_indices.tolist(),
        )
        actual = self.predictions["actual_price"].to_numpy(dtype=float)
        predicted = self.predictions["recommended_price"].to_numpy(dtype=float)
        reproduced_mae = float(np.mean(np.abs(actual - predicted)))
        self.assertAlmostEqual(
            reproduced_mae, self.evaluation["catboost_metrics"]["mae_inr"], places=8
        )
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted >= 0).all())

    def test_final_model_passes_predefined_selection_checks(self) -> None:
        decision = self.summary["selection_decision"]
        self.assertTrue(decision["rule_fixed_before_test_evaluation"])
        self.assertTrue(decision["final_model_selected"])
        self.assertTrue(all(decision["checks"].values()))
        model_mae = self.evaluation["catboost_metrics"]["mae_inr"]
        dummy_mae = self.evaluation["dummy_baseline_metrics"]["mae_inr"]
        self.assertLess(model_mae, dummy_mae)

    def test_schema_order_ranges_and_brand_model_mapping_are_complete(self) -> None:
        self.assertEqual(self.schema["feature_order"], list(BASE_FEATURE_COLUMNS))
        self.assertTrue(self.schema["exact_feature_match_required"])
        self.assertEqual(set(self.schema["numeric_features"]), {
            "vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"
        })
        self.assertEqual(self.allowed["training_rows"], 10_670)
        self.assertIn("Maruti", self.allowed["brand_to_models"])
        self.assertIn("Swift", self.allowed["brand_to_models"]["Maruti"])
        self.assertNotIn("City", self.allowed["brand_to_models"]["Maruti"])

    def test_inference_returns_numeric_and_indian_currency_outputs(self) -> None:
        data = load_cleaned_features(PROCESSED_DATA_PATH)
        assignment = load_fixed_split(SPLIT_ASSIGNMENT_PATH)
        validation_index = int(
            assignment.loc[assignment["split"].eq("validation"), "row_index"].iloc[0]
        )
        sample = data.loc[validation_index, list(BASE_FEATURE_COLUMNS)].to_dict()
        result = self.engine.recommend(sample)
        self.assertGreaterEqual(result["recommended_price"], 0)
        self.assertLessEqual(result["range_lower"], result["recommended_price"])
        self.assertGreaterEqual(result["range_upper"], result["recommended_price"])
        self.assertEqual(
            result["recommended_price_display"],
            format_indian_rupees(result["recommended_price"]),
        )
        self.assertTrue(result["recommended_price_display"].startswith("₹"))
        self.assertEqual(len(result["top_shap_drivers"]), 5)

    def test_invalid_inputs_are_rejected_and_unseen_categories_are_low(self) -> None:
        verification = json.loads(
            PRODUCTION_VERIFICATION_PATH.read_text(encoding="utf-8")
        )
        sample = verification["sample"]
        with self.assertRaisesRegex(ValueError, "Feature mismatch"):
            self.engine.recommend({key: value for key, value in sample.items() if key != "seats"})
        for feature in ("vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"):
            invalid = dict(sample)
            invalid[feature] = -1
            with self.assertRaisesRegex(ValueError, feature):
                self.engine.recommend(invalid)
        for feature in ("engine", "max_power", "seats"):
            invalid = dict(sample)
            invalid[feature] = 0
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                self.engine.recommend(invalid)
        impossible = dict(sample, brand="Maruti", model="City")
        with self.assertRaisesRegex(ValueError, "Impossible brand-model"):
            self.engine.recommend(impossible)
        unseen = dict(sample, brand="Unknown Brand", model="Unknown Model")
        result = self.engine.recommend(unseen)
        self.assertEqual(result["confidence"], "Low")
        self.assertGreaterEqual(len(result["warnings"]), 2)

    def test_interval_and_segment_audits_are_complete(self) -> None:
        self.assertEqual(self.bands["segment"].tolist(), list(PRICE_BAND_LABELS))
        self.assertEqual(self.bands["rows"].astype(int).sum(), 2_287)
        self.assertEqual(
            set(self.segments["group_type"]),
            {"Actual price band", "Brand market", "Confidence", "Training support"},
        )
        overall = self.coverage.loc[self.coverage["group_type"].eq("Overall")].iloc[0]
        self.assertEqual(int(overall["rows"]), 2_287)
        self.assertLessEqual(abs(float(overall["empirical_coverage"]) - 0.8), 0.10)
        self.assertTrue(
            self.predictions["range_lower"].le(self.predictions["recommended_price"]).all()
        )
        self.assertTrue(
            self.predictions["recommended_price"].le(self.predictions["range_upper"]).all()
        )

    def test_manifest_hashes_and_zip_integrity(self) -> None:
        self.assertTrue(self.manifest["final_model_selected"])
        for entry in self.manifest["files"].values():
            path = PRODUCTION_MANIFEST_PATH.parent / entry["path"]
            self.assertTrue(path.exists(), path.name)
            self.assertEqual(file_sha256(path), entry["sha256"])
        with zipfile.ZipFile(PRODUCTION_BUNDLE_PATH) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("models/production/production_manifest.json", archive.namelist())
            self.assertIn("src/config.py", archive.namelist())
            self.assertIn("src/predict.py", archive.namelist())
        self.assertEqual(
            self.manifest["minimum_inference_dependencies"],
            ["catboost==1.2.10", "numpy==2.3.5", "pandas==2.2.3"],
        )
        for relative, entry in self.manifest["runtime_files"].items():
            self.assertEqual(file_sha256(Path(relative)), entry["sha256"])

    def test_extracted_bundle_loads_in_a_fresh_process(self) -> None:
        verification = json.loads(
            PRODUCTION_VERIFICATION_PATH.read_text(encoding="utf-8")
        )
        code = (
            "import json; from src.predict import PriceRecommendationEngine; "
            f"sample=json.loads({json.dumps(json.dumps(verification['sample']))}); "
            "result=PriceRecommendationEngine().recommend(sample); "
            "print(json.dumps(result['recommended_price']))"
        )
        dependency_path = next(
            Path(path).resolve()
            for path in sys.path
            if Path(path).name == ".python_packages"
        )
        with tempfile.TemporaryDirectory() as temporary:
            with zipfile.ZipFile(PRODUCTION_BUNDLE_PATH) as archive:
                archive.extractall(temporary)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [temporary, str(dependency_path)]
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temporary,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        reproduced = float(json.loads(completed.stdout.strip()))
        self.assertAlmostEqual(
            reproduced,
            verification["fresh_process"]["recommended_price"],
            places=6,
        )

    def test_verify_only_never_reopens_test_rows(self) -> None:
        verification = verify_only()
        self.assertTrue(verification["verified"])
        self.assertEqual(verification["test_rows_accessed"], 0)
        self.assertEqual(verification["zip_integrity"], "passed")
        self.assertLessEqual(verification["maximum_prediction_difference"], 1e-6)

    def test_all_phase10_outputs_exist(self) -> None:
        expected = (
            FINAL_TEST_ACCESS_PATH,
            FINAL_TEST_EVALUATION_PATH,
            PRODUCTION_SUMMARY_PATH,
            PRODUCTION_REPORT_PATH,
            PRODUCTION_MODEL_PATH,
            PRODUCTION_FEATURE_SCHEMA_PATH,
            PRODUCTION_ALLOWED_INPUTS_PATH,
            PRODUCTION_PREPROCESSING_PATH,
            PRODUCTION_INTERVAL_PATH,
            PRODUCTION_CONFIDENCE_PATH,
            PRODUCTION_METRICS_PATH,
            PRODUCTION_METADATA_PATH,
            PRODUCTION_VERIFICATION_PATH,
            PRODUCTION_MANIFEST_PATH,
            PRODUCTION_MODEL_CARD_PATH,
            PRODUCTION_BUNDLE_PATH,
            TABLES_DIR / FINAL_METRICS_TABLE_FILENAME,
            TABLES_DIR / FINAL_BAND_TABLE_FILENAME,
            TABLES_DIR / FINAL_SEGMENT_TABLE_FILENAME,
            TABLES_DIR / FINAL_COVERAGE_TABLE_FILENAME,
            TABLES_DIR / FINAL_PREDICTIONS_TABLE_FILENAME,
            TABLES_DIR / FINAL_LARGEST_ERRORS_FILENAME,
        )
        for path in expected:
            self.assertTrue(path.exists(), path.name)
            self.assertGreater(path.stat().st_size, 10, path.name)


if __name__ == "__main__":
    unittest.main()
