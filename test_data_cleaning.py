"""Automated checks for deterministic Phase 2 cleaning."""

from __future__ import annotations

import unittest

import pandas as pd

from src.clean_data import build_odometer_review, clean_dataframe
from src.config import (
    PRIMARY_MODEL_COLUMNS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.validate_data import file_sha256, load_raw_data


EXPECTED_RAW_SHA256 = "72481e7f9dbe00166742504d68fa5be466a0f271271c536316e1bf1af96a38b0"


class CleanDataTests(unittest.TestCase):
    """Verify every Phase 2 change and important non-change."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_raw_data(RAW_DATA_PATH)
        cls.cleaned, cls.audit = clean_dataframe(cls.raw)
        cls.saved = pd.read_csv(PROCESSED_DATA_PATH)
        cls.odometer_review = build_odometer_review(cls.raw)

    def test_raw_file_is_unchanged(self) -> None:
        self.assertEqual(file_sha256(RAW_DATA_PATH), EXPECTED_RAW_SHA256)

    def test_cleaned_shape(self) -> None:
        self.assertEqual(self.cleaned.shape, (15_244, 12))

    def test_cleaned_schema(self) -> None:
        self.assertEqual(list(self.cleaned.columns), list(PRIMARY_MODEL_COLUMNS))
        self.assertNotIn("Unnamed: 0", self.cleaned.columns)
        self.assertNotIn("car_name", self.cleaned.columns)

    def test_duplicates_are_removed(self) -> None:
        self.assertEqual(self.audit["duplicates_removed"], 167)
        self.assertEqual(int(self.cleaned.duplicated().sum()), 0)

    def test_zero_seats_become_missing_without_imputation(self) -> None:
        self.assertEqual(int(self.cleaned["seats"].eq(0).sum()), 0)
        self.assertEqual(int(self.cleaned["seats"].isna().sum()), 2)
        self.assertFalse(self.audit["statistical_imputation_performed"])

    def test_only_seats_contains_missing_values(self) -> None:
        missing = self.cleaned.isna().sum()
        self.assertEqual(int(missing["seats"]), 2)
        self.assertEqual(int(missing.drop(index="seats").sum()), 0)

    def test_extreme_odometer_values_are_retained(self) -> None:
        self.assertEqual(int(self.cleaned["km_driven"].gt(500_000).sum()), 12)
        self.assertFalse(self.audit["outlier_capping_or_removal_performed"])

    def test_odometer_review_is_complete(self) -> None:
        self.assertEqual(len(self.odometer_review), 12)
        self.assertEqual(
            int(self.odometer_review["km_equals_selling_price"].sum()), 5
        )
        self.assertTrue(
            self.odometer_review["phase_2_action"]
            .eq("retained_unchanged_correction_not_provable")
            .all()
        )

    def test_no_negative_numeric_values(self) -> None:
        numeric = self.cleaned.select_dtypes(include="number")
        self.assertEqual(int(numeric.lt(0).sum().sum()), 0)

    def test_target_is_complete_and_positive(self) -> None:
        self.assertFalse(self.cleaned["selling_price"].isna().any())
        self.assertTrue(self.cleaned["selling_price"].gt(0).all())

    def test_saved_csv_matches_cleaning_function(self) -> None:
        pd.testing.assert_frame_equal(
            self.saved,
            self.cleaned.astype({"seats": "float64"}),
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
