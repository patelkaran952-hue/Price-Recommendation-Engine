"""Automated checks for the Phase 1 raw-data validation."""

from __future__ import annotations

import unittest

from src.config import EXPECTED_COLUMNS, RAW_DATA_PATH
from src.validate_data import load_raw_data, summarize_dataframe


class RawDataValidationTests(unittest.TestCase):
    """Confirm that the supplied dataset matches the recorded raw profile."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_raw_data(RAW_DATA_PATH)
        cls.summary = summarize_dataframe(cls.dataframe, RAW_DATA_PATH)

    def test_expected_schema_is_present(self) -> None:
        self.assertEqual(list(self.dataframe.columns), list(EXPECTED_COLUMNS))
        self.assertTrue(self.summary["integrity_checks"]["required_schema_present"])

    def test_raw_shape_is_stable(self) -> None:
        self.assertEqual(self.dataframe.shape, (15_411, 14))

    def test_no_blank_cells_in_supplied_csv(self) -> None:
        self.assertEqual(int(self.dataframe.isna().sum().sum()), 0)

    def test_duplicate_count_excludes_export_index(self) -> None:
        self.assertEqual(
            self.summary["duplicates"]["listings_after_export_index_removed"], 167
        )

    def test_known_invalid_seat_values_are_detected(self) -> None:
        self.assertEqual(self.summary["integrity_checks"]["zero_seat_rows"], 2)

    def test_extreme_odometer_values_are_detected(self) -> None:
        self.assertEqual(
            self.summary["integrity_checks"]["km_driven_above_500000_rows"], 12
        )

    def test_car_name_redundancy_is_detected(self) -> None:
        self.assertTrue(
            self.summary["integrity_checks"]["car_name_redundant_for_all_rows"]
        )

    def test_numeric_values_are_non_negative(self) -> None:
        counts = self.summary["integrity_checks"]["negative_numeric_value_counts"]
        self.assertTrue(all(count == 0 for count in counts.values()))

    def test_target_is_positive(self) -> None:
        self.assertTrue(self.summary["integrity_checks"]["target_values_all_positive"])


if __name__ == "__main__":
    unittest.main()
