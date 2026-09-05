"""Automated checks for the Phase 3 exploratory analysis."""

from __future__ import annotations

import json
import unittest

from src.config import (
    EDA_SUMMARY_PATH,
    FIGURES_DIR,
    PROCESSED_DATA_PATH,
    TABLES_DIR,
)
from src.eda import FIGURE_FILENAMES, create_eda_tables, load_cleaned_data
from src.validate_data import file_sha256


EXPECTED_CLEANED_SHA256 = (
    "7a9742b34ebf4017cbd97dec129f0040fc661e582ab749e4f04365b7612b42a0"
)
EXPECTED_TABLES = (
    "price_band_summary",
    "brand_price_summary",
    "model_price_summary",
    "category_price_summary",
    "vehicle_age_price_summary",
    "category_coverage",
    "pearson_correlation",
    "spearman_correlation",
    "matched_seller_comparison",
)


class ExploratoryDataAnalysisTests(unittest.TestCase):
    """Verify the Phase 3 calculations, outputs, and non-modeling scope."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_cleaned_data(PROCESSED_DATA_PATH)
        cls.tables = create_eda_tables(cls.dataframe)
        cls.summary = json.loads(EDA_SUMMARY_PATH.read_text(encoding="utf-8"))

    def test_cleaned_input_is_unchanged(self) -> None:
        self.assertEqual(file_sha256(PROCESSED_DATA_PATH), EXPECTED_CLEANED_SHA256)
        self.assertEqual(self.summary["source"]["sha256"], EXPECTED_CLEANED_SHA256)

    def test_price_distribution_summary(self) -> None:
        price = self.summary["price_distribution"]
        self.assertEqual(price["minimum"], 40_000)
        self.assertEqual(price["median"], 559_000.0)
        self.assertAlmostEqual(price["mean"], 774_701.45, places=2)
        self.assertEqual(price["maximum"], 39_500_000)
        self.assertGreater(price["skewness"], 10)

    def test_price_bands_cover_every_row(self) -> None:
        bands = self.tables["price_band_summary"]
        self.assertEqual(int(bands["count"].sum()), 15_244)
        self.assertAlmostEqual(float(bands["percentage"].sum()), 100.0, places=8)
        self.assertEqual(
            bands["count"].astype(int).tolist(), [6_556, 6_207, 1_704, 777]
        )

    def test_key_category_medians_and_counts(self) -> None:
        categories = self.summary["category_price_summary"]
        self.assertEqual(
            categories["transmission_type"]["Automatic"]["median_price"],
            1_050_000.0,
        )
        self.assertEqual(
            categories["transmission_type"]["Manual"]["median_price"],
            500_000.0,
        )
        self.assertEqual(categories["fuel_type"]["Electric"]["count"], 4)
        self.assertEqual(categories["seller_type"]["Dealer"]["count"], 9_459)

    def test_numeric_relationships(self) -> None:
        correlations = self.summary["numeric_correlations_with_price"]
        self.assertAlmostEqual(correlations["spearman"]["max_power"], 0.7211, places=4)
        self.assertAlmostEqual(correlations["spearman"]["engine"], 0.6429, places=4)
        self.assertAlmostEqual(correlations["spearman"]["vehicle_age"], -0.4778, places=4)
        self.assertAlmostEqual(correlations["pearson"]["max_power"], 0.7511, places=4)

    def test_age_comparison_is_recorded_as_cross_sectional(self) -> None:
        age = self.summary["vehicle_age"]
        self.assertEqual(age["median_price_age_2"], 820_000.0)
        self.assertEqual(age["median_price_age_10"], 335_500.0)
        self.assertAlmostEqual(
            age["cross_sectional_change_age_2_to_10_percentage"], -59.09, places=2
        )
        self.assertIn("not a measured depreciation rate", age["warning"])

    def test_category_coverage_and_matched_seller_result(self) -> None:
        coverage = self.summary["coverage"]
        self.assertEqual(coverage["brands"], 32)
        self.assertEqual(coverage["models"], 120)
        self.assertEqual(coverage["brands_under_50_rows"], 16)
        self.assertEqual(coverage["models_under_50_rows"], 65)

        seller = self.summary["matched_seller_analysis"]
        self.assertEqual(seller["groups_with_at_least_5_each"], 258)
        self.assertEqual(seller["median_dealer_minus_individual"], 27_500.0)
        self.assertAlmostEqual(seller["dealer_median_higher_percentage"], 75.97, places=2)

    def test_phase_three_does_not_model_or_impute(self) -> None:
        quality = self.summary["data_quality_context"]
        self.assertFalse(self.summary["modeling_or_imputation_performed"])
        self.assertEqual(quality["missing_seat_values_left_unimputed"], 2)
        self.assertEqual(quality["odometer_rows_above_500000_retained"], 12)

    def test_all_figures_and_tables_exist(self) -> None:
        self.assertEqual(set(self.tables), set(EXPECTED_TABLES))
        for filename in FIGURE_FILENAMES:
            path = FIGURES_DIR / filename
            self.assertTrue(path.exists(), filename)
            self.assertGreater(path.stat().st_size, 10_000, filename)
        for table_name in EXPECTED_TABLES:
            path = TABLES_DIR / f"{table_name}.csv"
            self.assertTrue(path.exists(), table_name)
            self.assertGreater(path.stat().st_size, 10, table_name)


if __name__ == "__main__":
    unittest.main()
