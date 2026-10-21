from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.analytics.patient_analytics import build_patient_analytics, get_ai_context
from src.database.database import initialize_database


ROOT = Path(__file__).resolve().parents[1]
CSV_FILENAMES = ("patients.csv", "meals.csv", "glucose.csv", "lifestyle.csv", "food_nutrition.csv")


class PatientAnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        data_dir = root / "data"
        data_dir.mkdir()
        for filename in CSV_FILENAMES:
            shutil.copyfile(ROOT / "data" / filename, data_dir / filename)
        self.database_path = root / "glucocare.sqlite3"
        initialize_database(data_dir, self.database_path)
        connection = sqlite3.connect(self.database_path)
        self.start_date, self.end_date = connection.execute(
            "SELECT MIN(date), MAX(date) FROM meals WHERE patient_id = 'P001'"
        ).fetchone()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_combined_patient_analytics_and_ai_context(self) -> None:
        analytics = build_patient_analytics(
            "P001", self.start_date, self.end_date, database_path=self.database_path
        )
        self.assertIsNotNone(analytics)
        assert analytics is not None
        self.assertEqual(analytics["meal_summary"]["recorded_meal_count"], 21)
        self.assertEqual(analytics["glucose_summary"]["reading_count"], 14)
        self.assertEqual(analytics["lifestyle_summary"]["record_count"], 7)
        self.assertEqual(analytics["data_quality"]["meal_record_count"], 21)
        self.assertEqual(analytics["data_quality"]["missing_meal_timestamps"], 21)
        self.assertEqual(analytics["meal_glucose_associations"], [])

        context = get_ai_context(
            "P001", self.start_date, self.end_date, database_path=self.database_path
        )
        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context["patient_profile"]["patient_id"], "P001")
        self.assertIn("meal", context["analytics"])
        self.assertIn("data_quality", context)
        self.assertTrue(context["is_demo"])
        self.assertIsNone(build_patient_analytics("INVALID", database_path=self.database_path))
        self.assertIsNone(get_ai_context("INVALID", database_path=self.database_path))

    def test_empty_histories_are_reported_without_fabricated_values(self) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute("DELETE FROM meals WHERE patient_id = 'P001'")
        connection.execute("DELETE FROM glucose WHERE patient_id = 'P001'")
        connection.execute("DELETE FROM lifestyle WHERE patient_id = 'P001'")
        connection.commit()
        connection.close()
        analytics = build_patient_analytics(
            "P001", self.start_date, self.end_date, database_path=self.database_path
        )
        self.assertIsNotNone(analytics)
        assert analytics is not None
        self.assertEqual(analytics["meal_summary"]["recorded_meal_count"], 0)
        self.assertIsNone(analytics["glucose_summary"]["average_recorded_glucose"])
        self.assertIsNone(analytics["lifestyle_summary"]["average_sleep_hours"])
        self.assertEqual(analytics["data_quality"]["missing_date_count"], 7)


if __name__ == "__main__":
    unittest.main()
