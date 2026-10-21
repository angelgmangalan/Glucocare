from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.analytics.lifestyle_analytics import (
    get_exercise_summary,
    get_lifestyle_summary,
    get_sleep_summary,
    get_stress_summary,
    get_water_summary,
)
from src.database.database import initialize_database


ROOT = Path(__file__).resolve().parents[1]
CSV_FILENAMES = ("patients.csv", "meals.csv", "glucose.csv", "lifestyle.csv", "food_nutrition.csv")


class LifestyleAnalyticsTests(unittest.TestCase):
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
            "SELECT MIN(date), MAX(date) FROM lifestyle WHERE patient_id = 'P001'"
        ).fetchone()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_lifestyle_averages_and_filtered_window(self) -> None:
        summary = get_lifestyle_summary("P001", database_path=self.database_path)
        self.assertEqual(summary["record_count"], 7)
        self.assertIsNotNone(summary["average_sleep_hours"])
        self.assertIsNotNone(summary["average_exercise_minutes"])
        self.assertIsNotNone(summary["average_water_intake"])
        self.assertGreaterEqual(summary["average_stress_score"], 1)
        one_day = get_sleep_summary(
            "P001", self.start_date, self.start_date, database_path=self.database_path
        )
        self.assertEqual(one_day["record_count"], 1)
        self.assertEqual(get_exercise_summary("INVALID", database_path=self.database_path)["record_count"], 0)
        self.assertEqual(get_water_summary("INVALID", database_path=self.database_path)["average"], None)
        self.assertEqual(get_stress_summary("INVALID", database_path=self.database_path)["record_count"], 0)

    def test_empty_lifestyle_history_has_no_averages(self) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute("DELETE FROM lifestyle WHERE patient_id = 'P001'")
        connection.commit()
        connection.close()
        summary = get_lifestyle_summary("P001", database_path=self.database_path)
        self.assertEqual(summary["record_count"], 0)
        self.assertIsNone(summary["average_sleep_hours"])
        self.assertIsNone(summary["average_stress_score"])


if __name__ == "__main__":
    unittest.main()
