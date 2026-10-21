from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.analytics.glucose_analytics import (
    get_daily_glucose_statistics,
    get_glucose_by_context,
    get_glucose_reading_count,
    get_glucose_summary,
    get_glucose_trend,
    get_meal_glucose_pairs,
)
from src.database.database import initialize_database


ROOT = Path(__file__).resolve().parents[1]
CSV_FILENAMES = ("patients.csv", "meals.csv", "glucose.csv", "lifestyle.csv", "food_nutrition.csv")


class GlucoseAnalyticsTests(unittest.TestCase):
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
            "SELECT MIN(date), MAX(date) FROM glucose WHERE patient_id = 'P001'"
        ).fetchone()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_summary_count_trend_and_context_stats(self) -> None:
        summary = get_glucose_summary(
            "P001", self.start_date, self.end_date, database_path=self.database_path
        )
        self.assertEqual(summary["reading_count"], 14)
        self.assertIsNotNone(summary["average_recorded_glucose"])
        self.assertLessEqual(summary["minimum_recorded_glucose"], summary["maximum_recorded_glucose"])
        self.assertEqual(get_glucose_reading_count("P001", database_path=self.database_path), 14)
        self.assertEqual(len(get_glucose_trend("P001", database_path=self.database_path)), 14)
        self.assertEqual(len(get_daily_glucose_statistics("P001", database_path=self.database_path)), 7)
        self.assertEqual(
            {row["context"] for row in get_glucose_by_context("P001", database_path=self.database_path)},
            {"After meal", "Fasting"},
        )
        self.assertEqual(get_glucose_summary("INVALID", database_path=self.database_path)["reading_count"], 0)

    def test_one_reading_has_no_standard_deviation(self) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """
            DELETE FROM glucose WHERE patient_id = ?
            AND reading_id != (
                SELECT reading_id FROM glucose WHERE patient_id = ? ORDER BY date, time LIMIT 1
            )
            """,
            ("P001", "P001"),
        )
        connection.commit()
        connection.close()
        summary = get_glucose_summary("P001", database_path=self.database_path)
        self.assertEqual(summary["reading_count"], 1)
        self.assertIsNone(summary["standard_deviation_recorded_glucose"])

    def test_empty_history_and_association_requires_meal_timestamps(self) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute("DELETE FROM glucose WHERE patient_id = 'P001'")
        connection.execute("DELETE FROM meals WHERE patient_id = 'P001'")
        connection.commit()
        connection.close()
        summary = get_glucose_summary("P001", database_path=self.database_path)
        self.assertEqual(summary["reading_count"], 0)
        self.assertIsNone(summary["average_recorded_glucose"])
        self.assertEqual(get_meal_glucose_pairs("P001", database_path=self.database_path), [])

    def test_meal_glucose_association_matches_configured_window(self) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "UPDATE meals SET meal_time = '12:00' WHERE patient_id = 'P001' AND date = ? AND meal_type = 'Lunch'",
            (self.end_date,),
        )
        connection.execute(
            """
            INSERT INTO glucose (reading_id, patient_id, date, time, glucose_value, context, is_demo)
            VALUES ('TEST-NEAR', 'P001', ?, '13:00', 121, 'After meal', 1)
            """,
            (self.end_date,),
        )
        connection.commit()
        connection.close()
        pairs = get_meal_glucose_pairs(
            "P001",
            self.end_date,
            self.end_date,
            window_minutes=90,
            database_path=self.database_path,
        )
        lunch_pair = next(pair for pair in pairs if pair["meal_type"] == "Lunch")
        self.assertEqual(lunch_pair["after_meal_readings"][0]["glucose_value"], 121)
        self.assertIn("does not establish causation", lunch_pair["label"])


if __name__ == "__main__":
    unittest.main()
