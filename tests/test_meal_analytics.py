from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.analytics.meal_analytics import (
    get_average_meal_nutrition,
    get_daily_nutrition_summary,
    get_food_frequency,
    get_meal_count,
    get_meal_type_distribution,
    get_recent_meals,
    get_meal_timing_summary,
)
from src.database.database import initialize_database


ROOT = Path(__file__).resolve().parents[1]
CSV_FILENAMES = ("patients.csv", "meals.csv", "glucose.csv", "lifestyle.csv", "food_nutrition.csv")


class MealAnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.data_dir = root / "data"
        self.data_dir.mkdir()
        for filename in CSV_FILENAMES:
            shutil.copyfile(ROOT / "data" / filename, self.data_dir / filename)
        self.database_path = root / "glucocare.sqlite3"
        initialize_database(self.data_dir, self.database_path)
        connection = sqlite3.connect(self.database_path)
        self.start_date, self.end_date = connection.execute(
            "SELECT MIN(date), MAX(date) FROM meals WHERE patient_id = 'P001'"
        ).fetchone()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_meal_counts_and_date_filtering(self) -> None:
        self.assertEqual(get_meal_count("P001", database_path=self.database_path), 21)
        self.assertEqual(
            get_meal_count("P001", self.start_date, self.start_date, database_path=self.database_path),
            3,
        )
        self.assertEqual(get_meal_count("INVALID", database_path=self.database_path), 0)
        self.assertEqual(
            len(get_recent_meals("P001", self.start_date, self.start_date, database_path=self.database_path)),
            3,
        )
        self.assertEqual(
            get_meal_type_distribution("P001", self.start_date, self.start_date, database_path=self.database_path),
            {"Breakfast": 1, "Dinner": 1, "Lunch": 1},
        )

    def test_food_frequency_and_recorded_meal_nutrition(self) -> None:
        frequency = get_food_frequency(
            "P001", self.start_date, self.start_date, database_path=self.database_path
        )
        meals = get_recent_meals("P001", self.start_date, self.start_date, database_path=self.database_path)
        self.assertEqual(
            sum(row["count"] for row in frequency),
            sum(len(meal["food"].split(";")) for meal in meals),
        )
        nutrition = get_average_meal_nutrition(
            "P001", self.start_date, self.start_date, database_path=self.database_path
        )
        self.assertEqual(nutrition["recorded_meal_count"], 3)
        self.assertAlmostEqual(
            nutrition["calories"],
            sum(meal["calories"] for meal in meals) / len(meals),
        )
        self.assertIn("recorded meals", nutrition["label"])
        self.assertEqual(nutrition["recorded_day_count"], 1)
        self.assertEqual(
            nutrition["average_per_recorded_day"]["calories"],
            sum(meal["calories"] for meal in meals),
        )
        daily = get_daily_nutrition_summary(
            "P001", self.start_date, self.start_date, database_path=self.database_path
        )
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily[0]["meal_count"], 3)
        timing = get_meal_timing_summary(
            "P001", self.start_date, self.start_date, database_path=self.database_path
        )
        self.assertEqual(timing["timed_meal_count"], 0)
        self.assertIn("unavailable", timing["status"])

    def test_timing_summary_only_uses_recorded_times(self) -> None:
        import sqlite3

        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "UPDATE meals SET meal_time = CASE meal_type "
            "WHEN 'Breakfast' THEN '08:00' WHEN 'Lunch' THEN '13:00' ELSE '19:00' END "
            "WHERE patient_id = 'P001' AND date = ?",
            (self.start_date,),
        )
        connection.commit()
        connection.close()
        timing = get_meal_timing_summary(
            "P001", self.start_date, self.start_date, database_path=self.database_path
        )
        self.assertEqual(timing["timed_meal_count"], 3)
        self.assertEqual(timing["average_time_by_meal_type"]["Breakfast"], "08:00")
        self.assertEqual(timing["average_gap_between_recorded_meals_minutes"], 330)

    def test_invalid_date_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_meal_count("P001", self.end_date, self.start_date, database_path=self.database_path)


if __name__ == "__main__":
    unittest.main()
