from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from src.database.database import (
    DataValidationError,
    get_all_food_information,
    get_all_patients,
    get_food_information,
    get_patient,
    get_patient_glucose,
    get_patient_lifestyle,
    get_patient_meals,
    get_recent_glucose,
    get_recent_meals,
    initialize_database,
)
from src.patient_profile.context import build_patient_context
from src.patient_profile.profile import generate_patient_summary, get_patient_profile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data"
CSV_FILENAMES = (
    "patients.csv",
    "meals.csv",
    "glucose.csv",
    "lifestyle.csv",
    "food_nutrition.csv",
)


class PhaseTwoDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        for filename in CSV_FILENAMES:
            shutil.copyfile(SOURCE_DATA / filename, self.data_dir / filename)
        self.database_path = self.root / "database" / "test.sqlite3"
        self.assertTrue(initialize_database(self.data_dir, self.database_path))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_get_patient_and_invalid_patient(self) -> None:
        patient = get_patient("P001", self.database_path)
        self.assertIsNotNone(patient)
        assert patient is not None
        self.assertEqual(patient["patient_id"], "P001")
        self.assertEqual(patient["allergies"], "None")
        self.assertIsNone(get_patient("INVALID", self.database_path))
        self.assertIsNone(get_patient("", self.database_path))

    def test_patient_history_queries_and_food_lookup(self) -> None:
        self.assertEqual(len(get_all_patients(self.database_path)), 10)
        meals = get_patient_meals("P001", self.database_path)
        glucose = get_patient_glucose("P001", self.database_path)
        lifestyle = get_patient_lifestyle("P001", self.database_path)
        self.assertEqual(len(meals), 21)
        self.assertEqual(len(glucose), 14)
        self.assertEqual(len(lifestyle), 7)
        self.assertIn("food", meals[0])
        self.assertIn("portion", meals[0])
        self.assertEqual(get_patient_meals("INVALID", self.database_path), [])
        food = get_food_information("puttu", self.database_path)
        self.assertIsNotNone(food)
        assert food is not None
        self.assertEqual(food["food_name"], "Puttu")
        self.assertIsNone(get_food_information("Not a food", self.database_path))
        self.assertEqual(len(get_all_food_information(self.database_path)), 34)

    def test_recent_history_and_invalid_day_window(self) -> None:
        self.assertEqual(len(get_recent_meals("P001", 3, self.database_path)), 9)
        self.assertEqual(len(get_recent_glucose("P001", 3, self.database_path)), 6)
        self.assertEqual(len(get_recent_meals("P001", 20, self.database_path)), 21)
        with self.assertRaises(ValueError):
            get_recent_meals("P001", 0, self.database_path)
        with self.assertRaises(ValueError):
            get_recent_glucose("P001", True, self.database_path)

    def test_summary_and_profile_context(self) -> None:
        summary = generate_patient_summary("P001", self.database_path)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["meal_count"], 21)
        self.assertEqual(summary["glucose_reading_count"], 14)
        self.assertEqual(summary["lifestyle_record_count"], 7)
        self.assertIsInstance(summary["average_recorded_glucose_mg_dl"], float)
        self.assertTrue(summary["most_frequently_recorded_foods"])
        self.assertIsNone(generate_patient_summary("INVALID", self.database_path))

        profile = get_patient_profile("P001", database_path=self.database_path)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["patient"]["patient_id"], "P001")
        self.assertEqual(len(profile["recent_meals"]), 21)
        self.assertEqual(len(profile["glucose_history"]), 14)
        self.assertEqual(len(profile["lifestyle_history"]), 7)

        context = build_patient_context("P001", database_path=self.database_path)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context["patient_profile"]["patient_id"], "P001")
        self.assertEqual(context["dietary_preferences"]["cuisine_preference"], "Kerala")
        self.assertIn("recent_glucose_history", context)
        self.assertIsNone(build_patient_context("INVALID", database_path=self.database_path))

    def test_missing_history_is_returned_as_empty_not_fabricated(self) -> None:
        import sqlite3

        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("DELETE FROM meals WHERE patient_id = ?", ("P001",))
            connection.execute("DELETE FROM glucose WHERE patient_id = ?", ("P001",))
            connection.execute("DELETE FROM lifestyle WHERE patient_id = ?", ("P001",))
            connection.commit()
        finally:
            connection.close()
        self.assertEqual(get_patient_meals("P001", self.database_path), [])
        self.assertEqual(get_patient_glucose("P001", self.database_path), [])
        self.assertEqual(get_patient_lifestyle("P001", self.database_path), [])
        summary = generate_patient_summary("P001", self.database_path)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIsNone(summary["average_recorded_glucose_mg_dl"])
        self.assertIsNone(summary["average_sleep_hours"])
        self.assertEqual(summary["most_frequently_recorded_foods"], [])

    def test_csv_import_is_skipped_when_unchanged_and_reloaded_when_changed(self) -> None:
        self.assertFalse(initialize_database(self.data_dir, self.database_path))
        patients_path = self.data_dir / "patients.csv"
        with patients_path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        rows[0]["health_goals"] = "Updated test goal"
        with patients_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.assertTrue(initialize_database(self.data_dir, self.database_path))
        patient = get_patient("P001", self.database_path)
        assert patient is not None
        self.assertEqual(patient["health_goals"], "Updated test goal")

    def test_invalid_csv_values_and_references_are_rejected_without_import(self) -> None:
        meals_path = self.data_dir / "meals.csv"
        with meals_path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
            fieldnames = list(rows[0])
        rows[0]["date"] = "not-a-date"
        with meals_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(DataValidationError, "date must use YYYY-MM-DD"):
            initialize_database(self.data_dir, self.database_path)
        self.assertEqual(len(get_patient_meals("P001", self.database_path)), 21)

        rows[0]["date"] = "2026-01-01"
        rows[0]["patient_id"] = "UNKNOWN"
        with meals_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(DataValidationError, "does not exist in patients.csv"):
            initialize_database(self.data_dir, self.database_path)

    def test_invalid_numeric_values_and_duplicate_ids_are_rejected(self) -> None:
        glucose_path = self.data_dir / "glucose.csv"
        with glucose_path.open(encoding="utf-8-sig", newline="") as source:
            glucose_rows = list(csv.DictReader(source))
            glucose_fields = list(glucose_rows[0])
        glucose_rows[0]["glucose_mg_dl"] = "not-a-number"
        with glucose_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=glucose_fields)
            writer.writeheader()
            writer.writerows(glucose_rows)
        with self.assertRaisesRegex(DataValidationError, "glucose_mg_dl must be a valid number"):
            initialize_database(self.data_dir, self.database_path)

        patient_path = self.data_dir / "patients.csv"
        with patient_path.open(encoding="utf-8-sig", newline="") as source:
            patient_rows = list(csv.DictReader(source))
            patient_fields = list(patient_rows[0])
        patient_rows[1]["patient_id"] = patient_rows[0]["patient_id"]
        with patient_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=patient_fields)
            writer.writeheader()
            writer.writerows(patient_rows)
        with self.assertRaisesRegex(DataValidationError, "duplicate patient_id"):
            initialize_database(self.data_dir, self.database_path)

    def test_missing_csv_file_has_a_controlled_validation_error(self) -> None:
        (self.data_dir / "glucose.csv").unlink()
        with self.assertRaisesRegex(DataValidationError, "Required CSV file is missing"):
            initialize_database(self.data_dir, self.database_path)

    def test_optional_meal_time_is_imported_when_csv_provides_it(self) -> None:
        meals_path = self.data_dir / "meals.csv"
        with meals_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = [*reader.fieldnames, "meal_time"]
            rows = list(reader)
        rows[0]["meal_time"] = "08:15"
        with meals_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.assertTrue(initialize_database(self.data_dir, self.database_path))
        meals = get_patient_meals("P001", self.database_path)
        recorded_time = next(meal["meal_time"] for meal in meals if meal["meal_id"] == rows[0]["meal_id"])
        self.assertEqual(recorded_time, "08:15")


if __name__ == "__main__":
    unittest.main()
