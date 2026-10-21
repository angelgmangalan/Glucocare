from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from datetime import date, time
from pathlib import Path

from src.analytics.meal_analytics import get_daily_nutrition_summary
from src.database.database import (
    DEFAULT_DATA_DIR,
    DatabaseError,
    get_patient_meals,
    initialize_database,
)
from src.food_vision.service import save_confirmed_meal


class FoodVisionPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data_dir = self.root / "source"
        self.data_dir.mkdir()
        for filename in (
            "patients.csv", "meals.csv", "glucose.csv", "lifestyle.csv", "food_nutrition.csv"
        ):
            shutil.copyfile(DEFAULT_DATA_DIR / filename, self.data_dir / filename)
        self.database_path = self.root / "glucocare.sqlite3"
        initialize_database(self.data_dir, self.database_path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_explicit_save_reuses_history_and_preserves_unknown_nutrition(self) -> None:
        before = len(get_patient_meals("P001", self.database_path))
        save_confirmed_meal(
            patient_id="P001",
            meal_date=date(2026, 10, 19),
            meal_time=time(12, 30),
            meal_type="Lunch",
            items=[{
                "label": "Unknown curry",
                "original_prediction": "Likely vegetable curry",
                "match_choice": "unmatched",
                "include": True,
                "food_id": None,
                "serving_quantity": 1,
            }],
            submission_id="vision-test-success",
            database_path=self.database_path,
        )
        meals = get_patient_meals("P001", self.database_path)
        self.assertEqual(len(meals), before + 1)
        saved = next(meal for meal in meals if meal["meal_id"] == "vision-test-success")
        self.assertEqual(saved["food"], "Unknown curry")
        self.assertIsNone(saved["calories"])
        self.assertIsNone(saved["carbohydrates"])
        connection = sqlite3.connect(self.database_path)
        try:
            details = connection.execute(
                """
                SELECT original_prediction, corrected_label, selected_food_id,
                       identification_source, serving_quantity, quantity_source, nutrition_status
                FROM food_image_meal_items WHERE meal_id = ?
                """,
                ("vision-test-success",),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(
            details,
            ("Likely vegetable curry", "Unknown curry", "vision", None, "unknown", "unmatched"),
        )
        daily = get_daily_nutrition_summary(
            "P001", "2026-10-19", "2026-10-19", database_path=self.database_path
        )
        self.assertEqual(len(daily), 1)
        self.assertIsNone(daily[0]["calories"])

    def test_database_match_must_exist_and_patient_is_valid(self) -> None:
        for patient_id, food_id in (("BAD", None), ("P001", "bad-food")):
            with self.subTest(patient_id=patient_id, food_id=food_id):
                match_choice = "matched" if food_id else "unmatched"
                with self.assertRaises(ValueError):
                    save_confirmed_meal(
                        patient_id=patient_id,
                        meal_date=date(2026, 10, 19),
                        meal_time=time(12),
                        meal_type="Lunch",
                        items=[{
                            "label": "Food", "match_choice": match_choice, "include": True,
                            "food_id": food_id, "quantity_confirmed": bool(food_id),
                            "serving_quantity": 1,
                        }],
                        submission_id=f"bad-{patient_id}-{food_id}",
                        database_path=self.database_path,
                    )

    def test_duplicate_confirmation_is_rejected_without_duplicate_meal(self) -> None:
        options = {
            "patient_id": "P001",
            "meal_date": date(2026, 10, 19),
            "meal_time": time(12),
            "meal_type": "Lunch",
            "items": [{
                "label": "Rice", "match_choice": "unmatched", "include": True,
                "food_id": None, "serving_quantity": 1,
            }],
            "submission_id": "vision-once",
            "database_path": self.database_path,
        }
        save_confirmed_meal(**options)
        with self.assertRaisesRegex(DatabaseError, "already submitted"):
            save_confirmed_meal(**options)
        self.assertEqual(
            sum(meal["meal_id"] == "vision-once" for meal in get_patient_meals("P001", self.database_path)),
            1,
        )

    def test_failed_save_rolls_back_without_a_partial_row(self) -> None:
        with self.assertRaises(ValueError):
            save_confirmed_meal(
                patient_id="P001",
                meal_date=date(2026, 10, 19),
                meal_time=time(12),
                meal_type="Lunch",
                items=[{
                    "label": "Rice", "match_choice": "matched", "quantity_confirmed": True,
                        "include": True, "identification_source": "invalid",
                        "food_id": None, "match_choice": "unmatched",
                        "serving_quantity": -1,
                }],
                submission_id="vision-rollback",
                database_path=self.database_path,
            )
        self.assertFalse(any(
            meal["meal_id"] == "vision-rollback"
            for meal in get_patient_meals("P001", self.database_path)
        ))

    def test_unverified_match_preserves_selection_and_confirmed_portion_without_nutrients(self) -> None:
        save_confirmed_meal(
            patient_id="P001",
            meal_date=date(2026, 10, 19),
            meal_time=time(12),
            meal_type="Lunch",
            items=[{
                "label": "Puttu",
                "original_prediction": "Puttu-like food",
                "identification_source": "vision",
                "match_choice": "matched",
                "include": True,
                "food_id": "F001",
                "quantity_confirmed": True,
                "serving_quantity": 2,
            }],
            submission_id="vision-unverified",
            database_path=self.database_path,
        )
        saved = next(
            meal for meal in get_patient_meals("P001", self.database_path)
            if meal["meal_id"] == "vision-unverified"
        )
        self.assertIsNone(saved["calories"])
        connection = sqlite3.connect(self.database_path)
        try:
            details = connection.execute(
                """
                SELECT selected_food_id, serving_quantity, quantity_source, nutrition_status
                FROM food_image_meal_items WHERE meal_id = ?
                """,
                ("vision-unverified",),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(details, ("F001", 2.0, "user_confirmed", "unverified"))

    def test_verified_source_and_confirmed_servings_are_scaled_before_save(self) -> None:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                "UPDATE food_nutrition SET is_demo = 0, source = ? WHERE food_id = ?",
                ("Verified source for deterministic test", "F001"),
            )
            connection.commit()
        finally:
            connection.close()
        save_confirmed_meal(
            patient_id="P001",
            meal_date=date(2026, 10, 19),
            meal_time=time(12),
            meal_type="Lunch",
            items=[{
                "label": "Puttu",
                "original_prediction": "Puttu",
                "match_choice": "matched",
                "include": True,
                "food_id": "F001",
                "quantity_confirmed": True,
                "serving_quantity": 2,
            }],
            submission_id="vision-verified",
            database_path=self.database_path,
        )
        saved = next(
            meal for meal in get_patient_meals("P001", self.database_path)
            if meal["meal_id"] == "vision-verified"
        )
        self.assertEqual(saved["calories"], 480)
        self.assertEqual(saved["carbohydrates"], 94)
        connection = sqlite3.connect(self.database_path)
        try:
            status = connection.execute(
                "SELECT nutrition_status FROM food_image_meal_items WHERE meal_id = ?",
                ("vision-verified",),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(status, "verified")

    def test_initialize_migrates_historical_not_null_meal_nutrition(self) -> None:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("DROP INDEX idx_meals_patient_date")
            connection.execute("ALTER TABLE meals RENAME TO meals_new_schema")
            connection.execute(
                """
                CREATE TABLE meals (
                    meal_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                    date TEXT NOT NULL, meal_type TEXT NOT NULL, meal_time TEXT,
                    food TEXT NOT NULL, portion TEXT NOT NULL,
                    calories REAL NOT NULL, carbohydrates REAL NOT NULL, protein REAL NOT NULL,
                    fat REAL NOT NULL, fiber REAL NOT NULL,
                    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
                )
                """
            )
            columns = (
                "meal_id", "patient_id", "date", "meal_type", "meal_time", "food", "portion",
                "calories", "carbohydrates", "protein", "fat", "fiber", "is_demo",
            )
            column_list = ", ".join(columns)
            connection.execute(
                f"INSERT INTO meals ({column_list}) SELECT {column_list} FROM meals_new_schema"
            )
            connection.execute("DROP TABLE meals_new_schema")
            connection.execute(
                "CREATE INDEX idx_meals_patient_date ON meals(patient_id, date)"
            )
            connection.commit()
        finally:
            connection.close()

        initialize_database(self.data_dir, self.database_path)
        connection = sqlite3.connect(self.database_path)
        try:
            schema = {
                row[1]: row for row in connection.execute("PRAGMA table_info(meals)")
            }
            self.assertEqual(schema["calories"][3], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM meals").fetchone()[0], 210)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
