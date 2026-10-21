from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "data" / "generate_demo_data.py"
SPEC = importlib.util.spec_from_file_location("generate_demo_data", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class PhaseOneDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.today = date(2026, 10, 15)
        self.tables = generator.build_demo_data(self.today)

    def test_expected_demo_record_counts(self) -> None:
        self.assertEqual(len(self.tables["patients.csv"]), 10)
        self.assertEqual(len(self.tables["meals.csv"]), 210)
        self.assertEqual(len(self.tables["glucose.csv"]), 140)
        self.assertEqual(len(self.tables["lifestyle.csv"]), 70)
        self.assertGreaterEqual(len(self.tables["food_nutrition.csv"]), 30)

    def test_all_records_are_demo_and_patient_references_are_valid(self) -> None:
        patients = self.tables["patients.csv"]
        patient_ids = {row["patient_id"] for row in patients}
        for rows in self.tables.values():
            self.assertTrue(all(row["is_demo"] is True for row in rows))
        for table in ("meals.csv", "glucose.csv", "lifestyle.csv"):
            self.assertTrue(all(row["patient_id"] in patient_ids for row in self.tables[table]))

    def test_each_patient_has_seven_days_and_expected_meals(self) -> None:
        for patient in self.tables["patients.csv"]:
            patient_id = patient["patient_id"]
            for table, expected_count in (("meals.csv", 21), ("glucose.csv", 14), ("lifestyle.csv", 7)):
                records = [row for row in self.tables[table] if row["patient_id"] == patient_id]
                self.assertEqual(len(records), expected_count)
                self.assertEqual(
                    {row["date"] for row in records},
                    {(self.today.fromordinal(self.today.toordinal() - offset)).isoformat() for offset in range(7)},
                )

    def test_every_meal_food_exists_in_reference_table(self) -> None:
        known_foods = {row["food_name"] for row in self.tables["food_nutrition.csv"]}
        known_foods.add("Dal")
        for meal in self.tables["meals.csv"]:
            self.assertTrue(set(meal["food_items"].split("; ")).issubset(known_foods))

    def test_csvs_are_written_with_headers_and_reloadable(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as temporary_directory:
            generator.write_demo_data(Path(temporary_directory), self.today)
            for filename, rows in self.tables.items():
                with (Path(temporary_directory) / filename).open(encoding="utf-8-sig", newline="") as csv_file:
                    written = list(csv.DictReader(csv_file))
                self.assertEqual(len(written), len(rows), filename)
                self.assertIn("is_demo", written[0], filename)


if __name__ == "__main__":
    unittest.main()
