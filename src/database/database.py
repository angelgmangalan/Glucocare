"""SQLite database and validated CSV import for GlucoCare."""

from __future__ import annotations

import csv
import hashlib
import math
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATABASE_PATH = DEFAULT_DATA_DIR / "glucocare.sqlite3"

CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "patients.csv": (
        "patient_id", "age", "gender", "diabetes_type", "diet_preference",
        "cuisine_preference", "activity_level", "allergies_restrictions",
        "health_goals", "lifestyle_notes", "is_demo",
    ),
    "meals.csv": (
        "meal_id", "patient_id", "date", "meal_type", "food_items",
        "portion_size", "calories", "carbohydrates_g", "protein_g", "fat_g",
        "fiber_g", "is_demo",
    ),
    "glucose.csv": (
        "glucose_id", "patient_id", "date", "time", "glucose_mg_dl",
        "context", "is_demo",
    ),
    "lifestyle.csv": (
        "lifestyle_id", "patient_id", "date", "sleep_hours",
        "exercise_minutes", "water_liters", "stress_level", "is_demo",
    ),
    "food_nutrition.csv": (
        "food_id", "food_name", "regional_name", "category", "serving_size",
        "calories", "carbohydrates_g", "protein_g", "fat_g", "fiber_g",
        "source", "is_demo",
    ),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL CHECK (age > 0),
    gender TEXT NOT NULL,
    diabetes_type TEXT NOT NULL,
    diet_preference TEXT NOT NULL,
    cuisine_preference TEXT NOT NULL,
    activity_level TEXT NOT NULL,
    allergies TEXT NOT NULL DEFAULT '',
    health_goals TEXT NOT NULL DEFAULT '',
    lifestyle_notes TEXT NOT NULL DEFAULT '',
    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
);

CREATE TABLE IF NOT EXISTS meals (
    meal_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    meal_type TEXT NOT NULL,
    meal_time TEXT,
    food TEXT NOT NULL,
    portion TEXT NOT NULL,
    calories REAL CHECK (calories >= 0),
    carbohydrates REAL CHECK (carbohydrates >= 0),
    protein REAL CHECK (protein >= 0),
    fat REAL CHECK (fat >= 0),
    fiber REAL CHECK (fiber >= 0),
    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
);

CREATE TABLE IF NOT EXISTS glucose (
    reading_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    glucose_value REAL NOT NULL CHECK (glucose_value > 0),
    context TEXT NOT NULL,
    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
);

CREATE TABLE IF NOT EXISTS lifestyle (
    lifestyle_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    sleep_hours REAL NOT NULL CHECK (sleep_hours >= 0),
    exercise_minutes REAL NOT NULL CHECK (exercise_minutes >= 0),
    water_intake REAL NOT NULL CHECK (water_intake >= 0),
    stress_level TEXT NOT NULL,
    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
);

CREATE TABLE IF NOT EXISTS food_nutrition (
    food_id TEXT PRIMARY KEY,
    food_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    regional_name TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    serving_size TEXT NOT NULL,
    calories REAL NOT NULL CHECK (calories >= 0),
    carbohydrates REAL NOT NULL CHECK (carbohydrates >= 0),
    protein REAL NOT NULL CHECK (protein >= 0),
    fat REAL NOT NULL CHECK (fat >= 0),
    fiber REAL NOT NULL CHECK (fiber >= 0),
    source TEXT NOT NULL,
    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
);

CREATE TABLE IF NOT EXISTS food_image_meal_items (
    meal_id TEXT NOT NULL REFERENCES meals(meal_id) ON DELETE CASCADE,
    item_index INTEGER NOT NULL CHECK (item_index >= 0),
    original_prediction TEXT NOT NULL,
    corrected_label TEXT NOT NULL,
    identification_source TEXT NOT NULL CHECK (identification_source IN ('vision', 'user')),
    selected_food_id TEXT REFERENCES food_nutrition(food_id) ON DELETE SET NULL,
    serving_quantity REAL CHECK (serving_quantity > 0),
    quantity_source TEXT NOT NULL CHECK (quantity_source IN ('user_confirmed', 'unknown')),
    nutrition_status TEXT NOT NULL CHECK (nutrition_status IN ('verified', 'unverified', 'unmatched')),
    PRIMARY KEY (meal_id, item_index)
);

CREATE TABLE IF NOT EXISTS import_metadata (
    filename TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_meals_patient_date ON meals(patient_id, date);
CREATE INDEX IF NOT EXISTS idx_glucose_patient_date ON glucose(patient_id, date);
CREATE INDEX IF NOT EXISTS idx_lifestyle_patient_date ON lifestyle(patient_id, date);
"""


class DatabaseError(RuntimeError):
    """Raised when database initialization or access cannot be completed."""


class DataValidationError(ValueError):
    """Raised when source CSV data does not satisfy the expected schema."""


def _connect(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    path = Path(database_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    except (OSError, sqlite3.Error) as error:
        raise DatabaseError(f"Unable to open GlucoCare database at {path}: {error}") from error


def _required_text(row: dict[str, str], column: str, filename: str, row_number: int) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise DataValidationError(f"{filename}, row {row_number}: {column} must not be empty.")
    return value


def _number(
    row: dict[str, str],
    column: str,
    filename: str,
    row_number: int,
    *,
    integer: bool = False,
    positive: bool = False,
) -> int | float:
    raw_value = (row.get(column) or "").strip()
    try:
        value = int(raw_value) if integer else float(raw_value)
    except ValueError as error:
        raise DataValidationError(
            f"{filename}, row {row_number}: {column} must be a valid "
            f"{'integer' if integer else 'number'}."
        ) from error
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        constraint = "greater than zero" if positive else "zero or greater"
        raise DataValidationError(f"{filename}, row {row_number}: {column} must be {constraint}.")
    return value


def _iso_date(value: str, filename: str, row_number: int) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise DataValidationError(
            f"{filename}, row {row_number}: date must use YYYY-MM-DD format."
        ) from error
    if parsed.isoformat() != value:
        raise DataValidationError(
            f"{filename}, row {row_number}: date must use YYYY-MM-DD format."
        )
    return value


def _demo_flag(row: dict[str, str], filename: str, row_number: int) -> int:
    value = (row.get("is_demo") or "").strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise DataValidationError(f"{filename}, row {row_number}: is_demo must be True or False.")
    return int(value in {"true", "1"})


def _read_csv(data_dir: Path, filename: str) -> tuple[list[dict[str, Any]], str]:
    path = data_dir / filename
    if not path.is_file():
        raise DataValidationError(f"Required CSV file is missing: {path}")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DataValidationError(f"Unable to read {path}: {error}") from error
    digest = hashlib.sha256(content).hexdigest()

    import csv
    import io

    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        headers = set(reader.fieldnames or [])
        missing = set(CSV_COLUMNS[filename]).difference(headers)
        if missing:
            raise DataValidationError(
                f"{filename} is missing required columns: {', '.join(sorted(missing))}."
            )
        raw_rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise DataValidationError(f"Unable to parse {filename}: {error}") from error
    if any(None in row for row in raw_rows):
        raise DataValidationError(f"{filename} contains a row with more fields than its header.")
    return raw_rows, digest


def _validated_source(data_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    raw: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    for filename in CSV_COLUMNS:
        raw[filename], hashes[filename] = _read_csv(data_dir, filename)

    primary_keys = {
        "patients.csv": "patient_id",
        "meals.csv": "meal_id",
        "glucose.csv": "glucose_id",
        "lifestyle.csv": "lifestyle_id",
        "food_nutrition.csv": "food_id",
    }
    for filename, key in primary_keys.items():
        seen: set[str] = set()
        for row_number, row in enumerate(raw[filename], start=2):
            value = _required_text(row, key, filename, row_number)
            if value in seen:
                raise DataValidationError(f"{filename}, row {row_number}: duplicate {key} {value!r}.")
            seen.add(value)

    patient_ids = {row["patient_id"].strip() for row in raw["patients.csv"]}
    transformed: dict[str, list[dict[str, Any]]] = {
        "patients": [],
        "meals": [],
        "glucose": [],
        "lifestyle": [],
        "food_nutrition": [],
    }

    for row_number, row in enumerate(raw["patients.csv"], start=2):
        transformed["patients"].append({
            "patient_id": _required_text(row, "patient_id", "patients.csv", row_number),
            "age": _number(row, "age", "patients.csv", row_number, integer=True, positive=True),
            "gender": _required_text(row, "gender", "patients.csv", row_number),
            "diabetes_type": _required_text(row, "diabetes_type", "patients.csv", row_number),
            "diet_preference": _required_text(row, "diet_preference", "patients.csv", row_number),
            "cuisine_preference": _required_text(row, "cuisine_preference", "patients.csv", row_number),
            "activity_level": _required_text(row, "activity_level", "patients.csv", row_number),
            "allergies": (row.get("allergies_restrictions") or "").strip(),
            "health_goals": (row.get("health_goals") or "").strip(),
            "lifestyle_notes": (row.get("lifestyle_notes") or "").strip(),
            "is_demo": _demo_flag(row, "patients.csv", row_number),
        })

    related = (
        ("meals.csv", "meal_id", "meals"),
        ("glucose.csv", "glucose_id", "glucose"),
        ("lifestyle.csv", "lifestyle_id", "lifestyle"),
    )
    for filename, id_column, table in related:
        for row_number, row in enumerate(raw[filename], start=2):
            patient_id = _required_text(row, "patient_id", filename, row_number)
            if patient_id not in patient_ids:
                raise DataValidationError(
                    f"{filename}, row {row_number}: patient_id {patient_id!r} "
                    "does not exist in patients.csv."
                )
            record: dict[str, Any] = {
                "patient_id": patient_id,
                "date": _iso_date(_required_text(row, "date", filename, row_number), filename, row_number),
                "is_demo": _demo_flag(row, filename, row_number),
            }
            record[id_column if table != "glucose" else "reading_id"] = _required_text(
                row, id_column, filename, row_number
            )
            if table == "meals":
                meal_time = (row.get("meal_time") or "").strip()
                if meal_time:
                    try:
                        if time.fromisoformat(meal_time).isoformat(timespec="minutes") != meal_time:
                            raise ValueError
                    except ValueError as error:
                        raise DataValidationError(
                            f"{filename}, row {row_number}: meal_time must use HH:MM format."
                        ) from error
                record["meal_time"] = meal_time or None
                for column in ("meal_type", "food_items", "portion_size"):
                    record[{"food_items": "food", "portion_size": "portion"}.get(column, column)] = (
                        _required_text(row, column, filename, row_number)
                    )
                for source, target in (
                    ("calories", "calories"),
                    ("carbohydrates_g", "carbohydrates"),
                    ("protein_g", "protein"),
                    ("fat_g", "fat"),
                    ("fiber_g", "fiber"),
                ):
                    record[target] = _number(row, source, filename, row_number)
            elif table == "glucose":
                time_text = _required_text(row, "time", filename, row_number)
                try:
                    if time.fromisoformat(time_text).isoformat(timespec="minutes") != time_text:
                        raise ValueError
                except ValueError as error:
                    raise DataValidationError(
                        f"{filename}, row {row_number}: time must use HH:MM format."
                    ) from error
                record["time"] = time_text
                record["glucose_value"] = _number(
                    row, "glucose_mg_dl", filename, row_number, positive=True
                )
                record["context"] = _required_text(row, "context", filename, row_number)
            else:
                for source, target in (
                    ("sleep_hours", "sleep_hours"),
                    ("exercise_minutes", "exercise_minutes"),
                    ("water_liters", "water_intake"),
                ):
                    record[target] = _number(row, source, filename, row_number)
                record["stress_level"] = _required_text(row, "stress_level", filename, row_number)
            transformed[table].append(record)

    for row_number, row in enumerate(raw["food_nutrition.csv"], start=2):
        food = {
            "food_id": _required_text(row, "food_id", "food_nutrition.csv", row_number),
            "food_name": _required_text(row, "food_name", "food_nutrition.csv", row_number),
            "regional_name": (row.get("regional_name") or "").strip(),
            "category": _required_text(row, "category", "food_nutrition.csv", row_number),
            "serving_size": _required_text(row, "serving_size", "food_nutrition.csv", row_number),
            "source": _required_text(row, "source", "food_nutrition.csv", row_number),
            "is_demo": _demo_flag(row, "food_nutrition.csv", row_number),
        }
        for source, target in (
            ("calories", "calories"),
            ("carbohydrates_g", "carbohydrates"),
            ("protein_g", "protein"),
            ("fat_g", "fat"),
            ("fiber_g", "fiber"),
        ):
            food[target] = _number(row, source, "food_nutrition.csv", row_number)
        transformed["food_nutrition"].append(food)

    food_names: set[str] = set()
    for row_number, food in enumerate(transformed["food_nutrition"], start=2):
        normalized = food["food_name"].casefold()
        if normalized in food_names:
            raise DataValidationError(
                f"food_nutrition.csv, row {row_number}: duplicate food_name {food['food_name']!r}."
            )
        food_names.add(normalized)
    return transformed, hashes


def initialize_database(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    force: bool = False,
) -> bool:
    """Create the schema and import CSVs only when new/changed; return whether imported."""
    source_dir = Path(data_dir)
    connection = _connect(database_path)
    try:
        connection.executescript(SCHEMA)
        meal_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(meals)").fetchall()
        }
        if "meal_time" not in meal_columns:
            connection.execute("ALTER TABLE meals ADD COLUMN meal_time TEXT")
        meal_info = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(meals)").fetchall()
        }
        if any(meal_info[column]["notnull"] for column in (
            "calories", "carbohydrates", "protein", "fat", "fiber"
        )):
            connection.execute("BEGIN")
            connection.execute(
                """
                CREATE TABLE meals_nullable_nutrition (
                    meal_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                    date TEXT NOT NULL,
                    meal_type TEXT NOT NULL,
                    meal_time TEXT,
                    food TEXT NOT NULL,
                    portion TEXT NOT NULL,
                    calories REAL CHECK (calories >= 0),
                    carbohydrates REAL CHECK (carbohydrates >= 0),
                    protein REAL CHECK (protein >= 0),
                    fat REAL CHECK (fat >= 0),
                    fiber REAL CHECK (fiber >= 0),
                    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
                )
                """
            )
            columns = (
                "meal_id", "patient_id", "date", "meal_type", "meal_time", "food",
                "portion", "calories", "carbohydrates", "protein", "fat", "fiber", "is_demo",
            )
            column_list = ", ".join(columns)
            connection.execute(
                f"INSERT INTO meals_nullable_nutrition ({column_list}) "
                f"SELECT {column_list} FROM meals"
            )
            connection.execute("DROP TABLE meals")
            connection.execute("ALTER TABLE meals_nullable_nutrition RENAME TO meals")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_meals_patient_date ON meals(patient_id, date)"
            )
            connection.commit()
        stored_hashes = {
            row["filename"]: row["sha256"]
            for row in connection.execute("SELECT filename, sha256 FROM import_metadata")
        }
        if not force and len(stored_hashes) == len(CSV_COLUMNS):
            current_hashes: dict[str, str] = {}
            for filename in CSV_COLUMNS:
                path = source_dir / filename
                if not path.is_file():
                    raise DataValidationError(f"Required CSV file is missing: {path}")
                try:
                    current_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as error:
                    raise DataValidationError(f"Unable to read {path}: {error}") from error
            if current_hashes == stored_hashes:
                return False

        source, hashes = _validated_source(source_dir)
        connection.execute("BEGIN")
        for table in ("meals", "glucose", "lifestyle", "food_nutrition", "patients"):
            connection.execute(f"DELETE FROM {table}")

        for table, rows in source.items():
            if not rows:
                continue
            columns = tuple(rows[0])
            placeholders = ", ".join("?" for _ in columns)
            column_names = ", ".join(columns)
            connection.executemany(
                f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})",
                ([row[column] for column in columns] for row in rows),
            )

        imported_at = datetime.now().isoformat(timespec="seconds")
        connection.executemany(
            """
            INSERT INTO import_metadata (filename, sha256, imported_at)
            VALUES (?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                sha256 = excluded.sha256,
                imported_at = excluded.imported_at
            """,
            [(filename, digest, imported_at) for filename, digest in hashes.items()],
        )
        connection.commit()
        return True
    except (DataValidationError, DatabaseError):
        connection.rollback()
        raise
    except sqlite3.Error as error:
        connection.rollback()
        raise DatabaseError(f"Unable to initialize GlucoCare database: {error}") from error
    finally:
        connection.close()


def _fetch_all(
    query: str,
    parameters: tuple[Any, ...] = (),
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    connection = _connect(database_path)
    try:
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]
    except sqlite3.Error as error:
        raise DatabaseError(f"Unable to read GlucoCare database: {error}") from error
    finally:
        connection.close()


def _fetch_one(
    query: str,
    parameters: tuple[Any, ...] = (),
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    connection = _connect(database_path)
    try:
        row = connection.execute(query, parameters).fetchone()
        return dict(row) if row is not None else None
    except sqlite3.Error as error:
        raise DatabaseError(f"Unable to read GlucoCare database: {error}") from error
    finally:
        connection.close()


def get_all_patients(database_path: str | Path = DEFAULT_DATABASE_PATH) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM patients ORDER BY patient_id", database_path=database_path)


def load_patients(database_path: str | Path = DEFAULT_DATABASE_PATH) -> list[dict[str, Any]]:
    """Compatibility-friendly alias for obtaining all patient records."""
    return get_all_patients(database_path)


def get_patient(
    patient_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    if not isinstance(patient_id, str) or not patient_id.strip():
        return None
    return _fetch_one(
        "SELECT * FROM patients WHERE patient_id = ?",
        (patient_id.strip(),),
        database_path,
    )


def get_patient_meals(
    patient_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    return _fetch_all(
        "SELECT * FROM meals WHERE patient_id = ? ORDER BY date DESC, meal_type",
        (patient_id,),
        database_path,
    )


def get_patient_glucose(
    patient_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    return _fetch_all(
        "SELECT * FROM glucose WHERE patient_id = ? ORDER BY date DESC, time DESC",
        (patient_id,),
        database_path,
    )


def get_patient_lifestyle(
    patient_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    return _fetch_all(
        "SELECT * FROM lifestyle WHERE patient_id = ? ORDER BY date DESC",
        (patient_id,),
        database_path,
    )


def _recent_records(
    table: str,
    patient_id: str,
    days: int,
    database_path: str | Path,
) -> list[dict[str, Any]]:
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer.")
    start_date = (date.today() - timedelta(days=days - 1)).isoformat()
    return _fetch_all(
        f"SELECT * FROM {table} WHERE patient_id = ? AND date >= ? AND date <= ? "
        "ORDER BY date DESC",
        (patient_id, start_date, date.today().isoformat()),
        database_path,
    )


def get_recent_meals(
    patient_id: str,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    return _recent_records("meals", patient_id, days, database_path)


def get_recent_glucose(
    patient_id: str,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    return _recent_records("glucose", patient_id, days, database_path)


def get_recent_lifestyle(
    patient_id: str,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    return _recent_records("lifestyle", patient_id, days, database_path)


def get_food_information(
    food_name: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    if not isinstance(food_name, str) or not food_name.strip():
        return None
    return _fetch_one(
        "SELECT * FROM food_nutrition WHERE food_name = ? COLLATE NOCASE",
        (food_name.strip(),),
        database_path,
    )


def get_food_information_by_id(
    food_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    if not isinstance(food_id, str) or not food_id.strip():
        return None
    return _fetch_one(
        "SELECT * FROM food_nutrition WHERE food_id = ?",
        (food_id.strip(),),
        database_path,
    )


def get_all_food_information(database_path: str | Path = DEFAULT_DATABASE_PATH) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM food_nutrition ORDER BY food_name", database_path=database_path)


def save_food_image_meal(
    *,
    meal_id: str,
    patient_id: str,
    meal_date: str,
    meal_time: str,
    meal_type: str,
    food: str,
    portion: str,
    nutrition: dict[str, float | None],
    food_ids: list[str],
    item_details: list[dict[str, Any]],
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Persist one explicitly confirmed meal, retaining unavailable nutrition as NULL."""
    nutrient_columns = ("calories", "carbohydrates", "protein", "fat", "fiber")
    if any(column not in nutrition for column in nutrient_columns):
        raise ValueError("All meal nutrition fields must be provided, including unavailable values.")
    if not all(isinstance(value, str) and value.strip() for value in (
        meal_id, patient_id, meal_date, meal_time, meal_type, food, portion
    )):
        raise ValueError("Meal and patient details must not be empty.")
    connection = _connect(database_path)
    try:
        connection.execute("BEGIN")
        patient = connection.execute(
            "SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        if patient is None:
            raise ValueError("The selected patient is not available.")
        if food_ids:
            placeholders = ", ".join("?" for _ in food_ids)
            matched = connection.execute(
                f"SELECT COUNT(*) FROM food_nutrition WHERE food_id IN ({placeholders})",
                tuple(food_ids),
            ).fetchone()[0]
            if matched != len(set(food_ids)):
                raise ValueError("A selected food record is no longer available.")
        values: list[float | None] = []
        for column in nutrient_columns:
            value = nutrition[column]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Meal {column} must be non-negative or unavailable.")
            values.append(value)
        connection.execute(
            """
            INSERT INTO meals (
                meal_id, patient_id, date, meal_type, meal_time, food, portion,
                calories, carbohydrates, protein, fat, fiber, is_demo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                meal_id, patient_id, meal_date, meal_type, meal_time, food, portion,
                *values,
            ),
        )
        for item_index, item in enumerate(item_details):
            selected_food_id = item.get("selected_food_id")
            quantity = item.get("serving_quantity")
            quantity_source = item.get("quantity_source")
            nutrition_status = item.get("nutrition_status")
            identification_source = item.get("identification_source")
            if (
                not isinstance(item.get("original_prediction"), str)
                or not item["original_prediction"].strip()
                or not isinstance(item.get("corrected_label"), str)
                or not item["corrected_label"].strip()
                or identification_source not in {"vision", "user"}
                or quantity_source not in {"user_confirmed", "unknown"}
                or nutrition_status not in {"verified", "unverified", "unmatched"}
                or (quantity is not None and (
                    isinstance(quantity, bool)
                    or not isinstance(quantity, (int, float))
                    or not math.isfinite(quantity)
                    or quantity <= 0
                ))
            ):
                raise ValueError("A confirmed food item contains invalid review details.")
            connection.execute(
                """
                INSERT INTO food_image_meal_items (
                    meal_id, item_index, original_prediction, corrected_label,
                    identification_source, selected_food_id, serving_quantity,
                    quantity_source, nutrition_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meal_id, item_index, item["original_prediction"], item["corrected_label"],
                    identification_source, selected_food_id, quantity, quantity_source,
                    nutrition_status,
                ),
            )
        connection.commit()
    except ValueError:
        connection.rollback()
        raise
    except sqlite3.IntegrityError as error:
        connection.rollback()
        if "meal_id" in str(error):
            raise DatabaseError("This meal confirmation was already submitted.") from error
        raise DatabaseError("The confirmed meal could not be saved.") from error
    except sqlite3.Error as error:
        connection.rollback()
        raise DatabaseError("The confirmed meal could not be saved.") from error
    finally:
        connection.close()


def main() -> None:
    imported = initialize_database()
    if imported:
        print(f"Initialized GlucoCare SQLite database at {DEFAULT_DATABASE_PATH}")
    else:
        print(f"GlucoCare database is up to date at {DEFAULT_DATABASE_PATH}")


if __name__ == "__main__":
    main()
