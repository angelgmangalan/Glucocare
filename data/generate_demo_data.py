"""Generate a small, reproducible, clearly synthetic GlucoCare dataset."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path(__file__).resolve().parent
DEMO_SOURCE = "Synthetic demo estimate; not clinically validated"

PATIENTS: list[dict[str, Any]] = [
    {"patient_id": "P001", "age": 42, "gender": "Female", "diabetes_type": "Type 2", "diet_preference": "Vegetarian", "cuisine_preference": "Kerala", "activity_level": "Moderate", "allergies_restrictions": "None", "health_goals": "Learn about balanced meal patterns", "lifestyle_notes": "Desk-based work; evening walks"},
    {"patient_id": "P002", "age": 55, "gender": "Male", "diabetes_type": "Type 2", "diet_preference": "Non-vegetarian", "cuisine_preference": "Kerala", "activity_level": "Light", "allergies_restrictions": "Fish", "health_goals": "Understand meal and glucose records", "lifestyle_notes": "Mostly sedentary; occasional walks"},
    {"patient_id": "P003", "age": 36, "gender": "Female", "diabetes_type": "Type 2", "diet_preference": "Vegetarian", "cuisine_preference": "South Indian", "activity_level": "Active", "allergies_restrictions": "Peanuts", "health_goals": "Explore regional meal variety", "lifestyle_notes": "Regular yoga practice"},
    {"patient_id": "P004", "age": 61, "gender": "Male", "diabetes_type": "Type 2", "diet_preference": "Non-vegetarian", "cuisine_preference": "Kerala", "activity_level": "Light", "allergies_restrictions": "None", "health_goals": "Build consistent tracking habits", "lifestyle_notes": "Retired; morning gardening"},
    {"patient_id": "P005", "age": 47, "gender": "Female", "diabetes_type": "Type 2", "diet_preference": "Vegetarian", "cuisine_preference": "Kerala", "activity_level": "Moderate", "allergies_restrictions": "Dairy", "health_goals": "Review meal and lifestyle patterns", "lifestyle_notes": "Walks several times a week"},
    {"patient_id": "P006", "age": 51, "gender": "Male", "diabetes_type": "Type 2", "diet_preference": "Non-vegetarian", "cuisine_preference": "South Indian", "activity_level": "Moderate", "allergies_restrictions": "Eggs", "health_goals": "Learn general nutrition basics", "lifestyle_notes": "Commutes by bicycle occasionally"},
    {"patient_id": "P007", "age": 29, "gender": "Female", "diabetes_type": "Type 2", "diet_preference": "Vegetarian", "cuisine_preference": "South Indian", "activity_level": "Active", "allergies_restrictions": "None", "health_goals": "Track daily routines", "lifestyle_notes": "Enjoys dance and walking"},
    {"patient_id": "P008", "age": 68, "gender": "Male", "diabetes_type": "Type 2", "diet_preference": "Non-vegetarian", "cuisine_preference": "Kerala", "activity_level": "Light", "allergies_restrictions": "Shellfish", "health_goals": "Keep a simple food diary", "lifestyle_notes": "Short walks when convenient"},
    {"patient_id": "P009", "age": 39, "gender": "Female", "diabetes_type": "Type 2", "diet_preference": "Vegetarian", "cuisine_preference": "Kerala", "activity_level": "Moderate", "allergies_restrictions": "Tree nuts", "health_goals": "Explore meal timing patterns", "lifestyle_notes": "Works hybrid; weekend hikes"},
    {"patient_id": "P010", "age": 58, "gender": "Male", "diabetes_type": "Type 2", "diet_preference": "Non-vegetarian", "cuisine_preference": "South Indian", "activity_level": "Moderate", "allergies_restrictions": "None", "health_goals": "Review weekly records", "lifestyle_notes": "Regular evening walks"},
]

# Values are intentionally marked as unverified illustrative estimates. They
# are dataset examples, not clinical nutrition advice or a substitute for IFCT.
FOODS: list[dict[str, Any]] = [
    {"food_id": "F001", "food_name": "Puttu", "regional_name": "പുട്ട്", "category": "Breakfast", "serving_size": "150 g serving", "calories": 240, "carbohydrates_g": 47, "protein_g": 5, "fat_g": 3, "fiber_g": 4, "allergens": "Coconut"},
    {"food_id": "F002", "food_name": "Appam", "regional_name": "അപ്പം", "category": "Breakfast", "serving_size": "2 medium pieces (120 g)", "calories": 190, "carbohydrates_g": 39, "protein_g": 4, "fat_g": 2, "fiber_g": 2, "allergens": ""},
    {"food_id": "F003", "food_name": "Idiyappam", "regional_name": "ഇടിയപ്പം", "category": "Breakfast", "serving_size": "2 nests (120 g)", "calories": 180, "carbohydrates_g": 39, "protein_g": 3, "fat_g": 1, "fiber_g": 2, "allergens": ""},
    {"food_id": "F004", "food_name": "Dosa", "regional_name": "ദോശ", "category": "Breakfast", "serving_size": "2 medium pieces (130 g)", "calories": 210, "carbohydrates_g": 34, "protein_g": 6, "fat_g": 5, "fiber_g": 3, "allergens": ""},
    {"food_id": "F005", "food_name": "Idli", "regional_name": "ഇഡ്ഡലി", "category": "Breakfast", "serving_size": "2 pieces (120 g)", "calories": 150, "carbohydrates_g": 30, "protein_g": 5, "fat_g": 1, "fiber_g": 2, "allergens": ""},
    {"food_id": "F006", "food_name": "Matta rice", "regional_name": "മട്ട അരി", "category": "Grain", "serving_size": "1 cup cooked (160 g)", "calories": 210, "carbohydrates_g": 45, "protein_g": 4, "fat_g": 1, "fiber_g": 3, "allergens": ""},
    {"food_id": "F007", "food_name": "Brown rice", "regional_name": "തവിടുള്ള അരി", "category": "Grain", "serving_size": "1 cup cooked (160 g)", "calories": 215, "carbohydrates_g": 45, "protein_g": 5, "fat_g": 2, "fiber_g": 4, "allergens": ""},
    {"food_id": "F008", "food_name": "White rice", "regional_name": "വെളുത്ത അരി", "category": "Grain", "serving_size": "1 cup cooked (160 g)", "calories": 205, "carbohydrates_g": 45, "protein_g": 4, "fat_g": 0, "fiber_g": 1, "allergens": ""},
    {"food_id": "F009", "food_name": "Chapati", "regional_name": "ചപ്പാത്തി", "category": "Grain", "serving_size": "2 medium pieces (100 g)", "calories": 220, "carbohydrates_g": 38, "protein_g": 7, "fat_g": 5, "fiber_g": 6, "allergens": "Wheat"},
    {"food_id": "F010", "food_name": "Kadala curry", "regional_name": "കടല കറി", "category": "Legume", "serving_size": "1 cup (180 g)", "calories": 260, "carbohydrates_g": 35, "protein_g": 12, "fat_g": 8, "fiber_g": 10, "allergens": ""},
    {"food_id": "F011", "food_name": "Avial", "regional_name": "അവിയൽ", "category": "Vegetable", "serving_size": "1 cup (180 g)", "calories": 150, "carbohydrates_g": 16, "protein_g": 4, "fat_g": 8, "fiber_g": 6, "allergens": "Coconut"},
    {"food_id": "F012", "food_name": "Thoran", "regional_name": "തോരൻ", "category": "Vegetable", "serving_size": "1 cup (150 g)", "calories": 130, "carbohydrates_g": 13, "protein_g": 4, "fat_g": 7, "fiber_g": 6, "allergens": "Coconut"},
    {"food_id": "F013", "food_name": "Sambar", "regional_name": "സാമ്പാർ", "category": "Legume", "serving_size": "1 cup (200 g)", "calories": 140, "carbohydrates_g": 22, "protein_g": 7, "fat_g": 3, "fiber_g": 6, "allergens": ""},
    {"food_id": "F014", "food_name": "Fish curry", "regional_name": "മീൻ കറി", "category": "Protein", "serving_size": "1 piece with gravy (150 g)", "calories": 190, "carbohydrates_g": 5, "protein_g": 24, "fat_g": 8, "fiber_g": 1, "allergens": "Fish"},
    {"food_id": "F015", "food_name": "Vegetable curry", "regional_name": "പച്ചക്കറി കറി", "category": "Vegetable", "serving_size": "1 cup (180 g)", "calories": 160, "carbohydrates_g": 20, "protein_g": 5, "fat_g": 7, "fiber_g": 7, "allergens": ""},
    {"food_id": "F016", "food_name": "Green gram curry", "regional_name": "ചെറുപയർ കറി", "category": "Legume", "serving_size": "1 cup (180 g)", "calories": 220, "carbohydrates_g": 30, "protein_g": 13, "fat_g": 5, "fiber_g": 9, "allergens": ""},
    {"food_id": "F017", "food_name": "Egg curry", "regional_name": "മുട്ട കറി", "category": "Protein", "serving_size": "2 eggs with gravy (180 g)", "calories": 240, "carbohydrates_g": 8, "protein_g": 15, "fat_g": 16, "fiber_g": 2, "allergens": "Eggs"},
    {"food_id": "F018", "food_name": "Chicken curry", "regional_name": "കോഴി കറി", "category": "Protein", "serving_size": "1 cup (180 g)", "calories": 280, "carbohydrates_g": 8, "protein_g": 27, "fat_g": 16, "fiber_g": 2, "allergens": ""},
    {"food_id": "F019", "food_name": "Paneer curry", "regional_name": "പനീർ കറി", "category": "Protein", "serving_size": "1 cup (180 g)", "calories": 300, "carbohydrates_g": 12, "protein_g": 17, "fat_g": 21, "fiber_g": 3, "allergens": "Dairy"},
    {"food_id": "F020", "food_name": "Vegetable stew", "regional_name": "പച്ചക്കറി സ്റ്റ്യൂ", "category": "Vegetable", "serving_size": "1 cup (180 g)", "calories": 170, "carbohydrates_g": 18, "protein_g": 4, "fat_g": 9, "fiber_g": 5, "allergens": "Coconut"},
    {"food_id": "F021", "food_name": "Coconut chutney", "regional_name": "തേങ്ങ ചമ്മന്തി", "category": "Side", "serving_size": "2 tbsp (30 g)", "calories": 90, "carbohydrates_g": 4, "protein_g": 2, "fat_g": 8, "fiber_g": 2, "allergens": "Coconut"},
    {"food_id": "F022", "food_name": "Tomato chutney", "regional_name": "തക്കാളി ചമ്മന്തി", "category": "Side", "serving_size": "2 tbsp (30 g)", "calories": 35, "carbohydrates_g": 6, "protein_g": 1, "fat_g": 1, "fiber_g": 1, "allergens": ""},
    {"food_id": "F023", "food_name": "Oats porridge", "regional_name": "ഓട്സ് കഞ്ഞി", "category": "Breakfast", "serving_size": "1 bowl (200 g)", "calories": 190, "carbohydrates_g": 32, "protein_g": 7, "fat_g": 4, "fiber_g": 5, "allergens": "Oats"},
    {"food_id": "F024", "food_name": "Ragi dosa", "regional_name": "റാഗി ദോശ", "category": "Breakfast", "serving_size": "2 medium pieces (140 g)", "calories": 200, "carbohydrates_g": 35, "protein_g": 5, "fat_g": 4, "fiber_g": 5, "allergens": ""},
    {"food_id": "F025", "food_name": "Sprouts salad", "regional_name": "മുളപ്പിച്ച പയർ സാലഡ്", "category": "Legume", "serving_size": "1 cup (150 g)", "calories": 120, "carbohydrates_g": 20, "protein_g": 9, "fat_g": 2, "fiber_g": 7, "allergens": ""},
    {"food_id": "F026", "food_name": "Banana", "regional_name": "വാഴപ്പഴം", "category": "Fruit", "serving_size": "1 medium (120 g)", "calories": 105, "carbohydrates_g": 27, "protein_g": 1, "fat_g": 0, "fiber_g": 3, "allergens": ""},
    {"food_id": "F027", "food_name": "Guava", "regional_name": "പേരയ്ക്ക", "category": "Fruit", "serving_size": "1 medium (100 g)", "calories": 68, "carbohydrates_g": 14, "protein_g": 3, "fat_g": 1, "fiber_g": 5, "allergens": ""},
    {"food_id": "F028", "food_name": "Papaya", "regional_name": "പപ്പായ", "category": "Fruit", "serving_size": "1 cup (140 g)", "calories": 60, "carbohydrates_g": 15, "protein_g": 1, "fat_g": 0, "fiber_g": 3, "allergens": ""},
    {"food_id": "F029", "food_name": "Orange", "regional_name": "ഓറഞ്ച്", "category": "Fruit", "serving_size": "1 medium (130 g)", "calories": 62, "carbohydrates_g": 15, "protein_g": 1, "fat_g": 0, "fiber_g": 3, "allergens": ""},
    {"food_id": "F030", "food_name": "Peanuts", "regional_name": "നിലക്കടല", "category": "Nuts", "serving_size": "30 g handful", "calories": 170, "carbohydrates_g": 5, "protein_g": 7, "fat_g": 14, "fiber_g": 3, "allergens": "Peanuts"},
    {"food_id": "F031", "food_name": "Almonds", "regional_name": "ബദാം", "category": "Nuts", "serving_size": "28 g handful", "calories": 165, "carbohydrates_g": 6, "protein_g": 6, "fat_g": 14, "fiber_g": 4, "allergens": "Tree nuts"},
    {"food_id": "F032", "food_name": "Curd", "regional_name": "തൈര്", "category": "Dairy", "serving_size": "1 small bowl (150 g)", "calories": 95, "carbohydrates_g": 7, "protein_g": 5, "fat_g": 5, "fiber_g": 0, "allergens": "Dairy"},
    {"food_id": "F033", "food_name": "Boiled egg", "regional_name": "പുഴുങ്ങിയ മുട്ട", "category": "Protein", "serving_size": "1 large egg (50 g)", "calories": 78, "carbohydrates_g": 1, "protein_g": 6, "fat_g": 5, "fiber_g": 0, "allergens": "Eggs"},
    {"food_id": "F034", "food_name": "Grilled fish", "regional_name": "ഗ്രിൽ ചെയ്ത മീൻ", "category": "Protein", "serving_size": "1 fillet (120 g)", "calories": 200, "carbohydrates_g": 0, "protein_g": 26, "fat_g": 10, "fiber_g": 0, "allergens": "Fish"},
]

MEAL_OPTIONS: dict[str, list[list[str]]] = {
    "Breakfast": [
        ["Puttu", "Kadala curry"],
        ["Idli", "Sambar"],
        ["Appam", "Egg curry"],
        ["Dosa", "Tomato chutney"],
        ["Idiyappam", "Vegetable stew"],
        ["Oats porridge", "Guava"],
        ["Ragi dosa", "Sambar"],
    ],
    "Lunch": [
        ["Matta rice", "Fish curry", "Avial"],
        ["Brown rice", "Green gram curry", "Thoran"],
        ["White rice", "Chicken curry", "Vegetable curry"],
        ["Chapati", "Dal", "Avial"],
        ["Matta rice", "Egg curry", "Thoran"],
        ["Brown rice", "Fish curry", "Vegetable curry"],
        ["Chapati", "Paneer curry", "Thoran"],
    ],
    "Dinner": [
        ["Chapati", "Vegetable curry", "Paneer curry"],
        ["Appam", "Fish curry", "Thoran"],
        ["Dosa", "Sambar", "Sprouts salad"],
        ["Puttu", "Green gram curry"],
        ["Idiyappam", "Chicken curry", "Vegetable curry"],
        ["Ragi dosa", "Egg curry", "Avial"],
        ["Brown rice", "Green gram curry", "Thoran"],
    ],
}

FOOD_LOOKUP = {food["food_name"]: food for food in FOODS}
FOOD_LOOKUP["Dal"] = {
    "category": "Legume",
    "calories": 180, "carbohydrates_g": 27, "protein_g": 10,
    "fat_g": 4, "fiber_g": 8, "allergens": "",
}


def _choose_meal(patient: dict[str, Any], meal_type: str, day_index: int) -> list[str]:
    restrictions = {
        item.strip().lower()
        for item in patient["allergies_restrictions"].split(",")
        if item.strip() and item.strip().lower() != "none"
    }
    options = MEAL_OPTIONS[meal_type]
    for offset in range(len(options)):
        candidate = options[(day_index + offset + int(patient["patient_id"][-1])) % len(options)]
        allergens = {
            allergen.strip().lower()
            for item in candidate
            for allergen in FOOD_LOOKUP[item]["allergens"].split(",")
            if allergen.strip()
        }
        if patient["diet_preference"] == "Vegetarian" and any(
            FOOD_LOOKUP[item]["category"] == "Protein"
            and item not in {"Paneer curry"}
            for item in candidate
        ):
            continue
        if not restrictions.intersection(allergens):
            return candidate
    return ["Brown rice", "Green gram curry", "Thoran"]


def build_demo_data(today: date | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return linked synthetic records covering today and the preceding six days."""
    today = today or date.today()
    meals: list[dict[str, Any]] = []
    glucose: list[dict[str, Any]] = []
    lifestyle: list[dict[str, Any]] = []

    for patient_index, patient in enumerate(PATIENTS):
        for day_index in range(7):
            record_date = today - timedelta(days=6 - day_index)
            date_text = record_date.isoformat()
            for meal_index, meal_type in enumerate(("Breakfast", "Lunch", "Dinner")):
                foods = _choose_meal(patient, meal_type, day_index)
                selected = [FOOD_LOOKUP[name] for name in foods]
                meals.append({
                    "meal_id": f"M-{patient['patient_id']}-{date_text}-{meal_type[:1]}",
                    "patient_id": patient["patient_id"],
                    "date": date_text,
                    "meal_type": meal_type,
                    "food_items": "; ".join(foods),
                    "portion_size": "; ".join(FOOD_LOOKUP[name].get("serving_size", "1 serving") for name in foods),
                    "calories": sum(int(food["calories"]) for food in selected),
                    "carbohydrates_g": sum(int(food["carbohydrates_g"]) for food in selected),
                    "protein_g": sum(int(food["protein_g"]) for food in selected),
                    "fat_g": sum(int(food["fat_g"]) for food in selected),
                    "fiber_g": sum(int(food["fiber_g"]) for food in selected),
                    "is_demo": True,
                })

            baseline = 92 + patient_index * 3 + (day_index % 3) * 4
            for context, hour, offset in (
                ("Fasting", "07:00", 0),
                ("After meal", "14:00", 26 + (day_index % 4) * 3),
            ):
                glucose.append({
                    "glucose_id": f"G-{patient['patient_id']}-{date_text}-{context.replace(' ', '')}",
                    "patient_id": patient["patient_id"],
                    "date": date_text,
                    "time": hour,
                    "glucose_mg_dl": baseline + offset,
                    "context": context,
                    "is_demo": True,
                })

            lifestyle.append({
                "lifestyle_id": f"L-{patient['patient_id']}-{date_text}",
                "patient_id": patient["patient_id"],
                "date": date_text,
                "sleep_hours": round(6.0 + ((patient_index + day_index) % 5) * 0.5, 1),
                "exercise_minutes": (patient_index * 7 + day_index * 13) % 61,
                "water_liters": round(1.5 + ((patient_index + day_index) % 4) * 0.3, 1),
                "stress_level": ("Low", "Moderate", "High")[(patient_index + day_index) % 3],
                "is_demo": True,
            })

    food_rows = [
        {
            **food,
            "source": DEMO_SOURCE,
            "is_demo": True,
        }
        for food in FOODS
    ]
    patient_rows = [{**patient, "is_demo": True} for patient in PATIENTS]
    return {
        "patients.csv": patient_rows,
        "meals.csv": meals,
        "glucose.csv": glucose,
        "lifestyle.csv": lifestyle,
        "food_nutrition.csv": food_rows,
    }


def write_demo_data(output_dir: Path = OUTPUT_DIR, today: date | None = None) -> None:
    """Write all demo tables as UTF-8 CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in build_demo_data(today).items():
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote 10 synthetic patients and 7 days of linked records to {output_dir}")


if __name__ == "__main__":
    write_demo_data()
