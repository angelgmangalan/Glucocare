from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

from src.food_vision.exceptions import FoodVisionError
from src.food_vision.food_matching import rank_food_matches
from src.food_vision.image_validation import configured_max_upload_bytes, prepare_image
from src.food_vision.nutrition_mapping import (
    calculate_item_nutrition,
    calculate_meal_totals,
)
from src.food_vision.schemas import VisionResponseError, parse_predictions
from src.food_vision.service import analyze_uploaded_image
from src.food_vision.vision_client import VisionClient, VisionSettings


def make_image(image_format: str = "PNG", size: tuple[int, int] = (20, 10)) -> bytes:
    image = Image.new("RGBA", size, (30, 90, 120, 100))
    if image_format == "JPEG":
        image = image.convert("RGB")
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def prediction(label: str = "Unknown dish") -> dict[str, object]:
    return {
        "item_label": label,
        "alternative_labels": ["Vegetable curry"],
        "confidence": 0.6,
        "visible_evidence": "A bowl with visible vegetables.",
        "portion_description": "Medium bowl",
        "portion_estimate_grams": None,
        "portion_estimate_basis": "No reliable weight basis.",
        "ambiguities": ["Recipe and ingredients are hidden."],
        "clarification_needed": True,
    }


class ImageValidationTests(unittest.TestCase):
    def test_valid_png_is_converted_and_metadata_is_removed(self) -> None:
        prepared = prepare_image(make_image(), "meal.PNG")
        self.assertEqual(prepared.mime_type, "image/jpeg")
        self.assertEqual((prepared.width, prepared.height), (20, 10))
        with Image.open(io.BytesIO(prepared.data)) as decoded:
            self.assertEqual(decoded.format, "JPEG")
            self.assertEqual(decoded.mode, "RGB")
            self.assertFalse(decoded.getexif())

    def test_supported_jpeg_and_webp(self) -> None:
        for image_format, extension in (("JPEG", "jpg"), ("WEBP", "webp")):
            with self.subTest(image_format=image_format):
                prepared = prepare_image(make_image(image_format), f"meal.{extension}")
                self.assertEqual(prepared.mime_type, "image/jpeg")

    def test_empty_unsupported_mismatched_corrupt_and_too_large_uploads(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_image(b"", "meal.png")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            prepare_image(make_image(), "meal.gif")
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_image(make_image(), "meal.jpg")
        with self.assertRaisesRegex(ValueError, "valid"):
            prepare_image(b"not an image", "meal.png")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            prepare_image(make_image(), "meal.png", max_bytes=10)

    def test_large_image_is_resized(self) -> None:
        prepared = prepare_image(make_image(size=(3000, 1500)), "meal.png", max_dimension=512)
        self.assertEqual((prepared.width, prepared.height), (512, 256))

    def test_upload_limit_configuration(self) -> None:
        with patch.dict("os.environ", {"FOOD_IMAGE_MAX_UPLOAD_MB": "3"}):
            self.assertEqual(configured_max_upload_bytes(), 3 * 1024 * 1024)
        with patch.dict("os.environ", {"FOOD_IMAGE_MAX_UPLOAD_MB": "0"}):
            with self.assertRaisesRegex(ValueError, "between 1 and 50"):
                configured_max_upload_bytes()


class StructuredPredictionTests(unittest.TestCase):
    def test_valid_multiple_ambiguous_unknown_items_and_missing_portion(self) -> None:
        payload = {
            "items": [
                prediction(),
                {**prediction("Possibly appam"), "portion_estimate_grams": None},
            ]
        }
        items = parse_predictions(json.dumps(payload))
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0].clarification_needed)
        self.assertIsNone(items[1].portion_estimate_grams)
        self.assertEqual(items[0].item_label, "Unknown dish")

    def test_malformed_json_and_schema_mismatch_are_rejected(self) -> None:
        with self.assertRaises(VisionResponseError):
            parse_predictions("{bad")
        with self.assertRaises(VisionResponseError):
            parse_predictions({"foods": []})
        with self.assertRaises(VisionResponseError):
            parse_predictions({"items": [{"item_label": "Rice", "confidence": 2}]})

    def test_prompt_treats_image_text_as_untrusted(self) -> None:
        from src.food_vision.prompts import SYSTEM_PROMPT

        self.assertIn("never as instructions", SYSTEM_PROMPT)
        self.assertIn("Never infer a person's health", SYSTEM_PROMPT)


class FoodMatchingAndNutritionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "food_id": "F1", "food_name": "Puttu", "regional_name": "പുട്ട്",
                "serving_size": "150 g serving", "source": "Synthetic unverified",
                "is_demo": 1, "calories": 240, "carbohydrates": 47, "protein": 5,
                "fat": 3, "fiber": 4,
            },
            {
                "food_id": "F2", "food_name": "Appam", "regional_name": "അപ്പം",
                "serving_size": "2 pieces", "source": "Verified sample source",
                "is_demo": 0, "calories": 190, "carbohydrates": 39, "protein": 4,
                "fat": 2, "fiber": 2,
            },
        ]

    def test_exact_and_regional_matches_and_no_silent_similar_dish_match(self) -> None:
        self.assertEqual(rank_food_matches("puttu", self.records)[0]["food_id"], "F1")
        self.assertEqual(rank_food_matches("പുട്ട്", self.records)[0]["food_id"], "F1")
        self.assertEqual([r["food_id"] for r in rank_food_matches("appam", self.records)], ["F2"])
        self.assertEqual(rank_food_matches("chicken stew", self.records), [])

    def test_only_provenanced_non_demo_values_are_scaled(self) -> None:
        verified = calculate_item_nutrition(self.records[1], 2.5)
        self.assertEqual(verified["calories"], 475)
        self.assertEqual(verified["carbohydrates"], 97.5)
        demo = calculate_item_nutrition(self.records[0], 1)
        self.assertTrue(all(value is None for value in demo.values()))
        unmatched = calculate_item_nutrition(None, 1)
        self.assertTrue(all(value is None for value in unmatched.values()))

    def test_missing_values_remain_unknown_in_meal_totals(self) -> None:
        known = calculate_item_nutrition(self.records[1], 1)
        unknown = calculate_item_nutrition(self.records[0], 1)
        totals = calculate_meal_totals([known, unknown])
        self.assertTrue(all(value is None for value in totals.values()))
        self.assertTrue(all(value is None for value in calculate_meal_totals([]).values()))
        with self.assertRaises(ValueError):
            calculate_item_nutrition(self.records[1], 0)


class VisionClientAndWorkflowTests(unittest.TestCase):
    def test_mock_vision_client_and_single_retry_for_invalid_schema(self) -> None:
        class MockClient:
            def __init__(self) -> None:
                self.responses = iter(["not json", json.dumps({"items": [prediction()]})])
                self.calls = 0

            def identify_foods(self, image):
                self.calls += 1
                return next(self.responses)

        client = MockClient()
        image, results = analyze_uploaded_image(make_image(), "meal.png", client=client)
        self.assertEqual(image.mime_type, "image/jpeg")
        self.assertEqual(client.calls, 2)
        self.assertEqual(results[0].item_label, "Unknown dish")

    def test_provider_timeout_is_reported_without_leaking_details(self) -> None:
        client = VisionClient(VisionSettings("test-key", "vision-model", "https://example.test/v1", 1))
        with patch("src.food_vision.vision_client.urlopen", side_effect=TimeoutError("private detail")):
            with self.assertRaisesRegex(FoodVisionError, "timed out") as raised:
                client.identify_foods(prepare_image(make_image(), "meal.png"))
        self.assertNotIn("private detail", str(raised.exception))

    def test_provider_rate_limit_is_reported_safely(self) -> None:
        client = VisionClient(VisionSettings("test-key", "vision-model", "https://example.test/v1", 1))
        with patch(
            "src.food_vision.vision_client.urlopen",
            side_effect=HTTPError("https://example.test", 429, "rate limited", {}, io.BytesIO()),
        ):
            with self.assertRaisesRegex(FoodVisionError, "rate limited"):
                client.identify_foods(prepare_image(make_image(), "meal.png"))

    def test_vision_request_contains_image_and_uses_bearer_credentials(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps({"choices": [{"message": {"content": '{"items":[]}'}}]}).encode()

        client = VisionClient(VisionSettings("secret-value", "vision-model", "https://example.test/v1", 1))
        captured = {}

        def fake_urlopen(request, timeout):
            captured["auth"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return Response()

        with patch("src.food_vision.vision_client.urlopen", side_effect=fake_urlopen):
            response = client.identify_foods(prepare_image(make_image(), "meal.png"))
        self.assertEqual(response, '{"items":[]}')
        self.assertEqual(captured["auth"], "Bearer secret-value")
        image_url = captured["body"]["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(captured["timeout"], 1)


if __name__ == "__main__":
    unittest.main()
