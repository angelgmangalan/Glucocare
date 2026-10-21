from __future__ import annotations

import unittest

from src.chatbot.chatbot import _source_records
from src.chatbot.safety import validate_response


class GroundingTests(unittest.TestCase):
    def test_only_retrieved_metadata_becomes_a_citation(self) -> None:
        sources = _source_records([
            {
                "source": "WHO",
                "title": "Diabetes",
                "url": "https://www.who.int/example",
                "topic": "diabetes",
                "text": "Verified retrieved text.",
            },
            {
                "source": "Invented Organization",
                "title": "No URL",
                "url": "javascript:alert(1)",
                "topic": "nutrition",
                "text": "Retrieved text.",
            },
        ])
        self.assertEqual(sources[0]["url"], "https://www.who.int/example")
        self.assertEqual(sources[1]["url"], "")
        self.assertEqual(len(sources), 2)

    def test_medical_response_requires_retrieved_source(self) -> None:
        self.assertFalse(
            validate_response("General medical statement.", requires_sources=True)[0]
        )
        self.assertTrue(
            validate_response(
                "General medical statement.",
                requires_sources=True,
                has_sources=True,
            )[0]
        )

    def test_unknown_patient_measurement_is_rejected(self) -> None:
        context = {"patient_data": {"glucose_history": [{"glucose_value": 132}]}}
        self.assertFalse(
            validate_response(
                "Your reading was 999 mg/dL.",
                patient_context=context,
            )[0]
        )
        self.assertTrue(
            validate_response(
                "Your recorded reading was 132 mg/dL.",
                patient_context=context,
            )[0]
        )


if __name__ == "__main__":
    unittest.main()
