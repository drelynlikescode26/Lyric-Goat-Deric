import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services import generate


class StructuredGenerationTests(unittest.TestCase):
    def test_three_versions_use_one_structured_request(self):
        payload = {
            "melodic": "I keep moving",
            "rap": "Still on the run",
            "punchy": "Won't fold",
        }
        messages = MagicMock()
        messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload))]
        )
        fake_client = SimpleNamespace(messages=messages)
        flow = {
            "phrase_map": [{"syllables": 4, "max_words": 5, "max_syllables": 5}],
            "tempo_bpm": 90,
            "flow_style": "steady",
        }

        with patch("services.generate._client", return_value=fake_client):
            results = generate.generate_lyrics("", flow)

        self.assertEqual(messages.create.call_count, 1)
        kwargs = messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], generate.GENERATION_MODEL)
        self.assertEqual(kwargs["output_config"]["format"]["type"], "json_schema")
        self.assertEqual({item["name"] for item in results}, {"melodic", "rap", "punchy"})

    def test_missing_key_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(generate.GenerationError, "ANTHROPIC_API_KEY"):
                generate._client()
