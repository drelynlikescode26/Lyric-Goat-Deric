import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services import generate


class StructuredGenerationTests(unittest.TestCase):
    @patch.dict(os.environ, {"LYRIC_GOAT_WRITER": "anthropic"})
    def test_three_versions_use_one_structured_request(self):
        payload = {
            "melodic": "I keep moving",
            "rap": "Still on the run",
            "punchy": "I won't fold now",
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

    @patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test", "LYRIC_GOAT_WRITER": "auto"})
    @patch("services.generate.OpenAI")
    def test_one_openai_key_writes_three_versions_with_bounded_private_request(self, factory):
        payload = {"melodic": "I keep moving", "rap": "Still on the run", "punchy": "I won't fold now"}
        factory.return_value.responses.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(payload))
        flow = {"phrase_map": [{"syllables": 4, "max_words": 5, "max_syllables": 5}]}
        results = generate.generate_lyrics("", flow)
        self.assertEqual(len(results), 3)
        request = factory.return_value.responses.create
        self.assertEqual(request.call_count, 1)
        kwargs = request.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-6-luna")
        self.assertFalse(kwargs["store"])
        self.assertEqual(kwargs["max_output_tokens"], 4096)
        self.assertTrue(kwargs["text"]["format"]["strict"])
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test", "LYRIC_GOAT_WRITER": "openai"})
    @patch("services.generate.OpenAI")
    def test_incomplete_or_refused_output_is_not_presented_as_lyrics(self, factory):
        request = factory.return_value.responses.create
        for status, output in [("incomplete", ""), ("completed", "")]:
            request.return_value = SimpleNamespace(status=status, output_text=output)
            with self.assertRaises(generate.GenerationError):
                generate._request("test", "rnb")

    @patch.dict(os.environ, {}, clear=True)
    def test_existing_anthropic_setup_still_selects_legacy_writer(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "offline-test"}):
            self.assertEqual(generate.writer_provider(), "anthropic")
