import unittest
from unittest.mock import patch

from services.transcribe import TranscriptionError, transcribe_audio


class TranscriptionModeTests(unittest.TestCase):
    @patch("services.transcribe._transcribe_timed")
    def test_timed_mode_is_one_explicit_provider(self, timed):
        timed.return_value = {"text": "rough words", "words": [{"word": "rough"}], "duration": 2.0}
        result = transcribe_audio("unused.wav", mode="timed")
        self.assertEqual(result["text"], "rough words")
        self.assertEqual(result["diagnostics"]["timing_model"], "whisper-1")
        self.assertEqual(result["diagnostics"]["status"], "ok")

    @patch("services.transcribe._transcribe_timed")
    @patch("services.transcribe._transcribe_semantic")
    def test_hybrid_surfaces_partial_failure(self, semantic, timed):
        semantic.side_effect = RuntimeError("provider rejected request")
        timed.return_value = {"text": "fallback timing text", "words": [], "duration": 1.0}
        result = transcribe_audio("unused.wav", mode="hybrid")
        self.assertEqual(result["diagnostics"]["status"], "partial")
        self.assertIn("provider rejected request", result["diagnostics"]["errors"][0]["message"])

    @patch("services.transcribe._transcribe_timed")
    def test_total_failure_raises_with_diagnostics(self, timed):
        timed.side_effect = RuntimeError("network down")
        with self.assertRaises(TranscriptionError) as caught:
            transcribe_audio("unused.wav", mode="timed")
        self.assertEqual(caught.exception.diagnostics["errors"][0]["stage"], "timing")


if __name__ == "__main__":
    unittest.main()
