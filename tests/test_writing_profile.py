import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import writing_profile


class WritingProfileTests(unittest.TestCase):
    def test_records_and_reads_accepted_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            feedback_file = directory / "writing_feedback.jsonl"
            with patch.object(writing_profile, "PROFILE_DIR", directory), patch.object(
                writing_profile, "FEEDBACK_FILE", feedback_file
            ):
                entry = writing_profile.record_feedback({
                    "action": "accepted",
                    "song_id": "song1",
                    "final_line": "I kept this line",
                })
                recent = writing_profile.recent_feedback()

            self.assertEqual(entry["final_line"], "I kept this line")
            self.assertEqual(recent[0]["id"], entry["id"])
