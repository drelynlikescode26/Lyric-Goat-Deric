import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from services.preprocess import preprocess_branches


class PreprocessBranchTests(unittest.TestCase):
    def test_speech_and_melody_stay_time_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "tone.wav"
            sample_rate = 44100
            duration = 0.6
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            sf.write(source, 0.2 * np.sin(2 * np.pi * 220 * t), sample_rate)

            branches = preprocess_branches(str(source))
            speech, speech_rate = sf.read(branches["speech_path"])
            melody, melody_rate = sf.read(branches["melody_path"])

            self.assertEqual(speech_rate, 16000)
            self.assertEqual(melody_rate, 22050)
            self.assertAlmostEqual(len(speech) / speech_rate, duration, places=2)
            self.assertAlmostEqual(len(melody) / melody_rate, duration, places=2)

            Path(branches["speech_path"]).unlink(missing_ok=True)
            Path(branches["melody_path"]).unlink(missing_ok=True)
