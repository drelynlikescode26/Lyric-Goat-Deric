import unittest

from scripts.benchmark_detection import score_phrase_maps


class BenchmarkScoreTests(unittest.TestCase):
    def test_phrase_and_syllable_errors(self):
        expected = [
            {"start_time": 0.0, "end_time": 1.0, "syllables": 4},
            {"start_time": 2.0, "end_time": 3.0, "syllables": 6},
        ]
        detected = [
            {"start_time": 0.1, "end_time": 1.2, "target_syllables": 5},
            {"start_time": 2.2, "end_time": 3.1, "target_syllables": 6},
        ]
        score = score_phrase_maps(expected, detected)
        self.assertEqual(score["phrase_count_error"], 0)
        self.assertEqual(score["mean_syllable_error"], 0.5)
        self.assertEqual(score["mean_boundary_error_seconds"], 0.15)


if __name__ == "__main__":
    unittest.main()
