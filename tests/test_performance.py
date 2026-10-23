import unittest
import numpy as np
from services.performance import vocal_phrases, attach_performance
from services.analyze import _build_note_map

SR = 16000


def vocal(duration=2, pulses=0, run=False):
    t = np.arange(int(SR * duration)) / SR
    hz = np.where(t < .7, 220, np.where(t < 1.4, 277.18, 329.63)) if run else np.full_like(t, 220.)
    carrier = np.sin(2*np.pi*np.cumsum(hz)/SR)
    envelope = .22 * (1 + .06*np.sin(2*np.pi*5*t))
    if pulses:
        envelope = .22 * np.sin(np.pi*pulses*t/duration)**4
    envelope *= np.minimum(1,t/.04) * np.minimum(1,(duration-t)/.04)
    return carrier * envelope


def connected_vocal(intervals):
    """Articulations over an unbroken voiced floor: no silence to segment on."""
    centers = .3 + np.r_[0., np.cumsum(intervals)]
    t = np.arange(round((centers[-1] + .3) * SR)) / SR
    envelope = np.full_like(t, .055)
    for center in centers:
        envelope += .18 * np.exp(-.5 * ((t - center) / .045) ** 2)
    envelope *= np.minimum(1, t / .04) * np.minimum(1, (t[-1] - t) / .04)
    return np.sin(2 * np.pi * 220 * t) * envelope, centers


class PerformanceTests(unittest.TestCase):
    def test_continuous_flow_switch_splits_before_the_sixteen_syllable_cap(self):
        for before, after in [(.4, .18), (.18, .4)]:
            with self.subTest(before=before):
                y, centers = connected_vocal([before] * 5 + [(before + after) / 2] + [after] * 5)
                phrases = vocal_phrases(y, SR)
                changes = [p for p in phrases if p.get("boundary_source") == "cadence_change"]
                self.assertEqual(len(changes), 1)
                self.assertAlmostEqual(changes[0]["start_time"], (centers[5] + centers[6]) / 2, delta=.08)
                self.assertEqual(sum(p["target_syllables"] for p in phrases), 12)
                self.assertTrue(changes[0]["continuous_from_previous"])
                for left, right in zip(phrases, phrases[1:]):
                    self.assertEqual(left["end_time"], right["start_time"])

    def test_long_steady_mumble_gets_shorter_lines_without_waiting_for_silence(self):
        y, _ = connected_vocal([.4] * 13)
        phrases = vocal_phrases(y, SR, tempo=90)
        self.assertGreater(len(phrases), 1)
        self.assertEqual(sum(p["target_syllables"] for p in phrases), 14)
        self.assertTrue(all(p["end_time"] - p["start_time"] <= 3.34 for p in phrases))
        self.assertTrue(any(p.get("boundary_source") == "phrase_length" for p in phrases))

    def test_isolated_quick_syllable_does_not_become_a_flow_switch(self):
        y, _ = connected_vocal([.25] * 4 + [.17] + [.25] * 4)
        phrases = vocal_phrases(y, SR)
        self.assertFalse(any(p.get("boundary_source") == "cadence_change" for p in phrases))
        self.assertEqual(sum(p["target_syllables"] for p in phrases), 10)

    def test_long_held_run_is_not_cut_into_fake_bars(self):
        phrases = vocal_phrases(vocal(duration=7, run=True), SR)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(phrases[0]["target_syllables"], 1)

    def test_straight_to_swung_flow_can_change_without_changing_average_speed(self):
        y, centers = connected_vocal([.3] * 6 + [.18, .42] * 4)
        phrases = vocal_phrases(y, SR)
        changes = [p for p in phrases if p.get("boundary_source") == "cadence_change"]
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(changes[0]["start_time"], centers[6], delta=.2)
        self.assertEqual(sum(p["target_syllables"] for p in phrases), 15)

    def test_one_held_syllable_can_span_many_notes(self):
        y = vocal(run=True)
        phrases = vocal_phrases(y, SR)
        notes = [{"start_time": a, "end_time": b, "pitch_midi": p}
                 for a,b,p in [(0,.7,57),(.7,1.4,61),(1.4,2,64)]]
        attach_performance(phrases, notes)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(phrases[0]["target_syllables"], 1)
        self.assertEqual(phrases[0]["note_count"], 3)
        self.assertTrue(phrases[0]["syllable_slots"][0]["held"])

    def test_repeated_syllables_on_one_pitch_remain_separate(self):
        phrases = vocal_phrases(vocal(pulses=6), SR)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(phrases[0]["target_syllables"], 6)

    def test_silence_and_original_time_origin(self):
        self.assertEqual(vocal_phrases(np.zeros(SR), SR), [])
        y = np.r_[np.zeros(SR//2), vocal(1), np.zeros(SR//2), vocal(1)]
        phrases = vocal_phrases(y, SR)
        self.assertEqual(len(phrases), 2)
        self.assertAlmostEqual(phrases[0]["start_time"], .5, delta=.06)
        self.assertAlmostEqual(phrases[1]["start_time"], 2., delta=.06)
        self.assertLess(phrases[0]["end_time"], 1.6)

    def test_pitch_changes_without_onsets_are_still_notes(self):
        times = np.arange(100) * .032
        pitch = np.r_[np.full(30,220.), np.full(30,277.18), np.full(40,329.63)]
        notes = _build_note_map(np.zeros(51200), SR, np.array([]), np.array([]), 3.2,
                               pitch_track=(pitch, np.ones(100,dtype=bool), times))
        self.assertEqual(len(notes), 3)
        self.assertEqual([n["pitch_midi"] for n in notes], [57,61,64])
