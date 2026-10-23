import unittest
from services.lyric_craft import (count_syllables, rhyme_match, clean_lyrics,
    validate_lyrics, line_syllable_options, ending_vowels)


class LyricCraftTests(unittest.TestCase):
    def test_syllabic_le_and_contractions(self):
        for word in ["bottle", "little", "people"]:
            self.assertEqual(count_syllables(word), 2)
        self.assertEqual(count_syllables("don't"), 1)
        self.assertEqual(count_syllables("tryna"), 2)

    def test_rhyme_uses_sound_not_spelling(self):
        self.assertEqual(rhyme_match("love", "move"), 0.)
        self.assertEqual(rhyme_match("love", "above"), 1.)
        self.assertEqual(rhyme_match("rain", "pain"), 1.)
        self.assertIn("oh", ending_vowels("I wanna go"))
        self.assertIn("ee", ending_vowels("lonely"))
        self.assertIn("oh", ending_vowels("tomorrow"))

    def test_alternate_pronunciations_are_not_forced_to_one_count(self):
        self.assertTrue({2,3} & line_syllable_options("every"))

    def test_stock_phrases_and_structural_misses_are_flagged(self):
        phrase = {"syllables": 4, "target_syllables": 4, "max_words": 4, "max_syllables": 5}
        codes = {x["code"] for x in validate_lyrics("Chasing dreams", [phrase, phrase])}
        self.assertIn("line_count", codes)
        self.assertIn("cliche", codes)

    def test_no_decorative_dashes_or_markdown_bullets(self):
        self.assertEqual(clean_lyrics('- I stayed\u2014you left\n2. Don\u2019t call'), "I stayed you left\nDon't call")
