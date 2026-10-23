import unittest
from services.rhyme_radar import analyze

class RadarTests(unittest.TestCase):
    def test_sound_not_spelling_and_internal_end_links(self):
        result=analyze('Through the night I hold the light\nI get it right')
        group=next(f for f in result['families'] if {'night','light','right'}<=set(f['examples']))
        self.assertEqual(group['label'],'Internal / end rhyme')
        self.assertTrue(result['lines'][0]['hits'])

    def test_multisyllable_sound_and_separate_repetition(self):
        result=analyze('I feel the motion\nI cross the ocean\nI cross the ocean')
        self.assertTrue(any(f['kind']=='multi' for f in result['families']))
        repeated=analyze('A light\nA light');self.assertFalse(repeated['families']);self.assertTrue(repeated['repetitions'])

    def test_cross_word_multisyllable_clue_does_not_color_unmatched_prefixes(self):
        result=analyze('I take the highway\nI go my way')
        multi=next(f for f in result['families'] if f['kind']=='multi')
        self.assertIn('my way',multi['examples']);self.assertIn('highway',multi['examples'])
        self.assertNotIn('the highway',multi['examples'])

    def test_slant_is_tentative_and_unknown_is_not_guessed(self):
        result=analyze('I carry hope\nI go home\nBrrmzzzz grixxy')
        self.assertTrue(any(f['kind']=='near' for f in result['families']))
        self.assertIn('grixxy',result['unknown']);self.assertFalse(result['lines'][2]['hits'])

    def test_unicode_indices_point_to_the_original_word(self):
        result=analyze('🔥 A bright light\nI stay all night')
        for row in result['lines']:
            for hit in row['hits']:self.assertEqual(row['text'][hit['start']:hit['end']],hit['text'])
