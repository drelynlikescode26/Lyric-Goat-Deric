import unittest
from unittest.mock import patch
import json
from services.generate import generate_lyrics, generate_single_line, GenerationError


class GenerationFitTests(unittest.TestCase):
    def setUp(self):
        self.flow={"phrase_map":[{"syllables":4,"target_syllables":4,"max_words":4,"max_syllables":5}]}

    @patch('services.generate._request')
    def test_repair_preserves_good_versions_and_locked_lines(self, request):
        request.side_effect=[
            {"melodic":"I keep moving","rap":"Chasing dreams","punchy":"I won't fold now"},
            {"melodic":"The model changed this","rap":"You left last night","punchy":"Another wrong change"}]
        result=generate_lyrics('',self.flow,locked_versions={"melodic":{"0":"Don't call me back"}})
        by_name={v['name']:v for v in result}
        self.assertEqual(by_name['melodic']['lyrics'],"Don't call me back")
        self.assertEqual(by_name['punchy']['lyrics'],"I won't fold now")
        self.assertEqual(by_name['rap']['lyrics'],"You left last night")
        self.assertEqual(request.call_count,2)

    @patch('services.generate._request')
    def test_bad_line_count_is_not_returned_as_synced_lyrics(self, request):
        request.return_value={k:"First line\nSecond line" for k in ('melodic','rap','punchy')}
        with self.assertRaises(GenerationError):
            generate_lyrics('',self.flow)
        self.assertEqual(request.call_count,2)

    @patch('services.generate._request')
    def test_single_locked_line_never_calls_model(self, request):
        result=generate_single_line(0,4,['Keep this exact'],{'0':'Keep this exact'},'',self.flow,
                                    'melodic','hook','sad','cadence','auto','rnb')
        self.assertEqual(result,'Keep this exact')
        request.assert_not_called()

    @patch('services.generate._request')
    def test_failed_repair_is_reported_and_bounded(self, request):
        request.side_effect=[{k:'Chasing dreams' for k in ('melodic','rap','punchy')},RuntimeError('offline')]
        results=generate_lyrics('',self.flow)
        self.assertTrue(all(v['quality_status']=='needs_review' and v['repair_unavailable'] for v in results))
        self.assertEqual(request.call_count,2)

    @patch('services.generate._request')
    def test_wordless_sound_and_theme_reach_writer_without_guessed_words(self, request):
        self.flow['phrase_map'][0].update(phonetic_hint='nah na nah oh', heard_words='I know your name')
        request.return_value = {k:"I won't go home" for k in ('melodic','rap','punchy')}
        generate_lyrics('hallucinated transcript', self.flow, theme='Leaving a friendship')
        payload = json.loads(request.call_args.args[0])
        self.assertEqual(payload['theme'], 'Leaving a friendship')
        self.assertEqual(payload['phrases'][0]['sound'], 'nah na nah oh')
        self.assertNotIn('heard_words', payload['phrases'][0])
        self.assertEqual(payload['rough_transcript'], '')
        self.assertEqual(payload['phrases'][0]['max_syllables'], 4)
        self.assertEqual(request.call_count, 1)

    @patch('services.generate._request')
    def test_one_missing_or_extra_syllable_requires_repair_instead_of_passing(self, request):
        for wrong in ('You let go', 'You let me go home'):
            request.reset_mock()
            request.side_effect = [{k:wrong for k in ('melodic','rap','punchy')},
                                   {k:"I won't go home" for k in ('melodic','rap','punchy')}]
            result = generate_lyrics('', self.flow, theme='Letting go')
            self.assertEqual(request.call_count, 2)
            self.assertTrue(all(v['lyrics']=="I won't go home" and v['quality_status']=='checked' for v in result))
            repair = json.loads(request.call_args.args[0])
            self.assertEqual(repair['original_request']['theme'], 'Letting go')

    @patch('services.generate._request')
    def test_line_rewrite_keeps_theme_and_exact_pocket(self, request):
        request.side_effect = [{k:'You let go' for k in ('melodic','rap','punchy')},
                               {k:"I won't go home" for k in ('melodic','rap','punchy')}]
        line = generate_single_line(0,4,['Stay right by me'],{},'',self.flow,
            'melodic','hook','sad','cadence','auto','rnb',theme='Choosing to leave')
        self.assertEqual(line,"I won't go home")
        self.assertEqual(request.call_count,2)
        payload = json.loads(request.call_args_list[0].args[0])
        self.assertEqual(payload['theme'],'Choosing to leave')
        self.assertEqual(payload['phrase']['max_syllables'],4)
