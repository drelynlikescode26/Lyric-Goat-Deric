import unittest
from unittest.mock import patch
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
