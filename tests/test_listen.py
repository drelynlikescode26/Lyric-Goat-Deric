import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import soundfile as sf
from services import listen
from services.listen import (listen_acoustic_performance as listen_performance,
                             apply_listening, _validated)


class ListeningTests(unittest.TestCase):
    @patch.dict('os.environ', {"OPENAI_API_KEY": "offline-test", "LYRIC_GOAT_AUDIO_LISTENER": "auto"})
    @patch('services.listen.OpenAI')
    def test_actual_audio_is_sent_and_timing_is_not_overwritten(self, client):
        payload = {"input_type":"mumble", "feel":["aching"], "phrases":[
            {"index":0,"phonetic_hint":"da oh","ending_vowel":"oh", "confidence":.9,
             "delivery":"held ending","heard_words":"", "start_time":999}]}
        client.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d)/'test.wav'; audio.write_bytes(b'RIFF-offline-fixture')
            phrases=[{"start_time":.5,"end_time":2.,"syllables":2,"target_syllables":2}]
            result=listen_performance(str(audio),phrases)
        kwargs=client.return_value.chat.completions.create.call_args.kwargs
        part=kwargs['messages'][0]['content'][1]['input_audio']
        self.assertEqual(base64.b64decode(part['data']), b'RIFF-offline-fixture')
        self.assertEqual(kwargs['modalities'], ['text'])
        flow=apply_listening({"phrase_map":phrases},result)
        self.assertEqual(flow['phrase_map'][0]['start_time'], .5)
        self.assertEqual(flow['phrase_map'][0]['target_syllables'], 2)
        self.assertEqual(flow['phrase_map'][0]['ending_vowel'],'oh')

    @patch.dict('os.environ', {}, clear=True)
    @patch('services.listen.OpenAI')
    def test_no_key_falls_back_without_model_call(self, client):
        self.assertEqual(listen_performance('unused.wav',[])['status'],'local_only')
        client.assert_not_called()

    def test_invalid_indices_and_confidence_never_become_targets(self):
        result=_validated({"phrases":[{"index":99},{"index":0,"confidence":float('nan'),"ending_vowel":"oh"}]},1,'test')
        self.assertEqual(len(result['phrases']),1)
        self.assertEqual(result['phrases'][0]['ending_vowel'],'unknown')

    def test_only_confident_plausible_articulations_adjust_counts(self):
        flow={"phrase_map":[{"start_time":.5,"end_time":2.,"syllables":1,"target_syllables":1}]}
        payload={"input_type":"mumble","phrases":[{"index":0,"confidence":.9,
                    "syllables":3,"count_confidence":.95}]}
        result=apply_listening(flow,_validated(payload,1,'test'))
        phrase=result['phrase_map'][0]
        self.assertEqual(phrase['target_syllables'],3)
        self.assertEqual((phrase['start_time'],phrase['end_time']),(.5,2.))
        self.assertEqual(phrase['acoustic_syllables'],1)
        self.assertEqual(phrase['syllable_slots'][0]['timing_source'],'estimated_subdivision')
        payload['phrases'][0]['count_confidence']=.4
        other={"phrase_map":[{"start_time":.5,"end_time":2.,"syllables":1}]}
        self.assertEqual(apply_listening(other,_validated(payload,1,'test'))['syllable_count'],1)

    @patch.dict('os.environ', {"OPENAI_API_KEY":"offline-test","LYRIC_GOAT_AUDIO_LISTENER":"auto"})
    @patch('services.listen.OpenAI', side_effect=RuntimeError('provider failed'))
    def test_provider_failure_remains_visible(self, client):
        p=[{"start_time":0,"end_time":1,"syllables":1}]
        result=listen_performance('unused.wav',p)
        self.assertEqual(result['status'],'local_only')
        self.assertEqual(result['reason'],'listening_unavailable')


class DetailedListeningTests(unittest.TestCase):
    @patch.dict('os.environ', {'OPENAI_API_KEY':'offline-test','LYRIC_GOAT_AUDIO_LISTENER':'auto'})
    @patch('services.listen.OpenAI')
    def test_default_uses_detailed_sounds_and_own_continuous_windows(self, client):
        detail = {'summary':'Wordless repeated sounds','feel':['soft'], 'phrases':[
            {'start_time':.1,'end_time':1.,'syllables':3,'phonetic_hint':'na na oh','heard_words':''},
            {'start_time':1.,'end_time':1.9,'syllables':4,'phonetic_hint':'da ni na oh','heard_words':''}]}
        client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(detail)))])
        reference = [{'start_time':.1,'end_time':1.9,'syllables':2,'target_syllables':2}]
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d)/'test.wav'
            sf.write(audio, np.zeros(32000), 16000)
            result = listen.listen_performance(str(audio), reference)
            part = client.return_value.chat.completions.create.call_args.kwargs['messages'][0]['content'][1]['input_audio']
            self.assertEqual(base64.b64decode(part['data']),audio.read_bytes())
        self.assertEqual(result['strategy'],'detailed')
        flow = apply_listening({'phrase_map':reference},result)
        self.assertEqual([p['target_syllables'] for p in flow['phrase_map']],[3,4])
        self.assertEqual(flow['phrase_map'][1]['phonetic_hint'],'da ni na oh')
        self.assertTrue(flow['phrase_map'][1]['continuous_from_previous'])
        self.assertEqual(flow['phrase_map'][0]['syllable_slots'],[])
        self.assertEqual(flow['phrase_map'][1]['max_syllables'],4)
        self.assertEqual(flow['syllable_count'],7)
        self.assertEqual(len(flow['debug_phrases']),2)

    def test_uncertain_count_uses_labeled_acoustic_estimate_without_changing_sound(self):
        detail = listen._detail_result(json.dumps({'phrases':[
            {'start_time':0,'end_time':2,'syllables':None,'phonetic_hint':'na nee na oh'}]}), 2)
        flow = apply_listening({'phrase_map':[{'start_time':0,'end_time':2,'syllables':4}]},
                              dict(detail, status='ok',strategy='detailed'))
        p = flow['phrase_map'][0]
        self.assertEqual(p['target_syllables'],4)
        self.assertEqual(p['syllable_source'],'acoustic_estimate')
        self.assertEqual(p['phonetic_hint'],'na nee na oh')

    @patch.dict('os.environ', {'OPENAI_API_KEY':'offline-test','LYRIC_GOAT_AUDIO_LISTENER':'auto'})
    @patch('services.listen.OpenAI')
    def test_truncated_detail_never_replaces_acoustic_phrases(self, client):
        client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='length',message=SimpleNamespace(content='{}'))])
        with tempfile.TemporaryDirectory() as d:
            audio=Path(d)/'test.wav'; sf.write(audio,np.zeros(16000),16000)
            result=listen.listen_performance(str(audio),[{'start_time':0,'end_time':1,'syllables':1}])
        self.assertEqual(result['status'],'local_only')
        self.assertEqual(result['reason'],'listening_unavailable')
