import io
import json
import unittest
import numpy as np
import soundfile as sf
from types import SimpleNamespace
from unittest.mock import patch
from app import app
from test_performance import connected_vocal


def wav(samples):
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format='WAV')
    buffer.seek(0)
    return buffer


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch('app.listen_performance', return_value={"status": "local_only", "phrases": []})
    @patch('services.generate._request')
    def test_connected_mumble_reaches_writer_as_contiguous_shorter_lines(self, writer, listener):
        samples, _ = connected_vocal([.4] * 13)
        text = "I won't let you go home\nYou keep on calling out my name"
        writer.return_value = {k: text for k in ('melodic', 'rap', 'punchy')}
        response = self.client.post('/process', data={
            "audio": (wav(samples), 'connected.wav'), "hum_mode": "true", "manual_bpm": "90", "theme": "Leaving a one-sided friendship"})
        self.assertEqual(response.status_code, 200, response.json)
        phrases = response.json['phrase_map']
        self.assertEqual([p['target_syllables'] for p in phrases], [6, 8])
        self.assertEqual(phrases[0]['end_time'], phrases[1]['start_time'])
        self.assertTrue(phrases[1]['continuous_from_previous'])
        request = json.loads(writer.call_args_list[0].args[0])
        self.assertTrue(request['phrases'][1]['continuous_from_previous'])
        self.assertEqual(request['phrases'][1]['boundary_source'], 'phrase_length')
        self.assertEqual(request['theme'], 'Leaving a one-sided friendship')
        self.assertEqual(response.json['theme'], request['theme'])
        self.assertEqual(listener.call_count, 1)
        self.assertEqual(writer.call_count, 1)
        self.assertTrue(all(v['quality_status'] == 'checked' for v in response.json['versions']))

    @patch('app.listen_performance')
    @patch('app.generate_lyrics')
    def test_silence_returns_actionable_error_without_paid_calls(self, writer, listener):
        response=self.client.post('/process',data={"audio":(wav(np.zeros(16000)), 'silence.wav'),"hum_mode":"true"})
        self.assertEqual(response.status_code,422)
        self.assertIn('No vocal',response.json['error'])
        writer.assert_not_called(); listener.assert_not_called()

    @patch.dict('os.environ', {"OPENAI_API_KEY":"offline-test","LYRIC_GOAT_AUDIO_LISTENER":"auto"})
    @patch('services.listen.OpenAI')
    @patch('services.generate._request')
    def test_upload_to_lyrics_runs_automatically_on_synthetic_vocal(self, writer, listener):
        t=np.arange(32000)/16000
        samples=.2*np.sin(2*np.pi*220*t)*np.sin(4*np.pi*t/2)**4
        samples=np.r_[np.zeros(4800),samples,np.zeros(4000)]
        listener.return_value.chat.completions.create.return_value=SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps({"feel":["soft"],
                "phrases":[{"start_time":.35,"end_time":2.3,"syllables":4,"phonetic_hint":"mm mm mm oh","ending_vowel":"oh","heard_words":""}]})))])
        writer.return_value={k:"I won't go home" for k in ('melodic','rap','punchy')}
        response=self.client.post('/process',data={"audio":(wav(samples),'hum.wav'),"hum_mode":"true","genre":"rnb"})
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['listening']['status'],'ok')
        self.assertEqual(response.json['listening']['strategy'],'detailed')
        self.assertEqual(len(response.json['versions']),3)
        self.assertEqual(response.json['phrase_map'][0]['target_syllables'],4)
        # The smooth attack becomes audible after its nominal .3s start.
        self.assertAlmostEqual(response.json['phrase_map'][0]['start_time'],.3,delta=.10)
        self.assertEqual(response.json['phrase_map'][0]['ending_vowel'],'oh')
        self.assertTrue(all(v['quality_status']=='checked' for v in response.json['versions']))
        self.assertEqual(writer.call_count,1)

    @patch('app.generate_lyrics',return_value=[])
    def test_regeneration_keeps_audio_context_and_locked_lines(self, writer):
        flow={"phrase_map":[{"syllables":1}],"listening":{"feel":["aching"]}}
        locks={"melodic":{"0":"Go"}}
        result=self.client.post('/regenerate-all',json={"flow_data":flow,"locked_versions":locks,"theme":"Moving on","language":"explicit"})
        self.assertEqual(result.status_code,200)
        self.assertEqual(writer.call_args.kwargs['locked_versions'],locks)
        self.assertEqual(writer.call_args.kwargs['theme'],'Moving on')
        self.assertEqual(writer.call_args.kwargs['language'],'explicit')
        self.assertEqual(writer.call_args.args[1]['listening']['feel'],['aching'])

    @patch('app.generate_single_line', return_value='Go')
    def test_line_rewrite_receives_theme(self, writer):
        result = self.client.post('/regenerate-line', json={"bar_index":0,"syllable_count":1,
            "context_lines":["Stay"],"theme":"Moving on","language":"explicit"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(writer.call_args.kwargs['theme'], 'Moving on')
        self.assertEqual(writer.call_args.kwargs['language'], 'explicit')
