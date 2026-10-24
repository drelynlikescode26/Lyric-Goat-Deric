import base64
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from app import app
from services import comparisons as module


REFERENCE = [{"start_time": .1, "end_time": 1.8, "syllables": 2,
              "target_syllables": 2, "syllable_slots": []}]
DETAIL = {"summary": "Two soft articulations", "feel": ["soft"], "phrases": [
    {"start_time": .1, "end_time": 1.8, "syllables": 2, "phonetic_hint": "na oh", "heard_words": ""}]}


def take():
    t = np.arange(32000) / 16000
    output = io.BytesIO()
    sf.write(output, .2 * np.sin(2 * np.pi * 220 * t), 16000, format="WAV")
    output.seek(0)
    return output


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), "LYRIC_GOAT_LOCAL_SETUP": "1", "OPENAI_API_KEY": "sk-private-test",
                                          "GEMINI_API_KEY": "google-private-test"}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.config = patch.dict(app.config, {"COMPARISONS_DIR": self.root, "TESTING": True,
                                             "TRUSTED_HOSTS": ["localhost", "127.0.0.1"]})
        self.config.start(); self.addCleanup(self.config.stop)
        self.client = app.test_client()
        self.csrf = self.client.get('/api/comparisons/options').json['csrf']
        self.headers = {'X-Comparison-CSRF': self.csrf}
        module._active_runs.clear()
        if module._busy.locked(): module._busy.release()
        self.addCleanup(lambda: module._busy.release() if module._busy.locked() else None)

    def upload(self):
        with patch.object(module, 'preanalyze_audio', return_value={}), patch.object(module, 'analyze_flow', return_value={'phrase_map': REFERENCE}):
            response = self.client.post('/api/comparisons', headers=self.headers,
                                       data={'audio': (take(), 'mumble.wav'), 'bpm': '90'})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json

    def queue(self, identity, providers):
        with patch.object(module, 'Thread'):
            return self.client.post(f'/api/comparisons/{identity}/runs', headers=self.headers,
                                    json={'providers': providers})

    def test_upload_history_and_reopening_never_call_paid_models(self):
        with patch.object(module, 'OpenAI') as ai, patch.object(module.httpx, 'post') as google:
            report = self.upload()
            self.client.get('/compare')
            self.client.get('/api/comparisons')
            reopened = self.client.get(f'/api/comparisons/{report["id"]}').json
            response = self.client.get(f'/api/comparisons/{report["id"]}/audio')
            self.assertEqual(hashlib.sha256(response.data).hexdigest(), report['audio_sha256'])
            response.close()
            self.assertEqual(reopened['runs'], [])
            ai.assert_not_called(); google.assert_not_called()
        folder = self.root / report['id']
        for path in [folder / 'take.wav', folder / 'report.json']:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(set(p.name for p in folder.iterdir()), {'take.wav', 'report.json'})

    def test_private_routes_csrf_and_model_allowlist(self):
        self.assertEqual(self.client.post('/api/comparisons', data={'audio': (take(), 'm.wav')}).status_code, 403)
        self.assertEqual(self.client.get('/api/comparisons', environ_base={'REMOTE_ADDR': '192.168.1.5'}).status_code, 403)
        with patch.dict(os.environ, {'LYRIC_GOAT_LOCAL_SETUP': '0'}):
            self.assertEqual(self.client.get('/compare').status_code, 404)
        report = self.upload()
        self.assertEqual(self.queue(report['id'], ['untrusted-model']).status_code, 400)
        self.assertEqual(self.queue(report['id'], ['openai_detail', 'openai_detail']).status_code, 400)
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
            self.assertEqual(self.queue(report['id'], ['gemini']).status_code, 400)
        self.assertEqual(self.client.get('/api/comparisons/not-a-valid-id/audio').status_code, 404)

    def test_every_adapter_gets_the_identical_audio_and_no_fake_speech_timestamps(self):
        report = self.upload(); audio = self.root / report['id'] / 'take.wav'
        content = audio.read_bytes()
        heard = {'status': 'ok', 'phrases': [{'index': 0, 'confidence': .9, 'phonetic_hint': 'na oh',
                 'delivery': 'held ending', 'ending_vowel': 'oh', 'heard_words': ''}]}
        with patch.object(module, 'listen_performance', return_value=heard) as baseline, \
             patch.object(module, 'OpenAI') as ai, patch.object(module.httpx, 'post') as google:
            ai.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[
                SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(DETAIL)))])
            ai.return_value.audio.transcriptions.create.return_value = SimpleNamespace(text='No, oh')
            google.return_value.json.return_value = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(DETAIL)}]}}]}
            results = [module._one(option, audio, report) for option in module.provider_options() if option['id'] != 'vocalparse']
            self.assertTrue(all(r['status'] == 'ok' for r in results), results)
            baseline.assert_called_once_with(str(audio), report['reference'])
            kwargs = ai.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(base64.b64decode(kwargs['messages'][0]['content'][1]['input_audio']['data']), content)
            google_args = google.call_args.kwargs
            self.assertEqual(base64.b64decode(google_args['json']['contents'][0]['parts'][1]['inlineData']['data']), content)
            self.assertEqual(kwargs['messages'][0]['content'][0]['text'], google_args['json']['contents'][0]['parts'][0]['text'])
            self.assertEqual(ai.call_args.kwargs['max_retries'], 0)
        self.assertEqual(results[2]['phrases'], [])
        self.assertEqual({r['audio_sha256'] for r in results}, {report['audio_sha256']})
        self.assertEqual(results[1]['phrases'][0]['timing_source'], 'model_estimate')
        rendered = json.dumps(results)
        self.assertNotIn('sk-private-test', rendered)
        self.assertNotIn('google-private-test', rendered)

    def test_failures_do_not_become_successful_local_fallbacks(self):
        report = self.upload(); audio = self.root / report['id'] / 'take.wav'
        option = module.provider_options()[0]
        with patch.object(module, 'listen_performance', return_value={'status': 'local_only', 'phrases': []}):
            result = module._one(option, audio, report)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['phrases'], [])
        with patch.object(module, 'listen_performance', side_effect=RuntimeError('sk-private-test and private body')):
            result = module._one(option, audio, report)
        self.assertNotIn('sk-private-test', json.dumps(result))

    def test_invalid_model_times_are_rejected_instead_of_repaired_into_truth(self):
        for start, end in [(1., .5), (-1., 1.), (0., 999.), (float('nan'), 1.)]:
            payload = {'phrases': [dict(start_time=start, end_time=end, syllables=2)]}
            with self.assertRaises(ValueError): module._detail_result(json.dumps(payload), 2.)
        with self.assertRaises(ValueError):
            module._detail_result(json.dumps({'phrases': [DETAIL['phrases'][0], DETAIL['phrases'][0]]}), 2.)

    def test_reruns_keep_old_results_and_reviews_and_refresh_changed_detection(self):
        report = self.upload(); identity = report['id']; folder = self.root / identity
        run = self.queue(identity, ['openai_detail', 'transcribe']).json
        def result(option, audio, data):
            return dict(option, status='ok', transcript='test', phrases=[], elapsed_seconds=.1)
        with patch.object(module, '_one', side_effect=result):
            module._run(folder, report, run)
        review = self.client.post(f'/api/comparisons/{identity}/review', headers=self.headers, json={
            'run_id': run['id'], 'provider': 'openai_detail', 'preferred': True,
            'scores': {'flow': 4, 'sounds': 5, 'syllables': 3}, 'notes': 'Second flow starts earlier.'})
        self.assertEqual(review.status_code, 200)
        na_review = self.client.post(f'/api/comparisons/{identity}/review', headers=self.headers, json={
            'run_id': run['id'], 'provider': 'transcribe', 'scores': {'flow': None, 'sounds': 3, 'syllables': None}})
        self.assertEqual(na_review.status_code, 200)
        self.assertIsNone(na_review.json['runs'][0]['review']['ratings']['transcribe']['scores']['flow'])
        review = na_review
        first = review.json['runs'][0]
        second = self.queue(identity, ['openai_detail']).json
        latest = module._load(folder)
        with patch.object(module, '_one', side_effect=result), patch.object(module, 'preanalyze_audio', return_value={}) as pre, \
             patch.object(module, 'analyze_flow', return_value={'phrase_map': REFERENCE}):
            latest['reference_revision'] = 'old-setup'
            module._run(folder, latest, second)
            pre.assert_called_once()
        reopened = self.client.get(f'/api/comparisons/{identity}').json
        self.assertEqual(reopened['runs'][0], first)
        self.assertEqual(len(reopened['runs']), 2)
        self.assertEqual({r['audio_sha256'] for r in reopened['runs']}, {report['audio_sha256']})

    def test_interrupted_runs_are_visible_and_can_be_rerun(self):
        report = self.upload(); run = self.queue(report['id'], ['transcribe']).json
        module._active_runs.clear()  # simulate a server restart
        module._busy.release()
        reopened = self.client.get(f'/api/comparisons/{report["id"]}').json
        self.assertEqual(reopened['runs'][0]['status'], 'failed')
        self.assertIn('restarted', reopened['runs'][0]['error'])
        self.assertEqual(self.queue(report['id'], ['transcribe']).status_code, 202)

    def test_silence_is_rejected_without_creating_an_experiment(self):
        output = io.BytesIO(); sf.write(output, np.zeros(16000), 16000, format='WAV'); output.seek(0)
        response = self.client.post('/api/comparisons', headers=self.headers, data={'audio': (output, 'silence.wav')})
        self.assertEqual(response.status_code, 422)
        self.assertIn('No vocal', response.json['error'])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_artist_boundary_marks_roundtrip_without_ai_or_audio_changes(self):
        report=self.upload();identity=report['id'];before=(self.root/identity/'take.wav').read_bytes()
        with patch.object(module,'OpenAI') as ai,patch.object(module.httpx,'post') as google:
            response=self.client.post(f'/api/comparisons/{identity}/boundaries',headers=self.headers,
                json={'cuts':[.5,1.25],'revision':0,'audio_sha256':report['audio_sha256']})
            self.assertEqual(response.status_code,200,response.json);ai.assert_not_called();google.assert_not_called()
        reopened=self.client.get(f'/api/comparisons/{identity}').json
        self.assertEqual(reopened['boundary_reference']['cuts'],[.5,1.25]);self.assertEqual(reopened['boundary_reference']['source'],'artist_marks')
        self.assertEqual((self.root/identity/'take.wav').read_bytes(),before);self.assertEqual(reopened['runs'],[])
        cleared=self.client.post(f'/api/comparisons/{identity}/boundaries',headers=self.headers,
            json={'cuts':[],'revision':1,'audio_sha256':report['audio_sha256']})
        self.assertEqual(cleared.status_code,200);self.assertEqual(cleared.json['boundary_reference']['cuts'],[])

    def test_boundary_marks_reject_wrong_take_stale_edits_and_invalid_times(self):
        report=self.upload();identity=report['id'];endpoint=f'/api/comparisons/{identity}/boundaries'
        payload={'cuts':[.5],'revision':0,'audio_sha256':report['audio_sha256']}
        self.assertEqual(self.client.post(endpoint,json=payload).status_code,403)
        for cuts in [[True],[float('nan')],[-.1],[0],[2.1],[1,.5],[.5,.51],['.5'],[.1]*65]:
            response=self.client.post(endpoint,headers=self.headers,json={**payload,'cuts':cuts})
            self.assertEqual(response.status_code,400,response.json)
        self.assertEqual(self.client.post(endpoint,headers=self.headers,json={**payload,'audio_sha256':'different'}).status_code,409)
        self.assertEqual(self.client.post(endpoint,headers=self.headers,json={**payload,'revision':1}).status_code,409)
        self.assertEqual(self.client.post(endpoint,headers=self.headers,json=payload).status_code,200)
        self.assertEqual(self.client.post(endpoint,headers=self.headers,json=payload).status_code,409)
