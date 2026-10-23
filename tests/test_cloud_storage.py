import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from dotenv import dotenv_values
from services import cloud_storage as storage
from app import app


class FakeR2:
    def __init__(self): self.objects = {}; self.fail = False; self.after_put = None
    def put_object(self, **args):
        if self.fail: raise RuntimeError('private-provider-error')
        self.objects[args['Key']] = (args['Body'], args.get('Metadata', {}))
        if self.after_put: self.after_put(args['Key'])
    def get_object(self, **args):
        data, metadata = self.objects[args['Key']]
        return {'Body': io.BytesIO(data), 'ContentLength': len(data), 'Metadata': metadata}
    def head_object(self, **args):
        data, metadata = self.objects[args['Key']]
        return {'ContentLength': len(data), 'Metadata': metadata}
    def delete_object(self, **args): self.objects.pop(args['Key'], None)
    def get_paginator(self, name):
        owner = self
        class Paginator:
            def paginate(self, **args):
                return [{'Contents': [{'Key': k} for k in owner.objects if k.startswith(args['Prefix'])]}]
        return Paginator()


class CloudStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.remote = FakeR2()
        self.library = storage.Library(self.root, self.remote, 'private-test', cache_bytes=20)
        self.addCleanup(self.library.db.close)
        self.audio_key = 'audio/aaaaaaaaaaaa_bbbbbbbbbbbb_0123456789abcdef.wav'
    def file(self, key, content):
        path = self.root / key; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content); return path

    def test_only_library_files_can_be_uploaded(self):
        for key in ['.env', '../.env', '/audio/a.wav', 'songs/../../.env', 'songs/not-an-id.json', 'history/private', 'comparisons/x/take.wav']:
            with self.assertRaises(storage.StorageError): storage.valid_key(key)

    def test_failed_upload_retains_original_and_durable_pending_save(self):
        path = self.file(self.audio_key, b'vocal take'); self.library.enqueue(path, self.audio_key)
        self.remote.fail = True
        self.assertFalse(self.library.sync_once()); self.assertEqual(path.read_bytes(), b'vocal take')
        self.assertEqual(self.library.status()['pending'], 1)
        self.assertNotIn('private-provider-error', self.library.status()['error'])
        reopened = storage.Library(self.root, self.remote, 'private-test'); self.addCleanup(reopened.db.close)
        self.remote.fail = False; self.assertTrue(reopened.sync_once())
        self.assertEqual(reopened.status()['pending'], 0)

    def test_verified_audio_can_be_evicted_and_reopened_byte_for_byte(self):
        content = b'original performance'; path = self.file(self.audio_key, content)
        self.library.enqueue(path, self.audio_key); self.assertTrue(self.library.sync_once())
        self.assertEqual(self.library.prune(clear=True), len(content)); self.assertFalse(path.exists())
        self.assertEqual(self.library.ensure(path, self.audio_key).read_bytes(), content)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_audio_in_use_pending_or_modified_is_never_evicted(self):
        path = self.file(self.audio_key, b'take'); self.library.enqueue(path, self.audio_key)
        self.assertEqual(self.library.prune(clear=True), 0)
        self.library.sync_once()
        with self.library.pin(path): self.assertEqual(self.library.prune(clear=True), 0)
        path.write_bytes(b'new unsynced performance')
        self.assertEqual(self.library.prune(clear=True), 0); self.assertTrue(path.exists())

    def test_bad_download_is_not_installed_as_a_valid_recording(self):
        self.remote.objects[storage.PREFIX+self.audio_key] = (b'corrupt', {'sha256': 'not-the-checksum'})
        path = self.root / self.audio_key
        with self.assertRaises(storage.StorageError): self.library.ensure(path, self.audio_key)
        self.assertFalse(path.exists())

    def test_streaming_audio_holds_a_pin_until_wsgi_closes_the_response(self):
        path=self.file(self.audio_key,b'audio stream');self.library.enqueue(path,self.audio_key);self.library.sync_once()
        with patch.object(storage,'library',return_value=self.library),app.test_request_context('/audio'):
            response=storage.audio_response(path,mimetype='audio/wav',key=self.audio_key)
            self.assertEqual(self.library.prune(clear=True),0)
            environ={'REQUEST_METHOD':'GET'}
            iterable=response.get_app_iter(environ)
            self.assertEqual(b''.join(iterable),b'audio stream')
            iterable.close()
            self.assertEqual(self.library.prune(clear=True),len(b'audio stream'))

    def test_server_must_confirm_checksum_before_local_cleanup(self):
        path = self.file(self.audio_key, b'take'); self.library.enqueue(path, self.audio_key)
        with patch.object(self.remote, 'head_object', return_value={'ContentLength': 4, 'Metadata': {}}):
            self.assertFalse(self.library.sync_once())
        self.assertEqual(self.library.prune(clear=True), 0); self.assertTrue(path.exists())

    def test_newer_edits_during_upload_remain_pending(self):
        key='songs/aaaaaaaaaaaa.json'; path=self.file(key,b'{"lyrics":"first"}')
        self.library.enqueue(path,key)
        def newer(remote_key):
            if remote_key == storage.PREFIX+key:
                self.remote.after_put=None; path.write_bytes(b'{"lyrics":"second"}'); self.library.enqueue(path,key)
        self.remote.after_put=newer
        self.assertTrue(self.library.sync_once()); self.assertEqual(self.library.status()['pending'],1)
        self.assertTrue(self.library.sync_once()); self.assertEqual(self.library.status()['pending'],0)
        self.assertEqual(self.remote.objects[storage.PREFIX+key][0],path.read_bytes())
        self.assertEqual(len([k for k in self.remote.objects if '/history/' in k]),2)

    def test_catalog_restores_metadata_and_respects_deletions_and_existing_edits(self):
        key='songs/aaaaaaaaaaaa.json'; path=self.file(key,b'{"lyrics":"first"}')
        self.library.enqueue(path,key); self.library.sync_once(); path.unlink()
        self.assertEqual(self.library.restore_catalog(),1)
        path.write_bytes(b'{"lyrics":"local draft"}'); self.assertEqual(self.library.restore_catalog(),0)
        self.assertEqual(json.loads(path.read_text())['lyrics'],'local draft')
        self.library.tombstone(key); self.library.sync_once(); path.unlink()
        fresh=storage.Library(self.root,self.remote,'private-test','fresh'); self.addCleanup(fresh.db.close)
        self.assertEqual(fresh.restore_catalog(),0); self.assertFalse(path.exists())
        self.assertIn(storage.PREFIX+key,self.remote.objects)

    def test_connection_probe_uses_only_disposable_text(self):
        with patch.object(storage,'make_client',return_value=self.remote): storage.probe({'R2_BUCKET':'private-test'})
        self.assertEqual(self.remote.objects,{})


class CloudSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.env_path=Path(self.temp.name)/'.env'
        env=patch.dict(os.environ,{'LYRIC_GOAT_LOCAL_SETUP':'1'},clear=True);env.start();self.addCleanup(env.stop)
        config=patch.dict(app.config,{'TESTING':True,'LOCAL_ENV_FILE':self.env_path,'TRUSTED_HOSTS':['localhost','127.0.0.1']});config.start();self.addCleanup(config.stop)
        self.client=app.test_client()
    def token(self):
        self.client.get('/storage')
        with self.client.session_transaction() as s: return s['storage_csrf']
    def test_storage_setup_is_private_and_csrf_protected(self):
        self.assertEqual(self.client.post('/storage',data={}).status_code,403)
        self.assertEqual(self.client.get('/storage',environ_base={'REMOTE_ADDR':'192.168.1.3'}).status_code,403)
        self.assertEqual(self.client.post('/storage',data={'csrf':self.token()},headers={'Origin':'https://evil.example'}).status_code,403)
        self.assertFalse(self.env_path.exists())
    def test_failed_verification_does_not_save_credentials_or_enable_cloud(self):
        with patch.object(storage,'probe',side_effect=storage.StorageError('Connection failed')):
            response=self.client.post('/storage',data={'csrf':self.token(),'R2_ACCESS_KEY_ID':'a'*32,'R2_SECRET_ACCESS_KEY':'b'*64})
        self.assertIn('Connection failed',response.text);self.assertNotIn('b'*64,response.text)
        self.assertFalse(self.env_path.exists());self.assertFalse(storage.configured())
    def test_verified_credentials_are_private_and_migration_starts_once(self):
        values={'R2_ACCOUNT_ID':'a'*32,'R2_BUCKET':'lyric-goat-private','R2_ACCESS_KEY_ID':'c'*32,'R2_SECRET_ACCESS_KEY':'d'*64}
        with patch.object(storage,'probe') as probe,patch.object(storage,'migrate_saved',return_value=2) as migrate,patch.object(storage,'status',return_value={'enabled':False,'pending':0,'error':''}):
            response=self.client.post('/storage',data={'csrf':self.token(),**values},follow_redirects=True)
        probe.assert_called_once_with(values);migrate.assert_called_once()
        self.assertNotIn(values['R2_SECRET_ACCESS_KEY'],response.text)
        self.assertEqual(dotenv_values(self.env_path)['LYRIC_GOAT_STORAGE'],'r2')
        self.assertEqual(stat.S_IMODE(self.env_path.stat().st_mode),0o600)
