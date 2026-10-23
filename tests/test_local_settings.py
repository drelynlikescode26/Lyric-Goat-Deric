import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from dotenv import dotenv_values
from app import app


class LocalSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / ".env"
        self.env = patch.dict(os.environ, {"LYRIC_GOAT_LOCAL_SETUP": "1"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = patch.dict(app.config, {"LOCAL_ENV_FILE": self.path,
            "TRUSTED_HOSTS": ["localhost", "127.0.0.1"], "TESTING": True})
        self.config.start()
        self.addCleanup(self.config.stop)
        self.client = app.test_client()

    def token(self):
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["setup_csrf"]

    def test_disabled_remote_and_rebound_hosts_cannot_configure_keys(self):
        with patch.dict(os.environ, {"LYRIC_GOAT_LOCAL_SETUP": "0"}):
            self.assertEqual(self.client.get("/settings").status_code, 404)
        self.assertEqual(self.client.get("/settings", environ_base={"REMOTE_ADDR": "192.168.1.2"}).status_code, 403)
        self.assertEqual(self.client.get("/settings", base_url="http://evil.example").status_code, 400)
        self.assertFalse(self.path.exists())

    def test_other_sites_cannot_trigger_paid_generation_on_localhost(self):
        with patch("app.generate_lyrics") as writer:
            response = self.client.post("/process", headers={"Origin": "https://evil.example"})
            self.assertEqual(response.status_code, 403)
            writer.assert_not_called()

    def test_cross_site_and_missing_csrf_posts_cannot_change_settings(self):
        data = {"api_key": "sk-" + "offline" * 5, "model": "gpt-6-luna"}
        self.assertEqual(self.client.post("/settings", data=data).status_code, 403)
        data["csrf"] = self.token()
        self.assertEqual(self.client.post("/settings", data=data, headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertFalse(self.path.exists())

    def test_key_is_private_persistent_and_never_returned_to_browser(self):
        key = "sk-" + "offline-test" * 5
        response = self.client.post("/settings", data={"csrf": self.token(), "api_key": key,
            "model": "gpt-6-luna"}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(key, response.text)
        self.assertIn("Saved.", response.text)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(dotenv_values(self.path)["OPENAI_API_KEY"], key)
        self.assertEqual(os.environ["OPENAI_API_KEY"], key)
        self.assertEqual(os.environ["LYRIC_GOAT_WRITER"], "openai")
        # Blank keeps a saved key while the chosen writer changes immediately.
        self.client.post("/settings", data={"csrf": self.token(), "api_key": "", "model": "gpt-5.4-mini"})
        self.assertEqual(dotenv_values(self.path)["OPENAI_API_KEY"], key)
        self.assertEqual(os.environ["OPENAI_GENERATION_MODEL"], "gpt-5.4-mini")

    def test_invalid_form_never_echoes_secret_or_uses_arbitrary_model(self):
        key = "sk-" + "offline-test" * 5
        response = self.client.post("/settings", data={"csrf": self.token(), "api_key": key,
            "model": "expensive-arbitrary-model"})
        self.assertNotIn(key, response.text)
        self.assertFalse(self.path.exists())
