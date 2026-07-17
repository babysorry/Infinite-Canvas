import os
import unittest
from unittest.mock import patch

import main


class CodexCliIntegrationTests(unittest.TestCase):
    def test_proxy_url_validation(self):
        self.assertEqual(
            main.normalize_proxy_url("http://127.0.0.1:7897/"),
            "http://127.0.0.1:7897",
        )
        self.assertEqual(
            main.normalize_proxy_url("socks5://127.0.0.1:7897"),
            "socks5://127.0.0.1:7897",
        )
        with self.assertRaises(main.HTTPException):
            main.normalize_proxy_url("127.0.0.1:7897")

    def test_codex_subprocess_inherits_resolved_proxy(self):
        proxy = "http://127.0.0.1:7897"
        with patch.dict(os.environ, {}, clear=True), patch.object(
            main,
            "codex_cli_proxy_settings",
            return_value={"http": proxy, "https": proxy, "source": "provider"},
        ):
            env = main.codex_cli_subprocess_env()

        self.assertEqual(env["HTTP_PROXY"], proxy)
        self.assertEqual(env["HTTPS_PROXY"], proxy)
        self.assertEqual(env["http_proxy"], proxy)
        self.assertEqual(env["https_proxy"], proxy)

    def test_native_image_feature_detection(self):
        feature_text = """
image_generation stable true
other_feature experimental false
"""
        self.assertTrue(main.codex_feature_enabled(feature_text))
        self.assertFalse(main.codex_feature_enabled(feature_text, "other_feature"))

    def test_native_channel_and_quality_normalization(self):
        self.assertEqual(main.codex_image_channel(), "native")
        self.assertEqual(main.normalize_codex_image_quality("HIGH"), "high")
        self.assertEqual(main.normalize_codex_image_quality("unsupported"), "auto")

    def test_codex_model_catalog_payload(self):
        payload = main.codex_models_payload(
            chat_models=["gpt-5.5", "gpt-5.4"],
            model_names={"gpt-5.5": "GPT 5.5"},
            catalog_source="refreshed",
        )

        self.assertEqual(payload["chat_models"], ["gpt-5.5", "gpt-5.4"])
        self.assertEqual(payload["model_names"], {"gpt-5.5": "GPT 5.5"})
        self.assertEqual(payload["image_models"], ["gpt-image-2"])
        self.assertEqual(payload["video_models"], [])

    def test_codex_json_events_extract_thread(self):
        stdout = "\n".join(
            [
                "not json",
                '{"type":"thread.started","thread_id":"thread_12345678"}',
                '{"type":"item.completed","item":{"type":"agent_message"}}',
            ]
        )

        self.assertEqual(main.codex_thread_id_from_events(stdout), "thread_12345678")
        self.assertEqual(len(main.codex_jsonl_objects(stdout)), 2)


if __name__ == "__main__":
    unittest.main()
