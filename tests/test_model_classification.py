import asyncio
import unittest
from unittest.mock import patch

import main
from fastapi import Response


class UpstreamModelClassificationTests(unittest.TestCase):
    def test_seedance_mini_is_video(self):
        self.assertEqual(main.classify_upstream_model("seedance-2.0-mini"), "video")

    def test_doubao_seedance_is_video(self):
        self.assertEqual(
            main.classify_upstream_model("doubao-seedance-2-0-fast-260128"),
            "video",
        )

    def test_grok_imagine_video_is_video(self):
        self.assertEqual(main.classify_upstream_model("grok-imagine-video"), "video")

    def test_regular_chat_model_stays_chat(self):
        self.assertEqual(main.classify_upstream_model("gpt-5.5"), "chat")

    def test_seedream_model_stays_image(self):
        self.assertEqual(main.classify_upstream_model("seedream-4.5"), "image")

    def test_upstream_models_are_grouped_without_cross_category_regressions(self):
        grouped, model_ids = main.parse_upstream_models(
            {
                "data": [
                    {"id": "seedance-2.0-mini"},
                    {"id": "doubao-seedance-1-0-pro-250528"},
                    {"id": "grok-imagine-video"},
                    {"id": "grok-imagine-video-reference"},
                    {"id": "gpt-5.5"},
                    {"id": "seedream-4.5"},
                ]
            }
        )

        self.assertEqual(
            grouped,
            {
                "image": ["seedream-4.5"],
                "chat": ["gpt-5.5"],
                "video": [
                    "doubao-seedance-1-0-pro-250528",
                    "grok-imagine-video",
                    "grok-imagine-video-reference",
                    "seedance-2.0-mini",
                ],
                "audio": [],
            },
        )
        self.assertEqual(model_ids, sorted(model_ids))

    def test_upstream_category_takes_priority_over_model_name_guessing(self):
        grouped, _ = main.parse_upstream_models(
            {
                "data": [
                    {
                        "id": "looks-like-video",
                        "category": "audio",
                        "display_name": "Dialogue",
                        "description": "Multi-role speech",
                    },
                    {
                        "id": "minimax-speech-2.6-hd",
                        "category": "audio",
                    },
                ]
            }
        )

        self.assertEqual(grouped["audio"], ["looks-like-video", "minimax-speech-2.6-hd"])
        self.assertEqual(grouped["video"], [])

    def test_model_metadata_is_preserved_from_models_protocol(self):
        raw = {
            "data": [
                {
                    "id": "minimax-speech-2.6-hd",
                    "display_name": "MiniMax Speech 2.6 HD",
                    "description": "中文与多语言高质量旁白和配音。",
                    "category": "audio",
                    "input_modalities": ["text"],
                    "output_modalities": ["audio"],
                    "capabilities": ["text-to-speech", "voice-control"],
                }
            ]
        }

        metadata = main.upstream_model_metadata(raw)

        self.assertEqual(metadata["minimax-speech-2.6-hd"]["category"], "audio")
        self.assertEqual(metadata["minimax-speech-2.6-hd"]["display_name"], "MiniMax Speech 2.6 HD")
        self.assertEqual(metadata["minimax-speech-2.6-hd"]["description_override"], "")


class SavedProviderCompatibilityTests(unittest.TestCase):
    def test_misclassified_seedance_is_exposed_to_canvas_as_video(self):
        provider = main.normalize_provider(
            {
                "id": "local-fal",
                "name": "Fal.ai 本地中转",
                "base_url": "http://127.0.0.1:8787/v1",
                "protocol": "openai",
                "image_models": ["seedream-4.5"],
                "chat_models": ["gpt-5.5", "seedance-2.0-mini"],
                "video_models": ["grok-imagine-video"],
            }
        )

        self.assertEqual(provider["image_models"], ["seedream-4.5"])
        self.assertEqual(provider["chat_models"], ["gpt-5.5"])
        self.assertEqual(
            provider["video_models"],
            ["grok-imagine-video", "seedance-2.0-mini"],
        )

    def test_audio_metadata_migrates_models_out_of_legacy_chat_list(self):
        provider = main.normalize_provider(
            {
                "id": "local-fal",
                "name": "Local Fal",
                "base_url": "http://127.0.0.1:8787/v1",
                "chat_models": [
                    "gpt-5.5",
                    "eleven-v3-dialogue",
                    "minimax-speech-2.6-hd",
                ],
                "audio_models": [],
                "model_metadata": {
                    "eleven-v3-dialogue": {
                        "id": "eleven-v3-dialogue",
                        "category": "audio",
                        "output_modalities": ["audio"],
                    },
                    "minimax-speech-2.6-hd": {
                        "id": "minimax-speech-2.6-hd",
                        "category": "chat",
                        "output_modalities": ["audio"],
                    },
                },
            }
        )

        self.assertEqual(provider["chat_models"], ["gpt-5.5"])
        self.assertEqual(
            provider["audio_models"],
            ["eleven-v3-dialogue", "minimax-speech-2.6-hd"],
        )
        self.assertEqual(provider["default_audio_model"], "eleven-v3-dialogue")

    def test_video_with_audio_output_is_not_migrated_to_audio(self):
        provider = main.normalize_provider(
            {
                "id": "local-fal",
                "name": "Local Fal",
                "base_url": "http://127.0.0.1:8787/v1",
                "chat_models": ["multimodal-generator"],
                "model_metadata": {
                    "multimodal-generator": {
                        "id": "multimodal-generator",
                        "category": "chat",
                        "output_modalities": ["video", "audio"],
                    }
                },
            }
        )

        self.assertEqual(provider["chat_models"], ["multimodal-generator"])
        self.assertEqual(provider["audio_models"], [])

    def test_canvas_config_response_disables_stale_browser_cache(self):
        response = Response()

        asyncio.run(main.ai_config(response))

        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_canvas_config_uses_only_saved_codex_chat_models(self):
        saved_providers = [
            {
                "id": "codex",
                "name": "GPT CLI",
                "protocol": "codex",
                "enabled": True,
                "chat_models": ["gpt-5.6-sol"],
                "image_models": ["gpt-image-2"],
            }
        ]
        with patch.object(
            main,
            "public_api_providers",
            return_value=saved_providers,
        ), patch.object(
            main,
            "codex_cli_model_catalog",
        ) as catalog:
            response = Response()
            payload = asyncio.run(main.ai_config(response))

        codex = next(
            provider
            for provider in payload["api_providers"]
            if provider["id"] == "codex"
        )
        self.assertEqual(codex["chat_models"], ["gpt-5.6-sol"])
        catalog.assert_not_called()

    def test_legacy_string_models_and_audio_metadata_normalize_together(self):
        provider = main.normalize_provider(
            {
                "id": "local-fal",
                "name": "Local Fal",
                "base_url": "http://127.0.0.1:8787/v1",
                "image_models": ["gpt-image-2"],
                "video_models": ["seedance-2.0-mini"],
                "audio_models": ["minimax-speech-2.6-hd", "eleven-v3-dialogue"],
                "default_audio_model": "eleven-v3-dialogue",
                "model_metadata": {
                    "minimax-speech-2.6-hd": {
                        "id": "minimax-speech-2.6-hd",
                        "description": "remote default",
                        "description_override": "my note",
                        "category": "audio",
                    }
                },
            }
        )

        self.assertEqual(provider["image_models"], ["gpt-image-2"])
        self.assertEqual(provider["audio_models"], ["minimax-speech-2.6-hd", "eleven-v3-dialogue"])
        self.assertEqual(provider["default_audio_model"], "eleven-v3-dialogue")
        self.assertEqual(
            provider["model_metadata"]["minimax-speech-2.6-hd"]["description_override"],
            "my note",
        )


class CanvasAudioTests(unittest.TestCase):
    def test_minimax_request_contains_only_supported_mvp_fields(self):
        body = main.canvas_audio_request_body(
            main.CanvasAudioRequest(
                model="minimax-speech-2.6-hd",
                prompt="欢迎来到今天的节目。",
                voice="Chinese (Mandarin)_Warm_Bestie",
                speed=1.1,
                language_boost="Chinese",
                format="mp3",
                language_code="zh",
                stability=0.5,
            )
        )

        self.assertEqual(body["voice_setting"]["voice_id"], "Chinese (Mandarin)_Warm_Bestie")
        self.assertEqual(body["voice_setting"]["speed"], 1.1)
        self.assertEqual(body["audio_setting"], {"format": "mp3"})
        self.assertNotIn("language_code", body)
        self.assertNotIn("stability", body)

    def test_eleven_single_role_request(self):
        body = main.canvas_audio_request_body(
            main.CanvasAudioRequest(
                model="eleven-v3-dialogue",
                prompt="[excited] 我们开始吧！",
                voice="Aria",
            )
        )

        self.assertEqual(
            body,
            {
                "model": "eleven-v3-dialogue",
                "prompt": "[excited] 我们开始吧！",
                "voice": "Aria",
            },
        )

    def test_eleven_multi_role_request_preserves_order(self):
        body = main.canvas_audio_request_body(
            main.CanvasAudioRequest(
                model="eleven-v3-dialogue",
                inputs=[
                    main.CanvasAudioDialogueInput(voice="Aria", text="你准备好了吗？"),
                    main.CanvasAudioDialogueInput(voice="Charlotte", text="[laughs] 当然。"),
                ],
                language_code="zh",
                stability=0.5,
                use_speaker_boost=True,
            )
        )

        self.assertEqual([item["voice"] for item in body["inputs"]], ["Aria", "Charlotte"])
        self.assertEqual(body["language_code"], "zh")
        self.assertTrue(body["use_speaker_boost"])
        self.assertNotIn("prompt", body)

    def test_audio_result_normalizes_relative_url_and_media_shape(self):
        result = main.normalize_canvas_audio_result(
            {
                "id": "audio_123",
                "status": "completed",
                "audio": {"url": "/output/result.mp3", "content_type": "audio/mpeg"},
            },
            {"base_url": "http://127.0.0.1:8787/v1"},
        )

        self.assertEqual(result["audio"]["kind"], "audio")
        self.assertEqual(result["audio"]["url"], "http://127.0.0.1:8787/output/result.mp3")
        self.assertEqual(result["audios"][0]["mime"], "audio/mpeg")

    def test_polling_queries_existing_task_without_recreating_it(self):
        class FakeResponse:
            status_code = 200
            text = "json"

            @staticmethod
            def json():
                return {
                    "id": "audio_poll",
                    "status": "in_progress",
                    "model": "minimax-speech-2.6-hd",
                }

        class FakeClient:
            posts = 0
            gets = 0

            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, *args, **kwargs):
                FakeClient.gets += 1
                return FakeResponse()

            async def post(self, *args, **kwargs):
                FakeClient.posts += 1
                return FakeResponse()

        main.CANVAS_AUDIO_TASKS["audio_poll"] = {
            "provider_id": "local-fal",
            "model": "minimax-speech-2.6-hd",
            "status": "queued",
        }
        provider = {
            "id": "local-fal",
            "base_url": "http://127.0.0.1:8787/v1",
        }
        try:
            with patch.object(main, "get_api_provider_exact", return_value=provider), patch.object(
                main.httpx, "AsyncClient", FakeClient
            ), patch.object(main, "api_headers", return_value={}):
                result = asyncio.run(main.canvas_audio_status("audio_poll"))
        finally:
            main.CANVAS_AUDIO_TASKS.pop("audio_poll", None)

        self.assertEqual(result["status"], "in_progress")
        self.assertEqual(FakeClient.gets, 1)
        self.assertEqual(FakeClient.posts, 0)


class LocalFalVideoPayloadTests(unittest.TestCase):
    def setUp(self):
        self.provider = {
            "id": "custom-api-4",
            "name": "Fal.ai本地中转",
            "base_url": "http://127.0.0.1:8787/v1",
        }
        self.image_url = "data:image/png;base64,aW1hZ2U="

    def video_payload(self, model, images=None, multimodal=False):
        return main.CanvasVideoRequest(
            prompt="A woman walks through a courtyard",
            model=model,
            images=(
                [main.AIReference(url=self.image_url, name="courtyard.png")]
                if images is None
                else images
            ),
            multimodal=multimodal,
        )

    def test_grok_image_to_video_includes_singular_image_url(self):
        body = main.openai_compatible_video_body(
            self.video_payload("grok-imagine-video"),
            self.provider,
        )

        self.assertEqual(body["images"], [self.image_url])
        self.assertEqual(body["image_url"], self.image_url)

    def test_seedance_image_to_video_includes_singular_image_url(self):
        body = main.openai_compatible_video_body(
            self.video_payload("seedance-2.0-mini"),
            self.provider,
        )

        self.assertEqual(body["images"], [self.image_url])
        self.assertEqual(body["image_url"], self.image_url)

    def test_seedance_first_and_last_frames_map_to_lfp_fields(self):
        end_image_url = "data:image/png;base64,ZW5kLWltYWdl"
        body = main.openai_compatible_video_body(
            self.video_payload(
                "seedance-2.0-mini",
                images=[
                    main.AIReference(url=self.image_url, role="first_frame"),
                    main.AIReference(url=end_image_url, role="last_frame"),
                ],
            ),
            self.provider,
        )

        self.assertEqual(body["image_url"], self.image_url)
        self.assertEqual(body["end_image_url"], end_image_url)
        self.assertEqual(body["images"], [self.image_url, end_image_url])

    def test_grok_rejects_first_and_last_frames(self):
        with self.assertRaisesRegex(main.HTTPException, "不支持尾帧"):
            main.openai_compatible_video_body(
                self.video_payload(
                    "grok-imagine-video",
                    images=[
                        main.AIReference(url=self.image_url, role="first_frame"),
                        main.AIReference(
                            url="data:image/png;base64,ZW5kLWltYWdl",
                            role="last_frame",
                        ),
                    ],
                ),
                self.provider,
            )

    def test_grok_reference_model_preserves_all_reference_images(self):
        reference_urls = [
            f"data:image/png;base64,cmVmZXJlbmNlLTA{index}"
            for index in range(1, 8)
        ]
        body = main.openai_compatible_video_body(
            self.video_payload(
                "grok-imagine-video-reference",
                images=[main.AIReference(url=url) for url in reference_urls],
                multimodal=True,
            ),
            self.provider,
        )

        self.assertEqual(body["images"], reference_urls)
        self.assertEqual(body["reference_image_urls"], reference_urls)
        self.assertNotIn("image_url", body)

    def test_grok_reference_model_requires_an_image(self):
        with self.assertRaisesRegex(main.HTTPException, "至少需要 1 张"):
            main.openai_compatible_video_body(
                self.video_payload(
                    "grok-imagine-video-reference",
                    images=[],
                    multimodal=True,
                ),
                self.provider,
            )

    def test_grok_reference_model_rejects_more_than_seven_images(self):
        with self.assertRaisesRegex(main.HTTPException, "最多支持 7 张"):
            main.openai_compatible_video_body(
                self.video_payload(
                    "grok-imagine-video-reference",
                    images=[
                        main.AIReference(
                            url=f"data:image/png;base64,cmVmZXJlbmNlLTA{index}",
                        )
                        for index in range(8)
                    ],
                    multimodal=True,
                ),
                self.provider,
            )

    def test_grok_reference_model_rejects_frame_roles(self):
        with self.assertRaisesRegex(main.HTTPException, "不支持首尾帧角色"):
            main.openai_compatible_video_body(
                self.video_payload(
                    "grok-imagine-video-reference",
                    images=[
                        main.AIReference(url=self.image_url, role="first_frame"),
                    ],
                ),
                self.provider,
            )

    def test_seedance_rejects_unlabelled_multiple_reference_images(self):
        with self.assertRaisesRegex(main.HTTPException, "不支持多图参考"):
            main.openai_compatible_video_body(
                self.video_payload(
                    "seedance-2.0-mini",
                    images=[
                        main.AIReference(url=self.image_url),
                        main.AIReference(url="data:image/png;base64,c2Vjb25kLWltYWdl"),
                    ],
                ),
                self.provider,
            )

    def test_local_fal_models_reject_omni_reference_mode(self):
        with self.assertRaisesRegex(main.HTTPException, "全能/多图参考"):
            main.openai_compatible_video_body(
                self.video_payload("seedance-2.0-mini", multimodal=True),
                self.provider,
            )

    def test_other_openai_video_models_keep_existing_images_contract(self):
        body = main.openai_compatible_video_body(
            self.video_payload("other-video-model"),
            self.provider,
        )

        self.assertEqual(body["images"], [self.image_url])
        self.assertNotIn("image_url", body)


if __name__ == "__main__":
    unittest.main()
