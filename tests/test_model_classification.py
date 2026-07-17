import asyncio
import unittest

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
            },
        )
        self.assertEqual(model_ids, sorted(model_ids))


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

    def test_canvas_config_response_disables_stale_browser_cache(self):
        response = Response()

        asyncio.run(main.ai_config(response))

        self.assertEqual(response.headers["cache-control"], "no-store")


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
