"""CPU-only isolated acceptance. No real encoder, VAE, GPU or remote endpoint."""
import asyncio
import base64
import importlib
import json
import math
import pathlib
import struct
import sys
import types
import unittest
import urllib.error
from unittest.mock import AsyncMock, Mock, patch

import torch
from safetensors.torch import load, save

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from comfyui_nodes import NODE_CLASS_MAPPINGS
from comfyui_nodes.client import RemoteClient
from comfyui_nodes.codec import decode_conditioning, encode_images
from comfyui_nodes.errors import ConfigurationError, ProtocolError, RetryableRemoteError, VersionMismatch
from comfyui_nodes.native import attach_references, native_encode, resized_references
from comfyui_nodes.remote_te_node import QwenImage21Remote, TextEncodeQwenImage21Remote, _RETRY_KEYS
from comfyui_nodes.remote_te_edit import TextEncodeQwenImageEditRemote
from comfyui_nodes.remote_h3 import MiniMaxH3ImageToVideoRemote

FP = "a" * 64
NEW_FP = "b" * 64
URL = "http://127.0.0.1:18765"


def response(*, mode="t2i", fp=FP, extras=None, dtype=torch.float32, attention=True, meta_extra=None):
    tensors = {"hidden_states": torch.arange(3 * 4096, dtype=torch.float32).reshape(1, 3, 4096).to(dtype)}
    extras = {"pooled_output": None} if extras is None else extras
    if attention:
        tensors["attention_mask"] = torch.tensor([[1, 1, 0]], dtype=torch.int64)
    meta = {"protocol": "remote-te/1", "model": "qwen3vl_8b", "fingerprint": fp,
            "mode": mode, "seq_len": "3", "extras": json.dumps(extras),
            "encode_ms": "12.25", "queue_wait_ms": "0.0", "server_version": "test",
            "has_mask": str(attention), "has_slots": str("image_slots" in extras)}
    meta.update(meta_extra or {})
    return save(tensors, metadata=meta)


def decoded(**kwargs):
    return decode_conditioning(response(**kwargs), model="qwen3vl_8b", mode=kwargs.get("mode", "t2i"),
                               fingerprint=FP, reference_count=len(kwargs.get("extras", {}).get("image_slots", [])))


class CodecTests(unittest.TestCase):
    def test_h3_hidden_and_token_tags_roundtrip_and_reject_corruption(self):
        hidden = torch.arange(3 * 5120, dtype=torch.float32).reshape(1, 3, 5120)
        tags = torch.tensor([1, 0, 1], dtype=torch.int64)
        meta = {"protocol": "remote-te/1", "model": "minimax_h3", "fingerprint": FP,
                "mode": "fl2va", "seq_len": "3", "extras": '{"pooled_output":null}',
                "encode_ms": "1", "queue_wait_ms": "0", "server_version": "test",
                "has_mask": "False", "has_slots": "False"}
        blob = save({"hidden_states": hidden, "minimax_token_tags": tags}, metadata=meta)
        conditioning, _ = decode_conditioning(blob, model="minimax_h3", mode="fl2va",
                                               fingerprint=FP, reference_count=1)
        self.assertTrue(torch.equal(conditioning[0][0], hidden))
        self.assertTrue(torch.equal(conditioning[0][1]["minimax_token_tags"], tags))
        for bad_tensors in ({"hidden_states": hidden},
                            {"hidden_states": hidden, "minimax_token_tags": torch.tensor([1, 2, 1])},
                            {"hidden_states": hidden, "minimax_token_tags": tags,
                             "attention_mask": torch.ones(1, 3)}):
            with self.assertRaises(ProtocolError):
                decode_conditioning(save(bad_tensors, metadata=meta), model="minimax_h3",
                                    mode="fl2va", fingerprint=FP, reference_count=1)

    def test_image_float32_is_lossless_and_first_batch_only(self):
        image = torch.tensor([-0.1, 0.33333334, 1.000001, 0.7], dtype=torch.float32).repeat(2, 32, 64, 1)
        original = image.clone()
        tensor = load(base64.b64decode(encode_images([image])))["image_0"]
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (1, 32, 64, 4))
        self.assertTrue(torch.equal(tensor, image[:1]))
        self.assertTrue(torch.equal(image, original))

    def test_hidden_mask_slots_restore_exactly(self):
        conditioning, meta = decoded(mode="edit", extras={"pooled_output": None, "image_slots": [1, 2]})
        self.assertEqual(conditioning[0][0].dtype, torch.float32)
        self.assertEqual(conditioning[0][1]["image_slots"], [1, 2])
        self.assertEqual(conditioning[0][1]["attention_mask"].dtype, torch.int64)
        self.assertEqual(meta["fingerprint"], FP)

    def test_rejects_unknown_extras_and_tensors(self):
        with self.assertRaises(ProtocolError):
            decoded(extras={"pooled_output": None, "hooks": {}})
        blob = response()
        tensors = load(blob)
        tensors["unknown_hook_tensor"] = torch.ones(1)
        n = struct.unpack("<Q", blob[:8])[0]
        meta = json.loads(blob[8:8 + n])["__metadata__"]
        with self.assertRaises(ProtocolError):
            decode_conditioning(save(tensors, metadata=meta), model="qwen3vl_8b", mode="t2i", fingerprint=FP)

    def test_mismatched_fingerprint_and_dtype_are_loud(self):
        with self.assertRaises(VersionMismatch):
            decoded(fp=NEW_FP)
        with self.assertRaises(ProtocolError):
            decoded(dtype=torch.float16)
        with self.assertRaises(ProtocolError):
            decoded(meta_extra={"has_mask": "False"})

    def test_slot_semantics_and_header_limits(self):
        for slots in ([True], [4], [2, 1]):
            with self.assertRaises(ProtocolError):
                decoded(mode="edit", extras={"pooled_output": None, "image_slots": slots})
        with self.assertRaises(ProtocolError):
            decode_conditioning(struct.pack("<Q", 999999) + b"{}", model="qwen3vl_8b", mode="t2i", fingerprint=FP)
        with self.assertRaises(ProtocolError):
            decoded(extras={"pooled_output": [0]})

    def test_input_limits(self):
        for images in ([torch.ones(1, 32, 32, 3, dtype=torch.float16)],
                       [torch.full((1, 2, 2, 3), float("nan"))],
                       [torch.ones(1, 2, 2, 3)] * 17):
            with self.assertRaises(ConfigurationError):
                encode_images(images)


class H3NodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_frame_remote_encode_and_local_vae_match_native_order(self):
        image = torch.rand(1, 32, 64, 3)
        resized = torch.rand(1, 480, 864, 3)
        calls = []
        parent = types.ModuleType("comfy_extras")
        parent.__path__ = []
        native = types.ModuleType("comfy_extras.nodes_minimax_h3")
        native._empty_av_latent = lambda width, height, length: ({"samples": "av"}, 124)
        native._resize = lambda pixels, width, height, crop: calls.append(("resize", crop)) or resized
        helpers = types.ModuleType("node_helpers")
        helpers.conditioning_set_values = lambda cond, extra: [[cond[0][0], {**cond[0][1], **extra}]]

        class Client:
            url = URL
            def __init__(self, url, timeout):
                calls.append(("client", url, timeout))
            async def health(self, model):
                return {"fingerprint": FP, "loaded": True}
            async def encode(self, prompt, **kwargs):
                calls.append(("encode", prompt, kwargs))
                return [[torch.ones(1, 3, 5120), {"minimax_token_tags": torch.tensor([1, 0, 1])}]], {"encode_ms": "12"}
            async def aclose(self):
                pass

        vae = Mock()
        vae.encode.return_value = torch.ones(1, 24, 2, 30, 54)
        with patch.dict(sys.modules, {"comfy_extras": parent, "comfy_extras.nodes_minimax_h3": native,
                                      "node_helpers": helpers}), \
             patch("comfyui_nodes.remote_h3.RemoteClient", Client):
            result = await MiniMaxH3ImageToVideoRemote().encode(
                vae, "motion", 864, 480, 124, server_url=URL, first_frame=image)
        conditioning, latent = result["result"]
        self.assertEqual(latent, {"samples": "av"})
        self.assertEqual(calls[0], ("resize", "disabled"))
        self.assertEqual(calls[2][2]["mode"], "fl2va")
        self.assertTrue(torch.equal(calls[2][2]["images"][0], resized))
        vae.encode.assert_called_once_with(resized)
        self.assertEqual(conditioning[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 0)



class HTTPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = RemoteClient(URL)

    def test_http_status_policy_never_reads_error_body(self):
        for status, expected in [(409, VersionMismatch), (400, ProtocolError), (401, ProtocolError),
                                 (422, ProtocolError), (302, ProtocolError), (429, RetryableRemoteError),
                                 (500, RetryableRemoteError), (503, RetryableRemoteError)]:
            body = Mock()
            error = urllib.error.HTTPError(URL, status, "error", {}, body)
            with patch.object(self.client.opener, "open", side_effect=error):
                with self.assertRaises(expected):
                    self.client.health_sync()
            body.read.assert_not_called()

    def test_network_policy_and_url_credentials(self):
        with patch.object(self.client.opener, "open", side_effect=urllib.error.URLError("do not echo this")):
            with self.assertRaisesRegex(RetryableRemoteError, "network") as exc:
                self.client.health_sync()
            self.assertNotIn("echo", str(exc.exception))
        for url in ("file:///tmp/a", "http://user:password@example.com", URL + "?token=secret"):
            with self.assertRaises(ConfigurationError):
                RemoteClient(url)

    def test_health_requires_bool_and_fingerprint(self):
        for loaded in (None, 1, "True"):
            with patch.object(self.client, "_request", return_value=json.dumps({"profiles": {"qwen3vl_8b": {"fingerprint": FP, "loaded": loaded}}})):
                with self.assertRaises(ProtocolError):
                    self.client.health_sync()

    async def test_async_network_runs_in_another_thread(self):
        import threading
        current_thread = threading.get_ident()
        with patch.object(self.client, "health_sync", side_effect=lambda model: threading.get_ident()):
            self.assertNotEqual(await self.client.health(), current_thread)

    def test_encode_sends_raw_safetensors_not_png(self):
        raw = torch.rand(1, 16, 32, 4)
        with patch.object(self.client, "_request", return_value=response(mode="edit", extras={"pooled_output": None, "image_slots": [1]})) as request:
            self.client.encode_sync("test", mode="edit", fingerprint=FP, images=[raw])
        payload = request.call_args.args[2]
        self.assertNotIn("ref_image_b64", payload)
        tensors = load(base64.b64decode(payload["ref_images_safetensors_b64"]))
        self.assertTrue(torch.equal(tensors["image_0"], raw))


class NodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _RETRY_KEYS.clear()
        self.remote_result = decoded()
        self.latent = {"samples": torch.zeros(1, 64, 2, 2)}
        self.attach = patch("comfyui_nodes.remote_te_node.attach_references", side_effect=lambda c, images, vae, resolution: (c, self.latent))
        self.attach_mock = self.attach.start()

    def tearDown(self):
        self.attach.stop()

    async def test_remote_success_does_not_load_local_clip(self):
        node = TextEncodeQwenImage21Remote()
        with patch("comfyui_nodes.remote_te_node.RemoteClient.health", AsyncMock(return_value={"fingerprint": FP, "loaded": True})), \
             patch("comfyui_nodes.remote_te_node.RemoteClient.encode", AsyncMock(return_value=self.remote_result)), \
             patch("comfyui_nodes.remote_te_node.load_fallback_clip") as loader:
            result = await node.encode("test", server_url=URL, unique_id="n")
        loader.assert_not_called()
        self.assertIn("REMOTE | fingerprint=" + FP, result["ui"]["text"][0])
        self.assertEqual(len(result["result"]), 1)

    async def test_pin_stays_old_and_explicit_expected_approves_new(self):
        node = TextEncodeQwenImage21Remote()
        with patch("comfyui_nodes.remote_te_node.RemoteClient.health", AsyncMock(return_value={"fingerprint": FP, "loaded": True})) as health, \
             patch("comfyui_nodes.remote_te_node.RemoteClient.encode", AsyncMock(return_value=self.remote_result)) as encode:
            await node.encode("test", server_url=URL)
            health.return_value = {"fingerprint": NEW_FP, "loaded": True}
            await node.encode("test2", server_url=URL)
            self.assertEqual(encode.call_args.kwargs["fingerprint"], FP)
            await node.encode("test3", server_url=URL, expected_fingerprint=NEW_FP)
            self.assertEqual(encode.call_args.kwargs["fingerprint"], NEW_FP)

    async def test_failed_fallback_retries_even_when_health_healthy(self):
        node = TextEncodeQwenImage21Remote()
        with patch("comfyui_nodes.remote_te_node.RemoteClient.health", AsyncMock(return_value={"fingerprint": FP, "loaded": True})), \
             patch("comfyui_nodes.remote_te_node.RemoteClient.encode", AsyncMock(side_effect=RetryableRemoteError("network"))) as encode, \
             patch.object(node, "_local", return_value=("local+", "local-", self.latent)):
            result = await node.encode("test", server_url=URL, unique_id="retry")
            self.assertIn("REMOTE FALLBACK", result["ui"]["text"][0])
            self.assertTrue(math.isnan(await node.IS_CHANGED(server_url=URL, unique_id="retry")))
            encode.side_effect = None
            encode.return_value = self.remote_result
            await node.encode("test", server_url=URL, unique_id="retry")
            self.assertEqual(await node.IS_CHANGED(server_url=URL, unique_id="retry"), (FP, True))

    async def test_409_and_bad_protocol_never_fallback(self):
        for failure in (VersionMismatch("HTTP 409"), ProtocolError("bad metadata")):
            node = TextEncodeQwenImage21Remote()
            with patch("comfyui_nodes.remote_te_node.RemoteClient.health", AsyncMock(return_value={"fingerprint": FP, "loaded": True})), \
                 patch("comfyui_nodes.remote_te_node.RemoteClient.encode", AsyncMock(side_effect=failure)), \
                 patch.object(node, "_local") as local:
                with self.assertRaises(type(failure)):
                    await node.encode("test", server_url=URL)
                local.assert_not_called()

    async def test_health_failure_can_fallback_without_loading_early(self):
        node = TextEncodeQwenImage21Remote()
        with patch("comfyui_nodes.remote_te_node.RemoteClient.health", AsyncMock(side_effect=RetryableRemoteError("network"))), \
             patch.object(node, "_local", return_value=("local+", "local-", self.latent)) as local:
            result = await node.encode("test", server_url=URL)
            self.assertEqual(result["result"], ("local+",))
            local.assert_called_once()

    async def test_pair_failure_uses_complete_native_pair(self):
        node = QwenImage21Remote()
        image = torch.rand(1, 32, 32, 4)
        vae = object()
        with patch("comfyui_nodes.remote_te_node.RemoteClient.health", AsyncMock(return_value={"fingerprint": FP, "loaded": True})), \
             patch("comfyui_nodes.remote_te_node.RemoteClient.encode", AsyncMock(side_effect=[self.remote_result, RetryableRemoteError("HTTP 503")])) as encode, \
             patch.object(node, "_local", return_value=("native+", "native-", self.latent)) as local:
            result = await node.encode("positive", "negative", server_url=URL, vae=vae, image_1=image)
            self.assertEqual(result["result"], ("native+", "native-", self.latent))
            self.assertEqual(encode.call_args.kwargs["mode"], "edit")
            self.assertEqual(local.call_args.args[2:4], ("positive", "negative"))
            self.attach_mock.assert_not_called()

    async def test_image_without_vae_keeps_vision(self):
        node = QwenImage21Remote()
        with patch("comfyui_nodes.remote_te_node.RemoteClient.health", AsyncMock(return_value={"fingerprint": FP, "loaded": True})), \
             patch("comfyui_nodes.remote_te_node.RemoteClient.encode", AsyncMock(return_value=self.remote_result)) as encode:
            await node.encode("positive", "negative", server_url=URL, image_1=torch.rand(1, 32, 32, 3))
            self.assertEqual(encode.call_args.kwargs["mode"], "t2i")

    def test_schema_four_nodes_and_clip_priority(self):
        fake_folder = types.ModuleType("folder_paths")
        fake_folder.get_filename_list = lambda _: ["other.safetensors", "qwen3vl_8b_w4a8.safetensors", "qwen3vl_8b_int8_convrot.safetensors"]
        with patch.dict(sys.modules, {"folder_paths": fake_folder}):
            self.assertEqual(len(NODE_CLASS_MAPPINGS), 4)
            self.assertIn("MiniMaxH3ImageToVideoRemote", NODE_CLASS_MAPPINGS)
            self.assertIn("first_frame", MiniMaxH3ImageToVideoRemote.INPUT_TYPES()["optional"])
            schema = QwenImage21Remote.INPUT_TYPES()
            self.assertEqual(schema["required"]["clip_name"][0][0], "qwen3vl_8b_int8_convrot.safetensors")
            self.assertIn("image_16", schema["optional"])
            self.assertIn("unique_id", schema["hidden"])
            self.assertEqual(schema["required"]["fallback_local"][1]["default"], True)
            self.assertEqual(schema["required"]["timeout_s"][1]["default"], 30)

    async def test_public_options_and_legacy_aliases(self):
        node = TextEncodeQwenImage21Remote()
        with patch.object(node, "_run", AsyncMock(return_value=(["cond"], self.latent, "ok"))) as run:
            await node.encode("test", fallback_local=False, timeout_s=45)
            self.assertFalse(run.call_args.kwargs["fallback"])
            self.assertEqual(run.call_args.kwargs["timeout"], 45)
            await node.encode("test", fallback=False, timeout=15)
            self.assertFalse(run.call_args.kwargs["fallback"])
            self.assertEqual(run.call_args.kwargs["timeout"], 15)


class NativeSemanticsTests(unittest.TestCase):
    def setUp(self):
        comfy = types.ModuleType("comfy")
        utils = types.ModuleType("comfy.utils")
        mm = types.ModuleType("comfy.model_management")
        utils.common_upscale = Mock(side_effect=lambda samples, width, height, method, crop:
            torch.nn.functional.interpolate(samples, size=(height, width), mode="nearest"))
        mm.intermediate_device = lambda: "cpu"
        comfy.utils, comfy.model_management = utils, mm
        helpers = types.ModuleType("node_helpers")
        def set_values(conditioning, values, append=False):
            return [[h, dict(extras, **values)] for h, extras in conditioning]
        helpers.conditioning_set_values = Mock(side_effect=set_values)
        self.modules = {"comfy": comfy, "comfy.utils": utils, "comfy.model_management": mm, "node_helpers": helpers}
        self.scope = patch.dict(sys.modules, self.modules)
        self.scope.start()
        self.utils, self.helpers = utils, helpers

    def tearDown(self):
        self.scope.stop()

    def test_resize_matches_native_dimensions_and_vae_keeps_rgba(self):
        first, second = torch.rand(2, 64, 128, 4), torch.rand(1, 96, 32, 3)
        vae = Mock()
        vae.encode.side_effect = lambda image: image.mean(dim=-1)
        pos, _ = decoded()
        neg, _ = decoded()
        conditions, latent = attach_references([pos, neg], [first, second], vae, 256)
        expected_width = max(32, round(math.sqrt(256 * 256 * 2) / 32) * 32)
        expected_height = max(32, round(math.sqrt(256 * 256 / 2) / 32) * 32)
        self.assertEqual(tuple(vae.encode.call_args_list[0].args[0].shape), (1, expected_height, expected_width, 4))
        self.assertEqual(tuple(latent["samples"].shape), (1, 64, expected_height // 16, expected_width // 16))
        self.assertIs(conditions[0][0][1]["reference_latents"], conditions[1][0][1]["reference_latents"])
        self.assertEqual(self.utils.common_upscale.call_args.args[-2:], ("lanczos", "disabled"))

    def test_native_execute_is_direct_fallback(self):
        qwen = types.ModuleType("comfy_extras.nodes_qwen")
        native_node = Mock()
        native_node.execute.return_value = types.SimpleNamespace(result=("native+", "native-", {"samples": "native"}))
        qwen.TextEncodeQwenImage21 = native_node
        image = torch.rand(1, 32, 32, 3)
        with patch.dict(sys.modules, {"comfy_extras.nodes_qwen": qwen}):
            result = native_encode("clip", "p", "n", "vae", 1024, [image])
        self.assertEqual(result, ("native+", "native-", {"samples": "native"}))
        self.assertEqual(native_node.execute.call_args.kwargs["prompt"], "p")
        self.assertIn("image_1", native_node.execute.call_args.kwargs["images"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
