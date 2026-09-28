from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / ".deps"), str(root)]
import base64
import concurrent.futures
import json
import struct
import tempfile
import threading
import time
import unittest
import torch
import safetensors.torch as st
from fastapi.testclient import TestClient
from te_server.server import create_app
from te_server.fingerprint import fingerprint


class FakeEncoder:
    fingerprint = "1234567890abcdef"
    loading = False
    def __init__(self):
        self.loaded = False
        self.calls = 0
        self.concurrent = 0
        self.peak = 0
        self.started = threading.Event()
        self.release = None
    def changed(self): return False
    def refresh(self): pass
    def load(self): self.loaded = True
    def unload(self): self.loaded = False
    def stats(self): return {"fingerprint": self.fingerprint, "loaded": self.loaded, "vram_mb": 0}
    def encode(self, prompt, resolution, mode, images):
        self.load()
        self.calls += 1
        self.concurrent += 1
        self.peak = max(self.peak, self.concurrent)
        self.started.set()
        if self.release:
            self.release.wait(3)
        time.sleep(0.01)
        self.concurrent -= 1
        extras = {"pooled_output": None}
        if mode == "edit": extras["image_slots"] = [1]
        return [[torch.full((1, 3, 4096), float(len(prompt))), extras]]


class ServerTests(unittest.TestCase):
    def test_h3_profile_preserves_token_tags_and_rejects_wrong_mode(self):
        encoder = FakeEncoder()
        encoder.encode = lambda *args: [[torch.ones(1, 3, 5120),
                                         {"pooled_output": None,
                                          "minimax_token_tags": torch.tensor([1, 0, 1])}]]
        client = TestClient(create_app({}, {"minimax_h3": encoder}))
        payload = base64.b64encode(st.save({"image_0": torch.rand(1, 32, 32, 3)})).decode()
        r = client.post("/encode", json={"prompt": "test", "model": "minimax_h3",
                                          "mode": "fl2va", "resolution": 0,
                                          "ref_images_safetensors_b64": payload})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(tuple(st.load(r.content)["hidden_states"].shape), (1, 3, 5120))
        self.assertTrue(torch.equal(st.load(r.content)["minimax_token_tags"], torch.tensor([1, 0, 1])))
        self.assertEqual(client.get("/health").json()["reserved_profiles"], [])
        self.assertEqual(client.post("/encode", json={"prompt": "x", "model": "minimax_h3",
                                                       "mode": "t2i"}).status_code, 400)

    def setUp(self):
        self.encoder = FakeEncoder()
        self.client = TestClient(create_app({"queue_depth": 1}, {"qwen3vl_8b": self.encoder}))

    def test_roundtrip_float32_extras(self):
        r = self.client.post("/encode", json={"prompt": "abc", "fingerprint": self.encoder.fingerprint})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(torch.equal(st.load(r.content)["hidden_states"], torch.full((1, 3, 4096), 3.0)))
        n = struct.unpack("<Q", r.content[:8])[0]
        meta = json.loads(r.content[8:8+n])["__metadata__"]
        self.assertEqual(json.loads(meta["extras"]), {"pooled_output": None})
        self.assertEqual(meta["protocol"], "remote-te/1")
        self.assertIn("queue_wait_ms", meta)

    def test_wrong_fingerprint_never_encodes(self):
        r = self.client.post("/encode", json={"prompt": "a", "fingerprint": "ffffffffffffffff"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.encoder.calls, 0)
        self.assertEqual(r.json()["detail"]["expected"], self.encoder.fingerprint)

    def test_validation_and_reserved_profile(self):
        for body in ({"prompt": "x", "resolution": -1}, {"prompt": "x", "mode": "bad"}, {"prompt": "x", "unexpected": True}):
            self.assertEqual(self.client.post("/encode", json=body).status_code, 400)
        self.assertEqual(self.client.post("/encode", json={"prompt": "a", "model": "minimax_h3"}).status_code, 404)
        self.assertEqual(self.client.post("/encode", json={"prompt": "a", "mode": "edit"}).status_code, 400)
        self.assertEqual(self.client.post("/encode", json={"prompt": "a", "mode": "edit", "ref_images_safetensors_b64": "YmFk"}).status_code, 400)

    def test_unknown_native_contract_is_not_retryable(self):
        self.encoder.encode = lambda *args: [[torch.ones(1, 3, 4096), {"hooks": "unsupported"}]]
        self.assertEqual(self.client.post("/encode", json={"prompt": "x"}).status_code, 400)

    def test_edit_float_image(self):
        pixels = torch.rand(1, 32, 32, 4)
        payload = base64.b64encode(st.save({"image_0": pixels})).decode()
        r = self.client.post("/encode", json={"prompt": "edit", "mode": "edit", "ref_images_safetensors_b64": payload})
        self.assertEqual(r.status_code, 200)
        bad = base64.b64encode(st.save({"image_0": pixels * float("nan")})).decode()
        self.assertEqual(self.client.post("/encode", json={"prompt": "edit", "mode": "edit", "ref_images_safetensors_b64": bad}).status_code, 400)

    def test_serial_queue_and_429(self):
        self.encoder.release = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(self.client.post, "/encode", json={"prompt": "a"})
            self.assertTrue(self.encoder.started.wait(2))
            second = pool.submit(self.client.post, "/encode", json={"prompt": "bb"})
            rt = self.client.app.state.runtimes["qwen3vl_8b"]
            deadline = time.monotonic() + 2
            while rt.pending != 1 and time.monotonic() < deadline: time.sleep(.01)
            third = self.client.post("/encode", json={"prompt": "ccc"})
            self.assertEqual(third.status_code, 429)
            self.assertEqual(third.headers["retry-after"], "1")
            self.encoder.release.set()
            self.assertEqual(first.result().status_code, 200)
            self.assertEqual(second.result().status_code, 200)
        self.assertEqual(self.encoder.peak, 1)
        self.assertEqual(rt.pending, 0)

    def test_resize_aggregate_rejected_before_encoder(self):
        pixels = torch.zeros(1, 32, 32, 3)
        payload = base64.b64encode(st.save({f"image_{i}": pixels.clone() for i in range(16)})).decode()
        r = self.client.post("/encode", json={"prompt": "edit", "mode": "edit", "resolution": 4096,
                                              "ref_images_safetensors_b64": payload})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.encoder.calls, 0)
        self.assertIn("resized", r.json()["detail"])

    def test_auth_and_unload(self):
        client = TestClient(create_app({"auth_token": "test-only"}, {"qwen3vl_8b": self.encoder}))
        self.assertEqual(client.get("/health").status_code, 401)
        self.assertEqual(client.get("/health", headers={"Authorization": "Bearer test-only"}).status_code, 200)
        self.assertEqual(self.client.post("/load", json={}).status_code, 200)
        self.assertTrue(self.encoder.loaded)
        self.assertEqual(self.client.post("/unload", json={}).status_code, 200)
        self.assertFalse(self.encoder.loaded)

    def test_exclusive_profiles_switch_on_demand(self):
        image = FakeEncoder()
        h3 = FakeEncoder()
        def encode_h3(*args):
            h3.load()
            return [[torch.ones(1, 3, 5120),
                     {"pooled_output": None, "minimax_token_tags": torch.tensor([1, 0, 1])}]]
        h3.encode = encode_h3
        client = TestClient(create_app({"exclusive_profiles": True, "idle_unload_min": 10},
                                       {"qwen3vl_8b": image, "minimax_h3": h3}))
        self.assertEqual(client.post("/load", json={"model": "qwen3vl_8b"}).status_code, 200)
        self.assertTrue(image.loaded)
        self.assertFalse(h3.loaded)
        response = client.post("/encode", json={"prompt": "video", "model": "minimax_h3", "mode": "t2va"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(image.loaded)
        self.assertTrue(h3.loaded)
        self.assertEqual(client.post("/encode", json={"prompt": "image"}).status_code, 200)
        self.assertTrue(image.loaded)
        self.assertFalse(h3.loaded)
        self.assertEqual(client.app.state.runtimes["qwen3vl_8b"].idle_unload_s, 600)

    def test_exclusive_profiles_wait_for_active_forward(self):
        image = FakeEncoder()
        image.release = threading.Event()
        h3 = FakeEncoder()
        client = TestClient(create_app({"exclusive_profiles": True},
                                       {"qwen3vl_8b": image, "minimax_h3": h3}))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/encode", json={"prompt": "image"})
            self.assertTrue(image.started.wait(2))
            second = pool.submit(client.post, "/load", json={"model": "minimax_h3"})
            time.sleep(.05)
            self.assertTrue(image.loaded)
            self.assertFalse(h3.loaded)
            image.release.set()
            self.assertEqual(first.result().status_code, 200)
            self.assertEqual(second.result().status_code, 200)
        self.assertFalse(image.loaded)
        self.assertTrue(h3.loaded)

    def test_fingerprint_cache_invalidates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            path.write_bytes(b"first")
            first = fingerprint(path)
            self.assertEqual(first, fingerprint(path))
            path.write_bytes(b"different weight")
            self.assertNotEqual(first, fingerprint(path))


if __name__ == "__main__": unittest.main(verbosity=2)
