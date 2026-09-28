"""In-memory ASGI acceptance: no listener, real encoder, remote host, or GPU."""
import asyncio
import concurrent.futures
import json
import threading
import time
import unittest
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / ".deps"), str(root)]

import torch

from te_server.server import create_app


class BlockingEncoder:
    fingerprint = "1234567890abcdef"
    loading = False

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts = []
        self.concurrent = 0
        self.peak = 0
        self.load_calls = 0
        self.unload_calls = 0
        self.refresh_started = None
        self.refresh_release = None

    def changed(self): return False
    def refresh(self):
        if self.refresh_started is not None:
            self.refresh_started.set()
            self.refresh_release.wait(3)
    def load(self): self.load_calls += 1
    def unload(self): self.unload_calls += 1
    def stats(self): return {"fingerprint": self.fingerprint, "loaded": True}

    def encode(self, prompt, *args):
        self.prompts.append(prompt)
        self.concurrent += 1
        self.peak = max(self.peak, self.concurrent)
        try:
            if prompt == "active":
                self.started.set()
                if not self.release.wait(3):
                    raise RuntimeError("Test release deadline exceeded")
            return [[torch.ones(1, 1, 4096), {"pooled_output": None}]]
        finally:
            self.concurrent -= 1


class ASGIRequest:
    def __init__(self, app, prompt, path="/encode"):
        self.disconnected = asyncio.Event()
        self.sent = []
        self.body = json.dumps({"prompt": prompt} if path == "/encode" else {}).encode()
        self.delivered = False
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json"),
                             (b"content-length", str(len(self.body)).encode())],
                 "client": ("127.0.0.1", 1234), "server": ("isolated", 80)}
        self.task = asyncio.create_task(app(scope, self.receive, self.send))

    async def receive(self):
        if not self.delivered:
            self.delivered = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        self.sent.append(message)

    @property
    def response(self):
        return next((m for m in self.sent if m["type"] == "http.response.start"), None)


class QueueCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.encoder = BlockingEncoder()
        self.app = create_app({"queue_depth": 1, "queue_wait_timeout_s": 2},
                              {"qwen3vl_8b": self.encoder})
        self.rt = self.app.state.runtimes["qwen3vl_8b"]
        self.requests = []
        self.extra_releases = []

    async def asyncTearDown(self):
        self.encoder.release.set()
        if self.encoder.refresh_release is not None:
            self.encoder.refresh_release.set()
        for release in self.extra_releases:
            release.set()
        await asyncio.wait_for(asyncio.gather(*(r.task for r in self.requests),
                                             return_exceptions=True), 3)
        await self.wait_until(lambda: not self.rt.active and self.rt.pending == 0)

    async def wait_until(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("State did not reach its expected value")
            await asyncio.sleep(.005)

    def request(self, prompt, path="/encode"):
        request = ASGIRequest(self.app, prompt, path)
        self.requests.append(request)
        return request

    async def active_request(self):
        request = self.request("active")
        await self.wait_until(self.encoder.started.is_set)
        return request

    def assert_all_slots_returned(self):
        self.assertEqual(self.rt.pending, 0)
        self.assertFalse(self.rt.active)
        self.assertFalse(self.rt.serial.locked())
        self.assertTrue(self.rt.slots.acquire(blocking=False))
        self.assertTrue(self.rt.slots.acquire(blocking=False))
        self.assertFalse(self.rt.slots.acquire(blocking=False))
        self.rt.slots.release()
        self.rt.slots.release()

    async def test_disconnect_drops_queued_work_and_reuses_slot(self):
        active = await self.active_request()
        abandoned = self.request("abandoned")
        await self.wait_until(lambda: self.rt.pending == 1)
        abandoned.disconnected.set()
        await self.wait_until(lambda: self.rt.pending == 0)
        self.assertEqual(self.encoder.prompts, ["active"])
        replacement = self.request("replacement")
        await self.wait_until(lambda: self.rt.pending == 1)
        self.encoder.release.set()
        await asyncio.gather(active.task, abandoned.task, replacement.task)
        self.assertEqual(self.encoder.prompts, ["active", "replacement"])
        self.assertEqual(replacement.response["status"], 200)
        self.assert_all_slots_returned()

    async def test_already_disconnected_request_never_starts_encoder(self):
        abandoned = self.request("abandoned")
        abandoned.disconnected.set()
        await abandoned.task
        self.assertEqual(self.encoder.prompts, [])
        self.assert_all_slots_returned()

    async def test_asgi_cancellation_releases_pending_work(self):
        active = await self.active_request()
        abandoned = self.request("abandoned")
        await self.wait_until(lambda: self.rt.pending == 1)
        abandoned.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await abandoned.task
        await self.wait_until(lambda: self.rt.pending == 0)
        self.assertTrue(self.rt.active)
        self.encoder.release.set()
        await active.task
        self.assertEqual(self.encoder.prompts, ["active"])
        self.assert_all_slots_returned()

    async def test_disconnected_load_and_unload_do_not_execute(self):
        active = await self.active_request()
        for path in ("/load", "/unload"):
            with self.subTest(path=path):
                abandoned = self.request("unused", path=path)
                await self.wait_until(lambda: self.rt.pending == 1)
                abandoned.disconnected.set()
                await abandoned.task
                self.assertEqual(self.rt.pending, 0)
        self.encoder.release.set()
        await active.task
        self.assertEqual((self.encoder.load_calls, self.encoder.unload_calls), (0, 0))
        self.assert_all_slots_returned()

    async def test_running_work_finishes_safely_after_asgi_cancellation(self):
        active = await self.active_request()
        active.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active.task
        self.assertTrue(self.rt.active)
        self.assertTrue(self.rt.serial.locked())
        replacement = self.request("replacement")
        await self.wait_until(lambda: self.rt.pending == 1)
        self.assertEqual(self.encoder.prompts, ["active"])
        self.encoder.release.set()
        await replacement.task
        self.assertEqual(self.encoder.prompts, ["active", "replacement"])
        self.assertEqual(self.encoder.peak, 1)
        self.assert_all_slots_returned()

    async def test_running_work_finishes_safely_after_disconnect(self):
        active = await self.active_request()
        active.disconnected.set()
        await asyncio.sleep(.03)
        self.assertTrue(self.rt.active)
        self.assertTrue(self.rt.serial.locked())
        replacement = self.request("replacement")
        await self.wait_until(lambda: self.rt.pending == 1)
        self.encoder.release.set()
        await asyncio.gather(active.task, replacement.task)
        self.assertEqual(self.encoder.peak, 1)
        self.assertEqual(self.encoder.prompts, ["active", "replacement"])
        self.assert_all_slots_returned()

    async def test_wait_deadline_returns_503_and_releases_slot(self):
        self.rt.queue_wait_timeout_s = .08
        active = await self.active_request()
        expired = self.request("expired")
        await expired.task
        self.assertEqual(expired.response["status"], 503)
        self.assertIn((b"retry-after", b"1"), expired.response["headers"])
        self.assertEqual(self.rt.pending, 0)
        self.encoder.release.set()
        await active.task
        self.assertEqual(self.encoder.prompts, ["active"])
        self.assert_all_slots_returned()

    async def test_cancel_before_executor_start_releases_serial_and_slot(self):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(pool)
        release = threading.Event()
        self.extra_releases.append(release)
        blocker = asyncio.get_running_loop().run_in_executor(None, release.wait)
        abandoned = self.request("abandoned")
        await self.wait_until(lambda: self.rt.active)
        self.assertEqual(self.encoder.prompts, [])
        abandoned.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await abandoned.task
        self.assert_all_slots_returned()
        replacement = self.request("replacement")
        await self.wait_until(lambda: self.rt.active)
        release.set()
        await asyncio.gather(blocker, replacement.task)
        self.assertEqual(self.encoder.prompts, ["replacement"])
        self.assert_all_slots_returned()

    async def test_disconnect_during_cpu_preflight_skips_encoder(self):
        self.encoder.refresh_started = threading.Event()
        self.encoder.refresh_release = threading.Event()
        abandoned = self.request("abandoned")
        await self.wait_until(self.encoder.refresh_started.is_set)
        abandoned.disconnected.set()
        await asyncio.sleep(.03)
        self.encoder.refresh_release.set()
        await abandoned.task
        self.assertEqual(self.encoder.prompts, [])
        self.assert_all_slots_returned()


if __name__ == "__main__":
    unittest.main(verbosity=2)
