"""In-memory HTTP and blocking CPU fakes. No listener, remote endpoint or CUDA work."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import threading
import time
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import torch

from te_server.encoder import TEService
from te_server.server import Runtime, _QueueJob, create_app
from test_queue_cancel import ASGIRequest


class Encoder:
    fingerprint = "1234567890abcdef"
    loading = False

    def __init__(self, loaded=True):
        self.loaded = loaded
        self.weight_changed = False
        self.maintenance_calls = 0
        self.encode_calls = 0
        self.load_calls = 0
        self.refresh_calls = 0
        self.unload_calls = 0
        self.running = 0
        self.peak = 0
        self.events = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False

    def changed(self): return self.weight_changed
    def stats(self): return {"loaded": self.loaded, "fingerprint": self.fingerprint}
    def refresh(self): self.refresh_calls += 1
    def load(self): self.load_calls += 1; self.loaded = True
    def unload(self):
        if self.running:
            raise AssertionError("Unload overlapped encoder work")
        self.unload_calls += 1
        self.loaded = False
        self.events.append("unload")

    def maintain_loaded(self):
        if not self.loaded or self.loading:
            return False
        self.maintenance_calls += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        self.events.append("maintenance-start")
        self.started.set()
        try:
            if self.block and not self.release.wait(4):
                raise RuntimeError("Fixture release timeout")
            if self.fail:
                raise RuntimeError("Injected native maintenance failure")
            return True
        finally:
            self.running -= 1
            self.events.append("maintenance-end")

    def encode(self, prompt, *args):
        self.encode_calls += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        self.events.append("business")
        try:
            return [[torch.ones(1, 1, 4096), {"pooled_output": None}]]
        finally:
            self.running -= 1


def slots_available(runtime, expected=2):
    acquired = 0
    while runtime.slots.acquire(blocking=False):
        acquired += 1
        if acquired > expected:
            raise AssertionError("Extra semaphore slot created")
    for _ in range(acquired):
        runtime.slots.release()
    return acquired


class MaintenanceUnitTests(unittest.TestCase):
    def test_disabled_and_interval_bounds(self):
        encoder = Encoder()
        runtime = Runtime(encoder)
        self.assertFalse(runtime.maintain_once())
        self.assertEqual(encoder.maintenance_calls, 0)
        for value in (True, False, -1, .5, 3601, float("nan"), float("inf"), "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Runtime(encoder, maintenance_interval_s=value)
        for value in (0, 1, 3600):
            Runtime(encoder, maintenance_interval_s=value)

    def test_unloaded_loading_changed_pending_and_busy_skip_without_refresh(self):
        encoder = Encoder()
        runtime = Runtime(encoder, queue_depth=1, maintenance_interval_s=1)
        for field, value in (("loaded", False), ("loading", True), ("weight_changed", True)):
            old = getattr(encoder, field)
            setattr(encoder, field, value)
            self.assertFalse(runtime.maintain_once())
            setattr(encoder, field, old)
        queued = _QueueJob(runtime)
        self.assertFalse(runtime.maintain_once())
        queued.abandon()
        runtime.serial.acquire()
        self.assertFalse(runtime.maintain_once())
        runtime.serial.release()
        with runtime.state:
            runtime.active = True
        self.assertFalse(runtime.maintain_once())
        with runtime.state:
            runtime.active = False
        self.assertEqual((encoder.load_calls, encoder.refresh_calls, encoder.maintenance_calls), (0, 0, 0))
        self.assertEqual(slots_available(runtime), 2)

    def test_no_last_used_or_slots_side_effect_and_idle_expiry_really_unloads(self):
        encoder = Encoder()
        runtime = Runtime(encoder, queue_depth=1, maintenance_interval_s=1, idle_unload_min=1)
        previous = runtime.last_used
        with patch.object(runtime, "leave", side_effect=AssertionError("Maintenance must not leave business slot")):
            self.assertTrue(runtime.maintain_once())
        self.assertEqual(runtime.last_used, previous)
        self.assertEqual(runtime.maintenance_count, 1)
        self.assertGreater(runtime.maintenance_total_ms, 0)
        self.assertEqual(slots_available(runtime), 2)
        runtime.last_used = time.monotonic() - 61
        self.assertFalse(runtime.maintain_once())
        runtime.reap_idle()
        self.assertEqual(encoder.unload_calls, 1)
        self.assertFalse(runtime.maintain_once())
        self.assertEqual(encoder.maintenance_calls, 1)
        self.assertEqual(encoder.load_calls, 0)

    def test_exception_stops_profile_once_and_releases_lock_and_exact_slots(self):
        encoder = Encoder()
        encoder.fail = True
        runtime = Runtime(encoder, queue_depth=1, maintenance_interval_s=1)
        previous = runtime.last_used
        with self.assertLogs(level="ERROR") as logs:
            self.assertFalse(runtime.maintain_once())
        for _ in range(10):
            self.assertFalse(runtime.maintain_once())
        self.assertEqual(encoder.maintenance_calls, 1)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(runtime.maintenance_error, "RuntimeError")
        self.assertTrue(runtime.maintenance_stopped)
        self.assertFalse(runtime.maintenance_active)
        self.assertEqual(runtime.last_used, previous)
        self.assertTrue(runtime.serial.acquire(blocking=False))
        runtime.serial.release()
        self.assertEqual(slots_available(runtime), 2)

    def test_health_reports_maintenance_as_busy_and_preserves_business_marker(self):
        encoder = Encoder()
        encoder.block = True
        app = create_app({"maintenance_interval_s": 1}, {"qwen3vl_8b": encoder})
        runtime = app.state.runtimes["qwen3vl_8b"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime.maintain_once)
            try:
                self.assertTrue(encoder.started.wait(1))
                profile = TestClient(app).get("/health").json()["profiles"]["qwen3vl_8b"]
                self.assertTrue(profile["maintenance_active"])
                self.assertTrue(profile["busy"])
                self.assertFalse(profile["business_active"])
                self.assertEqual(profile["queue_len"], 0)
            finally:
                encoder.release.set()
            self.assertTrue(future.result(timeout=1))
        profile = TestClient(app).get("/health").json()["profiles"]["qwen3vl_8b"]
        self.assertEqual(profile["maintenance_count"], 1)
        self.assertGreaterEqual(profile["maintenance_total_ms"], profile["maintenance_last_ms"])
        self.assertIsNone(profile["maintenance_error"])

    def test_native_entrypoint_uses_current_clip_only_without_lazy_load_or_timing_mutation(self):
        service = TEService.__new__(TEService)
        service.loading = False
        service.last_timing = {"business": "unchanged"}
        service.load = Mock(side_effect=AssertionError("no load"))
        service.refresh = Mock(side_effect=AssertionError("no refresh"))
        service.clip = None
        with patch("torch.cuda.synchronize") as sync:
            self.assertFalse(service.maintain_loaded())
            sync.assert_not_called()
            service.clip = Mock()
            service.clip.tokenize.return_value = "native-tokens"
            self.assertTrue(service.maintain_loaded())
            service.clip.tokenize.assert_called_once_with("warmup", images=[], keep_vision=True, prevent_empty_text=True)
            service.clip.encode_from_tokens_scheduled.assert_called_once_with("native-tokens")
            sync.assert_called_once()
            service.loading = True
            self.assertFalse(service.maintain_loaded())
        service.load.assert_not_called()
        service.refresh.assert_not_called()
        self.assertEqual(service.last_timing, {"business": "unchanged"})
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_forward_error_still_synchronizes_and_sync_error_stops_profile(self):
        encoder = Encoder()
        encoder.clip = Mock()
        encoder.clip.encode_from_tokens_scheduled.side_effect = ValueError("original forward error")
        with patch("torch.cuda.synchronize") as sync:
            with self.assertRaisesRegex(ValueError, "original forward error"):
                TEService.maintain_loaded(encoder)
            sync.assert_called_once()
        with patch("torch.cuda.synchronize", side_effect=RuntimeError("sync failed")):
            with self.assertRaisesRegex(RuntimeError, "sync failed") as failure:
                TEService.maintain_loaded(encoder)
            self.assertIsInstance(failure.exception.__context__, ValueError)
        encoder.clip.encode_from_tokens_scheduled.side_effect = None
        encoder.maintain_loaded = lambda: TEService.maintain_loaded(encoder)
        runtime = Runtime(encoder, queue_depth=1, maintenance_interval_s=1)
        with patch("torch.cuda.synchronize", side_effect=RuntimeError("sync failed")), self.assertLogs(level="ERROR"):
            self.assertFalse(runtime.maintain_once())
        self.assertTrue(runtime.maintenance_stopped)
        self.assertEqual(runtime.maintenance_error, "RuntimeError")
        self.assertFalse(runtime.maintenance_active)
        self.assertEqual(slots_available(runtime), 2)


class MaintenanceConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.encoder = Encoder()
        self.app = create_app({"queue_depth": 1, "maintenance_interval_s": 1}, {"qwen3vl_8b": self.encoder})
        self.runtime = self.app.state.runtimes["qwen3vl_8b"]
        self.requests = []
        self.tasks = []

    async def asyncTearDown(self):
        self.encoder.release.set()
        await asyncio.wait_for(asyncio.gather(*(r.task for r in self.requests), *self.tasks, return_exceptions=True), 4)

    async def wait_until(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("State did not reach expected value")
            await asyncio.sleep(.005)

    def request(self, path="/encode"):
        request = ASGIRequest(self.app, "business", path)
        self.requests.append(request)
        return request

    async def maintenance(self):
        self.encoder.block = True
        task = asyncio.create_task(asyncio.to_thread(self.runtime.maintain_once))
        self.tasks.append(task)
        await self.wait_until(self.encoder.started.is_set)
        return task

    async def test_later_business_waits_visibly_without_overlap_then_runs(self):
        maintenance = await self.maintenance()
        request = self.request()
        await self.wait_until(lambda: self.runtime.pending == 1)
        self.assertFalse(self.runtime.active)
        self.assertTrue(self.runtime.health_state()["busy"])
        self.assertEqual(self.encoder.encode_calls, 0)
        observed_wait_start = time.perf_counter()
        await asyncio.sleep(.06)
        observed_wait_ms = (time.perf_counter() - observed_wait_start) * 1000
        self.assertEqual(self.runtime.pending, 1)
        self.assertEqual(self.encoder.encode_calls, 0)
        self.encoder.release.set()
        await asyncio.wait_for(asyncio.gather(maintenance, request.task), 2)
        self.assertEqual(request.response["status"], 200)
        body = b"".join(message.get("body", b"") for message in request.sent
                        if message["type"] == "http.response.body")
        header_size = struct.unpack("<Q", body[:8])[0]
        metadata = json.loads(body[8:8 + header_size])["__metadata__"]
        # The actual transport receipt must include observed maintenance wait;
        # a successful response or visible pending count alone cannot prove it.
        self.assertGreaterEqual(float(metadata["queue_wait_ms"]), observed_wait_ms - 1)
        self.assertEqual(self.encoder.events, ["maintenance-start", "maintenance-end", "business"])
        self.assertEqual(self.encoder.peak, 1)
        self.assertEqual(slots_available(self.runtime), 2)

    async def test_pending_unload_prevents_next_maintenance_and_never_reloads(self):
        maintenance = await self.maintenance()
        request = self.request("/unload")
        await self.wait_until(lambda: self.runtime.pending == 1)
        self.encoder.release.set()
        await asyncio.wait_for(maintenance, 1)
        self.assertFalse(await asyncio.to_thread(self.runtime.maintain_once))
        await asyncio.wait_for(request.task, 2)
        self.assertEqual(request.response["status"], 200)
        self.assertFalse(await asyncio.to_thread(self.runtime.maintain_once))
        self.assertEqual(self.encoder.maintenance_calls, 1)
        self.assertEqual(self.encoder.unload_calls, 1)
        self.assertEqual(self.encoder.load_calls, 0)
        self.assertEqual(slots_available(self.runtime), 2)

    async def test_cancelling_queued_business_does_not_release_maintenance_serial_or_extra_slot(self):
        maintenance = await self.maintenance()
        request = self.request()
        await self.wait_until(lambda: self.runtime.pending == 1)
        request.task.cancel()
        await asyncio.gather(request.task, return_exceptions=True)
        await self.wait_until(lambda: self.runtime.pending == 0)
        self.assertTrue(self.runtime.maintenance_active)
        self.assertFalse(self.runtime.serial.acquire(blocking=False))
        self.assertEqual(slots_available(self.runtime), 2)
        self.encoder.release.set()
        await asyncio.wait_for(maintenance, 1)
        self.assertEqual(self.encoder.encode_calls, 0)

    async def test_cancel_shutdown_waits_for_inflight_maintenance_without_blocking_loop(self):
        context = self.app.router.lifespan_context(self.app)
        await context.__aenter__()
        maintenance = await self.maintenance()
        queued = self.request()
        await self.wait_until(lambda: self.runtime.pending == 1)
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        self.tasks.append(shutdown)
        await self.wait_until(lambda: self.runtime.closing)
        await asyncio.wait_for(queued.task, 1)
        self.assertEqual(queued.response["status"], 503)
        shutdown.cancel()
        await asyncio.sleep(.02)
        shutdown.cancel()
        await asyncio.sleep(.02)
        self.assertFalse(shutdown.done())
        self.assertEqual(self.encoder.unload_calls, 0)
        with self.assertRaises(HTTPException) as error:
            _QueueJob(self.runtime)
        self.assertEqual(error.exception.status_code, 503)
        self.encoder.release.set()
        await asyncio.wait_for(maintenance, 1)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(shutdown, 2)
        self.assertEqual(self.encoder.events, ["maintenance-start", "maintenance-end", "unload"])
        self.assertEqual(slots_available(self.runtime), 2)

    async def test_shutdown_wakes_long_interval_without_waiting_for_tick(self):
        self.runtime.maintenance_interval_s = 3600
        context = self.app.router.lifespan_context(self.app)
        await context.__aenter__()
        await asyncio.wait_for(context.__aexit__(None, None, None), .5)
        self.assertEqual(self.encoder.maintenance_calls, 0)
        self.assertEqual(self.encoder.unload_calls, 1)

    async def test_failed_native_forward_keeps_serial_until_blocked_sync_finishes(self):
        context = self.app.router.lifespan_context(self.app)
        await context.__aenter__()
        sync_started, sync_release, sync_finished = threading.Event(), threading.Event(), threading.Event()
        self.encoder.clip = Mock()
        self.encoder.clip.encode_from_tokens_scheduled.side_effect = RuntimeError("partial native forward")
        self.encoder.maintain_loaded = lambda: TEService.maintain_loaded(self.encoder)
        original_unload = self.encoder.unload
        def unload_after_sync():
            self.assertTrue(sync_finished.is_set(), "Unload ran before failed forward finished synchronizing")
            original_unload()
        def blocked_sync():
            sync_started.set()
            if not sync_release.wait(3):
                raise RuntimeError("Fixture sync release timeout")
            sync_finished.set()
        self.encoder.unload = unload_after_sync
        with patch("torch.cuda.synchronize", side_effect=blocked_sync), self.assertLogs(level="ERROR"):
            maintenance = asyncio.create_task(asyncio.to_thread(self.runtime.maintain_once))
            self.tasks.append(maintenance)
            shutdown = None
            try:
                await self.wait_until(sync_started.is_set)
                queued = self.request("/unload")
                await self.wait_until(lambda: self.runtime.pending == 1)
                await asyncio.sleep(.03)
                self.assertEqual(self.encoder.unload_calls, 0)
                self.assertTrue(self.runtime.maintenance_active)
                self.assertFalse(self.runtime.serial.acquire(blocking=False))
                shutdown = asyncio.create_task(context.__aexit__(None, None, None))
                self.tasks.append(shutdown)
                await self.wait_until(lambda: self.runtime.closing)
                await asyncio.wait_for(queued.task, 1)
                self.assertEqual(queued.response["status"], 503)
                self.assertFalse(shutdown.done())
                self.assertEqual(self.encoder.unload_calls, 0)
            finally:
                sync_release.set()
            self.assertFalse(await asyncio.wait_for(maintenance, 1))
            if shutdown is None:
                await context.__aexit__(None, None, None)
            else:
                await asyncio.wait_for(shutdown, 1)
        self.assertTrue(self.runtime.maintenance_stopped)
        self.assertEqual(self.runtime.maintenance_error, "RuntimeError")
        self.assertEqual(self.encoder.unload_calls, 1)
        self.assertEqual(slots_available(self.runtime), 2)

    async def test_shutdown_before_queued_business_worker_starts_single_executor(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        occupied, release = threading.Event(), threading.Event()
        def blocker():
            occupied.set()
            release.wait(4)
        blocker_future = loop.run_in_executor(None, blocker)
        await self.wait_until(occupied.is_set)
        context = self.app.router.lifespan_context(self.app)
        await context.__aenter__()
        request = self.request()
        await self.wait_until(lambda: self.runtime.active)
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        self.tasks.append(shutdown)
        try:
            await self.wait_until(lambda: self.runtime.closing)
            await asyncio.sleep(.02)
            self.assertFalse(shutdown.done())
            self.assertEqual(self.encoder.encode_calls, 0)
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(blocker_future, request.task, shutdown), 2)
        self.assertEqual(request.response["status"], 503)
        self.assertEqual(self.encoder.encode_calls, 0)
        self.assertEqual(self.encoder.unload_calls, 1)

    async def test_shutdown_before_scheduled_maintenance_worker_starts(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        occupied, release = threading.Event(), threading.Event()
        def blocker():
            occupied.set()
            release.wait(4)
        blocker_future = loop.run_in_executor(None, blocker)
        await self.wait_until(occupied.is_set)
        context = self.app.router.lifespan_context(self.app)
        await context.__aenter__()
        # The real 1-second scheduler queues its worker behind this fixture.
        await asyncio.sleep(1.1)
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        self.tasks.append(shutdown)
        try:
            await self.wait_until(lambda: self.runtime.closing)
            self.assertEqual(self.encoder.maintenance_calls, 0)
            self.assertFalse(shutdown.done())
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(blocker_future, shutdown), 2)
        self.assertEqual(self.encoder.maintenance_calls, 0)
        self.assertEqual(self.encoder.unload_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
