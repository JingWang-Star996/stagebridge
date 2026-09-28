import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt
import psutil
from . import __version__
from .codec import decode_images, pack


class EncodeReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "qwen3vl_8b"
    mode: Literal["t2i", "edit", "t2va", "fl2va"] = "t2i"
    prompt: str = Field(max_length=32768)
    resolution: int = Field(default=1024, ge=0, le=4096, strict=True)
    fingerprint: str | None = Field(default=None, min_length=16, max_length=64)
    ref_image_b64: str | None = None
    ref_image_format: str | None = None
    ref_images_safetensors_b64: str | None = None


class ModelReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "qwen3vl_8b"


class _QueueJob:
    """Own a queue slot and transfer the serial lock to the actual worker."""
    def __init__(self, runtime, cancelled=None):
        self.runtime = runtime
        self.guard = threading.Lock()
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.phase = "pending"
        with runtime.state:
            if runtime.closing:
                raise HTTPException(503, "TE service is shutting down")
            if not runtime.slots.acquire(blocking=False):
                raise HTTPException(429, "TE queue is full", headers={"Retry-After": "1"})
            runtime.pending += 1

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise HTTPException(503, "Request disconnected before execution")
        with self.runtime.state:
            if self.runtime.closing:
                raise HTTPException(503, "TE service is shutting down")

    def acquire(self):
        with self.guard:
            if self.phase != "pending" or not self.runtime.serial.acquire(blocking=False):
                return False
            with self.runtime.state:
                if self.runtime.closing:
                    self.runtime.serial.release()
                    raise HTTPException(503, "TE service is shutting down")
                self.runtime.pending -= 1
                self.runtime.active = True
            self.phase = "dispatched"
            return True

    def _release(self):
        # Caller holds guard. Exactly one owner releases each reservation.
        if self.phase == "done":
            return
        if self.phase == "pending":
            with self.runtime.state:
                self.runtime.pending -= 1
            self.runtime.slots.release()
        else:
            self.runtime.leave()
        self.phase = "done"

    def abandon(self):
        with self.guard:
            self.cancelled.set()
            if self.phase != "running":
                self._release()

    def invoke(self, operation, wait_ms):
        with self.guard:
            self.check_cancelled()
            self.phase = "running"
        try:
            return operation(wait_ms, self.check_cancelled)
        finally:
            with self.guard:
                self._release()


class Runtime:
    def __init__(self, encoder, queue_depth=16, queue_wait_timeout_s=60,
                 maintenance_interval_s=0, idle_unload_min=0):
        self.encoder = encoder
        self.serial = threading.Lock()
        self.state = threading.Lock()
        self.slots = threading.BoundedSemaphore(queue_depth + 1)
        self.pending = 0
        self.active = False
        self.closing = False
        self.last_used = time.monotonic()
        if type(maintenance_interval_s) not in (int, float) or not math.isfinite(maintenance_interval_s) or (
            maintenance_interval_s != 0 and not 1 <= maintenance_interval_s <= 3600
        ):
            raise ValueError("maintenance_interval_s must be 0 (disabled) or between 1 and 3600 seconds")
        self.maintenance_interval_s = float(maintenance_interval_s)
        self.idle_unload_s = float(idle_unload_min) * 60
        if not math.isfinite(self.idle_unload_s) or self.idle_unload_s < 0:
            raise ValueError("idle_unload_min must be finite and nonnegative")
        self.maintenance_active = False
        self.maintenance_count = 0
        self.maintenance_last_ms = 0.0
        self.maintenance_total_ms = 0.0
        self.maintenance_error = None
        self.maintenance_stopped = False
        self.queue_wait_timeout_s = float(queue_wait_timeout_s)
        if not math.isfinite(self.queue_wait_timeout_s) or self.queue_wait_timeout_s <= 0:
            raise ValueError("queue_wait_timeout_s must be finite and positive")

    def _idle_due(self):
        return bool(self.idle_unload_s and time.monotonic() - self.last_used >= self.idle_unload_s)

    def maintain_once(self):
        if not self.maintenance_interval_s or not self.serial.acquire(blocking=False):
            return False
        started = None
        try:
            with self.state:
                if self.pending or self.active or self.closing or self.maintenance_stopped or self._idle_due():
                    return False
                if self.encoder.loading or self.encoder.changed() or not self.encoder.stats().get("loaded", False):
                    return False
                self.maintenance_active = True
                started = time.perf_counter()
            completed = self.encoder.maintain_loaded()
            if completed:
                with self.state:
                    self.maintenance_count += 1
            return bool(completed)
        except Exception as exc:
            with self.state:
                self.maintenance_error = type(exc).__name__
                self.maintenance_stopped = True
            logging.exception("TE maintenance stopped after failure; restart required to re-enable this profile")
            return False
        finally:
            with self.state:
                if started is not None:
                    self.maintenance_last_ms = (time.perf_counter() - started) * 1000
                    self.maintenance_total_ms += self.maintenance_last_ms
                self.maintenance_active = False
            # Maintenance owns only serial: no business slot and no last_used update.
            self.serial.release()

    def reap_idle(self):
        if not self.serial.acquire(blocking=False):
            return
        try:
            with self.state:
                due = not (self.pending or self.active or self.closing) and self._idle_due()
            if due:
                self.encoder.unload()
        finally:
            self.serial.release()

    async def unload_for_shutdown(self):
        # Poll on the loop rather than occupying an executor worker waiting for
        # serial: a dispatched business worker may still need that executor.
        while not self.serial.acquire(blocking=False):
            await asyncio.sleep(0.01)
        def unload_owned():
            try:
                self.encoder.unload()
            finally:
                self.serial.release()
        await asyncio.to_thread(unload_owned)

    def health_state(self):
        with self.state:
            return {"queue_len": self.pending, "busy": self.active or self.maintenance_active,
                    "business_active": self.active, "closing": self.closing,
                    "maintenance_interval_s": self.maintenance_interval_s,
                    "maintenance_active": self.maintenance_active, "maintenance_count": self.maintenance_count,
                    "maintenance_last_ms": round(self.maintenance_last_ms, 2),
                    "maintenance_total_ms": round(self.maintenance_total_ms, 2),
                    "maintenance_error": self.maintenance_error, "maintenance_stopped": self.maintenance_stopped}

    async def run(self, request, operation):
        job = _QueueJob(self, request.scope.get("te_disconnected"))
        start = time.perf_counter()

        async def watch_disconnect():
            # FastAPI has already consumed the body. Unlike is_disconnected(),
            # this works through BaseHTTPMiddleware's cancellable receive wrapper.
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    job.abandon()
                    return

        watcher = asyncio.create_task(watch_disconnect())
        try:
            await asyncio.sleep(0)
            while True:
                job.check_cancelled()
                if time.perf_counter() - start >= self.queue_wait_timeout_s:
                    raise HTTPException(503, "TE queue wait timed out", headers={"Retry-After": "1"})
                if job.acquire():
                    break
                await asyncio.sleep(0.02)
            wait_ms = (time.perf_counter() - start) * 1000
            worker = asyncio.create_task(asyncio.to_thread(job.invoke, operation, wait_ms))
            # Cancellation can arrive before the executor starts the worker. The
            # job releases that reservation immediately and invoke then skips it.
            worker.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            return await asyncio.shield(worker)
        finally:
            job.abandon()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    def leave(self):
        with self.state:
            self.active = False
            self.last_used = time.monotonic()
        self.serial.release()
        self.slots.release()


def create_app(config, encoders=None):
    runtimes = {}
    started = time.monotonic()
    process = psutil.Process()
    token = os.environ.get("TE_AUTH_TOKEN") or config.get("auth_token")
    if encoders is None:
        from .encoder import TEService
        encoders = {name: TEService(p["ckpt_path"], model=name)
                    for name, p in config["profiles"].items()
                    if name in ("qwen3vl_8b", "minimax_h3") and not p.get("reserved", False) and p.get("ckpt_path")}
    for name, encoder in encoders.items():
        runtimes[name] = Runtime(encoder, int(config.get("queue_depth", 16)),
                                 config.get("queue_wait_timeout_s", 60),
                                 config.get("maintenance_interval_s", 0), config.get("idle_unload_min", 0))

    exclusive_profiles = bool(config.get("exclusive_profiles", False))
    if exclusive_profiles:
        # Queue dispatch, maintenance, idle cleanup and shutdown must all use
        # the same lock: no profile may unload weights during another forward.
        shared_serial = threading.Lock()
        for rt in runtimes.values():
            rt.serial = shared_serial

    def select_profile(selected):
        if exclusive_profiles:
            # The caller owns the shared serial lock. Pending jobs cannot start
            # until the old weights have been released.
            for name, rt in runtimes.items():
                if name != selected and rt.encoder.stats().get("loaded", False):
                    rt.encoder.unload()

    def profile(name):
        if name not in runtimes:
            raise HTTPException(404, "Unknown or reserved model profile")
        return runtimes[name]

    async def wait_or_stop(stop, interval):
        try:
            await asyncio.wait_for(stop.wait(), interval)
            return True
        except asyncio.TimeoutError:
            return False

    async def idle_reaper(stop):
        while not await wait_or_stop(stop, 5):
            for rt in runtimes.values():
                await asyncio.to_thread(rt.reap_idle)

    async def maintenance_loop(rt, stop):
        while not await wait_or_stop(stop, rt.maintenance_interval_s):
            await asyncio.to_thread(rt.maintain_once)
            with rt.state:
                if rt.maintenance_stopped:
                    return

    @asynccontextmanager
    async def lifespan(app):
        stop = asyncio.Event()
        background = [asyncio.create_task(idle_reaper(stop))]
        background.extend(asyncio.create_task(maintenance_loop(rt, stop))
                          for rt in runtimes.values() if rt.maintenance_interval_s)
        try:
            yield
        finally:
            for rt in runtimes.values():
                with rt.state:
                    rt.closing = True
            stop.set()

            async def finish():
                results = await asyncio.gather(*background, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        logging.error("TE background task ended with %s", type(result).__name__)
                unloaded = await asyncio.gather(*(rt.unload_for_shutdown() for rt in runtimes.values()),
                                                return_exceptions=True)
                for result in unloaded:
                    if isinstance(result, BaseException):
                        raise result

            cleanup = asyncio.create_task(finish())
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError()

    app = FastAPI(title="Remote TE", version=__version__, lifespan=lifespan)
    app.state.runtimes = runtimes

    @app.middleware("http")
    async def validate_request(request, call_next):
        if token and not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {token}"):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > 135 * 1024 * 1024:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 135 * 1024 * 1024:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
            request._body = bytes(body)
        # Observe the transport before BaseHTTPMiddleware adds receive wrappers.
        # Replay the cached POST body as usual; subsequent reads receive only the
        # disconnect signal, so exactly one task consumes the raw ASGI channel.
        raw_receive = request._receive
        disconnected = asyncio.Event()
        cancelled = threading.Event()
        request.scope["te_disconnected"] = cancelled

        async def watch_transport():
            while True:
                if (await raw_receive())["type"] == "http.disconnect":
                    cancelled.set()
                    disconnected.set()
                    return

        async def receive_disconnect():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        request._receive = receive_disconnect
        watcher = asyncio.create_task(watch_transport())
        try:
            # Let an already-delivered disconnect become visible before routing.
            await asyncio.sleep(0)
            return await call_next(request)
        finally:
            cancelled.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        # Do not echo user prompts or image base64 into errors/logs.
        return JSONResponse({"detail": [{"loc": e["loc"], "type": e["type"], "msg": e["msg"]} for e in exc.errors()]}, status_code=400)

    @app.get("/health")
    def health():
        profiles = {}
        for name, rt in runtimes.items():
            # Refresh an externally replaced checkpoint only when no forward is active.
            if rt.encoder.changed() and rt.serial.acquire(blocking=False):
                try:
                    with rt.state:
                        refresh_allowed = not rt.closing
                    if refresh_allowed:
                        rt.encoder.refresh()
                finally:
                    rt.serial.release()
            profiles[name] = {**rt.encoder.stats(), **rt.health_state()}
        return {"status": "ok", "profiles": profiles, "queue_len": sum(r.pending for r in runtimes.values()),
                "uptime_s": round(time.monotonic() - started, 1), "rss_mb": round(process.memory_info().rss / 2**20, 1),
                "server_version": __version__, "reserved_profiles": ["minimax_h3"] if "minimax_h3" not in runtimes else []}

    @app.post("/load")
    async def load(req: ModelReq, request: Request):
        rt = profile(req.model)
        def operation(wait_ms, check_cancelled):
            try:
                check_cancelled()
                rt.encoder.refresh()
                check_cancelled()
                select_profile(req.model)
                rt.encoder.load()
                return rt.encoder.stats()
            except HTTPException:
                raise
            except Exception as e:
                logging.exception("TE load failed")
                raise HTTPException(503, "Encoder load failed; inspect server log") from e
        return await rt.run(request, operation)

    @app.post("/unload")
    async def unload(req: ModelReq, request: Request):
        rt = profile(req.model)
        def operation(wait_ms, check_cancelled):
            check_cancelled()
            rt.encoder.unload()
            return rt.encoder.stats()
        return await rt.run(request, operation)

    @app.post("/encode")
    async def encode(req: EncodeReq, request: Request):
        rt = profile(req.model)
        if (req.model == "qwen3vl_8b" and req.mode not in ("t2i", "edit")) or (
                req.model == "minimax_h3" and req.mode not in ("t2va", "fl2va")):
            raise HTTPException(400, "Mode does not match model profile")
        if rt.encoder.loading:
            raise HTTPException(503, "Encoder is loading", headers={"Retry-After": "1"})
        return await rt.run(request, lambda wait_ms, check_cancelled: encode_work(req, rt, wait_ms, check_cancelled))

    def encode_work(req, rt, wait_ms, check_cancelled):
        try:
            check_cancelled()
            rt.encoder.refresh()
            if req.fingerprint and req.fingerprint != rt.encoder.fingerprint:
                raise HTTPException(409, {"error": "fingerprint_mismatch", "expected": rt.encoder.fingerprint})
            try:
                images = decode_images(req)
            except Exception as e:
                raise HTTPException(400, str(e)) from e
            check_cancelled()
            select_profile(req.model)
            start = time.perf_counter()
            result = rt.encoder.encode(req.prompt, req.resolution, req.mode, images)
            encode_ms = (time.perf_counter() - start) * 1000
            serialize_start = time.perf_counter()
            try:
                blob = pack(result, {"model": req.model, "fingerprint": rt.encoder.fingerprint, "mode": req.mode,
                                     "encode_ms": round(encode_ms, 2), "queue_wait_ms": round(wait_ms, 2), "server_version": __version__})
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            serialize_ms = (time.perf_counter() - serialize_start) * 1000
            phases = getattr(rt.encoder, "last_timing", {})
            logging.info("encode model=%s mode=%s seq_len=%d queue_wait_ms=%.2f encode_ms=%.2f serialize_ms=%.2f bytes=%d native_phases=%s",
                         req.model, req.mode, result[0][0].shape[1], wait_ms, encode_ms, serialize_ms, len(blob), phases)
            headers = {"X-TE-Serialize-Ms": f"{serialize_ms:.2f}", "X-TE-Fingerprint": rt.encoder.fingerprint}
            for field, header in (("load_ms", "X-TE-Load-Ms"), ("prepare_ms", "X-TE-Prepare-Ms"),
                                  ("tokenize_ms", "X-TE-Tokenize-Ms"), ("forward_sync_ms", "X-TE-Forward-Sync-Ms")):
                if field in phases:
                    headers[header] = f"{phases[field]:.2f}"
            return Response(blob, media_type="application/octet-stream", headers=headers)
        except HTTPException:
            raise
        except Exception as e:
            logging.exception("TE encode failed")
            raise HTTPException(503, "Encoder failed; inspect server log") from e

    return app


def main():
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sys.argv = [sys.argv[0]]
    root = str(Path(config["comfy_root"]).resolve())
    sys.path.insert(0, root)
    os.chdir(root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = create_app(config)
    import uvicorn
    uvicorn.run(app, host=config.get("host", "127.0.0.1"), port=int(config.get("port", 8765)), workers=1, access_log=False)


if __name__ == "__main__":
    main()
