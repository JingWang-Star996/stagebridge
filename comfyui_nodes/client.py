"""Bounded stdlib HTTP client; all calls from async nodes run in worker threads."""
import asyncio
import http.client
import json
import os
import re
import socket
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request

from .codec import MAX_RESPONSE_BYTES, MODEL, decode_conditioning, encode_images, strict_json, validate_request
from .errors import ConfigurationError, ProtocolError, RetryableRemoteError, VersionMismatch

DEFAULT_SERVER_URL = os.environ.get("REMOTE_TE_SERVER_URL", "http://127.0.0.1:8765")
FINGERPRINT_RE = re.compile(r"[A-Za-z0-9_.:+-]{1,256}\Z")


def fingerprint_value(value):
    if not isinstance(value, str) or not FINGERPRINT_RE.fullmatch(value):
        raise ProtocolError("Encoder fingerprint is missing or invalid.")
    return value


def server_url(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise ConfigurationError("Server URL must be an HTTP(S) base URL.")
    try:
        parsed = urllib.parse.urlsplit(value.strip())
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError()
        parsed.port
    except ValueError:
        raise ConfigurationError("Use an HTTP(S) base URL without credentials, query or fragment.") from None
    return value.strip().rstrip("/")


class _PersistentOpener:
    """One direct connection; caller serializes open + complete response read."""
    def __init__(self, url):
        self.base = urllib.parse.urlsplit(url)
        self._state = threading.Lock()
        self._connection = None
        self._socket = None
        self._closed = False

    def abort(self, *, close=False):
        # Never take the request lock: close must interrupt a blocked response.
        with self._state:
            if close:
                self._closed = True
            connection, active_socket = self._connection, self._socket
            self._connection = self._socket = None
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if connection is not None:
            connection.close()

    def open(self, request, timeout):
        target = urllib.parse.urlsplit(request.full_url)
        if (target.scheme, target.hostname, target.port) != (self.base.scheme, self.base.hostname, self.base.port):
            raise ConfigurationError("Remote request origin changed.")
        with self._state:
            if self._closed:
                raise RetryableRemoteError("Remote client is closed.")
            connection = self._connection
            if connection is None:
                if self.base.scheme == "https":
                    connection = http.client.HTTPSConnection(self.base.hostname, self.base.port, timeout=timeout,
                                                             context=ssl.create_default_context())
                else:
                    connection = http.client.HTTPConnection(self.base.hostname, self.base.port, timeout=timeout)
                # A stale socket may fail, but request() must never reconnect/replay.
                connection.auto_open = 0
                self._connection = connection
        connection.timeout = timeout
        if connection.sock is None:
            connection.connect()
        with self._state:
            if self._closed or connection is not self._connection:
                connection.close()
                raise RetryableRemoteError("Remote client was closed while connecting.")
            self._socket = connection.sock
            self._socket.settimeout(timeout)
        path = urllib.parse.urlunsplit(("", "", target.path or "/", target.query, ""))
        connection.request(request.get_method(), path, body=request.data, headers=dict(request.header_items()))
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            raise urllib.error.HTTPError(request.full_url, response.status, response.reason, response.headers, response)
        return response


class RemoteClient:
    def __init__(self, url, timeout=30.0):
        self.url = server_url(url)
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 900:
            raise ConfigurationError("Timeout must be between 1 and 900 seconds.")
        self.timeout = float(timeout)
        # Local/private endpoints must not inherit system proxies. Credentials stay
        # in the environment and are never workflow widgets or log messages.
        self.opener = _PersistentOpener(self.url)
        self._request_lock = threading.RLock()

    def close(self):
        self.opener.abort(close=True)

    async def aclose(self):
        # A separate short-lived closer cannot queue behind a saturated pool of
        # to_thread network calls. Socket shutdown/close never runs on the loop.
        loop = asyncio.get_running_loop()
        finished = loop.create_future()

        def complete(error):
            if error is None:
                finished.set_result(None)
            else:
                finished.set_exception(error)

        def close_in_thread():
            try:
                self.close()
            except BaseException as exc:
                loop.call_soon_threadsafe(complete, exc)
            else:
                loop.call_soon_threadsafe(complete, None)

        threading.Thread(target=close_in_thread, name="remote-te-close", daemon=True).start()
        cancelled = False
        while not finished.done():
            try:
                await asyncio.shield(finished)
            except asyncio.CancelledError:
                cancelled = True
        finished.result()
        if cancelled:
            raise asyncio.CancelledError()

    async def _async_call(self, function, *args, **kwargs):
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Shielded to_thread work must have its later exception retrieved.
            worker.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            await self.aclose()
            raise

    def _request(self, method, path, payload=None, health=False):
        with self._request_lock:
            try:
                return self._request_locked(method, path, payload, health)
            except BaseException:
                self.opener.abort()
                raise

    def _request_locked(self, method, path, payload=None, health=False):
        headers = {"Accept": "application/json" if health else "application/octet-stream"}
        token = os.environ.get("REMOTE_TE_API_TOKEN", "")
        if token:
            if "\r" in token or "\n" in token:
                raise ConfigurationError("REMOTE_TE_API_TOKEN contains invalid characters.")
            headers["Authorization"] = "Bearer " + token
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=min(self.timeout, 5.0) if health else self.timeout) as response:
                limit = 64 * 1024 if health else MAX_RESPONSE_BYTES
                raw_length = response.headers.get("Content-Length")
                if len(response.headers.get_all("Content-Length", [])) > 1 or (
                    raw_length is not None and response.headers.get("Transfer-Encoding") is not None
                ):
                    raise ProtocolError("Ambiguous response body framing.")
                content_length = None
                if raw_length is not None:
                    try:
                        content_length = int(raw_length)
                    except ValueError:
                        raise ProtocolError("Invalid response Content-Length.") from None
                    if content_length < 0 or content_length > limit:
                        raise ProtocolError("Remote response exceeds the permitted size.")
                content = response.read(limit + 1)
                if len(content) > limit:
                    raise ProtocolError("Remote response exceeds the permitted size.")
                if content_length is not None and len(content) != content_length:
                    raise http.client.IncompleteRead(content, max(0, content_length - len(content)))
                return content
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            if status == 409:
                raise VersionMismatch("Remote encoder rejected the pinned fingerprint (HTTP 409). Set expected_fingerprint explicitly to approve a new version.") from None
            if status == 429 or 500 <= status <= 599:
                raise RetryableRemoteError(f"Remote encoder temporarily unavailable (HTTP {status}).") from None
            raise ProtocolError(f"Remote encoder rejected the request (HTTP {status}); local fallback is disabled for this error.") from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError, http.client.IncompleteRead):
            raise RetryableRemoteError("Remote encoder network connection failed or timed out.") from None
        except http.client.HTTPException:
            raise ProtocolError("Malformed HTTP response from remote encoder.") from None

    def health_sync(self, model=MODEL):
        with self._request_lock:
            try:
                return self._health_sync(model)
            except BaseException:
                self.opener.abort()
                raise

    def _health_sync(self, model):
        payload = strict_json(self._request("GET", "/health", health=True))
        if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
            raise ProtocolError("Health response is missing profiles.")
        profile = payload["profiles"].get(model)
        if not isinstance(profile, dict) or type(profile.get("loaded")) is not bool:
            raise ProtocolError("Health response is missing a valid encoder profile.")
        return {"fingerprint": fingerprint_value(profile.get("fingerprint")), "loaded": profile["loaded"]}

    async def health(self, model=MODEL):
        return await self._async_call(self.health_sync, model)

    def encode_sync(self, prompt, *, model=MODEL, mode="t2i", resolution=1024, fingerprint, images=()):
        with self._request_lock:
            try:
                return self._encode_sync(prompt, model=model, mode=mode, resolution=resolution,
                                         fingerprint=fingerprint, images=images)
            except BaseException:
                self.opener.abort()
                raise

    def _encode_sync(self, prompt, *, model, mode, resolution, fingerprint, images):
        validate_request(prompt, model, mode, resolution)
        fingerprint_value(fingerprint)
        if model == "minimax_h3" and ((mode == "t2va" and images) or
                                       (mode == "fl2va" and not 1 <= len(images) <= 2)):
            raise ConfigurationError("MiniMax H3 t2va needs no frames; fl2va needs one or two keyframes.")
        image_data = encode_images(images)
        if mode == "edit" and not image_data:
            raise ConfigurationError("Edit mode requires at least one reference image.")
        payload = {"model": model, "mode": mode, "prompt": prompt, "resolution": resolution, "fingerprint": fingerprint}
        if image_data is not None:
            payload["ref_images_safetensors_b64"] = image_data
        blob = self._request("POST", "/encode", payload)
        return decode_conditioning(blob, model=model, mode=mode, fingerprint=fingerprint, reference_count=len(images))

    async def encode(self, prompt, **kwargs):
        return await self._async_call(self.encode_sync, prompt, **kwargs)
