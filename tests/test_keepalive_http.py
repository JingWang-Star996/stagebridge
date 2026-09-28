"""Real loopback HTTP tests; no external host, model, or CUDA work."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import ssl
import threading
import time
import unittest
from unittest.mock import patch

import comfyui_nodes.client as client_module
from comfyui_nodes.client import RemoteClient
from comfyui_nodes.errors import ProtocolError, RetryableRemoteError, VersionMismatch
from comfyui_nodes.remote_te_node import QwenImage21Remote, TextEncodeQwenImage21Remote
from test_client import FP, response


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, behavior):
        super().__init__(("127.0.0.1", 0), Handler)
        self.behavior = behavior
        self.records = []
        self.finished_ports = set()
        self.entered = threading.Event()
        self.peer_closed = threading.Event()
        self.release = threading.Event()
        self.binary = response()

    def handle_error(self, request, address):
        pass  # Deliberate socket shutdowns are asserted through receipts below.


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def finish(self):
        try:
            super().finish()
        finally:
            self.server.finished_ports.add(self.client_address[1])

    def do_GET(self):
        self.handle_call()

    def do_POST(self):
        self.handle_call()

    def handle_call(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.records.append({"method": self.command, "path": self.path, "body": body,
                                    "port": self.client_address[1], "auth": self.headers.get("Authorization")})
        behavior = self.server.behavior
        if behavior == "blocked":
            self.send_response(200)
            self.send_header("Content-Length", "16")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            self.server.entered.set()
            self.connection.settimeout(4)
            if self.rfile.read(1) == b"":
                self.server.peer_closed.set()
            self.close_connection = True
            return
        if behavior == "drop" and self.command == "POST":
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        if behavior == "409" and self.command == "POST":
            self.reply(b"fingerprint rejected", status=409)
            return
        if behavior == "redirect":
            self.reply(b"", status=302, extra={"Location": "/never-follow"})
            return
        if behavior == "long-header":
            self.reply(b"x" * 1025)
            return
        if behavior in ("duplicate-length", "conflicting-framing"):
            self.send_response(200)
            self.send_header("Content-Length", "4")
            if behavior == "duplicate-length":
                self.send_header("Content-Length", "400")
            else:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"4\r\nabcd\r\n0\r\n\r\n" if behavior == "conflicting-framing" else b"abcd")
            self.wfile.flush()
            return
        if behavior == "long-unframed":
            self.send_response(200)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"x" * 1025)
            self.wfile.flush()
            self.close_connection = True
            return
        if behavior == "truncated":
            self.send_response(200)
            self.send_header("Content-Length", "128")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"abcd")
            self.wfile.flush()
            self.close_connection = True
            return
        if behavior == "bad-health":
            self.reply(b'{"profiles": {}}')
            return
        payload = json.dumps({"profiles": {"qwen3vl_8b": {"loaded": True, "fingerprint": FP}}}).encode() if self.command == "GET" else self.server.binary
        self.reply(payload)
        if behavior == "stale":
            # Deliberately omit Connection: close, emulating idle peer shutdown.
            self.close_connection = True

    def reply(self, body, status=200, extra=None):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


@contextmanager
def server(behavior="normal"):
    instance = Server(behavior)
    thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield instance, "http://127.0.0.1:" + str(instance.server_port)
    finally:
        instance.release.set()
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def wait_for(predicate, seconds=1):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class LoopbackTests(unittest.TestCase):
    def test_four_requests_share_socket_proxy_ignored_and_close_reaches_peer(self):
        with server() as (remote, url), patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1", "REMOTE_TE_API_TOKEN": "loopback-secret"}):
            client = RemoteClient(url)
            try:
                client.health_sync()
                client.encode_sync("positive", fingerprint=FP)
                client.encode_sync("negative", fingerprint=FP)
                client.health_sync()
                self.assertEqual(len(remote.records), 4)
                self.assertEqual(len({r["port"] for r in remote.records}), 1)
                self.assertTrue(all(r["auth"] == "Bearer loopback-secret" for r in remote.records))
            finally:
                client.close()
            self.assertTrue(wait_for(lambda: remote.records[0]["port"] in remote.finished_ports))

    def test_409_discards_connection_without_reposting_then_explicit_call_can_reconnect(self):
        with server("409") as (remote, url):
            client = RemoteClient(url)
            try:
                with self.assertRaises(VersionMismatch):
                    client.encode_sync("one submission", fingerprint=FP)
                self.assertIsNone(client.opener._connection)
                client.health_sync()
                self.assertEqual([r["method"] for r in remote.records], ["POST", "GET"])
                self.assertNotEqual(remote.records[0]["port"], remote.records[1]["port"])
            finally:
                client.close()

    def test_oversized_declared_and_unframed_body_close_connection(self):
        for behavior in ("long-header", "long-unframed"):
            with self.subTest(behavior=behavior), server(behavior) as (remote, url), patch.object(client_module, "MAX_RESPONSE_BYTES", 1024):
                client = RemoteClient(url)
                try:
                    with self.assertRaises(ProtocolError):
                        client.encode_sync("large", fingerprint=FP)
                    self.assertIsNone(client.opener._connection)
                    self.assertEqual(len(remote.records), 1)
                finally:
                    client.close()

    def test_truncated_body_is_network_failure_and_not_retried(self):
        with server("truncated") as (remote, url):
            client = RemoteClient(url)
            try:
                with self.assertRaises(RetryableRemoteError):
                    client.encode_sync("truncated", fingerprint=FP)
                self.assertIsNone(client.opener._connection)
                self.assertEqual(len(remote.records), 1)
            finally:
                client.close()

    def test_ambiguous_body_framing_is_not_reused(self):
        for behavior in ("duplicate-length", "conflicting-framing"):
            with self.subTest(behavior=behavior), server(behavior) as (remote, url):
                client = RemoteClient(url)
                try:
                    with self.assertRaises(ProtocolError):
                        client.encode_sync("ambiguous", fingerprint=FP)
                    self.assertIsNone(client.opener._connection)
                    self.assertEqual(len(remote.records), 1)
                finally:
                    client.close()

    def test_received_post_then_disconnect_never_replayed(self):
        with server("drop") as (remote, url):
            client = RemoteClient(url)
            try:
                with self.assertRaises(RetryableRemoteError):
                    client.encode_sync("only once", fingerprint=FP)
                self.assertEqual(len(remote.records), 1)
                self.assertEqual(json.loads(remote.records[0]["body"])["prompt"], "only once")
            finally:
                client.close()

    def test_stale_keepalive_is_failure_without_implicit_reconnect(self):
        with server("stale") as (remote, url):
            client = RemoteClient(url)
            try:
                client.health_sync()
                self.assertTrue(wait_for(lambda: len(remote.finished_ports) == 1))
                with self.assertRaises(RetryableRemoteError):
                    client.encode_sync("must not reconnect", fingerprint=FP)
                self.assertEqual(len(remote.records), 1)
            finally:
                client.close()

    def test_redirect_is_rejected_without_followup(self):
        with server("redirect") as (remote, url):
            client = RemoteClient(url)
            try:
                with self.assertRaises(ProtocolError):
                    client.health_sync()
                self.assertEqual(len(remote.records), 1)
                self.assertIsNone(client.opener._connection)
            finally:
                client.close()

    def test_protocol_decode_error_also_discards_socket(self):
        with server("bad-health") as (remote, url):
            client = RemoteClient(url)
            try:
                with self.assertRaises(ProtocolError):
                    client.health_sync()
                self.assertIsNone(client.opener._connection)
                self.assertTrue(wait_for(lambda: len(remote.finished_ports) == 1))
            finally:
                client.close()

    def test_parallel_calls_are_serialized_on_one_connection(self):
        with server() as (remote, url):
            client = RemoteClient(url)
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda _: client.health_sync(), range(12)))
                self.assertEqual(len(results), 12)
                self.assertEqual(len({r["port"] for r in remote.records}), 1)
            finally:
                client.close()

    def test_https_default_context_keeps_certificate_verification(self):
        client = RemoteClient("https://127.0.0.1:1")
        with patch.object(client_module.http.client, "HTTPSConnection") as factory:
            factory.return_value.sock = None
            factory.return_value.connect.side_effect = OSError("offline fixture")
            with self.assertRaises(RetryableRemoteError):
                client.health_sync()
            context = factory.call_args.kwargs["context"]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
        client.close()


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def wait_event(self, event):
        async def poll():
            while not event.is_set():
                await asyncio.sleep(0.005)
        await asyncio.wait_for(poll(), 1)

    async def test_two_real_nodes_reuse_then_close_their_own_connections(self):
        with server() as (remote, url), patch("comfyui_nodes.remote_te_node.attach_references", side_effect=lambda c, *args: (c, None)):
            for node, prompts, expected_calls in ((TextEncodeQwenImage21Remote(), ("one",), 2),
                                                   (QwenImage21Remote(), ("positive", "negative"), 3)):
                before = len(remote.records)
                await node.encode(*prompts, server_url=url, fallback_local=False)
                calls = remote.records[before:]
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(len({r["port"] for r in calls}), 1)
                deadline = asyncio.get_running_loop().time() + 1
                while calls[0]["port"] not in remote.finished_ports:
                    self.assertLess(asyncio.get_running_loop().time(), deadline)
                    await asyncio.sleep(0.005)
            self.assertEqual(len({r["port"] for r in remote.records}), 2)

    async def test_cancel_closes_active_response_with_saturated_single_worker(self):
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        with server("blocked") as (remote, url):
            client = RemoteClient(url, timeout=5)
            task = asyncio.create_task(client.health())
            await self.wait_event(remote.entered)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            await self.wait_event(remote.peer_closed)
            self.assertIsNone(client.opener._connection)
            self.assertEqual(len(remote.records), 1)
            with self.assertRaises(RetryableRemoteError):
                await client.health()

    async def test_cancel_waiting_call_cannot_submit_late_post(self):
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        with server("blocked") as (remote, url):
            client = RemoteClient(url, timeout=5)
            first = asyncio.create_task(client.health())
            await self.wait_event(remote.entered)
            queued = asyncio.create_task(client.encode("must never reach server", fingerprint=FP))
            await asyncio.sleep(0.03)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(queued, 1)
            with self.assertRaises(RetryableRemoteError):
                await asyncio.wait_for(first, 1)
            await self.wait_event(remote.peer_closed)
            await asyncio.sleep(0.03)
            self.assertEqual([r["method"] for r in remote.records], ["GET"])

    async def test_async_close_does_not_block_loop_and_survives_repeated_cancel(self):
        client = RemoteClient("http://127.0.0.1:1")
        entered = threading.Event()
        release = threading.Event()
        original = client.close
        def slow_close():
            entered.set()
            release.wait(1)
            original()
        with patch.object(client, "close", side_effect=slow_close):
            task = asyncio.create_task(client.aclose())
            await self.wait_event(entered)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        self.assertTrue(client.opener._closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
