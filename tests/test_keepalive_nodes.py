"""Offline node lifetime tests: mocked transport and native paths, no GPU or HTTP."""
import asyncio
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


CANDIDATE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE))
from comfyui_nodes import remote_te_node as nodes
from comfyui_nodes.errors import ProtocolError, RetryableRemoteError, VersionMismatch

if Path(nodes.__file__).resolve().parent != CANDIDATE / "comfyui_nodes":
    raise RuntimeError("Node lifetime tests must import the candidate package")

URL = "http://127.0.0.1:18765"
FP = "a" * 64
NODE_CASES = (
    (nodes.TextEncodeQwenImage21Remote, ("positive",), 1),
    (nodes.QwenImage21Remote, ("positive", "negative"), 2),
)
REMOTE_RESULT = ("remote-condition", {"encode_ms": "1.5"})


class NodeClientLifetimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        nodes._RETRY_KEYS.clear()
        self.addCleanup(nodes._RETRY_KEYS.clear)
        self.latent = {"samples": "mocked-latent"}
        self.attach = self.enterContext(patch.object(
            nodes, "attach_references",
            side_effect=lambda conditions, images, vae, resolution: (conditions, self.latent),
        ))
        self.loader = self.enterContext(patch.object(nodes, "load_fallback_clip"))

    def client(self):
        return SimpleNamespace(
            url=URL,
            health=AsyncMock(return_value={"fingerprint": FP, "loaded": True}),
            encode=AsyncMock(return_value=REMOTE_RESULT),
            aclose=AsyncMock(),
            close=Mock(),
        )

    async def cancel_while_blocked(self, invoke, method, preceding_successes=0):
        entered = asyncio.Event()
        calls = 0

        async def block(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls <= preceding_successes:
                return REMOTE_RESULT
            entered.set()
            await asyncio.Future()

        method.side_effect = block
        task = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_nodes_close_after_success(self):
        for node_type, args, count in NODE_CASES:
            with self.subTest(node=node_type.__name__):
                client = self.client()
                node = node_type()
                with patch.object(nodes, "RemoteClient", return_value=client) as factory, \
                        patch.object(node, "_local") as local:
                    result = await node.encode(*args, server_url=URL)
                factory.assert_called_once()
                client.health.assert_awaited_once()
                self.assertEqual(client.encode.await_count, count)
                client.aclose.assert_awaited_once_with()
                local.assert_not_called()
                self.assertEqual(result["result"][:count], ("remote-condition",) * count)
                if count == 2:
                    self.assertEqual(result["result"][2], self.latent)
        self.loader.assert_not_called()

    async def test_nodes_await_close_before_returning_success(self):
        for node_type, args, _ in NODE_CASES:
            with self.subTest(node=node_type.__name__):
                client = self.client()
                close_started, release_close = asyncio.Event(), asyncio.Event()

                async def delayed_close():
                    close_started.set()
                    await release_close.wait()

                client.aclose.side_effect = delayed_close
                with patch.object(nodes, "RemoteClient", return_value=client):
                    task = asyncio.create_task(node_type().encode(*args, server_url=URL))
                    try:
                        await asyncio.wait_for(close_started.wait(), timeout=2)
                        self.assertFalse(task.done(), "Node returned before aclose completed")
                        release_close.set()
                        await asyncio.wait_for(task, timeout=2)
                    finally:
                        release_close.set()
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                client.aclose.assert_awaited_once_with()

    async def test_nodes_close_after_health_error_without_fallback(self):
        for node_type, args, _ in NODE_CASES:
            for failure in (RetryableRemoteError("network"), ProtocolError("bad health")):
                with self.subTest(node=node_type.__name__, error=type(failure).__name__):
                    client = self.client()
                    client.health.side_effect = failure
                    node = node_type()
                    with patch.object(nodes, "RemoteClient", return_value=client), \
                            patch.object(node, "_local") as local:
                        with self.assertRaises(type(failure)):
                            await node.encode(*args, server_url=URL, fallback_local=False)
                    client.encode.assert_not_awaited()
                    client.aclose.assert_awaited_once_with()
                    local.assert_not_called()

    async def test_nodes_close_after_each_encode_error_without_fallback(self):
        for node_type, args, count in NODE_CASES:
            for failed_call in range(count):
                for failure in (RetryableRemoteError("network"), ProtocolError("bad metadata")):
                    with self.subTest(node=node_type.__name__, call=failed_call, error=type(failure).__name__):
                        client = self.client()
                        client.encode.side_effect = [REMOTE_RESULT] * failed_call + [failure]
                        node = node_type()
                        with patch.object(nodes, "RemoteClient", return_value=client), \
                                patch.object(node, "_local") as local:
                            with self.assertRaises(type(failure)):
                                await node.encode(*args, server_url=URL, fallback_local=False)
                        self.assertEqual(client.encode.await_count, failed_call + 1)
                        client.aclose.assert_awaited_once_with()
                        local.assert_not_called()

    async def test_nodes_close_on_fallback_return_and_fallback_failure(self):
        for node_type, args, count in NODE_CASES:
            for failed_stage in ("health", "encode"):
                for local_fails in (False, True):
                    with self.subTest(node=node_type.__name__, stage=failed_stage, local_fails=local_fails):
                        client = self.client()
                        if failed_stage == "health":
                            client.health.side_effect = RetryableRemoteError("network")
                        else:
                            client.encode.side_effect = [REMOTE_RESULT] * (count - 1) + [RetryableRemoteError("HTTP 503")]
                        node = node_type()
                        with patch.object(nodes, "RemoteClient", return_value=client), \
                                patch.object(node, "_local", return_value=("native+", "native-", self.latent)) as local:
                            if local_fails:
                                local.side_effect = RuntimeError("native failure")
                                with self.assertRaisesRegex(RuntimeError, "native failure"):
                                    await node.encode(*args, server_url=URL, fallback_local=True)
                            else:
                                result = await node.encode(*args, server_url=URL, fallback_local=True)
                                self.assertEqual(result["result"][:count], ("native+", "native-")[:count])
                                self.assertIn("REMOTE FALLBACK", result["ui"]["text"][0])
                        local.assert_called_once()
                        client.aclose.assert_awaited_once_with()

    async def test_nodes_close_on_409_without_fallback_or_further_encode(self):
        for node_type, args, count in NODE_CASES:
            for failed_call in range(count):
                with self.subTest(node=node_type.__name__, call=failed_call):
                    client = self.client()
                    client.encode.side_effect = [REMOTE_RESULT] * failed_call + [VersionMismatch("HTTP 409")]
                    node = node_type()
                    with patch.object(nodes, "RemoteClient", return_value=client), \
                            patch.object(node, "_local") as local:
                        with self.assertRaisesRegex(VersionMismatch, "409"):
                            await node.encode(*args, server_url=URL, fallback_local=True)
                    self.assertEqual(client.encode.await_count, failed_call + 1)
                    client.aclose.assert_awaited_once_with()
                    local.assert_not_called()
        self.loader.assert_not_called()

    async def test_nodes_close_on_health_cancellation_without_fallback(self):
        for node_type, args, _ in NODE_CASES:
            with self.subTest(node=node_type.__name__):
                client = self.client()
                node = node_type()
                with patch.object(nodes, "RemoteClient", return_value=client), \
                        patch.object(node, "_local") as local:
                    await self.cancel_while_blocked(
                        lambda: node.encode(*args, server_url=URL, fallback_local=True), client.health,
                    )
                client.encode.assert_not_awaited()
                client.aclose.assert_awaited_once_with()
                local.assert_not_called()

    async def test_nodes_close_on_each_encode_cancellation_without_fallback(self):
        for node_type, args, count in NODE_CASES:
            for canceled_call in range(count):
                with self.subTest(node=node_type.__name__, call=canceled_call):
                    client = self.client()
                    node = node_type()
                    with patch.object(nodes, "RemoteClient", return_value=client), \
                            patch.object(node, "_local") as local:
                        await self.cancel_while_blocked(
                            lambda: node.encode(*args, server_url=URL, fallback_local=True),
                            client.encode, preceding_successes=canceled_call,
                        )
                    self.assertEqual(client.encode.await_count, canceled_call + 1)
                    client.aclose.assert_awaited_once_with()
                    local.assert_not_called()

    async def test_is_changed_closes_on_success_and_retry_key_return(self):
        for node_type, _, _ in NODE_CASES:
            for pending_retry in (False, True):
                with self.subTest(node=node_type.__name__, pending_retry=pending_retry):
                    nodes._RETRY_KEYS.clear()
                    client = self.client()
                    if pending_retry:
                        nodes._RETRY_KEYS.add(node_type._retry_key("node", URL, "qwen3vl_8b"))
                    with patch.object(nodes, "RemoteClient", return_value=client):
                        result = await node_type.IS_CHANGED(server_url=URL, unique_id="node")
                    if pending_retry:
                        self.assertTrue(math.isnan(result))
                    else:
                        self.assertEqual(result, (FP, True))
                    client.aclose.assert_awaited_once_with()

    async def test_is_changed_closes_on_retryable_and_nonretryable_errors(self):
        for node_type, _, _ in NODE_CASES:
            for failure in (RetryableRemoteError("network"), ProtocolError("bad health"), VersionMismatch("HTTP 409")):
                with self.subTest(node=node_type.__name__, error=type(failure).__name__):
                    client = self.client()
                    client.health.side_effect = failure
                    with patch.object(nodes, "RemoteClient", return_value=client):
                        if isinstance(failure, RetryableRemoteError):
                            self.assertTrue(math.isnan(await node_type.IS_CHANGED(server_url=URL)))
                        else:
                            with self.assertRaises(type(failure)):
                                await node_type.IS_CHANGED(server_url=URL)
                    client.aclose.assert_awaited_once_with()

    async def test_is_changed_closes_on_cancellation(self):
        for node_type, _, _ in NODE_CASES:
            with self.subTest(node=node_type.__name__):
                client = self.client()
                with patch.object(nodes, "RemoteClient", return_value=client):
                    await self.cancel_while_blocked(
                        lambda: node_type.IS_CHANGED(server_url=URL), client.health,
                    )
                client.aclose.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
