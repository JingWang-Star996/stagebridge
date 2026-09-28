"""MiniMax H3 first/last-frame node with remote text/vision encoding."""

import os

from .client import RemoteClient
from .codec import H3_MODEL, checked_images, validate_request
from .errors import ConfigurationError, RetryableRemoteError
from .remote_te_node import RemoteNodeBase, _RETRY_KEYS

H3_SERVER_URL = os.environ.get("REMOTE_TE_H3_SERVER_URL", os.environ.get("REMOTE_TE_SERVER_URL", "http://127.0.0.1:8765"))


class MiniMaxH3ImageToVideoRemote(RemoteNodeBase):
    CATEGORY = "model/conditioning/minimax"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")

    @classmethod
    async def IS_CHANGED(cls, server_url=H3_SERVER_URL, model=H3_MODEL, **kwargs):
        return await super().IS_CHANGED(server_url=server_url, model=model, **kwargs)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "server_url": ("STRING", {"default": H3_SERVER_URL}),
                "model": ([H3_MODEL],),
                "expected_fingerprint": ("STRING", {"default": ""}),
                "timeout_s": ("INT", {"default": 900, "min": 1, "max": 1800}),
            },
            "optional": {"first_frame": ("IMAGE",), "last_frame": ("IMAGE",)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    async def encode(self, vae, prompt, width, height, length, server_url=H3_SERVER_URL,
                     model=H3_MODEL, expected_fingerprint="", timeout_s=900,
                     first_frame=None, last_frame=None, unique_id=""):
        if type(width) is not int or type(height) is not int or min(width, height) < 32 or \
                max(width, height) > 8192 or width % 32 or height % 32 or \
                type(length) is not int or not 5 <= length <= 3600:
            raise ConfigurationError("MiniMax H3 canvas and length must match native node bounds.")
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize

        latent, frame_count = _empty_av_latent(width, height, length)
        images, keyframes = [], []
        if first_frame is not None:
            image = _resize(first_frame[:1], width, height, "disabled")
            images.append(image)
            keyframes.append({"resolved_frame_index": 0, "image": image})
        if last_frame is not None:
            image = _resize(last_frame[:1], width, height, "center")
            images.append(image)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": image})
        mode = "fl2va" if images else "t2va"
        images = checked_images(images)
        validate_request(prompt, model, mode, 0)

        client = RemoteClient(server_url, timeout_s)
        key = self._retry_key(unique_id, client.url, model)
        try:
            fingerprint = await self._fingerprint(client, model, expected_fingerprint)
            conditioning, meta = await client.encode(
                prompt, model=model, mode=mode, resolution=0,
                fingerprint=fingerprint, images=images)
        except RetryableRemoteError:
            _RETRY_KEYS.add(key)
            raise
        finally:
            await client.aclose()
        _RETRY_KEYS.discard(key)
        if keyframes:
            import node_helpers
            for keyframe in keyframes:
                keyframe["latent"] = vae.encode(keyframe.pop("image"))
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"minimax_keyframes": keyframes})
        status = f"REMOTE H3 | fingerprint={fingerprint} | encode_ms={meta['encode_ms']}"
        return {"ui": {"text": [status]}, "result": (conditioning, latent)}
