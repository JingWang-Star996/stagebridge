"""Legacy ComfyUI nodes with async transport, version pinning and lazy fallback."""
from .client import DEFAULT_SERVER_URL, RemoteClient, fingerprint_value, server_url
from .codec import MODEL, checked_images, validate_request
from .errors import RetryableRemoteError
from .native import attach_references, clip_names, load_fallback_clip, native_encode

# IS_CHANGED is invoked on the class, not its cached instance. Only retry state
# belongs here; version pins and loaded CLIP objects stay on each node instance.
_RETRY_KEYS = set()


def common_inputs():
    return {
        "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
        "model": ([MODEL],),
        "resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32}),
        "fallback_local": ("BOOLEAN", {"default": True}),
        "clip_name": (clip_names(),),
        "expected_fingerprint": ("STRING", {"default": "", "tooltip": "Empty pins the first server version for this node instance. Enter a new fingerprint explicitly to approve an upgrade."}),
        "timeout_s": ("INT", {"default": 30, "min": 1, "max": 900}),
        "fallback_device": (["default", "cpu"],),
    }


class RemoteNodeBase:
    CATEGORY = "conditioning/remote Qwen Image 2.1"
    FUNCTION = "encode"

    def __init__(self):
        self._pins = {}
        self._clip = None
        self._clip_key = None

    @classmethod
    def _retry_key(cls, unique_id, url, model):
        return cls.__name__, str(unique_id), server_url(url), model

    @classmethod
    async def IS_CHANGED(cls, server_url=DEFAULT_SERVER_URL, model=MODEL, unique_id="", timeout=None, timeout_s=30, **kwargs):
        key = cls._retry_key(unique_id, server_url, model)
        client = RemoteClient(server_url, timeout if timeout is not None else timeout_s)
        try:
            health = await client.health(model)
        except RetryableRemoteError:
            return float("nan")
        finally:
            await client.aclose()
        # Keep retrying after encode failures even when /health remains available.
        if key in _RETRY_KEYS:
            return float("nan")
        return (health["fingerprint"], health["loaded"])

    async def _fingerprint(self, client, model, expected_fingerprint):
        key = client.url, model
        if expected_fingerprint:
            self._pins[key] = fingerprint_value(expected_fingerprint)
        if key not in self._pins:
            self._pins[key] = (await client.health(model))["fingerprint"]
        return self._pins[key]

    def _local(self, clip_name, fallback_device, prompt, negative_prompt, vae, resolution, images):
        key = clip_name, fallback_device
        if self._clip is None or self._clip_key != key:
            self._clip = load_fallback_clip(clip_name, fallback_device)
            self._clip_key = key
        return native_encode(self._clip, prompt, negative_prompt, vae, resolution, images)

    async def _run(self, prompts, *, server_url, model, resolution, fallback, clip_name,
                   expected_fingerprint, timeout, fallback_device, unique_id, images=(), vae=None):
        mode = "edit" if images and vae is not None else "t2i"
        images = checked_images(images)
        for prompt in prompts:
            validate_request(prompt, model, mode, resolution)
        client = RemoteClient(server_url, timeout)
        key = self._retry_key(unique_id, client.url, model)
        fingerprint = None
        try:
            fingerprint = await self._fingerprint(client, model, expected_fingerprint)
            conditions, metadata = [], []
            for prompt in prompts:
                condition, meta = await client.encode(prompt, model=model, mode=mode,
                    resolution=resolution, fingerprint=fingerprint, images=images)
                conditions.append(condition)
                metadata.append(meta)
        except RetryableRemoteError as exc:
            _RETRY_KEYS.add(key)
            if not fallback:
                raise
            positive, negative, latent = self._local(clip_name, fallback_device, prompts[0],
                prompts[1] if len(prompts) > 1 else "", vae, resolution, images)
            local_conditions = [positive, negative][:len(prompts)]
            return local_conditions, latent, f"REMOTE FALLBACK | fingerprint={fingerprint or 'unavailable'} | local={clip_name} | {exc}"
        finally:
            await client.aclose()
        _RETRY_KEYS.discard(key)
        conditions, latent = attach_references(conditions, images, vae, resolution)
        duration = sum(float(meta["encode_ms"]) for meta in metadata)
        return conditions, latent, f"REMOTE | fingerprint={fingerprint} | encode_ms={duration:.2f}"


class TextEncodeQwenImage21Remote(RemoteNodeBase):
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True}), **common_inputs()},
                "hidden": {"unique_id": "UNIQUE_ID"}}

    async def encode(self, text, server_url=DEFAULT_SERVER_URL, model=MODEL, resolution=1024,
                     fallback=True, clip_name="", expected_fingerprint="", timeout=30,
                     fallback_device="default", unique_id="", fallback_local=None, timeout_s=None):
        fallback = fallback if fallback_local is None else fallback_local
        timeout = timeout if timeout_s is None else timeout_s
        conditions, _, status = await self._run([text], server_url=server_url, model=model,
            resolution=resolution, fallback=fallback, clip_name=clip_name, expected_fingerprint=expected_fingerprint,
            timeout=timeout, fallback_device=fallback_device, unique_id=unique_id)
        return {"ui": {"text": [status]}, "result": (conditions[0],)}


class QwenImage21Remote(RemoteNodeBase):
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"vae": ("VAE",)}
        optional.update({f"image_{i}": ("IMAGE",) for i in range(1, 17)})
        return {"required": {"prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                             "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}), **common_inputs()},
                "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    async def encode(self, prompt, negative_prompt, server_url=DEFAULT_SERVER_URL, model=MODEL,
                     resolution=1024, fallback=True, clip_name="", expected_fingerprint="", timeout=30,
                     fallback_device="default", unique_id="", vae=None, fallback_local=None, timeout_s=None, **kwargs):
        fallback = fallback if fallback_local is None else fallback_local
        timeout = timeout if timeout_s is None else timeout_s
        from .errors import ConfigurationError
        unknown = set(kwargs) - {f"image_{i}" for i in range(1, 17)}
        if unknown:
            raise ConfigurationError("Unknown image or conditioning input cannot be silently discarded.")
        images = [kwargs[f"image_{i}"] for i in range(1, 17) if kwargs.get(f"image_{i}") is not None]
        conditions, latent, status = await self._run([prompt, negative_prompt], server_url=server_url,
            model=model, resolution=resolution, fallback=fallback, clip_name=clip_name,
            expected_fingerprint=expected_fingerprint, timeout=timeout, fallback_device=fallback_device,
            unique_id=unique_id, images=images, vae=vae)
        return {"ui": {"text": [status]}, "result": (conditions[0], conditions[1], latent)}
