"""Single-reference edit convenience node; VAE execution stays in ComfyUI."""
from .client import DEFAULT_SERVER_URL
from .codec import MODEL
from .remote_te_node import RemoteNodeBase, common_inputs


class TextEncodeQwenImageEditRemote(RemoteNodeBase):
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                             "ref_image": ("IMAGE",), "vae": ("VAE",), **common_inputs()},
                "hidden": {"unique_id": "UNIQUE_ID"}}

    async def encode(self, text, ref_image, vae, server_url=DEFAULT_SERVER_URL, model=MODEL,
                     resolution=1024, fallback=True, clip_name="", expected_fingerprint="", timeout=30,
                     fallback_device="default", unique_id="", fallback_local=None, timeout_s=None):
        from .errors import ConfigurationError
        fallback = fallback if fallback_local is None else fallback_local
        timeout = timeout if timeout_s is None else timeout_s
        if vae is None:
            raise ConfigurationError("Edit node requires a local VAE.")
        conditions, latent, status = await self._run([text], server_url=server_url, model=model,
            resolution=resolution, fallback=fallback, clip_name=clip_name,
            expected_fingerprint=expected_fingerprint, timeout=timeout, fallback_device=fallback_device,
            unique_id=unique_id, images=[ref_image], vae=vae)
        return {"ui": {"text": [status]}, "result": (conditions[0], latent)}
