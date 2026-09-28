"""Native ComfyUI integration. Imports that can initialize models are lazy."""
import math

import torch

from .codec import checked_images
from .errors import ConfigurationError

NO_CLIP = "(no text encoder installed)"


def clip_names():
    import folder_paths
    names = list(folder_paths.get_filename_list("text_encoders"))
    def rank(name):
        normalized = name.lower().replace("-", "_")
        if "qwen3vl_8b_int8_convrot" in normalized:
            return 0, name.lower()
        if "qwen3vl_8b" in normalized and "w4a8" in normalized:
            return 1, name.lower()
        if "qwen3vl_8b" in normalized or "qwen3_vl_8b" in normalized:
            return 2, name.lower()
        return 3, name.lower()
    return sorted(names, key=rank) or [NO_CLIP]


def load_fallback_clip(name, device="default"):
    import folder_paths
    if name == NO_CLIP or name not in folder_paths.get_filename_list("text_encoders"):
        raise ConfigurationError("Local fallback requested, but the selected text encoder is not installed. Select an installed Qwen3-VL-8B encoder in clip_name.")
    import comfy.sd
    from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
    options = {}
    if device == "cpu":
        options = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}
    clip = comfy.sd.load_clip(
        ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", name)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE, model_options=options,
    )
    if not isinstance(clip.tokenizer, QwenImage21Tokenizer):
        raise ConfigurationError("Selected fallback file did not load the native Qwen Image 2.1 tokenizer. Select Qwen3-VL-8B weights.")
    return clip


def resized_references(images, resolution):
    import comfy.utils
    result = []
    for image in checked_images(images):
        samples = image.movedim(-1, 1)
        if resolution > 0:
            ratio = samples.shape[3] / samples.shape[2]
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
        width, height = max(32, width), max(32, height)
        if width * height > 16_777_216 or max(width, height) > 16384:
            raise ConfigurationError("Resized reference image exceeds the local safety limit.")
        if (width, height) == (samples.shape[3], samples.shape[2]):
            resized = image
        else:
            resized = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
        result.append(resized)
    return result


def attach_references(conditionings, images, vae, resolution):
    import comfy.model_management
    import node_helpers
    resized = resized_references(images, resolution)
    latent_w = latent_h = resolution or 1024
    if resized:
        latent_h, latent_w = resized[0].shape[1:3]
    with torch.inference_mode():
        ref_latents = [vae.encode(image) for image in resized] if vae is not None else []
        if ref_latents:
            conditionings = [node_helpers.conditioning_set_values(cond, {"reference_latents": ref_latents}, append=True)
                             for cond in conditionings]
        latent = torch.zeros([1, 64, latent_h // 16, latent_w // 16],
                             device=comfy.model_management.intermediate_device())
    return conditionings, {"samples": latent}


def native_encode(clip, prompt, negative_prompt, vae, resolution, images):
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21
    with torch.inference_mode():
        result = TextEncodeQwenImage21.execute(
            clip=clip, prompt=prompt, negative_prompt=negative_prompt, vae=vae,
            resolution=resolution, images={f"image_{i + 1}": image for i, image in enumerate(checked_images(images))},
        )
    return result.result
