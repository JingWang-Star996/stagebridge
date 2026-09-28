import base64
import io
import json
import math
import struct
import warnings
import numpy as np
from PIL import Image
import torch
import safetensors.torch as st

PROTOCOL = "remote-te/1"
MAX_IMAGE_PIXELS = 16_777_216
MAX_IMAGE_BYTES = 96 * 1024 * 1024


def _base64(value):
    if len(value) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
        raise ValueError("Reference payload exceeds limit")
    try:
        data = base64.b64decode(value, validate=True)
    except ValueError as e:
        raise ValueError("Invalid base64 reference") from e
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Reference payload exceeds limit")
    return data


def decode_images(req):
    if req.ref_images_safetensors_b64 and req.ref_image_b64:
        raise ValueError("Choose one reference image transport")
    images = []
    if req.ref_images_safetensors_b64:
        tensors = st.load(_base64(req.ref_images_safetensors_b64))
        keys = [f"image_{i}" for i in range(len(tensors))]
        if not 1 <= len(keys) <= 16 or set(keys) != set(tensors):
            raise ValueError("Expected consecutive image_0 through image_N, at most 16")
        images = [tensors[k] for k in keys]
    elif req.ref_image_b64:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(_base64(req.ref_image_b64))) as img:
                if img.format not in ("PNG", "JPEG"):
                    raise ValueError("Only PNG/JPEG images are accepted")
                if req.ref_image_format and req.ref_image_format.upper() not in (img.format, "JPG" if img.format == "JPEG" else img.format):
                    raise ValueError("Image format does not match payload")
                if img.width * img.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Reference image is too large")
                mode = "RGBA" if "A" in img.getbands() else "RGB"
                images = [torch.from_numpy(np.array(img.convert(mode), dtype=np.float32) / 255.0).unsqueeze(0)]
    total_pixels = 0
    for image in images:
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4) or image.dtype != torch.float32:
            raise ValueError("References must be float32 [1,H,W,3|4]")
        pixels = image.shape[1] * image.shape[2]
        total_pixels += pixels
        if pixels < 1 or total_pixels > MAX_IMAGE_PIXELS:
            raise ValueError("Invalid reference dimensions or total pixel limit")
        if not torch.isfinite(image).all():
            raise ValueError("Reference pixels must be finite")
    if req.mode == "edit" and not images:
        raise ValueError("edit requires at least one reference image")
    if req.model == "minimax_h3":
        if req.mode == "t2va" and images:
            raise ValueError("H3 t2va does not accept reference frames")
        if req.mode == "fl2va" and not 1 <= len(images) <= 2:
            raise ValueError("H3 fl2va requires one or two keyframes")
    else:
        resized_sizes(images, req.resolution)
    return images


def resized_sizes(images, resolution):
    """Validate the entire expanded request before allocating any resized image."""
    sizes = []
    total_pixels = 0
    total_bytes = 0
    for image in images:
        source_h, source_w = image.shape[1:3]
        if resolution > 0:
            ratio = source_w / source_h
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width, height = round(source_w / 32) * 32, round(source_h / 32) * 32
        width, height = max(32, width), max(32, height)
        total_pixels += width * height
        total_bytes += width * height * image.shape[-1] * 4
        if total_pixels > MAX_IMAGE_PIXELS or total_bytes > MAX_IMAGE_BYTES:
            raise ValueError("Combined resized references exceed the pixel or byte limit")
        sizes.append((width, height))
    return sizes


def prepare_vision(images, resolution):
    """Same resize and alpha composition as native TextEncodeQwenImage21."""
    sizes = resized_sizes(images, resolution)
    import comfy.utils
    result = []
    for image, (width, height) in zip(images, sizes):
        samples = image[:1].movedim(-1, 1)
        s = image[:1] if (width, height) == (samples.shape[3], samples.shape[2]) else comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
        rgb = s[:, :, :, :3]
        if s.shape[-1] > 3:
            rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
        result.append(rgb)
    return result


def pack(conditioning, metadata):
    if len(conditioning) != 1:
        raise ValueError("Scheduled/hooked CLIP conditioning is not supported by this profile")
    hidden, extra = conditioning[0]
    h3 = metadata.get("model") == "minimax_h3"
    expected_width = 5120 if h3 else 4096
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != expected_width or hidden.dtype != torch.float32:
        raise ValueError("Unexpected native TE tensor contract")
    if not torch.isfinite(hidden).all():
        raise ValueError("Non-finite encoder output")
    tensors = {"hidden_states": hidden.detach().cpu().contiguous()}
    extras = {}
    for key, value in extra.items():
        if h3 and key == "minimax_token_tags":
            if not torch.is_tensor(value) or list(value.shape) != [hidden.shape[1]] or value.dtype != torch.int64:
                raise ValueError("Unexpected MiniMax H3 token tags")
            if not bool(((value == 0) | (value == 1)).all()):
                raise ValueError("MiniMax H3 token tags must be 0 or 1")
            tensors[key] = value.detach().cpu().contiguous()
        elif not h3 and key == "attention_mask":
            if not torch.is_tensor(value) or list(value.shape) != list(hidden.shape[:2]):
                raise ValueError("Unexpected attention mask")
            tensors[key] = value.detach().cpu().contiguous()
        elif key == "pooled_output" and value is None:
            extras[key] = None
        elif not h3 and key == "image_slots" and isinstance(value, list) and all(type(v) is int and 0 <= v <= hidden.shape[1] for v in value):
            extras[key] = value
        else:
            raise ValueError(f"Unsupported native conditioning extra: {key}")
    if h3 and "minimax_token_tags" not in tensors:
        raise ValueError("MiniMax H3 conditioning is missing token tags")
    meta = {**metadata, "protocol": PROTOCOL, "seq_len": hidden.shape[1],
            "has_mask": "attention_mask" in tensors, "has_slots": "image_slots" in extras,
            "extras": json.dumps(extras, separators=(",", ":"))}
    return st.save(tensors, metadata={k: str(v) for k, v in meta.items()})
