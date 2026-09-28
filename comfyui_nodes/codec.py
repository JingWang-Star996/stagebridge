"""Strict, pickle-free remote-te/1 transport. IMAGE values are never quantized."""
import base64
import json
import math
import struct

import torch
from safetensors.torch import load, save

from .errors import ConfigurationError, ProtocolError, VersionMismatch

PROTOCOL = "remote-te/1"
MODEL = "qwen3vl_8b"
H3_MODEL = "minimax_h3"
MAX_IMAGES = 16
MAX_IMAGE_EDGE = 8192
MAX_IMAGE_PIXELS = 16_777_216
MAX_IMAGE_BYTES = 96 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_SEQUENCE = 32768
MAX_PROMPT_BYTES = 256 * 1024
HIDDEN_WIDTH = 4096
METADATA_KEYS = {"protocol", "model", "fingerprint", "mode", "seq_len", "extras",
                 "encode_ms", "queue_wait_ms", "server_version"}
OPTIONAL_METADATA_KEYS = {"has_mask", "has_slots"}


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON key in remote response.")
        result[key] = value
    return result


def strict_json(data):
    try:
        return json.loads(data, object_pairs_hook=_no_duplicates,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("Invalid JSON in remote response.") from None


def validate_request(prompt, model, mode, resolution):
    if not ((model == MODEL and mode in ("t2i", "edit")) or
            (model == H3_MODEL and mode in ("t2va", "fl2va"))):
        raise ConfigurationError("Unsupported TE model and mode combination.")
    if not isinstance(prompt, str) or len(prompt) > 32768 or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ConfigurationError("Prompt must be UTF-8 text no longer than 32768 characters or 256 KiB.")
    if type(resolution) is not int or not 0 <= resolution <= 4096:
        raise ConfigurationError("Resolution must be an integer from 0 to 4096.")


def checked_images(images):
    """Return each input's first image, matching native Qwen Image 2.1 semantics."""
    if len(images) > MAX_IMAGES:
        raise ConfigurationError("At most 16 reference image inputs are supported.")
    result, total_bytes, total_pixels = [], 0, 0
    for image in images:
        if not isinstance(image, torch.Tensor) or image.dtype != torch.float32:
            raise ConfigurationError("Reference IMAGE tensors must be float32 NHWC.")
        if image.ndim != 4 or image.shape[0] < 1 or image.shape[-1] not in (3, 4):
            raise ConfigurationError("Reference IMAGE shape must be [N,H,W,3 or 4], N >= 1.")
        _, height, width, channels = image.shape
        if min(height, width) < 1 or max(height, width) > MAX_IMAGE_EDGE:
            raise ConfigurationError("Reference image dimensions must be between 1 and 8192.")
        total_pixels += height * width
        if total_pixels > MAX_IMAGE_PIXELS:
            raise ConfigurationError("Reference images exceed the total 16 megapixel limit.")
        total_bytes += height * width * channels * 4
        if total_bytes + MAX_HEADER_BYTES + 8 > MAX_IMAGE_BYTES:
            raise ConfigurationError("Selected reference images exceed the 96 MiB transport limit.")
        first = image[:1].detach()
        if not bool(torch.isfinite(first).all()):
            raise ConfigurationError("Reference IMAGE contains non-finite values.")
        result.append(first)
    return result


def encode_images(images):
    images = checked_images(images)
    if not images:
        return None
    # Preserve original float32 values, channels and dimensions. Server owns resize.
    tensors = {f"image_{i}": image.to(device="cpu").contiguous()
               for i, image in enumerate(images)}
    return base64.b64encode(save(tensors)).decode("ascii")


def _read_header(blob):
    if not isinstance(blob, bytes) or not 10 <= len(blob) <= MAX_RESPONSE_BYTES:
        raise ProtocolError("Invalid or oversized safetensors response.")
    size = struct.unpack("<Q", blob[:8])[0]
    if size < 2 or size > MAX_HEADER_BYTES or 8 + size > len(blob):
        raise ProtocolError("Invalid safetensors header length.")
    header = strict_json(blob[8:8 + size])
    if not isinstance(header, dict):
        raise ProtocolError("Safetensors header must be an object.")
    return header, len(blob) - 8 - size


def _descriptor(header, name, shapes, allowed_dtypes, payload_bytes):
    entry = header.get(name)
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise ProtocolError(f"Invalid {name} descriptor.")
    shape, dtype, offsets = entry["shape"], entry["dtype"], entry["data_offsets"]
    if not isinstance(shape, list) or any(type(x) is not int or x < 1 for x in shape):
        raise ProtocolError(f"Invalid {name} shape.")
    if shape not in shapes or dtype not in allowed_dtypes:
        raise ProtocolError(f"Unsupported {name} shape or dtype.")
    if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
        raise ProtocolError(f"Invalid {name} offsets.")
    itemsize = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "BOOL": 1}[dtype]
    if offsets[0] < 0 or offsets[1] > payload_bytes or offsets[1] - offsets[0] != math.prod(shape) * itemsize:
        raise ProtocolError(f"Invalid {name} payload length.")


def decode_conditioning(blob, *, model, mode, fingerprint, reference_count=0):
    header, payload_bytes = _read_header(blob)
    h3 = model == H3_MODEL
    allowed_tensors = {"__metadata__", "hidden_states", "minimax_token_tags"} if h3 else {"__metadata__", "hidden_states", "attention_mask"}
    if set(header) - allowed_tensors:
        raise ProtocolError("Unknown response tensor; refusing to discard conditioning fields.")
    meta = header.get("__metadata__")
    if not isinstance(meta, dict) or not METADATA_KEYS.issubset(meta) or set(meta) - METADATA_KEYS - OPTIONAL_METADATA_KEYS or any(not isinstance(v, str) for v in meta.values()):
        raise ProtocolError("Missing, unknown or non-string protocol metadata.")
    if meta["protocol"] != PROTOCOL or meta["model"] != model or meta["mode"] != mode:
        raise ProtocolError("Response protocol/model/mode does not match the request.")
    if meta["fingerprint"] != fingerprint:
        raise VersionMismatch("Remote encoder response fingerprint does not match the pinned version.")
    try:
        seq_len = int(meta["seq_len"])
        if str(seq_len) != meta["seq_len"] or not 1 <= seq_len <= MAX_SEQUENCE:
            raise ValueError()
        for field in ("encode_ms", "queue_wait_ms"):
            duration = float(meta[field])
            if not math.isfinite(duration) or duration < 0:
                raise ValueError()
        if not meta["server_version"] or len(meta["server_version"]) > 128:
            raise ValueError()
    except ValueError:
        raise ProtocolError("Invalid sequence length, timing or server version metadata.") from None
    _descriptor(header, "hidden_states", [[1, seq_len, 5120 if h3 else HIDDEN_WIDTH]], {"F32"}, payload_bytes)
    if h3:
        _descriptor(header, "minimax_token_tags", [[seq_len]], {"I64"}, payload_bytes)
    elif "attention_mask" in header:
        _descriptor(header, "attention_mask", [[1, seq_len]], {"BOOL", "I32", "I64", "F32", "F16", "BF16"}, payload_bytes)
    extras = strict_json(meta["extras"])
    if h3:
        if not isinstance(extras, dict) or set(extras) - {"pooled_output"} or extras.get("pooled_output") is not None and "pooled_output" in extras:
            raise ProtocolError("Unsupported MiniMax H3 conditioning extras.")
        if meta.get("has_mask", "False") != "False" or meta.get("has_slots", "False") != "False":
            raise ProtocolError("MiniMax H3 must not return image masks or slots.")
        try:
            tensors = load(blob)
        except Exception:
            raise ProtocolError("Malformed safetensors payload.") from None
        hidden, tags = tensors["hidden_states"], tensors["minimax_token_tags"]
        if not bool(torch.isfinite(hidden).all()) or not bool(((tags == 0) | (tags == 1)).all()):
            raise ProtocolError("Invalid MiniMax H3 hidden states or token tags.")
        return [[hidden, {**extras, "minimax_token_tags": tags}]], meta
    if not isinstance(extras, dict) or set(extras) - {"pooled_output", "image_slots"}:
        raise ProtocolError("Unknown conditioning extras or hooks cannot be represented by remote-te/1.")
    if "pooled_output" not in extras or extras["pooled_output"] is not None:
        raise ProtocolError("remote-te/1 requires pooled_output=null for this profile.")
    slots = extras.get("image_slots", [])
    if not isinstance(slots, list) or any(type(x) is not int or not 0 <= x <= seq_len for x in slots):
        raise ProtocolError("image_slots must be an integer list within the sequence.")
    if slots != sorted(slots) or len(slots) != (reference_count if mode == "edit" else 0):
        raise ProtocolError("image_slots do not match reference images and mode.")
    for key, actual in (("has_mask", "attention_mask" in header), ("has_slots", "image_slots" in extras)):
        if key in meta and meta[key] != str(actual):
            raise ProtocolError("Response presence flags disagree with tensors or extras.")
    try:
        tensors = load(blob)
    except Exception:
        raise ProtocolError("Malformed safetensors payload.") from None
    hidden = tensors["hidden_states"]
    if not bool(torch.isfinite(hidden).all()):
        raise ProtocolError("Non-finite hidden states in remote response.")
    extra = {"pooled_output": None}
    if "image_slots" in extras:
        extra["image_slots"] = list(slots)
    if "attention_mask" in tensors:
        mask = tensors["attention_mask"]
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ProtocolError("Attention mask must contain only zero and one.")
        extra["attention_mask"] = mask
    return [[hidden, extra]], meta
