import hashlib
import json
import os
from pathlib import Path


def fingerprint(path):
    """Cache by nanosecond mtime+size; fail if the file changes while hashing."""
    path = Path(path)
    before = path.stat()
    signature = {"size": before.st_size, "mtime_ns": before.st_mtime_ns}
    sidecar = path.with_suffix(path.suffix + ".fp")
    try:
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
        value = cached["sha256"]
        if all(cached.get(k) == v for k, v in signature.items()) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
            return value
    except (OSError, ValueError, KeyError, TypeError):
        pass
    with path.open("rb") as f:
        value = hashlib.file_digest(f, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while hashing")
    temp = sidecar.with_name(sidecar.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps({**signature, "sha256": value}), encoding="utf-8")
        os.replace(temp, sidecar)
    except OSError:
        # Read-only checkpoint storage remains supported; only caching is optional.
        temp.unlink(missing_ok=True)
    return value
