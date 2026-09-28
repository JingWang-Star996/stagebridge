import gc
import logging
from pathlib import Path
import time
import threading
import torch
from .fingerprint import fingerprint
from .codec import prepare_vision


class TEService:
    def __init__(self, checkpoint, model="qwen3vl_8b"):
        if model not in ("qwen3vl_8b", "minimax_h3"):
            raise ValueError("Unknown TE model profile")
        self.model = model
        self.checkpoint = Path(checkpoint).resolve()
        self.clip = None
        self.digest = fingerprint(self.checkpoint)
        self.signature = self._signature()
        self.load_ms = 0
        self.loading = False
        self.last_timing = {}

    def _signature(self):
        s = self.checkpoint.stat()
        return s.st_mtime_ns, s.st_size

    @property
    def fingerprint(self):
        return self.digest[:16]

    def changed(self):
        return self._signature() != self.signature

    def refresh(self):
        if self.changed():
            self.unload()
            self.digest = fingerprint(self.checkpoint)
            self.signature = self._signature()

    def load(self):
        if self.clip is not None:
            return
        import comfy.sd
        import comfy.model_management as mm
        minimum_free = 10 if self.model == "qwen3vl_8b" else 14
        if torch.cuda.mem_get_info()[0] < minimum_free * 1024**3:
            raise RuntimeError(f"TE needs at least {minimum_free} GiB free VRAM; existing workloads were left running")
        start = time.perf_counter()
        self.loading = True
        try:
            with torch.inference_mode():
                clip_type = comfy.sd.CLIPType.MINIMAX if self.model == "minimax_h3" else comfy.sd.CLIPType.QWEN_IMAGE
                self.clip = comfy.sd.load_clip(ckpt_paths=[str(self.checkpoint)], embedding_directory=None,
                                              clip_type=clip_type)
                # Warm native forward and resident weights before marking ready.
                self.clip.encode_from_tokens_scheduled(self._warmup_tokens())
                torch.cuda.synchronize()
            self.load_ms = (time.perf_counter() - start) * 1000
            logging.info("TE loaded fp=%s load_ms=%.1f free_total_mb=%s", self.fingerprint, self.load_ms, [v / 2**20 for v in torch.cuda.mem_get_info()])
        except Exception:
            self.unload()
            raise
        finally:
            self.loading = False

    def encode(self, prompt, resolution, mode, images):
        started = time.perf_counter()
        self.load()
        loaded = time.perf_counter()
        with torch.inference_mode():
            visual = images if self.model == "minimax_h3" else prepare_vision(images, resolution)
            prepared = time.perf_counter()
            if self.model == "minimax_h3":
                tokens = self.clip.tokenize(prompt, images=visual)
            else:
                tokens = self.clip.tokenize(prompt, images=visual, keep_vision=mode != "edit", prevent_empty_text=True)
            tokenized = time.perf_counter()
            result = self.clip.encode_from_tokens_scheduled(tokens)
            torch.cuda.synchronize()
            completed = time.perf_counter()
            self.last_timing = {
                "load_ms": (loaded - started) * 1000,
                "prepare_ms": (prepared - loaded) * 1000,
                "tokenize_ms": (tokenized - prepared) * 1000,
                "forward_sync_ms": (completed - tokenized) * 1000,
                "thread_id": threading.get_ident(),
            }
            return result

    def maintain_loaded(self):
        """Caller owns Runtime.serial; never load, refresh, or change business timing."""
        clip = self.clip
        if clip is None or self.loading:
            return False
        with torch.inference_mode():
            try:
                clip.encode_from_tokens_scheduled(TEService._warmup_tokens(self))
            finally:
                # A failed forward may already have queued GPU work. Keep the
                # caller's serial ownership until synchronization returns/errors.
                torch.cuda.synchronize()
        return True

    def _warmup_tokens(self):
        if getattr(self, "model", "qwen3vl_8b") == "minimax_h3":
            return self.clip.tokenize("warmup", images=[])
        return self.clip.tokenize("warmup", images=[], keep_vision=True, prevent_empty_text=True)

    def unload(self):
        if self.clip is not None:
            import comfy.model_management as mm
            mm.unload_model_and_clones(self.clip.patcher)
            self.clip = None
            gc.collect()
            mm.cleanup_models_gc()
            mm.soft_empty_cache(force=True)

    def stats(self):
        return {"fingerprint": self.fingerprint, "loaded": self.clip is not None and not self.loading,
                "loading": self.loading, "weight_changed": self.changed(), "load_ms": round(self.load_ms, 1),
                "vram_mb": round(torch.cuda.memory_allocated() / 2**20, 1) if torch.cuda.is_initialized() else 0,
                "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1) if torch.cuda.is_initialized() else 0}
