"""mackit — EmbeddingGemma 2 on a Mac: MLX, ONNX Runtime and PyTorch MPS side by side (Week 30, Module 08).

Three runtimes, three homes:
  • PyTorch + MPS      → week30_multimodal_embeddings_edge/.venv-embed  (embedkit.load)
  • MLX (Apple's own)  → week30_multimodal_embeddings_edge/.venv-mlx    (mlx-community/embeddinggemma-2-{bf16,4bit}, via mlx-vlm)
  • ONNX Runtime       → week30_multimodal_embeddings_edge/.venv-embed  (onnx-community/embeddinggemma-2-ONNX, downloaded to week30_multimodal_embeddings_edge/.models)

The runner starts every lab with .venv-embed's Python. A lab that needs MLX calls `ensure_mlx()` first: if MLX
is not importable it re-runs the same script with week30_multimodal_embeddings_edge/.venv-mlx/bin/python (same arguments, same output).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

COMMON = Path(__file__).resolve().parent
WEEK = COMMON.parent
MLX_PY = WEEK / ".venv-mlx" / "bin" / "python"
EMBED_PY = WEEK / ".venv-embed" / "bin" / "python"
MODELS = WEEK / ".models"
ONNX_DIR = MODELS / "embeddinggemma-2-ONNX"
MLX_REPOS = {"bf16": "mlx-community/embeddinggemma-2-bf16", "4bit": "mlx-community/embeddinggemma-2-4bit"}
_EMPTY_MEDIA = ("image_features", "video_features", "audio_features")


def ensure_mlx() -> None:
    """Re-exec this script under week30_multimodal_embeddings_edge/.venv-mlx if MLX is not importable here."""
    try:
        import mlx.core  # noqa: F401
        import mlx_vlm  # noqa: F401
        return
    except ImportError:
        pass
    if not MLX_PY.exists():
        print("✕ week30_multimodal_embeddings_edge/.venv-mlx is missing — see week30_multimodal_embeddings_edge/README.md (Module 08 setup)", flush=True)
        sys.exit(1)
    print(f"→ MLX lives in week30_multimodal_embeddings_edge/.venv-mlx — re-running this lab with {MLX_PY.relative_to(WEEK.parent)}", flush=True)
    os.execv(str(MLX_PY), [str(MLX_PY), *sys.argv])


def _quiet() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# ── MLX ──────────────────────────────────────────────────────────────────────
class MLXEmbedder:
    """EmbeddingGemma 2 in MLX. Text via the tokenizer; images/audio/video via the model's processor."""

    def __init__(self, variant: str = "bf16", multimodal: bool = False):
        _quiet()
        import mlx.core as mx
        from mlx_vlm.utils import get_model_path
        self.mx, self.variant, self.repo = mx, variant, MLX_REPOS[variant]
        t0 = time.perf_counter()
        self.path = get_model_path(self.repo)
        if multimodal:
            from mlx_vlm import load
            self.model, self.processor = load(self.repo)
            self.tokenizer = self.processor.tokenizer
        else:
            from mlx_vlm.embedding_loader import load_embedding_model
            from transformers import AutoTokenizer
            self.model = load_embedding_model(self.path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
            self.processor = None
        self.load_s = time.perf_counter() - t0
        self.disk_mb = sum(f.stat().st_size for f in Path(self.path).glob("*.safetensors")) / 1e6

    def _run(self, inputs) -> "object":
        mx = self.mx
        out = self.model(**{k: mx.array(v) for k, v in inputs.items()}).text_embeds
        mx.eval(out)
        import numpy as np
        return np.array(out.astype(mx.float32))

    def encode(self, texts: list[str]):
        return self._run(self.tokenizer(texts, padding=True, return_tensors="np"))

    def encode_image(self, path: str):
        from PIL import Image
        return self._run(self.processor(images=[Image.open(path).convert("RGB")], return_tensors="np"))[0]

    def encode_audio(self, path: str):
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        return self._run(self.processor(audio=[audio], sampling_rate=sr, return_tensors="np"))[0]

    def peak_mb(self) -> float:
        return self.mx.get_peak_memory() / 2**20


# ── ONNX Runtime ─────────────────────────────────────────────────────────────
def onnx_download() -> Path:
    """The text model in q4 and fp16 into week30_multimodal_embeddings_edge/.models (ONNX Runtime refuses the HF cache's symlinked blobs)."""
    from huggingface_hub import snapshot_download
    _quiet()
    snapshot_download("onnx-community/embeddinggemma-2-ONNX", local_dir=str(ONNX_DIR),
                      allow_patterns=["onnx/model_q4.onnx*", "onnx/model_fp16.onnx*", "*.json", "tokenizer*"])
    return ONNX_DIR


class ONNXEmbedder:
    """The ONNX text model. Media inputs are passed as empty (0 × 512) feature tensors."""

    def __init__(self, dtype: str = "q4", providers=("CPUExecutionProvider",)):
        _quiet()
        import onnxruntime as ort
        from transformers import AutoTokenizer
        if not (ONNX_DIR / "onnx" / f"model_{dtype}.onnx").exists():
            onnx_download()
        t0 = time.perf_counter()
        so = ort.SessionOptions()
        so.log_severity_level = 2 if "CoreMLExecutionProvider" in providers else 3   # keep CoreML's coverage report
        self.session = ort.InferenceSession(str(ONNX_DIR / "onnx" / f"model_{dtype}.onnx"), so, providers=list(providers))
        self.tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
        self.load_s = time.perf_counter() - t0
        self.dtype = dtype
        self.disk_mb = sum(f.stat().st_size for f in (ONNX_DIR / "onnx").glob(f"model_{dtype}.onnx*")) / 1e6
        self.provider = self.session.get_providers()[0]

    def encode(self, texts: list[str]):
        import numpy as np
        enc = self.tokenizer(texts, padding=True, return_tensors="np")
        empty = np.zeros((0, 512), dtype=np.float32)
        feeds = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
        feeds.update({k: empty for k in _EMPTY_MEDIA})
        return self.session.run(["sentence_embedding"], feeds)[0]


# ── macOS facts (no sudo) ────────────────────────────────────────────────────
def power_source() -> str:
    import subprocess
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        return "AC" if "AC Power" in out else ("battery" if "Battery Power" in out else "unknown")
    except Exception:
        return "unknown"


def battery_watts() -> float | None:
    """Instantaneous draw from the battery (W) when on battery power; None on AC or without a battery."""
    import re
    import subprocess
    try:
        out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    amp = re.search(r'"InstantAmperage" = (-?\d+)', out) or re.search(r'"Amperage" = (-?\d+)', out)
    volt = re.search(r'"Voltage" = (\d+)', out)
    if not amp or not volt:
        return None
    a = int(amp.group(1))
    if a > 2**63:                       # ioreg prints negative values as unsigned 64-bit
        a -= 2**64
    if a >= 0:                          # charging or idle on AC: not a draw from the battery
        return None
    return abs(a) / 1000 * int(volt.group(1)) / 1000
