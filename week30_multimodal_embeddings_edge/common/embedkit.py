"""embedkit — the lab helper for Week 30 (multimodal embeddings on local, sovereign and edge hardware).

Output helpers use the same glyphs as weeks 25–26, so the Embed Lab Runner colours them:
  ━ banner · ▣ step · ✓ pass · ✕ fail · ⚠ warn · ◆ metric · ═ result · → action · │ table · $ command

Model helpers wrap sentence-transformers' EmbeddingGemma 2 with selective encoder loading:

    import embedkit as ek
    model = ek.load(("text",))               # 270M text-only
    model = ek.load(("text", "image"))       # 440M
    model = ek.load(("text", "image", "audio"))   # 740M, everything (video goes through the vision encoder)

Labs run in week30_multimodal_embeddings_edge/.venv-embed (transformers ≥ 5.19, sentence-transformers ≥ 6.1). Once the model is in the
Hugging Face cache, load() sets HF_HUB_OFFLINE=1 so nothing is fetched from the network.
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

COMMON = Path(__file__).resolve().parent            # …/week30_multimodal_embeddings_edge/common
WEEK = COMMON.parent                                # …/week30_multimodal_embeddings_edge
ROOT = WEEK.parent                                  # …/agenticaicodingfitness
MEDIA = COMMON / "media"                            # SYNTHETIC hotel media (from week 24, module 13)
SOURCES = COMMON / "sources"
MODEL_ID = "google/embeddinggemma-2"
DIMS = (768, 512, 256, 128)                         # Matryoshka dimensions the model was trained for
OLLAMA = "http://localhost:11434"
GGUF_TAG = "hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0"   # text-only GGUF (310 MB)

# Modalities → the encoders sentence-transformers must keep (model card, "Selective Encoder Loading").
_CONFIG_KWARGS = {
    frozenset({"text"}): {"vision_config": None, "audio_config": None},          # 270M
    frozenset({"text", "image"}): {"audio_config": None},                        # 440M
    frozenset({"text", "audio"}): {"vision_config": None},                       # 570M
    frozenset({"text", "image", "audio"}): {},                                   # 740M
}


# ── output ───────────────────────────────────────────────────────────────────
def banner(title: str, sub: str = "") -> None:
    print("━" * 72)
    print(f"━━ {title}")
    if sub:
        print(f"   {sub}")
    print("━" * 72, flush=True)


def step(n, text: str) -> None:
    print(f"\n▣ STEP {n} · {text}", flush=True)


def ok(msg: str) -> None:
    print(f"✓ {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"⚠ {msg}", flush=True)


def note(msg: str) -> None:
    print(f"◆ {msg}", flush=True)


metric = note


def result(msg: str) -> None:
    print(f"═ {msg}", flush=True)


def action(msg: str) -> None:
    print(f"→ {msg}", flush=True)


def table(rows: list[list], headers: list[str]) -> None:
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [min(max(len(r[i]) for r in cols), 52) for i in range(len(headers))]
    clip = lambda s, w: s if len(s) <= w else s[:w - 1] + "…"          # noqa: E731
    fmt = lambda r: "  ".join(clip(c, widths[i]).ljust(widths[i]) for i, c in enumerate(r))  # noqa: E731
    print("│ " + fmt(headers))
    print("│ " + "  ".join("─" * w for w in widths))
    for r in cols[1:]:
        print("│ " + fmt(r))
    sys.stdout.flush()


def bar(value: float, vmax: float, width: int = 28) -> str:
    n = 0 if vmax <= 0 else max(0, min(width, round(width * value / vmax)))
    return "█" * n + "░" * (width - n)


def check(cond: bool, good: str, bad: str) -> bool:
    """Exercise checker line: ✓ or ✕."""
    print(("✓ " + good) if cond else ("✕ " + bad), flush=True)
    return bool(cond)


# ── paths ────────────────────────────────────────────────────────────────────
def media(name: str) -> str:
    path = MEDIA / name
    if not path.exists():
        raise FileNotFoundError(f"no media file {name} in {MEDIA}")
    return str(path)


def load_kb() -> dict:
    """The SYNTHETIC hotel knowledge base: {"docs": [{title, text}], "queries": [{q, gold}]}."""
    return json.loads((COMMON / "data" / "hotel_kb.json").read_text(encoding="utf-8"))


def runs(module_file: str) -> Path:
    """NN_folder/.runs/ for a lab file (gitignored)."""
    d = Path(module_file).resolve().parents[1] / ".runs"
    d.mkdir(exist_ok=True)
    return d


# ── the model ────────────────────────────────────────────────────────────────
def _cached() -> bool:
    """Checked on the filesystem, BEFORE huggingface_hub is imported (it reads HF_HUB_OFFLINE at import time)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    snaps = hub / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any(snaps.glob("*/model.safetensors"))


if _cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")    # sovereign by default: the local cache is enough


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def safe_dtype(dev: str | None = None):
    """bfloat16 where the hardware has it, else float32 — never float16 (model card: NaN / silent degradation)."""
    import torch
    dev = dev or device()
    if dev == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dev == "mps":
        return torch.bfloat16
    return torch.float32


_MODELS: dict = {}


def load(modalities=("text",), *, dtype=None, dev: str | None = None, quiet: bool = False):
    """A cached SentenceTransformer for EmbeddingGemma 2 with only the encoders you need."""
    key_mod = frozenset(modalities) | {"text"}
    if "video" in key_mod:                               # video frames go through the vision encoder
        key_mod = (key_mod - {"video"}) | {"image"}
    if key_mod not in _CONFIG_KWARGS:
        raise ValueError(f"unsupported modality set {sorted(key_mod)}")
    dev = dev or device()
    k = (key_mod, str(dtype), dev)
    if k in _MODELS:
        return _MODELS[k]
    import warnings
    warnings.filterwarnings("ignore")
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    kwargs = {"config_kwargs": dict(_CONFIG_KWARGS[key_mod]), "device": dev}
    if dtype is not None:
        kwargs["model_kwargs"] = {"dtype": dtype}
    model = SentenceTransformer(MODEL_ID, **kwargs)
    secs = time.perf_counter() - t0
    n = sum(p.numel() for p in model.parameters())
    if not quiet:
        ok(f"loaded {MODEL_ID} · encoders {'+'.join(sorted(key_mod))} · {n / 1e6:.1f}M params · "
           f"{dev} · {secs:.1f}s")
    _MODELS[k] = model
    return model


def params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ── vector maths (numpy) ─────────────────────────────────────────────────────
def renorm(vectors):
    import numpy as np
    v = np.asarray(vectors, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def top_k(query, matrix, k: int = 3) -> list[tuple[int, float]]:
    """Indices and cosine scores of the k rows of `matrix` closest to `query`."""
    import numpy as np
    q = renorm(query)
    m = renorm(matrix)
    scores = m @ q
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def recall_at_1(query_vecs, doc_vecs, gold: list[int]) -> float:
    import numpy as np
    s = renorm(query_vecs) @ renorm(doc_vecs).T
    return float(np.mean(np.argmax(s, axis=1) == np.asarray(gold)))


def mrr(query_vecs, doc_vecs, gold: list[int]) -> float:
    import numpy as np
    s = renorm(query_vecs) @ renorm(doc_vecs).T
    ranks = [int(np.sum(row > row[g])) + 1 for row, g in zip(s, gold)]
    return float(np.mean([1.0 / r for r in ranks]))


# ── measuring ────────────────────────────────────────────────────────────────
@contextmanager
def timer():
    """with ek.timer() as t: ...; t.s → seconds."""
    class T:
        s = 0.0
    t = T()
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t.s = time.perf_counter() - t0


def current_rss_mb() -> float:
    """Resident memory of this process right now, in MB."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        return int(out) / 1024
    except Exception:
        return math.nan


def param_mb(model) -> float:
    """Bytes the parameters occupy once materialised, in MB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20


def rss_mb() -> float:
    """Peak resident memory of this process in MB (macOS reports bytes, Linux KB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return math.nan
    i = (len(s) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def machine() -> dict:
    """What this machine is, from the OS (no network)."""
    import platform
    import subprocess
    info = {"os": f"{platform.system()} {platform.release()}", "arch": platform.machine(),
            "cpu": platform.processor() or platform.machine(), "ram_gb": None}
    if sys.platform == "darwin":
        def sysctl(k):
            try:
                return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception:
                return ""
        info["cpu"] = sysctl("machdep.cpu.brand_string") or info["cpu"]
        mem = sysctl("hw.memsize")
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
        info["cores"] = sysctl("hw.ncpu")
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 2**20, 1)
        except Exception:
            pass
        info["cores"] = str(os.cpu_count())
    return info


# ── local Ollama (laptop) ────────────────────────────────────────────────────
def _http(method: str, path: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def ollama_up() -> bool:
    try:
        _http("GET", "/api/tags", timeout=3)
        return True
    except Exception:
        return False


def ollama_models() -> list[str]:
    try:
        return [m["name"] for m in _http("GET", "/api/tags", timeout=5).get("models", [])]
    except Exception:
        return []


def ollama_embed(model: str, texts: list[str], timeout: float = 300) -> list[list[float]]:
    return _http("POST", "/api/embed", {"model": model, "input": texts}, timeout=timeout)["embeddings"]


def ollama_chat(model: str, messages: list[dict], *, tools: list | None = None, max_tokens: int = 1024,
                temperature: float = 0.2, think: bool | None = None, timeout: float = 600) -> dict:
    """Ollama's native /api/chat (no streaming). Returns the `message` dict (content, thinking, tool_calls)."""
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature}}
    if tools:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    out = _http("POST", "/api/chat", body, timeout=timeout)
    msg = out.get("message", {})
    msg["_eval_count"] = out.get("eval_count")
    msg["_secs"] = (out.get("total_duration") or 0) / 1e9
    return msg


def pick_chat_model(preferred=("gemma4:12b", "gemma4:latest", "nemotron-3-nano:latest", "gemma3:4b")) -> str:
    have = ollama_models()
    for m in preferred:
        if m in have and not m.endswith(":cloud") and "-cloud" not in m:
            return m
    local = [m for m in have if "cloud" not in m]
    return local[0] if local else ""
