#!/usr/bin/env python3
"""A faithful **DGX Spark simulator** — so every concept is learnable with no GPU.

When no real endpoint is reachable, the demos fall back to this module. It does
NOT pretend to be a smart LLM; instead it makes the *mechanics* of running models
on a DGX honest and visible:

  • MODEL_REGISTRY — real model names + their on-disk / in-VRAM footprints and the
    decode tok/s you'd actually see on a DGX Spark (memory-bandwidth bound).
  • gpu_telemetry() — an nvidia-smi-shaped snapshot for the GB10 (128 GB unified).
  • installed_models() — the models a fresh DGX Spark would have pulled.
  • stream_generate() — streams a short, topic-aware answer token-by-token at a
    plausible rate so latency / tok/s feel real.

Everything it prints is clearly marked as simulated. The exact real commands are
always shown by the demos, so SIM mode is a true dry-run of the real hardware.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

# ── the model registry (numbers are representative DGX Spark figures) ─────────
# tok_s is single-stream decode on a GB10 (273 GB/s unified) — decode is
# memory-bandwidth bound, so tok/s ≈ bandwidth / active-bytes-per-token.
@dataclass
class ModelSpec:
    name: str
    params_b: float          # total parameters (billions)
    active_b: float          # active params/token (= params_b for dense; less for MoE)
    quant: str               # weight format
    bytes_per_param: float   # effective bytes/param at this quant
    ctx: int                 # default context window
    tok_s: float             # ~single-stream decode tok/s on a DGX Spark
    arch: str                # "dense" or "MoE"

    @property
    def vram_gb(self) -> float:
        # weights + ~18% runtime overhead (kv-cache, activations, fragmentation)
        return round(self.params_b * self.bytes_per_param * 1.18, 1)

    @property
    def disk_gb(self) -> float:
        return round(self.params_b * self.bytes_per_param, 1)


REGISTRY: dict[str, ModelSpec] = {
    "qwen3.6:35b-a3b-q8_0":   ModelSpec("qwen3.6:35b-a3b-q8_0",   35, 3.0, "Q8_0",   1.0,  40960, 46.0, "MoE"),
    "qwen3.6:35b-a3b-nvfp4":  ModelSpec("qwen3.6:35b-a3b-nvfp4",  35, 3.0, "NVFP4",  0.55, 40960, 78.0, "MoE"),
    "qwen3.6:35b-a3b-bf16":   ModelSpec("qwen3.6:35b-a3b-bf16",   35, 3.0, "BF16",   2.0,  40960, 24.0, "MoE"),
    "llama3.3:70b":           ModelSpec("llama3.3:70b",           70, 70.0, "Q4_K_M", 0.55, 8192, 6.5, "dense"),
    "llama3.1:8b":            ModelSpec("llama3.1:8b",             8,  8.0, "Q4_K_M", 0.55, 8192, 38.0, "dense"),
    "gemma4:12b":             ModelSpec("gemma4:12b",             12, 12.0, "Q4_K_M", 0.55, 8192, 31.0, "dense"),
    "qwen2.5:32b":            ModelSpec("qwen2.5:32b",            32, 32.0, "Q4_K_M", 0.55, 8192, 14.0, "dense"),
    "phi3.5:3.8b":            ModelSpec("phi3.5:3.8b",           3.8, 3.8, "Q4_K_M", 0.55, 4096, 62.0, "dense"),
}

# What a freshly-onboarded DGX Spark would have pulled (the coding-agent default).
_INSTALLED = ["qwen3.6:35b-a3b-q8_0", "llama3.1:8b", "gemma4:12b"]


def installed_models() -> list[str]:
    return list(_INSTALLED)


def spec_for(name: str) -> ModelSpec:
    if name in REGISTRY:
        return REGISTRY[name]
    # unknown → approximate from the leading number in the tag
    import re
    m = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
    p = float(m.group(1)) if m else 8.0
    return ModelSpec(name, p, p, "Q4_K_M", 0.55, 8192, round(150 / max(p, 1), 1), "dense")


def gpu_telemetry(model: str | None = None, busy: bool = False) -> dict:
    """An nvidia-smi-shaped snapshot for a DGX Spark GB10 (128 GB unified)."""
    used = 0.0
    if model:
        used = spec_for(model).vram_gb
    return {
        "name": "NVIDIA GB10 (DGX Spark)",
        "driver": "580.95.05",
        "cuda": "13.0",
        "mem_total_gb": 128.0,
        "mem_used_gb": round(used, 1),
        "mem_free_gb": round(128.0 - used, 1),
        "util_pct": 94 if busy else 2,
        "temp_c": 71 if busy else 38,
        "power_w": 210 if busy else 28,
        "power_cap_w": 240,
    }


def render_smi(t: dict) -> str:
    """Render the telemetry dict as a compact nvidia-smi-style block."""
    bar_n = int(28 * t["mem_used_gb"] / t["mem_total_gb"])
    bar = "█" * bar_n + "·" * (28 - bar_n)
    return (
        f"  +-----------------------------------------------------------------+\n"
        f"  | NVIDIA-SMI 580.95   Driver {t['driver']}   CUDA {t['cuda']}            |\n"
        f"  |-----------------------------------------------------------------|\n"
        f"  | GPU  {t['name']:<40} |\n"
        f"  |   Temp {t['temp_c']}C   Power {t['power_w']}W / {t['power_cap_w']}W   "
        f"Util {t['util_pct']}%        |\n"
        f"  |   Mem  [{bar}] {t['mem_used_gb']:.1f} / {t['mem_total_gb']:.0f} GB |\n"
        f"  +-----------------------------------------------------------------+"
    )


# ── simulated generation (mechanics, not intelligence) ────────────────────────
_CANNED = {
    "privacy": (
        "Running the model on this DGX means the prompt, the activations, and the "
        "answer never leave the box. There is no network hop to a third party, so "
        "regulated data (PII, PHI, trade secrets) stays inside your security "
        "perimeter by physical design rather than by policy."
    ),
    "nvfp4": (
        "NVFP4 is NVIDIA's 4-bit floating-point weight format for Blackwell. It "
        "packs each weight into ~0.5 bytes with a per-block scale, cutting VRAM "
        "~4x versus BF16 while keeping accuracy within ~1-2% — and Blackwell has "
        "native NVFP4 tensor-core kernels, so it is fast, not just small."
    ),
    "serving": (
        "An OpenAI-compatible endpoint means any app that already speaks the "
        "OpenAI API works unchanged — you only swap the base_url to your DGX. "
        "Ollama, vLLM, and llama.cpp all expose /v1/chat/completions, so the "
        "client code is identical to the cloud; only the address changes."
    ),
    "default": (
        "This is a simulated answer from the DGX Spark simulator — it demonstrates "
        "the streaming, latency, and tok/s mechanics without a real model. Point "
        "the demo at a live Ollama or DGX endpoint to get genuine generations."
    ),
}


def _pick_canned(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ("privacy", "cloud", "sovereign", "leave")):
        return _CANNED["privacy"]
    if "nvfp4" in p or "quantiz" in p:
        return _CANNED["nvfp4"]
    if any(w in p for w in ("endpoint", "serv", "api", "openai")):
        return _CANNED["serving"]
    return _CANNED["default"]


def stream_generate(prompt: str, model: str | None = None, *, rate_tok_s: float | None = None):
    """Yield answer chunks (words) at a plausible DGX rate. Returns nothing; the
    caller prints chunks. Use ``rate`` from the model spec so tok/s feels real."""
    model = model or "qwen3.6:35b-a3b-q8_0"
    rate = rate_tok_s or spec_for(model).tok_s
    answer = _pick_canned(prompt)
    delay = 1.0 / max(rate, 1) * 1.3      # ~1.3 words/token is a rough mapping
    for word in answer.split(" "):
        yield word + " "
        time.sleep(min(delay, 0.05))      # cap so SIM never feels sluggish
