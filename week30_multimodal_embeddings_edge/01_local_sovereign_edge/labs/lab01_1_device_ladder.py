#!/usr/bin/env python3
"""Lab 01-1 · The device ladder: which AI fits on which hardware, and where an embedding model lands.

Offline arithmetic, no model needed. It walks Tina Huang's device ladder (Every Size Local AI In 24 Minutes),
applies her rule of thumb — usable RAM = RAM − 25 %, max parameters (billions) ≈ usable RAM ÷ 0.6 — and then
asks the question the video does not: where does an EMBEDDING model fit? EmbeddingGemma 2 is 270M parameters
for text and 740M with every encoder, so it lands several rungs lower than any chat model.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_1_device_ladder.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from embedkit import banner, bar, note, ok, result, step, table, warn  # noqa: E402

# (class, device, memory in GB as the video quotes it, peak bandwidth GB/s from vendor specs — RESEARCH.md §4)
LADDER = [
    ("chip", "Arduino Uno R4", 32 / 1024 / 1024, None),
    ("chip", "ESP32-S3", 8 / 1024, None),
    ("mini computer", "Raspberry Pi 5", 8, 17),
    ("mini computer", "iPhone 16 (apps get ~4–5 GB)", 4.5, None),
    ("personal computer", "MacBook Pro (the video's)", 32, None),
    ("home server", "Mac Studio", 64, None),
    ("home server", "AMD Ryzen AI Halo", 128, 256),
    ("GPU", "RTX 4090", 24, 1008),
    ("GPU", "8× H100", 640, 8 * 3350),
]


def usable_gb(ram_gb: float) -> float:
    """Tina's rule: about a quarter of the RAM goes to the OS and everything else."""
    return ram_gb * 0.75


def max_params_b(ram_gb: float) -> float:
    """Her rule of thumb: billions of parameters ≈ usable GB ÷ 0.6 (≈ a 4-bit model plus overhead)."""
    return usable_gb(ram_gb) / 0.6


def human(gb: float) -> str:
    if gb < 1 / 1024:
        return f"{gb * 1024 * 1024:.0f} KB"
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.0f} GB" if gb == int(gb) or gb >= 10 else f"{gb:.1f} GB"


banner("Lab 01-1 · the device ladder", "Tina Huang's tiers + her fit rule · then: where does an embedding model land?")

step(1, "her rule of thumb, rung by rung")
rows = []
for cls, dev, ram, _bw in LADDER:
    p = max_params_b(ram)
    size = "—" if p < 0.001 else f"{p * 1000:.1f}M" if p < 1 else f"{p:.1f}B"
    rows.append([cls, dev, human(ram), human(usable_gb(ram)), size])
table(rows, ["class", "device", "memory", "usable (−25 %)", "max params ≈ usable ÷ 0.6"])
note("worked example from the video: 32 GB → 24 GB usable → 24 ÷ 0.6 = 40B parameters")
check40 = round(max_params_b(32), 6)
ok(f"our function agrees: max_params_b(32) = {check40}") if check40 == 40 else warn(f"got {check40}")

step(2, "an embedding model is not a chat model — weights per EmbeddingGemma 2 configuration")
# parameters per configuration (model card, Selective Encoder Loading)
CONFIGS = [("text only", 270), ("text + image", 440), ("text + audio", 570), ("all four (incl. video)", 740)]
BYTES = {"float32": 4, "bfloat16": 2, "int8 / Q8_0": 1, "4-bit / Q4": 0.5}
rows = []
for name, params_m in CONFIGS:
    rows.append([name, f"{params_m}M"] + [f"{params_m * b:.0f} MB" for b in BYTES.values()])
table(rows, ["configuration", "params"] + list(BYTES))
note("weights only — activations, the tokenizer and the runtime add more on top")
note("REFERENCE (launch blog): quantized on a Pixel 11 Pro, text-only uses ~191MB active RAM, full multimodal ~567MB")
note("REFERENCE (GGUF cards): text-only Q8_0 = 310 MB, BF16 = 558 MB, UD-Q4_K_XL = 176 MB")

step(3, "which rungs can host EmbeddingGemma 2?")
NEED_GB = {"text, 4-bit": 0.176, "text, bf16": 0.54, "full, bf16": 1.48}
rows = []
for cls, dev, ram, _bw in LADDER:
    u = usable_gb(ram)
    cells = ["✓" if u >= need else "✕" for need in NEED_GB.values()]
    rows.append([dev, human(u)] + cells)
table(rows, ["device", "usable"] + [f"{k} ({human(v)})" for k, v in NEED_GB.items()])
warn("the ESP32-S3 has 6 MB usable — not even the 4-bit text model fits. It can still SEND a sentence to a "
     "bigger machine on the Wi-Fi and get a vector back (the video's Arduino → Mac Studio trick).")

step(4, "chat model vs embedding model on the same Raspberry Pi 5")
pi = usable_gb(8)
for name, gb in (("Qwen3 1.7B (4-bit, the video's Pi voice loop)", 1.7 * 0.6),
                 ("EmbeddingGemma 2 text (bf16)", 0.54),
                 ("EmbeddingGemma 2 full (bf16)", 1.48)):
    print(f"  {name:48s} {bar(gb, pi)} {gb:.2f} / {pi:.0f} GB")
result("an embedding model fits one rung lower than the smallest chat model you would want — "
       "that is why search, memory and RAG can live on the edge even when the LLM cannot.")
