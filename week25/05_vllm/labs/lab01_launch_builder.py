#!/usr/bin/env python3
"""Lab 05-1 · Launch-command builder: size vLLM's memory flags for a 128 GB unified-memory Spark.

Pure arithmetic, runs anywhere (no Spark needed). vLLM reserves a fixed slice of memory at start-up
(--gpu-memory-utilization), loads the weights into it, and turns the rest into KV cache. This lab does
that sum for models from the vLLM playbook's support matrix, shows how many full-length conversations fit,
and prints a ready-to-paste `docker run` built from the playbook's base configuration.

Numbers are estimates, not measurements: NVFP4/FP8 checkpoints keep a few layers at higher precision, and
the real KV size is printed by vLLM at start-up. Use this to pick sensible flags before you download 40 GB.

Run: .venv/bin/python week25/05_vllm/labs/lab01_launch_builder.py [--pick 3] [--ctx 32768] [--util 0.8] [--kv fp8]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import SPEC, banner, bar, kv_cache_gb, note, result, step, table, warn, weights_gb  # noqa: E402

# HF handle (from the vLLM playbook's model support matrix), params (B), weight format, layers, KV heads, head dim.
# Layer/head numbers are from each model's config.json on Hugging Face — check them there for other models.
MODELS = [
    ("nvidia/Llama-3.1-8B-Instruct-FP8",     8.0,  "fp8",   32, 8, 128),
    ("nvidia/Qwen3-14B-NVFP4",              14.8,  "nvfp4", 40, 8, 128),
    ("nvidia/Qwen3-32B-NVFP4",              32.8,  "nvfp4", 64, 8, 128),
    ("nvidia/Llama-3.3-70B-Instruct-NVFP4", 70.6,  "nvfp4", 80, 8, 128),
    ("meta-llama/Llama-3.3-70B-Instruct",   70.6,  "bf16",  80, 8, 128),
]
# Two-Spark extension (course addition, Section 6): NVIDIA's own 120B model in FP8. It is a Mamba-2/MoE hybrid:
# only 8 of its 88 layers are attention layers, so only those 8 hold KV cache (2 KV heads each, config.json).
# The 40 Mamba layers keep a fixed-size state per sequence instead, which this sum leaves out.
NEMOTRON_FP8 = ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8", 120.0, "fp8", 8, 2, 128)
RUNTIME_GB = 4         # course assumption: activations, CUDA graphs, sampler buffers inside vLLM's slice
KV_BYTES = {"auto": 2, "fp8": 1}     # --kv-cache-dtype: auto = model dtype (bf16 → 2 bytes); fp8 → 1 byte


def plan(params_b, fmt, layers, kvh, hd, *, ctx, util, kv="auto", tp=1):
    """Per-node budget for vLLM: util × 128 GB, minus weights (split over tp nodes), minus runtime."""
    budget = util * SPEC["memory_gb"]
    w = weights_gb(params_b, fmt) / tp
    kv_per_seq = kv_cache_gb(layers, kvh // tp, hd, ctx, 1, KV_BYTES[kv])     # tensor parallel splits KV heads
    kv_room = budget - w - RUNTIME_GB
    fits = kv_room / kv_per_seq if kv_room > 0 else 0.0
    return {"budget": budget, "weights": w, "kv_room": kv_room, "kv_per_seq": kv_per_seq, "fits": fits}


def seqs_flag(fits: float) -> int:
    """Largest power of two ≤ the number of full-length sequences that fit (never preempts at full length)."""
    return 2 ** int(math.log2(fits)) if fits >= 1 else 0


ap = argparse.ArgumentParser()
ap.add_argument("--pick", type=int, default=3, help="row 1-5 of the model table for the generated command")
ap.add_argument("--ctx", type=int, default=32_768, help="--max-model-len")
ap.add_argument("--util", type=float, default=0.8, help="--gpu-memory-utilization (playbook base config: 0.8)")
ap.add_argument("--kv", choices=["auto", "fp8"], default="auto", help="--kv-cache-dtype")
args = ap.parse_args()

banner("Lab 05-1 · vLLM launch-command builder",
       "memory flags from arithmetic · no Spark needed · the same on every machine", status=False)

step(1, f"what fits in --gpu-memory-utilization {args.util} × {SPEC['memory_gb']} GB, "
        f"--max-model-len {args.ctx}, --kv-cache-dtype {args.kv}")
rows = []
for i, (name, p, fmt, L, kvh, hd) in enumerate(MODELS, 1):
    r = plan(p, fmt, L, kvh, hd, ctx=args.ctx, util=args.util, kv=args.kv)
    if r["kv_room"] <= 0:
        verdict = "✕ weights alone exceed the slice"
    else:
        verdict = f"{r['fits']:6.1f} full-length seqs"
    rows.append([i, name, fmt, f"{r['weights']:5.1f} GB", f"{max(r['kv_room'], 0):5.1f} GB",
                 f"{r['kv_per_seq']:5.2f} GB", verdict])
table(rows, ["#", "model (HF handle)", "format", "weights", "KV room", "KV / seq", "fits at full context"])
note(f"slice = {args.util} × {SPEC['memory_gb']} GB = {args.util * SPEC['memory_gb']:.1f} GB; "
     f"KV room = slice − weights − ~{RUNTIME_GB} GB runtime (course assumption).")
note("vLLM prints its own numbers at start-up (KV cache size in tokens, maximum concurrency). "
     "Trust those over this table.")

step(2, "the three levers — Qwen3-32B NVFP4 at 32K context")
name, p, fmt, L, kvh, hd = MODELS[2]
for util in (0.5, 0.8, 0.9):
    for kv in ("auto", "fp8"):
        r = plan(p, fmt, L, kvh, hd, ctx=32_768, util=util, kv=kv)
        print(f"│ util {util:.1f} · kv {kv:4s}  {r['fits']:5.1f} × 32K seqs  {bar(r['fits'], 30)}")
for ctx in (8_192, 131_072):
    r = plan(p, fmt, L, kvh, hd, ctx=ctx, util=0.8)
    print(f"│ util 0.8 · kv auto · ctx {ctx // 1024:>3}K → {r['fits']:6.1f} full-length seqs")
note("Raising --gpu-memory-utilization takes memory from the OS and every other process — on a Spark the "
     "'GPU memory' IS the system memory. The Nemotron playbook lowers it to 0.70 first when it hits OOM.")
note("--kv-cache-dtype fp8 halves the KV cache per token; the agent-ready Qwen3.6 and Nemotron Super "
     "recipes both use it.")

step(3, "two Sparks, tensor parallel 2 — the playbook's Llama 3.3 70B (bf16)")
name, p, fmt, L, kvh, hd = MODELS[4]
one = plan(p, fmt, L, kvh, hd, ctx=2048, util=0.8)
two = plan(p, fmt, L, kvh, hd, ctx=2048, util=0.8, tp=2)
table([["1 Spark", f"{one['weights']:.1f} GB", f"{one['budget']:.1f} GB",
        "✕ does not fit" if one["kv_room"] <= 0 else f"{one['fits']:.0f} seqs"],
       ["2 Sparks, TP=2", f"{two['weights']:.1f} GB per node", f"{two['budget']:.1f} GB per node",
        f"{two['fits']:.0f} × 2K seqs"]],
      ["setup", "weights", "vLLM slice", "at --max-model-len 2048"])
for ctx in (8_192, 32_768):
    r = plan(p, fmt, L, kvh, hd, ctx=ctx, util=0.8, tp=2)
    print(f"│ TP=2 · ctx {ctx // 1024:>2}K → {r['fits']:5.1f} full-length seqs")
note("Tensor parallelism splits every layer's weights AND its KV heads across the nodes, so each Spark holds "
     "half. The price is an all-reduce over the 200 Gb/s QSFP link on every layer of every token (Module 02).")
name, p, fmt, L, kvh, hd = NEMOTRON_FP8
one = plan(p, fmt, L, kvh, hd, ctx=262_144, util=0.8, kv="fp8")
two = plan(p, fmt, L, kvh, hd, ctx=262_144, util=0.8, kv="fp8", tp=2)
print(f"│ {name} · ctx 256K · kv fp8")
one_verdict = "✕ does not fit" if one["kv_room"] <= 0 else f"{one['fits']:.0f} seqs"
print(f"│   1 Spark         {one['weights']:5.1f} GB          {one_verdict}")
print(f"│   2 Sparks, TP=2  {two['weights']:5.1f} GB per node  {two['kv_per_seq']:.2f} GB KV per 256K seq → "
      f"{two['fits']:4.1f} full-length seqs")
note("Only 8 of Nemotron 3 Super's 88 layers keep a KV cache, so a 256K conversation costs ~0.5 GB, not tens of GB. "
     "On two Sparks vLLM reported 60.84x maximum concurrency at 262,144 tokens (Section 6).")

step(4, f"your command — row {args.pick}")
pick = MODELS[max(1, min(args.pick, len(MODELS))) - 1]
name, p, fmt, L, kvh, hd = pick
r = plan(p, fmt, L, kvh, hd, ctx=args.ctx, util=args.util, kv=args.kv)
n = seqs_flag(r["fits"])
if n == 0:
    warn(f"{name} does not fit one full {args.ctx}-token sequence at util {args.util}. "
         "Lower --ctx, use fp8 KV, a smaller format, or two Sparks (step 3).")
else:
    kv_flag = f" \\\n    --kv-cache-dtype {args.kv}" if args.kv != "auto" else ""
    print(f"""$ docker run -d \\
  --name vllm-server \\
  --gpus all \\
  --ipc host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  --entrypoint "" \\
  -p 8000:8000 \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  vllm/vllm-openai:latest \\
  vllm serve {name} \\
    --max-model-len {args.ctx} \\
    --gpu-memory-utilization {args.util} \\
    --max-num-seqs {n}{kv_flag}""")
    note(f"--max-num-seqs {n}: the largest power of two ≤ {r['fits']:.1f} full-length sequences, so the KV cache "
         "never runs out even if every user sends a full context. Short chats could safely run more.")
    note("Shape = the playbook's base configuration. Course change: the whole ~/.cache/huggingface is mounted "
         "(not only hub/) so the token from `hf auth login` on the Spark is visible; no -e HF_TOKEN needed.")
result("Weights are fixed; the KV cache is what you tune. Context × concurrent sequences × bytes per token "
       "must fit in the slice you give vLLM.")
