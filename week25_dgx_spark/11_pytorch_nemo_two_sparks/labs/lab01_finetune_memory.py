#!/usr/bin/env python3
"""Lab 11-1 · Fine-tune memory: full vs LoRA vs QLoRA, on one Spark or sharded over two.

Pure arithmetic, runs anywhere. For the six recipes in this module (four PyTorch playbook
scripts, two NeMo AutoModel examples) it counts parameters from each model's shape, adds up

    weights + gradients + optimizer states   (bytes per parameter depend on the method)
    + activations                            (tokens in flight × hidden × layers — a rough rule)

and says whether the job fits on one Spark (128 GB), needs two with FSDP (256 GB), or neither.
It then shows what FSDP sends over the QSFP link for each forward + backward pass.

Every number is arithmetic, not a measurement. Lab 03 reads what your Spark really used.

Run: .venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab01_finetune_memory.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import SPEC, banner, bar, note, result, step, table  # noqa: E402

GB = 1e9
ONE = SPEC["memory_gb"]
HEADROOM_GB = 10                    # OS, CUDA context, the Python process, dataloader
LINK_GBS = 21.875                   # busbw pass mark from Module 02 (NVIDIA's cluster script)

# name: layers, hidden, kv_dim (kv heads × head dim), MLP intermediate, vocab, tied embeddings?
ARCH = {
    "Llama 3.2 3B":  (28, 3072, 1024, 8192, 128256, True),
    "Llama 3.1 8B":  (32, 4096, 1024, 14336, 128256, False),
    "Llama 3.1 70B": (80, 8192, 1024, 28672, 128256, False),
    "Qwen3 8B":      (36, 4096, 1024, 12288, 151936, False),
}


def linear_shapes(name: str) -> list[tuple[int, int]]:
    """(d_in, d_out) of the seven projections in one decoder layer: q k v o gate up down."""
    L, h, kv, ff, *_ = ARCH[name]
    return [(h, h), (h, kv), (h, kv), (h, h), (h, ff), (h, ff), (ff, h)]


def params(name: str) -> tuple[float, float]:
    """(linear-layer params, embedding + lm_head params). Norms are too small to matter."""
    L, h, _, _, vocab, tied = ARCH[name]
    lin = L * sum(i * o for i, o in linear_shapes(name))
    emb = vocab * h * (1 if tied else 2)
    return lin, emb


def lora_params(name: str, r: int) -> int:
    """LoRA adds A (r × d_in) and B (d_out × r) to every targeted projection: r × (d_in + d_out)."""
    return ARCH[name][0] * sum(r * (i + o) for i, o in linear_shapes(name))


def activations_gb(name: str, tokens: int, checkpointing: bool) -> float:
    """Rough rule (Korthikanti et al. 2022, flash attention): ~34 × hidden bytes per token per layer
    without checkpointing; with it, one bf16 layer input per layer plus one layer recomputed.
    Plus the logits for the loss in fp32 (tokens × vocab × 4 bytes)."""
    L, h, *_, vocab, _ = ARCH[name]
    per_layer = 34 * h * tokens
    acts = (2 * h * tokens * L + per_layer) if checkpointing else per_layer * L
    return (acts + tokens * vocab * 4) / GB


# recipe: (label, model, method, micro-batch, seq cap, checkpointing, source)
#   methods: full = bf16 weights + bf16 grads + AdamW m, v in bf16 (the scripts load bf16, optim adamw_torch)
#            lora = frozen bf16 base + LoRA r=8 · qlora = frozen NF4 base (~4.13 bits) + LoRA r=8
RECIPES = [
    ("PyTorch full SFT", "Llama 3.2 3B", "full", 8, 2048, False, "Llama3_3B_full_finetuning.py defaults"),
    ("PyTorch LoRA", "Llama 3.1 8B", "lora", 8, 2048, False, "Llama3_8B_LoRA_finetuning.py defaults"),
    ("PyTorch QLoRA", "Llama 3.1 70B", "qlora", 8, 2048, False, "Llama3_70B_qLoRA_finetuning.py defaults"),
    ("PyTorch QLoRA +ckpt", "Llama 3.1 70B", "qlora", 8, 2048, True, "same, with --gradient_checkpointing"),
    ("PyTorch LoRA (FSDP)", "Llama 3.1 70B", "lora", 4, 2048, True, "Llama3_70B_LoRA_finetuning.py defaults"),
    ("NeMo full SFT", "Qwen3 8B", "full", 1, 1024, False, "packed 1024, local_batch_size 1"),
    ("NeMo LoRA", "Llama 3.1 8B", "lora", 1, 1024, False, "packed 1024, batch 1 assumed"),
]
ALPACA_TOKENS = 300                 # ASSUMPTION: Alpaca samples are short — the longest in a batch ~300 tokens, not 2048


def state_gb(name: str, method: str, r: int = 8) -> tuple[float, float, float]:
    """(weights, grads + optimizer, trainable params) in GB / GB / count."""
    lin, emb = params(name)
    total = lin + emb
    if method == "full":
        return total * 2 / GB, total * 6 / GB, total             # grads 2 B + AdamW m, v 2 B each
    lp = lora_params(name, r)
    base = total * 2 if method == "lora" else lin * 4.127 / 8 + emb * 2   # NF4 + double-quant constants
    return (base + lp * 2) / GB, lp * 6 / GB, lp


banner("Lab 11-1 · fine-tune memory — full vs LoRA vs QLoRA, one Spark or two",
       "arithmetic only · no Spark needed · the same on every machine", status=False)

step(1, "parameters, counted from each model's shape")
rows = []
for name in ARCH:
    lin, emb = params(name)
    rows.append([name, f"{(lin + emb) / 1e9:.2f}B", f"{lin / 1e9:.2f}B", f"{emb / 1e9:.2f}B",
                 f"{lora_params(name, 8):,}"])
table(rows, ["model", "total", "linear layers", "embeddings", "LoRA r=8 params (all 7 projections)"])
note("The 8B LoRA script prints 'Trainable parameters = …' when it starts: compare it with this column in lab 03.")

step(2, "bytes per parameter, by method")
table([["full, as the playbook runs it", "2", "2", "4 (m, v in bf16)", "8"],
       ["full, classic mixed precision", "2 + 4 fp32 master", "2", "8 (m, v in fp32)", "16"],
       ["LoRA (frozen base)", "2", "0", "0", "2 + tiny adapter"],
       ["QLoRA (frozen NF4 base)", "~0.52", "0", "0", "~0.52 + tiny adapter"]],
      ["method", "weights", "grads", "AdamW states", "total B/param"])
note("The PyTorch scripts load the model in bfloat16 and use optim='adamw_torch', whose states take the parameter "
     "dtype. Recipes that keep fp32 master weights need 16 B/param — twice as much.")

step(3, f"the recipes: state + activations (+ {HEADROOM_GB} GB headroom)")
rows, chart = [], []
for label, name, method, mb, seq, ckpt, src in RECIPES:
    w, go, trainable = state_gb(name, method)
    worst = activations_gb(name, mb * seq, ckpt)
    typical = activations_gb(name, mb * min(seq, ALPACA_TOKENS), ckpt) if "PyTorch" in label else worst
    need = w + go + typical + HEADROOM_GB
    if need <= ONE:
        where = "1 Spark"
    elif "FSDP" not in label:                                 # a single-node script: it has to fit one Spark
        where = "✕ over 128 GB"
    elif (w + go) / 2 + typical + HEADROOM_GB <= ONE:        # FSDP halves the state; activations stay per Spark
        where = "2 Sparks (FSDP)"
    else:
        where = "✕ too big"
    rows.append([f"{label} · {name}", f"{w:6.1f}", f"{go:6.1f}", f"{typical:6.1f}", f"{worst:6.1f}",
                 f"{need:6.1f} GB", where])
    chart.append((f"{label} · {name}", need))
table(rows, ["recipe", "weights", "grads+opt", "acts (typ)", "acts (cap)", "total (typ)", "fits on"])
note(f"'acts (typ)' assumes Alpaca samples of ≤{ALPACA_TOKENS} tokens (padding goes to the longest in the batch); "
     "'acts (cap)' fills every sequence to the cap. NeMo packs sequences to exactly 1024 tokens, so typ = cap.")
q_lin, q_emb = params("Qwen3 8B")
note(f"The NeMo rows assume the same 8 B/param as the PyTorch scripts; the recipe YAML decides. At 16 B/param "
     f"(fp32 master + states) Qwen3 8B full SFT would need ~{(q_lin + q_emb) * 16 / GB + 5.8 + HEADROOM_GB:.0f} GB, "
     "so a recipe named *_spark.yaml must be leaner than that. Lab 03 shows what it really used.")
for label, need in chart:
    print(f"│ {label:36s} {bar(min(need, 2 * ONE), 2 * ONE)} {need:6.1f} GB")
print(f"│ {'one Spark':36s} {bar(ONE, 2 * ONE)} {ONE:6.1f} GB")

step(4, "what checkpointing and micro-batch size do to activations (Llama 3.1 70B QLoRA)")
for mb, seq, ck in [(8, 2048, False), (8, ALPACA_TOKENS, False), (8, 2048, True), (1, 2048, True)]:
    a = activations_gb("Llama 3.1 70B", mb * seq, ck)
    print(f"│ batch {mb} × {seq:4d} tokens · checkpointing {'on ' if ck else 'off'}  {bar(min(a, 400), 400)} {a:6.1f} GB")
note("The QLoRA script leaves --gradient_checkpointing off. That works for short Alpaca samples (NVIDIA's benchmark "
     "guide runs the defaults on one Spark), but not for your own long data: at 8 × 2048 tokens the activations alone "
     "exceed 128 GB. Turn checkpointing on, or lower --batch_size, before you change the dataset.")

step(5, "two Sparks with FSDP: memory per Spark, and what crosses the cable per forward + backward pass")
rows = []
for label, name, method, cfg in [("LoRA (FSDP FULL_SHARD)", "Llama 3.1 70B", "lora", "config_fsdp_lora.yaml"),
                                 ("LoRA (FSDP FULL_SHARD)", "Llama 3.1 8B", "lora", "config_fsdp_lora.yaml"),
                                 ("full SFT (FSDP2)", "Llama 3.2 3B", "full", "config_finetuning.yaml")]:
    w, go, _ = state_gb(name, method)
    per_spark = (w + go) / 2
    if method == "lora":        # weights gathered in forward AND again in backward (reshard after forward)
        moved = 2 * w * 0.5
    else:                       # FSDP2, reshard_after_forward false: one gather + one reduce-scatter of grads
        moved = w * 0.5 + w * 0.5
    rows.append([f"{name} · {label}", cfg, f"{w + go:6.1f} GB", f"{per_spark:6.1f} GB", f"{moved:6.1f} GB",
                 f"{moved / LINK_GBS:5.2f} s"])
table(rows, ["job", "Accelerate config", "state, 1 Spark", "state per Spark", "link per pass", f"at {LINK_GBS} GB/s"])
note("FSDP sends weight shards on EVERY forward/backward pass, and gradient accumulation does not reduce it. "
     "Only more tokens per pass (bigger micro-batch) amortises it. The LoRA gradients themselves are tiny.")

result("Two Sparks help training when the job does not fit one Spark at the precision you want — Llama 3.1 70B "
       "LoRA in bf16 is the playbook's case. When it fits (8B LoRA, 70B QLoRA), one Spark avoids the link cost; "
       "two only pay off for speed when each pass computes for much longer than its link time above.")
