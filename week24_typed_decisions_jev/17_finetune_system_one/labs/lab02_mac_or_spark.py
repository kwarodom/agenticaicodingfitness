#!/usr/bin/env python3
"""Lab 17-2 · Mac or DGX Spark? Measure before you decide.

These are ~300M-parameter ENCODERS, not 8B LLMs. That changes the answer, so we measure:
  1. what this machine is, and which device each trainer will actually use on it
  2. memory: what a full fine-tune and a LoRA fine-tune need (counted from the weight files)
  3. speed: one real training pass of each model on 48 tickets, timed here
  4. projections for 456 / 5,000 / 50,000 tickets, next to what is known about a DGX Spark
  5. the decision rules

Needs week24_typed_decisions_jev/.venv-local (Lab 16-0) to measure; otherwise replays the 2026-10-17 measurement on an
Apple M4 Max (labelled RECORDED). The timed pass takes ~40 s.
Run: .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab02_mac_or_spark.py
"""
import glob
import json
import os
import platform
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, step, table  # noqa: E402

banner("Lab 17-2 · Mac or DGX Spark?", "measure memory and speed here, then decide")
F.L.banner_local()
rec = F.Recorder("lab02_speed")

# ── 1 · this machine ──────────────────────────────────────────────────────────
step(1, "this machine, and the device each trainer picks")
live = bool(F.L.where())
if live:
    import torch
    devs = ["cuda" if torch.cuda.is_available() else None,
            "mps" if torch.backends.mps.is_available() else None, "cpu"]
    devs = [d for d in devs if d]
    laya_dev = "cuda" if "cuda" in devs else ("mps" if "mps" in devs else "cpu")
    gl_dev = "cuda" if "cuda" in devs else "cpu"             # gliner2 ExtractorTrainer: CUDA or CPU, nothing else
    me = {"chip": F.gpu_name(), "mem_gb": round(F.mem_gb() or 0, 1), "torch": torch.__version__,
          "devices": devs, "laya_dev": laya_dev, "gl_dev": gl_dev, "os": f"{platform.system()} {platform.machine()}"}
    rec.data["machine"] = me
else:
    me = rec.data["machine"]
    print("» no week24_typed_decisions_jev/.venv-local here — showing the machine this lab was recorded on (RECORDED)")
table([["chip", me["chip"]], ["memory (unified)", f"{me['mem_gb']:.0f} GiB"], ["torch devices", ", ".join(me["devices"])],
       ["Laya trainer uses", f"{me['laya_dev']}  (order: cuda → mps → cpu)"],
       ["GLiNER2 trainer uses", f"{me['gl_dev']}  (cuda, else cpu — never mps)"],
       ["mixed precision", "CUDA only — on a Mac both train in fp32"]],
      ["", "this machine"])

# ── 2 · memory ────────────────────────────────────────────────────────────────
step(2, "memory — counted from the weight files, not guessed")


def params_in(pattern: str) -> int | None:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    files = glob.glob(os.path.join(hub, pattern), recursive=True)
    if not files:
        return None
    with open(files[0], "rb") as f:                           # a safetensors header lists every tensor's shape
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    total = 0
    for k, v in header.items():
        if k != "__metadata__":
            n = 1
            for d in v["shape"]:
                n *= d
            total += n
    return total


P = {"Laya multilingual": params_in("models--convaiinnovations--laya/snapshots/*/multilingual/model.safetensors")
     or rec.data.get("params", {}).get("Laya multilingual"),
     "GLiNER2.5-multi-Decide": params_in("models--fastino--GLiNER2.5-multi-Decide/snapshots/*/model.safetensors")
     or rec.data.get("params", {}).get("GLiNER2.5-multi-Decide")}
rec.data["params"] = P
LORA_TRAINABLE = 1_357_832                                    # printed by ExtractorTrainer for r=8 (Lab 17-5)
rows = []
for name, n in P.items():
    full = n * 16 / 1e9                                      # fp32 weights 4 + grads 4 + Adam moments 8 bytes
    lora = (n * 4 + LORA_TRAINABLE * 16) / 1e9
    rows.append([name, f"{n / 1e6:.0f} M", f"{n * 4 / 1e9:.1f} GB", f"≈ {full:.1f} GB", f"≈ {lora:.1f} GB" if "GLiNER" in name else "—"])
table(rows, ["model", "parameters", "weights fp32", "full fine-tune state", "LoRA r=8 state"])
print("» Add activations (a few GB at batch 8, 512 tokens). Even the full fine-tune fits a 16 GB Mac;")
print("  an 8B LLM's full fine-tune state is ~130 GB — that is the job a 128 GB Spark exists for.")

# ── 3 · speed: one real pass ──────────────────────────────────────────────────
step(3, "speed — one training pass of each model on 48 tickets, timed on this machine")
sample = F.split("train")[::9][:48]
speed = rec.data.get("speed", {})
if live:
    import contextlib
    import io
    import warnings
    warnings.filterwarnings("ignore")
    from laya.train import TrainConfig, items_from_rows, load_checkpoint, train_model
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model, tok, cfg = load_checkpoint(F.LAYA_BASE)
        items, _ = items_from_rows(tok, F.to_laya_rows(sample), cfg.get("max_len", 512), cfg.get("head_max_len", 192))
        dev = torch.device(me["laya_dev"])
        t0 = time.perf_counter()
        train_model(model, tok, items, TrainConfig(epochs=1, micro_batch=8, grad_accum=1, log_every=10 ** 9), dev,
                    cfg.get("max_len", 512), cfg.get("head_max_len", 192))
        if dev.type == "mps":
            torch.mps.synchronize()
        laya_s = time.perf_counter() - t0
    del model
    print(f"   ✓ Laya multilingual · full fine-tune · {me['laya_dev']} · {len(items)} items (48 tickets × 3 questions) "
          f"in {laya_s:.1f} s")
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    import logging
    logging.disable(logging.INFO)
    gl = {}
    for mode in ("lora", "full"):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            m = AutoExtractor.from_pretrained(F.GLINER_BASE)
            tr = ExtractorTrainer(m, TrainingConfig(
                output_dir=str(F.RUNS / "speedtest"), num_epochs=1, batch_size=8, eval_strategy="no", save_best=False,
                num_workers=0, fp16=False, bf16=False, use_lora=mode == "lora", lora_r=8, lora_alpha=16,
                logging_steps=1000, report_to_wandb=False))
            t0 = time.perf_counter()
            tr.train(train_data=F.to_gliner_records(sample))
            gl[mode] = time.perf_counter() - t0
        print(f"   ✓ GLiNER2.5-multi-Decide · {mode:<4} · {me['gl_dev']} · 48 tickets in {gl[mode]:.1f} s")
    import shutil
    shutil.rmtree(F.RUNS / "speedtest", ignore_errors=True)
    speed = {"laya_items_per_s": len(items) / laya_s, "gl_lora_tickets_per_s": 48 / gl["lora"],
             "gl_full_tickets_per_s": 48 / gl["full"], "measured": time.strftime("%Y-%m-%d"), "on": me["chip"]}
    if os.environ.get("JEV_RECORD") == "1":
        rec.data["speed"] = speed
        rec.dirty = True
    src = "LOCAL"
else:
    src = f"RECORDED {speed.get('measured', '')} on {speed.get('on', '?')}"
table([["Laya multilingual · full", me["laya_dev"], f"{speed['laya_items_per_s'] / 3:.1f} tickets/s",
        f"{speed['laya_items_per_s']:.1f} question-items/s"],
       ["GLiNER2.5-multi-Decide · LoRA r=8", me["gl_dev"], f"{speed['gl_lora_tickets_per_s']:.1f} tickets/s", "3 tasks per ticket"],
       ["GLiNER2.5-multi-Decide · full", me["gl_dev"], f"{speed['gl_full_tickets_per_s']:.1f} tickets/s", "3 tasks per ticket"]],
      ["training", "device", "throughput", "note"])
print(f"» source: {src}")

# Short passes include warm-up (first MPS kernels, dataset preparation). If the real training runs
# from Labs 17-4/17-5 exist, project from those instead — they include evaluation and saving too.
full = {}
for name, path in (("laya", F.RECORDED / "lab04_laya.json"), ("gl", F.RECORDED / "lab05_gliner.json")):
    if path.is_file():
        full[name] = json.loads(path.read_text(encoding="utf-8"))
rate = dict(speed)
if full.get("laya", {}).get("seconds"):
    lr = full["laya"]
    rate["laya_items_per_s"] = lr["train_items"] * 3 / lr["seconds"]
for mode in ("lora", "full"):
    if full.get("gl", {}).get(mode, {}).get("tickets_per_s"):
        rate[f"gl_{mode}_tickets_per_s"] = full["gl"][mode]["tickets_per_s"]
if rate != speed:
    print("» whole training runs (Labs 17-4/17-5, incl. evaluation + saving): "
          f"Laya {rate['laya_items_per_s'] / 3:.1f} · GLiNER LoRA {rate['gl_lora_tickets_per_s']:.1f} · "
          f"GLiNER full {rate['gl_full_tickets_per_s']:.1f} tickets/s — the projections below use these")
speed = rate
EP = {m: len(full.get("gl", {}).get(m, {}).get("eval_history") or []) or n for m, n in (("lora", 8), ("full", 4))}

# ── 4 · projections ───────────────────────────────────────────────────────────
step(4, "how long would it take? (this machine, measured rate × size × epochs; + ~10 s load/save)")


def hm(s: float) -> str:
    return f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


spark = json.loads((F.RECORDED / "lab08_spark.json").read_text(encoding="utf-8")) if (F.RECORDED / "lab08_spark.json").is_file() else {}
rows = []
for n in (456, 5_000, 50_000):
    rows.append([f"{n:,} tickets",
                 hm(n * 3 * 3 / speed["laya_items_per_s"]),                 # 3 questions, 3 epochs
                 hm(n * EP["lora"] / speed["gl_lora_tickets_per_s"]),        # epochs the real run used
                 hm(n * EP["full"] / speed["gl_full_tickets_per_s"]),
                 hm(n * 3 * 3 / spark["laya_items_per_s"]) if spark.get("laya_items_per_s") else "not measured yet"])
table(rows, ["dataset", "Laya full · 3 epochs", f"GLiNER LoRA · {EP['lora']} epochs", f"GLiNER full · {EP['full']} epochs",
             "DGX Spark (Laya)"])
print("» DGX Spark facts we can stand behind (NVIDIA spec, see week25_dgx_spark Module 01): GB10 Grace Blackwell,")
print("  128 GB unified LPDDR5x, 273 GB/s, CUDA — so both trainers get the GPU and bf16 autocast there.")
print("  How much FASTER it is for these models is not on any spec sheet: Lab 17-8 measures it.")

# ── 5 · decide ────────────────────────────────────────────────────────────────
step(5, "the decision")
table([["first fine-tune, < ~5k tickets, one model", "Mac", "minutes; same machine you serve from; data never leaves"],
       ["GLiNER2 on > ~5k tickets, or many runs", "Spark", "its trainer gets CUDA there; on a Mac it is CPU-only"],
       ["hyper-parameter sweeps, 3+ seeds, nightly re-training", "Spark", "unattended, bf16, leaves the laptop free"],
       ["the data may not leave the building", "either", "both are on-prem — never a reason to pick the cloud"],
       ["an LLM teacher or 8B+ model in the same pipeline", "Spark", "128 GB holds what 36 GB cannot (week 25)"],
       ["you only need better confidence, not new skills", "neither", "re-fit temperatures/thresholds on val (Lab 17-6)"]],
      ["situation", "train on", "why"])
print("» For this module's 456 tickets the Mac is the right tool. The Spark earns its place when the")
print("  data or the number of runs grows — and Lab 17-8 shows the same code running there unchanged.")
rec.save()
