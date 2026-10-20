#!/usr/bin/env python3
"""Lab 11-3 · Watch the job: tail the training log on Spark A, parse it, and check the machine.

Read-only. Finds the newest ~/w25/logs/m11_*.log that lab 02 started, reads its last 400 lines,
and parses what the trainers print:

  • Hugging Face / TRL (the PyTorch scripts): one dict per logging step — {'loss': …, 'grad_norm': …,
    'learning_rate': …, 'epoch': …} — and the scripts' own "TRAINING COMPLETED" block.
  • NeMo AutoModel: lines with `step N` and `loss X` (the exact layout changes between releases).

It draws the loss curve, flags NaN or a rising loss, and shows memory (`free -g`), GPU use and the
container state. In DRY mode it parses an EXAMPLE log (illustrative numbers, not a Spark run) —
so the parser itself is tested on every machine.

Run: .venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab03_watch_log.py
"""
import ast
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, bar, check, note, result, sh, step, table, warn  # noqa: E402

# EXAMPLE — the SHAPE of a Llama3_8B_LoRA_finetuning.py log (dataset_size 100, batch 2 → 50 steps).
# The header and summary lines follow the script's print() calls; every number is illustrative.
EX_LOG = """
============================================================
LLAMA 3.1 8B LoRA FINE-TUNING CONFIGURATION
============================================================
Model: meta-llama/Llama-3.1-8B-Instruct
Batch size: 2
Sequence length: 2048
Number of epochs: 1
Learning rate: 0.0001
LoRA rank: 8
Dataset size: 100
============================================================

Loading model: meta-llama/Llama-3.1-8B-Instruct
Trainable parameters = 20,971,520
Loading dataset with 100 samples...
Compiling model with torch.compile()...
Running warmup for torch.compile()...
{'loss': 1.7102, 'grad_norm': 1.0412, 'learning_rate': 0.0001, 'epoch': 0.02}
{'loss': 1.6627, 'grad_norm': 0.9876, 'learning_rate': 6.667e-05, 'epoch': 0.04}
{'loss': 1.5931, 'grad_norm': 0.9713, 'learning_rate': 3.333e-05, 'epoch': 0.06}
{'train_runtime': 41.2, 'train_samples_per_second': 0.146, 'train_steps_per_second': 0.073, 'train_loss': 1.6553, 'epoch': 0.06}

Starting LoRA fine-tuning for 1 epoch(s)...
{'loss': 1.6801, 'grad_norm': 1.0225, 'learning_rate': 0.0001, 'epoch': 0.02}
{'loss': 1.6135, 'grad_norm': 0.9510, 'learning_rate': 9.8e-05, 'epoch': 0.04}
{'loss': 1.5420, 'grad_norm': 0.8844, 'learning_rate': 9.6e-05, 'epoch': 0.06}
{'loss': 1.4987, 'grad_norm': 0.8391, 'learning_rate': 9.4e-05, 'epoch': 0.08}
{'loss': 1.4305, 'grad_norm': 0.7908, 'learning_rate': 9.2e-05, 'epoch': 0.1}
{'loss': 1.3912, 'grad_norm': 0.7650, 'learning_rate': 9e-05, 'epoch': 0.12}
{'loss': 1.3570, 'grad_norm': 0.7221, 'learning_rate': 8.8e-05, 'epoch': 0.14}
{'loss': 1.3398, 'grad_norm': 0.7010, 'learning_rate': 8.6e-05, 'epoch': 0.16}
{'loss': 1.3104, 'grad_norm': 0.6893, 'learning_rate': 8.4e-05, 'epoch': 0.18}
{'loss': 1.2877, 'grad_norm': 0.6702, 'learning_rate': 8.2e-05, 'epoch': 0.2}
[EXAMPLE trimmed: steps 11–47 left out]
{'loss': 1.1932, 'grad_norm': 0.6015, 'learning_rate': 6e-06, 'epoch': 0.96}
{'loss': 1.2011, 'grad_norm': 0.6127, 'learning_rate': 4e-06, 'epoch': 0.98}
{'loss': 1.1876, 'grad_norm': 0.5980, 'learning_rate': 2e-06, 'epoch': 1.0}
{'train_runtime': 180.0, 'train_samples_per_second': 0.556, 'train_steps_per_second': 0.278, 'train_loss': 1.2764, 'epoch': 1.0}

============================================================
TRAINING COMPLETED
============================================================
Training runtime: 180.00 seconds
Samples per second: 0.56
Steps per second: 0.28
Train loss: 1.2764
============================================================
"""
# EXAMPLE — a NeMo AutoModel-style line. The layout is NOT from the playbook; check it against your own log.
EX_NEMO = """step 1 | epoch 0 | loss 1.9012 | grad_norm 4.1250 | lr 1.00e-05
step 2 | epoch 0 | loss 1.7730 | grad_norm 3.5625 | lr 1.00e-05
step 3 | epoch 0 | loss 1.6214 | grad_norm 2.9375 | lr 1.00e-05"""

EX_FREE = """               total        used        free      shared  buff/cache   available
Mem:             119          52          59           0           8          66"""
EX_SMI = "96 %, 61, 88.40 W"
EX_PS = "w25-m11-pytorch-lora-8b Up 4 minutes"


def parse_log(text: str) -> dict:
    """→ {'steps': [dict per logged step], 'phases': [[…], […]], 'summary': {…}, 'trainable': int|None}."""
    phases, cur, summary, trainable, done = [], [], {}, None, False
    for raw in re.split(r"[\r\n]+", text):                 # tqdm bars rewrite the line with \r
        line = raw.strip()
        m = re.search(r"Trainable parameters[ =:]+([\d,]+)", line)
        if m:
            trainable = int(m.group(1).replace(",", ""))
        if line.startswith("Starting ") and cur:            # the scripts' warmup pass ends here
            phases.append(cur)
            cur = []
        if line.startswith("{") and line.endswith("}"):
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            d = {k: _num(v) for k, v in d.items()}
            if "train_runtime" in d:
                summary = d
            elif "loss" in d:
                cur.append(d)
            continue
        m = re.search(r"\bstep[\s:=]+(\d+)\b.*?\bloss[\s:=]+([-\d.eE+naNinf]+)", line)
        if m:
            cur.append({"step": int(m.group(1)), "loss": _num(m.group(2))})
        if "TRAINING COMPLETED" in line:
            done = True
    if cur:
        phases.append(cur)
    return {"phases": phases, "steps": phases[-1] if phases else [], "summary": summary,
            "trainable": trainable, "completed": done}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


banner("Lab 11-3 · watch the fine-tuning job", "tail + parse the log · memory · GPU · container state")

step(1, "which job, and is it still running?")
ps = sh("docker ps -a --filter name=w25-m11 --format '{{.Names}} {{.Status}}'", timeout=30, example=EX_PS)
lg = sh("ls -t ~/w25/logs/m11_*.log 2>/dev/null | head -1", timeout=30,
        example="/home/nvidia/w25/logs/m11_pytorch-lora-8b.log")
path = lg.out.strip().splitlines()[-1] if lg.out.strip() else ""
if lg.live and not path:
    warn("no ~/w25/logs/m11_*.log on Spark A yet — start a job with lab 02 first.")
    sys.exit(0)

step(2, "the last 400 lines of the log")
tail = sh(f"tail -n 400 {path}", timeout=30, example=EX_LOG, quiet=True)
print(f"◆ read {len(tail.out.splitlines())} lines from {path} ({'live' if tail.live else 'EXAMPLE text'})")
for ln in [x for x in tail.out.splitlines() if x.strip()][-6:]:
    print(f"  {ln[:140]}")
if re.search(r"out of memory|CUDA error|Traceback|Killed", tail.out, re.I):
    warn("the log shows an error (out of memory / Traceback / Killed). Read the lines above it; lab 01 shows "
         "what to shrink (batch size, sequence length, --gradient_checkpointing).")
if re.search(r"gated repo|401 Client Error|Cannot access gated", tail.out, re.I):
    warn("gated model: accept the licence on huggingface.co and run `hf auth login` on the Spark.")

step(3, "parse it")
p = parse_log(tail.out)
steps = p["steps"]
if p["trainable"]:
    print(f"◆ trainable parameters: {p['trainable']:,}"
          + ("  (= lab 01's count for Llama 3.1 8B at r=8)" if p["trainable"] == 20_971_520 else ""))
if len(p["phases"]) > 1:
    print(f"◆ {len(p['phases']) - 1} warmup pass(es) for torch.compile() skipped ({len(p['phases'][0])} steps) — "
          "the scripts train once to compile, then again for real")
if steps:
    losses = [s["loss"] for s in steps if isinstance(s.get("loss"), float)]
    lo, hi = min(losses), max(losses)
    shown = steps if len(steps) <= 14 else steps[:7] + steps[-7:]
    for s_ in shown:
        where = f"epoch {s_['epoch']:.2f}" if "epoch" in s_ else f"step {s_.get('step', '?')}"
        print(f"│ {where:11s} loss {s_['loss']:7.4f}  {bar(s_['loss'] - lo * 0.9, hi - lo * 0.9)}")
    nan = any(math.isnan(x) or math.isinf(x) for x in losses)
    k = max(1, len(losses) // 5)
    first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
    check(not nan and last < first,
                f"loss is finite and falling: first {k} avg {first:.3f} → last {k} avg {last:.3f} "
                f"({(first - last) / first:.0%} lower)",
                "loss is NaN/inf or not falling — lower --learning_rate, check the data, or look for OOM retries")
else:
    warn("no training steps logged yet — the model may still be downloading or compiling.")
if p["summary"]:
    sm = p["summary"]
    table([[f"{sm.get('train_runtime', 0):.1f} s", f"{sm.get('train_samples_per_second', 0):.3f}",
            f"{sm.get('train_steps_per_second', 0):.3f}", f"{sm.get('train_loss', 0):.4f}"]],
          ["runtime", "samples/s", "steps/s", "train loss"])
print(f"◆ finished: {'yes — TRAINING COMPLETED' if p['completed'] else 'not yet (or a NeMo job)'}")

step(4, "the machine while it trains")
sh("free -g | head -2", timeout=30, example=EX_FREE)
sh("nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader", timeout=30,
   example=EX_SMI)
note("On a Spark the GPU and CPU share one 128 GB pool: `free -g` is the honest memory meter (Module 01).")

step(5, "parser self-test — runs on every machine")
t = parse_log(EX_LOG)
check(len(t["phases"]) == 2 and len(t["steps"]) == 13 and t["trainable"] == 20_971_520 and t["completed"]
      and abs(t["summary"]["train_loss"] - 1.2764) < 1e-9,
      "HF/TRL format: 3 warmup + 13 training steps, trainable count, summary and TRAINING COMPLETED all found",
      "the HF/TRL parser missed something in the EXAMPLE log — parser bug")
n = parse_log(EX_NEMO)
check([s_["step"] for s_ in n["steps"]] == [1, 2, 3] and n["steps"][-1]["loss"] == 1.6214,
      "NeMo-style `step N | loss X` lines parsed (3 steps)", "the NeMo-style parser failed — parser bug")

print()
if tail.live:
    result("that is your run. Save the final train loss — Module 13 compares the fine-tune with the base model.")
else:
    result("DRY: the log above is an EXAMPLE (illustrative numbers, not a Spark run). Start lab 02 on your Spark, "
           "then run this lab again.")
