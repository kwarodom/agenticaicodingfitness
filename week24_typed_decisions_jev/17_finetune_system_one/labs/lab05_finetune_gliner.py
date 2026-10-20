#!/usr/bin/env python3
"""Lab 17-5 · Fine-tune GLiNER2.5-multi-Decide two ways: a 5 MB LoRA adapter vs a full fine-tune.

Two ways to run it:
  --train lora | --train full   from the ⌨ terminal (CPU on a Mac: ≈ 6–9 min each), saved under
                                17_finetune_system_one/.runs/gliner-lora (≈ 5 MB) or .runs/gliner-full (≈ 1.1 GB)
  (no flag)                     the ▶ Run button: reads your runs, or the 2026-10-17 runs (RECORDED)

What it shows:
  1. the two recipes side by side, and what LoRA actually trains
  2. the runs: time, trainable parameters, validation loss per epoch, which epoch was kept ("best")
  3. size on disk — and why that matters when you serve five customers from one base model
  4. accuracy on validation: base vs LoRA vs full
  5. one base, many adapters: switch the adapter off and the base model comes back unchanged

Run:  week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train lora
      week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train full
      .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py
"""
import contextlib
import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, step, table  # noqa: E402

MODE = sys.argv[sys.argv.index("--train") + 1] if "--train" in sys.argv else None
if MODE not in (None, "lora", "full"):
    raise SystemExit("usage: lab05_finetune_gliner.py [--train lora|full]")
banner("Lab 17-5 · fine-tune GLiNER2.5-multi-Decide", "LoRA adapter vs full fine-tune, CPU on a Mac"
       + (f" · TRAINING {MODE.upper()} NOW" if MODE else ""))
F.L.banner_local()
rec = F.Recorder("lab05_gliner")

# ── 1 · two recipes ───────────────────────────────────────────────────────────
step(1, "two recipes")
RECIPES = {
    "lora": dict(num_epochs=8, batch_size=8, encoder_lr=1e-5, task_lr=5e-4, use_lora=True, lora_r=8, lora_alpha=16.0,
                 lora_dropout=0.0, save_adapter_only=True),
    "full": dict(num_epochs=4, batch_size=8, encoder_lr=1e-5, task_lr=5e-4, use_lora=False),
}
COMMON = dict(eval_strategy="epoch", save_best=True, early_stopping=True, early_stopping_patience=2, metric_for_best="eval_loss", greater_is_better=False,
              save_total_limit=1, warmup_ratio=0.1, scheduler_type="linear", num_workers=0, fp16=False, bf16=False,
              logging_steps=20, seed=42, report_to_wandb=False)
table([["what trains", "rank-8 matrices beside every encoder layer + heads", "all 287 M weights"],
       ["epochs", "8 (few weights move: needs more passes)", "4 (every weight moves: overfits sooner)"],
       ["learning rate", "5e-4 for the adapter and task heads", "encoder 1e-5 · task heads 5e-4"],
       ["kept checkpoint", "lowest validation loss (save_best)", "lowest validation loss (save_best)"],
       ["early stopping", "stop after 2 epochs without a new best", "same"],
       ["precision", "fp32 on CPU · bf16 on CUDA", "fp32 on CPU · bf16 on CUDA"]],
      ["", "LoRA (r = 8, α = 16)", "full fine-tune"])
print("» LoRA: W stays frozen; training learns a tiny B·A (8 columns wide) added on top. GLiNER2's recipe")
print("  table suggests rank 4 below ~1k examples and 8 for 1k–10k; 456 tickets × 3 tasks ≈ 1.4k → 8.")

# ── 2 · train (or read the runs) ──────────────────────────────────────────────
step(2, "the runs")


def train(mode: str) -> dict:
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    out = F.CKPT[f"gliner_{mode}"]
    shutil.rmtree(out, ignore_errors=True)
    model = AutoExtractor.from_pretrained(F.GLINER_BASE)
    logging.getLogger("gliner2").setLevel(logging.WARNING)
    import torch
    precision = dict(fp16=False, bf16=torch.cuda.is_available())   # bf16 on CUDA (a Spark), fp32 on a Mac's CPU
    cfg = TrainingConfig(output_dir=str(out), **dict(COMMON, **precision), **RECIPES[mode])
    trainer = ExtractorTrainer(model, cfg)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"   training {mode}: {trainable:,} of {total:,} weights trainable ({trainable / total:.2%})", flush=True)
    t0 = time.perf_counter()
    res = trainer.train(train_data=str(F.DATA / "gliner_train.jsonl"), eval_data=str(F.DATA / "gliner_val.jsonl"))
    secs = time.perf_counter() - t0
    hist = [{"epoch": i, "eval_loss": round(m["eval_loss"], 4)}
            for i, m in enumerate((m for m in trainer.eval_metrics_history if "eval_loss" in m), 1)]
    for d in out.glob("checkpoint-*"):                       # keep only best/ and final/
        shutil.rmtree(d, ignore_errors=True)
    kept = F.gl_dir(f"gliner_{mode}")
    size = sum(f.stat().st_size for f in kept.rglob("*") if f.is_file())
    run = {"mode": mode, "seconds": round(secs, 1), "trainable": trainable, "total": total,
           "eval_history": hist, "best_metric": res.get("best_metric"), "kept": kept.name, "bytes": size,
           "tickets_per_s": round(456 * max(1, len(hist)) / secs, 2),        # epochs actually run
           "device": "cuda" if __import__("torch").cuda.is_available() else "cpu", "chip": F.gpu_name(),
           "trained": time.strftime("%Y-%m-%d %H:%M")}
    (out / "lab05_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    return run


runs = {}
if MODE:
    if not F.L.where():
        raise SystemExit("✗ --train needs week24_typed_decisions_jev/.venv-local (Module 16, step 0)")
    with contextlib.redirect_stderr(io.StringIO()):
        runs[MODE] = train(MODE)
    if os.environ.get("JEV_RECORD") == "1" or MODE not in rec.data:
        rec.data[MODE], rec.dirty = runs[MODE], True
        os.environ["JEV_RECORD"] = "1"
        rec.save()
for mode in ("lora", "full"):
    if mode in runs:
        continue
    mine = F.CKPT[f"gliner_{mode}"] / "lab05_run.json"
    if mine.is_file():
        runs[mode] = dict(json.loads(mine.read_text()), _src="LOCAL (your run)")
    elif mode in rec.data:
        runs[mode] = dict(rec.data[mode], _src=f"RECORDED {rec.data[mode]['trained']}")
rows = []
for mode, r in runs.items():
    curve = " → ".join(f"{h['eval_loss']:.3f}" for h in r["eval_history"])
    rows.append([mode, f"{r['trainable'] / 1e6:.2f} M ({r['trainable'] / r['total']:.1%})", r["device"],
                 f"{r['seconds'] / 60:.1f} min", f"{r['tickets_per_s']:.1f}/s", curve, r.get("_src", "LOCAL (just trained)")])
table(rows, ["run", "trainable", "device", "time", "tickets", "validation loss per epoch", "source"])
if not runs:
    print("» no runs yet — train with --train lora and --train full from the ⌨ terminal")
for mode, r in runs.items():
    h = r["eval_history"]
    if h:
        best = min(h, key=lambda x: x["eval_loss"])
        print(f"» {mode}: lowest validation loss at epoch {best['epoch']} of {len(h)} → that checkpoint is kept as '{r['kept']}/'"
              + ("  (later epochs were already overfitting)" if best["epoch"] < len(h) else ""))
        planned = RECIPES[mode]["num_epochs"]
        if len(h) < planned:
            print(f"   early stopping ended it after epoch {len(h)} of {planned}: two epochs in a row without a new best")

# ── 3 · size ──────────────────────────────────────────────────────────────────
step(3, "size on disk")
if runs:
    table([[m, f"{r['bytes'] / 1e6:,.1f} MB", "needs the 1.1 GB base next to it" if m == "lora" else "stands alone"]
           for m, r in runs.items()], ["run", "kept checkpoint", ""])
    if "lora" in runs:
        print(f"» Five hotels, five vocabularies: 1 base + 5 adapters ≈ {1100 + 5 * runs['lora']['bytes'] / 1e6:,.0f} MB."
              " Five full fine-tunes ≈ 5,500 MB.")

# ── 4 · validation accuracy ───────────────────────────────────────────────────
step(4, "validation accuracy: base vs LoRA vs full (Lab 17-6 is the real exam)")
val = F.split("val")
rows = []
for eng in ("gliner_base", "gliner_lora", "gliner_full"):
    p, src = F.predict(eng, val)
    if p:
        rows.append(F.summary_row(eng, val, p, src))
table(rows, F.SUMMARY_HEADERS)

# ── 5 · adapters switch off ──────────────────────────────────────────────────
step(5, "one base, many adapters: switch the adapter off and the base model is back")
probe = "มีควันออกจากตู้ไฟใกล้ลิฟต์ชั้น 9"
switch, src = None, ""
if F.available("gliner_lora"):
    pm = F._load("gliner_lora")
    tasks = {"queue": F.L.to_gliner_schema(F.Q)["queue"]}
    on = pm.classify_text(probe, tasks, include_confidence=True)["queue"]
    with pm.disable_adapter():
        off = pm.classify_text(probe, tasks, include_confidence=True)["queue"]
    switch, src = {"on": on, "off": off}, "LOCAL"
    if os.environ.get("JEV_RECORD") == "1":
        rec.data["switch"], rec.dirty = switch, True
        rec.save()
elif "switch" in rec.data:
    switch, src = rec.data["switch"], "RECORDED"
if switch:
    print(f"   \"{probe}\"  (smoke from the electrical cabinet near the lift) · {src}")
    print(f"   adapter ON  → {switch['on']['label']:<12} {switch['on']['confidence']:.2f}")
    print(f"   adapter OFF → {switch['off']['label']:<12} {switch['off']['confidence']:.2f}   (the untouched base model)")
print("» the code: pm = PeftModel.from_pretrained(base, path) … with pm.disable_adapter(): …")
print("» GLiNER2.5 boundary checkpoints load adapters through PEFT (the old .load_adapter() is span-only).")
