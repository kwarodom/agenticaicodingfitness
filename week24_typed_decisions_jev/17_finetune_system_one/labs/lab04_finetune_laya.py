#!/usr/bin/env python3
"""Lab 17-4 · Fine-tune Laya on your tickets (full fine-tune, Apple GPU or CUDA).

Two ways to run it:
  --train   from the ⌨ terminal (≈ 4 min on an M4 Max, longer than the runner's 150 s ▶ limit):
            laya.train.finetune(...) on data/laya_train.jsonl, judged on data/laya_val.jsonl,
            saved to 17_finetune_system_one/.runs/laya-ft (≈ 1.3 GB, gitignored)
  (no flag) the ▶ Run button: reads the checkpoint you trained, or the 2026-10-17 run (RECORDED)

What it shows:
  1. the recipe — every knob, and why it is set that way for 456 tickets
  2. the training run: loss per epoch, the optimizer-update budget, Laya's own warnings
  3. Laya's before/after table on VALIDATION, and why the in-sample version of it lies
  4. the generalisation gap: train vs val vs held-out with the same checkpoint
  5. what training changed in the confidence: fitted temperatures and abstention cuts

Run:  week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py --train
      .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py
"""
import contextlib
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, step, table  # noqa: E402

TRAIN = "--train" in sys.argv
banner("Lab 17-4 · fine-tune Laya", "full fine-tune of the multilingual checkpoint on 456 tickets"
       + (" · TRAINING NOW" if TRAIN else ""))
F.L.banner_local()
rec = F.Recorder("lab04_laya")
OUT = F.CKPT["laya_ft"]

# ── 1 · the recipe ────────────────────────────────────────────────────────────
step(1, "the recipe — every knob, and why")
RECIPE = dict(epochs=3, micro_batch=8, grad_accum=2, encoder_lr=2.5e-5, head_lr=1e-4, loss="rlcd",
              shuffle_options=("choice",), calib_frac=0.12, seed=0, target_error=0.05)
table([["base", F.LAYA_BASE, "one checkpoint for EN + TH (the Router would need two fine-tunes)"],
       ["epochs", RECIPE["epochs"], "small data: 3 passes; more starts memorising the 72 situations"],
       ["batch", f"{RECIPE['micro_batch']} × {RECIPE['grad_accum']} = 16", "Laya's default 64 gives only ~22 updates/epoch here"],
       ["learning rates", "encoder 2.5e-5 · heads 1e-4", "Laya's defaults: the heads learn faster than the encoder"],
       ["loss", "rlcd", "Laya's default (proper-scoring reward + soft cross-entropy)"],
       ["shuffle_options", "choice", "re-orders queue options each epoch so it cannot learn 'first = hvac'"],
       ["calibration slice", "12 % of train items", "≥ 50 per question type, or the temperature fit is noise"],
       ["eval_data", "data/laya_val.jsonl", "held-out situations → Laya labels the table 'held_out'"],
       ["target_error", "0.05", "abstention cuts that keep accepted answers ≤ 5 % wrong"]],
      ["knob", "value", "why"])

# ── 2 · train (or read what was trained) ──────────────────────────────────────
step(2, "the training run")
summary = None
if TRAIN:
    if not F.L.where():
        raise SystemExit("✗ --train needs week24_typed_decisions_jev/.venv-local (Module 16, step 0)")
    import laya.train as LT
    warnings.simplefilter("always")
    t0 = time.perf_counter()
    buf, caught = io.StringIO(), []

    class Tee(io.TextIOBase):                                   # live progress AND a copy for the record
        def write(self, s):
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
            return buf.write(s)

    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(Tee()):
        warnings.simplefilter("always")
        summary = LT.finetune(str(F.DATA / "laya_train.jsonl"), F.LAYA_BASE, str(OUT),
                              LT.TrainConfig(eval_data=str(F.DATA / "laya_val.jsonl"), **RECIPE), device="auto")
        caught = sorted({" ".join(str(x.message).split())[:400] for x in w if "laya" in str(x.message).lower()
                         or "calibration" in str(x.message).lower()})
    secs = time.perf_counter() - t0
    log = buf.getvalue()
    import shutil
    shutil.rmtree(OUT / "checkpoint_latest", ignore_errors=True)   # an extra 1.3 GB copy we do not need
    import torch
    run = {"seconds": round(secs, 1), "device": str(LT.resolve_device("auto")), "chip": F.gpu_name(),
           "torch": torch.__version__, "log": log, "warnings": caught,
           "epoch_loss": summary.get("epoch_loss"), "train_items": summary.get("train_items"),
           "calibration_items": summary.get("calibration_items"), "temperature": summary.get("temperature"),
           "abstention_thresholds": summary.get("abstention_thresholds"),
           "report": json.loads((OUT / "rl_agent_config.json").read_text())["training"]["train_report"],
           "trained": time.strftime("%Y-%m-%d %H:%M")}
    (OUT / "lab04_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    if os.environ.get("JEV_RECORD") == "1" or not rec.path.is_file():
        rec.data, rec.dirty = run, True
        os.environ["JEV_RECORD"] = "1"
        rec.save()
    src = "LOCAL (just trained)"
elif (OUT / "lab04_run.json").is_file():
    run, src = json.loads((OUT / "lab04_run.json").read_text(encoding="utf-8")), "LOCAL (your checkpoint)"
else:
    run, src = rec.data, f"RECORDED {rec.data.get('trained', '')}"
    if not run:
        raise SystemExit("✗ no checkpoint and no recording — run this lab with --train from the ⌨ terminal")
    print("» no .runs/laya-ft here → replaying the 2026-10-17 training run. Train your own with --train.")
if not TRAIN:
    for line in run["log"].splitlines():
        if line.startswith(("train items", "optimizer budget", "epoch ")):
            print("   " + line)
print(f"» {run['train_items']} train items + {run['calibration_items']} calibration items · {run['device']} · "
      f"{run['seconds'] / 60:.1f} min · source: {src}")
loss = run.get("epoch_loss") or []
for i, v in enumerate(loss, 1):
    print(f"   epoch {i}  loss {v:.4f}  {'█' * max(1, round(v / max(loss) * 30))}")
import textwrap
for msg in run.get("warnings", []):
    print(textwrap.fill(msg, 100, initial_indent="   ⚠ ", subsequent_indent="     "))

# ── 3 · before / after on validation ─────────────────────────────────────────
step(3, "Laya's own before/after table — on held-out VALIDATION situations")
rep = run["report"]
print(f"» eval_mode = {rep['eval_mode']} · {rep['eval_items']} items · {rep['note']}")
b, a = rep["before"], rep["after"]
table([[m, f"{b[k]:.3f}", f"{a[k]:.3f}", f"{a[k] - b[k]:+.3f}"] for m, k in
       (("accuracy (all 3 questions)", "accuracy"), ("loss", "loss"), ("ECE", "ece"), ("Brier", "brier"),
        ("mean confidence", "mean_confidence")) if b.get(k) is not None and a.get(k) is not None],
      ["metric", "base multilingual", "fine-tuned", "change"])
bt, at = b.get("by_type", {}), a.get("by_type", {})
if bt and at:
    table([[f"{q} ({name})", f"{bt[q]['accuracy']:.2f} → {at[q]['accuracy']:.2f}",
            f"{bt[q]['mean_confidence']:.2f} → {at[q]['mean_confidence']:.2f}", f"{bt[q]['ece']:.2f} → {at[q]['ece']:.2f}"]
           for q, name in (("choice", "queue"), ("noul", "safety"), ("score", "urgency")) if q in bt and q in at],
          ["question", "accuracy", "mean confidence", "ECE"])
print("» Without --eval, Laya scores the in-sample calibration slice instead and SAYS so ('not held-out').")
print("  A smoke run that did that on copies of its own data printed 0.98 accuracy. That number meant nothing.")

# ── 4 · the generalisation gap ───────────────────────────────────────────────
step(4, "the same checkpoint on data it trained on, near data, and far data")
sets = [("train (seen)", F.split("train")[::4]), ("val (unseen situations)", F.split("val")),
        ("held-out (hand-written)", F.split("test") + F.split("m16"))]
rows = []
for name, ts in sets:
    p, s = F.predict("laya_ft", ts)
    ok, n = F.accuracy(ts, p)
    rows.append([name, len(ts), f"{ok}/{n}" if n else "—", f"{ok / n:.0%}" if n else "—", s])
table(rows, ["data", "tickets", "queue correct", "", "source"])
print("» The gap between rows is the honest summary of a fine-tune. Train ≈ 100 % only says it learned")
print("  the homework; held-out says whether it learned the JOB. Lab 17-6 does this properly, with error bars.")

# ── 5 · confidence after training ────────────────────────────────────────────
step(5, "what training did to confidence — temperatures and abstention cuts (fitted on the calibration slice)")
temps = run.get("temperature") or []
table([[q, f"{t:.2f}"] for q, t in zip(("choice", "score", "noul"), temps)], ["question type", "fitted temperature"])
cuts = run.get("abstention_thresholds") or {}
if cuts:
    print("» abstention cuts at ≤ 5 % error: " + " · ".join(f"{k} {v:.3f}" for k, v in cuts.items()))
if temps and all(abs(t - 1.0) < 1e-6 for t in temps):
    print("» Every temperature stayed at 1.0 and every cut is ~1.000: the calibration slice is cut from the")
    print("  TRAINING situations, the model already gets all of them right with ~100 % confidence, so there")
    print("  was nothing to calibrate on. Compare step 3: mean confidence 0.99+, accuracy lower — overconfident.")
elif temps and any(t <= 0.5 + 1e-6 or t >= 5.0 - 1e-6 for t in temps):
    print("» A temperature on the [0.5, 5] clamp means the fit ran out of room — too little or too easy data.")
print("» So the gate has to come from somewhere else: Lab 17-6 fits the cut on VALIDATION (unseen situations)")
print("  and checks it on held-out. Validation loss going UP while accuracy goes up is the same story:")
print("  fewer mistakes, but each one made with near-certainty.")
print(f"» serve it: laya.load('{OUT.relative_to(F.WEEK24.parent)}').predict(state, Q)  — same call as Module 16.")
