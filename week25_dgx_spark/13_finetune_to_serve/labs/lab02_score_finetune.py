#!/usr/bin/env python3
"""Lab 13-2 · Score the fine-tune: read LLaMA Factory's predictions from the Spark and apply the gate.

Module 09's lab 04 runs `llamafactory-cli train configs/hotel_predict.yaml` on the Spark. That writes
one prediction per held-out request to
    ~/w25/m09/saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl
This lab reads that file over ssh (read-only), scores it with exactly the rules lab 13-1 used for the
baseline, and saves a local copy in .runs/ for lab 13-4. In DRY mode there is no fine-tune to score,
so it says so and scores the baseline file instead, labelled as such, to show what the report looks like.

Run: .venv/bin/python week25/13_finetune_to_serve/labs/lab02_score_finetune.py [--file path.jsonl]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _hotel_eval as H  # noqa: E402
from sparkkit import banner, note, result, sh, step, table, warn, where  # noqa: E402

REMOTE = "~/w25/m09/saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl"
LOCAL = Path(sys.argv[sys.argv.index("--file") + 1]) if "--file" in sys.argv else None

banner("Lab 13-2 · score the fine-tune", "LLaMA Factory's predictions from the Spark, scored by the same gate")

step(1, "get the fine-tune's predictions")
label = "fine-tune (Spark)"
if LOCAL:
    rows = H.read_predictions(LOCAL)
    label = f"file {LOCAL.name}"
else:
    r = sh(f"test -f {REMOTE} && wc -l < {REMOTE} && cat {REMOTE} || echo MISSING", quiet=True,
           example='{"prompt": "…", "predict": "{\\"department\\": \\"security\\", …}", "label": "…"}  (×60)')
    rows = H.read_predictions(r.out) if r.live else []
    if r.live and "MISSING" in r.out:
        warn(f"{REMOTE} does not exist yet on {where()} — run Module 09 lab 04's predict step first:")
        print("$ cd ~/w25/m09 && llamafactory-cli train configs/hotel_predict.yaml   # ~a few minutes")
    elif r.live:
        H.write_predictions(rows, "predictions_finetune_spark.jsonl")
        note(f"{len(rows)} predictions copied to .runs/predictions_finetune_spark.jsonl")
if not rows:
    fallback = sorted(H.RUNS.glob("predictions_baseline_*.jsonl"))
    if not fallback:
        warn("nothing to score: no Spark predictions and no baseline yet. Run lab 13-1 first.")
        raise SystemExit(0)
    rows = H.read_predictions(fallback[-1])
    label = f"BASELINE {fallback[-1].name} (no fine-tune available — shown so you can read the report)"
    print(f"◈ DRY: no fine-tune predictions reachable; scoring {fallback[-1].name} instead")

step(2, f"score {len(rows)} predictions · {label}")
scores = [H.score_item(r.get("predict", ""), r.get("label", ""), r.get("prompt", "")) for r in rows]
m = H.summarize(scores)
table([[k, f"{got:.0%}", f"≥ {need:.0%}", "✓" if ok else "✕"] for k, got, need, ok in H.gate(m)],
      ["gate metric", "score", "needed to ship", ""])
print(f"│ priority accuracy {m['priority_acc']:.0%} · urgent cases: {m['urgent_n']}")

step(3, "the mistakes, by department")
conf = H.confusion(scores)
for (want, got), n in list(conf.items())[:6]:
    print(f"│ expected {want:14s} got {got:14s} × {n}")
if not conf:
    print("│ no department mistakes")

step(4, "slice by guest language — an average can hide a bias")
langs = H.by_language(rows, scores)
table([[lang, v["n"], f"{v['department_acc']:.0%}", f"{v['priority_acc']:.0%}", f"{v['false_urgent']:.0%}",
        f"{v['urgent_recall']:.0%}" if v.get("urgent_n", 1) else "— (no urgent cases)"]
       for lang, v in langs.items()],
      ["language", "n", "department", "priority", "normal → urgent", "urgent recall"])

passed = all(ok for *_, ok in H.gate(m))
result(f"{label}: {'PASSES' if passed else 'does not pass'} the ship gate.")
note("LLaMA Factory's own metrics (BLEU/ROUGE in predict_results.json) measure word overlap with the "
     "label. For a router, the gate above (valid JSON, right department, no missed emergency) is what matters.")
