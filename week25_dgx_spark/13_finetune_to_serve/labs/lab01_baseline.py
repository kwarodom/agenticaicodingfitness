#!/usr/bin/env python3
"""Lab 13-1 · The baseline: how good is the base model with a good prompt and no fine-tune?

A fine-tune is only worth serving if it beats this number. The lab sends Module 09's 60 held-out guest
messages (with the same system prompt the training data uses) to a plain, un-fine-tuned model and
scores the answers with the shared scorer (_hotel_eval.py): JSON validity, department accuracy,
urgent recall, and the "never promise a time" reply rule.

Where it runs, in order: the base model on your Spark's vLLM (Qwen/Qwen3-4B-Instruct-2507, the model
Module 09 fine-tunes) → Ollama on this laptop (LAPTOP STAND-IN, a different model, so a different
baseline) → DRY. Predictions are saved in LLaMA Factory's predict format so lab 13-2 scores them
exactly like the fine-tune's file.

Run: .venv/bin/python week25/13_finetune_to_serve/labs/lab01_baseline.py [--limit 20] [--kind ollama]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _hotel_eval as H  # noqa: E402
from sparkkit import banner, chat_any, note, pick_laptop_model, result, step, table, warn  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"            # Module 09's base model, served un-fine-tuned
KIND = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "vllm"
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

banner("Lab 13-1 · the baseline", "prompt-only base model on the 60 held-out hotel requests")

items = H.load_eval(LIMIT)
step(1, f"send {len(items)} held-out guest messages with Module 09's system prompt")
laptop = pick_laptop_model(["gemma4:12b", "nemotron-3-nano"])
rows, sources, t0 = [], set(), time.time()
for i, it in enumerate(items, 1):
    msgs = [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["instruction"]}]
    r = chat_any(KIND, BASE, msgs, max_tokens=160, laptop_model=laptop, quiet=i > 1, temperature=0.0)
    sources.add(r["source"])
    if r["source"] == "reference":
        break
    rows.append({"prompt": it["instruction"], "predict": r.get("text", ""), "label": it["output"],
                 "model": r.get("model"), "source": r["source"]})
    if i % 10 == 0:
        print(f"│ {i}/{len(items)} answered · {time.time() - t0:.0f}s")

if not rows:
    warn("no endpoint answered (DRY): nothing to score. Start vLLM on the Spark (Module 05) or Ollama here.")
    note("With a Spark: serve the BASE model without the adapter, e.g. `vllm serve Qwen/Qwen3-4B-Instruct-2507`.")
    raise SystemExit(0)

src = "LAPTOP STAND-IN" if "laptop" in sources else "Spark"
path = H.write_predictions(rows, f"predictions_baseline_{'laptop' if 'laptop' in sources else 'spark'}.jsonl")
note(f"{len(rows)} predictions from {rows[0]['model']} ({src}) → {path.relative_to(H.MOD.parents[1])}")

step(2, "score them with the same rules the fine-tune will face")
scores = [H.score_item(r["predict"], r["label"], r["prompt"]) for r in rows]
m = H.summarize(scores)
table([[k, f"{got:.0%}", f"≥ {need:.0%}", "✓" if ok else "✕"] for k, got, need, ok in H.gate(m)],
      ["gate metric", "baseline", "needed to ship", ""])
print(f"│ priority accuracy {m['priority_acc']:.0%} · urgent cases in this set: {m['urgent_n']}")

step(3, "where does it go wrong?")
conf = H.confusion(scores)
if conf:
    for (want, got), n in list(conf.items())[:5]:
        print(f"│ expected {want:14s} got {got:14s} × {n}")
else:
    print("│ no department mistakes")
bad_json = [r["predict"][:90].replace("\n", " ") for r, s in zip(rows, scores) if not s["json_valid"]]
for b in bad_json[:2]:
    print(f"│ not valid router JSON: {b!r}")

passed = all(ok for *_, ok in H.gate(m))
result(f"baseline {'PASSES' if passed else 'does not pass'} the ship gate ({src}). "
       "Lab 13-2 scores the fine-tune's predictions with the same rules; lab 13-4 puts them side by side.")
if "laptop" in sources:
    note(f"This baseline is {rows[0]['model']} on your laptop, not Qwen3-4B on the Spark. Re-run with a Spark "
         "for the real before/after comparison.")
