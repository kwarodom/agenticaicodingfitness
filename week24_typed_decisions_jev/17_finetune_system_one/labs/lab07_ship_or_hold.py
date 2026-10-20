#!/usr/bin/env python3
"""Lab 17-7 · Ship or hold? Grade every fine-tune against the criteria you locked BEFORE training.

  1. check the goalposts did not move: criteria.json still matches the lock, and the lock is older
     than every checkpoint it is about to judge
  2. grade each candidate (laya_ft, gliner_lora, gliner_full) on C1–C7 → SHIP or HOLD, with reasons
  3. write a model card per candidate: data, base, recipe, metrics, verdict, date (.runs/model_card_*.json)
  4. what shipping looks like: the serving call, the confidence gate, and the rollback

Reads .runs/lab06_results.json (run Lab 17-6 first), else the 2026-10-17 results (RECORDED). Free, instant.
Run: .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab07_ship_or_hold.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, step, table  # noqa: E402

banner("Lab 17-7 · ship or hold", "the locked criteria decide — not how the numbers feel")

# ── 1 · goalposts ─────────────────────────────────────────────────────────────
step(1, "did the goalposts move?")
crit_path = HERE / "criteria.json"
crit = json.loads(crit_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(crit_path.read_bytes()).hexdigest()[:16]
lock_path = F.RUNS / "criteria.lock"
lock = json.loads(lock_path.read_text()) if lock_path.is_file() else None
if not lock:
    print("⚠ no .runs/criteria.lock — run Lab 17-3 before training, so the criteria are on record first.")
elif lock["sha"] != digest:
    print(f"✗ criteria.json changed after it was locked ({lock['sha']} → {digest}). Grading is void until the")
    print("  original file is restored (git checkout criteria.json). Change criteria for the NEXT model, not this one.")
    raise SystemExit(1)
else:
    print(f"✓ criteria.json matches the lock (sha256 {digest}, locked {lock['locked']})")
for eng in ("laya_ft", "gliner_lora", "gliner_full"):
    d = F.CKPT[eng] if eng == "laya_ft" else F.gl_dir(eng)
    weights = sorted(d.glob("*.safetensors")) if d.is_dir() else []     # when the WEIGHTS were written
    if lock and weights:
        t = weights[0].stat().st_mtime
        before = lock["locked_at"] < t
        print(f"{'✓' if before else '✗'} {eng}: weights written {time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}"
              f" — {'after' if before else 'BEFORE'} the criteria were locked")

# ── 2 · grade ─────────────────────────────────────────────────────────────────
step(2, "grade each candidate on C1–C7")
res_path = F.RUNS / "lab06_results.json"
if res_path.is_file():
    results, src = json.loads(res_path.read_text()), "LOCAL (.runs/lab06_results.json)"
else:
    results, src = json.loads((F.RECORDED / "lab06_results.json").read_text()), "RECORDED 2026-10-17"
print(f"» results: {src}")
verdicts = {}
for eng in ("laya_ft", "gliner_lora", "gliner_full"):
    m = dict(results["engines"].get(eng, {}))
    if not m:
        continue
    verdict, rows = F.ship_decision(m, crit["criteria"])
    verdicts[eng] = (verdict, rows, m)
    print(f"\n   {eng}  →  {'🟢 SHIP' if verdict == 'SHIP' else '🔴 HOLD'}")
    table([[cid, status, detail] for cid, status, detail in rows], ["", "result", "measured"])
if not verdicts:
    print("» nothing to grade — train in Labs 17-4/17-5, then run Lab 17-6.")
print("\n» A HOLD is a result, not a failure: it names exactly what to fix (data for the broken tickets,")
print("  more Thai examples, a confidence gate) before the next attempt — under the SAME criteria.")

# ── 3 · model cards ───────────────────────────────────────────────────────────
step(3, "write a model card per candidate")
data_sha = hashlib.sha256((F.DATA / "laya_train.jsonl").read_bytes()).hexdigest()[:16]
for eng, (verdict, rows, m) in verdicts.items():
    card = {"model": eng, "base": F.LAYA_BASE if eng == "laya_ft" else F.GLINER_BASE,
            "checkpoint": str(F.CKPT[eng].relative_to(F.WEEK24.parent)),
            "training_data": {"file": "data/laya_train.jsonl" if eng == "laya_ft" else "data/gliner_train.jsonl",
                              "tickets": 456, "sha256_laya_jsonl": data_sha, "generator": "datagen.py seed 17, split by situation"},
            "evaluated_on": crit["held_out"], "criteria_sha": digest, "verdict": verdict,
            "criteria": [{"id": c, "result": s, "measured": d} for c, s, d in rows],
            "metrics": {k: v for k, v in m.items() if not isinstance(v, list)},
            "known_failures": m.get("broke", []), "written": time.strftime("%Y-%m-%d %H:%M"),
            "intended_use": "routing synthetic hotel-engineering tickets in a classroom; NOT validated on real guests"}
    out = F.RUNS / f"model_card_{eng}.json"
    F.RUNS.mkdir(exist_ok=True)
    out.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"   ✓ {out.relative_to(HERE)}  ({verdict})")
print("» The card travels with the checkpoint: anyone who loads it can see what it was trained on, how it")
print("  was judged, and which tickets it still gets wrong.")

# ── 4 · shipping ──────────────────────────────────────────────────────────────
step(4, "what shipping looks like")
print("""   # serve — the same call as Module 16, pointed at your checkpoint
   agent = laya.load("week24_typed_decisions_jev/17_finetune_system_one/.runs/laya-ft")
   res = agent.predict({"message": text}, Q, min_confidence={"default": CUT})   # CUT from Lab 17-6 (val)
   if res["answers"]["queue"].get("low_confidence"): send_to_person(...)        # never quietly to the cloud

   # GLiNER LoRA: one base in memory, one small adapter per site
   pm = PeftModel.from_pretrained(AutoExtractor.from_pretrained(BASE), ".runs/gliner-lora/best")

   # roll back = point at the previous checkpoint (keep it!), or switch the adapter off""")
print("» Shadow first: run the fine-tune next to the current model for a week, log both, compare on")
print("  tickets people actually resolved. Promote only if it holds on THAT data too.")
