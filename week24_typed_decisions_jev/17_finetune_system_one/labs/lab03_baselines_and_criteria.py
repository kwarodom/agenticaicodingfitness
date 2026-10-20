#!/usr/bin/env python3
"""Lab 17-3 · Before training: write down what success means, then measure the baselines.

  1. read criteria.json and LOCK it (hash + time in .runs/criteria.lock) — no moving goalposts later
  2. two floors every model must clear: always-the-majority-label, and hand-written keyword rules
  3. the zero-shot models on validation (120 generated tickets from situations never trained on)
  4. the same models on the 66 held-out tickets, plus Jev — the numbers fine-tuning has to beat
  5. what to expect: is there room to improve, and where?

Needs week24_typed_decisions_jev/.venv-local for live local answers (else RECORDED); Jev replays in DRY mode.
First run ~60–90 s (186 tickets × 3 local models); later runs read the .runs/ cache.
Run: .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab03_baselines_and_criteria.py
"""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, step, table  # noqa: E402

banner("Lab 17-3 · baselines and success criteria", "decide what 'better' means before you train anything")
F.L.banner_local()

# ── 1 · lock the criteria ────────────────────────────────────────────────────
step(1, "the success criteria, written before training — and locked")
crit_path = HERE / "criteria.json"
crit = json.loads(crit_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(crit_path.read_bytes()).hexdigest()[:16]
lock = F.RUNS / "criteria.lock"
F.RUNS.mkdir(exist_ok=True)
if lock.is_file() and json.loads(lock.read_text())["sha"] != digest:
    print("⚠ criteria.json CHANGED since it was locked. If no model is trained yet, delete .runs/criteria.lock")
    print("  and re-run. If one is, Lab 17-7 will grade against the ORIGINAL criteria — that is the point.")
elif not lock.is_file():
    lock.write_text(json.dumps({"sha": digest, "locked_at": time.time(),
                                "locked": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
table([[c["id"], c["what"]] for c in crit["criteria"]], ["", f"must hold on {crit['held_out'].split('=')[1].strip()}"])
print(f"» locked: sha256 {digest} · {json.loads(lock.read_text())['locked']}  (.runs/criteria.lock)")
print("» C1–C2 say 'better', C3 says 'never miss a fire', C4–C6 say 'and nothing else got worse'.")

# ── 2 · two floors ────────────────────────────────────────────────────────────
step(2, "two floors: the majority label, and keyword rules")
train, val = F.split("train"), F.split("val")
held = F.split("test") + F.split("m16")
majority = collections.Counter(t["queue"] for t in train).most_common(1)[0][0]
rows = []
for eng in ("majority", "keywords"):
    p, src = F.predict(eng, held, majority=majority)
    rows.append(F.summary_row(eng, held, p, src))
table(rows, F.SUMMARY_HEADERS)
print(f"» The majority label ('{majority}') is ~{max(collections.Counter(t['queue'] for t in held).values()) / len(held):.0%} "
      "on held-out. A model that cannot beat ten lines of keywords is not worth fine-tuning — or deploying.")

# ── 3 · zero-shot models on validation ────────────────────────────────────────
step(3, "zero-shot models on VALIDATION (120 tickets, situations the fine-tune will never see)")
base = {}
rows = []
for eng in ("laya_router", "laya_base", "gliner_base"):
    p, src = F.predict(eng, val)
    base[eng] = p
    rows.append(F.summary_row(eng, val, p, src))
table(rows, F.SUMMARY_HEADERS)
print("» Validation is for DECISIONS during training (which epoch, which threshold). It is generated, so it")
print("  flatters every model a little. The verdict comes from the hand-written held-out set below.")

# ── 4 · held-out, plus Jev ────────────────────────────────────────────────────
step(4, "the bar: every baseline on the 66 HELD-OUT tickets")
rows, held_preds = [], {}
for eng in ("majority", "keywords", "laya_router", "laya_base", "gliner_base", "jev"):
    p, src = F.predict(eng, held, majority=majority)
    held_preds[eng] = p
    rows.append(F.summary_row(eng, held, p, src))
table(rows, F.SUMMARY_HEADERS)
for eng in ("laya_router", "laya_base", "gliner_base", "jev"):
    e = F.queue_ece(held, held_preds[eng])
    ms = F.median_ms(held_preds[eng])
    print(f"   {eng:<12} queue ECE {e:.2f}" + (f" · median {ms:.0f} ms" if ms else "") if e is not None else f"   {eng}")
print("» ECE = |confidence − accuracy| averaged over bins: 0 means 'a 0.9 is right 90 % of the time'.")

# ── 5 · where is the room? ────────────────────────────────────────────────────
step(5, "where can fine-tuning help? (queue mistakes of the zero-shot models, by true team)")
rows = []
for eng in ("laya_router", "gliner_base"):
    miss = collections.Counter(t["queue"] for t in held if not F.correct(t, held_preds[eng].get(t["id"]), "queue"))
    rows.append([eng] + [miss.get(q, 0) for q in F.QUEUES] + [sum(miss.values())])
table(rows, ["engine", *F.QUEUES, "total wrong"])
print("» Concentrated mistakes (one team, one language) are what a few hundred examples can fix.")
print("  Mistakes spread evenly are usually a label-definition problem — fix the instructions first.")
