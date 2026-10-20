#!/usr/bin/env python3
"""Lab 09-1 · A smoke evaluation — measure before you automate.

12 human-labeled intent requests (8 English, 4 Thai) → one Jev call each →
the numbers that decide whether a router may act on its own:

  raw accuracy        correct / successful calls
  coverage            auto-routed / ALL rows   (failures and 'review' count against it)
  accepted accuracy   correct among the auto-routed ones
  wrong accepted      the dangerous number: routed automatically AND wrong

The answers are saved to .runs/intent_answers.json so Lab 09-2 can re-threshold
them WITHOUT calling Jev again.

Run: .venv/bin/python week24/09_evaluate_and_cost/labs/lab01_smoke_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evalkit as ev                                          # noqa: E402
from jevkit import banner, step, table                        # noqa: E402

banner("Lab 09-1 · smoke evaluation of the intent router",
       "12 labeled rows · same 7 questions as week24/jev_lab/jev_lab.py intent")

# ── 1. the labeled data ─────────────────────────────────────────────────────
step(1, "load the human-labeled test rows")
rows = ev.load_rows()
langs = {}
for r in rows:
    langs[r.get("language", "en")] = langs.get(r.get("language", "en"), 0) + 1
print(f"│ {len(rows)} rows · languages {langs} · labels are TEACHING labels, not a benchmark")
print("│ the 'expected' field is stripped by jev_lab.prepare() — Jev never sees the answer key")

# ── 2. one call per row ─────────────────────────────────────────────────────
step(2, f"ask Jev — {len(rows)} calls, one per row")
recs = ev.run_intent_eval(rows)
table([[r["id"], r["language"], r["gold"], r.get("pred", "ERROR"),
        f"{max(r['probabilities'].values()):.2f}" if r.get("ok") else "—",
        "✓" if r.get("ok") and r["pred"] == r["gold"] else "✕",
        "auto" if ev.gate(r) else "review"] for r in recs],
      ["id", "lang", "gold", "predicted", "top p", "match", "gate"])
path = ev.save_run(recs)
print(f"◆ saved {len(recs)} answers → {path.relative_to(ev.WEEK24.parent)} (Lab 09-2 reuses them)")

# ── 3. the headline numbers ─────────────────────────────────────────────────
step(3, "score it — reference gate: confidence ≥ .75 AND top ≥ .80 AND margin ≥ .20")
m = ev.metrics(recs)
print(f"◆ raw accuracy (successes only) : {ev.pct(m['raw_accuracy'])}")
print(f"◆ coverage (auto-routed / all)  : {ev.pct(m['coverage'])}  ({m['accepted']} of {m['rows']})")
print(f"◆ accepted accuracy             : {ev.pct(m['accepted_accuracy'])}")
print(f"◆ wrong accepted                : {m['wrong_accepted']}   ← routed automatically AND wrong")
print(f"◆ API / validation failures     : {m['failures']}")

by_lang = {}
for r in recs:
    if r.get("ok"):
        by_lang.setdefault(r["language"], []).append(r["pred"] == r["gold"])
for lang, hits in by_lang.items():
    print(f"◆ accuracy [{lang}]                 : {sum(hits)}/{len(hits)}")

# ── 4. where the mistakes are ───────────────────────────────────────────────
step(4, "confusion matrix — rows = gold label, columns = what Jev chose")
cm = ev.confusion(recs)
seen = [l for l in ev.LABELS if l in cm or any(l in row for row in cm.values())]
table([[g] + [cm.get(g, {}).get(p, "") or "·" for p in seen] for g in seen], ["gold \\ pred"] + seen)
for r in recs:
    if r.get("ok") and r["pred"] != r["gold"]:
        print(f"⚠ {r['id']}: gold {r['gold']} but Jev chose {r['pred']} "
              f"(p={max(r['probabilities'].values()):.2f}) — is the MODEL wrong, or the LABEL?")

# ── 5. latency and cost ─────────────────────────────────────────────────────
step(5, "latency and cost (successful calls)")
lat = [r["latency_ms"] for r in recs if r.get("ok")]
toks = sum(r.get("input_tokens", 0) for r in recs if r.get("ok"))
p50, p95 = ev.percentile(lat, .5), ev.percentile(lat, .95)
if p50 is None:
    print("◈ no latency in DRY mode — answers were replayed, not timed")
else:
    print(f"◆ p50 {p50:.0f} ms · p95 {p95:.0f} ms   (nearest-rank, n={len(lat)})")
print(f"◆ {toks} input tokens total · avg {toks / max(len(lat) or len(recs), 1):.0f}/call · "
      f"${toks * 0.042 / 1e6:.6f} for the whole eval")
print("\n═ 12 rows is a SMOKE test. It tells you the plumbing works, not that the router is")
print("  safe. A pilot needs several hundred independently labeled rows per language.")
