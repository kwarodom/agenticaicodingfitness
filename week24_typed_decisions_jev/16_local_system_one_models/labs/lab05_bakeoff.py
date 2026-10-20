#!/usr/bin/env python3
"""Lab 16-5 · A fair bake-off: Jev vs Laya vs GLiNER2.5-Decide on the same 18 tickets.

Fair means: same text, same three questions (via one adapter), same labels, same scoring,
every engine run zero-shot, Thai sent to each vendor's MULTILINGUAL model. Then:
  1. run every engine on every ticket (local engines on this machine, Jev over the API)
  2. accuracy per question and per language
  3. speed and cost per 1,000 tickets
  4. calibration on `safety` (Jev and Laya only — Decide has no probability distribution)

Needs week24_typed_decisions_jev/.venv-local + Lab 16-0 for the local engines (otherwise replays the 2026-10-17 run).
Run: .venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab05_bakeoff.py
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import localkit as L  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
import jevkit  # noqa: E402
from jevkit import banner, choice, noul, score, step, table  # noqa: E402

DATA = json.loads((HERE / "tickets.json").read_text(encoding="utf-8"))
T = DATA["tickets"]
banner("Lab 16-5 · bake-off", f"{len(T)} tickets · Jev (hosted) vs Laya vs GLiNER2.5-Decide (local)")
L.banner_local()
rec = L.Recorder("lab05_bakeoff")

Q = {
    "queue": choice("Which team should handle `message`?", {
        "hvac": "air conditioning, heating, temperature", "housekeeping": "towels, cleaning, amenities",
        "plumbing": "water leaks, drains, toilets, showers",
        "safety": "smoke, burning smell, fire, sparks, or water near electricity",
        "front_desk": "check-in, check-out, billing, room changes, stay extensions"}),
    "safety": noul("Does `message` describe smoke, burning, fire, sparks or water near electricity?"),
    "urgency": score("How urgent is `message`?", [
        "routine — whenever convenient, or nothing needs doing", "today — a problem that should be fixed today",
        "immediate — danger or damage happening now"]),
}


def norm_jev(a: dict) -> dict:
    return {"queue": a["queue"]["choice"], "safety_p": a["safety"]["noul"],
            "safety": a["safety"]["noul"] >= 0.5, "urgency": round(a["urgency"]["score"])}


def norm_decide(a: dict) -> dict:
    return {"queue": a["queue"]["choice"], "safety_p": None, "safety": a["safety"]["label"] == "yes",
            "urgency": a["urgency"]["level"]}


# ── 1 · run ──────────────────────────────────────────────────────────────────
step(1, "run every engine on every ticket")
router, dec_en, dec_multi = L.laya(), L.decide(False), L.decide(True)
results = {"jev": {}, "laya": {}, "decide": {}}
times = {"jev": [], "laya": [], "decide": []}
src = {"jev": jevkit.mode(), "laya": "local" if router else "recorded", "decide": "local" if dec_en else "recorded"}
if router:                                                  # warm both Laya checkpoints before timing
    for t in (T[0], T[-1]):
        L.run_laya(router, {"message": t["text"]}, Q)
for m in (dec_en, dec_multi):
    if m:
        m.classify_text("warm up", L.to_gliner_schema(Q))

for t in T:
    state = {"message": t["text"]}
    r = jevkit.ask(state, Q, quiet=True)
    results["jev"][t["id"]] = norm_jev(jevkit.validate(r, Q))
    if r.get("_latency_ms"):
        times["jev"].append(r["_latency_ms"])

    key = f"laya:{t['id']}"
    if router:
        res, ms = L.run_laya(router, state, Q)
        out = {"a": norm_jev(res["answers"]), "ms": ms, "ckpt": res.get("routing", {}).get("model")}
        rec.put(key, out)
    else:
        out = rec.get(key)
    if out:
        results["laya"][t["id"]] = out["a"]
        times["laya"].append(out["ms"])

    key = f"decide:{t['id']}"
    model = dec_multi if t["lang"] != "en" else dec_en      # each vendor's own advice: multi for Thai
    if model:
        ans, ms = L.run_decide(model, t["text"], Q)
        out = {"a": norm_decide(ans), "ms": ms}
        rec.put(key, out)
    else:
        out = rec.get(key)
    if out:
        results["decide"][t["id"]] = out["a"]
        times["decide"].append(out["ms"])
    print(f"   {t['id']}  jev {results['jev'][t['id']]['queue']:<12} "
          f"laya {results['laya'].get(t['id'], {}).get('queue', '—'):<12} "
          f"decide {results['decide'].get(t['id'], {}).get('queue', '—'):<12} gold {t['queue']}")

# ── 2 · accuracy ─────────────────────────────────────────────────────────────
step(2, "accuracy — same labels, same scoring for every engine")


def acc(engine: str, field: str, lang: str | None = None) -> str:
    rows = [t for t in T if (lang is None or t["lang"] == lang) and t["id"] in results[engine]]
    if not rows:
        return "—"
    ok = sum(results[engine][t["id"]][field] == t[field] for t in rows)
    return f"{ok}/{len(rows)}"


table([[f, acc("jev", f), acc("laya", f), acc("decide", f)] for f in ("queue", "safety", "urgency")]
      + [[f"queue · {lg.upper()} only", acc("jev", "queue", lg), acc("laya", "queue", lg), acc("decide", "queue", lg)]
         for lg in ("en", "th")],
      ["question", "Jev 1.13", "Laya (Router)", "GLiNER2.5-Decide (+multi for TH)"])
mae = {e: statistics.mean(abs(results[e][t["id"]]["urgency"] - t["urgency"]) for t in T     # skip a label
                          if t["id"] in results[e] and results[e][t["id"]]["urgency"] is not None)  # outside the set
       for e in results if results[e]}
print("» urgency mean absolute error (0 = perfect, levels 0–2): "
      + " · ".join(f"{e} {v:.2f}" for e, v in mae.items()))

# ── 3 · speed and cost ───────────────────────────────────────────────────────
step(3, "speed per ticket (3 questions) and cost per 1,000 tickets")
med = {e: (statistics.median(v) if v else None) for e, v in times.items()}
jev_tok = 480                                                # measured: ~480 input tokens per ticket here
table([["Jev 1.13", f"{med['jev']:.0f} ms" if med["jev"] else "recorded", "internet round trip",
        f"${jev_tok * 1000 * jevkit.PRICE_PER_MTOK / 1e6:.3f}", L.tag(src["jev"])],
       ["Laya", f"{med['laya']:.0f} ms" if med["laya"] else "—", f"this machine, {L.device() if router else 'GPU/CPU'}", "$0 + your hardware", L.tag(src["laya"])],
       ["GLiNER2.5-Decide", f"{med['decide']:.0f} ms" if med["decide"] else "—", "this machine, CPU",
        "$0 + your hardware", L.tag(src["decide"])]],
      ["engine", "median time", "where", "per 1,000 tickets", "source"])
print("» local times are model time on THIS laptop; Jev's include the network. Neither is 'the' speed.")

# ── 4 · calibration ──────────────────────────────────────────────────────────
step(4, "is a 0.9 really 90 %? Brier score on `safety` (lower is better; 0.25 = always saying 0.5)")
for e in ("jev", "laya"):
    ps = [(results[e][t["id"]]["safety_p"], t["safety"]) for t in T if t["id"] in results[e]]
    if ps:
        brier = statistics.mean((p - (1.0 if y else 0.0)) ** 2 for p, y in ps)
        confident_wrong = sum(1 for p, y in ps if (p >= 0.8 and not y) or (p <= 0.2 and y))
        print(f"» {e:<5} Brier {brier:.3f} · confidently wrong {confident_wrong}/{len(ps)}")
print("» GLiNER2.5-Decide is left out on purpose: it returns one score for the winning label, not P(yes).")
print(f"» 18 tickets is a classroom sample, not a verdict. Use Module 09's method on a few hundred of YOUR")
print("  tickets, fine-tune the local model if it is close, and keep the data boundary decision separate.")
rec.save()
