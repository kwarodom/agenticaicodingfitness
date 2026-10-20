#!/usr/bin/env python3
"""Lab 16-2 · Laya on your laptop: the same request shape as Jev, answered in tens of milliseconds, offline.

  1. one English ticket → the Router picks the English checkpoint → typed answers + latency
  2. the same request to Jev → compare the two answers side by side
  3. a Thai ticket → the Router switches to the multilingual checkpoint; forcing English fails quietly
  4. the truncation trap: the key fact sits after ~320 tokens of state

Needs week24_typed_decisions_jev/.venv-local + Lab 16-0 (otherwise replays answers recorded on 2026-10-17).
Run: .venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab02_laya_local.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import localkit as L  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
import jevkit  # noqa: E402
from jevkit import banner, choice, noul, score, show, step, table  # noqa: E402

banner("Lab 16-2 · Laya, locally", "Convai Innovations · Apache-2.0 · ModernBERT-large / mmBERT-base")
L.banner_local()
rec = L.Recorder("lab02_laya")
router = L.laya()

Q = {
    "queue": choice("Which team should handle `message`?", {
        "hvac": "air conditioning, heating, temperature", "housekeeping": "towels, cleaning, amenities",
        "plumbing": "water leaks, drains, toilets", "safety": "smoke, burning smell, fire, electrical danger",
        "front_desk": "check-in, check-out, billing, room changes"}),
    "wants_move": noul("Does the guest in `message` ask to move to another room?"),
    "urgency": score("How urgent is `message`?", ["routine", "today", "immediate"]),
}


def laya_ask(key: str, state, questions, *, repeat: int = 1, **kw) -> dict:
    """Run Laya locally (median of `repeat` warm calls) or replay the recording."""
    if router is not None:
        try:
            times, res = [], None
            for _ in range(repeat):
                res, ms = L.run_laya(router, state, questions, **kw)
                times.append(ms)
            out = {"answers": res["answers"], "routing": res.get("routing", {}), "ms": statistics.median(times)}
            rec.put(key, out)
            return {**L.as_resp(out["answers"], f"laya/{out['routing'].get('model', '?')}", "local", out["ms"]),
                    "routing": out["routing"]}
        except Exception as e:  # noqa: BLE001  (e.g. a checkpoint missing in offline mode)
            print(f"   ◇ Laya failed locally ({type(e).__name__}: {str(e)[:90]}) → recording")
    out = rec.get(key)
    if not out:
        raise SystemExit("✕ no local Laya and no recording — see localkit.py for setup")
    return {**L.as_resp(out["answers"], f"laya/{out['routing'].get('model', '?')}", "recorded", out["ms"]),
            "routing": out["routing"]}


# ── 1 · English ──────────────────────────────────────────────────────────────
EN = {"message": "Guest in room 1408 says the AC has been out since yesterday and they want to move "
                 "to another room tonight."}
step(1, "an English ticket — Router → English checkpoint (median of 5 warm calls)")
laya_ask("en_warmup", EN, Q)                      # first call loads the checkpoint
r_laya = laya_ask("en", EN, Q, repeat=5)
show(r_laya)
print(f"» routing: {r_laya['routing'].get('model')} — {r_laya['routing'].get('reason')}")

# ── 2 · same request to Jev ──────────────────────────────────────────────────
step(2, "the SAME state and questions to Jev")
r_jev = jevkit.ask(EN, Q, quiet=True)
jevkit.validate(r_jev, Q)
show(r_jev, top=2)
la, ja = r_laya["answers"], r_jev["answers"]
table([["queue", f"{la['queue']['choice']} (peak {max(la['queue']['probabilities'].values()):.2f})",
        f"{ja['queue']['choice']} (peak {max(ja['queue']['probabilities'].values()):.2f})"],
       ["wants_move", f"{la['wants_move']['noul']:.2f}", f"{ja['wants_move']['noul']:.2f}"],
       ["urgency", f"{la['urgency']['score']:.2f} / 2", f"{ja['urgency']['score']:.2f} / 2"],
       ["latency", f"{r_laya['_latency_ms']:.0f} ms (no network"
        + (f", {L.device()})" if r_laya["_source"] == "local" else ", recorded)"),
        f"{r_jev['_latency_ms']:.0f} ms (internet)" if r_jev.get("_latency_ms") else "recorded"]],
      ["question", "Laya (local)", "Jev (hosted)"])
print("» same JSON in, same answer shapes out. But Laya's `confidence` is its own statistic (it ships")
print("  `answer_confidence` = the peak too). Compare PROBABILITIES across models, not `confidence` fields.")

# ── 3 · Thai ─────────────────────────────────────────────────────────────────
TH = {"message": "ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 ใกล้ห้อง 1203 รีบมาดูด่วนค่ะ"}
Q3 = {"queue": Q["queue"]}
step(3, "a Thai ticket (burning smell near the electrical cabinet) — routed vs forced English")
laya_ask("th_warmup", TH, Q3)                     # loads the multilingual checkpoint once (~seconds)
r_routed = laya_ask("th_routed", TH, Q3, repeat=3)
r_forced = laya_ask("th_forced_en", TH, Q3, repeat=3, model="english")
rows = []
for name, r in (("Router (auto)", r_routed), ("forced english", r_forced)):
    a = r["answers"]["queue"]
    rows.append([name, r["routing"].get("model"), a["choice"], f"{max(a['probabilities'].values()):.2f}",
                 f"{r['_latency_ms']:.0f} ms"])
table(rows, ["how", "checkpoint", "queue", "peak prob", "time"])
for name, r in (("Router (auto)", r_routed), ("forced english", r_forced)):
    a = r["answers"]["queue"]
    peak = max(a["probabilities"].values())
    verdict = ("✓ right" if a["choice"] == "safety" else
               "✕ wrong AND confident — the dangerous kind" if peak >= 0.7 else "✕ wrong, but unsure — a gate would catch it")
    print(f"» {name:<15} {verdict}")
print("» the right answer is `safety`. Laya's card warns the English checkpoint scored 0.000 on Khmer at")
print("  0.952 confidence — confidence alone can't protect you. Test the language you actually serve.")

# ── 4 · truncation ───────────────────────────────────────────────────────────
filler = " ".join(f"Line {i}: chiller CH-{i % 3 + 1} supply 6.{i % 9} C, status normal." for i in range(60))
LONG = {"message": filler + " FINAL NOTE: there is visible smoke coming from the electrical panel in room 1203."}
step(4, f"the truncation trap — {len(LONG['message'].split())} words of state, the smoke is in the LAST sentence")
r_short = laya_ask("long_default", LONG, Q3, model="english")
r_wide = laya_ask("long_multi_8k", LONG, Q3, model="multilingual", max_len=8192)
r_jev_long = jevkit.ask(LONG, Q3, quiet=True)
rows = [["laya english (512 tok)", r_short["answers"]["queue"]["choice"],
         f"{max(r_short['answers']['queue']['probabilities'].values()):.2f}"],
        ["laya multilingual, max_len=8192", r_wide["answers"]["queue"]["choice"],
         f"{max(r_wide['answers']['queue']['probabilities'].values()):.2f}"],
        ["Jev (64k context)", r_jev_long["answers"]["queue"]["choice"],
         f"{max(r_jev_long['answers']['queue']['probabilities'].values()):.2f}"]]
table(rows, ["engine", "queue", "peak prob"])
print("» Laya's English checkpoint keeps ~320 tokens of state and quietly drops the rest — the smoke")
print("  sentence never reaches the model. Module 10's benchmark runner REFUSES rows that would truncate.")
if r_wide["answers"]["queue"]["choice"] != "safety":
    print("» the 8k multilingual run saw the whole text and STILL missed it: base Laya is a weak zero-shot")
    print("  model (Jev Decision Index 4.43 vs Jev 60.11). Convai calls it 'a fast base to specialise'.")
rec.save()
