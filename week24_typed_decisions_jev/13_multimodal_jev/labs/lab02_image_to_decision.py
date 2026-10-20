#!/usr/bin/env python3
"""Lab 13-2 · Images → decisions: a vision model writes, code reads the numbers, Jev judges the meaning.

  1. thermostat photo → vision one-liner → CODE parses numbers + computes deviation → Jev: was it readable?
  2. leak photo       → vision description → Jev: queue · safety · severity · hedged → policy
  3. invoice scan     → vision transcription → CODE checks the arithmetic → Jev: suspicious? → policy
  4. same thermostat through EVERY vision provider you have → do they agree?

Run: .venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab02_image_to_decision.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perception as P  # noqa: E402
from jevkit import banner, show, step, table  # noqa: E402
import mediakit as mk  # noqa: E402

prov = mk.pick("see")
banner("Lab 13-2 · image → decision", f"vision: {prov} · decisions: Jev · rules: your code")

# ── 1 · thermostat ───────────────────────────────────────────────────────────
step(1, "thermostat display (media/thermostat_room1203.png)")
try:
    r = mk.see(prov, P.MEDIA / "thermostat_room1203.png", P.SEE_THERMOSTAT, max_tokens=300)
    mk.show(r, "see")
    facts = P.parse_thermostat(r["text"])
    print(f"» code facts  {facts}")
    a = P.validate(P.ask({"vision_text": r["text"]}, P.THERMO_QUESTIONS), P.THERMO_QUESTIONS)
    print(f"» Jev readable {a['readable']['noul']:.2f}")
    if a["readable"]["noul"] < 0.8 or facts.get("measured_c") is None:
        print("→ proposal: ask a person to read the display — the vision output is not trustworthy enough")
    elif facts.get("warm_deviation"):
        print(f"→ proposal: HVAC ticket for room {facts['room']} — {facts['deviation_c']:+.1f} °C above setpoint "
              f"(computed in code, not by any model)")
    else:
        print("→ proposal: no action — room within 2 °C of setpoint")
except mk.MediaError as e:
    print(f"✕ {e}")

# ── 2 · leak photo ───────────────────────────────────────────────────────────
step(2, "maintenance photo (media/leak_under_ahu.jpg)")
try:
    r = mk.see(prov, P.MEDIA / "leak_under_ahu.jpg", P.SEE_SCENE, max_tokens=400)
    mk.show(r, "see")
    resp = P.ask({"description": r["text"]}, P.SCENE_QUESTIONS)
    a = P.validate(resp, P.SCENE_QUESTIONS)
    show(resp, top=2)
    decision = P.policy({"modality": "image", "queue": a["queue"]["choice"], "safety": a["safety"]["noul"],
                         "perception_ok": bool(r["text"].strip())})
    print(f"→ proposal: {decision}")
except mk.MediaError as e:
    print(f"✕ {e}")

# ── 3 · invoice ──────────────────────────────────────────────────────────────
step(3, "supplier invoice (media/invoice_bank_change.png)")
try:
    r = mk.see(prov, P.MEDIA / "invoice_bank_change.png", P.SEE_INVOICE, max_tokens=500)
    mk.show(r, "see")
    math = P.check_invoice_math(r["text"])
    print(f"» code math   {math}")
    resp = P.ask({"invoice_text": r["text"]}, P.INVOICE_QUESTIONS)
    a = P.validate(resp, P.INVOICE_QUESTIONS)
    show(resp, top=2)
    decision = P.policy({"modality": "image", "queue": "finance", "suspicious": a["suspicious"]["noul"],
                         "perception_ok": bool(r["text"].strip())})
    print(f"→ proposal: {decision}")
    print("  The arithmetic is fine — which is exactly why a maths check alone would miss this fraud.")
except mk.MediaError as e:
    print(f"✕ {e}")

# ── 4 · every vision provider on the same image ──────────────────────────────
step(4, "same thermostat, every vision provider with a key")
rows = []
for p in [x for x in mk.capable("see") if x != "openrouter"]:
    try:
        r = mk.see(p, P.MEDIA / "thermostat_room1203.png", P.SEE_THERMOSTAT, max_tokens=320)
        f = P.parse_thermostat(r["text"])
        rows.append([p, r.get("model"), f.get("room"), f.get("measured_c"), f.get("setpoint_c"), f.get("mode"),
                     f"{(r.get('latency_ms') or 0) / 1000:.1f} s", r.get("_source")])
    except mk.MediaError as e:
        rows.append([p, "—", "error", "", "", "", "", str(e)[:30]])
table(rows, ["provider", "model", "room", "measured", "setpoint", "mode", "time", "source"])
parsed = [r for r in rows if None not in r[2:6] and r[2] != "error"]
bad = [r[0] for r in rows if r not in parsed]
agree = {tuple(r[2:6]) for r in parsed}
if bad:
    print(f"⚠ could not parse the reply from: {', '.join(bad)} → treat as 'unreadable' and ask a person")
print(f"═ {len(parsed)} provider(s) parsed: "
      f"{'they AGREE on every fact' if len(agree) == 1 else 'they DISAGREE on at least one fact → send to a person'}"
      f" (compared in code).")
print("  The Jev questions and the policy did not change when the vision model did.")
print("═ execute: False — every step above is a proposal for staff.")
