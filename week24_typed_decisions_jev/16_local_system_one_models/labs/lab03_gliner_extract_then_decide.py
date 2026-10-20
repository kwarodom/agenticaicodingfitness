#!/usr/bin/env python3
"""Lab 16-3 · Classic GLiNER finds the spans, a decision model picks the right one.

Jev never returns a span or an offset. GLiNER (urchade/gliner_multi-v2.1) does nothing BUT that:
it marks every "room number", "equipment" or "time" in the text with start/end positions.
  1. extract: GLiNER pulls candidate spans (with offsets) from an English and a Thai ticket
  2. decide:  Jev answers a Choice whose options ARE those candidates — "which room needs the repair?"
  3. code:    keeps the offsets, highlights the chosen span, and refuses when the candidates are empty

Needs week24_typed_decisions_jev/.venv-local + Lab 16-0 for GLiNER (otherwise replays the recorded spans).
Run: .venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab03_gliner_extract_then_decide.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import localkit as L  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
import jevkit  # noqa: E402
from jevkit import banner, choice, noul, show, step, table  # noqa: E402

banner("Lab 16-3 · extract with GLiNER, decide with Jev", "spans + offsets locally → a typed choice over them")
L.banner_local()
rec = L.Recorder("lab03_gliner")
ner = L.gliner()
LABELS = ["room number", "floor", "equipment", "time"]

TICKETS = {
    "en": "Guest moved from 1203 to 1408 last night. The AC in 1408 works fine, but the bathroom ceiling in "
          "1203 is still dripping water onto the floor. Please check it before 3pm.",
    "th": "ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 ใกล้ห้อง 1203 รีบมาดูด่วนค่ะ",
}


def spans(key: str, text: str) -> tuple[list[dict], str]:
    if ner is not None:
        out = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}
               for e in ner.predict_entities(text, LABELS, threshold=0.3)]
        rec.put(key, out)
        return out, "local"
    out = rec.get(key)
    if out is None:
        raise SystemExit("✕ no local GLiNER and no recording — see localkit.py for setup")
    return out, "recorded"


def highlight(text: str, marks: list[dict]) -> str:
    for s in sorted(marks, key=lambda m: -m["start"]):   # right to left keeps earlier offsets valid
        text = text[:s["start"]] + "⟦" + text[s["start"]:s["end"]] + "⟧" + text[s["end"]:]
    return text


for lang, text in TICKETS.items():
    step(f"{lang.upper()}-1", f"extract — GLiNER over: {text[:70]}…")
    found, src = spans(lang, text)
    table([[e["label"], e["text"], f"{e['start']}–{e['end']}", f"{e['score']:.2f}"] for e in found],
          ["label", "span", "offsets", "score"])
    print(f"   ({L.tag(src)} · GLiNER never decides which room is the problem — it only finds candidates)")

    rooms = {}                                        # code: normalise + de-duplicate, keep EVERY offset
    for e in found:
        digits = "".join(ch for ch in e["text"] if ch.isdigit())
        if e["label"] in ("room number", "floor") and digits:
            rooms.setdefault(digits, []).append(e)

    step(f"{lang.upper()}-2", "decide — Jev chooses among the candidates")
    if not rooms:
        print("→ no candidate rooms: ask a person (never let a decision model invent one)")
        continue
    options = {f"room_{rid}": f"room {rid}" for rid in rooms}
    options["none_of_these"] = "the problem is not in any of these rooms, or the rooms are not stated"
    qs = {"problem_room": choice("Which room in `message` has the problem that needs a technician?", options),
          "safety": noul("Does `message` describe smoke, burning, fire, sparks or water near electricity?")}
    resp = jevkit.ask({"message": text}, qs, quiet=True)
    jevkit.validate(resp, qs)
    show(resp, top=3)
    pick = resp["answers"]["problem_room"]["choice"]
    chosen = next((rid for rid in rooms if f"room_{rid}" == pick), None)
    if chosen:
        offs = ", ".join(f"{m['start']}–{m['end']}" for m in rooms[chosen])
        print(f"→ code: room {chosen}, mentioned at offsets {offs}")
        print(f"   {highlight(text, rooms[chosen])}")
    else:
        print("→ code: Jev said none of the candidates — send to a person")

step(3, "who does what")
table([["find every room / time / equipment mention + offsets", "GLiNER (local, multilingual)", "spans"],
       ["which mention is the problem?", "Jev Choice over the candidates", "typed + probability"],
       ["no candidates → no guess; highlight; write the ticket", "your code", "deterministic"]],
      ["job", "who", "output"])
print("» the Thai ticket shows why: GLiNER marked '3' (a FLOOR) next to '1203'. A span model finds; it does")
print("  not judge. systemonemodels.org calls this order 'extract, then decide'.")
rec.save()
