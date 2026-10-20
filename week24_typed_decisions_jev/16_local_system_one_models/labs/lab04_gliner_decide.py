#!/usr/bin/env python3
"""Lab 16-4 · GLiNER2.5-Decide: Fastino's open decision model, fed from the SAME Jev questions.

  1. the adapter: Jev choice/noul/score → GLiNER2.5-Decide classification tasks
  2. one English ticket on Decide (local) vs Jev (hosted)
  3. a Thai ticket: the English model vs GLiNER2.5-multi-Decide
  4. what Decide has that Jev doesn't (multi-label, spans in the same model) — and what its scores are NOT

Needs week24_typed_decisions_jev/.venv-local + Lab 16-0 (otherwise replays answers recorded on 2026-10-17).
Run: .venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab04_gliner_decide.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import localkit as L  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
import jevkit  # noqa: E402
from jevkit import banner, choice, noul, score, step, table  # noqa: E402

banner("Lab 16-4 · GLiNER2.5-Decide", "Fastino Labs · 340M DeBERTa-v3 · Apache-2.0 · CPU")
L.banner_local()
rec = L.Recorder("lab04_decide")

Q = {
    "queue": choice("Which team should handle `message`?", {
        "hvac": "air conditioning, heating, temperature", "housekeeping": "towels, cleaning, amenities",
        "plumbing": "water leaks, drains, toilets", "safety": "smoke, burning smell, fire, electrical danger",
        "front_desk": "check-in, check-out, billing, room changes"}),
    "safety": noul("Does `message` describe smoke, burning, fire, sparks or water near electricity?"),
    "urgency": score("How urgent is `message`?", ["routine", "today", "immediate"]),
}


def decide_ask(key: str, multi: bool, text: str, questions: dict) -> tuple[dict, float, str]:
    model = L.decide(multi=multi)
    if model is not None:
        model.classify_text(text, L.to_gliner_schema(questions))      # warm-up
        answers, ms = L.run_decide(model, text, questions)
        rec.put(key, {"answers": answers, "ms": ms})
        return answers, ms, "local"
    r = rec.get(key)
    if r is None:
        raise SystemExit("✕ no local GLiNER2.5-Decide and no recording — see localkit.py for setup")
    return r["answers"], r["ms"], "recorded"


def fmt(a: dict) -> str:
    if a["type"] == "choice":
        return f"{a['choice']} ({a['confidence']:.2f})"
    if a["type"] == "noul":
        return f"{a['label']} ({a['confidence']:.2f})"
    return f"{a['label']} ({a['confidence']:.2f})"


# ── 1 · adapter ──────────────────────────────────────────────────────────────
step(1, "the adapter — the same three questions, translated for GLiNER2.5-Decide")
print(json.dumps(L.to_gliner_schema(Q), indent=1, ensure_ascii=False)[:900])
print("» choice → labels with descriptions · noul → ['yes','no'] + prompt · score → ordered labels as plain classes")
print("  (Decide's own card: an ordinal head 'does not know that 6 is closer to 7 than 0' — measure it with MAE).")

# ── 2 · English ──────────────────────────────────────────────────────────────
EN = "There is smoke coming out of the electrical panel next to room 1203 and the lights are flickering."
step(2, f"English: {EN}")
d_en, ms_en, src = decide_ask("en", False, EN, Q)
j_en = jevkit.ask({"message": EN}, Q, quiet=True)
ja = jevkit.validate(j_en, Q)
table([["queue", fmt(d_en["queue"]), f"{ja['queue']['choice']} (p {max(ja['queue']['probabilities'].values()):.2f})"],
       ["safety", fmt(d_en["safety"]), f"noul {ja['safety']['noul']:.2f}"],
       ["urgency", fmt(d_en["urgency"]), f"score {ja['urgency']['score']:.2f} / 2"],
       ["time", f"{ms_en:.0f} ms · {L.tag(src)} · CPU", (f"{j_en['_latency_ms']:.0f} ms · LIVE" if j_en.get("_latency_ms") else "RECORDED")]],
      ["question", "GLiNER2.5-Decide", "Jev"])
print("» Decide returns a LABEL and one confidence for that label. Jev returns a probability for every option.")

# ── 3 · Thai ─────────────────────────────────────────────────────────────────
TH = "ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 ใกล้ห้อง 1203 รีบมาดูด่วนค่ะ"
step(3, f"Thai: {TH}")
rows = []
for key, multi, name in (("th_en_model", False, "GLiNER2.5-Decide (English)"),
                         ("th_multi_model", True, "GLiNER2.5-multi-Decide")):
    a, ms, src = decide_ask(key, multi, TH, Q)
    rows.append([name, fmt(a["queue"]), fmt(a["safety"]), f"{ms:.0f} ms · {L.tag(src)}"])
j_th = jevkit.validate(jevkit.ask({"message": TH}, Q, quiet=True), Q)
rows.append(["Jev 1.13", f"{j_th['queue']['choice']} ({max(j_th['queue']['probabilities'].values()):.2f})",
             f"noul {j_th['safety']['noul']:.2f}", "hosted"])
table(rows, ["engine", "queue", "safety", "time"])
print("» expected: safety / yes. Fastino's suite is English; its card says use the multi model for other")
print("  languages. Never send Thai to an English-only checkpoint just because it returns *an* answer.")

# ── 4 · beyond Jev, and the caveat ───────────────────────────────────────────
step(4, "multi-label topics + spans in the same model — and what the scores are not")
model = L.decide(multi=False)
GUEST = ("Guest in 1408 says the AC has been out since yesterday, wants to move tonight, "
         "and asks us to release the incidentals hold on the card.")
TASKS = {"topics": {"labels": ["hvac", "billing", "housekeeping", "room_change", "safety"],
                    "multi_label": True, "cls_threshold": 0.4}}
ALL = {"topics": {"labels": TASKS["topics"]["labels"], "multi_label": True, "cls_threshold": 0.0}}
if model is not None:
    topics = model.classify_text(GUEST, TASKS, include_confidence=True)
    allsc = model.classify_text(GUEST, ALL, include_confidence=True)
    ents = model.extract_entities(GUEST, ["room number", "equipment", "time"])
    rec.put("multi", {"topics": topics, "all": allsc, "entities": ents})
    src = "local"
else:
    r = rec.get("multi") or {}
    topics, allsc, ents, src = r.get("topics"), r.get("all"), r.get("entities"), "recorded"
print(f"» topics (multi-label, ≥0.4): {topics}   [{L.tag(src)}]")
print(f"» spans from the SAME model:  {ents}")
scores = {d["label"]: round(d["confidence"], 2) for d in (allsc or {}).get("topics", [])}
print(f"» every topic's score: {scores} → sum {sum(scores.values()):.2f}")
print("» independent per-label scores, not a distribution that sums to 1. Don't threshold them like Jev")
print("  probabilities, and don't average them with Jev's — tune Decide's thresholds on your own data.")
rec.save()
