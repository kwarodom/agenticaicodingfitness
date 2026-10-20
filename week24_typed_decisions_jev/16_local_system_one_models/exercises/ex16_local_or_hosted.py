#!/usr/bin/env python3
"""Exercise 16 · Write the adapter and the engine picker for a local-first decision service.

Fill in the two TODOs, save, then run:
    .venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/exercises/ex16_local_or_hosted.py

The checker runs 11 offline tests first (free, no model, no API call). When they pass it sends one
ticket through YOUR adapter to GLiNER2.5-Decide on this machine (if week24_typed_decisions_jev/.venv-local is set up).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import localkit as L  # noqa: E402  (first: may re-run this file inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, check, choice, noul, score  # noqa: E402


# ── TODO 1 ── to_gliner(questions) → GLiNER2.5-Decide tasks for model.classify_text(text, tasks)
#   For each Jev question (same key in the output):
#     choice → {"labels": {option: description}, "prompt": instructions}
#     noul   → {"labels": ["yes", "no"],          "prompt": instructions}
#     score  → {"labels": [level, …] (same order), "prompt": instructions}
#   Remove backticks from the prompt (GLiNER reads one text, not a state object: "`message`" → "message").
#   Any other question type (e.g. "box") → raise ValueError — never silently drop a question.
def to_gliner(questions: dict) -> dict:
    return None


# ── TODO 2 ── pick_engine(task) → "jev" | "laya" | "gliner_decide" | "gliner_ner" | "human"
#   `task` keys: data ("public" | "internal" | "confidential"), lang ("en" | "th"),
#                state_tokens (int), needs_offsets (bool), latency_ms (int budget), labelled_examples (int)
#   Rules, checked IN THIS ORDER:
#     1. needs_offsets                          → "gliner_ner"   (decision models return no spans)
#     2. data == "confidential"                 → a LOCAL engine, never "jev":
#          state_tokens > 1000                  → "human"        (too long for the local models' windows)
#          lang == "th"                         → "laya"         (its Router sends Thai to the multilingual checkpoint)
#          otherwise                            → "gliner_decide"
#     3. latency_ms < 100                       → "laya"         (no network round trip)
#     4. otherwise                              → "jev"          (best zero-shot, 64k context)
def pick_engine(task: dict) -> str:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
Q = {"queue": choice("Which team should handle `message`?", {"hvac": "air conditioning", "safety": "smoke, fire"}),
     "safety": noul("Does `message` describe smoke or fire?"),
     "urgency": score("How urgent is `message`?", ["routine", "today", "immediate"])}
WANT = {"queue": {"labels": {"hvac": "air conditioning", "safety": "smoke, fire"},
                  "prompt": "Which team should handle message?"},
        "safety": {"labels": ["yes", "no"], "prompt": "Does message describe smoke or fire?"},
        "urgency": {"labels": ["routine", "today", "immediate"], "prompt": "How urgent is message?"}}
BASE = {"data": "public", "lang": "en", "state_tokens": 200, "needs_offsets": False, "latency_ms": 2000,
        "labelled_examples": 0}
PICKS = [
    ({}, "jev"),
    ({"needs_offsets": True, "data": "confidential"}, "gliner_ner"),
    ({"data": "confidential"}, "gliner_decide"),
    ({"data": "confidential", "lang": "th"}, "laya"),
    ({"data": "confidential", "state_tokens": 5000}, "human"),
    ({"latency_ms": 40}, "laya"),
    ({"data": "internal", "lang": "th"}, "jev"),
    ({"data": "internal", "state_tokens": 20000}, "jev"),
]


def main() -> None:
    banner("Exercise 16 · adapter + engine picker")
    ok = True
    try:
        out = to_gliner(Q)
    except Exception as e:  # noqa: BLE001
        out = None
        print(f"   to_gliner raised {type(e).__name__}: {e}")
    ok &= check(out == WANT, "to_gliner maps choice, noul and score exactly",
                f"TODO 1: to_gliner(Q) should be {WANT} · got {out}")
    try:
        to_gliner({"where": {"type": "box", "instructions": "the mug in `photo`"}})
        raised = False
    except ValueError:
        raised = True
    except Exception:  # noqa: BLE001
        raised = False
    ok &= check(raised, "to_gliner raises ValueError for a box question",
                "TODO 1: a 'box' question must raise ValueError, not disappear")
    try:
        same_order = list((to_gliner({"u": score("x", ["c", "a", "b"])}) or {}).get("u", {}).get("labels", []))
    except Exception:  # noqa: BLE001
        same_order = []
    ok &= check(same_order == ["c", "a", "b"], "score levels keep their order",
                f"TODO 1: score levels must keep their order · got {same_order}")
    for n, (delta, want) in enumerate(PICKS, 1):
        task = {**BASE, **delta}
        try:
            got = pick_engine(task)
        except Exception as e:  # noqa: BLE001
            got = f"raised {type(e).__name__}"
        ok &= check(got == want, f"pick {delta or 'default'} → {want}",
                    f"TODO 2: test {n} {delta or 'default'} expected {want!r} · got {got!r}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again — nothing was sent anywhere.")
        return

    print("\n✓ all offline tests pass — one ticket through YOUR adapter on this machine")
    model = L.decide(multi=False)
    text = "Sparks came out of the hair dryer socket in 702 when my daughter plugged it in."
    if model is None:
        print("◈ GLiNER2.5-Decide is not set up here (see localkit.py) — your adapter output would be:")
        print(f"   {to_gliner(Q)}")
        return
    print(f"» {model.classify_text(text, to_gliner(Q), include_confidence=True)}")
    print("» confidential guest text never left this laptop. Now ask: would you trust these three labels")
    print("  without a person? Lab 16-5's numbers say what to check before you do.")


if __name__ == "__main__":
    main()
