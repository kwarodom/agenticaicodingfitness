#!/usr/bin/env python3
"""Exercise 07 · The memory tool's two rules: one hit per kind, and no citation the index never returned.

Fill in the three TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/exercises/ex07_memory_tool.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from embedkit import banner, check  # noqa: E402

import re


# ── TODO 1 ── keep the best `per_kind` hits of EACH kind (hits are dicts with kind, source, score),
#   then return them all sorted by score, highest first. This is search_memory's modality-gap fix.
def per_kind_top(hits: list, per_kind: int = 2) -> list:
    return hits


# ── TODO 2 ── cited(answer): every "[...]" source in the answer, in order of appearance.
#   unknown_citations(answer, sources): the cited ones that are NOT in `sources` (hallucinated citations).
def cited(answer: str) -> list:
    return []


def unknown_citations(answer: str, sources: set) -> list:
    return []


# ── TODO 3 ── Lab 07-3 gave 100/100. Read Lab 07-2's Thai answer and its search_memory query: what was wrong
#   with it that the grader did not check? 1–3 sentences.
GRADER_BLIND_SPOT = """
"""


# ─────────────────────────── checker — no need to edit below ────────────────
HITS = [{"kind": "note", "source": "hotel_kb.json#2", "score": 0.80},
        {"kind": "note", "source": "hotel_kb.json#0", "score": 0.71},
        {"kind": "note", "source": "hotel_kb.json#3", "score": 0.70},
        {"kind": "audio", "source": "media/voicemail_burning_th.wav", "score": 0.66},
        {"kind": "audio", "source": "media/request_towels_en.wav", "score": 0.61},
        {"kind": "image", "source": "media/plantroom_smoke.jpg", "score": 0.62}]
SOURCES = {h["source"] for h in HITS} | {"incident log"}


def main() -> None:
    banner("Exercise 07 · the memory tool's two rules", "offline checker · no model, no LLM")
    good = True
    try:
        m = per_kind_top(HITS, per_kind=1)
        got = [(h["kind"], h["source"]) for h in m]
    except Exception as e:  # noqa: BLE001
        got = f"error: {e}"
    good &= check(got == [("note", "hotel_kb.json#2"), ("audio", "media/voicemail_burning_th.wav"),
                          ("image", "media/plantroom_smoke.jpg")],
                  "per_kind_top: the best hit OF EACH KIND, then sorted by score (the modality-gap fix)",
                  f"TODO 1: got {got}")
    a1 = "Haze in B2 [incident log]. Evacuate and call 2222 [hotel_kb.json#2]."
    a2 = "Room 1203 reported a burning smell [media/voicemail_burning_th.wav] — see the CCTV [media/cctv_lobby.mp4]."
    try:
        c = [cited(a1), unknown_citations(a1, SOURCES), unknown_citations(a2, SOURCES)]
    except Exception as e:  # noqa: BLE001
        c = f"error: {e}"
    good &= check(c == [["incident log", "hotel_kb.json#2"], [], ["media/cctv_lobby.mp4"]],
                  "cited / unknown_citations: every [source] in order · flag the ones the index never returned",
                  f"TODO 2: got {c}")
    t = GRADER_BLIND_SPOT.lower()
    good &= check(len(t.split()) >= 15 and any(w in t for w in ("wrong note", "plant room", "procedure", "relevant",
                                                                  "query", "burning", "irrelevant", "right note")),
                  "your note names what Lab 07-3 scored 100 but should not have",
                  "TODO 3: ≥ 15 words — read Lab 07-2's Thai answer and its search query again")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Add a check for your blind spot to lab07_3_grade.py — that is how a grader earns trust.")


if __name__ == "__main__":
    main()
