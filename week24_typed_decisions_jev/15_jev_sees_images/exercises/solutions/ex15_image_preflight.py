#!/usr/bin/env python3
"""Reference solution · Exercise 15 · Preflight an image request, then gate a Box answer — write both checks yourself.

If ANY image in a request breaks a limit, the whole request fails: there are no partial answers.
So production code checks every image BEFORE it spends a call, and never trusts a Box on its own.
Fill in the two TODOs, save, then run:
    .venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/exercises/ex15_image_preflight.py

The checker runs 12 offline tests first (free, no API call). Only when they pass does it send
one CCTV frame to jev-1.14.0 and run YOUR gate on the answer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vision as V  # noqa: E402  (first: it puts week24_typed_decisions_jev/common on the path)
from jevkit import banner, box, check, noul, show  # noqa: E402


# ── TODO 1 ── preflight(width, height, fmt, file_bytes) → list of problems ([] = OK to send)
#   The API limits (docs.typesafe.ai/go/images, checked 2026-10-17). Add one short string per broken rule:
#     • fmt must be "png", "jpg" or "jpeg" (lower-case)                   → a problem containing "format"
#     • width AND height each between 280 and 1568 px, ends included       → a problem containing "side"
#     • width × height between 78,400 and 2,458,624 px, ends included     → a problem containing "pixels"
#     • file_bytes ≤ 16 MiB (16 * 1024 * 1024), after base64 decoding      → a problem containing "size"
#   (A 2000×1500 photo breaks "side" AND "pixels" — report both.)
def preflight(width: int, height: int, fmt: str, file_bytes: int) -> list[str]:
    problems = []
    if fmt not in ("png", "jpg", "jpeg"):
        problems.append(f"format {fmt!r} — send PNG or JPEG")
    if not (280 <= width <= 1568 and 280 <= height <= 1568):
        problems.append(f"side {width}×{height} — each side must be 280…1568 px")
    if not (78_400 <= width * height <= 2_458_624):
        problems.append(f"pixels {width * height:,} — must be 78,400…2,458,624")
    if file_bytes > 16 * 1024 * 1024:
        problems.append(f"size {file_bytes / 2**20:.1f} MiB — max 16 MiB per image")
    return problems


# ── TODO 2 ── gate(answers) → {"act": bool, "edges": dict|None, "reason": str}
#   `answers` holds a noul "visible" and a box "where" from the SAME request.
#     1. visible < 0.80                  → act False, edges None, reason contains "absent"
#                                          (a Box answer is always a box — even around nothing)
#     2. where.confidence < 0.60         → act True, edges = the box padded 0.05 on every side,
#                                          clamped to 0…1 (V.pad does exactly this), reason contains "padded"
#     3. otherwise                       → act True, edges = the box unchanged, reason contains "confident"
def gate(answers: dict) -> dict:
    visible = answers["visible"]["noul"]
    where = answers["where"]
    if visible < 0.80:
        return {"act": False, "edges": None, "reason": f"target absent (visible {visible:.2f})"}
    if where["confidence"] < 0.60:
        return {"act": True, "edges": V.pad(where["box"]),
                "reason": f"padded — box confidence {where['confidence']:.2f}"}
    return {"act": True, "edges": where["box"], "reason": f"confident box ({where['confidence']:.2f})"}


# ─────────────────────────── checker — no need to edit below ────────────────
PRE = [  # (width, height, fmt, bytes) → words that must appear (empty = must pass)
    ((1152, 768, "jpg", 125_582), []),
    ((280, 280, "png", 9_000), []),
    ((1568, 1568, "jpeg", 2_000_000), []),
    ((279, 600, "png", 50_000), ["side"]),
    ((2000, 1500, "jpg", 900_000), ["side", "pixels"]),
    ((800, 600, "webp", 80_000), ["format"]),
    ((1200, 900, "png", 17 * 1024 * 1024), ["size"]),
]
B = {"top": 0.2, "bottom": 0.6, "left": 0.1, "right": 0.5}
GATES = [  # (visible, confidence, edges) → (act, padded?, reason word)
    ((0.05, 0.9, B), (False, None, "absent")),
    ((0.79, 0.9, B), (False, None, "absent")),
    ((0.95, 0.4, B), (True, True, "padded")),
    ((0.95, 0.4, {"top": 0.02, "bottom": 0.99, "left": 0.0, "right": 0.97}), (True, True, "padded")),
    ((0.95, 0.8, B), (True, False, "confident")),
]


def run_gate_test(vis, conf, edges):
    return gate({"visible": {"type": "noul", "noul": vis},
                 "where": {"type": "box", "box": dict(edges), "confidence": conf}})


def main() -> None:
    banner("Exercise 15 · image preflight + Box gate")
    ok = True
    for n, (args, words) in enumerate(PRE, 1):
        try:
            out = preflight(*args)
        except Exception as e:  # noqa: BLE001
            out = None
            print(f"   preflight test {n} raised {type(e).__name__}: {e}")
        good = isinstance(out, list) and (not words and out == [] or
                                          words and all(any(w in p for p in out) for w in words))
        ok &= check(good, f"preflight {args} → {out}",
                    f"TODO 1: preflight{args} should report {words or 'nothing'} · got {out}")
    for n, ((vis, conf, edges), (act, padded, word)) in enumerate(GATES, 1):
        try:
            out = run_gate_test(vis, conf, edges)
        except Exception as e:  # noqa: BLE001
            out = None
            print(f"   gate test {n} raised {type(e).__name__}: {e}")
        want_edges = None if not act else (V.pad(edges) if padded else edges)
        good = (isinstance(out, dict) and out.get("act") is act and word in str(out.get("reason", ""))
                and (out.get("edges") == want_edges if act else out.get("edges") is None))
        ok &= check(good, f"gate visible={vis} conf={conf} → act={act}{' padded' if padded else ''}",
                    f"TODO 2: gate test {n} expected act={act} edges={want_edges} reason~{word!r} · got {out}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again — no API call was made.")
        return

    print("\n✓ all offline tests pass — now one real frame (media/cctv_t7s.jpg) through jev-1.14.0")
    from PIL import Image
    with Image.open(V.MEDIA / "cctv_t7s.jpg") as im:
        w, h = im.size
    problems = preflight(w, h, "jpg", (V.MEDIA / "cctv_t7s.jpg").stat().st_size)
    if problems:
        print(f"✕ your preflight blocks this frame: {problems}")
        return
    ref, assets = V.load("cctv_t7s.jpg")
    qs = {"visible": noul("Is there smoke, steam or haze in `frame`?"),
          "where": box("the densest cloud of smoke, steam or haze in `frame`")}
    resp = V.ask_image({"frame": ref}, qs, assets)
    show(resp)
    decision = gate(resp["answers"])
    print(f"→ your gate: {decision}")
    if decision.get("act"):
        crop, drawn = V.crop_to_box("cctv_t7s.jpg", decision["edges"], "ex15_haze")
        print(f"» saved {drawn.relative_to(V.HERE)} — open it: is the red box on the haze?")
    print("» execute stays False: a person looks at the crop before anyone is sent to the plant room.")


if __name__ == "__main__":
    main()
