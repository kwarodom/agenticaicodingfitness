#!/usr/bin/env python3
"""Lab 15-2 · An image can go anywhere text can go: state, instructions, criteria.

  1. STATE        — a photo is evidence; every question is asked about it
  2. INSTRUCTIONS — "is `photo` the same room as THIS picture: <ref>?" (compare to a known image)
  3. CRITERIA     — define Choice options by EXAMPLE pictures instead of words
  4. FRAMES       — two CCTV frames in order: "what changed?"

Run: .venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab02_where_images_go.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision as V  # noqa: E402  (first: it puts week24_typed_decisions_jev/common on the path)
from jevkit import banner, choice, noul, show, step  # noqa: E402

banner("Lab 15-2 · where images go", f"model {V.MODEL} · all media synthetic")

# ── 1 · state ────────────────────────────────────────────────────────────────
step(1, "STATE — the leak photo, with Module 13's three questions")
leak, a_leak = V.load("leak_under_ahu.jpg")
show(V.ask_image({"photo": leak, "location": "Basement B1, AHU-2"}, V.SCENE_QUESTIONS, a_leak), top=2)
print("» state can mix text and images: `location` is a string, `photo` is a picture.")

# ── 2 · instructions ─────────────────────────────────────────────────────────
step(2, "INSTRUCTIONS — compare a live CCTV frame against a known reference photo")
frame, a_frame = V.load("cctv_t7s.jpg")
normal, a_normal = V.load("plantroom_normal.jpg")
qs = {
    "same_room": noul(f"Is `photo` the same plant room, with the same equipment layout, as this reference "
                      f"picture taken on a normal day: {normal}?"),
    "differs_from_normal": noul(f"Compared with this reference picture of a normal day: {normal}, does `photo` "
                                f"show smoke, steam or haze that is not in the reference?"),
}
show(V.ask_image({"photo": frame}, qs, {**a_frame, **a_normal}))
print("» the reference image is written INSIDE the sentence. Each use adds another copy of the image "
      "(more tokens) — when several questions need it, put it in `state` once instead.")

# ── 3 · criteria by example ──────────────────────────────────────────────────
step(3, "CRITERIA — options defined by example pictures (and one by words)")
smoke, a_smoke = V.load("plantroom_smoke.jpg")
qs = {"condition": choice("Which example does the plant room in `photo` look most like?", {
    "normal": {"description": "Normal operation, clear air", "examples": [normal]},
    "haze": {"description": "Smoke, steam or haze in the room", "examples": [smoke]},
    "water": f"Liquid pooling on the floor, like this: {leak}",
    "none": "Not a plant room, or matches none of the examples"})}
first, a_first = V.load("cctv_t1s.jpg")
for name, ref, assets in (("cctv_t1s.jpg", first, a_first), ("cctv_t7s.jpg", frame, a_frame)):
    print(f"\n   photo = media/{name}")
    show(V.ask_image({"photo": ref}, qs, {**assets, **a_normal, **a_smoke, **a_leak}))
print("» the option NAMES (normal, haze, …) are what your code receives; the pictures are what Jev compares "
      "against. Option names are object keys, so they are never replaced by images.")

# ── 4 · frames ───────────────────────────────────────────────────────────────
step(4, "FRAMES — a short history, oldest first: what changed?")
state = {"camera": "CAM-03 plant room B2",
         "frames": [{"seconds_ago": 6, "frame": first}, {"seconds_ago": 0, "frame": frame}]}
qs = {
    "haze_appeared": noul("Did smoke, steam or haze appear between the first and the last of `frames`?"),
    "haze_now": noul("Is there smoke, steam or haze in `frames[1].frame`?"),
    "haze_before": noul("Is there smoke, steam or haze in `frames[0].frame`?"),
}
show(V.ask_image(state, qs, {**a_first, **a_frame}))
print("» `frames[1].frame` is a path into the state — the same backtick paths you used for text in Module 03.")
