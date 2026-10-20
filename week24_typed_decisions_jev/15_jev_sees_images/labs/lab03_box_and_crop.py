#!/usr/bin/env python3
"""Lab 15-3 · Box: WHERE is it? One bounding box, a Noul gate, a crop, and edge checks.

  1. find the puddle: Box + "is there a puddle?" Noul in ONE request → draw it, crop it
  2. ask for something that is NOT there → Box still answers; the Noul gate saves you
  3. check the crop with one Noul per edge ("does it continue past the left edge?")
  4. a policy that only acts on a box when the gate AND the box confidence agree

Box needs jev-1.14.0 and the images header. Crops go to week24_typed_decisions_jev/15_jev_sees_images/.runs/.
Run: .venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab03_box_and_crop.py
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision as V  # noqa: E402  (first: it puts week24_typed_decisions_jev/common on the path)
from jevkit import banner, box, image, noul, show, step, table  # noqa: E402

banner("Lab 15-3 · Box", f"model {V.MODEL} · answer = 4 edges as fractions of the image")

MIN_GATE, MIN_BOX = 0.80, 0.60      # example thresholds — tune them on YOUR images (the docs give none)


def located(a: dict, gate: str, where: str) -> dict:
    """Code decides: use the box only if the target is there AND the box is confident; pad when unsure."""
    p, conf = a[gate]["noul"], a[where]["confidence"]
    if p < MIN_GATE:
        return {"use_box": False, "why": f"gate {p:.2f} < {MIN_GATE} → target probably absent, ignore the box"}
    edges = a[where]["box"]
    if conf < MIN_BOX:
        return {"use_box": True, "edges": V.pad(edges), "why": f"box confidence {conf:.2f} < {MIN_BOX} → padded 5 %"}
    return {"use_box": True, "edges": edges, "why": f"gate {p:.2f} · box confidence {conf:.2f}"}


# ── 1 · find the puddle ──────────────────────────────────────────────────────
step(1, "media/leak_under_ahu.jpg — where is the puddle?")
leak, a_leak = V.load("leak_under_ahu.jpg")
qs = {"puddle_visible": noul("Is there a puddle of liquid on the floor in `photo`?"),
      "puddle": box("the puddle of liquid on the floor in `photo`")}
resp = V.ask_image({"photo": leak}, qs, a_leak)
show(resp)
decision = located(resp["answers"], "puddle_visible", "puddle")
print(f"→ {decision['why']}")
if decision["use_box"]:
    crop, drawn = V.crop_to_box("leak_under_ahu.jpg", decision["edges"], "puddle")
    print(f"» saved {crop.relative_to(V.HERE)} and {drawn.relative_to(V.HERE)}")
    from PIL import Image
    with Image.open(V.MEDIA / "leak_under_ahu.jpg") as im:
        print(f"» in pixels on the {im.size[0]}×{im.size[1]} file: {V.to_pixels(decision['edges'], im.size)}"
              "  (left, top, right, bottom)")

# ── 2 · absent target ────────────────────────────────────────────────────────
step(2, "ask for something that is not in the photo")
qs = {"extinguisher_visible": noul("Is there a red fire extinguisher in `photo`?"),
      "extinguisher": box("the red fire extinguisher in `photo`")}
resp = V.ask_image({"photo": leak}, qs, a_leak)
show(resp)
print(f"→ {located(resp['answers'], 'extinguisher_visible', 'extinguisher')['why']}")
print("» a Box answer is ALWAYS a box — there is no 'not found'. Without the Noul in the same request "
      "you would draw a rectangle around nothing.")

# ── 3 · edge checks on the crop ──────────────────────────────────────────────
step(3, "check the crop: does the puddle continue past any edge?")
crop_path = V.RUNS / "puddle_crop.jpg"
if crop_path.is_file():
    edges_used = decision["edges"] if decision.get("use_box") else {}
    recipe = "leak_under_ahu.jpg|" + "|".join(f"{edges_used.get(e, 0):.2f}" for e in ("left", "top", "right", "bottom"))
    cref, casset = image(crop_path, ref="img-crop-" + hashlib.sha256(recipe.encode()).hexdigest()[:12])
    edges = {e: noul(f"Does the puddle of liquid in `crop` continue past the {e} edge of the image?")
             for e in ("left", "right", "top", "bottom")}
    resp = V.ask_image({"crop": cref}, edges, {cref: casset})
    show(resp)
    cut = [e for e, a in resp["answers"].items() if a["noul"] >= 0.5]
    print(f"→ edges to move outward: {cut or 'none'}  "
          "(the docs: 4 edge questions caught 30/36 cut-offs; one 'is all of it inside?' caught 26)")
else:
    print("» no crop yet — step 1 decided not to use the box")

# ── 4 · the thermostat: Box finds it, but Jev never READS it ─────────────────
step(4, "media/thermostat_room1203.png — find the reading, then ask about it")
th, a_th = V.load("thermostat_room1203.png")
qs = {"reading": box("the large measured-temperature number on the display in `photo`"),
      "band": V.choice("Which range contains the large measured temperature shown in `photo`?", {
          "below_22": "below 22 °C", "22_to_26": "22 to 26 °C", "26_to_28": "26 to 28 °C", "above_28": "above 28 °C"}),
      "above_setpoint": noul("Is the large measured temperature in `photo` more than 3 °C above the SET value?")}
resp = V.ask_image({"photo": th}, qs, a_th)
show(resp)
table([["29.1 (the exact number)", "a vision LLM or OCR → code (Module 13)", "Jev returns no text"],
       ["which band / above setpoint?", "Jev Choice / Noul on the image", "typed, cheap, thresholdable"],
       ["where on the screen?", "Jev Box", "crop for a person or for OCR"]],
      ["you need", "use", "why"])
