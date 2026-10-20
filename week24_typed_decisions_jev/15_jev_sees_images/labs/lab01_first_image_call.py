#!/usr/bin/env python3
"""Lab 15-1 · Your first image call: a photo goes straight into Jev, no vision LLM in between.

  1. prepare the photo the way the docs ask (upright, ≤1568 px, content-hash reference)
  2. look at the raw request: the reference sits in `state`, the bytes sit in `assets`
  3. forget the danger-zone header → HTTP 400 (images are opt-in)
  4. send it properly → typed answers about the photo, plus what one image costs

Run: .venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab01_first_image_call.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision as V  # noqa: E402  (first: it puts week24_typed_decisions_jev/common on the path)
import jevkit  # noqa: E402
from jevkit import banner, show, step, table  # noqa: E402

banner("Lab 15-1 · first image call", f"model {V.MODEL} · header {jevkit.IMAGES_HEADER}: true")

# ── 1 · prepare ──────────────────────────────────────────────────────────────
step(1, "prepare media/leak_under_ahu.jpg")
from PIL import Image  # noqa: E402

with Image.open(V.MEDIA / "leak_under_ahu.jpg") as im:
    print(f"» file: {im.size[0]}×{im.size[1]} px · {(V.MEDIA / 'leak_under_ahu.jpg').stat().st_size / 1024:.0f} KB")
ref, assets = V.load("leak_under_ahu.jpg")
a = assets[ref]
print(f"» reference: {ref}   (sha256 of the file → the same photo always gets the same reference)")
print(f"» asset:     format={a['format']} · {len(a['data_base64']) / 1024:.0f} KB of base64 (≈ ⅓ bigger than the file)")
print("  limits: PNG/JPEG still image · each side 280…1568 px · 16 MiB per file · 32 MiB per request")

# ── 2 · the request ──────────────────────────────────────────────────────────
step(2, "the raw request body (base64 shortened for the screen)")
questions = {"leak": jevkit.noul("Is liquid pooling on the floor in `photo`?"),
             "queue": V.SCENE_QUESTIONS["queue"]}
shown = {"model": V.MODEL, "state": {"photo": ref},
         "assets": {ref: {**a, "data_base64": a["data_base64"][:24] + "…"}},
         "questions": questions}
print(json.dumps(shown, indent=1, ensure_ascii=False)[:1100])
print("» the model receives the IMAGE wherever the reference string appears. "
      "Questions point at it by name: `photo`.")

# ── 3 · no header ────────────────────────────────────────────────────────────
step(3, "send it WITHOUT the danger-zone header")
if jevkit.mode() == "live":
    try:
        jevkit.ask({"photo": ref}, questions, model_id=V.MODEL, assets=assets, images_header=False, quiet=True)
        print("» unexpected: the API accepted it — images may have left the danger zone. Check the docs.")
    except jevkit.JevError as e:
        print(f"✕ {e}")
else:   # an error can't be replayed from the cache — this is the message we got live on 2026-10-17
    print('✕ HTTP 400 {"detail":{"error_type":"api_usage_error","message":"Images are experimental. '
          'Send the header TypeSafe-Danger-Zone-Images: true to use them."}}   (recorded text)')
print("» images and Box are OFF by default. TypeSafe calls them 'dreamer-ready': fine for prototypes, "
      "not yet vouched for in production.")

# ── 4 · with header ──────────────────────────────────────────────────────────
step(4, "send it WITH the header (jevkit adds it when assets are present)")
resp = V.ask_image({"photo": ref}, questions, assets)
show(resp, top=3)

text_tokens = 180   # a short text-only call in Modules 01–05 is ~150–250 input tokens
img_tokens = (resp.get("usage") or {}).get("input_tokens", 0)
table([["text-only lab call", f"~{text_tokens}", f"${text_tokens * jevkit.PRICE_PER_MTOK / 1e6:.6f}"],
       ["this photo + 2 questions", img_tokens or "(dry: see live)", f"${jevkit.cost_usd(resp):.6f}"]],
      ["request", "input tokens", "cost"])
print("» images bill as ordinary input tokens — no separate image price. A 1152×768 photo is ~2.6k tokens, "
      "so it is ~15× a short text call but still ≈ $0.0001.")
