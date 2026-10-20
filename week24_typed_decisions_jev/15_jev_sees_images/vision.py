#!/usr/bin/env python3
"""Shared pieces for Module 15 · Jev sees images (Jev 1.14 "danger zone").

Everything a lab needs to send a photo to Jev directly — no vision LLM in between:

    from vision import MEDIA, RUNS, load, ask_image, crop_to_box

    ref, assets = load("leak_under_ahu.jpg")              # (reference, {ref: asset})
    resp = ask_image({"photo": ref}, QUESTIONS, assets)   # images header added for you

All media is synthetic (drawn with code, an image model, or rendered with ffmpeg).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import jevkit  # noqa: E402
from jevkit import box, choice, image, noul, score  # noqa: E402,F401

MEDIA = HERE / "media"
RUNS = HERE / ".runs"                                 # gitignored; the runner can show files here
MODEL = jevkit.VISION_MODEL                           # jev-1.14.0 — Box questions need it

# The same leak questions Module 13 asked about a vision model's DESCRIPTION — now asked about the photo.
SCENE_QUESTIONS = {
    "queue": choice("Which engineering queue should handle what `photo` shows?", {
        "water_leak": "Water or another liquid leaking, dripping or pooling",
        "smoke_or_fire": "Smoke, steam, haze, burning, fire or sparks",
        "comfort": "Temperature or comfort complaint with no visible damage",
        "none": "Nothing wrong is visible",
        "unclear": "The picture is too dark, blurred or ambiguous to choose"}),
    "safety": noul("Does `photo` show something that could injure a person or damage electrical "
                   "equipment — smoke, steam, haze, fire, sparks, or liquid near electrical equipment?"),
    "severity": score("How severe is the problem visible in `photo`?", [
        "nothing wrong", "cosmetic, fix when convenient", "needs a technician today",
        "immediate hazard, clear the area"]),
}


def load(name: str) -> tuple[str, dict]:
    """media/<name> → (reference, {reference: asset}). The reference is a content hash."""
    ref, asset = image(MEDIA / name)
    return ref, {ref: asset}


def ask_image(state, questions: dict, assets: dict, *, model_id: str | None = None, **kw) -> dict:
    """jevkit.ask pinned to jev-1.14.0 unless you say otherwise. Validates before returning."""
    resp = jevkit.ask(state, questions, model_id=model_id or MODEL, assets=assets, **kw)
    jevkit.validate(resp, questions)
    return resp


def to_pixels(edges: dict, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Box fractions → (left, top, right, bottom) pixels for PIL. Same answer works at any size."""
    w, h = size
    return (round(edges["left"] * w), round(edges["top"] * h),
            round(edges["right"] * w), round(edges["bottom"] * h))


def pad(edges: dict, amount: float = 0.05) -> dict:
    """Move every edge outward by `amount` of the image, staying inside 0…1 (docs pattern)."""
    return {"left": max(0.0, edges["left"] - amount), "top": max(0.0, edges["top"] - amount),
            "right": min(1.0, edges["right"] + amount), "bottom": min(1.0, edges["bottom"] + amount)}


def crop_to_box(name: str, edges: dict, out_name: str, *, outline: bool = True) -> tuple[Path, Path | None]:
    """Save the crop (and, optionally, the full photo with the box drawn) under .runs/."""
    from PIL import Image, ImageDraw
    RUNS.mkdir(exist_ok=True)
    with Image.open(MEDIA / name) as im:
        im = im.convert("RGB")
        px = to_pixels(edges, im.size)
        crop_path = RUNS / f"{out_name}_crop.jpg"
        im.crop(px).save(crop_path, "JPEG", quality=90)
        drawn = None
        if outline:
            d = ImageDraw.Draw(im)
            d.rectangle(px, outline=(255, 64, 64), width=5)
            drawn = RUNS / f"{out_name}_box.jpg"
            im.save(drawn, "JPEG", quality=88)
    return crop_path, drawn
