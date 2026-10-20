#!/usr/bin/env python3
"""Lab 15-4 · Direct (photo → Jev) vs pipeline (photo → vision LLM → text → Jev), same photos, same questions.

Module 13 taught the pipeline: a multimodal model DESCRIBES the photo, Jev judges the description.
Jev 1.14 can now look at the photo itself. Which is better? Measure, don't guess:

  1. three plant-room photos through BOTH paths
  2. compare the answers, the time and the tokens side by side
  3. the rule of thumb you take away

Run: .venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab04_direct_vs_pipeline.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "13_multimodal_jev"))
import vision as V  # noqa: E402  (first: it puts week24_typed_decisions_jev/common on the path)
import perception as P  # noqa: E402  Module 13's prompts and questions, unchanged
import mediakit as mk  # noqa: E402
from jevkit import banner, step, table  # noqa: E402

prov = mk.pick("see", "gemini")
banner("Lab 15-4 · direct vs pipeline", f"pipeline: {prov} describes → Jev 1.13 judges text · direct: Jev 1.14 judges the photo")

PHOTOS = ["leak_under_ahu.jpg", "plantroom_smoke.jpg", "plantroom_normal.jpg"]
rows, agree = [], 0

step(1, "both paths, three photos")
for name in PHOTOS:
    print(f"\n   media/{name}")
    # pipeline — exactly Module 13's lab 13-2 path
    seen = mk.see(prov, P.MEDIA / name if (P.MEDIA / name).is_file() else V.MEDIA / name, P.SEE_SCENE, max_tokens=400)
    if seen.get("_source") == "placeholder":
        print(f"   pipeline: {seen['text']}")
        pipe = None
    else:
        print(f"   pipeline · {prov} wrote: {seen['text'][:150]}…")
        pipe = P.ask({"description": seen["text"]}, P.SCENE_QUESTIONS, quiet=True)
        pa = P.validate(pipe, P.SCENE_QUESTIONS)
    # direct — the photo itself
    ref, assets = V.load(name)
    direct = V.ask_image({"photo": ref}, V.SCENE_QUESTIONS, assets, quiet=True)
    da = direct["answers"]
    if pipe:
        same = pa["queue"]["choice"] == da["queue"]["choice"] and (pa["safety"]["noul"] >= 0.5) == (da["safety"]["noul"] >= 0.5)
        agree += same
        see_ms = seen.get("latency_ms") or 0
        rows.append([name, f"{pa['queue']['choice']} · safety {pa['safety']['noul']:.2f}",
                     f"{da['queue']['choice']} · safety {da['safety']['noul']:.2f}", "✓" if same else "✕",
                     f"{(see_ms + (pipe.get('_latency_ms') or 0)) / 1000:.1f} s" if see_ms else "recorded",
                     f"{(direct.get('_latency_ms') or 0) / 1000:.1f} s" if direct.get("_latency_ms") else "recorded",
                     f"{(seen.get('usage') or {}).get('input_tokens', 0)}+{(pipe.get('usage') or {}).get('input_tokens', 0)}",
                     (direct.get("usage") or {}).get("input_tokens", 0)])
    print(f"   direct   · queue {da['queue']['choice']} · safety {da['safety']['noul']:.2f} · "
          f"severity {da['severity']['score']:.2f}/3")

step(2, "side by side")
table(rows, ["photo", "pipeline (text)", "direct (image)", "agree", "pipe time", "direct time",
             "pipe tok (LLM+Jev)", "direct tok"])
if rows:
    print(f"» {agree}/{len(rows)} photos: same queue and same side of 0.5 on safety")

step(3, "what to take away")
table([["you need the exact text/number (29.1 °C, an invoice total)", "pipeline — Jev writes nothing"],
       ["an audit trail a person can read ('what did the model see?')", "pipeline — the description IS the log"],
       ["audio or video", "pipeline — Jev takes still PNG/JPEG only (export frames)"],
       ["a fast typed judgment on a photo, one call, one bill", "direct — ~1 s, ~2.7k tokens ≈ $0.0001"],
       ["compare against a known picture / define options by example", "direct — no LLM can pass that through text"],
       ["production today", "pipeline, or direct behind a human gate — images are 'dreamer-ready'"]],
      ["situation", "pick"])
print("» TypeSafe's own warning: every model is worse on images than on text, and simple, decomposed "
      "questions help most. Keep the Module 13 policy — only the evidence path changes.")
