#!/usr/bin/env python3
"""Lab 15-5 · Jev 1.14.0: what changed besides images — versions, Thai, and long context.

  1. which names your key can send (aliases move; versioned ids don't)
  2. images on 1.13 vs 1.14 — and Box, which only 1.14 knows
  3. Thai hotel tickets on 1.13 vs 1.14 (the release notes promise the biggest gain in multilingual)
  4. a 47k-token BMS log: over the default 32k limit → 400; with the long-context header → answered

Run: .venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab05_versions_and_long_context.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision as V  # noqa: E402  (first: it puts week24_typed_decisions_jev/common on the path)
import jevkit  # noqa: E402
from jevkit import ask, banner, box, choice, noul, step, table, validate  # noqa: E402

OLD, NEW = "jev-1.13.0", "jev-1.14.0"
banner("Lab 15-5 · versions, Thai and long context", f"{OLD} (pinned course default) vs {NEW} (jev-preview)")


def live() -> bool:
    return jevkit.mode() == "live"


# ── 1 · names ────────────────────────────────────────────────────────────────
step(1, "model names on your key")
if live():
    try:
        for m in jevkit.list_models():
            print(f"» {m.get('name') or m.get('id'):<14} {m.get('release_date', ''):<12} {str(m.get('description', ''))[:70]}")
    except Exception as e:  # noqa: BLE001
        print(f"✕ {e}")
else:
    print("» (DRY) on 2026-10-17 the list showed the aliases jev-latest and jev-preview")
print("» jev-preview → jev-1.14.0 since 2026-10-14. jev-latest still pointed at jev-1.13.0 on 2026-10-17 and "
      "TypeSafe said it moves to 1.14.0 'in a week'. The course pins jev-1.13.0 so recordings stay reproducible; "
      "every response's `model` field tells you which version actually answered.")

# ── 2 · images and Box per version ───────────────────────────────────────────
step(2, "the same photo on both versions")
ref, assets = V.load("leak_under_ahu.jpg")
q_img = {"leak": noul("Is liquid pooling on the floor in `photo`?")}
q_box = {"puddle": box("the puddle of liquid on the floor in `photo`")}
rows = []
for m in (OLD, NEW):
    r = ask({"photo": ref}, q_img, model_id=m, assets=assets, quiet=True)
    validate(r, q_img)
    if live():
        try:
            b = ask({"photo": ref}, q_box, model_id=m, assets=assets, quiet=True)["answers"]["puddle"]
            box_txt = f"✓ conf {b['confidence']:.2f}"
        except jevkit.JevError as e:
            msg = re.search(r'"message":"([^"]+)"', str(e))
            box_txt = "✕ HTTP 400 " + (f'"{msg.group(1)}"' if msg else str(e)[:40])
    else:
        box_txt = "✓ (Lab 15-3)" if m == NEW else '✕ HTTP 400 "Invalid request." (recorded)'
    rows.append([m, f"{r['answers']['leak']['noul']:.2f}", (r.get("usage") or {}).get("input_tokens", 0), box_txt])
table(rows, ["model", "leak noul", "input tok", "Box question"])
print("» 'jev-1.13.0 was multimodal all along' — images work on both. Box is new in 1.14.")

# ── 3 · Thai ─────────────────────────────────────────────────────────────────
step(3, "Thai guest messages on both versions (synthetic, labels written by hand)")
THAI = [
    ("แอร์ห้อง 1203 ไม่เย็นเลย เป่าลมร้อนออกมา", "hvac"),
    ("ขอผ้าเช็ดตัวเพิ่มอีกสองผืนที่ห้อง 812 ค่ะ", "housekeeping"),
    ("มีน้ำหยดจากเพดานในห้องน้ำ พื้นเปียกหมดแล้ว", "plumbing"),
    ("ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 รีบมาดูด่วน", "safety"),
    ("ขอเลื่อนเวลาเช็คเอาท์เป็นบ่ายสองโมงได้ไหมคะ", "front_desk"),
    ("ไม่ต้องส่งช่างมาแล้วนะคะ แอร์กลับมาเย็นแล้ว", "no_action"),
]
Q = {"queue": choice("Which team should handle `message`?", {
    "hvac": "air conditioning, heating, temperature", "housekeeping": "towels, cleaning, amenities",
    "plumbing": "water leaks, drains, toilets", "safety": "smoke, burning smell, fire, electrical danger",
    "front_desk": "check-in, check-out, billing, room changes",
    "no_action": "the guest says nothing needs to be done (any more)"})}
rows, right = [], {OLD: 0, NEW: 0}
for text, gold in THAI:
    row = [text[:30], gold]
    for m in (OLD, NEW):
        a = validate(ask({"message": text}, Q, model_id=m, quiet=True), Q)["queue"]
        right[m] += a["choice"] == gold
        row.append(f"{'✓' if a['choice'] == gold else '✕'} {a['choice']} {a['confidence']:.2f}")
    rows.append(row)
table(rows, ["message", "expected", OLD, NEW])
print(f"» correct: {OLD} {right[OLD]}/{len(THAI)} · {NEW} {right[NEW]}/{len(THAI)} — six messages prove nothing; "
      "they show you HOW to check. Watch the confidence column as much as the ✓.")

# ── 4 · long context ─────────────────────────────────────────────────────────
step(4, "a 47k-token BMS log with one alarm buried in it")
lines = [f"2026-10-{1 + i // 400:02d} {i % 24:02d}:{i % 60:02d} AHU-{i % 9 + 1} SAT 13.{i % 10} C  "
         f"CHW valve {i % 100}%  status OK" for i in range(1200)]
lines.insert(911, "2026-10-03 03:12 AHU-3 SMOKE DETECTOR ALARM duct smoke detected, unit tripped")
log = "\n".join(lines)
q = {"alarm": noul("Does `log` contain a smoke or fire alarm?")}
print(f"» {len(lines)} lines · {len(log) / 1024:.0f} KB · default limits: 64k per request, 32k per state+question")
if live():
    try:
        ask({"log": log}, q, quiet=True)
        print("» accepted without the header (limits may have changed)")
    except jevkit.JevError as e:
        print(f"✕ without header: {e}")
else:
    print('✕ without header: HTTP 400 {"detail":{"error_type":"max_tokens_exceeded"}}   (recorded text)')
r = ask({"log": log}, q, long_context=True, timeout=120, quiet=True)
validate(r, q)
print(f"» with {jevkit.LONG_CONTEXT_HEADER}: true → alarm noul {r['answers']['alarm']['noul']:.2f}")
jevkit.summary(r)
print("» long context costs the same per token but is slower and less accurate (TypeSafe's words). Prefer: "
      "filter the log in code first (grep for ALARM), then ask Jev about the 20 lines that matter.")
