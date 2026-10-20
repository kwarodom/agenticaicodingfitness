#!/usr/bin/env python3
"""Lab 04-3 · Thai vs English — the same request, two languages.

TypeSafe says English is Jev's primary training language; other languages
work but must be TESTED on your own content. Alto Copilot users write in
Thai, English and a mix of both — so we measure, pair by pair:

  • does the topic stay the same?
  • how far does the probability distribution move? (total variation distance)
  • does confidence drop in Thai?

Run: .venv/bin/python week24/04_intent_routing/labs/lab03_thai_vs_english.py
"""
import sys
from pathlib import Path

WEEK24 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK24 / "common"))
sys.path.insert(0, str(WEEK24 / "jev_lab"))
import jev_lab as ref  # noqa: E402
from jevkit import ask, banner, step, table, validate  # noqa: E402

# Only the questions we compare — fewer questions, cheaper calls.
QUESTIONS = {k: ref.QUESTIONS["intent"][k] for k in ("intent", "needs_live_data")}

PAIRS = [
    ("Why is AHU-3 not cooling the hotel lobby?",
     "ทำไม AHU-3 ไม่เย็นที่ล็อบบี้โรงแรม"),
    ("Calculate verified electricity savings against the adjusted baseline.",
     "คำนวณผลการประหยัดไฟฟ้าที่ผ่านการตรวจสอบ เทียบกับ baseline ที่ปรับแล้ว"),
    ("Schedule preventive maintenance for the chiller next week.",
     "นัดบำรุงรักษาเชิงป้องกันชิลเลอร์สัปดาห์หน้า"),
    ("Summarize our hotel's carbon emissions for the ESG report.",
     "สรุปการปล่อยคาร์บอนของโรงแรมสำหรับรายงาน ESG"),
    ("Please check the chiller plant kW/ton for yesterday.",
     "ช่วย check chiller plant kW/ton ของเมื่อวานหน่อย"),      # mixed TH/EN, as people really type
]


def tvd(p: dict, q: dict) -> float:
    """Total variation distance: 0 = identical distributions, 1 = no overlap."""
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


banner("Lab 04-3 · Thai vs English", "same meaning, two languages — measure, don't assume")

step(1, f"ask {len(PAIRS)} pairs (2 calls per pair)")
rows, same = [], 0
for en, th in PAIRS:
    a_en = validate(ask({"message": en}, QUESTIONS, quiet=True), QUESTIONS)
    a_th = validate(ask({"message": th}, QUESTIONS, quiet=True), QUESTIONS)
    i_en, i_th = a_en["intent"], a_th["intent"]
    same += i_en["choice"] == i_th["choice"]
    rows.append([th, i_en["choice"], i_th["choice"],
                 f"{i_en['confidence']:.2f} → {i_th['confidence']:.2f}",
                 f"{tvd(i_en['probabilities'], i_th['probabilities']):.2f}",
                 f"{a_en['needs_live_data']['noul']:.2f} → {a_th['needs_live_data']['noul']:.2f}"])
table(rows, ["Thai / mixed message", "EN topic", "TH topic", "confidence EN→TH", "TVD", "live-data EN→TH"])

step(2, "read the table")
print(f"◆ same topic in {same}/{len(PAIRS)} pairs")
print("◆ TVD near 0 = the Thai distribution matches the English one; > 0.2 deserves a look")
print("═ Five pairs prove nothing statistically. Before production, measure per-class accuracy")
print("  on hundreds of real (de-identified) Thai and mixed messages — Lab 09 shows how.")
