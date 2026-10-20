#!/usr/bin/env python3
"""Lab 02-2 · Noul — the probability that a statement is TRUE.

A Noul answers "how likely is it that this proposition holds?" with one number
from 0 (no) to 1 (yes). Three lessons, each in one batched call:

  1. clear yes / clear no / genuinely unsure — 0.5 means UNSURE, not "medium"
  2. multi-label: one noul PER label. They are independent — no need to sum to 1
  3. a statement and its negation are asked separately — nothing forces
     p(yes) + p(not yes) = 1, so ask each decision ONE way in production

Run: .venv/bin/python week24/02_three_primitives/labs/lab02_noul.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, noul, show, step, table, validate  # noqa: E402

banner("Lab 02-2 · noul", "a probability that a proposition is true")

# ── 1. yes / no / unsure ───────────────────────────────────────────────────
step(1, "same question, three messages: clear yes, clear no, unsure")
Q1 = {"reports_problem": noul(
    "Does `message` report a problem with the room or its equipment?",
    true="The guest says something in the room is broken, dirty, too hot/cold or not working",
    false="No problem with the room is mentioned",
)}
rows = []
for msg in ["The TV remote is not working.",
            "What time is breakfast?",
            "The room is fine I guess, but it smells a bit different from last time."]:
    resp = ask({"message": msg}, Q1, quiet=True)
    v = validate(resp, Q1)["reports_problem"]["noul"]
    rows.append([msg, f"{v:.2f}", "yes" if v >= .8 else "no" if v <= .2 else "UNSURE → review"])
table(rows, ["message", "noul", "reading"])
print("◆ 0.5 does not mean 'half a problem'. It means the model cannot tell.")

# ── 2. multi-label: one noul per label ─────────────────────────────────────
step(2, "multi-label — a message can need SEVERAL teams: one noul per label")
MSG2 = {"message": "The AC is dripping water onto the carpet and the minibar bill looks wrong."}
Q2 = {
    "needs_hvac": noul("Does `message` describe a heating or air-conditioning problem?"),
    "needs_housekeeping": noul("Does `message` require cleaning or drying something in the room?"),
    "needs_billing": noul("Does `message` raise a question or complaint about a bill or charge?"),
    "needs_security": noul("Does `message` report a safety or security threat?"),
}
resp = ask(MSG2, Q2)
a = validate(resp, Q2)
show(resp)
total = sum(x["noul"] for x in a.values())
print(f"◆ sum of the four nouls = {total:.2f} — independent yes/no questions do NOT sum to 1")
chosen = [k for k, x in a.items() if x["noul"] >= 0.5]
print(f"→ labels at ≥ 0.5: {chosen}  (a Choice would have forced just ONE of these)")

# ── 3. proposition vs its negation ─────────────────────────────────────────
step(3, "ask a statement AND its negation — are they complementary?")
MSG3 = {"message": "I might be checking out a day early, not sure yet."}
Q3 = {
    "leaving_early": noul("Is the guest in `message` checking out early?"),
    "not_leaving_early": noul("Is the guest in `message` NOT checking out early?"),
}
resp = ask(MSG3, Q3)
a = validate(resp, Q3)
show(resp)
s = a["leaving_early"]["noul"] + a["not_leaving_early"]["noul"]
print(f"◆ p(yes) + p(negation) = {s:.2f} — nothing guarantees exactly 1.00")
print("═ Rule: ask each decision ONE way, and enforce logical identities in code.")
