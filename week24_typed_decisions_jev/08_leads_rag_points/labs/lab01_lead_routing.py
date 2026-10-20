#!/usr/bin/env python3
"""Lab 08-1 · Route sales inquiries to the right sales-engineering queue.

Five synthetic inquiries → solution family (choice) + buying stage (score) +
missing scope (noul) → a queue proposal and a follow-up checklist.
No email is sent, no price is quoted, no savings are promised.

Run: .venv/bin/python week24/08_leads_rag_points/labs/lab01_lead_routing.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, guarded, noul, score, show, step, table, top2, validate  # noqa: E402

LEADS = Path(__file__).resolve().parents[1] / "data" / "leads.jsonl"

QUESTIONS = guarded({
    "solution": choice("Which AltoTech solution family best matches the stated need in `message`?", {
        "air_side": "Split-type, VRF or room/zone air-conditioning optimization.",
        "water_side": "Chiller plant, pumps or cooling-tower optimization.",
        "portfolio": "Multi-property energy visibility and comparison.",
        "carbon": "Carbon baseline, emissions or sustainability reporting.",
        "unknown": "No clear fit, insufficient details or unrelated need.",
    }),
    "buying_stage": score("What buying intent is explicitly evidenced in `message`? Do not infer budget or purchase authority.", [
        "General information only.",
        "Exploring a concrete site requirement.",
        "Explicit request for proposal, pilot, quotation or procurement.",
    ]),
    "scope_missing": noul("Are important scoping details such as building type, HVAC system or target outcome absent from `message`?"),
})

FOLLOW_UP = ["HVAC configuration", "operating hours", "meter coverage", "building type", "target outcome"]


def accepted(a: dict) -> bool:
    p1, p2 = top2(a["probabilities"])
    return a["confidence"] >= 0.75 and p1 >= 0.80 and p1 - p2 >= 0.20


def policy(a: dict) -> dict:
    label = a["solution"]["choice"]
    return {"route": label if accepted(a["solution"]) and label != "unknown" else "sales_review",
            "scope_followup": a["scope_missing"]["noul"] >= 0.50,
            "execute": False, "no_outreach": True}


def main() -> None:
    banner("Lab 08-1 · sales-lead routing", "solution family · buying stage · missing scope → queue proposal")
    leads = [json.loads(l) for l in LEADS.read_text(encoding="utf-8").splitlines() if l.strip()]

    step(1, "one inquiry in detail")
    print(f"   “{leads[2]['state']['message']}”")
    r = ask(leads[2]["state"], QUESTIONS)
    validate(r, QUESTIONS)
    show(r, top=3)

    step(2, f"route all {len(leads)} inquiries")
    rows = []
    for lead in leads:
        a = validate(ask(lead["state"], QUESTIONS, quiet=True), QUESTIONS)
        rec = policy(a)
        rows.append([lead["id"], lead["state"]["message"][:44], a["solution"]["choice"],
                     f"{a['solution']['confidence']:.2f}", f"{a['buying_stage']['score']:.2f}",
                     f"{a['scope_missing']['noul']:.2f}", rec["route"], "yes" if rec["scope_followup"] else "-"])
    table(rows, ["id", "message", "solution", "conf", "stage 0-2", "scope?", "route", "follow-up"])

    step(3, "what the numbers do NOT mean")
    print("⚠ buying_stage 2.0 = 'the message explicitly asks for a quote'. It is NOT a win")
    print("  probability, a budget, purchase authority or creditworthiness.")
    print(f"→ follow-up drafts (by a generative model, left UNSENT) ask for: {', '.join(FOLLOW_UP)}")
    print("═ execute: False · no_outreach: True · no_price_quoted: True · no_savings_promised: True")


if __name__ == "__main__":
    main()
