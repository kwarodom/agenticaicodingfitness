#!/usr/bin/env python3
"""Lab 08-3 · Suggest a semantic kind for BMS points — for a curator to review.

Six point descriptions → an internal semantic kind (NOT an ontology URI) + an
ambiguity signal → curator_review. Nothing is written to the building graph.

Run: .venv/bin/python week24/08_leads_rag_points/labs/lab03_point_mapping.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, guarded, noul, step, table, top2, validate  # noqa: E402

POINTS = Path(__file__).resolve().parents[1] / "data" / "points.jsonl"

QUESTIONS = guarded({
    "kind": choice("Classify `point_description` by operational meaning. Select only an allowed semantic "
                   "kind, not a new ontology URI.", {
        "supply_air_temperature_sensor": "Measured supply air temperature, not its target.",
        "zone_air_temperature_sensor": "Measured room or zone air temperature, not its target.",
        "zone_air_temperature_setpoint": "Target room or zone air temperature.",
        "unknown": "Ambiguous acronym, insufficient context or none of these.",
    }),
    "ambiguous": noul("Is there insufficient evidence to distinguish the measured quantity, location or "
                      "sensor versus setpoint role?"),
})


def proposed_kind(a: dict, max_ambiguity: float = 0.20) -> str:
    """Accept Jev's kind only if it is confident AND the ambiguity signal is low."""
    p1, p2 = top2(a["kind"]["probabilities"])
    ok = a["kind"]["confidence"] >= 0.75 and p1 >= 0.80 and p1 - p2 >= 0.20
    return a["kind"]["choice"] if ok and a["ambiguous"]["noul"] <= max_ambiguity else "unknown"


def main() -> None:
    banner("Lab 08-3 · BMS point mapping", "semantic kind + ambiguity → curator_review (no graph write)")
    points = [json.loads(l) for l in POINTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    answers = [validate(ask(p["state"], QUESTIONS, quiet=True), QUESTIONS) for p in points]

    step(1, f"{len(points)} points → proposals (ambiguity gate ≤ 0.20)")
    rows = [[p["id"], p["state"]["point_description"][:46], a["kind"]["choice"],
             f"{a['kind']['confidence']:.2f}", f"{a['ambiguous']['noul']:.2f}", proposed_kind(a)]
            for p, a in zip(points, answers)]
    table(rows, ["id", "point_description", "jev kind", "conf", "ambig", "proposed"])

    step(2, "same answers, different gate — NO new model calls")
    for gate in (0.20, 0.40):
        kinds = [proposed_kind(a, gate) for a in answers]
        accepted = sum(k != "unknown" for k in kinds)
        print(f"◆ ambiguity ≤ {gate:.2f}: {accepted}/{len(points)} proposed → "
              + ", ".join(f"{p['id']}={k}" for p, k in zip(points, kinds)))
    print("   A gate chosen without data can silently reject EVERYTHING. Pick it on labelled")
    print("   examples (Lab 09) — changing a threshold never needs another API call.")

    step(3, "what a curator does next")
    print("→ map ACCEPTED kinds to classes via a registry pinned to your approved Brick version")
    print("→ verify unit, equipment, location, sensor-vs-setpoint role and provenance")
    print("→ validate the proposed graph separately (brickschema Graph.validate() / SHACL)")
    print("⚠ pt-5 is a SUPPLY-AIR SETPOINT — not one of the four options. Did 'unknown' win?")
    print("═ execute: False · route: curator_review · graph_write: False · needs_ontology_and_shacl_validation: True")


if __name__ == "__main__":
    main()
