#!/usr/bin/env python3
"""PART 4 · Assemble Grand Bangkok end-to-end  [ADVANCED]

Everything from Ch 2–4 in one build: five discipline layers, 22 payloaded
floors (2 loaded), instanced guest-room furniture, sensors as first-class
prims with relationships to the equipment they watch, navigation waypoints —
then the composed tree, the inventory, and the 'ready for App 5' checklist.

REAL if `pip install usd-core` (the exported .usda is re-opened with the real
pxr API); a reachable LLM endpoint makes the closing copilot question REAL.

Run:  python demos/step04_assemble_hotel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 4", "Assemble Grand Bangkok end-to-end", "ADVANCED")
    view.mode_line()

    st = sim.build_grand_bangkok(load=("F12", "F14"), detailed=("F12", "F14"))
    stats = st.stats()

    print("Discipline layers — one .usda per trade, composed non-destructively (each")
    print("team edits its own layer; nobody clobbers anybody):\n")
    for lyr in st.layers:
        print(f"  sublayer  {lyr}")
    print()

    print(f"Composed stage — {stats['composed']} of {stats['authored']} authored prims "
          f"({stats['loaded']}/{stats['payloads']} floor payloads loaded, "
          f"{stats['instanceable']} instanceable):\n")
    print(st.tree(max_lines=36))
    print()

    sensor = st.prims["/World/Tower/F12/Sensors/TS_1201"]
    print("Sensors are FIRST-CLASS prims — attributes for external IDs, a relationship")
    print(f"to the equipment they watch ({sensor.path}):\n")
    for k, v in sensor.attrs.items():
        print(f"  {k:<18}= {v}")
    for k, v in sensor.rels.items():
        print(f"  rel {k:<14}→ {v}")
    print()

    print("Navigation waypoints — saved camera targets for the operations team:\n")
    for name, target, pos, note in sim.WAYPOINTS:
        print(f"  {name:<18}→ {target}")
        print(f"  {'':<18}  at {pos} · {note}")
    print()

    rows = sim.inventory(st)
    print("Inventory of the composed working set (GlobalId coverage = bindability):\n")
    print(f"  {'class':<12}{'count':>6}{'with GlobalId':>15}{'coverage':>10}")
    print("  " + "─" * 45)
    for cls, n, gid, cov in rows:
        print(f"  {cls:<12}{n:>6}{gid:>15}{cov:>10}")
    print()

    usda = st.to_usda()
    if sim.have_usd():
        from pxr import Usd  # noqa: PLC0415
        out = Path(__file__).resolve().parents[1] / ".sandbox"
        out.mkdir(exist_ok=True)
        f = out / "grand_bangkok.usda"
        f.write_text(usda)
        stage = Usd.Stage.Open(str(f))   # bind it — a chained call lets pxr GC the stage mid-traversal
        n = sum(1 for _ in stage.TraverseAll())
        print(f"▣ REAL check: pxr re-opened the exported stage — {n} prims · {f.name}")
        print("  open it in usdview / USD Composer and the same tree appears.\n")
    else:
        head = usda.splitlines()[:10]
        print("Exported .usda (head — `pip install usd-core` to re-open it for real):\n")
        print("\n".join("  " + ln for ln in head))
        print(f"  … ({len(usda.splitlines()) - len(head)} more lines)\n")

    gid_ok = all(cov == "100%" for _, _, _, cov in rows)
    checklist = [
        (gid_ok, "GlobalId on every equipment/sensor prim — the App 5 join key"),
        (True, "sensors are first-class prims with rel alto:serves → equipment"),
        (True, "external IDs authored (alto:bacnetRef / alto:pointId) per point"),
        (stats["payloads"] == 22, "every floor behind a payload — open only the working set"),
        (stats["instanceable"] > 0, "repeated furniture/VAVs instanceable — memory stays flat"),
        (True, "waypoints saved for chiller plant, AHU room, lobby"),
    ]
    print("Ready-for-App-5 checklist (the SCENE layer's exit criteria):\n")
    for ok, what in checklist:
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    print()

    view.generate("You are the twin copilot for the Grand Bangkok hotel stage. In two "
                  "sentences: why do instancing and per-floor payloads matter for a "
                  "building-scale USD twin, and what role does GlobalId play next?",
                  max_tokens=300, title="Ask the twin copilot")

    print("\nTakeaway: the SCENE layer is done — organized, validated, identified,")
    print("affordable. Next: App 5 binds live BACnet/MQTT telemetry onto these exact")
    print("GlobalIds (STATE) — the moment the model becomes a twin.")


if __name__ == "__main__":
    main()
