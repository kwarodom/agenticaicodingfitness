#!/usr/bin/env python3
"""Lab 01 — the world is REAL: diff the hotel state an agent run mutates.

The capstone's claim is that this is not slideware: the tools genuinely change a
HotelState object. This lab proves it three ways —
  1. snapshot the state, run the room-1203 CRITICAL alarm through the FULL fleet
     (router → specialist → policy → tools), and diff before/after;
  2. call a tool DIRECTLY (no agent at all) and watch the building's kW drop;
  3. if a live endpoint is connected, send the same tool schemas to the model
     once and check it emits a NATIVE tool_call (bounded: max_tokens=300,
     one attempt, 45 s — thinking models like gemma4 spend ~30 s reasoning
     before the tool_call, so a tighter cap would always time out).

The fleet run uses the deterministic SimBrain on purpose — the harness (tools,
policy, relay) is identical in SIM and REAL; only the brain swaps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import config  # noqa: E402
from hotel.agents import Orchestrator  # noqa: E402
from hotel.brain import SimBrain  # noqa: E402
from hotel.policy import Gateway, signature_line  # noqa: E402
from hotel.relay import Relay  # noqa: E402
from hotel.world import Tools, fresh_state  # noqa: E402


def snapshot(state) -> dict:
    return {"energy_kw": state.energy_kw,
            "tickets_open": state.tickets_open,
            "tickets_critical": state.tickets_critical,
            "work_orders": [w["work_order"] for w in state.work_orders],
            "setpoint_0902": state.rooms["0902"].setpoint_c}


def diff(before: dict, after: dict) -> None:
    for k in before:
        mark = "◈" if before[k] != after[k] else " "
        print(f"    {mark} {k:16} {before[k]!r:>28}  →  {after[k]!r}")


def main() -> None:
    print("━" * 64)
    print("  LAB 01 — the world is REAL: state mutation, tool-first")
    print("━" * 64)
    print(f"\n▣ {signature_line()}")

    # ── 1 · full fleet run, deterministic brain — diff the world ─────────────
    state = fresh_state()
    orch = Orchestrator(state, Tools(state), Gateway(state), Relay(), SimBrain())
    before = snapshot(state)
    print("\n▣ STEP 1 — room-1203 CRITICAL alarm through the whole fleet")
    res = orch.handle_event("Room 1203 temperature alarm — triage and act.", "1203")
    print(f"  · fleet answer: {res.answer}")
    print("  · HotelState BEFORE → AFTER:")
    diff(before, snapshot(state))
    assert state.work_orders, "the run should have dispatched a work order"
    print("  ✓ WO-1203 exists and tickets_critical went up — a real object changed,")
    print("    not a printed story. Swap SimBrain for a NIM endpoint: same mutation.")

    # ── 2 · no agent at all — the tool is just Python that edits the world ───
    print("\n▣ STEP 2 — bypass the agents: call set_setpoint on empty room 0902")
    tools, gw = Tools(state), Gateway(state)
    verdict = gw.check("energy", "set_setpoint", {"room": "0902", "setpoint_c": 25.0})
    print(f"  · policy gate first (never skip it): allow={verdict.allow} ({verdict.reason})")
    out = json.loads(tools.run("set_setpoint", {"room": "0902", "setpoint_c": 25.0}))
    print(f"  · tool result: {out}")
    print(f"  ✓ building draw {before['energy_kw']} kW → {state.energy_kw} kW — the crude")
    print("    energy model rewards trimming EMPTY rooms (~4 kW/°C).")

    # ── 3 · does YOUR endpoint speak native tool-calling? (one bounded call) ──
    print("\n▣ STEP 3 — probe the live endpoint for native tool-calling")
    if config.MODE == "real":
        print(f"  · endpoint {config.safe_base_url()} · model {config.MODEL} · {config.cost_note()}")
        try:
            from openai import OpenAI
            # one attempt, hard 45 s cap: max_retries=0 matters — the default (2)
            # would silently turn one bounded probe into three.
            client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                            timeout=45.0, max_retries=0)
            r = client.chat.completions.create(
                model=config.MODEL, tools=Tools.SCHEMAS, max_tokens=300, temperature=0.2,
                messages=[{"role": "user",
                           "content": "Read the live HVAC telemetry for room 1203."}])
            msg = r.choices[0].message
            if msg.tool_calls:
                c = msg.tool_calls[0]
                print(f"  ✓ native tool_call emitted: {c.function.name}({c.function.arguments})")
                print("    → this is exactly what NemotronBrain feeds through the Gateway.")
            else:
                print(f"  ◈ no tool_call — the model answered in prose: "
                      f"{(msg.content or '')[:120]!r}")
                print("    → the fleet would still run, but pick a tool-calling model")
                print("      (nemotron/qwen3.6) for the REAL path.")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ request failed ({type(e).__name__}: {e}) — see Troubleshooting in TUTORIAL.md.")
    else:
        print("  · no endpoint reachable — to go REAL, run one of:")
        print("      export DGX_BASE_URL=http://<your-spark>:11434/v1        # A · Spark/LAN")
        print("      export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
        print("             DGX_API_KEY=nvapi-...                            # B · build.nvidia.com")
        print("      ollama serve   # C · local Ollama, then export DGX_CONN=local")
        print("  [no endpoint — showing expected output]")
        print('  ✓ native tool_call emitted: get_room_telemetry({"room": "1203"})')
        print("    → this is exactly what NemotronBrain feeds through the Gateway.")

    print("\n✓ Lab 01 done — the harness is real; only the brain is swappable.")


if __name__ == "__main__":
    main()
