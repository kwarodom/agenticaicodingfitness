#!/usr/bin/env python3
"""CHECKPOINT 6 — The full fleet: ADK long-running + A2A across services.

Goal: tie CP1–CP5 into one end-to-end story — a single hotel maintenance
work-order that is BOTH long-running (durable, pause/resume) AND distributed
(delegates across a service boundary to the vendor via A2A, and to an in-house
energy agent in-process). This is "L3 — Fleet Orchestration": the layer above
the in-process MAS from week15.

The architecture (MCP vertical · A2A horizontal):

    MaintenanceCoordinator  (ADK, durable session state — CP1/CP2)
       │
       ├─ in-process sub_agent ── EnergyAgent ──MCP──▶ occupancy/energy data   (CP4)
       │
       └─ A2A across services ──▶ Acme HVAC Parts Agent  (other org/framework)  (CP5)
                                     └─ input-required → resumed via webhook     (CP3)

The whole flow runs OFFLINE: durable state in SQLite, the vendor and LLM mocked.
Each step prints what happened and how the work-order's persisted step advances
OPEN → DIAGNOSED → AWAITING_PART → PART_DELIVERED → REPAIRED → CLOSED.

(Week 17 capstone · long-running agents + A2A · maps to week15 L3 fleet layer)

Run:  python week17/checkpoints/checkpoint6_fleet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling checkpoints
from config import DB_PATH, MockLLM, WorkOrderStep, banner, step

from checkpoint2_durable_sessions import DurableWorkOrderStore
from checkpoint4_sub_agents import find_low_occupancy_window
from checkpoint5_a2a_cards import VENDOR_CARD, MockVendorAgent, validate_card

WO_ID = "WO-2207"


def main():
    banner("CP6 · Full fleet — ADK long-running + A2A across services")

    # fresh durable store so the capstone is reproducible
    if DB_PATH.exists():
        DB_PATH.unlink()
    store = DurableWorkOrderStore()
    llm = MockLLM()

    # ── 1. OPEN → DIAGNOSED (coordinator, durable) ──────────────────────────
    step("[coordinator] guest reports R305 HVAC not cooling — opening work-order")
    store.create_session(WO_ID, {
        "work_order_id": WO_ID, "current_step": WorkOrderStep.OPEN,
        "fault_details": {}, "pending_signals": [],
    })
    state = store.get_session(WO_ID)
    state["fault_details"] = {"room": "R305", "symptom": "HVAC not cooling",
                              "part_needed": "COMP-24K-BTU"}
    state["current_step"] = WorkOrderStep.DIAGNOSED
    store.save_state(WO_ID, state)
    step(f"[coordinator] diagnosed → needs part COMP-24K-BTU  (persisted: {state['current_step']})")
    print()

    # ── 2. A2A across services: delegate the part order to the vendor ───────
    step("[A2A ▶ vendor] coordinator can't import the vendor's agent — delegating")
    step("              over A2A (horizontal). Discovering its card first:")
    validate_card(VENDOR_CARD)
    skill_id = llm("order the replacement compressor part from the vendor")
    step(f"              matched skill '{skill_id}' on {VENDOR_CARD['name']}")

    vendor = MockVendorAgent()
    task = vendor.submit(skill_id, {"sku": "COMP-24K-BTU", "room": "R305", "urgency": "rush"})
    step(f"              task {task['task_id']} → {task['state']}")

    # work-order PAUSES here while the vendor task is open (durable)
    state["current_step"] = WorkOrderStep.AWAITING_PART
    state["pending_signals"] = ["part_delivered"]
    store.save_state(WO_ID, state)
    step(f"[coordinator] PAUSED at {state['current_step']} — process can scale to zero")
    print()

    # vendor needs input (HITL across the network), caller answers, task completes
    if task["state"] == "input-required":
        step(f"[A2A ◀ vendor] input-required: {task['question']!r}")
        task = vendor.resume(task["task_id"], "B")
        step(f"[A2A ◀ vendor] {task['state']} — eta {task['result']['eta_days']}d, "
             f"tracking {task['result']['tracking_id']}")
    print()

    # ── 3. Days later: delivery webhook resumes the work-order (CP3) ─────────
    step("[webhook] *** 2 days pass — vendor's logistics POSTs part_delivered ***")
    state = store.get_session(WO_ID)           # fresh read (≙ woken container)
    assert state["current_step"] == WorkOrderStep.AWAITING_PART
    state.update({"current_step": WorkOrderStep.PART_DELIVERED, "pending_signals": []})
    state["fault_details"]["tracking_id"] = task["result"]["tracking_id"]
    store.save_state(WO_ID, state)
    step(f"[coordinator] resumed via state_delta → {state['current_step']}")
    print()

    # ── 4. In-process sub-agent: energy agent schedules the repair (CP4) ────
    step("[sub_agent ▶ energy] delegate scheduling to in-house EnergyAgent (in-process)")
    window = find_low_occupancy_window("R305")
    step(f"[sub_agent ◀ energy] low-occupancy window: {window['window']}")
    print()

    # ── 5. REPAIRED → CLOSED (coordinator, durable) ─────────────────────────
    state["current_step"] = WorkOrderStep.REPAIRED
    store.save_state(WO_ID, state)
    step(f"[coordinator] technician completed fix in {window['window']} → {state['current_step']}")
    state["current_step"] = WorkOrderStep.CLOSED
    state["pending_signals"] = []
    store.save_state(WO_ID, state)
    step(f"[coordinator] verified + guest notified → {state['current_step']}")
    print()

    # ── verify the durable trail ────────────────────────────────────────────
    final = store.get_session(WO_ID)
    assert final["current_step"] == WorkOrderStep.CLOSED
    step("FLEET SUMMARY")
    step(f"   work-order {WO_ID}: {' → '.join(WorkOrderStep.ORDER)}")
    step(f"   final persisted state: {final['current_step']}  ✅")
    print()
    step("WHAT YOU JUST SAW:")
    step("   • Long-running: state lived in SQLite across a multi-day pause (ADK")
    step("     DatabaseSessionService); resume was event-driven, not polled.")
    step("   • Distributed: the vendor was reached HORIZONTALLY via A2A (its card +")
    step("     task lifecycle + input-required), the energy agent VERTICALLY in-")
    step("     process. MCP would be each agent's link down to its own tools/data.")
    step("   • This is the L3 fleet layer above week15's single-process hotel MAS.")


if __name__ == "__main__":
    main()
