#!/usr/bin/env python3
"""CHECKPOINT 4 — Sub-agent delegation (in-process) and where it hits a wall.

Goal: a long-running coordinator shouldn't carry every skill itself. ADK lets it
delegate to focused `sub_agents`, each with a narrow prompt and tool set — which
keeps reasoning sharp over multi-day contexts (a fat do-everything prompt rots).
The coordinator delegates by transferring control to a named sub-agent.

The pivot: ADK sub_agents work beautifully WHEN YOU OWN EVERY AGENT and they run
in one Python process. But the parts vendor is a different company, on a
different framework, behind their own API. You cannot `import` their Agent into
your `sub_agents=[...]` list. That boundary is exactly what A2A solves — and is
why Checkpoint 5 exists.

What this demonstrates:
  Part 1 — build a coordinator with an in-house `energy_agent` sub-agent (real
           ADK wiring if installed).
  Part 2 — simulate the delegation decision offline.
  Part 3 — name the wall: the vendor can't be a sub_agent → A2A (CP5).

(Week 17 · fleet orchestration · ADK blog: sub_agents delegation)

Run:  python week17/checkpoints/checkpoint4_sub_agents.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # config.py
from config import MODEL, WorkOrderStep, banner, have_adk, step


# In-house tool the energy sub-agent owns (would reach Neo4j via MCP in prod).
def find_low_occupancy_window(room: str, tool_context=None) -> dict:
    """Find the next low-occupancy window to schedule a noisy repair in `room`.

    Args:
        room: Room id, e.g. 'R305'.
    """
    return {"room": room, "window": "Tue 02:00–05:00", "expected_occupancy": "low"}


def build_coordinator_with_subagent():
    """Real ADK wiring: a coordinator that owns an in-house energy sub-agent.
    Only importable with google-adk installed."""
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool

    energy_agent = Agent(
        name="energy_agent",
        model=MODEL,
        description="Schedules repairs into low-occupancy, low-energy windows.",
        instruction=(
            "You are the hotel Energy agent. Given a room, call "
            "find_low_occupancy_window to pick the least disruptive repair slot, "
            "then transfer control back to the coordinator."
        ),
        tools=[FunctionTool(find_low_occupancy_window)],
    )

    coordinator = Agent(
        name="hotel_maintenance_coordinator",
        model=MODEL,
        description="Coordinates long-running hotel maintenance work-orders.",
        instruction=(
            "Coordinate the work-order. When a repair is ready to schedule, "
            "delegate to the energy_agent to pick a low-occupancy window."
        ),
        sub_agents=[energy_agent],     # ← in-process delegation
    )
    return coordinator, energy_agent


def simulate_delegation():
    """Offline: show the coordinator deciding to hand a sub-task to energy_agent."""
    state = {"current_step": WorkOrderStep.PART_DELIVERED,
             "fault_details": {"room": "R305"}}
    step(f"coordinator at {state['current_step']} — repair is ready to schedule")
    step("   decision: delegate scheduling to in-house 'energy_agent'")
    result = find_low_occupancy_window(state["fault_details"]["room"])
    step(f"   energy_agent → {result['window']} (occupancy: {result['expected_occupancy']})")
    step("   control transferred back to coordinator")
    return result


def main():
    banner("CP4 · Sub-agent delegation (in-process) — and where it hits a wall")

    if have_adk():
        coordinator, energy_agent = build_coordinator_with_subagent()
        step("Real ADK hierarchy constructed:")
        step(f"   {coordinator.name}")
        step(f"     └─ sub_agent: {energy_agent.name}  "
             f"(tools={[t.name for t in energy_agent.tools]})")
    else:
        step("google-adk not installed — skipping live wiring "
             "(`pip install google-adk`).")

    print()
    step("Simulating delegation:")
    simulate_delegation()

    print()
    step("THE WALL — why sub_agents isn't enough for a real fleet:")
    step("   • sub_agents must be imported into ONE Python process.")
    step("   • The parts VENDOR is another company, another framework, own API.")
    step("   • You can't `import vendor.Agent` — there is a network + org boundary.")
    step("   ⇒ delegate across that boundary with A2A (agent cards). See CP5.")
    print()
    step("MCP vs A2A in one line: MCP is VERTICAL (an agent → its tools, e.g.")
    step("find_low_occupancy_window → Neo4j). A2A is HORIZONTAL (agent → another")
    step("agent across services). Production fleets use BOTH.")


if __name__ == "__main__":
    main()
