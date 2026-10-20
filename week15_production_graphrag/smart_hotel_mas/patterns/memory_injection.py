#!/usr/bin/env python3
"""Memory Injection Pattern. (smart_hotel_mas.pdf §"Memory Injection Pattern")

The four memory tiers (L1 working, L2 episodic, L3 semantic, L4 knowledge graph)
are useless to an LLM agent unless their contents are turned into *natural
language* and spliced into the agent's prompt at run time. This module shows the
canonical injection routine used by every agent in the MAS:

    build_agent_context()  -> assembles a compact, multi-tier context block
    inject_context_into_agent() -> prepends that block to a CrewAI agent backstory
    run_agent_cycle()      -> a demo of building context + measuring its size

Key Insight:
  * Convert stored data to natural language — agents reason over prose, not rows.
  * Keep the block under ~500 tokens; the prompt is precious budget.
  * L1 = what I am doing *now*; L2 = what happened *recently*; L4 = structured
    active alerts. (L3 semantic recall is injected per-query, not here.)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # patterns/ -> workshop root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "checkpoints"))

from config import check_neo4j, MODEL  # noqa: F401  (MODEL per workshop convention)
from checkpoint2_memory import WorkingMemory, EpisodicMemory

# checkpoint6 (HotelKGMemory) is the heavy L4 wrapper; it is optional for this
# pattern demo, so guard the import and degrade gracefully when absent.
try:
    from checkpoint6_full_mas import HotelKGMemory
except ImportError:
    HotelKGMemory = None  # type: ignore


def build_agent_context(working_mem, episodic_mem, kg_mem, agent_id: str) -> str:
    """Assemble a compact natural-language context block for ``agent_id``.

    Concatenates three tiers into a single prompt-ready string (target <500
    tokens):
        L1  working_mem.summary()
        L2  episodic_mem.recall_recent(hours=1, limit=5)  -> formatted lines
        L4  kg_mem.query_active_alerts()[:3]              -> formatted lines
    Returns the tiers joined by blank lines.
    """
    blocks = [f"[Agent: {agent_id}]"]

    # ── L1: Working Memory (current task state) ──────────────────────────────
    blocks.append("## CURRENT TASK STATE (L1)\n" + working_mem.summary())

    # ── L2: Episodic Memory (what happened recently) ─────────────────────────
    recent = episodic_mem.recall_recent(hours=1, limit=5)
    if recent:
        lines = [
            f"- [{ep['ts']}] {ep['agent']} · {ep['type']} → {ep['data']}"
            for ep in recent
        ]
        blocks.append("## RECENT EVENTS (L2, last hour)\n" + "\n".join(lines))
    else:
        blocks.append("## RECENT EVENTS (L2, last hour)\n- (none)")

    # ── L4: Knowledge Graph active alerts (structured) ───────────────────────
    if kg_mem is not None:
        try:
            alerts = kg_mem.query_active_alerts()[:3]
            if alerts:
                lines = [
                    f"- {a.get('alert_type', a.get('type', '?'))} "
                    f"severity={a.get('severity', '?')} "
                    f"room={a.get('room', a.get('room_id', '?'))}"
                    for a in alerts
                ]
                blocks.append("## ACTIVE ALERTS (L4)\n" + "\n".join(lines))
            else:
                blocks.append("## ACTIVE ALERTS (L4)\n- (none active)")
        except Exception as e:  # noqa: BLE001
            blocks.append(f"## ACTIVE ALERTS (L4)\n- (unavailable: {type(e).__name__})")
    else:
        blocks.append("## ACTIVE ALERTS (L4)\n- (L4 not connected)")

    return "\n\n".join(blocks)


def inject_context_into_agent(agent, context: str):
    """Prepend the assembled ``context`` to a CrewAI agent's backstory.

    Works on any object exposing a ``.backstory`` string attribute.
    """
    existing = getattr(agent, "backstory", "") or ""
    agent.backstory = f"{context}\n\n---\n\n{existing}"
    return agent


def run_agent_cycle(working_mem, episodic_mem, kg_mem):
    """Demo one cycle: build the context and report its approximate token count."""
    context = build_agent_context(working_mem, episodic_mem, kg_mem, "SensorAgent")
    approx_tokens = len(context.split())  # rough word-count proxy for tokens
    print(context)
    print("\n" + "─" * 60)
    print(f"Approx context size: {approx_tokens} tokens "
          f"({'OK' if approx_tokens < 500 else 'OVER BUDGET — trim!'} vs <500 target)")
    return context


if __name__ == "__main__":
    # ── L1: a couple of set() calls ──────────────────────────────────────────
    wm = WorkingMemory()
    wm.set("current_batch", ["R101", "R102", "R103"])
    wm.set("batch_avg_temp_c", 24.1)

    # ── L2: in-memory SQLite with a couple of store() calls ──────────────────
    em = EpisodicMemory(":memory:")
    em.store("SensorAgent", "sensor_batch_read",
             {"rooms": ["R101", "R102", "R103"], "count": 3, "avg_temp_c": 24.1},
             tags=["batch"])
    em.store("AlertAgent", "alert_triggered",
             {"room": "R301", "type": "HIGH_TEMP", "value": 28.5, "threshold": 27.0},
             tags=["alert"])

    # ── L4: only if Neo4j + HotelKGMemory are available ──────────────────────
    kg = None
    if HotelKGMemory is not None and check_neo4j():
        try:
            kg = HotelKGMemory("MemoryInjectionDemo")
        except Exception as e:  # noqa: BLE001
            print(f"(L4 skipped — HotelKGMemory init failed: {type(e).__name__})")
    else:
        print("(L4 skipped — Neo4j/HotelKGMemory unavailable; "
              "context will note no active alerts)")

    print()
    run_agent_cycle(wm, em, kg)
    em.close()
