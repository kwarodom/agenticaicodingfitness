#!/usr/bin/env python3
"""CHECKPOINT 2 — Durable sessions: pause for days, resume with no context loss.

Goal: prove that the work-order's state survives a process restart. A chatbot
that keeps state in memory loses everything when the container scales to zero
over a quiet weekend. A long-running agent persists state to a database, so the
"3-day pause" is just a row sitting in SQLite until something wakes it.

In ADK this is `DatabaseSessionService` — point it at a SQLAlchemy URL and your
session state is durable; the same config works for local SQLite and prod Cloud
SQL (only the URL changes). We show that real config, then run a zero-dependency
SQLite mirror so you can WATCH the state survive a restart with no credentials.

What this demonstrates:
  Phase A — create a work-order, advance it to AWAITING_PART, persist, "exit".
  Phase B — a fresh store object (≙ a new container) re-reads the SAME db file
            and finds the work-order exactly where it paused.

(Week 17 · long-running agents · ADK blog: DatabaseSessionService)

Run:  python week17/checkpoints/checkpoint2_durable_sessions.py
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import config.py
from config import APP_NAME, DB_PATH, DB_URL, WorkOrderStep, banner, have_adk, step


# ── How you'd configure the SAME persistence in real ADK ────────────────────
def show_adk_config() -> None:
    """ADK gives you durable sessions for free via DatabaseSessionService. This
    is the production wiring (from the blog's FastAPI server) — we print it so
    you see what the SQLite mirror below stands in for."""
    snippet = f'''    from google.adk.sessions.database_session_service import DatabaseSessionService

    # local dev → SQLite file; production → swap the URL for Cloud SQL.
    session_service = DatabaseSessionService(db_url="{DB_URL}")
    session = await session_service.create_session(
        app_name="{APP_NAME}", user_id="front_desk", session_id="WO-1042",
        state={{"current_step": "OPEN"}},
    )
    # ...state mutated by tools (CP1) is persisted automatically...
    # A brand-new process can later: await session_service.get_session(...)
    # and read session.state back — nothing is held in memory.'''
    step("Real ADK persistence (DatabaseSessionService):")
    print(snippet)


# ── Zero-dependency mirror of what DatabaseSessionService does ──────────────
class DurableWorkOrderStore:
    """A tiny SQLite-backed session store: one row per work-order, state kept as
    a JSON blob. This is exactly the shape DatabaseSessionService persists for
    you — shown directly so the demo runs with no google-adk and no network."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "  app_name TEXT, session_id TEXT, state_json TEXT,"
                "  PRIMARY KEY (app_name, session_id))"
            )

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def create_session(self, session_id: str, state: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
                (APP_NAME, session_id, json.dumps(state)),
            )

    def get_session(self, session_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT state_json FROM sessions WHERE app_name=? AND session_id=?",
                (APP_NAME, session_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_state(self, session_id: str, state: dict) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE sessions SET state_json=? WHERE app_name=? AND session_id=?",
                (json.dumps(state), APP_NAME, session_id),
            )


WO_ID = "WO-1042"


def phase_a_open_and_pause() -> None:
    step("PHASE A — open the work-order and run it to AWAITING_PART")
    store = DurableWorkOrderStore()
    store.create_session(WO_ID, {
        "work_order_id": WO_ID,
        "current_step": WorkOrderStep.OPEN,
        "fault_details": {},
        "pending_signals": [],
    })
    # advance (same transitions as CP1, persisted after each)
    state = store.get_session(WO_ID)
    state["fault_details"] = {"room": "R305", "symptom": "HVAC not cooling",
                              "part_needed": "COMP-24K-BTU"}
    state["current_step"] = WorkOrderStep.DIAGNOSED
    store.save_state(WO_ID, state)

    state["current_step"] = WorkOrderStep.AWAITING_PART
    state["pending_signals"] = ["part_delivered"]
    store.save_state(WO_ID, state)
    step(f"   persisted {WO_ID} at {state['current_step']} → {DB_PATH.name}")
    step("   *** process exits here — imagine a 3-day weekend pause ***")


def phase_b_fresh_process() -> None:
    step("PHASE B — a FRESH store object (≙ a brand-new container) re-reads the db")
    store = DurableWorkOrderStore()          # nothing in memory; reads the file
    state = store.get_session(WO_ID)
    assert state is not None, "session was not persisted!"
    step(f"   recovered {WO_ID}: current_step = {state['current_step']}")
    step(f"   fault_details survived: {state['fault_details']}")
    step(f"   still waiting on: {state['pending_signals']}")
    assert state["current_step"] == WorkOrderStep.AWAITING_PART
    step("   ✅ state survived the restart — no conversation replay needed.")


def main():
    banner("CP2 · Durable sessions — pause for days, resume with no context loss")
    if have_adk():
        step("google-adk is installed — you could use DatabaseSessionService directly.")
    show_adk_config()
    print()
    # clean slate so the demo is reproducible
    if DB_PATH.exists():
        DB_PATH.unlink()
    phase_a_open_and_pause()
    print()
    phase_b_fresh_process()
    print()
    step("KEY IDEA: durability lives in the DB, not the process. Checkpoint 3")
    step("wakes this paused work-order from an external delivery event.")


if __name__ == "__main__":
    main()
