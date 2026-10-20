#!/usr/bin/env python3
"""CHECKPOINT 2 — Working + Episodic Memory (L1 + L2).

Goal: give agents two fast, dependency-free memory tiers before we add the
heavier services. L1 = Working Memory (in-process dict, holds the current
task's transient state). L2 = Episodic Memory (SQLite, an append-only log of
"what happened" that survives process restarts). (smart_hotel_mas.pdf §CP2)

Run:  python week15/smart_hotel_mas/checkpoints/checkpoint2_memory.py

Pure stdlib — no Neo4j / ChromaDB / network needed for this checkpoint.
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import config.py
from config import EPISODIC_DB


# ── L1: Working Memory (transient, in-process) ───────────────────────────────
class WorkingMemory:
    """L1 scratchpad. A plain dict for the agent's current task state.

    Lives only as long as the process: when the batch finishes (or the process
    exits), this state is gone. That is exactly what we want for per-task
    bookkeeping like "which rooms am I processing right now".
    """

    def __init__(self):
        self._state = {}

    def set(self, key, value):
        self._state[key] = value

    def get(self, key, default=None):
        return self._state.get(key, default)

    def get_all(self) -> dict:
        return dict(self._state)

    def clear(self):
        self._state = {}

    def summary(self) -> str:
        items = list(self._state.items())
        return "Working Memory: {" + ", ".join(f"{k}={v}" for k, v in items[:10]) + "}"


# ── L2: Episodic Memory (persistent, SQLite) ─────────────────────────────────
class EpisodicMemory:
    """L2 event log. An append-only timeline of agent events in SQLite.

    Unlike L1, this persists to disk, so an agent can answer "what did we do
    today?" even after a restart.
    """

    def __init__(self, db_path=EPISODIC_DB):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                agent_id TEXT,
                event_type TEXT,
                data TEXT,
                tags TEXT DEFAULT ''
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS ep_ts_idx ON episodes(ts)")
        self.conn.commit()

    def store(self, agent_id, event_type, data: dict, tags: list = None):
        self.conn.execute(
            "INSERT INTO episodes (ts, agent_id, event_type, data, tags) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                agent_id,
                event_type,
                json.dumps(data),
                ",".join(tags or []),
            ),
        )
        self.conn.commit()

    def recall_recent(self, hours=24, event_type=None, limit=20) -> list:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        sql = "SELECT ts, agent_id, event_type, data FROM episodes WHERE ts > ?"
        params = [cutoff]
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {"ts": r[0], "agent": r[1], "type": r[2], "data": json.loads(r[3])}
            for r in rows
        ]

    def recall_by_room(self, room_id, limit=10) -> list:
        rows = self.conn.execute(
            "SELECT ts, agent_id, event_type, data FROM episodes "
            "WHERE data LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{room_id}%", limit),
        ).fetchall()
        return [
            {"ts": r[0], "agent": r[1], "type": r[2], "data": json.loads(r[3])}
            for r in rows
        ]

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    # ── L1: Working Memory ──────────────────────────────────────────────────
    print("── L1: Working Memory (in-process) ──")
    wm = WorkingMemory()
    wm.set("current_batch", ["R101", "R102", "R103"])
    wm.set("batch_start_ts", datetime.now().isoformat())
    print(wm.summary())

    # ── L2: Episodic Memory ─────────────────────────────────────────────────
    print("\n── L2: Episodic Memory (SQLite) ──")
    em = EpisodicMemory()
    em.store(
        "SensorAgent",
        "sensor_batch_read",
        {"rooms": ["R101", "R102", "R103"], "count": 3, "avg_temp_c": 24.1},
        tags=["batch", "sensors"],
    )
    em.store(
        "AlertAgent",
        "alert_triggered",
        {"room": "R301", "type": "HIGH_TEMP", "value": 28.5, "threshold": 27.0},
        tags=["alert", "hvac"],
    )

    recent = em.recall_recent(hours=1)
    print(f"Recalled {len(recent)} episode(s) from the last hour:")
    for ep in recent:
        print(f"  [{ep['ts']}] {ep['agent']} · {ep['type']} → {ep['data']}")
    em.close()

    # ── Key Insight ─────────────────────────────────────────────────────────
    # L1 (Working Memory) is the agent's transient task state: it vanishes when
    # the process ends, which is correct for "what am I doing right now".
    # L2 (Episodic Memory) persists to SQLite, so the system can later answer
    # "what did we do today?" — durable history that outlives any single run.
