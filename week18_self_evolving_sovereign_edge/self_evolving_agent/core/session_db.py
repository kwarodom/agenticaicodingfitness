#!/usr/bin/env python3
"""Episodic memory — the SessionDB engine  (Tutorial Part 3).

A highly-optimised SQLite store for the agent's raw, message-by-message
experience. SQLite is dismissed as a "toy", but configured correctly it is a
fast, serverless, crash-safe engine for local AI state. Two engineering problems
must be solved to make it safe for a concurrent agent (a foreground turn + a
background consolidator writing at once):

  1. write contention  → WAL mode + randomised jitter backoff (no convoy effect)
  2. schema evolution  → declarative SCHEMA_SQL reconciled on boot (no migrations)

Plus universal full-text search via DUAL FTS5 tokenizers: ``unicode61`` for
latin scripts and ``trigram`` for CJK / partial-match queries.

This module is pure stdlib — it runs with no API key and no network.
"""
from __future__ import annotations

import random
import sqlite3
import time
import uuid
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# ── Declarative schema — the single source of truth ──────────────────────────
# To add a column (e.g. feedback_score) just add it here and restart; the DB
# auto-evolves on next boot via _reconcile_columns(). No migration scripts.
SCHEMA_SQL: dict[str, dict[str, str]] = {
    "sessions": {
        "session_id": "TEXT PRIMARY KEY",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "model": "TEXT",
        "user_id": "TEXT",
        "system_prompt": "TEXT",
        "label": "TEXT",                 # human-readable run label (e.g. "run 1")
        "turns": "INTEGER",              # agent-loop turns this run took
        "cost": "REAL",                  # USD this run cost
        "skills_loaded": "TEXT",         # comma-joined skills injected for this run
    },
    "messages": {
        "message_id": "TEXT PRIMARY KEY",
        "session_id": "TEXT",
        "role": "TEXT",                  # 'user' | 'assistant' | 'tool' | 'summary'
        "content": "TEXT",
        "tokens": "INTEGER",
        "cost": "REAL",
        "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    },
}


class SessionDB:
    """Full episodic memory engine: WAL, jitter backoff, FTS5, schema evolution."""

    _WRITE_MAX_RETRIES = 5
    _WRITE_RETRY_MIN_S = 0.02            # 20 ms minimum jitter
    _WRITE_RETRY_MAX_S = 0.15            # 150 ms maximum jitter

    def __init__(self, db_path: str = "agent_state.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._setup_wal_mode(self.conn)
        self._init_schema()

    # ── 3.1 WAL mode — concurrent reads during writes ────────────────────────
    def _setup_wal_mode(self, conn: sqlite3.Connection) -> None:
        """Configure SQLite for high-concurrency agent workloads."""
        conn.execute("PRAGMA journal_mode=WAL;")     # readers never block writers
        conn.execute("PRAGMA synchronous=NORMAL;")   # survives OS crash; 3-5× faster
        conn.execute("PRAGMA cache_size=-32000;")    # 32 MB page cache
        conn.execute("PRAGMA foreign_keys=ON;")

    # ── 3.2 The convoy problem — randomised jitter backoff ───────────────────
    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run a write transaction with randomised jitter backoff.

        Under contention every thread otherwise retries at the same instant,
        collides, and the DB appears frozen (the convoy effect). Staggering
        retries randomly between 20–150 ms lets threads find open windows
        without coordinating.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(self.conn)
                    self.conn.commit()
                    return result
                except BaseException:
                    self.conn.rollback()
                    raise
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(self._WRITE_RETRY_MIN_S,
                                                self._WRITE_RETRY_MAX_S)
                        time.sleep(jitter)
                        continue
                raise
        raise last_err or RuntimeError("Write failed after max retries")

    # ── 3.3 Declarative schema evolution ─────────────────────────────────────
    def _init_schema(self) -> None:
        def setup(conn: sqlite3.Connection) -> None:
            for table, cols in SCHEMA_SQL.items():
                col_defs = ", ".join(f'"{c}" {t}' for c, t in cols.items())
                conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs})")
            # 3.4 dual FTS5 tokenizers — latin (unicode61) + CJK/partial (trigram)
            conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts '
                         'USING fts5(content, content="messages", tokenize="unicode61")')
            conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram '
                         'USING fts5(content, content="messages", tokenize="trigram")')
            self._reconcile_columns(conn.cursor())
        self._execute_write(setup)

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """Add any column declared in SCHEMA_SQL but missing in the live DB."""
        for table, declared in SCHEMA_SQL.items():
            cursor.execute(f"PRAGMA table_info({table})")
            live = {row[1] for row in cursor.fetchall()}
            for col, col_type in declared.items():
                if col not in live:
                    cursor.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {col_type}')

    # ── writes ───────────────────────────────────────────────────────────────
    def create_session(self, session_id: str, model: str, user_id: str = "user",
                        system_prompt: str = "", label: str = "") -> None:
        def insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_id, created_at, model, user_id, system_prompt, label) "
                "VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)",
                (session_id, model, user_id, system_prompt, label))
        self._execute_write(insert)

    def append_message(self, session_id: str, role: str, content: str,
                       tokens: int = 0, cost: float = 0.0) -> str:
        msg_id = str(uuid.uuid4())

        def insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO messages "
                "(message_id, session_id, role, content, tokens, cost, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (msg_id, session_id, role, content, tokens, cost))
            rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                         (rowid, content))
            conn.execute("INSERT INTO messages_fts_trigram(rowid, content) VALUES (?, ?)",
                         (rowid, content))
        self._execute_write(insert)
        return msg_id

    # ── reads ─────────────────────────────────────────────────────────────────
    def get_session_history(self, session_id: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT role, content, tokens, cost FROM messages "
            "WHERE session_id = ? ORDER BY timestamp ASC, rowid ASC", (session_id,))
        return [dict(row) for row in cursor.fetchall()]

    def session_stats(self, session_id: str) -> dict:
        """Aggregate episodic metrics for one run — turns, tokens, cost."""
        row = self.conn.execute(
            "SELECT COUNT(*) n, "
            "COALESCE(SUM(CASE WHEN role='tool' THEN 1 ELSE 0 END),0) tool_calls, "
            "COALESCE(SUM(tokens),0) tokens, COALESCE(SUM(cost),0) cost "
            "FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        return {"messages": row["n"], "tool_calls": row["tool_calls"],
                "tokens": row["tokens"], "cost": round(row["cost"], 4)}

    def _contains_cjk(self, text: str) -> bool:
        """Detect CJK characters to route to the correct FTS5 index."""
        return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)

    def search_messages(self, query: str, session_id: str | None = None,
                        limit: int = 20) -> list[dict]:
        """Search history with the optimal tokenizer for the query language."""
        if self._contains_cjk(query) and len(query.strip()) >= 3:
            fts = "messages_fts_trigram"   # partial matching + CJK (no word boundaries)
        else:
            fts = "messages_fts"           # better relevance ranking for latin scripts
        sql = (
            f"SELECT m.role, m.content, m.timestamp, bm25({fts}) AS relevance "
            f"FROM messages m JOIN {fts} ON m.rowid = {fts}.rowid "
            f"WHERE {fts} MATCH ? "
            + ("AND m.session_id = ? " if session_id else "")
            + "ORDER BY relevance LIMIT ?")
        params: tuple = (query, session_id, limit) if session_id else (query, limit)
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    # ── garbage collection support (Part 7) ───────────────────────────────────
    def prune_old_messages(self, ttl_days: int = 7) -> int:
        """Delete raw messages older than TTL (their value lives on in MEMORY.md)."""
        def delete(conn: sqlite3.Connection) -> int:
            result = conn.execute(
                "DELETE FROM messages WHERE timestamp < datetime('now', ?)",
                (f"-{ttl_days} days",))
            # FTS5 content tables stay consistent because they're contentless mirrors;
            # rebuild keeps them tidy after a bulk delete.
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) "
                         "VALUES('rebuild')")
            return result.rowcount
        return self._execute_write(delete)

    def prune_session_messages(self, session_id: str, keep_summary: bool = True) -> int:
        """Delete raw messages for one session after it has been summarised."""
        def delete(conn: sqlite3.Connection) -> int:
            clause = "AND role != 'summary'" if keep_summary else ""
            result = conn.execute(
                f"DELETE FROM messages WHERE session_id = ? {clause}", (session_id,))
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) "
                         "VALUES('rebuild')")
            return result.rowcount
        return self._execute_write(delete)

    def update_session_metrics(self, session_id: str, turns: int, cost: float,
                               skills_loaded: str = "") -> None:
        """Record run-level metrics so the compound-returns chart survives a
        service restart (the durable proof that the agent improved)."""
        def update(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET turns = ?, cost = ?, skills_loaded = ? "
                "WHERE session_id = ?", (turns, cost, skills_loaded, session_id))
        self._execute_write(update)

    def list_sessions(self) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT session_id, label, model, created_at, turns, cost, skills_loaded "
            "FROM sessions WHERE label IS NOT NULL AND label != '' "
            "ORDER BY created_at ASC, rowid ASC")
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()
