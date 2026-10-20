#!/usr/bin/env python3
"""Checkpoint 1 — Episodic memory: the SessionDB engine  (Part 3).

Builds the real episodic store and exercises the four engineering features that
make SQLite safe for a concurrent agent:

    • WAL mode               concurrent reads during writes
    • jitter backoff         no convoy effect under concurrent writers
    • schema reconciliation  add a column by editing SCHEMA_SQL, no migration
    • dual FTS5 tokenizers    latin (unicode61) + CJK / partial (trigram) search

Runs fully OFFLINE — pure stdlib, no API key, no network.
"""
from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from self_evolving_agent.core.session_db import SCHEMA_SQL, SessionDB


def main() -> None:
    print("● Checkpoint 1 — Episodic memory (SessionDB)\n")
    db_path = str(Path(tempfile.mkdtemp()) / "episodic.db")
    db = SessionDB(db_path)

    # 1. WAL mode is on (concurrent reads during writes)
    mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    print(f"  WAL mode .......... journal_mode={mode}")
    assert mode.lower() == "wal"

    # 2. Append messages — English + Chinese (CJK) to exercise both tokenizers
    sid = "ses-cp1"
    db.create_session(sid, "claude-haiku-4-5", label="cp1")
    db.append_message(sid, "user", "Deploy the FastAPI service to staging", tokens=12)
    db.append_message(sid, "assistant", "Building the Docker image now", tokens=8)
    db.append_message(sid, "user", "部署到生产环境需要多长时间")   # CJK query material
    print("  Appended .......... 3 messages (2 English, 1 Chinese)")

    # 3. Dual FTS5 search — unicode61 for latin, trigram auto-selected for CJK
    latin = db.search_messages("deploy staging", session_id=sid)
    cjk = db.search_messages("生产环境", session_id=sid)
    print(f"  FTS5 (unicode61) .. 'deploy staging' → {len(latin)} hit(s)")
    print(f"  FTS5 (trigram) .... '生产环境'       → {len(cjk)} hit(s)  (CJK auto-routed)")
    assert latin and cjk

    # 4. Jitter backoff under concurrent writers — no convoy collapse.
    # Each thread opens its OWN connection to the same DB file (the real scenario:
    # a gateway API + a CLI + a background worker all writing at once). WAL +
    # randomised jitter backoff let them interleave without the convoy effect.
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            wdb = SessionDB(db_path)
            for i in range(15):
                wdb.append_message(sid, "tool", f"writer{n}-msg{i}")
            wdb.close()
        except Exception as exc:                       # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f"  Jitter backoff .... 4 threads (own connections) × 15 writes = 60 "
          f"concurrent, errors={len(errors)}")
    assert not errors, f"convoy not handled: {errors}"

    # 5. Declarative schema evolution — pretend a new column was added to SCHEMA_SQL
    SCHEMA_SQL["messages"]["feedback_score"] = "REAL"
    try:
        db2 = SessionDB(db_path)                        # reopen → auto-reconciles
        cols = {r[1] for r in db2.conn.execute("PRAGMA table_info(messages)").fetchall()}
        print(f"  Schema evolve ..... added 'feedback_score' on reboot → present={('feedback_score' in cols)}")
        assert "feedback_score" in cols
        db2.close()
    finally:
        SCHEMA_SQL["messages"].pop("feedback_score", None)

    stats = db.session_stats(sid)
    print(f"\n  session stats ..... {stats}")
    db.close()
    print("\n✓ Checkpoint 1 passed — episodic engine is WAL-safe, concurrent, "
          "schema-evolving, and bilingually searchable.")


if __name__ == "__main__":
    main()
