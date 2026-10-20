#!/usr/bin/env python3
"""Checkpoint 5 — Memory garbage collection  (Part 7).

Unbounded memory has its own costs. GC keeps the agent healthy with:
  1. TTL episodic compression — summarise an old session into MEMORY.md, then
     delete the raw messages (the semantic value survives; the verbatim log
     doesn't). 60–80% token reduction per session.
  2. The right to forget — GDPR-safe surgical erasure with an audit trail.

Runs OFFLINE: a deterministic stub stands in for the LLM summariser.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from self_evolving_agent.core.garbage_collection import (
    EpisodicGarbageCollector, erase_user_fact)
from self_evolving_agent.core.semantic_memory import SemanticMemory
from self_evolving_agent.core.session_db import SessionDB


def main() -> None:
    print("● Checkpoint 5 — Memory garbage collection\n")
    root = Path(tempfile.mkdtemp())
    db = SessionDB(str(root / "gc.db"))
    sem = SemanticMemory(root)

    # Seed an OLD session (timestamps backdated 30 days) with lots of raw messages
    sid = "ses-old"
    db.create_session(sid, "claude-haiku-4-5", user_id="kwarodom", label="old")
    for i in range(40):
        db.append_message(sid, "user" if i % 2 else "assistant",
                          f"verbose message number {i} about the staging deploy")
    db.conn.execute("UPDATE messages SET timestamp = datetime('now','-30 days') "
                    "WHERE session_id = ?", (sid,))
    db.conn.commit()
    # add one fact to forget later
    db.create_session("ses-pii", "claude-haiku-4-5", user_id="kwarodom")
    db.append_message("ses-pii", "user", "My home address is 42 Sukhumvit Road Bangkok")
    sem.append_fact("USER.md", "Identity", "Home address: 42 Sukhumvit Road Bangkok")

    before = db.session_stats(sid)
    print(f"  Before GC ......... session '{sid}' has {before['messages']} raw messages")

    # 1. TTL compression — deterministic stub summariser
    def stub_summary(history, session_id):
        return (f"Session {session_id[:8]}: user worked through a staging deploy across "
                f"{len(history)} messages; key outcome recorded.")

    gc = EpisodicGarbageCollector(db, sem.memory_md, stub_summary, ttl_days=7)
    stats = gc.run_gc_cycle()
    after = db.session_stats(sid)
    print(f"  GC cycle .......... summarised={stats['sessions_summarised']} · "
          f"pruned={stats['messages_pruned']} raw msgs · ~{stats['tokens_saved']} tokens saved")
    print(f"  After GC .......... session '{sid}' raw messages now {after['messages']} "
          f"(summary retained)")
    assert stats["sessions_summarised"] == 1
    assert after["messages"] < before["messages"]
    assert "Session Summaries" in sem.memory_md.read_text()
    print("  MEMORY.md ......... gained a 'Session Summaries' section (value preserved)")

    # 2. The right to forget — GDPR-safe erasure with audit trail
    audit = root / "erasure_audit.jsonl"
    changes = erase_user_fact("address", "kwarodom", db, sem.user_md, audit)
    print(f"\n  Right to forget ... USER.md lines removed={changes['memory_lines_removed']} · "
          f"episodic redacted={changes['episodic_messages_redacted']}")
    assert changes["memory_lines_removed"] >= 1
    assert "address" not in sem.user_md.read_text().lower()
    assert audit.exists()
    print(f"  Audit trail ....... {audit.name} written (compliance-ready)")

    db.close()
    print("\n✓ Checkpoint 5 passed — old episodes compress into semantic memory and "
          "personal facts can be surgically, auditably erased.")


if __name__ == "__main__":
    main()
