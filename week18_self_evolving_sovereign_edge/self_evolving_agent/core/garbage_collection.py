#!/usr/bin/env python3
"""Memory garbage collection  (Tutorial Part 7).

Unconstrained memory accumulation has its own costs: the episodic DB grows to
hundreds of thousands of messages, vector indexes get noisy, and MEMORY.md
bloats with contradictory facts. GC solves this with three complementary
strategies:

  1. TTL-based episodic compression — summarise an old session into MEMORY.md,
     then delete the raw messages (60–80% token reduction, facts preserved).
  2. SKILL.md versioning — handled in skill_library.py (archive + rollback).
  3. The right to forget — GDPR-safe surgical erasure with an audit trail.

The summariser is injected as a callable so this module runs offline: pass a
real LLM summariser in production, or the deterministic stub used by the
checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .session_db import SessionDB


class EpisodicGarbageCollector:
    """Run a GC cycle: summarise old sessions into MEMORY.md, then prune raw msgs."""

    def __init__(self, db: SessionDB, memory_md: str | Path,
                 summarise_fn: Callable[[list[dict], str], str], ttl_days: int = 7):
        self.db = db
        self.memory_md = Path(memory_md)
        self.summarise = summarise_fn          # (history, session_id) -> summary text
        self.ttl_days = ttl_days

    def run_gc_cycle(self) -> dict:
        """Summarise sessions past TTL, append summaries, prune raw messages."""
        stats = {"sessions_summarised": 0, "messages_pruned": 0, "tokens_saved": 0}
        old = self.db.conn.execute(
            "SELECT DISTINCT session_id FROM messages "
            "WHERE timestamp < datetime('now', ?) "
            "AND session_id NOT IN (SELECT session_id FROM messages WHERE role='summary') "
            "LIMIT 10", (f"-{self.ttl_days} days",)).fetchall()
        for (session_id,) in old:
            history = self.db.get_session_history(session_id)
            if not history:
                continue
            summary = self.summarise(history, session_id)
            if not summary:
                continue
            # store the summary as a special 'summary' role message (survives prune)
            self.db.append_message(session_id, "summary", summary)
            self._merge_summary_into_memory(session_id, summary)
            pruned = self.db.prune_session_messages(session_id, keep_summary=True)
            stats["sessions_summarised"] += 1
            stats["messages_pruned"] += pruned
            raw_tokens = sum(len(m["content"]) for m in history) // 4
            stats["tokens_saved"] += max(0, raw_tokens - len(summary) // 4)
        return stats

    def _merge_summary_into_memory(self, session_id: str, summary: str) -> None:
        text = self.memory_md.read_text(errors="replace") if self.memory_md.exists() else ""
        if "## Session Summaries" not in text:
            text += "\n\n## Session Summaries\n"
        text += f"\n### {session_id}\n{summary}\n"
        self.memory_md.write_text(text)


def erase_user_fact(fact_keyword: str, user_id: str, db: SessionDB,
                    user_md: str | Path, audit_path: str | Path) -> dict:
    """Right to forget — remove a fact from USER.md + redact episodic messages,
    leaving a GDPR-compliance audit trail."""
    user_md = Path(user_md)
    changes = {"memory_lines_removed": 0, "episodic_messages_redacted": 0}

    # 1. Remove matching lines from USER.md
    if user_md.exists():
        lines = user_md.read_text(errors="replace").splitlines()
        kept = [l for l in lines if fact_keyword.lower() not in l.lower()]
        changes["memory_lines_removed"] = len(lines) - len(kept)
        user_md.write_text("\n".join(kept) + "\n")

    # 2. Redact matching episodic messages (keep the row, drop the content)
    def redact(conn) -> int:
        result = conn.execute(
            "UPDATE messages SET content = '[REDACTED per user request]' "
            "WHERE content LIKE ? AND session_id IN "
            "(SELECT session_id FROM sessions WHERE user_id = ?)",
            (f"%{fact_keyword}%", user_id))
        return result.rowcount
    changes["episodic_messages_redacted"] = db._execute_write(redact)

    # 3. Write the audit trail entry
    audit = {"erased_keyword": fact_keyword, "user_id": user_id, "changes": changes}
    with open(audit_path, "a") as f:
        f.write(json.dumps(audit) + "\n")
    return changes
