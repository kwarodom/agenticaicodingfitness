#!/usr/bin/env python3
"""Shared configuration for the Week 18 self-evolving-agent tutorial.

One place for: which MODEL the agent runs on, WHERE the three memory layers live
on disk, a zero-dependency .env loader (so the repo-root ANTHROPIC_API_KEY is
picked up without python-dotenv), and a tiny capability probe so every module
degrades gracefully when the Claude Agent SDK / `claude` CLI is unavailable.

The memory layers (Tripartite Memory Model — see TUTORIAL.md):

    memory/
    ├── MEMORY.md            semantic — world & project facts
    ├── USER.md              semantic — user profile & preferences
    ├── skills/              procedural — the SKILL.md library
    │   ├── index.json       O(1) skill lookup table
    │   └── archive/         previous skill versions (rollback)
    ├── snapshots/           PreCompact working-memory snapshots
    └── agent_state.db       episodic — the SessionDB (SQLite)
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

APP_NAME = "self_evolving_agent"

# ── models ───────────────────────────────────────────────────────────────────
# Fast + cheap: the default for foreground agent turns and routine work.
MODEL_FAST = "claude-haiku-4-5-20251001"
# More capable: background consolidation / GEPA, where reasoning quality matters.
MODEL_SMART = "claude-sonnet-4-6"

# ── safety rails ─────────────────────────────────────────────────────────────
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_BUDGET_USD = 0.50
CONSOLIDATION_BUDGET_USD = 0.15      # cap the background meta-cognitive agent
SUMMARISE_BUDGET_USD = 0.02          # cap the GC session-summariser

# ── paths ────────────────────────────────────────────────────────────────────
PKG = Path(__file__).resolve().parent                 # …/week18/self_evolving_agent
MEMORY_DIR = PKG / "memory"
SKILLS_DIR = MEMORY_DIR / "skills"
ARCHIVE_DIR = SKILLS_DIR / "archive"
SNAPSHOT_DIR = MEMORY_DIR / "snapshots"
DB_PATH = MEMORY_DIR / "agent_state.db"
VECTOR_DIR = MEMORY_DIR / "vector_db"


def ensure_memory_dirs() -> Path:
    """Create the on-disk memory tree (idempotent). Returns the memory root."""
    for d in (MEMORY_DIR, SKILLS_DIR, ARCHIVE_DIR, SNAPSHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return MEMORY_DIR


def wipe_memory() -> list[str]:
    """Delete every generated memory artifact — a clean amnesiac slate.

    Used by the 'reset' control so you can re-watch the agent learn from zero.
    """
    removed: list[str] = []
    for target in (DB_PATH, MEMORY_DIR / "MEMORY.md", MEMORY_DIR / "USER.md",
                   SKILLS_DIR, SNAPSHOT_DIR, VECTOR_DIR,
                   MEMORY_DIR / "erasure_audit.jsonl"):
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            removed.append(target.name + "/")
        elif target.exists():
            target.unlink()
            removed.append(target.name)
    # also drop SQLite side files
    for suffix in ("-wal", "-shm"):
        side = Path(str(DB_PATH) + suffix)
        if side.exists():
            side.unlink()
    ensure_memory_dirs()
    return removed or ["nothing (already clean)"]


# ── minimal .env loader (no python-dotenv dependency) ────────────────────────
def load_env() -> None:
    """Populate os.environ from the repo-root .env for any key not already set.
    Safe to call repeatedly; never overrides an already-exported variable."""
    root = PKG.parents[1]                               # …/agenticaicodingfitness
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


# ── capability probe ─────────────────────────────────────────────────────────
def sdk_available() -> bool:
    """True only if a LIVE agent turn can run: the SDK imports AND the `claude`
    CLI it drives is on PATH. Everything else (the engine) runs offline."""
    if shutil.which("claude") is None:
        return False
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except Exception:
        return False
