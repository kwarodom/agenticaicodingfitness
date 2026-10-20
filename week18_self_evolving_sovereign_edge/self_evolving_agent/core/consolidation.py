#!/usr/bin/env python3
"""The subconscious loop — background consolidation  (Tutorial Part 6).

The feature that separates a self-EVOLVING agent from a merely persistent one is
the background consolidation loop — the agent's equivalent of human sleep. While
the user reads the response, a background thread replays the conversation and
distils it into lasting knowledge:

    episodic transcript ──▶ semantic facts (MEMORY.md / USER.md)
                       └──▶ procedural skill (SKILL.md)

The foreground thread NEVER waits for this — the user gets their answer
immediately and the agent gets smarter in the background.

Two consolidators are provided:
  • SDKConsolidator   — the real meta-cognitive agent (claude_agent_sdk), edits
    the memory files itself with the Write/Edit tools.
  • HeuristicConsolidator — a deterministic, offline distiller so every
    checkpoint and the live demo work with no API key / no CLI.

Both implement ``consolidate(history, session_id) -> dict`` and report what they
learned so the visualizer can show the episodic→semantic→procedural flow.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .semantic_memory import SemanticMemory
from .skill_library import SkillLibrary

META_COGNITIVE_PROMPT = """You are a self-improving cognitive review engine. Your \
task is to review the conversation transcript that just occurred and extract \
permanent knowledge into the memory files in the current directory.

1. SEMANTIC MEMORY UPDATE:
Did the user share new personal facts, project details, technical decisions, or \
preferences? If yes, use the Edit/Write tool to update MEMORY.md (world/project \
facts) or USER.md (user profile/preferences).

2. PROCEDURAL MEMORY UPDATE:
Did you perform a multi-step technical task? Did you discover a better approach, \
hit a failure and find a fix, or identify a reusable workflow? If yes, write or \
update a SKILL.md file in the ./skills/ directory. Give it a name like \
skills/<task-class>.md with sections: Context, Preconditions, Optimal Steps, \
Known Pitfalls, Performance Notes — and a metrics header (version, success_rate, \
avg_turns, last_updated, usage_count, error_count).

3. ERROR LEARNING:
Did you make a mistake? Add the error pattern and its fix to the relevant \
SKILL.md's "Known Pitfalls" section.

IMPORTANT: If nothing of permanent value was discussed (small talk, trivial \
lookups), take NO action. Do not write empty or trivial memory entries."""


def _format_transcript(history: list[dict], limit: int = 20) -> str:
    recent = history[-limit:]
    return "\n".join(f"[{m['role'].upper()}]: {m['content'][:500]}" for m in recent)


# ── 6.3 PreCompact hook — protect working memory during compaction ───────────
def snapshot_before_compaction(session_id: str, history: list[dict],
                               snapshot_dir: str | Path) -> dict:
    """Snapshot in-flight working memory before the context window is compacted,
    so nothing is lost when older turns are summarised away."""
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"{session_id[:8]}_pre_compact.json"
    path.write_text(json.dumps({
        "session_id": session_id,
        "history_length": len(history),
        "last_message": history[-1] if history else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return {"snapshot": str(path)}


class HeuristicConsolidator:
    """Deterministic, offline meta-cognitive distiller.

    It mirrors what the real agent does — extract semantic facts, derive a
    procedural skill, record pitfalls — using simple rules so the closed loop is
    fully observable with zero dependencies. Good enough to *demonstrate*
    compound returns; swap in SDKConsolidator for genuine LLM distillation.
    """

    def __init__(self, memory_dir: str | Path, skills_dir: str | Path) -> None:
        self.semantic = SemanticMemory(memory_dir)
        self.skills = SkillLibrary(skills_dir)

    def consolidate(self, history: list[dict], session_id: str) -> dict:
        learned = {"facts": [], "skill": None, "pitfalls": []}
        if len(history) < 2:
            return learned

        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        tool_msgs = [m["content"] for m in history if m["role"] == "tool"]
        joined = " ".join(user_msgs).lower()

        # 1. SEMANTIC — pull obvious project/user facts out of the transcript.
        for pattern, section_file, section in [
            (r"project\s+(\w[\w\- ]{2,40})", "MEMORY.md", "Project Context"),
            (r"my name is\s+(\w[\w ]{1,40})", "USER.md", "Identity"),
            (r"i prefer\s+(\w[\w ,]{2,60})", "USER.md", "Preferences"),
            (r"we use\s+(\w[\w ,./]{2,60})", "MEMORY.md", "Technical Decisions"),
        ]:
            m = re.search(pattern, joined)
            if m:
                fact = m.group(1).strip().rstrip(".")
                self.semantic.append_fact(section_file, section, fact)
                learned["facts"].append(f"{section}: {fact}")

        # 2. PROCEDURAL — derive a skill from the task class + the tools used.
        task_class = self._task_class(joined)
        if task_class and tool_msgs:
            steps = "\n".join(f"{i+1}. {t[:90]}" for i, t in enumerate(tool_msgs[:6]))
            pitfalls = [m["content"] for m in history
                        if m["role"] == "tool" and "error" in m["content"].lower()]
            today = datetime.now(timezone.utc).date().isoformat()
            content = (
                f"# SKILL: {task_class}\n"
                f"version: 1.0\nsuccess_rate: 1.0\n"
                f"avg_turns: {len(tool_msgs)}\nlast_updated: {today}\n"
                f"usage_count: 1\nerror_count: {len(pitfalls)}\n\n"
                f"## Context\nLearned from session {session_id[:8]}.\n\n"
                f"## Optimal Steps\n{steps or '(steps observed in transcript)'}\n\n"
                f"## Known Pitfalls (auto-learned from failures)\n"
                + ("\n".join(f"{i+1}. [{today}] {p[:80]}" for i, p in enumerate(pitfalls))
                   or "(none yet)")
                + "\n")
            result = self.skills.update_skill_with_versioning(task_class, content)
            self.skills.update_skill_index(
                task_class, f"{task_class}.md",
                keywords=list(set(re.findall(r"\w+", task_class.replace("-", " ")))
                              | {w for w in re.findall(r"\w+", joined) if len(w) > 4})[:12])
            learned["skill"] = result
            learned["pitfalls"] = [p[:80] for p in pitfalls]
        return learned

    def _task_class(self, text: str) -> str | None:
        for kw, cls in [("deploy", "deploy-service"), ("test", "run-tests"),
                        ("analy", "analyse-codebase"), ("review", "code-review"),
                        ("refactor", "refactor-code"), ("fix", "fix-bug"),
                        ("summar", "summarise-data")]:
            if kw in text:
                return cls
        return None


class SDKConsolidator:
    """The real meta-cognitive agent: a background claude_agent_sdk run that edits
    MEMORY.md / USER.md / skills/*.md itself. Used when the SDK + CLI are present."""

    def __init__(self, memory_dir: str | Path, model: str,
                 budget_usd: float = 0.15) -> None:
        self.memory_dir = str(memory_dir)
        self.model = model
        self.budget_usd = budget_usd

    def consolidate(self, history: list[dict], session_id: str) -> dict:
        from ._aio import run_sync
        return run_sync(self._run, history, session_id)

    async def _run(self, history: list[dict], session_id: str) -> dict:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
        transcript = _format_transcript(history)
        before = self._snapshot_files()
        # Consume the generator fully (no early break) so the SDK's anyio task
        # group shuts down cleanly in this thread.
        async for msg in query(
            prompt=("Review this conversation and update memory + skills as "
                    f"appropriate. Work in the current directory.\n\n{transcript}"),
            options=ClaudeAgentOptions(
                system_prompt=META_COGNITIVE_PROMPT,
                allowed_tools=["Read", "Write", "Edit", "Glob"],
                permission_mode="bypassPermissions",
                max_turns=8,
                max_budget_usd=self.budget_usd,
                model=self.model,
                cwd=self.memory_dir,
            ),
        ):
            if isinstance(msg, ResultMessage):
                pass
        after = self._snapshot_files()
        return self._diff(before, after)

    def _snapshot_files(self) -> dict:
        root = Path(self.memory_dir)
        files = {}
        for p in list(root.glob("*.md")) + list((root / "skills").glob("*.md")):
            files[str(p.relative_to(root))] = p.read_text(errors="replace")
        return files

    def _diff(self, before: dict, after: dict) -> dict:
        learned = {"facts": [], "skill": None, "pitfalls": [], "files_changed": []}
        for name, text in after.items():
            if before.get(name) != text:
                learned["files_changed"].append(name)
                if name.startswith("skills/"):
                    learned["skill"] = {"skill": Path(name).stem}
        return learned
