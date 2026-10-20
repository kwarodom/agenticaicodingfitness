#!/usr/bin/env python3
"""The SelfEvolvingAgent  (Tutorial Part 10).

A drop-in replacement for a stateless agent — same interface, persistent memory.
It wires the three memory layers into every turn:

    system prompt = BASE_PROMPT
                  + <memory-context> fenced MEMORY.md + USER.md </memory-context>
                  + <skill> relevant SKILL.md files </skill>

After each turn it (a) logs the conversation to episodic SessionDB and (b) forks
background consolidation so the NEXT run starts from accumulated wisdom rather
than a blank slate. That feedback loop is what produces compound returns:
identical tasks get measurably faster and cheaper run over run.

Two execution modes, chosen automatically:
  • LIVE   — real claude_agent_sdk turn (needs the `claude` CLI). Authentic.
  • SIMULATED — deterministic offline turn. Always runs; cleanly demonstrates
    the mechanism (loaded skill ⇒ fewer turns) with no key / network.

The visualizer and checkpoints call: run_task() then consolidate().
"""
from __future__ import annotations

import uuid
from pathlib import Path

from .. import config
from ._aio import run_sync
from .consolidation import HeuristicConsolidator, SDKConsolidator
from .semantic_memory import SemanticMemory
from .session_db import SessionDB
from .skill_library import SkillLibrary

BASE_PROMPT = """You are a highly capable, self-improving AI assistant. Your \
persistent memory (MEMORY.md, USER.md, and SKILL.md files) is injected above as \
authoritative recalled knowledge. When you complete a complex task your learnings \
are preserved for future sessions — you improve with every interaction. Be \
concise and use any recalled procedural skill as your primary guide."""

# A small, realistic workspace the LIVE agent can actually act on.
_WORKSPACE_FILES = {
    "service.py": (
        "def parse(line):\n    a, b = line.split(',')\n    return int(a) / int(b)\n\n"
        "def run(lines):\n    return [parse(l) for l in lines]  # no error handling\n"
    ),
    "data.csv": "10,2\n9,3\n7,0\n",
    "TASK.md": "# Task\nFind the bug in service.py that crashes on data.csv.\n",
}


class SelfEvolvingAgent:
    def __init__(self, db_path: str | None = None, memory_dir: str | None = None,
                 skills_dir: str | None = None, model: str | None = None) -> None:
        config.ensure_memory_dirs()
        self.db = SessionDB(db_path or str(config.DB_PATH))
        self.mem_dir = Path(memory_dir or config.MEMORY_DIR)
        self.semantic = SemanticMemory(self.mem_dir)
        self.skills = SkillLibrary(skills_dir or config.SKILLS_DIR)
        self.model = model or config.MODEL_FAST
        self.live = config.sdk_available()
        self.workspace = self.mem_dir / "workspace"
        self._seed_workspace()

    def _seed_workspace(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        for name, body in _WORKSPACE_FILES.items():
            f = self.workspace / name
            if not f.exists():
                f.write_text(body)

    # ── memory injection ─────────────────────────────────────────────────────
    def build_system_prompt(self, prompt: str) -> str:
        """BASE_PROMPT + fenced semantic memory + relevant procedural skills."""
        mem_block = self.semantic.fenced_block()
        skills = self.skills.load_relevant_skills(prompt, max_skills=3)
        return f"{BASE_PROMPT}\n\n{mem_block}\n\n{skills}".strip()

    def memory_report(self, prompt: str) -> dict:
        """What memory WOULD be injected for this prompt — for the visualizer."""
        return {
            "has_semantic": self.semantic.has_learned_facts(),
            "skills_loaded": self.skills.matching_skill_names(prompt),
            "skill_count": self.skills.count(),
            "system_prompt_chars": len(self.build_system_prompt(prompt)),
        }

    # ── run one task ──────────────────────────────────────────────────────────
    def run_task(self, prompt: str, label: str = "") -> dict:
        session_id = "ses-" + uuid.uuid4().hex[:8]
        system_prompt = self.build_system_prompt(prompt)
        self.db.create_session(session_id, self.model, "user", system_prompt, label)
        self.db.append_message(session_id, "user", prompt)
        skills_loaded = self.skills.matching_skill_names(prompt)

        if self.live:
            outcome = run_sync(self._run_live, session_id, prompt, system_prompt)
        else:
            outcome = self._run_simulated(session_id, prompt, bool(skills_loaded))

        self.db.update_session_metrics(session_id, outcome["turns"], outcome["cost"],
                                       ", ".join(skills_loaded))
        stats = self.db.session_stats(session_id)
        return {"session_id": session_id, "label": label, "mode": outcome["mode"],
                "turns": outcome["turns"], "tool_calls": stats["tool_calls"],
                "cost": outcome["cost"], "skills_loaded": skills_loaded,
                "memory_injected": self.semantic.has_learned_facts(),
                "result": outcome["result"]}

    async def _run_live(self, session_id: str, prompt: str, system_prompt: str) -> dict:
        from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                                      ResultMessage, TextBlock, ToolResultBlock,
                                      ToolUseBlock, UserMessage, query)
        turns = cost = 0
        result = ""
        async for msg in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                system_prompt=system_prompt,
                allowed_tools=["Read", "Glob", "Grep", "Bash"],
                permission_mode="bypassPermissions",
                max_turns=config.DEFAULT_MAX_TURNS,
                max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
                model=self.model,
                cwd=str(self.workspace),
            ),
        ):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        self.db.append_message(session_id, "assistant", b.text)
                    elif isinstance(b, ToolUseBlock):
                        turns += 1
                        self.db.append_message(
                            session_id, "tool", f"{b.name} {str(b.input)[:200]}")
            elif isinstance(msg, UserMessage):
                blocks = msg.content if isinstance(msg.content, list) else []
                for b in blocks:
                    if isinstance(b, ToolResultBlock):
                        txt = b.content if isinstance(b.content, str) else str(b.content)
                        flag = "error " if b.is_error else ""
                        self.db.append_message(session_id, "tool", f"{flag}result: {txt[:200]}")
            elif isinstance(msg, ResultMessage):
                turns = msg.num_turns or turns
                cost = msg.total_cost_usd or 0.0
                result = msg.result or ""
        return {"mode": "live", "turns": turns, "cost": round(cost, 4), "result": result}

    def _run_simulated(self, session_id: str, prompt: str, has_skill: bool) -> dict:
        """Deterministic offline turn. A loaded SKILL.md lets the agent skip the
        discovery/trial-and-error turns — so identical work costs fewer turns.
        This is the compound-returns mechanism made fully observable."""
        # Cold (no skill): explore → trial → error → fix → verify. Warm: follow skill.
        cold_steps = [
            ("tool", "Glob **/* — discover the workspace"),
            ("tool", "Read service.py — inspect the code"),
            ("tool", "Bash python service.py — trial run"),
            ("tool", "error result: ZeroDivisionError: division by zero"),
            ("assistant", "Found it: parse() has no zero-division guard. Adding one."),
            ("tool", "Read data.csv — confirm the offending row 7,0"),
            ("tool", "Bash python -c 'verify the fix' — passes"),
        ]
        warm_steps = [
            ("assistant", "Recalled skill fix-bug — applying the known pitfall directly."),
            ("tool", "Read service.py — locate the unguarded division (per skill)"),
            ("tool", "Bash python service.py — confirm fix, passes"),
        ]
        steps = warm_steps if has_skill else cold_steps
        for role, content in steps:
            self.db.append_message(session_id, role, content)
        turns = sum(1 for r, _ in steps if r == "tool" and "result" not in _)
        # crude but consistent cost model: ~$0.012 per tool turn on the fast model
        cost = round(turns * 0.012, 4)
        result = ("Fixed: guarded the zero-division in parse(). "
                  + ("(used recalled skill)" if has_skill else "(discovered from scratch)"))
        return {"mode": "simulated", "turns": turns, "cost": cost, "result": result}

    # ── consolidate (the subconscious loop) ───────────────────────────────────
    def consolidate(self, session_id: str) -> dict:
        """Run the meta-cognitive consolidator over a finished session. In
        production this is the Stop hook firing in a background thread; here we
        call it explicitly so the visualizer can show what was learned."""
        history = self.db.get_session_history(session_id)
        heuristic = HeuristicConsolidator(self.mem_dir, self.skills.dir)
        if not self.live:
            return heuristic.consolidate(history, session_id)

        # LIVE: let the real meta-cognitive agent distil the run by editing the
        # memory files itself. It is stochastic — it may judge a trivial run not
        # worth saving — so if it wrote nothing we run the deterministic heuristic
        # to guarantee the learning loop is always observable.
        try:
            learned = SDKConsolidator(self.mem_dir, config.MODEL_SMART,
                                      config.CONSOLIDATION_BUDGET_USD
                                      ).consolidate(history, session_id)
        except Exception as exc:                           # never crash a run
            learned = {"facts": [], "skill": None, "files_changed": [],
                       "note": f"SDK consolidation error: {exc}"}
        wrote_nothing = not (learned.get("files_changed") or learned.get("skill"))
        if wrote_nothing:
            fallback = heuristic.consolidate(history, session_id)
            if fallback.get("skill") or fallback.get("facts"):
                fallback["note"] = "LLM pass found nothing reusable; heuristic distilled it"
                learned = fallback
        # The LIVE agent writes SKILL.md files directly — make sure they're indexed
        # so the NEXT run can actually match and load them.
        self.skills.reindex()
        return learned

    def close(self) -> None:
        self.db.close()
