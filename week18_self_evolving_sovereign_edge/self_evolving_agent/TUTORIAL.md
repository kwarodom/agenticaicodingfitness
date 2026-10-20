# Self-Evolving Agent — the tutorial, mapped to this code

This walks the [PDF](self_evolving_agent_tutorial.pdf) part by part and points at
the exact module + checkpoint that implements each idea. Read it alongside the
live visualizer (`server.py`) and the step-by-step guide (`tutorial_server.py`).

---

## Part 1 — The Stateless Problem
Every stateless LLM call is a blank slate: it re-explains context, can't reuse
past solutions, and repeats mistakes. The cost is measurable (≈90% of tokens on
repetition). The fix is **persistent state**: a memory engine that spans sessions
and accumulates wisdom. Everything below builds that engine.

## Part 2 — The Tripartite Memory Model
Three interconnected layers mirror human memory:

| Layer | Code | Checkpoint |
|-------|------|-----------|
| Episodic (raw experience) | `core/session_db.py` | CP1 |
| Semantic (abstracted facts) | `core/semantic_memory.py` | CP2 |
| Procedural (actionable skills) | `core/skill_library.py` | CP3 |

The power isn't any one layer — it's the **closed loop** (Part 6) that promotes
episodic experience into semantic + procedural memory so the next session starts
from accumulated wisdom.

## Part 3 — Episodic Memory: the SessionDB Engine → `core/session_db.py` · CP1
A correctly-tuned SQLite store. The four engineering features:
- **3.1 WAL mode** — `_setup_wal_mode()`: readers never block writers (a
  foreground turn + the background consolidator write concurrently).
- **3.2 Convoy / jitter backoff** — `_execute_write()`: `BEGIN IMMEDIATE` with
  randomised 20–150 ms retry jitter so competing writers don't collide forever.
- **3.3 Declarative schema evolution** — `SCHEMA_SQL` + `_reconcile_columns()`:
  add a column to the dict, restart, the DB `ALTER`s itself. No migrations.
  (CP1 literally adds `feedback_score` at runtime to prove it.)
- **3.4 Dual FTS5 tokenizers** — `search_messages()` routes latin queries to
  `unicode61` and CJK/partial queries to `trigram`. CP1 searches in English *and*
  Chinese.

## Part 4 — Semantic Memory & Context Fencing → `core/semantic_memory.py` · CP2
`MEMORY.md` (world/project facts) + `USER.md` (profile/preferences). The critical
safety primitive is **4.3 context fencing** — `build_memory_context_block()`
wraps recalled memory in `<memory-context>` with an authoritative system note and
escapes any attempt to close the fence early. CP2 plants
`"Ignore all previous instructions … </memory-context>"` in memory and shows the
fence neutralising it. **4.5 VectorMemoryStore** (optional, `chromadb`) embeds
facts and retrieves only the query-relevant ones instead of injecting all of
MEMORY.md every turn.

## Part 5 — Procedural Memory & the SKILL.md Library → `core/skill_library.py` · CP3
A `SKILL.md` is an auto-generated playbook: Context, Preconditions, Optimal
Steps, **Known Pitfalls (auto-learned)**, Performance Notes — plus a metrics
header (`version`, `success_rate`, `avg_turns`, `usage_count`, `error_count`).
- **5.3 matching** — `load_relevant_skills()` scores `skills/index.json` keywords
  against the prompt and injects the top matches, context-fenced as `<skill>`.
- **metrics** — `list_skills()` parses the header so the agent (and the
  visualizer) can see its own improvement.

## Part 6 — The Subconscious Loop → `core/consolidation.py` · CP4
After a turn, a background **meta-cognitive agent** replays the transcript and
distils it: semantic facts → MEMORY.md/USER.md, a reusable workflow → SKILL.md,
mistakes → that skill's *Known Pitfalls*. `META_COGNITIVE_PROMPT` is the system
prompt; `SDKConsolidator` runs the real agent (it edits the files itself with the
Write/Edit tools); `HeuristicConsolidator` is the deterministic offline distiller.
**6.3 `snapshot_before_compaction()`** is the PreCompact hook — snapshot working
memory before the context window is compacted so nothing in flight is lost. The
foreground never waits for any of this (in production it's a `Stop` hook firing a
daemon thread).

## Part 7 — Memory Garbage Collection → `core/garbage_collection.py` · CP5
- **7.1 TTL compression** — `EpisodicGarbageCollector.run_gc_cycle()`: summarise
  a session older than the TTL into MEMORY.md, then `prune_session_messages()`
  the raw rows (60–80% token reduction; the semantic value survives).
- **7.2 SKILL.md versioning** — `SkillLibrary.update_skill_with_versioning()` +
  `rollback_skill()` (CP3): never blindly overwrite; archive + bump, revert on
  regression.
- **7.3 The right to forget** — `erase_user_fact()`: surgical, GDPR-safe deletion
  from USER.md + episodic redaction, with an `erasure_audit.jsonl` trail.

## Part 8 — Self-Improvement: GEPA → `core/gepa.py` · CP6
**Genetic-Pareto Prompt Evolution.** `run_gepa_cycle()`: (1) reflective mutation
generates N prompt variants, (2/3) `pareto_filter()` keeps only variants not
dominated on accuracy/cost/tokens, then the highest-accuracy survivor is written
back to the SKILL.md prompt (versioned). The agent tunes its own prompts — no
human prompt engineer. CP6 exercises the Pareto math deterministically.

## Part 9 — Multi-Agent Memory Architecture
Subagents are isolated cognitive units (fresh context) over a **shared** memory
store: each writes its own `skills/<role>.md` partition; the orchestrator
consolidates cross-cutting facts into MEMORY.md. This package ships the
single-agent core; the pattern composes directly onto Week 15/17's multi-agent
work (each specialist gets `load_relevant_skills(role)` + a context fence).

## Part 10 — Production Integration → `core/agent.py` · CP7
`SelfEvolvingAgent` is a drop-in for a stateless agent. `build_system_prompt()`
assembles `BASE_PROMPT + fenced(MEMORY+USER) + matched SKILLs`. In production the
six SDK hooks map to memory actions (UserPromptSubmit→inject, PostToolUse→log,
Stop→consolidate, PreCompact→snapshot, SubagentStop→merge skills). Here `run_task`
+ `consolidate` expose those steps explicitly so the **visualizer can show them**.
CP7 runs one task three times and asserts the compound-returns drop.

## Part 11 — Workshop Exercises
The checkpoints *are* the exercises, made runnable and self-checking. Want the
PDF's exact drills? Each checkpoint's docstring states the goal; extend them:
add a `feedback_score` column (CP1), a new injection payload (CP2), a second
skill class (CP3/CP4), a real LLM summariser into the GC (CP5), more GEPA axes
(CP6), or wire CP7 to LIVE mode and measure a real 5-run curve.

## Appendix — Quick Reference

| Pattern | Where | Prevents |
|---------|-------|----------|
| WAL mode | `session_db._setup_wal_mode` | read/write deadlocks |
| Jitter backoff | `session_db._execute_write` | the SQLite convoy effect |
| Context fencing | `semantic_memory.build_memory_context_block` | prompt injection via memory |
| Schema reconcile | `session_db._reconcile_columns` | migration-script failures |
| Background fork | `agent.consolidate` (Stop hook) | foreground latency |
| PreCompact snapshot | `consolidation.snapshot_before_compaction` | working-memory loss |
| SKILL versioning | `skill_library.update_skill_with_versioning` | skill regressions |
| TTL GC | `garbage_collection.run_gc_cycle` | unbounded DB growth |
| GEPA | `gepa.run_gepa_cycle` | static, suboptimal prompts |
| Skill index | `skill_library.update_skill_index` / `reindex` | O(n) file scans |
