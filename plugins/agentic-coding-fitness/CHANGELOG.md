# Changelog

All notable changes to the **Agentic Coding Fitness** plugin.
This project adheres to [Semantic Versioning](https://semver.org/).

## [2.3.0] — 2026-10-10

Adds the Week 27 course module and takes the plugin from **20 → 21 skills**. The headline: from *one agent in a terminal* to a **software factory** — unattended lanes that take a linted ticket to a QA'd, reviewed, merged PR, with deterministic guards, behavioural evals and a telemetry-to-tickets loop.

### Added — 1 new skill
- **software-factory** (W27) — the five contracts (ticket, builder evidence, test-gap ledger, review report, merge decision), Build → QA → Review lanes as one Claude Code skill each with one worktree per issue and state kept only in GitHub, "review remembers" + the adversarial truth-check (min-confidence ≥ 70), the merge gate (`--match-head-commit` at the reviewed SHA; human-only labels `money`/`auth`/`schema`, plus AltoTech's `control`/`tenant`/`pdpa`), five `PreToolUse` guard hooks (exit 2 blocks), `claude plugin eval` cases with WITH/W-OUT deltas, and the self-improving loop (Sentry/PostHog → fresh verifier per candidate → tickets; lookback proposes one change). Grounded in `week27/` (Alto Mini target app, `factory/` starter kit, exercises with tests) and Eric Tech's open-source super-board; ships a $0 kata (ticket lint → guard fuzz → verifier).

### Added — course content (`week27/`)
- `00_alto_mini/` — the factory target: FastAPI + static page, SQLite seed, Makefile, 6 tests, four seeded ticketable gaps.
- `factory/` — the starter kit: five lane skills, five guard hooks with offline tests, `factory-run.sh`, `merge-gate.sh`, stub telemetry + collectors, three eval cases with an offline `gh` stub.
- Labs 01–07 (`TUTORIAL.md` + `exercises/` with tested solutions) and the research brief.

### Changed
- `guard-protected-push.py` is refspec-aware: the Lab 05 fuzz found `git push origin HEAD:main` slipping past the first version; now blocked, with four new test cases.

### Versioning
- Bumped `plugin.json` and `.claude-plugin/marketplace.json` to **2.3.0**; descriptions and keywords (`software-factory`, `agentic-sdlc`, `merge-gate`, `guard-hooks`, `plugin-evals`) refreshed. Skills are auto-discovered from `skills/*/SKILL.md`.

### Honest limits
- Nothing in `week27/factory` has been run against a live Claude Code CLI yet; `claude -p` flags and eval grader names follow the docs as read on 2026-10-10. Thai translations for Week 27 are pending.

## [2.1.0] — 2026-06-17

Extends the plugin to cover the two newest course weeks (16–17), taking it from **16 → 18 skills**. The headline: agents that survive **real time** (pause for days, resume without losing context) and **service boundaries** (delegate across teams/frameworks), plus a config-driven production framework.

### Added — 2 new skills
- **nemo-agent-toolkit** (W16) — building production multi-agent systems with the **NVIDIA NeMo Agent Toolkit (NAT)**: the four building blocks (tools, agents, workflows, observability), registering custom tools via `FunctionBaseConfig` + `@register_function` → `FunctionInfo`, RAG tools via LlamaIndex, supervisor → specialist orchestration (LangGraph `StateGraph` under the hood) with HITL, YAML workflow composition that separates logic from instantiation, the `Builder`/`LLMFrameworkEnum` cross-framework abstraction, and NVIDIA NIM models. Grounded in `week16/` notebooks; ships a $0 `MockLLM` register→compose→route drill.
- **long-running-and-distributed-agents** (W17) — agents that **pause for days and resume without losing context**: the durable state machine (`current_step` in persisted session state, never chat history), Google ADK's four primitives (`ToolContext.state`, `before_agent_callback`, `DatabaseSessionService`, `Runner.run_async(state_delta=…)`), event-driven webhook resume, pause gates, the credential problem **auth.md** solves (store the durable *grant*, re-mint a scoped short-lived token at every wake; least privilege per sub-agent; revocation), and in-process `sub_agents` vs. cross-service **A2A**. Grounded in `week17/checkpoints/` (offline), `week17/hr_onboarding/`, and `week17/authmd_adk/`; ships a $0 durable-state-machine drill you can kill and resume.

### Changed
- **a2a-protocol** — **upgraded from "conceptual" to grounded.** A2A is now runnable offline in `week17/checkpoints/checkpoint5_a2a_cards.py` (cards + six-state lifecycle) and `checkpoint6_fleet.py` (ADK long-running + A2A end-to-end); only the *remote peer* is mocked. Banner, frontmatter, and repo anchors updated; cross-linked to `long-running-and-distributed-agents`.
- **curriculum-and-periodization** — extended the week-by-week map and the kata `SYLLABUS` to weeks 16–17; added the two new skills to the Performance phase; corrected the blueprint-vs-repo divergence note (week16 exists; the repo runs one week past the blueprint's 16 with week17's fleet/long-running capstone).
- **multi-agent-systems** — added NeMo Agent Toolkit and Google ADK to the framework cheat-sheet, with a "beyond one process" pointer to the Week 17 fleet layer.

### Versioning
- Bumped `plugin.json` and `.claude-plugin/marketplace.json` (top-level + nested) to **2.1.0**; refreshed descriptions (16 → 18 skills, weeks 2–17) and keywords (`google-adk`, `nemo-agent-toolkit`, `authmd`, `long-running-agents`, `agent-card`, …). Skills are auto-discovered from `skills/*/SKILL.md`.

## [2.0.0] — 2026-05-31

A major expansion driven by the *Agentic AI Fitness Class: Claude Code Plugin & Skill Strategy* blueprint. The plugin grows from a 9-skill course recap into a **16-skill curriculum-and-practice platform**, and every existing skill gains production-grade depth.

### Added — 7 new skills
- **production-and-observability** (W10) — take a prototype to production: *see it* (LangSmith tracing, OpenTelemetry `gen_ai.*` conventions), *stop it* (HITL `interrupt()` + durable `SqliteSaver` checkpointing, progressive autonomy, the OWASP-ASI 4-layer guardrail stack), *afford it* (budgets, cost-per-successful-task, CI cost-regression gate). Grounded in `week10/`.
- **agent-evaluation** (W10 & W15) — measure and **gate** agent quality: golden datasets, the eval-framework landscape (RAGAS/DeepEval/Braintrust/Inspect AI), LLM-as-judge with bias mitigation, and the 5-gate CI/CD pipeline (lint → eval → cost → canary → shadow). Grounded in `week15/kg_mastery/part4_evaluation/`.
- **agent-drills** — the practice menu: warm-up/skill/endurance/sparring/capstone katas, grounded in the 14 real `week11/exercises` (ex01–ex14) plus the flagship drills authored into the other skills.
- **curriculum-and-periodization** ⚠️ — the syllabus: four phases (Foundation → Strength → Endurance → Performance), progressive overload, deload weeks, and a map bridging the blueprint's idealized 16 weeks to this repo's real weeks 2–15.
- **vibe-coding-and-security** ⚠️ — working *with* a coding agent: context engineering (CLAUDE.md, six-layer framework, Preferred/Avoid pairs), test-driven vibe coding, the ~45% AI-code vulnerability reality + a self-review security loop, and a hand-off-vs-take-control decision tree.
- **a2a-protocol** ⚠️ — the Agent-to-Agent protocol (cross-framework counterpart to MCP): Agent Cards at `/.well-known/agent.json`, the six-state task lifecycle, polling/SSE/webhooks, and A2A vs MCP. Conceptual/stretch — not yet implemented in this repo.
- **skill-authoring** (W7+) — the meta-skill: the two skill types (Capability Uplift vs Encoded Preference), full frontmatter + progressive disclosure, the 7-step creation pipeline, and how to evaluate/maintain a skill. This plugin is the worked example.

### Changed — all 9 existing skills enriched
- **models-and-patterns** — reconciled the loose catalog into the canonical **12-pattern, 3-tier taxonomy** (Core/Workflow/Advanced), added a 12×4 framework-support matrix, a 5-question pattern-selection decision tree (with the compounding-error rationale), concrete **cost economics** (3-tier cascade, savings levers, break-even, cost-per-successful-task), and real **HITL + guardrails** coverage (progressive autonomy, OWASP-ASI stack, hand-off tree).
- **mcp-and-skills** — added *Capability Uplift vs Encoded Preference*, fuller SKILL.md anatomy + progressive disclosure, **building an MCP server** with FastMCP (grounded in `week15/.../05_fastmcp_server.py`), a one-screen **MCP security** awareness section, and **CLAUDE.md context engineering**.
- **multi-agent-systems** — expanded to a 6-pattern orchestration framework with a "which pattern?" decision table, **communication topologies + state management** (chain/star/mesh; shared/message-passing/event-driven/blackboard), and real `week10` supervisor/checkpointer/HITL code + Claude Agent SDK primitives.
- **agent-loops** — labelled the loop as **ReAct / Bounded Execution**, added Karpathy's 4 LLM failure modes for the reflection loop, and 5 deterministic token-budget guardrails ("denial of wallet" prevention).
- **rag-knowledge-agents** — its first runnable grounding: a **"how do I know my RAG is good?"** section built on `week15/kg_mastery/part4_evaluation/` (RAGAS metrics, golden testset, CI threshold gate).
- **llm-fundamentals, tool-use, agent-memory-graphs, knowledge-graph-mastery** — standardized guided labs into the **kata template** ($0 `MockLLM` warm-up + skill drill + weighted rubric + pass threshold), added cross-links to the new skills, the **Tool-Definition-Mastery** flagship drill (`tool-use`), a security caveat for model-written Cypher (`agent-memory-graphs`), and a prompt-caching cost note (`llm-fundamentals`).

### Versioning
- Bumped `plugin.json` and `.claude-plugin/marketplace.json` (top-level + nested) to **2.0.0**; refreshed descriptions (9 → 16 skills) and keywords. Skills are auto-discovered from `skills/*/SKILL.md`, so no skill list is hardcoded in the manifest.

## [1.1.0]
- 9 concept skills recapping the course: llm-fundamentals, tool-use, agent-loops, mcp-and-skills, rag-knowledge-agents, multi-agent-systems, agent-memory-graphs, knowledge-graph-mastery, models-and-patterns.
