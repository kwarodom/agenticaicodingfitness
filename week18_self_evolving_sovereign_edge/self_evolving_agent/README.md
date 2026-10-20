# 🧠 Week 18 — Self-Evolving Agent with Persistent Memory

Runnable Python that turns a *stateless* agent into one that **remembers, learns,
and gets cheaper with every interaction** — the closed learning loop from the
[*Self-Evolving Agent*](self_evolving_agent_tutorial.pdf) tutorial (inspired by the
Hermes Agent, Nous Research), grounded in this course's hotel/dev domain.

> A stateless agent performs identically on task #1 and task #1000. A
> self-evolving agent exhibits **compound returns**: each run adds to its memory,
> so identical work gets measurably faster and cheaper. By run 5 of an identical
> task the tutorial reports **64% fewer turns and 66% lower cost — with no human
> tuning.**

This package makes that real two ways:

1. **A step-by-step explainable web app** (`tutorial_server.py`) — click through
   the architecture and run all 7 checkpoints, output streaming live.
2. **A live "watch it evolve" demo** (`server.py`) — drive the agent in the
   browser: run a task → consolidate (the subconscious loop) → run the *same*
   task again and watch it load the skill it wrote and finish faster. The three
   memory layers update in real time.

```
The problem week-2..17 left open, and what solves it here:

  A new session starts from a blank slate, every time   →  TRIPARTITE MEMORY
  ───────────────────────────────────────────────────      (episodic + semantic
  It re-explains context, repeats mistakes, never           + procedural) refreshed
  gets faster at a task it has done before                  by a background
                                                            CONSOLIDATION LOOP
```

---

## The mental model — the Tripartite Memory Model

```
                                ┌──────────────────────────────────────┐
   USER ──▶ ① RUN a task ──────▶│ system prompt = BASE                  │
                                │   + <memory-context> MEMORY/USER </>   │  ← semantic
                                │   + <skill> matched SKILL.md </skill>  │  ← procedural
                                └───────────────┬──────────────────────┘
                                                │ logged verbatim
                                                ▼
                                        EPISODIC SessionDB (SQLite)
                                                │
   ② CONSOLIDATE (background, the "subconscious")│  meta-cognitive agent
                                                ▼
            ┌───────────────────────────────────────────────┐
            │ MEMORY.md / USER.md  (facts)   skills/*.md (how)│  ← persisted
            └───────────────────────────────────────────────┘
                                                │
   ③ next run starts SMARTER  ◀──────── loads the above ──────┘  (compound returns)
```

| Layer | Brain analogue | Store | Answers |
|-------|----------------|-------|---------|
| **Episodic** | hippocampus | `SessionDB` (SQLite, WAL, FTS5) | "What exactly did we say 5 min ago?" |
| **Semantic** | neocortex | `MEMORY.md` + `USER.md` (+ optional vectors) | "Who is this user? What do I know?" |
| **Procedural** | cerebellum | `SKILL.md` library | "What's the optimal workflow for X?" |

---

## Layout — all the code in this subfolder

```
week18/self_evolving_agent/
├── config.py                       models, the on-disk memory tree, .env / capability probe
├── core/
│   ├── session_db.py               Part 3 — Episodic engine: WAL, jitter backoff,
│   │                                        declarative schema evolution, dual FTS5
│   ├── semantic_memory.py          Part 4 — MEMORY.md/USER.md + XML context fencing + vectors
│   ├── skill_library.py            Part 5/7.2 — SKILL.md matching, injection, versioning, rollback
│   ├── consolidation.py            Part 6 — the meta-cognitive loop + PreCompact snapshot
│   ├── garbage_collection.py       Part 7 — TTL compression + GDPR-safe erasure
│   ├── gepa.py                     Part 8 — Genetic-Pareto prompt evolution
│   ├── agent.py                    Part 10 — SelfEvolvingAgent (real SDK + offline sim)
│   └── _aio.py                     run each SDK call in its own thread/loop
├── checkpoints/                    7 self-contained steps (1–6 offline; 7 = capstone)
│   ├── checkpoint1_episodic.py         …7_self_evolving.py
├── server.py + static/index.html   the LIVE self-evolution visualizer  (port 8088)
├── tutorial_server.py + static/guide.html   the step-by-step guide      (port 8090)
└── memory/                         (generated) MEMORY.md, USER.md, skills/, agent_state.db …
```

---

## Setup

```bash
# from the repo root, using the repo's uv-managed .venv (Python 3.13)
uv pip install -r week18/self_evolving_agent/requirements.txt
```

**LIVE mode** (real LLM agent turns + real consolidation) needs the `claude` CLI,
which uses your Claude Code subscription auth — **no API key**:

```bash
npm install -g @anthropic-ai/claude-code   # then run `claude` once to sign in
```

Without it, everything still runs in **simulated** mode (deterministic, offline,
$0). The checkpoints 1–6 are pure stdlib and never need the CLI at all.

---

## Run it — two front doors

### A) The step-by-step guide (recommended start)

```bash
.venv/bin/python week18/self_evolving_agent/tutorial_server.py     # → http://127.0.0.1:8090
```

Click through *Understand the architecture* → run *Checkpoints 1–7* (output
streams live) → *Watch it self-evolve* (starts the visualizer + opens it).

### B) The live visualizer directly

```bash
.venv/bin/python week18/self_evolving_agent/server.py              # → http://127.0.0.1:8088
```

Then, in the browser: **Run the task → 🧠 Consolidate → Run the SAME task again.**
Watch the turns/cost bar drop and the new `SKILL.md` appear. Toggle **LIVE ⇄ SIM**
in the header.

### C) Just the checkpoints (no browser)

```bash
for n in 1 2 3 4 5 6 7; do
  .venv/bin/python week18/self_evolving_agent/checkpoints/checkpoint${n}_*.py
done
```

| CP | What you build | The one idea |
|----|----------------|--------------|
| 1 | **SessionDB** — WAL, jitter backoff, schema evolution, dual FTS5 | SQLite is a real concurrent agent store |
| 2 | **Semantic memory + context fencing** | recalled memory is *reference data*, not commands — defuse injection |
| 3 | **SKILL.md library** — match, inject, version, rollback | procedural memory is how the agent gets faster |
| 4 | **Background consolidation** | episodic → semantic + procedural, with zero foreground latency |
| 5 | **Memory GC** — TTL compression + GDPR erasure | unbounded memory has its own cost |
| 6 | **GEPA** — Pareto prompt evolution | the agent tunes its own prompts |
| 7 | **Capstone** — same task ×3 | **compound returns: fewer turns & lower cost each run** |

---

## LIVE vs SIMULATED — read this

| | **LIVE** (real LLM) | **SIMULATED** (offline) |
|---|---|---|
| Agent turns | real `claude_agent_sdk` runs in a sandbox | deterministic scripted turns |
| Consolidation | a real meta-cognitive agent **edits** MEMORY.md / SKILL.md | rule-based distiller |
| Cost | a few cents per run | $0 |
| Compound-returns curve | **real, but turn counts vary** run to run (it's a model) | **guaranteed** drop |
| Best for | proving the loop is *authentic* | teaching the *mechanism* cleanly |

The honest caveat (also shown in the UI): in LIVE mode a real model may not use
fewer turns every single run — the reliable learning signal is that the agent
**wrote a skill and loads it on the rerun**, and memory visibly grows. SIM mode
gives the clean, repeatable compound-returns curve the tutorial quantifies.

---

## Notes & honesty

- **No API key is used or wanted.** LIVE mode goes through the `claude` CLI's
  subscription auth; the apps deliberately drop `ANTHROPIC_API_KEY` so an
  unrelated key can't break the CLI.
- **The vector store (Part 4.5) is optional** — install `chromadb` to embed and
  retrieve only query-relevant facts instead of injecting all of MEMORY.md.
- **`memory/` is generated and gitignored.** "Reset" / "Start over" wipes it so
  you can re-watch the agent learn from an amnesiac slate.
- Model: `claude-haiku-4-5` for turns, `claude-sonnet-4-6` for consolidation/GEPA
  — swap freely in `config.py`.

## Where this sits in the course

```
agent-loops (W4–5) → tools (W3) → MCP (W7) → memory graphs (W14) → MAS (W9,W15)
→ long-running + distributed (W17)
                          │
                          ▼
        Week 18 — agents that PERSIST and IMPROVE across sessions
        (this folder + the sibling ../agent_loop comprehensive tutorial)
```
