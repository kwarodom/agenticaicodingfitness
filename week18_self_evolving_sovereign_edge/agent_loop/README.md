# 🔁 Week 18 — The Agent Loop (Claude Agent SDK)

An **interactive, explainable web app + runnable demos** for the
*Claude Agent SDK — Agent Loop Engineering* tutorial
(`week18/agent_loop_comprehensive_tutorial.pdf`).

Where the PDF *explains* the loop, this folder lets you **watch it run**. Every
demo makes a **real** call through the Claude Agent SDK and narrates the loop as
it turns:

```
RECEIVE PROMPT → REASON/PLAN → ACT (tool) → OBSERVE result → ↺ repeat → FINAL RESULT
```

> **These are real agent loops, not mocks.** The SDK drives your local `claude`
> CLI, which uses your existing Claude Code / subscription sign-in — so **no
> `ANTHROPIC_API_KEY` is required** when you're already signed in. Each demo is
> capped on turns *and* USD (`max_turns`, `max_budget_usd`), so a full
> click-through of the tutorial costs only a few cents.

---

## Quick start

```bash
# from the repo root, using the repo's uv-managed .venv (Python 3.13)
uv pip install -r week18/agent_loop/requirements.txt

# 1) the interactive web app  (recommended — click through all 10 parts)
.venv/bin/python week18/agent_loop/tutorial_server.py
# → open http://127.0.0.1:8090

# 2) or run any demo straight from the terminal
.venv/bin/python week18/agent_loop/demos/step01_hello_agent.py
```

Prerequisite for **live** runs: the `claude` CLI installed and signed in.

```bash
npm install -g @anthropic-ai/claude-code   # then run `claude` once to sign in
```

If the CLI is missing, the web app still loads and the demos print install
hints instead of calling the model.

---

## What the web app gives you

For each of the 10 tutorial parts the guide shows:

1. **The concept** — the *why*, distilled from the PDF.
2. **The exact source** — a "View demo source" toggle (no hidden magic).
3. **A live run** — click *Run this loop live* and watch the loop stream into
   the browser, colour-coded so you can read it at a glance:
   - <kbd>session</kbd> · <kbd>reasoning</kbd> · <kbd>ACT</kbd> (a tool call =
     one turn) · <kbd>OBSERVE</kbd> (the tool result) · <kbd>result</kbd>.

The shared engine is [`loopview.py`](loopview.py) — a thin wrapper over
`claude_agent_sdk.query()` that labels every message as **REASON / ACT /
OBSERVE / RESULT**, counts turns, and prints the real USD cost. Every demo uses
it, so the loop is never a black box.

---

## Layout

```
week18/agent_loop/
├── README.md                     → this file
├── requirements.txt              → claude-agent-sdk + FastAPI
├── config.py                     → models, budget caps, seeded sandbox
├── loopview.py                   → the "make the loop visible" engine (shared)
├── tutorial_server.py            → FastAPI control plane for the web app
├── static/guide.html             → the clickable, streaming tutorial UI
└── demos/                        → one runnable loop per tutorial part
    ├── step01_hello_agent.py        Part 1–2 · the smallest real loop
    ├── step02_turns_messages.py     Part 3   · turns, message types, results
    ├── step03_builtin_tools.py      Part 4a  · built-in tools & permissions
    ├── step04_custom_tool.py        Part 4b  · a custom @tool over a fake CRM
    ├── step05_hooks_safety.py       Part 5   · PreToolUse safety gate + audit log
    ├── step06_sessions.py           Part 6   · capture & resume a session
    ├── step07_multi_agent.py        Part 7   · orchestrate specialist subagents
    ├── step08_usecase_triage.py     Part 8   · production: clear a support queue
    └── step09_production.py         Part 9   · bounded execution & cost control
```

---

## The 10 parts → what you'll watch happen

| Part | Demo | The one thing to notice |
|------|------|--------------------------|
| 1 | *(concept)* | The loop = REASON → ACT → OBSERVE → repeat. |
| 2 | `step01` | Claude must **ACT** (use a tool) to learn what files exist — it can't guess. |
| 3 | `step02` | Every **message type** tagged live; what each `ResultMessage` subtype means. |
| 4a | `step03` | `disallowed_tools` **always wins** — Bash never runs, even when asked. |
| 4b | `step04` | A **custom tool** is just an async fn + a good description; the loop calls *your* systems. |
| 5 | `step05` | A **PreToolUse hook** blocks `rm -rf` before it runs; a PostToolUse hook audits everything. |
| 6 | `step06` | Run 2 **resumes** Run 1's session and recalls a fact never in its prompt. |
| 7 | `step07` | The orchestrator **delegates** to subagents with fresh contexts, then synthesises. |
| 8 | `step08` | One loop **clears a support queue** — resolve L1s, escalate the critical incident. |
| 9 | `step09` | `max_budget_usd` **stops a runaway loop**; you handle the stop subtype. |
| 10 | *(concept)* | The production deployment checklist. |

---

## Cost & safety notes

- Demos default to the **fast/cheap** model (`claude-haiku-4-5`) and small caps.
  A whole run of all 9 demos is typically well under **$0.20**.
- Tool-using demos work in a throwaway `.sandbox/` (seeded with sample files);
  `🧹 Clean sandbox` in the web app, or `POST /api/cleanup`, removes it.
- `step05` deliberately *asks* for a destructive command to prove the safety
  hook denies it — nothing dangerous actually runs, and it's confined to the
  sandbox regardless.

---

## Where this sits in the course

```
tool-use (W3) → agent loops (W4–5) ──────────────┐
                                                  ▼
   THIS WEEK (W18): the loop as an engineering discipline —
   tools, hooks, sessions, subagents, real use cases, production caps
                                                  │
   builds toward → long-running & distributed agents (W17), MAS (W9/W15)
```

Week 4–5 introduced the ReAct loop conceptually. Week 18 makes it **production
software**: bounded, observable, safe, resumable, and wired to real systems —
demonstrated on real loops you can watch run.

---

## One environment note

Recent `claude` CLI versions can *defer* tools and have the model load them
mid-loop via a built-in `ToolSearch` step. That's handy in environments with
hundreds of tools, but it adds noise to a tutorial about the REASON → ACT →
OBSERVE loop. These demos set `ENABLE_TOOL_SEARCH=0` (in `config.py`) so the
loop reads exactly as the PDF describes it. Remove that line if you want to see
the deferred-tool behaviour.
