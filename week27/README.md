# Week 27 · Software Factory — an org chart made of loops

Build the thing that builds the software: a kanban of linted tickets drained by unattended **Build → QA → Review**
lanes, guarded by deterministic hooks, measured by behavioural evals, and fed by production telemetry that files its
own verified tickets. The reference is Eric Tech's open-source **super-board**; the running target is **Alto Mini**,
a small hotel-energy app with seeded, ticketable bugs.

The course is the research brief [`Software Factory — Research Brief and Week 27 Tutorials.md`](Software%20Factory%20—%20Research%20Brief%20and%20Week%2027%20Tutorials.md)
(video §2, landscape §3, design principles §4, module design §5, labs §6–12, AltoTech notes §13, risks §14). The brief's Lab 0–5 are modules 01–06 here; its capstone is module 07.
turned into seven runnable modules plus the [`factory/`](factory/) starter kit.

## Start

```bash
# once — the target app
cd week27/00_alto_mini && make seed && make test        # 6 passed
make dev                                                 # http://127.0.0.1:8127

# once — the kit's offline checks (no Claude Code needed)
cd ../factory && ../../.venv/bin/python -m pytest -q tests          # guard hooks
cd ../ && ../.venv/bin/python -m pytest -q 0*/exercises/solutions    # exercise solutions

# Lab 01 — install the reference factory (needs Claude Code + gh)
curl -fsSL https://raw.githubusercontent.com/EricTechPro/super-board/main/get.sh | bash
```

## Modules

| # | module | brief | you build |
|---|---|---|---|
| 01 | [Run someone else's factory](01_run_someone_elses_factory/TUTORIAL.md) | §2, §6 | super-board on Alto Mini; one ticket drained; the trail reconstructed from GitHub alone |
| 02 | [The ticket and the Builder lane](02_ticket_and_builder/TUTORIAL.md) | §7 | five linted tickets, `CLAUDE.md`, `fitness-build`, headless `factory-run.sh` |
| 03 | [The QA lane](03_qa_lane/TUTORIAL.md) | §8 | test-gap ledger, red-first proof, Playwright forensics, a bounce |
| 04 | [Review and the merge gate](04_review_and_merge_gate/TUTORIAL.md) | §9 | "review remembers", adversarial truth-check, four merge-gate refusals |
| 05 | [Guards and evals](05_guards_and_evals/TUTORIAL.md) | §10 | five PreToolUse guards red-teamed; three `claude plugin eval` cases + your fourth |
| 06 | [Telemetry to tickets](06_telemetry_to_tickets/TUTORIAL.md) | §11 | verifier chain over stub Sentry/PostHog; `fitness-collect`; `fitness-lookback` |
| 07 | [Capstone](07_capstone/TUTORIAL.md) | §12 | a 48-hour factory run on a real project, reported honestly |

Each module has `TUTORIAL.md` and `exercises/` (with `solutions/` and tests).

## What is where

```text
week27/
  00_alto_mini/        the target: FastAPI + static page, SQLite seed, 6 tests, Makefile (dev/seed/test/e2e)
                       seeded gaps: None-reading TypeError · setpoint 31 → 500 · tz KeyError · "—" for no-data
  factory/             the starter kit (also shipped as a zip in class):
    CLAUDE.md, docs/agents/issue-tracker.md           project instructions + ticket format and lint rules
    skills/fitness-{build,qa,review,collect,lookback}  the five lanes as Claude Code skills
    .claude/settings.json, .claude/hooks/guard-*.py    PreToolUse guards (exit 2 blocks) + tests/test_hooks.py
    scripts/factory-run.sh, scripts/merge-gate.sh      headless drain of Ready; merge only at the reviewed SHA
    scripts/collect_*.py, telemetry/*.json             stub Sentry/PostHog feeds and collectors
    evals/                                              3 `claude plugin eval` cases, graders, offline gh stub
  0N_*/                TUTORIAL.md, exercises/ (+ solutions/ with pytest)
```

## Honesty rules (short version)

- Everything that runs offline here has been run: Alto Mini's tests, the guard tests, the collectors, the eval seeds, the exercise solutions.
- Nothing has been run against a **live Claude Code CLI** by the course authors yet: `claude -p` flags, `claude plugin eval` grader `type` names and the headless lane prompts are written from the docs as read on 10 Oct 2026. Lab 05 says so; your first run is the verification.
- super-board facts (version 3.1.1, model ladder, policies, eval costs) are as read from the repo on 10 Oct 2026; the per-run cost figures are the author's estimates.
- Thai translations (`TUTORIAL.th.md`) are not yet written for this week.

## Where it points next

The same lanes, pointed at AltoTech repos, add the human-merge labels `control`, `tenant` and `pdpa`, route the
Builder through the Week 25 LiteLLM gateway for the sovereign tier, and treat Alto Copilot's activity log and the
analytics MCP as the "signals" that `fitness-collect` turns into tickets. See brief §13.
