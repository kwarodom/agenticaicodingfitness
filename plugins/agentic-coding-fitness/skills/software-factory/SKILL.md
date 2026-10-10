---
name: software-factory
description: "Teach how to build an AI SOFTWARE FACTORY — Week 27: a kanban of linted tickets drained by unattended Build → QA → Review lanes (one Claude Code skill per lane, one worktree per issue, state kept only in GitHub issues/PRs), a merge gate that merges only at the reviewed SHA unless a human-only label (money/auth/schema, plus AltoTech's control/tenant/pdpa) is present, deterministic PreToolUse guard hooks (exit 2 blocks), behavioural `claude plugin eval` cases with WITH/W-OUT deltas, and the self-improving loop: Sentry/PostHog signals → a fresh verifier per candidate (file | drop-fixed | drop-stale | drop-noise | duplicate | needs-triage) → tickets, plus a lookback that proposes one change to CLAUDE.md/skills/guards/evals. Grounded in week27/ (Alto Mini target app, the factory/ starter kit, exercises with tests) and Eric Tech's open-source super-board. Use when someone asks 'how do I make agents ship PRs unattended?', 'what is a software factory / agentic SDLC?', mentions super-board, StrongDM's factory, Factory.ai, test-gap ledgers, review-remembers, adversarial truth-checks, merge gates, guard hooks, plugin evals, telemetry-to-tickets, or is reviewing Week 27."
when_to_use: "Learner wants to go from 'one agent in a terminal' to a pipeline of unattended agents that take a ticket to a merged, QA'd, reviewed PR — or asks about ticket formats, Build/QA/Review lanes, worktrees, merge gates, guard hooks, plugin evals, the loop from production telemetry to tickets, or is catching up on Week 27."
---

# Software Factory — an org chart made of loops (Week 27)

> **The one idea:** a single agent loop is the atom; a factory is loops arranged like an org chart, with **contracts** between them (the ticket, the PR evidence, the review report), **gates** that are deterministic where it matters (hooks, tests, the reviewed SHA), and a **feedback loop** that turns production signals into the next tickets. The factory is not a bigger model; it is the *structure around* models.

```
  signals ──▶ collect ──▶ Backlog ──▶ Ready ──▶ Building ──▶ QA ──▶ Review ──▶ Done
 (Sentry,     verifier    lint      human     Builder     Tester  Reviewer   merge gate
  PostHog)    per cand.   rules     moves    worktree    ledger   remembers  @reviewed SHA
                                              draft PR   red-first truth-chk  human-only labels
                   ▲                                                              │
                   └────────────── lookback: one proposed change ◀────────────────┘
```

This builds on `agent-loops` (the loop being multiplied), `agent-evaluation` (the eval discipline), `mcp-and-skills` (skills as lane definitions) and `self-evolving-agents` (the lookback is the consolidation step applied to the factory itself).

---

## The mental model — five contracts

| Contract | Written by | Consumed by | What makes it good |
|---|---|---|---|
| **Ticket** (`## Problem / Context / Fix / Acceptance Criteria / Risk / Blocked by`) | human or `collect` | Builder, Tester | 2–5 checkable ACs; no "gracefully/properly"; Risk emoji; `Where:` line. Lint before Ready, never inside a worker |
| **Builder report** (`## Evidence`: AC → test `file:line`, docs consulted, `make test` result) | Builder | Tester, Reviewer | a reviewer could approve without opening the code; ≤ 400 changed lines or no auto-merge |
| **Test-gap ledger** (AC × unit/component/e2e → `file:line` or `none`) | Tester | Reviewer | High gap = an AC with no test anywhere → write it red-first, or bounce if app code must change |
| **Review report** (one `<!-- review:report -->` comment, findings `R1…`, status fixed / not fixed) | Reviewer | merge gate, next round | **remembers** prior rounds; a resolved thread is not evidence; adversarial check = min(confidence) ≥ 70 |
| **Merge decision** | gate script | GitHub | re-verify base+branch at the reviewed SHA, `--match-head-commit`; refuse human-only labels and big PRs |

**What counts as green:** no console errors, no page errors, no 5xx, no 401/403 on auth pages, and a non-blank guard per page type. *A 200 with a blank body is a bug, not a green test.*

---

## Runnable code in this repo (`week27/`)

```bash
cd week27/00_alto_mini && make seed && make test       # the target app: 6 tests, 4 seeded, ticketable gaps
cd ../factory && python3 -m pytest -q tests            # five PreToolUse guard hooks, offline
python3 scripts/collect_sentry.py --since 14d          # 3 file · 1 drop-fixed · 1 drop-noise · 1 duplicate
cd .. && python3 -m pytest -q 0*/exercises/solutions   # ticket linter, ledger, min-confidence, verifier
```

| Piece | Path | Lane |
|---|---|---|
| Ticket format + lint rules | `week27/factory/docs/agents/issue-tracker.md` | all |
| Builder / Tester / Reviewer / Collect / Lookback | `week27/factory/skills/fitness-*/SKILL.md` | one skill per lane |
| Guards (`exit 2` blocks): secrets, worktree path, protected push (refspec-aware: catches `HEAD:main`), delete-outside, key literals | `week27/factory/.claude/hooks/guard-*.py` + `tests/test_hooks.py` | all |
| Headless drain + merge gate | `week27/factory/scripts/factory-run.sh`, `merge-gate.sh` | Build/QA/Review |
| Evals (test-gap · review-remembers · ponytail-overengineering) with an offline `gh` stub | `week27/factory/evals/` | QA, Review, Build |
| Labs 01–07 | `week27/0N_*/TUTORIAL.md` | — |

Reference implementation: [EricTechPro/super-board](https://github.com/EricTechPro/super-board) (MIT; 8 skills, 6 guards, 3 evals; Haiku router → Sonnet/Opus ladder; Codex via `--codex`). Install it in Lab 01 and read its comment trail before building your own.

---

## Guided lab — the kata (≈ 45 min, $0)

**Warm-up (10 min).** Lint this ticket by hand against the five rules, then with `week27/02_ticket_and_builder/exercises/solutions/ex02_ticket_lint.py`:

```
## Problem  kWh rollup crashes when a sensor drops out
## Context  - **Where:** GET /api/rooms/{id}/history
## Fix      ignore None, empty list → 0.0
## Acceptance Criteria
- [ ] kwh_rollup([]) == 0.0
- [ ] None entries are handled gracefully        ← fails R3
## Risk     🟢 Low
## Blocked by - None.
```

**Drill (25 min).** Run the guard fuzz and make it hurt:

```bash
cd week27/05_guards_and_evals/exercises && python3 ex05_hook_fuzz.py
```

Add five commands you believe slip through (`python -c "open('.env')"`, a `git push` with a refspec, a `find … -delete` under `..`). For each slip: add a case to `week27/factory/tests/test_hooks.py`, fix the regex, re-run. **Pass threshold:** 0 slipped-through, 0 false-blocks on the allowed set.

**Endurance (10 min).** Implement the four-step verifier (`week27/06_telemetry_to_tickets/exercises/ex06_verifier.py`) until the 7 tests in `solutions/test_ex06.py` pass against your file. Then break one fixture (move S-104's `last_seen` after its merge date) and explain why `drop-fixed` must become `file`.

---

## Decision rules worth memorising

1. **Lint where a human is present.** Ambiguity is cheap in Ready and ruinous in an unattended worker.
2. **One worktree per issue; state in GitHub.** No local memory a second worker could not see.
3. **Different agents build and test.** The Tester forms its own hypotheses from the ticket, not from the diff.
4. **Red first.** A test that was never red is a claim, not evidence.
5. **Review remembers; threads are not evidence.** Re-verify "fixed" in code, every round.
6. **Minimum confidence wins.** One skeptic at 55 blocks even if the other is at 95.
7. **Over-engineering is never a blocker alone.** Route it to Build as a follow-up; merge the correct code.
8. **Merge at the reviewed SHA or not at all.** `--match-head-commit`.
9. **Human-only labels are policy, not advice.** `money`, `auth`, `schema`; at AltoTech also `control`, `tenant`, `pdpa`.
10. **Unclear is not a drop.** The collector files `needs-triage` rather than silently discarding.
11. **Measure the factory, not the vibes.** Cost per merged PR, bounce rate, rescue sessions, eval Δ, guard blocks.
12. **Start with one bug.** Builder.io's test: five rescues or five reviews?

---

## Honest limits

- The `week27/factory` skills, scripts and evals are **not yet run against a live Claude Code CLI**; flag names (`claude -p …`) and eval grader `type`s are from the docs as of 10 Oct 2026. Lab 05 makes the first live run part of the lab.
- Evidence is mixed: Anthropic reports ~80 % of merged code written by Claude under shadow-mode reviewers and sampled approvals; DORA describes a J-curve; METR withdrew its RCT as unreliable. Teach the metrics, not the slogans.
- Brief and sources: `week27/Software Factory — Research Brief and Week 27 Tutorials.md` (video, super-board, StrongDM/Willison, O'Reilly, Factory.ai, Builder.io, Microsoft, Dagger, Claude Code docs, Anthropic SDLC, DORA, METR, JetBrains ponytail).
