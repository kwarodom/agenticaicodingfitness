# ▶ Factory Lab 01 — Run someone else's factory

> Part of Week 27 · Software Factory: an org chart made of loops. You install a working factory on **Alto Mini**, drain one ticket, and read every comment the lanes leave behind before you build anything yourself.

**What you'll actually do**
- Install Eric Tech's open-source `super-board` on `week27/00_alto_mini` and run its 8-step onboarding.
- Write one ticket in the standard format for a real seeded bug (the `—` on the Alerts page).
- Drain it Build → QA → Review → merge, and reconstruct what happened from the issue and PR threads alone.
- Record cost, bounces and where the human gate sat.

**Time** ~90 min · **Difficulty** beginner · **Needs** Claude Code (or Codex), `gh` logged in, `jq`, Python 3.9+, a GitHub repo you own

**Sources:** Eric Tech, [I Built a Self-Improving Software Factory](https://www.youtube.com/watch?v=aggJvNZxfKA) (7 Oct 2026) · [EricTechPro/super-board](https://github.com/EricTechPro/super-board) README, `skills/*/SKILL.md`, `references/ticket-format.md` · the course brief [`Software Factory — Research Brief and Week 27 Tutorials.md`](../Software%20Factory%20—%20Research%20Brief%20and%20Week%2027%20Tutorials.md) §2 and §6

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| A fork you own | `gh repo view --json nameWithOwner` inside your fork of this repo (or push `00_alto_mini` to a fresh repo) | the factory writes issues, branches and PRs |
| Alto Mini runs | `cd week27/00_alto_mini && make seed && make test` → `6 passed` | the QA lane needs a working app and a green baseline |
| Claude Code | `claude --version` | the lanes run as Claude Code skills (or Codex with `--codex`) |

```bash
# on: laptop
cd week27/00_alto_mini && make seed && make test && make dev   # leave running on :8127
```

## 1 · Install and onboard (15 min)

```bash
# on: laptop, in the repo root of your fork
curl -fsSL https://raw.githubusercontent.com/EricTechPro/super-board/main/get.sh | bash
# or, inside Claude Code:  /plugin marketplace add EricTechPro/super-board  →  /plugin install super-board@super-board
```

Then in Claude Code: `/super-board onboard`. The eight steps are Checks · GitHub · Board · Branch · AGENTS.md · Policies · Bug sources · Review. Accept the defaults, let it create the GitHub Project with columns Backlog / Ready / Building / QA / Review / Blocked / Done, and say **yes** to `guard-protected-push`. The plugin ships the skills only; the Checks step adds the guard hooks and scripts into your repo.

> 📌 **Versions move.** super-board was at 3.1.1 on 10 Oct 2026 and requires Claude Code, `gh`, `jq`, bash 3.2+ and Python 3.9+. If a step differs on your machine, the README's Quick start wins.

## 2 · Your first ticket (15 min)

Open `http://127.0.0.1:8127`. In the Alerts table, the `sensor_gap` alert for Deluxe 1205 shows `—` under "Yesterday kWh" because the room had no readings yesterday. That is your ticket. File it with the format in [`../factory/docs/agents/issue-tracker.md`](../factory/docs/agents/issue-tracker.md):

```markdown
## Problem
The Alerts page shows "—" for yesterday's kWh when a room had no readings, so staff cannot tell "no data" from "not loaded".

## Context
- **Where:** Alerts, `/` (table `data-testid="alerts"`), API `GET /api/alerts`
  - a. `make seed` then open `/`
  - b. find the `sensor_gap` row for Deluxe 1205
- **Who:** duty engineers reading the morning alerts

## Fix
Return `yesterday_kwh: null` plus `yesterday_status: "no_data"` from the API and render "no data" in the UI.

## Acceptance Criteria
- [ ] `GET /api/alerts` includes `yesterday_status` with values `ok` or `no_data` for every alert with a room
- [ ] the UI renders "no data" (not "—") when `yesterday_status == "no_data"`
- [ ] a pytest asserts `no_data` for room 1205 on the seeded database

## Risk
🟢 Low · read-only presentation change

## Blocked by
- None.
```

Title: `🐛 [bug] alerts: "—" shown when a room has no readings yesterday`. Run `/super-board lint` — it should pass; try deleting an AC and lint again to see it push back.

## 3 · Drain it (30 min)

Move the card to **Ready** and run `/super-board run alto-mini`. While it runs, keep two tabs open: the issue and the PR. Expected sequence (from `references/run.md` and the lane skills):

1. **Builder** creates `.claude/worktrees/issue-<N>-build/`, branch `issue-<N>-alerts-no-data`, implements test-first, opens a **draft PR**, posts `[builder] [report]`, moves the card to QA.
2. **Tester** fills the test-gap ledger (AC → test `file:line`), captures forensics, posts `[tester] ✅` or bounces with `[tester] ❌`.
3. **Reviewer** posts one `<!-- super-review:report -->` comment, may run the adversarial truth-check, then the merge gate re-runs tests on base + branch pinned to the reviewed SHA and merges.

**Expected output** (shape, not exact text)

```
🧱 Build  #12 → draft PR #13 · 3 files · 41 lines · tests: 7 passed
🧪 QA     #13 → ledger 3/3 ACs pinned · forensics docs/super-qa/report/… · ✅
🔍 Review #13 → 0 blockers · 1 nit · merge gate: tests green at a1b2c3d → merged
```

## 4 · Read the trail (20 min)

Answer in your lab notes, with links to the exact comments:
1. Where did state live? (Hint: nowhere local — issue + PR comments.) Quote the `[builder] [report]` block.
2. What did QA assert beyond "tests pass"? Find the ledger and the forensics path.
3. Did Review bounce? If yes, what was `not fixed`? If no, which finding was a nit?
4. Which column would a `money`- or `auth`-labelled card have landed in, and why?
5. Cost: `/super-board status` or `.claude/bin/super-board-usage.sh`. Record tokens and USD for one merged PR.

## 5 · Exercise

`ex01_trail.md`: a 10-line timeline of the run (who, when, which column, which artifact) plus the three numbers: changed lines, bounces, cost. Hand it in before Lab 02 — Lab 02 asks you to rebuild each of those steps as your own skill.
