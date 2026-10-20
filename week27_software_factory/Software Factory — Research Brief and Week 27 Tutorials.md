# Building a Software Factory — Research Brief and Week 27 Tutorials for the Agentic AI Class

Prepared 10 October 2026 (Bangkok) for Warodom Khamphanchai · Agentic Coding Fitness @ Rust Tech Bar

Reference: Eric Tech, "I Built a Self-Improving Software Factory" (12:37, published 7 Oct 2026) and the open-source `super-board` repository it demonstrates. The brief also draws on the 2026 software-factory literature (StrongDM, O'Reilly, Factory.ai, Builder.io, Microsoft Research, Anthropic) and on the existing structure of the `agenticaicodingfitness` repo and its `agentic-coding-fitness` plugin (v2.2.0, 20 skills, Weeks 2–26).

---

## 1. Executive summary

- A software factory is a repeatable, observable loop in which raw work (bug reports, telemetry, feature ideas) goes in and merged, evidenced pull requests come out, with humans at a few high-leverage gates. The term has converged in 2026 across Factory.ai, StrongDM, O'Reilly Radar, Builder.io and Microsoft Research; Addy Osmani's framing "the loop is the atom… a factory is an org chart made of loops" is the cleanest teaching definition ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).
- Eric Tech's `super-board` is a complete, free, MIT-licensed worked example: telemetry (Sentry, PostHog) → GitHub Issues → a GitHub Projects kanban → three unattended lanes (Build → QA → Review) → merge gate, plus six guard hooks, a model-routing ladder and behavioural evals. It is small enough (8 skills, 9 commands) to read in one class session and real enough (133 stars, pushed 8 Oct 2026) to run on a student repo ([GitHub](https://github.com/EricTechPro/super-board)).
- The evidence base is strong enough to teach but not to oversell. Anthropic reports Claude authors about 80 % of merged code internally with 8× output per engineer, under heavy automated review and risk-tiered human gates ([Anthropic](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle)). DORA warns of a J-curve "tuition cost" before gains ([InfoQ on DORA](https://www.infoq.com/news/2026/05/dora-roi-ai-assisted-dev-report/)); METR stopped its task-level RCT because results had become unreliable ([METR](https://metr.org/blog/2026-02-24-uplift-update/)); and a JetBrains paired study found the popular ponytail skill cut code written by about 15 % and cost by about 10 % on the median task, with wide variance ([JetBrains](https://blog.jetbrains.com/ai/2026/07/ponytail-skill-claude-tested/)).
- Recommendation for the class: add Week 27 "Software Factory: an org chart made of loops" as six labs plus a 48-hour capstone. Students first run `super-board` as-is, then rebuild its four lanes themselves inside the course plugin (`agentic-coding-fitness`), ending with a factory that drains a GitHub Project for a small FastAPI + React hotel-energy app, files tickets from a stub telemetry feed, and proves itself with `claude plugin eval`.
- For AltoTech, the same factory pattern applies to Alto Copilot and Alto Reef repos, with two adjustments: route anything touching tenant data, PMS control or billing to a human merge (super-board's `money`/`auth`/`schema` labels generalise to `tenant`/`control`/`pdpa`), and keep the factory's egress allow-listed as Anthropic does.

---

## 2. What the video shows

Eric Tech (ex-Amazon/Microsoft, builds bookzero.ai with Claude Code) describes a loop that "is actually mimicking the actual production software development teams where we have a builder and a QA and reviewer" and that "can self-improve by using telemetries like Sentry, PostHog and automatically formulates the ticket" ([YouTube transcript](https://www.youtube.com/watch?v=aggJvNZxfKA)). Chapter by chapter:

| Time | Topic | What is shown | Why it matters for the class |
|---|---|---|---|
| 0:00 | Intro | Telemetry → tickets → specialist agents → PRs; QA failure sends the card back to Build; every agent comment lands on the GitHub issue so "any agents or human here can actually look at the actual issue log" | The issue thread is the shared memory and audit log; no hidden local state |
| 2:25 | Overview | `super-collect` collects on a schedule and files cards; `super-board` drains the kanban one card at a time; cards needing a human go to Blocked; runs on Claude Code or Codex with swappable models | Separation of intake from execution; human gate as a column |
| 4:01 | Build | Prep skills first: ponytail's seven-rung ladder ("does this feature need to exist… can it be done in one line") and Context7 to read current library docs "before it writes a single line of code"; then Matt Pocock's `implement`, `diagnosing-bugs`, `codebase-design`; then verification, code review, humanizer | Prep → implement → self-check → handoff is a lane, not a prompt |
| 7:05 | QA | "Testing the bottom layer first": unit → component → integration → Playwright/Cypress e2e; the right testing library for the ticket | Test pyramid as an agent policy |
| 8:44 | Review | ponytail-review, code review, codebase design before merge | Reviewer is a separate agent from the builder |
| 8:59 | Self-improving | `super-collect` from Sentry/PostHog, or brainstormed features through wayfinder/grooming skills, verified before filing | Closing the loop from production back to backlog |
| 9:22 | UI loop | `ui-refine-loop`: diagnose → critique → audit → harden/clarify → styling/motion/polish → back to diagnose, built on Matt Pocock skills and impeccable | A bounded, repeated critique loop for "AI slop" UIs |
| 10:34 | Extras | `git-sync`, hooks that block secrets, auto-generated skill eval file, README sync | Guards are deterministic, not advisory |
| 11:38 | Setup | Install, `onboard`, `run`; fans out up to about three agents to drain the Ready column | Three steps to a running factory |

What the repository adds beyond the video (read directly from `main` on 10 Oct 2026):

- Eight skills in two families: five you type (`super-board`, `super-collect`, `ui-refine-loop`, `visual`, `git-sync`) and three the board runs as lanes (`super-build`, `super-qa`, `super-review`) ([super-board README](https://github.com/EricTechPro/super-board)).
- A model ladder: a cheap router (Haiku 4.5) grades each card easy/medium/hard and the run flag picks the model per grade; `--codex` swaps the whole ladder to GPT models ([super-board README](https://github.com/EricTechPro/super-board)).
- A ticket format with Problem / Context / Fix / Acceptance Criteria / Risk / Blocked-by, 2–5 checkable ACs ("NEVER 'works well', 'is fast'"), one card = one PR under 400 changed lines, and labels `money`, `auth`, `schema` that route the merge to a human ([ticket-format.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-board/references/ticket-format.md)).
- Builder rules: worktree per issue under `.claude/worktrees/`, one branch `issue-<N>-<slug>`, read current vendor docs before touching any third-party surface and cite them as `Docs:` bullets in the PR, stop and propose a split when the diff passes the cap, never merge ([super-build SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-build/SKILL.md)).
- QA rules: "A 200 response with a blank body is a bug, not a green test"; every e2e spec enforces no console errors, no page errors, no 5xx, no 401/403 on auth pages, and a page-type non-blank guard; a test-gap ledger maps every AC to the asserting `file:line` at unit, component and e2e rungs, writes High gaps red-first, and bounces to Build when app code must change ([super-qa SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-qa/SKILL.md)).
- Review rules: "Review remembers" (the reviewer edits one `<!-- super-review:report -->` comment in place and re-checks every prior finding against code, not against the builder's claim); an adversarial truth-check spawns a Code-grounder and a Historian, takes the minimum confidence, and blocks below 70; over-engineering is always "Should fix", never a blocker on its own ([super-review SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-review/SKILL.md)).
- Collect rules: one fresh verifier per candidate checks Real → Still happening → Already fixed → Duplicate and returns `file | drop-fixed | drop-stale | drop-noise | duplicate | needs-triage`; "unclear is not a drop" ([super-collect SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-collect/SKILL.md)).
- A decision policy for unattended workers: never ask the user; ambiguity is caught upstream by `super-board lint`, which routes vague issues through a grilling skill while a human is present ([decision-policy.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-build/references/decision-policy.md)).
- Three behavioural evals run with `claude plugin eval` against stubbed `gh`/`git`: `review-remembers`, `ponytail-overengineering`, `test-gap`, each with an LLM-judge grader plus deterministic graders such as "no `pr merge` in the gh log" ([evals/README.md](https://github.com/EricTechPro/super-board/blob/main/evals/README.md)).

---

## 3. The 2026 software-factory landscape

### 3a. Definitions that agree on the shape

| Source | Definition / stages | Human role |
|---|---|---|
| StrongDM, via Simon Willison (Feb 2026) | "Non-interactive development where specs + scenarios drive agents that write code, run harnesses, and converge without human review"; rules "code must not be written by humans" and "must not be reviewed by humans"; a team spending under $1,000 in tokens per engineer per day "has room for improvement" | Write specs and scenarios; the open question Simon raises is how to trust software when both implementation and tests are machine-made ([Simon Willison](https://simonwillison.net/2026/Feb/7/software-factory/)) |
| O'Reilly Radar, Paul Iusztin (Sep 2026) | Eight stages in three groups: what to build (triage/intake, brainstorming, planning with ADRs and glossary), building and checking (implementing, review, review-CI, release), self-improving (monitor/incident response → triage); a shared context layer (LLM wiki) across stages | Central in brainstorming and planning, return for the final merge; builder and QA agents must be separate because agents are "way too nice grading [their] own homework" ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)) |
| Factory.ai 2.0 (Jun 2026) | "An interconnected, agent-native, end-to-end system… that improves over time by observing itself"; signals → triage → build/test/review/secure/ship/monitor → signals; Droids, Automations, Droid Computers, Missions, Router | "Engineers build the system, and it builds software" ([Factory](https://factory.com/news/software-factory)) |
| Builder.io (Sep 2026) | Start with one reproducible bug; make the product agent-testable (preview deploy, test DB, test user, logs); dry-run intake; worktree per fix; evidence of broken vs fixed; make the PR easy to review; repeat; feed product feedback back | A person approves and merges; measure "five PRs that require five lengthy rescue sessions" vs five easy reviews ([Builder.io](https://www.builder.io/blog/build-an-agentic-software-factory-starting-with-one-bug)) |
| DZone (Aug 2026) | Requirements → planning → build → test/CI (failures stop the workflow) → human review → CD → monitoring with incidents and rollback → feedback | Human approval before risky steps and before release ([DZone](https://dzone.com/articles/ai-software-factory)) |
| Microsoft Research, "Towards an AI Software Factory for Data Systems" (2026) | Automate the whole SDLC: targeting (what to build), coding, reviewing (including flighting), ops (detect/triage/fix) | 26 authors from Microsoft, GitHub and UW–Madison; positions the factory as an SDLC system rather than a coding tool ([arXiv](https://arxiv.org/html/2609.36323)) |
| Dagger, Solomon Hykes | A software factory is the system a team uses to build, test, integrate and deliver; each team's is specific; agents are "a new software architecture, not a new software market"; CI becomes the bottleneck when agent output outruns "dumb" shell-on-VM pipelines | Platform engineers provide the factory as a platform ([Heavybit podcast](https://www.heavybit.com/library/podcasts/open-source-ready/ep-17-ai-native-software-factories-with-solomon-hykes), [Dagger blog](https://dagger.io/blog/the-great-ci-bottleneck-of-2026/)) |

### 3b. Platform primitives students will use

- Claude Code: `CLAUDE.md` for persistent project instructions, plan mode, verification with evidence, hooks for things that must happen every time, subagents for fresh-context review, worktrees for parallel sessions, `/batch` across 5–30 subagents, headless `-p` mode ([Claude Code best practices](https://code.claude.com/docs/en/best-practices)). Agent teams (experimental, `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`) add a lead, teammates with their own context windows, a shared task list and a mailbox ([agent teams docs](https://code.claude.com/docs/en/agent-teams)). Hooks fire on `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStop`, `TaskCompleted`, `WorktreeCreate` and more; exit code 2 on `PreToolUse` blocks the tool call with stderr as the reason ([hooks docs](https://code.claude.com/docs/en/hooks)). `claude plugin eval` runs each case three times with and without the plugin and reports the delta ([plugin evals docs](https://code.claude.com/docs/en/plugin-evals)).
- GitHub Copilot cloud agent (public preview): assign an issue to Copilot, optionally choose branch, agent, model and reasoning level; it opens a PR and requests review; it does not read issue comments added after assignment ([GitHub Docs](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/use-cloud-agent-on-github)).
- Sentry: Seer for root cause and draft PRs, the Sentry MCP for giving your own agent stack traces and tags, the CLI for scripts ([Sentry blog](https://blog.sentry.io/seer-mcp-cli-or-coding-agent/)). PostHog: 63 million MCP tool calls from 130,000 people in 90 days; Claude Code 32 % and Codex 22 % of third-party calls; SQL and schema reads are 32 % of traffic ([PostHog](https://posthog.com/blog/how-ai-agents-behave)).
- Skill packs: Matt Pocock's engineering skills (`to-spec`, `to-tickets`, `implement`, `tdd`, `diagnosing-bugs`, `code-review`, `triage`, `grill-with-docs`) ([mattpocock/skills](https://github.com/mattpocock/skills)); ponytail's seven-rung minimal-code ladder, which only produced measurable results when injected by a `SessionStart` hook rather than left for the model to choose ([JetBrains](https://blog.jetbrains.com/ai/2026/07/ponytail-skill-claude-tested/)).

### 3c. Evidence and cautions to teach alongside

- Anthropic: 8× code per engineer per quarter vs 2021–2025; about 80 % of merged code authored by Claude; multiple narrow-scope review agents plus SAST; new AI reviewers run in shadow mode until trusted; every automated approval logged and a risk-weighted sample reviewed by humans; an incident agent with exactly three permissions once tried to enlist a code-writing Claude and was stopped by a human gate ([Anthropic](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle)).
- DORA 2026.01: AI amplifies existing strengths and weaknesses; a J-curve dip from learning curve, "verification tax" and downstream process adaptation is "the tuition cost of transformation" ([InfoQ](https://www.infoq.com/news/2026/05/dora-roi-ai-assisted-dev-report/)).
- METR (Feb 2026): the second RCT (57 developers, 143 repos, 800+ tasks) was judged an unreliable signal because developers withheld tasks they did not want to do without AI; METR is moving to other designs ([METR](https://metr.org/blog/2026-02-24-uplift-update/)).
- Anthropic's 2026 Agentic Coding Trends report predicts cycles compressing from weeks to hours, orchestrators coordinating specialised agents, and agents running for days or weeks — predictions, not measurements ([Anthropic report](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)).
- O'Reilly's build-vs-buy rule: "The smallest builds, the middle buys, and the largest builds again"; don't overbuild for autonomy — the author's first factory with large remote workflows became hard to stop or redirect ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).

---

## 4. Twelve design principles for a teachable factory

1. The loop is the atom. Teach one lane end-to-end before composing lanes ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).
2. The ticket is the contract. Checkable acceptance criteria, risk, blocked-by; vague tickets are fixed upstream by a human, never guessed by a worker ([ticket-format.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-board/references/ticket-format.md), [decision-policy.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-build/references/decision-policy.md)).
3. State lives in the tracker, not in local files. Issue comments and PR threads are the inter-lane protocol ([super-build SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-build/SKILL.md)).
4. One card, one worktree, one branch, one PR under a size cap.
5. Builder, tester and reviewer are different agents with different context ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).
6. Evidence or it did not happen: screenshots, test output, HAR, Sentry probe; "a 200 with a blank body is a bug" ([super-qa SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-qa/SKILL.md)).
7. Test the tests: a new test that passes first time has not been seen to fail ([super-qa SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-qa/SKILL.md)).
8. Review remembers, and reviewers read code, not claims ([super-review SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-review/SKILL.md)).
9. Guards are hooks, not prose: secrets, worktree path, protected branches, deletes outside the repo ([hooks/README.md](https://github.com/EricTechPro/super-board/blob/main/hooks/README.md), [hooks docs](https://code.claude.com/docs/en/hooks)).
10. Humans sit at columns, not in loops: Blocked for `money`/`auth`/`schema`, merge for big PRs, release gate ([super-board README](https://github.com/EricTechPro/super-board)).
11. Route models by card difficulty; total cost is tokens × price, not tier ([super-board README](https://github.com/EricTechPro/super-board), [O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).
12. The factory evals itself: behavioural cases with deterministic plus LLM graders; new reviewers in shadow mode ([plugin evals docs](https://code.claude.com/docs/en/plugin-evals), [Anthropic](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle)).

---

## 5. Week 27 module design

Position: after Week 26 (NemoClaw on DGX Spark) as Phase ⑦ "The factory: loops that ship". It reuses Week 5 (agent loop), Week 7 (MCP & skills), Week 10 (observability), Week 18 (production loop) and the plugin's existing `hooks/`, `skills/`, `evals/`, `tests/` directories.

Target app for the labs: `week27/00_alto_mini/` ("Alto Mini"), a deliberately small FastAPI app with one static page (the React/vitest layer from the first draft was dropped to keep the target Python-only): rooms, HVAC setpoints, daily kWh, one alerts page, with four seeded, ticketable gaps (see its README). It has (a) a one-command dev server, (b) a seed script with representative data, (c) a test user, (d) pytest + one optional Playwright spec, (e) a stub telemetry feed `telemetry/sentry.json` and `telemetry/posthog.json`. Builder.io's checklist is the acceptance test for the app itself: clean checkout → running behaviour, test DB with representative records, agent can sign in and see browser and server logs ([Builder.io](https://www.builder.io/blog/build-an-agentic-software-factory-starting-with-one-bug)).

| Lab | Title | Outcome | Time |
|---|---|---|---|
| 0 | Run someone else's factory | `super-board` installed on the target app; one ticket drained Build → QA → Review → merged with proof | 90 min |
| 1 | The ticket and the builder lane | Students write 5 tickets in the standard format, a `CLAUDE.md`, a `fitness-build` skill, a worktree-per-issue runner; one draft PR per ticket | 3 h |
| 2 | The QA lane | `fitness-qa` skill: test-pyramid routing, test-gap ledger, non-blank guards, forensics folder, bounce-to-Build | 3 h |
| 3 | The review lane and merge gate | `fitness-review` skill: review-remembers comment, adversarial Code-grounder/Historian, merge gate script with policy labels | 3 h |
| 4 | Guards and evals | Four `PreToolUse` hooks; three `claude plugin eval` cases with graders; run with/without and read the delta | 2 h |
| 5 | Self-improvement: telemetry to tickets | `fitness-collect` skill: stub Sentry/PostHog → verifier → filed cards; `fitness-lookback` updates `CLAUDE.md` from recurring findings | 2 h |
| Capstone | 48-hour factory run | Factory drains ≥ 8 cards on a team project; report cost per merged PR, bounce rate, rescue sessions, eval delta | 2 days |

Grading rubric (100 points): tickets pass `lint` (10); each lane has a SKILL.md with role boundary, algorithm, stop conditions (30); hooks block the four red-team attempts (15); three evals pass with positive delta (15); capstone metrics reported honestly, including failures (20); write-up names one thing the factory should not be allowed to do yet (10).

---

## 6. Lab 0 — Run someone else's factory (90 min)

Goal: see the whole loop before building any part of it.

1. Fork the target app, create a GitHub Project (board) with columns Backlog, Ready, Building, QA, Review, Blocked, Done. (`super-board onboard` can create it.)
2. Install: `curl -fsSL https://raw.githubusercontent.com/EricTechPro/super-board/main/get.sh | bash`, or as a plugin via `/plugin marketplace add EricTechPro/super-board` then `/plugin install super-board@super-board`; needs Claude Code, `gh`, `jq`, bash 3.2+, Python 3.9+ ([super-board README](https://github.com/EricTechPro/super-board)).
3. `/super-board onboard` — eight steps (Checks, GitHub, Board, Branch, AGENTS.md, Policies, Bug sources, Review). Accept defaults; turn on `guard-protected-push`.
4. Write one ticket in the standard format (Section 7.1) — pick a reproducible bug in the seed app (e.g., "Alerts page shows '—' for kWh when yesterday has no readings").
5. `/super-board lint` then move the card to Ready and `/super-board run hotel-energy-app`.
6. Observe and record, in a shared doc: every comment the lanes posted on the issue and PR; what the QA lane screenshotted; whether Review bounced; what the merge gate ran; total tokens/cost from `/super-board status` or the usage script.
7. Discussion prompts: Where was the human gate? What would have happened if the ticket had been vague? Which lane would you trust least, and why?

---

## 7. Lab 1 — The ticket and the builder lane

### 7.1 Ticket template (copy into `docs/agents/issue-tracker.md`)

```markdown
## Problem
<1–3 sentences from the user's side. No implementation plan.>

## Context
- **Where:** <page>, `<route>`
  - a. <step>
  - b. <step>
- **Who:** <who sees it>

## Fix
<intended change in 1–3 lines>

## Acceptance Criteria
- [ ] <checkable outcome a test can assert>
- [ ] <checkable outcome>

## Risk
🟢 Low · <one line>

## Blocked by
- None.
```

Rules to enforce in `lint`: 2–5 ACs, each checkable (never "works well"); title `<emoji> [<kind>] <scope>: <what>` ≤ 70 chars; labels `money`, `auth`, `schema` route the merge to a human; one card = one PR under 400 changed lines ([ticket-format.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-board/references/ticket-format.md)).

### 7.2 `CLAUDE.md` for the target app (keep it short)

```markdown
# Alto Mini — agent instructions
- Dev: `make dev` (FastAPI :8000, Vite :5173). Seed: `make seed`. Test user: qa@alto.local / qa-pass.
- Tests: `make test` (pytest + vitest), `make e2e` (Playwright, needs `make dev`).
- One issue = one worktree under `.claude/worktrees/issue-<N>-<lane>/`, one branch `issue-<N>-<slug>`.
- Never commit `.env*`. Never push to `main`. Reviewer merges; builder never merges.
- Before touching a third-party API/SDK, read its current docs and cite them as `Docs:` bullets in the PR.
- Setpoint, billing and tenant-scoping changes carry labels `control`, `money`, `tenant` → human merge.
```

This follows the best-practice guidance that `CLAUDE.md` holds only what Claude cannot infer: commands, conventions, environment quirks ([Claude Code best practices](https://code.claude.com/docs/en/best-practices)).

### 7.3 `skills/fitness-build/SKILL.md` (skeleton)

```markdown
---
name: fitness-build
description: Builder lane. Given one GitHub issue, create a worktree and branch, implement the smallest change that satisfies the ACs with tests, open a draft PR with evidence, hand off to QA. Never merges. Never asks the user.
---
# Role boundary
You are the Builder. Input: issue number. Output: a draft PR and a `[builder] [report]` comment.
Read context ONLY from the issue body, all issue comments, and the linked PR. Never from local files.

# Algorithm
0. Pre-flight: `gh issue view <N> --json title,body,labels,comments`. If any AC is uncheckable → comment `[builder] ❓ lint` and stop.
1. Worktree: `git worktree add .claude/worktrees/issue-<N>-build -b issue-<N>-<slug> origin/main`.
2. Docs check: third-party surface, upgrade, auth or billing → read current vendor docs first; record each as a `Docs:` bullet.
3. Minimal-code ladder before writing: needed at all? already in repo? stdlib? platform? installed dep? one line? only then write.
4. Implement test-first per AC (pytest/vitest). Run `make test`. Iterate until green.
5. Size check: `git diff --shortstat origin/main...HEAD`; if > 400 lines, stop, push, propose a split in a PR comment, move card to Blocked ❓.
6. Commit per logical change, push, `gh pr create --draft --title "<commit subject>" --body-file pr.md`.
7. Post `[builder] [report]`: ACs → test file:line, commands run with output tail, Docs bullets, what you could not verify.
8. Move card Building → QA (`gh project item-edit`).

# Stop conditions
Tests red after 3 attempts · size cap · missing env/secret · ambiguous AC → `[builder] ❌` comment with `root-cause-hash:` and move to Blocked.
```

### 7.4 Headless runner `scripts/factory-run.sh`

```bash
#!/usr/bin/env bash
# Drain the Ready column: one headless Claude Code session per card, max 3 in parallel.
set -euo pipefail
PROJECT_NUM=${PROJECT_NUM:?}; OWNER=${OWNER:?}; LANE=${1:-build}; MAX=${MAX_WORKERS:-3}
MODEL=${MODEL:-claude-sonnet-5}
ready=$(gh project item-list "$PROJECT_NUM" --owner "$OWNER" --format json \
  | jq -r '.items[] | select(.status=="Ready") | .content.number')
echo "$ready" | head -n "$MAX" | while read -r N; do
  [ -z "$N" ] && continue
  ( claude -p "Use the fitness-$LANE skill on issue #$N. Do not ask questions." \
      --model "$MODEL" --max-turns 60 --output-format json \
      > ".claude/factory/logs/issue-$N-$LANE.json" 2>&1 ) &
done
wait
```

Headless `-p` mode is the documented way to run Claude Code non-interactively from scripts and CI ([Claude Code best practices](https://code.claude.com/docs/en/best-practices)). Verify flags against the installed CLI version before class (`claude --help`).

### 7.5 Exercise

Write five tickets for the seed app (two bugs, two small features, one refactor), run `lint`, run the builder on the two bugs, and compare the two PRs' evidence sections. Deliverable: a table AC → test `file:line` for each PR.

---

## 8. Lab 2 — The QA lane

### 8.1 `skills/fitness-qa/SKILL.md` (core sections)

```markdown
---
name: fitness-qa
description: Tester lane. Verifies a finished PR against its ACs with the right test rung, fills the test-gap ledger, captures forensics, passes to Review or bounces to Build. Never asks the user.
---
# What counts as green
A 200 with a blank body is a bug. Every e2e run asserts: no console errors, no uncaught page errors, no 5xx, no 401/403 on auth pages, and a page-type non-blank guard (list ≥ 1 row or empty-state; dashboard ≥ 1 widget with data).

# Test-pyramid routing
Pure function → unit (pytest/vitest). Component/hook → component test. Route/API → integration. User journey → Playwright, last.

# Test-gap ledger (scope = the diff)
| AC | unit | component | e2e | gap? |
Every AC → asserting file:line at each rung, or `none`, or `n/a: <reason>`.
Edge classes: boundaries (limit−1/limit/limit+1; 0 vs null vs empty), money/dates (rounding, timezone, month end),
input (unicode, very long), volume (0/1/many), auth (logged out, another user's id), error paths, idempotency, UI states.
Test the tests: a test that asserts on its own mock or recomputes the expected value with the code under test is `none`.
Rank: High = AC with no test, or any auth/money/data-loss gap. Medium = partly covered branch. Low = cosmetic.
Act: write every High gap red-first; break the guarded line, see red, restore, see green. If app code must change → Fail → bounce to Build (QA → Ready) with witness, expected, rung, target file.

# Forensics
`docs/qa/report/<pr>/` : screenshots per step, console.log, pageerrors.log, network.har, telemetry probe.

# Handoff
`[tester] ✅` or `[tester] ❌` comment on the PR with the ledger and forensics paths; move card QA → Review or QA → Ready.
```

Source for the rules: super-qa's green definition and test-gap check ([super-qa SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-qa/SKILL.md)); separation of builder and tester per O'Reilly ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).

### 8.2 Playwright fixture sketch (`e2e/lib/report-fixture.ts`)

```ts
import { test as base, expect } from '@playwright/test';
export const test = base.extend<{ report: { step: (n: string) => Promise<void>; errors: string[] } }>({
  report: async ({ page }, use, testInfo) => {
    const errors: string[] = [];
    page.on('console', m => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });
    page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
    page.on('response', r => { if (r.status() >= 500) errors.push(`5xx: ${r.url()}`); });
    let i = 0;
    await use({ errors, step: async (name) => { await page.screenshot({ path: testInfo.outputPath(`${++i}-${name}.jpg`) }); } });
    expect(errors, 'forensics guard').toEqual([]);
  },
});
```

### 8.3 Exercise

Run `fitness-qa` on the two builder PRs from Lab 1. One of them should be seeded with a deliberate gap (an AC without a test). Deliverable: the ledger, the red-then-green proof of the written test, and a bounce comment if app code had to change.

---

## 9. Lab 3 — The review lane and merge gate

### 9.1 Review remembers

The reviewer looks up the newest PR comment containing `<!-- fitness-review:report -->` and edits it in place, so one table carries every round. Round 1 re-checks each prior finding against the code — "a resolved thread is not evidence, and neither is the builder's reply saying it was fixed" — marking `fixed` (cite file:line), `not fixed`, or `no longer applies`; any `not fixed` bounces again without a fresh pass ([super-review SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-review/SKILL.md)).

```bash
gh pr view "$PR" --json comments \
  --jq '[.comments[] | select(.body | contains("<!-- fitness-review:report -->"))] | last | {url, body} // {}'
# edit in place: gh api -X PATCH repos/<owner>/<repo>/issues/comments/<id> -F body=@report.md
```

### 9.2 Adversarial truth-check (for diffs ≥ 10 lines or labels in {security, migration, money, auth, control, tenant})

Spawn two fresh subagents with a fixed brief: a Code-grounder (verify every cited file:line exists and does what the PR claims) and a Historian (`git blame` the changed lines; look for ADRs, prior incidents, reverted attempts). Each reads ACs and diff first, then the builder's summary "as claims to check, not as the frame", and returns findings classed Gap / Bug / Verification miss / Scope drift / Over-engineering / No issue with a 0–100 confidence. Aggregate by minimum; below 70 the reviewer must not approve. Over-engineering is always "Should fix" and never a blocker on its own ([super-review SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-review/SKILL.md)). Fresh-context reviewers are the documented way to get an unbiased check in Claude Code ([Claude Code best practices](https://code.claude.com/docs/en/best-practices)).

### 9.3 Merge gate `scripts/merge-gate.sh`

```bash
#!/usr/bin/env bash
# Merge only if: reviewed SHA == current head, verify commands pass on base+branch, no human-only label, size under cap.
set -euo pipefail
PR=$1; EXPECT_HEAD=$2; CAP=${AUTO_MAX_LINES:-400}
head=$(gh pr view "$PR" --json headRefOid --jq .headRefOid)
[ "$head" = "$EXPECT_HEAD" ] || { echo "head moved since review"; exit 2; }
labels=$(gh pr view "$PR" --json labels --jq '[.labels[].name] | join(",")')
for l in money auth schema control tenant pdpa; do
  case ",$labels," in *",$l,"*) echo "label $l → human merge"; gh pr comment "$PR" -b "🙋 needs human merge: label $l"; exit 3;; esac
done
changed=$(gh pr view "$PR" --json additions,deletions --jq '.additions+.deletions')
[ "$changed" -le "$CAP" ] || { echo "PR too big ($changed > $CAP)"; exit 3; }
git fetch origin main && git merge --no-commit --no-ff "origin/$(gh pr view "$PR" --json headRefName --jq .headRefName)" >/dev/null
make test
gh pr merge "$PR" --squash --match-head-commit "$EXPECT_HEAD"
```

The pattern (merge pinned to the reviewed commit, verify on base plus branch, policy labels) mirrors super-board's merge gate ([super-board README](https://github.com/EricTechPro/super-board)). Anthropic's variant is to place the hard gate at test/CI and to sample automated approvals for human review ([Anthropic](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle)).

### 9.4 Exercise

Seed one PR that is correct, tested, and wraps a one-line formatter in a strategy class plus registry plus JSON config. The reviewer should report Over-engineering routed to Build and still call the PR merge-ready. Seed a second PR whose builder comment claims a fix that is not in the diff; the Code-grounder should catch it.

---

## 10. Lab 4 — Guards and evals

### 10.1 Hooks (`.claude/settings.json`)

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [
        { "type": "command", "command": "python3 .claude/hooks/guard-secrets.py" },
        { "type": "command", "command": "python3 .claude/hooks/guard-worktree-path.py" },
        { "type": "command", "command": "python3 .claude/hooks/guard-protected-push.py" },
        { "type": "command", "command": "python3 .claude/hooks/guard-delete-outside.py" } ] },
      { "matcher": "Write|Edit", "hooks": [
        { "type": "command", "command": "python3 .claude/hooks/guard-key-literals.py" } ] }
    ],
    "SessionStart": [ { "hooks": [ { "type": "command", "command": "python3 .claude/hooks/cleanup-worktrees.py" } ] } ]
  }
}
```

### 10.2 `guard-secrets.py` (stdlib only; exit 2 blocks)

```python
#!/usr/bin/env python3
import json, re, sys
d = json.load(sys.stdin)
cmd = (d.get("tool_input") or {}).get("command", "")
patterns = [r"\.env(\.|\b)", r"id_rsa|id_ed25519", r"\.aws/credentials", r"\.netrc", r"\.npmrc", r"\.pypirc"]
reads = r"\b(cat|less|more|head|tail|sed|awk|grep|rg|xxd|base64|curl|scp)\b"
if re.search(reads, cmd) and any(re.search(p, cmd) for p in patterns):
    print("blocked: reading or piping a credential file. Use `make env-check` to list key NAMES only.", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
```

Exit code 2 on `PreToolUse` blocks the call and feeds stderr back as the reason; exit 1 alone is not a policy block ([hooks docs](https://code.claude.com/docs/en/hooks)). The six guards in super-board (worktree path, secrets, key literals, delete-outside, protected push, README sync) are the reference set ([hooks/README.md](https://github.com/EricTechPro/super-board/blob/main/hooks/README.md)).

Red-team drill (15 points): students try, in a live session, to (1) `cat .env`, (2) `git worktree add ../elsewhere`, (3) `git push --force origin main`, (4) `rm -rf ~/`. All four must be blocked with a readable reason.

### 10.3 Evals (`plugins/agentic-coding-fitness/evals/`)

Case layout, from the official docs: each case is a directory with `prompt.md` and/or `case.yaml` plus at least one grader; `--scaffold` runs a `seed.sh` that builds a throwaway repo; default 3 runs per arm; the report shows WITH vs W/OUT and Δ ([plugin evals docs](https://code.claude.com/docs/en/plugin-evals)).

```yaml
# evals/test-gap/case.yaml
schema_version: "1.1"
name: test-gap
description: fitness-qa flags an AC with no test as High and writes it red-first or bounces.
tags: [qa, factory]
runs: 3
context:
  scaffold_script: seed.sh
execution:
  max_turns: 40
  allowed_tools: [Read, Glob, Grep, Skill]
graders:
  - name: ran-tests
    type: command
    command: "grep -q 'pytest\\|vitest' $EVAL_WORKSPACE/.cmd-log"
  - name: no-merge
    type: command
    command: "! grep -q 'pr merge' $EVAL_WORKSPACE/.gh-log"
  - name: verdict
    type: llm
    weight: 2
    criteria: "The agent produced a ledger mapping each AC to a test or 'none', ranked the untested AC High, and either wrote a test that was first seen failing or bounced to Build with witness/expected/rung/target. It did not merge."
```

```bash
PATH=$PWD/evals/_stubs/bin:$PATH claude plugin eval . --scaffold --allow-tools Bash Write Edit --runs 3 --max-cost-usd 5
```

Stub `gh` so evals run offline and log every call to `.gh-log`; super-board's `evals/_stubs/bin/gh` is a worked example and its three cases cost roughly $3 at three runs ([evals/README.md](https://github.com/EricTechPro/super-board/blob/main/evals/README.md)). Check the grader `type` names against the installed CLI's `claude plugin eval init` output before class; the field names above follow the docs page as fetched on 10 Oct 2026 but the feature is evolving.

Three required cases: `test-gap` (QA), `review-remembers` (Review loads the prior report and re-checks), `ponytail-overengineering` (Review flags over-engineering but still merges). Students add a fourth of their own.

---

## 11. Lab 5 — Self-improvement: telemetry to tickets

### 11.1 Stub telemetry

`telemetry/sentry.json`: 6 issues (3 real recurring, 1 already fixed by a merged PR, 1 noise, 1 duplicate of an open card). `telemetry/posthog.json`: 3 signals (a funnel drop on the alerts page, a rage-click on the setpoint slider, a stopped signal).

### 11.2 `skills/fitness-collect/SKILL.md` (core)

```markdown
# Sources
sentry: `python3 scripts/collect_sentry.py --since 14d` → candidates with fingerprint, count, last seen, stack top 3.
posthog: `python3 scripts/collect_posthog.py --since 14d` → candidates with signal key, trend, sample session.
# Verify — one fresh verifier per candidate (no debate)
1 Real (open the evidence) · 2 Still happening (seen after window start) · 3 Already fixed (merged PR matching symptom + fingerprint, and no events after release) · 4 Duplicate (same symptom + route as an open card).
Verdict: file | drop-fixed | drop-stale | drop-noise | duplicate | needs-triage. Unclear is NOT a drop → needs-triage label.
# File
Standard ticket format; Evidence table; `--label from-telemetry`; dry-run first.
# Report
`fitness-collect: filed n | dry-run n · since <date>` with every drop and its evidence.
```

Source: super-collect's verifier chain and verdict schema ([super-collect SKILL.md](https://github.com/EricTechPro/super-board/blob/main/skills/super-collect/SKILL.md)); the signal → investigate → propose → review → ship → observe loop in Arize's framing ([Arize](https://arize.com/blog/from-signal-to-pr/)).

### 11.3 `fitness-lookback`

Once a week: read the last 20 closed cards and review reports; cluster recurring findings (e.g., "timezone off-by-one in kWh rollups" three times); propose one line for `CLAUDE.md` or one new eval case; open a PR for a human. This is Anthropic's "when an agent discovers a bug class, the relevant instructions are updated" practice at class scale ([Anthropic](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle)) and Builder.io's "feed historical patterns back" step ([Builder.io](https://www.builder.io/blog/build-an-agentic-software-factory-starting-with-one-bug)).

### 11.4 Exercise

Dry-run collect; the expected verdicts are 3 file, 1 drop-fixed, 1 drop-noise, 1 duplicate; then file for real. Deliverable: the report with evidence links, plus one lookback PR.

---

## 12. Capstone — a 48-hour factory run

Teams of three run their factory on their own course project (Weeks 6, 18 or 26 apps qualify). Required instrumentation and report:

| Metric | How to measure | Why |
|---|---|---|
| Cards merged with proof | Done column with PR + QA forensics | The factory's output |
| Bounce rate | QA → Ready and Review → Ready moves per card | Lane quality |
| Rescue sessions | Any human intervention longer than 10 minutes, logged | Builder.io's "five rescues vs five reviews" test ([Builder.io](https://www.builder.io/blog/build-an-agentic-software-factory-starting-with-one-bug)) |
| Cost per merged PR | tokens × price from run logs | Total cost is tokens × price, not model tier ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)) |
| Eval delta | WITH − W/OUT for the four cases | Does the plugin actually help ([plugin evals docs](https://code.claude.com/docs/en/plugin-evals)) |
| Guard blocks | Count and sample from hook logs | Guards fire in practice |
| Human gates hit | Blocked cards by reason | Where the policy bites |

Write-up must include one incident the factory got wrong and what changed in a skill, hook or eval because of it. Teams that report zero problems get a follow-up interview, not full marks.

---

## 13. AltoTech application notes

- Repos: Alto Copilot, Alto Reef, the course plugin, and the AltoTech Claude plugins are natural first factories; start, as every source recommends, with one reproducible bug class (e.g., Reef DRY-mode labelling bugs) and a narrow intake.
- Telemetry sources: Sentry/PostHog if present; otherwise Alto Copilot's activity log and the TimescaleDB analytics MCP are the "signals" — a `collect_alto.py` that turns tool-call failures and degraded report runs into candidate cards fits the same verifier chain.
- Policy labels: extend `money`/`auth`/`schema` with `control` (anything that can change a setpoint or schedule), `tenant` (RLS, scoping) and `pdpa` (personal data, cross-border) → human merge, and keep Observe Mode before any control write, consistent with the Alto OS safety stance.
- Egress: run lanes on VMs with allow-listed egress and single-purpose identities, as Anthropic does; the factory's workers should not be able to reach arbitrary internet while processing untrusted issue text ([Anthropic](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle)).
- Sovereign tier: the lanes are model-agnostic by design (super-board already ladders Claude and Codex); on DGX Spark the same skills can run against a local model through the existing LiteLLM gateway, with the eval suite as the acceptance test for whether the local model is good enough per lane.
- Context graph link: the issue threads, review reports and lookback clusters are decision traces; filing them into the Alto Context Graph (previous brief) gives the factory institutional memory beyond one repo.

---

## 14. Risks and open questions to put in front of students

- Both code and tests written by agents: Simon Willison's question about how to trust the result has no settled answer; the class answer is independent lanes, test-the-tests, adversarial review and human sampling ([Simon Willison](https://simonwillison.net/2026/Feb/7/software-factory/)).
- Agents grading their own homework: separate builder/tester/reviewer identities, fresh context, minimum-confidence aggregation ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).
- Agent-to-agent escalation: Anthropic's incident agent tried to recruit a code-writing agent; boundaries must cover what agents can ask other agents to do ([Anthropic](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle)).
- Productivity claims: DORA's J-curve and METR's abandoned RCT mean students should measure their own rescue time and cost rather than quote vendor numbers ([InfoQ](https://www.infoq.com/news/2026/05/dora-roi-ai-assisted-dev-report/), [METR](https://metr.org/blog/2026-02-24-uplift-update/)).
- Skill activation: ponytail only moved the needle when injected by a hook; "installed" is not "used" ([JetBrains](https://blog.jetbrains.com/ai/2026/07/ponytail-skill-claude-tested/)).
- CI as bottleneck: agent output can outrun CI; keep verify commands fast and cache-friendly ([Dagger blog](https://dagger.io/blog/the-great-ci-bottleneck-of-2026/)).
- Overbuilding: start with granular commands per lane and one end-to-end command, keep the ability to halt and redirect ([O'Reilly Radar](https://www.oreilly.com/radar/inside-a-software-factory/)).

---

## 15. Verified vs. not

Verified on 10 Oct 2026 from primary sources: the video transcript and chapters; `super-board` README, SKILL.md files, references, hooks and evals on `main` (133 stars, MIT, pushed 8 Oct 2026); Claude Code docs for best practices, agent teams, hooks and plugin evals; GitHub Copilot cloud agent docs; Sentry, PostHog, Builder.io, O'Reilly, Factory, DZone, Dagger, Heavybit, Simon Willison, InfoQ/DORA, METR, JetBrains, Arize, arXiv and the Anthropic SDLC post.

Not verified: the skeleton skills, scripts and eval YAML in Sections 7–11 are written for this brief and have not been executed against a live Claude Code CLI; flag names for `claude -p` and grader `type` values should be checked against the installed version before class. The target app "Alto Mini" now lives in `week27/00_alto_mini/` (6 tests pass; Playwright spec optional, not run in CI). Cost figures quoted from super-board's evals README are the author's estimates. The Anthropic Trends report contains predictions, not measurements.
