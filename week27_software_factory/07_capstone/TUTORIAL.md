# ▶ Factory Lab 07 — Capstone: a 48-hour factory run

> Part of Week 27. Teams of three run their factory on a real course project for two days and report what it shipped, what it cost, and what it got wrong.

**What you'll actually do**
- Pick a project (your Week 6, 18 or 26 app, or Alto Mini extended) and make it agent-testable.
- Write ≥ 8 tickets, lint them, and let the factory drain them with the lanes you built in Labs 02–06.
- Instrument the run: cost per merged PR, bounce rate, rescue sessions, guard blocks, human gates hit, eval delta.
- Write up one incident the factory got wrong and the skill/hook/eval change you made because of it.

**Time** 2 days wall-clock, ~6 h human time · **Difficulty** advanced · **Needs** Labs 01–06

**Sources:** Builder.io's "five rescues vs five reviews" test · O'Reilly's "total cost is tokens × price" and "don't overbuild for autonomy" · Anthropic's sampled approvals and single-purpose agent identities · DORA's J-curve · brief §12

## 1 · Make the project agent-testable (2 h)

Builder.io's checklist is the gate: a clean checkout reaches the behaviour under test with documented commands; a seed with representative records; a test user; the agent can see browser and server logs. Write it into `CLAUDE.md`. If your project fails this checklist, fix that first — "the app builds" is not a verification.

## 2 · The run (48 h)

- Columns and labels as in `docs/agents/issue-tracker.md`; human-merge labels `money`, `auth`, `schema`, `control`, `tenant`, `pdpa`.
- Run lanes on a schedule (`cron` or a Claude Code scheduled task): `factory-run.sh build` every 30 min, `qa` and `review` every 30 min offset, `fitness-collect` twice a day.
- Keep a **rescue log**: every human intervention over 10 minutes, with the card number and what you did.
- Keep a **guard log**: every hook block, with the command that was attempted.

## 3 · The report (required fields)

| Metric | Value | Evidence |
|---|---|---|
| Cards merged with proof | n | Done column; each PR's QA forensics path |
| Bounce rate | QA→Ready + Review→Ready moves ÷ cards | board history |
| Rescue sessions | n (minutes) | rescue log |
| Cost per merged PR | USD | run logs: tokens × list price |
| Eval delta | WITH − W/OUT per case | `evals/results/<ts>/aggregate-result.json` |
| Guard blocks | n, with 3 examples | guard log |
| Human gates hit | by label / reason | Blocked column |

Plus a one-page narrative: the incident the factory got wrong (a merged PR that broke something, a Tester that passed a blank page, a Reviewer that trusted a claim), the root cause, and the change you made to a skill, hook or eval. Then one paragraph on what the factory should **not** be allowed to do yet in your project and why.

## 4 · Grading (100)

Tickets pass lint (10) · each lane has a SKILL.md with role boundary, algorithm and stop conditions (30) · the four red-team attempts are blocked (15) · three evals pass with positive delta and a fourth case exists (15) · metrics reported honestly, including failures (20) · the "should not do yet" paragraph is specific (10). Teams that report zero problems get a follow-up interview, not full marks.

## 5 · Stretch goals

- Run the same lanes on a local model through your Week 25 LiteLLM gateway (`MODEL=` in `factory-run.sh`) and compare eval deltas per lane — the eval suite is the acceptance test for "is the local model good enough for this lane?".
- Replace the stub telemetry with a live Sentry or PostHog MCP.
- Add real auth to Alto Mini and a `tenant` label flow, then show the merge gate refusing to auto-merge it.
