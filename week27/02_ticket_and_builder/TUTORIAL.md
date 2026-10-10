# ▶ Factory Lab 02 — The ticket and the Builder lane

> Part of Week 27. You write the contract (the ticket), the instructions (`CLAUDE.md`), and the first lane (`fitness-build`), then run it headless on the three seeded bugs in Alto Mini.

**What you'll actually do**
- Write five tickets in the standard format and lint them by hand against the rules.
- Install the course's `fitness-build` skill and a short `CLAUDE.md` into Alto Mini.
- Run one Builder per ticket with `scripts/factory-run.sh build` (headless `claude -p`, worktree per issue, draft PR).
- Compare evidence sections across PRs and find the one that over-reached.

**Time** ~3 h · **Difficulty** intermediate · **Needs** Lab 01 done; `gh project` access to your board

**Sources:** super-board [`super-build/SKILL.md`](https://github.com/EricTechPro/super-board/blob/main/skills/super-build/SKILL.md) and [`ticket-format.md`](https://github.com/EricTechPro/super-board/blob/main/skills/super-board/references/ticket-format.md) · [Claude Code best practices](https://code.claude.com/docs/en/best-practices) (CLAUDE.md, worktrees, headless mode) · JetBrains on the ponytail ladder ([test](https://blog.jetbrains.com/ai/2026/07/ponytail-skill-claude-tested/)) · brief §7

## 0 · Install the kit into Alto Mini

```bash
# on: laptop, repo root
cp -r week27/factory/skills/* .claude/skills/          # fitness-build, -qa, -review, -collect, -lookback
mkdir -p .claude/hooks && cp week27/factory/.claude/hooks/*.py .claude/hooks/
cp week27/factory/.claude/settings.json .claude/settings.json   # or merge the "hooks" block into yours
cp -r week27/factory/scripts week27/factory/docs week27/factory/telemetry week27/00_alto_mini/
cp week27/factory/CLAUDE.md week27/00_alto_mini/CLAUDE.md
```

Edit `week27/00_alto_mini/CLAUDE.md` so the commands are true for this app: `make dev` (:8127), `make seed`, `make test`, test user `qa@example.local`. Keep it under 15 lines — `CLAUDE.md` holds only what Claude cannot infer.

## 1 · Five tickets (45 min)

File these as issues (titles given; you write the bodies). Lint each against the table in `docs/agents/issue-tracker.md`.

| # | Title | Seeded gap | Labels |
|---|---|---|---|
| T1 | `🐛 [bug] energy: kwh_rollup crashes on a None reading` | `app/energy/rollup.py` (S-101) | — |
| T2 | `🐛 [bug] energy: daily history 500s when tz is omitted` | `app/energy/daily.py` (S-103) | — |
| T3 | `🐛 [bug] rooms: setpoint 31 returns 500 instead of 422` | `app/hvac/setpoint.py` + slider max (S-102) | `control` |
| T4 | `✨ [feat] alerts: acknowledge asks for confirmation and can be undone for 10 s` | PostHog funnel | — |
| T5 | `♻️ [refactor] api: share the yesterday-window query between alerts and history` | design | — |

Lint rules that usually fail on first try: an AC like "handles None gracefully" (not checkable → "returns 0.0 for `[]` and ignores `None`"); a `## Fix` that is an implementation plan; a missing `Blocked by`. T3 must carry `control`: in Alto Mini anything that changes a setpoint is a human-merge label (see `scripts/merge-gate.sh`).

## 2 · Read the Builder skill (20 min)

Open `.claude/skills/fitness-build/SKILL.md`. Map each step to what you saw super-board do in Lab 01:

| fitness-build step | super-board equivalent | Why it exists |
|---|---|---|
| 0 pre-flight lint | `super-board lint` | ambiguity is resolved where a human is present, never inside an unattended worker |
| 1 worktree + branch | worktree `.claude/worktrees/issue-N-build`, branch `issue-N-slug` | parallel Builders never collide; the guard hook blocks other paths |
| 2 docs check | "Docs before outside-tool code" | vendor docs beat memory for third-party surfaces |
| 3 minimal-code ladder | ponytail's seven rungs | less code → fewer tokens, fewer review findings |
| 4 test-first per AC | `mattpocock-skills:tdd` | the AC → test mapping is the evidence QA checks |
| 5 size cap 400 | `merge_policy.auto_max_lines` | big PRs are never auto-merged |
| 6–8 draft PR, report, move card | PR template + `[builder] [report]` | state lives in GitHub |

## 3 · Run the Builder headless (60 min)

```bash
# on: laptop, in week27/00_alto_mini (board number from `gh project list --owner <you>`)
export OWNER=<you> PROJECT_NUM=<n> MAX_WORKERS=2 MODEL=claude-sonnet-5
gh project item-edit ...   # or drag T1 and T2 to Ready in the UI
bash scripts/factory-run.sh build
tail -f .claude/factory/logs/issue-*-build.err
```

`factory-run.sh` lists Ready cards, starts one `claude -p "Use the fitness-build skill on issue #N…"` per card (max `MAX_WORKERS`), and writes JSON logs. Check `claude --help` once: flag names (`--max-turns`, `--output-format`) are verified against the docs as of 10 Oct 2026, not against your installed version.

**Expected output** (from the PR the Builder opens for T1)

```
## Evidence
- AC1 kwh_rollup([]) == 0.0           → tests/test_rollup.py:4  (red → green)
- AC2 None entries ignored            → tests/test_rollup.py:5
- AC3 2-dp rounding                   → tests/test_rollup.py:3 (existing)
- `make test` → 8 passed
Docs: none needed — no third-party surface
```

## 4 · Compare the PRs (30 min)

Put the two or three PRs side by side and answer:
1. Did any Builder touch files outside the ticket's scope? (T5 is the usual over-reacher.)
2. For T3, did the Builder clamp or reject? The ticket said reject with 422 — a Builder that "improved" it to clamping made a product decision it was not given.
3. Which PR's evidence section would let a reviewer approve without opening the code? Why?

## 5 · Exercise

`ex02_ticket_lint.py` (in `exercises/`): a 40-line linter that reads an issue body from stdin and prints PASS/FAIL per rule (2–5 ACs, each `- [ ]`, no banned words, `Risk` present, `Blocked by` present). Run it on your five tickets. Solution in `exercises/solutions/`.
