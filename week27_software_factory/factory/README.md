# Week 27 — Software Factory starter kit (Agentic Coding Fitness)

A minimal, readable factory you can install into any repo that already has `make dev`, `make test`, `make e2e`.
It mirrors the lanes of EricTechPro/super-board (Build → QA → Review, plus Collect and Lookback) in course-sized skills.

Status: skeleton written for the Week 27 labs. Shell/Python files are syntax-checked and the hooks are unit-tested
offline; nothing here has been run against a live Claude Code CLI yet. Check `claude --help` and
`claude plugin eval init` for current flag/grader names before class.

Layout
- `CLAUDE.md` — project instructions template (edit commands for your app)
- `docs/agents/issue-tracker.md` — ticket format + lint rules
- `skills/fitness-*/SKILL.md` — the five lanes
- `.claude/settings.json`, `.claude/hooks/*.py` — guard hooks (PreToolUse, exit 2 blocks)
- `scripts/factory-run.sh` — headless drain of the Ready column, N workers
- `scripts/merge-gate.sh` — merge only at the reviewed SHA, after verify, unless a human-only label
- `scripts/collect_sentry.py`, `scripts/collect_posthog.py` — read stub telemetry, emit candidates
- `telemetry/*.json` — stub feeds for Lab 06 (3 file, 1 fixed, 1 noise, 1 duplicate)
- `e2e/report-fixture.ts` — Playwright forensics fixture for the QA lane
- `evals/` — three `claude plugin eval` cases with graders and an offline `gh` stub
- `tests/test_hooks.py` — offline tests for the guards (`python3 -m pytest tests`)

Install into a repo (Lab 02 §0 does this for week27/00_alto_mini)
    cp -r week27/factory/skills/* <repo>/.claude/skills/ ; cp -r week27/factory/.claude/hooks <repo>/.claude/ ; merge .claude/settings.json
    cp -r week27/factory/{scripts,docs,evals,telemetry} <repo>/ ; edit CLAUDE.md
