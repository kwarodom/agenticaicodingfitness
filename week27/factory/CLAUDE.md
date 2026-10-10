# <App> — agent instructions (keep short; only what Claude cannot infer)
- Dev: `make dev`. Seed: `make seed`. Test user: qa@example.local / qa-pass.
- Tests: `make test` (unit + component), `make e2e` (Playwright; needs `make dev`).
- One issue = one worktree under `.claude/worktrees/issue-<N>-<lane>/`, one branch `issue-<N>-<slug>`.
- Never commit `.env*`. Never push to `main`. Reviewer merges; Builder never merges.
- Before touching a third-party API/SDK/upgrade/auth/billing: read current vendor docs, cite as `Docs:` bullets in the PR.
- Labels `money`, `auth`, `schema`, `control`, `tenant`, `pdpa` → human merge (see scripts/merge-gate.sh).
- Ticket format and lint rules: docs/agents/issue-tracker.md
