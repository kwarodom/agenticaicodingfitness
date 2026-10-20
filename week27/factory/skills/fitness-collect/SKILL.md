---
name: fitness-collect
description: Intake lane of the Week 27 software factory. Reads telemetry (Sentry/PostHog stubs or live), verifies each candidate with one fresh verifier, and files standard-format Backlog cards. Supports --dry-run. Use on a schedule or on request.
---
# Sources
`python3 scripts/collect_sentry.py --since 14d` and `python3 scripts/collect_posthog.py --since 14d` print candidate JSON
(fingerprint/key, count, first/last seen, top frames or sample session, suggested route).

# Verify — one fresh verifier subagent per candidate, no debate
Give it ONE candidate, the board snapshot (`gh project item-list ... --format json`) and the window. It checks in order:
1. Real — open the cited event/replay/code; the evidence holds.
2. Still happening — seen after the window start.
3. Already fixed — a merged PR matches symptom + fingerprint AND no events after its merge.
4. Duplicate — same symptom + same route/module as an open card or issue.
Verdict: `file | drop-fixed | drop-stale | drop-noise | duplicate | needs-triage` + evidence links + note.
Unclear is NOT a drop: `needs-triage` files the card with that label.

# File
Standard ticket format (docs/agents/issue-tracker.md) with an Evidence table; label `from-telemetry`; `duplicate` →
comment on the existing card instead. `--dry-run` prints what would be filed.

# Report
`fitness-collect: filed n | dry-run n · since <date>` listing every candidate, verdict and evidence.
