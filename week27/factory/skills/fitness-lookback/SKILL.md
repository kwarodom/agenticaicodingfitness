---
name: fitness-lookback
description: Weekly self-improvement lane. Reads the last 20 closed cards, QA ledgers and review reports, clusters recurring findings, and proposes ONE change to CLAUDE.md, a skill, a hook or an eval as a PR for a human. Use weekly.
---
# Algorithm
1. `gh issue list --state closed --limit 20 --json number,title,labels,comments` and the matching PR review reports.
2. Cluster recurring findings (same class + same module/route; e.g. "timezone off-by-one in kWh rollups" ×3).
3. For the top cluster propose exactly one of: a line in CLAUDE.md; a step in a lane SKILL.md; a new guard hook; a new
   eval case under evals/. Explain the evidence (card numbers) in the PR body.
4. Open a PR labelled `lookback` for a human to merge. Never merge it yourself.
