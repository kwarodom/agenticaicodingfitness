# ▶ Factory Lab 04 — The Review lane and the merge gate

> Part of Week 27. The Reviewer owns the merge. It remembers prior rounds, reads code rather than claims, and on non-trivial diffs asks two fresh skeptics before it approves.

**What you'll actually do**
- Run `fitness-review` on a PR twice and watch "review remembers" edit one report in place.
- Trigger the adversarial truth-check (Code-grounder + Historian, minimum confidence) on a `control`-labelled PR.
- Seed two trap PRs: an over-engineered correct one, and one whose builder claims a fix that is not in the diff.
- Run `scripts/merge-gate.sh` and see it refuse a human-only label, a moved head, and a big PR.

**Time** ~3 h · **Difficulty** intermediate–advanced · **Needs** Lab 03 PRs

**Sources:** super-board [`super-review/SKILL.md`](https://github.com/EricTechPro/super-board/blob/main/skills/super-review/SKILL.md) ("Review remembers", "Adversarial mode") · Anthropic, [How Anthropic secures its AI-native SDLC](https://claude.com/blog/how-anthropic-secures-its-ai-native-software-development-lifecycle) (shadow mode, sampled approvals) · brief §9

## 1 · Review remembers (45 min)

Run `fitness-review` on T2's PR. It posts one comment beginning `<!-- fitness-review:report -->` with a findings table (`R1`, `R2`, …). Now have the Builder "fix" R1 only (rebuild lane), and run review again. Expected Round 1 behaviour:

```
<!-- fitness-review:report -->
| id | finding | file:line | status |
| R1 | default tz hard-coded instead of HOTEL_TZ env | app/energy/daily.py:6 | ✅ fixed (daily.py:6 reads os.environ) |
| R2 | history endpoint still 500s on malformed tz string | app/api/rooms.py:31 | ❌ not fixed — resolved thread is not evidence |
→ bounce to Build; fresh pass skipped (code about to change)
```

The lookup is one `gh pr view … --json comments --jq` call; the update is a `PATCH` on the same comment id. There is exactly one report per PR.

## 2 · Adversarial truth-check (45 min)

Run review on T3 (`control` label). The skill spawns two fresh subagents with the fixed brief: a **Code-grounder** (every cited `file:line` exists and does what the PR says) and a **Historian** (`git blame` the lines; ADRs, prior incidents, reverted attempts). Aggregate = **minimum** of the two confidences; below 70 the Reviewer must not approve.

Seed the trap first: edit the Builder's `[builder] [report]` comment on T3 to claim "added test for 15.5 boundary" when no such test exists. The Code-grounder should return a `Verification miss` and a confidence well under 70.

**Expected output**

```
truth-check: code-grounder 42/100 (Verification miss: claimed tests/test_setpoint.py::test_boundary_low not found) · historian 88/100
aggregate 42 < 70 → not approved · card → Blocked 🛡
```

## 3 · Over-engineering is never a blocker alone (30 min)

Seed PR T6: a correct, tested two-decimal kWh formatter wrapped in a strategy class, a registry, a factory and a JSON config file (the fixture in `../factory/evals/ponytail-overengineering/seed.sh` is exactly this). The expected review: one `Over-engineering · Should fix` finding routed to Build as a follow-up, **and** merge-ready. A reviewer that blocks here is wrong by policy.

## 4 · The merge gate (45 min)

```bash
# on: laptop, in week27/00_alto_mini
bash scripts/merge-gate.sh <PR> <reviewed-sha>
```

Try the four refusals and record each exit code and comment:

| Attempt | Expected |
|---|---|
| PR with label `control` (T3) | exit 3 · comment `🙋 needs human merge: label control` |
| Push one more commit after review, then run with the old SHA | exit 2 · `head moved since review` |
| A PR with > 400 changed lines | exit 3 · `🙋 big PR … please review` |
| Break a test on the branch | exit 4 · `verify failed on base+branch` |

Then merge T1 for real: the gate merges with `--match-head-commit <sha>`, so a commit that lands between review and merge can never ride in.

> 📌 Anthropic places its hard gate at test/CI rather than in a PreToolUse hook, runs new AI reviewers in **shadow mode** until trusted, and reviews a risk-weighted sample of automated approvals. Decide as a class where your gate lives and write it down in `docs/agents/issue-tracker.md`.

## 5 · Exercise

`ex04_min_confidence.py`: implement the aggregation rule (minimum, over-engineering excluded from confidence, reclass Blocker→Should fix for over-engineering) over a list of sub-agent findings; include a test where one skeptic at 55 blocks despite the other at 95. Solution provided.
