# ▶ Factory Lab 06 — Telemetry to tickets (the self-improving loop)

> Part of Week 27. A factory that only drains tickets humans wrote is a fast intern. One that files its own verified tickets from production signals, and edits its own instructions from what it learns, is the "self-improving" part of the title.

**What you'll actually do**
- Run the collectors over the stub Sentry and PostHog feeds and read their advisory hints.
- Run `fitness-collect --dry-run`: one fresh verifier per candidate, verdicts `file | drop-fixed | drop-stale | drop-noise | duplicate | needs-triage`.
- File the three real candidates as standard tickets with Evidence tables; comment on the duplicate.
- Run `fitness-lookback` to propose one change to `CLAUDE.md` or an eval from recurring findings.

**Time** ~2 h · **Difficulty** intermediate · **Needs** Labs 02–04 cards on the board

**Sources:** super-board [`super-collect/SKILL.md`](https://github.com/EricTechPro/super-board/blob/main/skills/super-collect/SKILL.md) · Sentry, [Seer, the Sentry MCP and CLI, or your coding agent](https://blog.sentry.io/seer-mcp-cli-or-coding-agent/) · PostHog, [How AI agents behave: 63M MCP tool calls](https://posthog.com/blog/how-ai-agents-behave) · Arize, [From Signal to PR](https://arize.com/blog/from-signal-to-pr/) · Builder.io, [start with one bug](https://www.builder.io/blog/build-an-agentic-software-factory-starting-with-one-bug) (step 8) · brief §11

## 1 · The feeds (15 min)

`week27/factory/telemetry/sentry.json` has six issues; `posthog.json` has three signals; `board_snapshot.json` is the kanban as the verifier sees it. Run the collectors:

```bash
# on: laptop
cd week27/factory/scripts
python3 collect_sentry.py --since 14d | jq -r '.candidates[] | "\(.id)  \(.hint)  \(.title)"'
python3 collect_posthog.py --since 14d | jq -r '.candidates[] | "\(.key)  \(.hint)"'
```

**Expected output**

```
S-101  file            TypeError: unsupported operand None in kwh_rollup
S-102  file            ValueError: setpoint 31.0 outside 16-30
S-103  file            KeyError: 'tz' in daily_kwh
S-104  drop-fixed      AttributeError: 'NoneType' has no attribute 'name'
S-105  drop-noise      ChunkLoadError: Loading chunk 7 failed (stale deploy)
S-106  duplicate:S-101 TypeError in kwh_rollup via /api/rooms/{id}/history
funnel_drop_alerts_ack      file
rage_click_setpoint_slider  file
export_csv_errors           drop-fixed
```

The `hint` is advisory. The skill still sends **each candidate to one fresh verifier** that checks, in order: Real (open the evidence) → Still happening (seen after the window start) → Already fixed (merged PR matches symptom and fingerprint, and no events after it) → Duplicate (same symptom and route as an open card). "Unclear is not a drop": it files with `needs-triage`.

## 2 · Dry run, then file (45 min)

In Claude Code, inside Alto Mini: `Use the fitness-collect skill with --dry-run --since 14d`. Compare the verifier verdicts to the hints; a verifier that disagrees with a hint must say why with evidence. Then run without `--dry-run`. Expected: three new Backlog cards (S-101, S-102, S-103 — notice S-101 is the same bug as Lab 02's T1; if T1 is still open the verifier should return `duplicate:#<T1>` and comment there instead), one comment on the open rollup card for S-106, and a report:

```
fitness-collect: filed 3 | dry-run 0 · since 2026-09-26 (14d)
  S-101 duplicate → commented on #<T1>   S-104 drop-fixed (PR #41, 0 events after 2026-09-26)
  S-105 drop-noise (3 events in 2 min around a deploy)   S-106 duplicate → S-101
  funnel_drop_alerts_ack file → #<new>   rage_click_setpoint_slider file → #<new>   export_csv_errors drop-fixed (PR #44)
```

Each filed ticket carries an **Evidence** table (source, id, count, first/last seen, route, top frames or sample session) and the label `from-telemetry`.

## 3 · Swap in a real source (optional, 30 min)

Both collectors read a JSON file "in the same shape as a live export". For Sentry, the Sentry CLI's `--json` output or the Sentry MCP's issue list maps onto the six fields the collector reads; for PostHog, the MCP's SQL tool can produce the funnel and rage-click rows. Keep the verifier chain identical — only the fetch changes. For AltoTech repos the equivalent "signals" are Alto Copilot's activity log and the TimescaleDB analytics MCP.

## 4 · Lookback (30 min)

`Use the fitness-lookback skill`. It reads the last 20 closed cards and review reports, clusters recurring findings, and opens **one** PR proposing one line in `CLAUDE.md`, one step in a lane skill, one guard, or one eval case — for a human to merge. After Labs 02–04 the obvious cluster is "timezone handling in energy rollups"; a good proposal is a `CLAUDE.md` line ("all kWh aggregation goes through `daily_kwh(rows, cfg)` with `HOTEL_TZ`") plus an eval case.

## 5 · Exercise

`ex06_verifier.py`: implement the four-step verifier as a pure function over the stub JSON and the board snapshot, returning the verdict schema; the test asserts the nine expected verdicts above. Then break one fixture (set S-104's `last_seen` after its merge date) and watch `drop-fixed` turn into `file`.
