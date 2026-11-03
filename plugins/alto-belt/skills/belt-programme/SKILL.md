---
name: belt-programme
description: AltoTech's self-guided Claude Code belt programme (White to Black) — the eight-competency rubric, Week 0 baseline, the scorecard script, eight practices, the 13-week path and the Friday check-in. Use when someone asks to be coached on their Claude Code habits, to score themselves, to run the belt check-in, or what to improve next to reach the next belt; /belt checkin fills the prompt for it.
---

# Claude Code Belt Programme: self-guided

A 13-week programme to go from wherever you are today to Black belt with Claude Code.
Run it yourself, in your own repository, with Claude Code as your coach.

Based on the research brief *Becoming a Top-1% Claude Code AI Developer* (AltoTech,
29 Oct 2026): rubric §6.2, belts §4, scorecard §2.4.

---

## How to use this file

1. Copy this file into your repository at `docs/claude-code-belt.md`, or keep it anywhere
   on your machine.
2. Open Claude Code in your repository and paste:

   > Read `<path to this file>`. Act as my coach for this programme. Start with
   > "Week 0: baseline". Run the scorecard, ask me the eight rubric questions one at a
   > time, then tell me my belt and the three changes that would raise it fastest.
   > Follow the coaching rules at the end of the file.

3. Every Friday, paste:

   > Read `<path to this file>`. Run the Friday check-in.

---

## The rubric

Score yourself 0 to 4 on each competency. 1 and 3 sit between the descriptions.

| Competency | 0 | 2 | 4 |
|---|---|---|---|
| Verification | Trusts "done" | Runs tests sometimes | Every change ships with runnable evidence, and screenshots for UI |
| Context hygiene | Never clears | Clears between tasks | Two-strike rule, `/compact` with instructions, checks `/context` |
| Instruction design | No CLAUDE.md | Long CLAUDE.md | Under 200 lines, rules scoped with `paths:` |
| Automation | Rules written as prose | Some skills | Hooks enforce the rules that must always hold; a skill for every playbook used three times |
| Parallel work | One session | Uses worktrees | Picks subagent, team or workflow by measured cost and quality |
| Multi-agent judgment | Adds agents freely | Uses subagents | Defends each design with ten test tasks, run twice |
| Review design | Reads every diff | Uses `/code-review` | REVIEW.md, risk tiers, review rounds per PR tracked |
| Accountability and safety | Bypasses permissions | Pre-allows commands | Containers for unattended runs, kill switches, owns what lands |

### Belts

The lowest score decides the belt, not the total.

| Belt | Identity | Needs |
|---|---|---|
| White | Operator | Every competency at 1 or more |
| Green | Harness Engineer | Every competency at 2 or more |
| Blue | Architect | Every competency at 3 or more |
| Brown | Designer and Evaluator | Every competency at 3 or more, and Review design at 4 |
| Black | FDE and Multiplier | Every competency at 4, and a shipped customer deployment with evals, a runbook and a cost model |

"Top 1%" has no published population benchmark. Treat the score as a training gauge you
re-run monthly, not a percentile.

---

## Week 0: baseline (60 minutes)

### 1. Run the scorecard

Save this as `scorecard.sh` outside the repository (for example `~/bin/scorecard.sh`) and
run it from the repository root: `bash ~/bin/scorecard.sh OWNER/REPO 30`. It needs `gh`
and `jq`.

```bash
#!/usr/bin/env bash
# Weekly Claude Code scorecard. Run from the repo root: ./scorecard.sh OWNER/REPO [days]
set -euo pipefail
REPO=${1:?usage: scorecard.sh OWNER/REPO [days]}
DAYS=${2:-30}
SINCE=$(date -u -v-"${DAYS}"d +%Y-%m-%d 2>/dev/null || date -u -d "-${DAYS} days" +%Y-%m-%d)

echo "Scorecard for $REPO since $SINCE"

# Instruction budget: CLAUDE.md plus every rule without paths: frontmatter loads in every session.
always=$(wc -c < CLAUDE.md)
for f in .claude/rules/*.md; do
  [ -f "$f" ] || continue
  grep -q '^paths:' "$f" || always=$((always + $(wc -c < "$f")))
done
echo "CLAUDE.md lines:            $(wc -l < CLAUDE.md)   (target < 200)"
echo "Always-on instruction bytes: $always   (target < 15000)"

# Work in progress
echo "Git worktrees:              $(git worktree list | wc -l | tr -d ' ')   (target <= 5)"
gh pr list -R "$REPO" --state open --limit 1000 --json isDraft \
  --jq '"Open PRs / drafts:          \(length) / \(map(select(.isDraft))|length)   (target drafts <= 5)"'

# Merged PRs: throughput, reverts, and evidence in the PR body
gh pr list -R "$REPO" --state merged --limit 1000 --search "merged:>=$SINCE" \
  --json title,body,createdAt,mergedAt --jq '
  def pct(f): ((map(select(f))|length) * 100 / (length|if .==0 then 1 else . end) | floor);
  "Merged PRs:                 \(length)",
  "Revert PRs:                 \(map(select(.title|test("^revert";"i")))|length)",
  "PRs with a Testing section: \(pct(.body|test("(?i)#+ *(test|testing|verification)")))%   (target 100%)",
  "PRs with a Risk section:    \(pct(.body|test("(?i)#+ *risk")))%   (target 100%)",
  "PRs with a revert plan:     \(pct(.body|test("(?i)(how to )?(revert|roll ?back)")))%   (target 100%)",
  "Median hours open->merged:  \(map(((.mergedAt|fromdate)-(.createdAt|fromdate))/3600)|sort|.[length/2|floor]|floor)"'
```

Note: the scorecard counts PRs for the whole repository, not just yours. Add
`--author @me` to both `gh pr list` lines to see only your own.

### 2. Measure your context

Start a fresh Claude Code session and run `/context` before typing anything. Write down how
much of the window is already used.

### 3. Score yourself

Answer the eight rubric rows honestly. Write the scores into your log (template at the end).

---

## Week 0, continued: changes that touch only your machine

These change nothing in the shared repository, so you can do them on day one.

1. **Turn off what this repository never uses.** In `.claude/settings.local.json` set
   `"enabledPlugins": { "<plugin>@<marketplace>": false }` for each unused plugin. In
   `/mcp`, disable connectors you don't use here. Run `/context` in a fresh session again
   and record the difference.
2. **Clean your memory.** Open `~/.claude/projects/<project>/memory/MEMORY.md`. Delete
   entries for work that is finished. Each remaining entry should say *why* and *how to
   apply*, not only "don't do X".
3. **List your worktrees and drafts.** `git worktree list` and
   `gh pr list --author @me --state open --draft`. Mark which can be closed. Don't delete
   another person's or agent's live worktree.

### Four habits that start with your next prompt

- **Name the verifier in the prompt.** "Done when `<test command>` passes and the screen
  shows `<what>`."
- **Two strikes, then `/clear`.** After two failed corrections, clear and write a better
  first prompt instead of correcting a third time.
- **"Prove to me this works"** before every PR. Expect pasted output, not a claim.
- **Every PR body has Testing, Risk and How to revert**, written by hand until a template
  exists.

---

## The eight practices

Each practice ends with a test for when it is done.

1. **Turn every repeated mistake into a hook.** A rule in CLAUDE.md is a request; a
   PreToolUse hook that exits 2 is enforcement. *Done when* you asked Claude to break the
   rule and the transcript shows the block, and a test proves it.

   ```bash
   #!/usr/bin/env bash
   # Example PreToolUse hook. Matcher: "Edit|Write|Read|Bash". Exit 2 blocks the call.
   input=$(cat)
   target=$(jq -r '.tool_input.file_path // .tool_input.command // empty' <<<"$input")
   if [[ "$target" == *".env"* && "$target" != *".env.example"* ]]; then
     echo "Blocked: .env holds secrets. Ask the user to run the step with '!'." >&2
     exit 2
   fi
   exit 0
   ```

2. **Write memory with the reason, then promote it.** End each correction with "save this
   as a feedback memory with Why and How to apply." When a lesson keeps coming back, make
   it a hook and shrink the memory to a pointer. Decisions teammates need belong in the
   repository, because teammates and subagents never see your memory. *Done when*
   MEMORY.md is under 150 lines with no finished entries.

3. **One owner per directory, a lead who integrates.** Agent teams share a working tree,
   so two agents editing one file lose work. Give each subagent a `tools:` list and a
   directory it owns; reviewers get `Read, Grep, Glob, Bash` only. *Done when* a month
   passes with no merge conflict between agents and every reviewer finding names
   `file:line`.

4. **Show the output before saying done.** Claude runs the tests, or drives the real app in
   a browser for UI, and pastes what it saw. *Done when* every PR body has a Testing section
   with a pasted summary line or a screenshot.

5. **Keep always-on instructions under 15 KB.** CLAUDE.md and every rule without `paths:`
   load in every session. Remove generic rules that contradict your project's own, scope
   language rules with `paths:`, and test each CLAUDE.md line with "would removing this
   cause a mistake?" *Done when* the scorecard shows under 15,000 bytes and CLAUDE.md is
   under 200 lines.

6. **Cap work in progress at five.** Parallel agents multiply token use, and every finished
   agent waits for you to review it. *Done when* you end each week with 5 or fewer
   worktrees and 5 or fewer draft PRs, and no uncommitted work older than 3 days.

7. **Every PR states Testing, Risk and How to revert.** Add
   `.github/pull_request_template.md` with the three headings, a `REVIEW.md` that says what
   failure looks like in your repository, and CODEOWNERS for auth, permissions and CI.
   *Done when* 100% of your merged PRs carry all three sections and your mean review rounds
   per PR is below 1.5.

8. **Ten test tasks before any architecture change.** Before adding or removing a
   subagent, skill or hook, write ten real tasks with expected outcomes and run each twice,
   before and after. Record which tasks flipped, plus time and cost. *Done when* each agent
   and skill you rely on has a ten-task set in the repository.

---

## The 13-week path

Shared-repository changes (practices 5 and 7) go through a normal PR with the owner of
those files. Check your team's freeze calendar first.

| Weeks | Gate | Ship |
|---|---|---|
| 0 | Baseline | Scorecard, `/context` figure, self-score, machine-only changes, four habits |
| 1–2 | | Practices 5 and 7 as PRs; practice 6 cleanup; Friday scorecard from now on |
| 3–4 | **Blue** | Ten test tasks for your most-used agent or skill (practice 8); review-round log started; a container and kill-switch file for any unattended run |
| 5–8 | **Brown** | Review rounds under 1.5; test-task sets for every agent you rely on; remove harness parts one at a time and keep only what moves the score |
| 9–13 | **Black** | One customer deployment written up with runbook, evals and cost model; teach a 60-minute session; a peer re-scores you |

---

## Weekly routine

| Day | Practice (30–60 min) | Log |
|---|---|---|
| Monday | Plan mode on the week's first ticket; `/context` at session start | Corrections before done; context used |
| Tuesday | Up to three worktrees in parallel on unrelated tasks | Time saved; minutes spent waiting on you |
| Wednesday | Fresh-context review of your own diff, then `/code-review` | Findings fixed vs dismissed, with reasons |
| Thursday | Promote one repeated correction to a skill or hook, with a test | What it replaced |
| Friday | Check-in: scorecard, prune worktrees and drafts, add one test task | Log entry below |

---

## Friday log template

Copy this block into `~/claude-code-belt-log.md` each week.

```markdown
### Week N · YYYY-MM-DD
Scorecard: always-on __ B · CLAUDE.md __ lines · worktrees __ · drafts __ ·
merged __ · reverts __ · Testing __% · Risk __% · revert plan __%
Context at session start: __% used
Rubric: Verif _ · Context _ · Instr _ · Auto _ · Parallel _ · Multi _ · Review _ · Safety _ = __/32
Belt: ____
Shipped this week:
Next week's one change:
```

---

## Coaching rules for Claude

When a learner asks you to coach them from this file:

1. **Measure before advising.** Run the scorecard and read `/context` output before
   suggesting changes. Quote the numbers.
2. **Score honestly.** Base each rubric score on evidence you can point to (a file, a PR,
   a count). When evidence is missing, say "not measured" and score conservatively.
3. **One change at a time.** Recommend at most three changes, ordered by which raises the
   lowest competency first, because the lowest score decides the belt.
4. **Respect the repository's rules.** Follow the project's CLAUDE.md, lane ownership and
   freeze calendar. Don't edit shared files (CLAUDE.md, rules, templates, CI) without the
   learner confirming they own them or have the owner's agreement. Never touch `.env` or
   secrets; hand those steps to the learner with `!`.
5. **Machine-only first.** Prefer changes to `settings.local.json`, personal memory and
   habits until the learner opens a PR for shared changes.
6. **Show evidence of done.** Before marking a practice done, show the command output or
   transcript that meets its "done when" test.
7. **Friday check-in:** run the scorecard, compare with last week's log entry, re-score the
   rubric, fill in the log template, and name next week's one change.

---

## With the Alto Belt mod

When the `alto-belt` mod is installed, start every coaching session from what it measured:
`/belt status` gives the belt, the eight scores with their evidence and the next three changes.
Use those numbers instead of guessing, and say which ones the mod marks "not measured" or
"declared". Record anything the learner proves that the mod cannot see with
`/belt declare <container|field|cost> <note>`. Review rounds are measured, not declared:
the mod counts a round each time a reviewer agent runs with the PR named in its description
(`#123`).
