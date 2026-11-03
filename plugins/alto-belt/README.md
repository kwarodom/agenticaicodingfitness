# Alto Belt

A Claude Code mod that coaches a developer from White to Black belt on the AltoTech Claude Code
rubric: eight competencies, each scored 0 to 4, with the belt set by the lowest score. It watches
how you work in each repository, shows a live score, says what to change next, and gives one quiet
nudge when a habit slips.

Install it once on your machine, then type `/belt` in any repository: it scores how you work there on
eight competencies, White to Black, and names the three changes that raise your belt fastest. Nothing
in the repository has to change before you start.

## Before you start

- Claude Code 2.1.287 or newer (`claude --version`). Alto Belt is a mod, and older versions don't load mods.
- The repository you want to score is a git repository.
- Optional: `gh auth status` says you are logged in to github.com. The mod reads your open draft PRs
  with `gh`; without it, Parallel work reads 0 drafts. No access to any private repository is needed.

## Install (once per machine)

Run these inside any Claude Code session. Choose **user scope** when asked, so it works in every
repository you open.

```text
/plugin marketplace add kwarodom/agenticaicodingfitness
/plugin install alto-belt@agentic-coding-fitness
/reload-plugins
/belt help
```

`/belt help` lists the commands; if it does, the mod is loaded.

From a clone of this repository instead:

```bash
claude plugin marketplace add .
claude plugin install alto-belt@agentic-coding-fitness
```

To update later: `/plugin marketplace update agentic-coding-fitness`, then
`/plugin install alto-belt@agentic-coding-fitness` again and `/reload-plugins`.

## Evaluate your repository

Open Claude Code from the repository's folder and work through these in your first session.

1. Type `/belt`. The pane shows your belt, the eight scores with the evidence behind each, and the
   next three changes, lowest score first. Your belt also appears in the status line.
2. Type `/belt web`, or press `w` in the pane, for the belt page in your browser. It is a local file
   that refreshes while you work.
3. Type `/belt checkin`. It fills the prompt with the check-in that the bundled `belt-programme`
   skill runs: a baseline the first time, then a Friday check-in each week.
4. Declare what the mod cannot see, with a short note each time:
   - `/belt declare container <note>`: unattended runs go in a container.
   - `/belt declare cost <note>`: you compared subagents, teams or workflows on measured cost. This
     is the only way to reach 4 in Parallel work.
   - `/belt declare field <note>`: a shipped deployment with evals, a runbook and a cost model.
     Black needs it.
5. Come back with `/belt refresh` after you change the repository's files.

Expect **not measured** on several competencies at first. Behaviour counts build over 14 days of
work in that repository, review rounds over 30 days. A belt above Green needs all eight measured.

### Habits the score watches

These count from the day you install it, so start them now.

- **Run the tests before each commit.** A commit with no test run since the last one shows a reminder.
- **Write Testing, Risk and How to revert into every PR body** you pass to `gh pr create`. A body
  passed with `--body-file` counts as not seen.
- **After two corrections in a row, `/clear`** and write a better first prompt instead of
  correcting a third time.
- **Name the PR in every reviewer run's description**, for example `Review #123, round 2`. Review
  rounds are counted from the description only.
- **Keep at most five worktrees and five draft PRs** open in the repository.

### Troubleshooting

| You see | Do this |
|---|---|
| `/belt` is an unknown command | Run `/reload-plugins`, or start a new session; check `claude --version` is 2.1.287+ |
| Draft PRs read as 0 when you have some | Run `gh auth status` in that repository; the count comes from `gh pr list --author @me --draft` |
| The pane did not change after you added a file | Run `/belt refresh` |
| A competency says not measured | Normal for the first days; it needs 14 days of work in that repository |
| Install says 3 userConfig options are not set | Nothing to do: the defaults apply (see Settings). Change them with `/plugin configure alto-belt@agentic-coding-fitness` |

## What you see

- **`/belt`** opens the pane (typing it again brings it back and reads the repository again; `/belt close` hides it): your belt, the eight scores with the evidence behind each, the next
  three changes (lowest score first), this session's live numbers, and your score over time.
- **The status line** shows your belt and score, for example `belt Green 20/32`.
- **One line above the prompt** when a habit slips (see Nudges). It clears once the habit is back.
- **`/belt web`** opens your belt page in the browser: the belt you hold, a score ring, the five-belt
  path, a card per skill with a mark at what the next belt needs, your next three moves, and today's
  habits. It is a local file (`~/.claude/alto-belt/<repo>/index.html`) that refreshes itself every few
  seconds while you work, and says when something went up. Press `w` in the pane for the same.
- **`/belt checkin`** fills the prompt with the Friday check-in, which the bundled
  `belt-programme` skill runs.

Commands: `/belt`, `/belt web`, `/belt status`, `/belt refresh`, `/belt checkin`, `/belt declare <what> [note]`,
`/belt undeclare <what>`, `/belt help`, `/belt close`.

## Privacy

- The web page is a file on your machine. It loads its fonts from Google Fonts when online and
  falls back to system fonts offline; nothing is sent.

- Everything stays on your machine, in Claude Code's own plugin store. Nothing is sent anywhere.
- Prompts are classified as they are submitted (a correction? does it name a check?) and only the
  **counts** are kept. No prompt text, command text or file content is stored.
- Scores are per repository and per person. There is no team view in this version.

## How each competency is scored

The score is an estimate from evidence, and it is conservative. A competency with too little
evidence shows **not measured** and is left out of the total. A belt above Green needs all eight
measured. Some things the mod cannot see; you **declare** them with `/belt declare`, and the pane
marks them as declared.

Behaviour counts cover the last 14 days in this repository, across all its worktrees (keyed by the
main checkout, via `git rev-parse --git-common-dir`).

| Competency | Evidence | 4 needs |
|---|---|---|
| Verification | Commits with a test run before them this session; PR bodies (from `gh pr create`) with Testing, Risk and How to revert | 90% of commits tested and 90% of PRs with all three sections |
| Context hygiene | Correction streaks, `/clear` after two corrections, turns with the context over 70%, manual `/compact` | No third correction in a row, regular `/clear`, context rarely over 70% |
| Instruction design | CLAUDE.md lines; bytes loaded every session (CLAUDE.md plus rules without `paths:`) | Under 200 lines, under 15,000 bytes, at most one unscoped rule |
| Automation | Hooks in `.claude/settings.json`, PreToolUse guards, hook tests, skills | 3+ hooks with a PreToolUse guard, hook tests and a skill |
| Parallel work | Worktrees of this repo, your open draft PRs, subagent use | Capped at 3 by the mod; 4 needs `/belt declare cost` (a measured cost comparison) |
| Multi-agent judgment | Subagents defined; golden-task sets in `.claude/golden/` with 10+ tasks | A golden set for every agent you define |
| Review design | `REVIEW.md`, a PR template, CODEOWNERS, reviewer agents defined, reviews run (`/code-review`, reviewer agents) | 4 needs the three files and review rounds measured: at least 10 PRs reviewed in 30 days, a mean under 1.5 rounds per PR, and at most one review in ten naming no PR |
| Accountability and safety | Broad pre-approved commands in local settings, `bypassPermissions` as default, guard hooks | No broad approvals, a secrets guard, and `/belt declare container` |

**How review rounds are counted.** Each time a reviewer agent (a type named review, referee,
security, trust or auditor) starts, the mod reads the PR number from the run's description, never from its prompt: a number
after `PR`, `pull request` or `pull/` wins, then a bare `#123` (two digits or more, and not
another repository's `name#22`). A PR's rounds are its busiest reviewer's reads, so two reviewers
reading the same head make one round. A description that names no PR and says golden, task N or
TN is a test replay and is not counted; any other run that names no PR is counted as unnamed.
Only the day of each read is kept, for 60 days.

**Black is never awarded automatically.** It needs every competency at 4 and `/belt declare field`
(a shipped customer deployment with evals, a runbook and a cost model).

### Belts

| Belt | Needs |
|---|---|
| White | Every measured competency at 1 or more |
| Green | Every measured competency at 2 or more |
| Blue | All eight measured and at 3 or more |
| Brown | Blue, with Review design at 4 |
| Black | All eight at 4, and a declared field deployment |

## Nudges

One line above the prompt, the newest first. Each clears by itself.

| When | Says | Clears |
|---|---|---|
| Two corrections in a row | If the next answer misses too, `/clear` and rewrite the first prompt | A prompt that is not a correction |
| Three or more in a row | Two-strike rule: `/clear` and write a better first prompt | `/clear`, or a prompt that is not a correction |
| Context over 70% | `/compact` with instructions, or `/clear` if the task changed | Context back under 70% |
| A commit with no test run since the last one | Run the tests and show the output | A test run |
| `gh pr create` without Testing, Risk or How to revert | Which sections are missing | Three prompts later |

Corrections are recognised in English and Thai. The word lists live in `hooks/signals.ts`, the one
place to tune them.

## Settings

| Option | Default | What it does |
|---|---|---|
| `band` | on | The line above the prompt |
| `status` | on | The belt in the status line |
| `coach_model` | off | Adds one short coaching section to what Claude reads: ask for a verifier before a non-trivial task, show output before saying done, suggest `/clear` after two failed corrections. Off keeps the score a measure of you, not of the mod |

## Not in this version

A team board, trend charts, and reading `--body-file` PR bodies from
disk (a PR made that way counts as "body not seen", never as missing sections).

## The programme

The bundled skill `belt-programme` is the full self-guided programme: the rubric, Week 0, the eight
practices, the 13-week path and the coaching rules. It comes from AltoTech's research brief
*Becoming a Top-1% Claude Code AI Developer* (29 Oct 2026).

## Credits

Belt badges (the Alto mascot in a medallion, one per belt) generated with Higgsfield (GPT Image 2.5). Layout patterns drawn from Fiverr's level overview,
Brilliant's and Noom's course paths, WHOOP's locked levels and Searchable's headline-plus-ring dashboard,
found on Mobbin.
