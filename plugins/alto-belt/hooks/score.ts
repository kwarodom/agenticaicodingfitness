// Evidence to scores: each competency 0 to 4 or not measured, the belt, and the next three changes.
// Conservative on purpose: too little evidence is "not measured", and Black is never automatic.

import type { Assessment, Belt, Competency, CompetencyScore, Declared, RepoScan, Tally } from '../types'
import { EMPTY_ROUNDS, ROUNDS_MIN_PRS, ROUNDS_TARGET, roundsLine, roundsMet } from './rounds'
import type { Rounds } from './rounds'

export const COMPETENCIES: { id: Competency; name: string }[] = [
  { id: 'verification', name: 'Verification' },
  { id: 'context', name: 'Context hygiene' },
  { id: 'instructions', name: 'Instruction design' },
  { id: 'automation', name: 'Automation' },
  { id: 'parallel', name: 'Parallel work' },
  { id: 'multiagent', name: 'Multi-agent judgment' },
  { id: 'review', name: 'Review design' },
  { id: 'safety', name: 'Accountability and safety' },
]

export const BELT_NAMES: Record<Belt, string> = {
  none: 'No belt yet',
  white: 'White',
  green: 'Green',
  blue: 'Blue',
  brown: 'Brown',
  black: 'Black',
}

// Thresholds the rubric names.
export const CLAUDE_MD_MAX_LINES = 200
export const ALWAYS_ON_MAX_BYTES = 15_000
export const WIP_CAP = 5
const MIN_PROMPTS = 20
const MIN_COMMITS = 3

const pct = (part: number, whole: number): number => (whole === 0 ? 0 : part / whole)
const ofN = (part: number, whole: number): string => `${part} of ${whole}`

type Scored = { score: number | null; evidence: string; next: string }

const verification = (t: Tally): Scored => {
  const commitRate = pct(t.testedCommits, t.commits)
  const prRate = pct(t.prsEvidenced, t.prsSeen)
  const evidence = `tests before commit ${ofN(t.testedCommits, t.commits)} · PRs with Testing, Risk and revert ${ofN(t.prsEvidenced, t.prsSeen)}`
  if (t.commits < MIN_COMMITS && t.prsSeen < 2) return { score: null, evidence: `${evidence} · too few commits to judge`, next: 'Commit a few changes here; the score needs 3 commits or 2 PRs.' }
  const next = 'Name the check in the prompt ("done when … passes"), run it before each commit, and put Testing, Risk and How to revert in every PR.'
  if (commitRate >= 0.9 && t.prsSeen >= 2 && prRate >= 0.9) return { score: 4, evidence, next: 'Keep it up: every change ships with its evidence.' }
  if (commitRate >= 0.7 || prRate >= 0.8) return { score: 3, evidence, next }
  if (commitRate >= 0.4 || prRate >= 0.4) return { score: 2, evidence, next }
  if (t.testedCommits > 0 || t.prsEvidenced > 0) return { score: 1, evidence, next }
  return { score: 0, evidence, next }
}

const context = (t: Tally, live: number | null): Scored => {
  const fullShare = pct(t.fullTurns, t.turns)
  const evidence = `third corrections in a row ${t.thirdStrikes} · /clear ${t.clears} · turns over 70% full ${ofN(t.fullTurns, t.turns)}${live === null ? '' : ` · now ${live}%`}`
  if (t.prompts < MIN_PROMPTS) return { score: null, evidence: `${evidence} · fewer than ${MIN_PROMPTS} prompts`, next: `Work here a little longer; the score needs ${MIN_PROMPTS} prompts.` }
  const next = 'After two failed corrections, /clear and rewrite the first prompt; /compact with instructions before the context passes 70%.'
  const clearsEnough = t.clears >= Math.max(1, Math.floor(t.prompts / 40))
  if (t.thirdStrikes === 0 && clearsEnough && fullShare < 0.1) return { score: 4, evidence, next: 'Context is clean. Keep clearing between tasks.' }
  if (t.thirdStrikes <= 1 && t.clears >= 1 && fullShare < 0.25) return { score: 3, evidence, next }
  if (t.clears >= 1) return { score: 2, evidence, next }
  if (t.thirdStrikes >= 3) return { score: 0, evidence, next }
  return { score: 1, evidence, next }
}

const instructions = (s: RepoScan): Scored => {
  if (s.claudeMdLines === null) return { score: 0, evidence: 'no CLAUDE.md', next: 'Write a CLAUDE.md under 200 lines: commands, conventions, non-goals, and how a change is verified.' }
  const evidence = `CLAUDE.md ${s.claudeMdLines} lines · ${s.alwaysOnBytes.toLocaleString('en-US')} bytes load every session · ${s.rulesUnscoped} of ${s.rulesTotal} rules unscoped`
  const next = s.claudeMdLines >= CLAUDE_MD_MAX_LINES
    ? `Cut CLAUDE.md under ${CLAUDE_MD_MAX_LINES} lines: ask of each line "would removing this cause a mistake?"`
    : `Get always-on text under ${ALWAYS_ON_MAX_BYTES.toLocaleString('en-US')} bytes: add paths: to rules that apply to some files only.`
  if (s.claudeMdLines < CLAUDE_MD_MAX_LINES && s.alwaysOnBytes < ALWAYS_ON_MAX_BYTES && s.rulesUnscoped <= 1) return { score: 4, evidence, next: 'Lean. Re-check after each rule you add.' }
  if (s.claudeMdLines < CLAUDE_MD_MAX_LINES && s.alwaysOnBytes < 25_000) return { score: 3, evidence, next }
  if (s.claudeMdLines <= 400) return { score: 2, evidence, next }
  return { score: 1, evidence, next }
}

const automation = (s: RepoScan): Scored => {
  const evidence = `${s.hooks} hooks, ${s.preToolUseHooks} before tool calls · hook tests ${s.hasHookTests ? 'yes' : 'no'} · ${s.skills} skills`
  const next = s.preToolUseHooks === 0
    ? 'Turn the rule you repeat most into a PreToolUse hook that exits 2, then prove it blocks.'
    : !s.hasHookTests
      ? 'Add a test that runs each guard hook and checks it blocks.'
      : 'Turn the next playbook you paste a third time into a skill.'
  if (s.hooks >= 3 && s.preToolUseHooks >= 1 && s.hasHookTests && s.skills >= 1) return { score: 4, evidence, next: 'Rules that must hold are hooks, with tests. Keep promoting repeats.' }
  if (s.hooks >= 2 && s.preToolUseHooks >= 1) return { score: 3, evidence, next }
  if (s.hooks >= 1 || s.skills >= 1) return { score: 2, evidence, next }
  if (s.agents >= 1 || s.rulesTotal >= 1) return { score: 1, evidence, next }
  return { score: 0, evidence, next }
}

const parallel = (s: RepoScan, t: Tally, declared: Declared): Scored => {
  const worktrees = s.worktrees
  const drafts = s.drafts
  const evidence = `${worktrees ?? '?'} worktrees · ${drafts ?? '?'} of your draft PRs open · ${t.agentSpawns} subagents run`
  if (worktrees === null) return { score: null, evidence, next: 'Run /belt refresh inside a git repository.' }
  const isCapped = worktrees <= WIP_CAP && (drafts === null || drafts <= WIP_CAP)
  const usesParallel = worktrees >= 2 || t.agentSpawns >= 1
  const next = isCapped
    ? 'Compare one task done by subagents and by one loop: tokens, time and result. Then /belt declare cost.'
    : `Cap work in progress at ${WIP_CAP}: land or close drafts, and git worktree remove merged ones.`
  if (isCapped && usesParallel && declared.cost !== undefined) return { score: 4, evidence: `${evidence} · cost comparison declared`, next: 'Parallel by choice, capped, and costed.' }
  if (isCapped && usesParallel) return { score: 3, evidence, next }
  if (usesParallel) return { score: 2, evidence, next }
  return { score: 1, evidence, next: 'Try two worktrees on two unrelated tasks, and compare the time.' }
}

const multiagent = (s: RepoScan, t: Tally): Scored => {
  const evidence = `${s.agents} subagents defined · ${s.goldenSets} golden-task sets with 10+ tasks · ${t.agentSpawns} subagents run`
  const next = 'Write ten real tasks with expected outcomes in .claude/golden/<agent>.md and run each twice before changing that agent.'
  if (s.goldenSets >= 1 && s.goldenSets >= s.agents) return { score: 4, evidence, next: 'Every agent has a golden set. Re-run it on each change.' }
  if (s.goldenSets >= 1) return { score: 3, evidence, next }
  if (s.agents >= 1 || t.agentSpawns >= 1) return { score: 2, evidence, next }
  return { score: 1, evidence, next: 'Use a read-only reviewer subagent on your next diff.' }
}

const review = (s: RepoScan, t: Tally, rounds: Rounds): Scored => {
  const have = [s.hasReviewMd ? 'REVIEW.md' : null, s.hasPrTemplate ? 'PR template' : null, s.hasCodeowners ? 'CODEOWNERS' : null].filter(Boolean)
  const evidence = `${have.length === 0 ? 'no review files' : have.join(', ')} · ${s.reviewerAgents} reviewer agents · ${t.reviews} fresh-context reviews run · ${roundsLine(rounds)}`
  const next = !s.hasReviewMd
    ? 'Write a REVIEW.md: what failure looks like here, file:line evidence, three nits at most, two rounds at most.'
    : !s.hasPrTemplate
      ? 'Add a PR template with Testing, Risk and How to revert.'
      : rounds.prs < ROUNDS_MIN_PRS
        ? `Name the PR (#123) in every reviewer's description so its rounds count; ${ROUNDS_MIN_PRS} PRs reviewed in 30 days are needed.`
        : rounds.mean !== null && rounds.mean >= ROUNDS_TARGET
          ? `Bring review rounds under ${ROUNDS_TARGET} per PR: after round 2, stop and settle the shape with the owner.`
          : 'Name the PR (#123) in every reviewer\'s description: at most one review in ten may go unnamed.'
  const full = s.hasReviewMd && s.hasPrTemplate && s.hasCodeowners
  if (full && t.reviews >= 1 && roundsMet(rounds)) return { score: 4, evidence, next: 'Review is designed and measured. Keep rounds under 1.5.' }
  if (s.hasReviewMd && (s.hasPrTemplate || s.hasCodeowners)) return { score: 3, evidence, next }
  if (have.length >= 1 || t.reviews >= 1 || s.reviewerAgents >= 1) return { score: 2, evidence, next }
  return { score: 1, evidence, next }
}

const safety = (s: RepoScan, declared: Declared): Scored => {
  const evidence = `${s.broadApprovals} broad pre-approved commands · ${s.guardHooks} guard hooks${s.isBypassDefault ? ' · bypassPermissions is the default' : ''}`
  const next = s.broadApprovals > 0
    ? 'Remove pre-approvals that run anything (any Python, any shell, gh api *) from your local settings.'
    : s.guardHooks === 0
      ? 'Add a hook that blocks reading or editing .env and other secrets.'
      : 'Run unattended loops only in a container with a kill-switch file, then /belt declare container.'
  if (s.isBypassDefault) return { score: 0, evidence, next: 'Stop defaulting to bypassPermissions; pre-allow the exact commands instead.' }
  if (s.broadApprovals === 0 && s.guardHooks >= 1 && declared.container !== undefined) return { score: 4, evidence: `${evidence} · container declared`, next: 'Autonomy is fenced.' }
  if (s.broadApprovals === 0 && s.guardHooks >= 1) return { score: 3, evidence, next }
  if (s.broadApprovals <= 2) return { score: 2, evidence, next }
  return { score: 1, evidence, next }
}

export const beltOf = (scores: (number | null)[], declared: Declared): Belt => {
  const measured = scores.filter((one): one is number => one !== null)
  if (measured.length === 0) return 'none'
  const low = Math.min(...measured)
  const isAll = measured.length === scores.length
  const reviewScore = scores[COMPETENCIES.findIndex(one => one.id === 'review')]
  if (isAll && low >= 4 && declared.field !== undefined) return 'black'
  if (isAll && low >= 3 && reviewScore === 4) return 'brown'
  if (isAll && low >= 3) return 'blue'
  if (low >= 2) return 'green'
  if (low >= 1) return 'white'
  return 'none'
}

const gateOf = (belt: Belt, scores: CompetencyScore[], declared: Declared): string => {
  const unmeasured = scores.filter(one => one.score === null).map(one => nameOf(one.id))
  const below = (n: number): string[] => scores.filter(one => one.score !== null && one.score < n).map(one => nameOf(one.id))
  if (belt === 'black') return 'Top of the ladder. Teach it.'
  if (belt === 'brown') return declared.field === undefined ? `Black needs every competency at 4${below(4).length ? ` (${below(4).join(', ')})` : ''} and /belt declare field.` : `Black needs ${below(4).join(', ')} at 4.`
  if (belt === 'blue') return 'Brown needs Review design at 4.'
  if (unmeasured.length > 0 && below(3).length === 0) return `Blue needs all eight measured: ${unmeasured.join(', ')}.`
  const nextLevel = belt === 'green' ? 3 : belt === 'white' ? 2 : 1
  const missing = below(nextLevel)
  return `${BELT_NAMES[belt === 'none' ? 'white' : belt === 'white' ? 'green' : 'blue']} needs ${missing.join(', ')} at ${nextLevel}${unmeasured.length ? `, and ${unmeasured.join(', ')} measured` : ''}.`
}

export const nameOf = (id: Competency): string => COMPETENCIES.find(one => one.id === id)?.name ?? id

/** Scores the evidence. The window is the last 14 days of counts in this repository; rounds cover 30 days. */
export const assess = (scan: RepoScan, window: Tally, declared: Declared, livePercent: number | null, rounds: Rounds = EMPTY_ROUNDS): Assessment => {
  const byId: Record<Competency, Scored> = {
    verification: verification(window),
    context: context(window, livePercent),
    instructions: instructions(scan),
    automation: automation(scan),
    parallel: parallel(scan, window, declared),
    multiagent: multiagent(scan, window),
    review: review(scan, window, rounds),
    safety: safety(scan, declared),
  }
  const scores: CompetencyScore[] = COMPETENCIES.map(({ id }) => ({ id, ...byId[id] }))
  const values = scores.map(one => one.score)
  const belt = beltOf(values, declared)
  const total = values.reduce<number>((sum, one) => sum + (one ?? 0), 0)
  const measured = values.filter(one => one !== null).length
  // Lowest first, since the lowest score sets the belt; unmeasured after the measured ones.
  const ranked = [...scores]
    .filter(one => one.score === null || one.score < 4)
    .sort((a, b) => (a.score ?? 5) - (b.score ?? 5))
  return { scores, total, measured, belt, gate: gateOf(belt, scores, declared), next: ranked.slice(0, 3).map(one => `${nameOf(one.id)}: ${one.next}`) }
}
