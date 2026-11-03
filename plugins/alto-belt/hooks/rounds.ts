// Review rounds per pull request: how many times each reviewer read each PR. A reviewer run counts
// toward a PR only when its description names it (#123); the prompt is never read, because test
// replays quote old PR numbers there. Days only are kept, never text.

/** PR number → reviewer → the day of each read, one entry per read. Key `?` holds reviews that named no PR. */
export type RoundsLog = Record<string, Record<string, string[]>>

export type Rounds = {
  /** PRs read by a reviewer in the window. */
  prs: number
  /** Mean rounds per PR, where a PR's rounds are its busiest reviewer's reads. */
  mean: number | null
  /** The PRs with the most rounds, most first. */
  worst: { pr: string; rounds: number }[]
  /** Reviewer runs in the window that named no PR. */
  unnamed: number
  /** Reviewer runs in the window that named a PR. */
  named: number
}

export const EMPTY_ROUNDS: Rounds = { prs: 0, mean: null, worst: [], unnamed: 0, named: 0 }

/** Rounds are judged over 30 days, so a quiet fortnight does not take the belt away. */
export const ROUNDS_WINDOW_DAYS = 30
export const ROUNDS_MIN_PRS = 10
export const ROUNDS_TARGET = 1.5
const KEEP_DAYS = 60
const UNNAMED = '?'

const REVIEWERS = /(review|referee|security|trust|auditor)/i
// "PR 869", "PR#869", "PR-872", "pull request 872", "pull/872": the PR word wins over any other number.
const PR_WORD = /\b(?:PR|pull request|pull)[\s#:/-]*(\d{1,6})\b/i
// A bare "#869", not another repository's "plugins#22" or a one-digit "round #2".
const HASH = /(?<![\w/.-])#(\d{2,6})\b/
// A golden-set replay or other test run of a reviewer, when it names no PR.
const TEST_RUN = /\bgolden\b|\btask \d+\b|\bT\d{1,2}\b/i

/** The reviewer a tool call starts, or null when it does not start one. */
export const reviewerOf = (tool: string, input: Record<string, unknown>): string | null => {
  if (tool !== 'Agent' && tool !== 'Task') return null
  const type = String(input.subagent_type ?? '')
  return REVIEWERS.test(type) ? type : null
}

/** The PR a reviewer run names in its description: a number, `?` when it names none, null for a test run. */
export const prOfReview = (input: Record<string, unknown>): string | null => {
  const description = String(input.description ?? '')
  const pr = PR_WORD.exec(description)?.[1] ?? HASH.exec(description)?.[1]
  if (pr !== undefined) return pr
  return TEST_RUN.test(description) ? null : UNNAMED
}

/** The log with one more read by `reviewer` of `pr` on `day`, and reads older than the kept days dropped. */
export const withRead = (log: RoundsLog, pr: string, reviewer: string, day: string, today: string = day): RoundsLog => {
  const from = shiftedDay(today, -(KEEP_DAYS - 1))
  const next: RoundsLog = {}
  for (const [key, byReviewer] of Object.entries(log)) {
    const kept = Object.fromEntries(
      Object.entries(byReviewer)
        .map(([who, days]) => [who, days.filter(one => one >= from)] as const)
        .filter(([, days]) => days.length > 0),
    )
    if (Object.keys(kept).length > 0) next[key] = kept
  }
  const byReviewer = { ...(next[pr] ?? {}) }
  byReviewer[reviewer] = [...(byReviewer[reviewer] ?? []), day]
  next[pr] = byReviewer
  return next
}

/** Rounds over the `span` days up to `today`, inclusive. */
export const roundsOf = (log: RoundsLog, today: string, span: number = ROUNDS_WINDOW_DAYS): Rounds => {
  const from = shiftedDay(today, -(span - 1))
  const inWindow = (days: string[]): number => days.filter(one => one >= from && one <= today).length
  let unnamed = 0
  let named = 0
  const perPr: { pr: string; rounds: number }[] = []
  for (const [pr, byReviewer] of Object.entries(log)) {
    const reads = Object.values(byReviewer).map(inWindow)
    if (pr === UNNAMED) {
      unnamed += reads.reduce((sum, one) => sum + one, 0)
      continue
    }
    named += reads.reduce((sum, one) => sum + one, 0)
    const rounds = Math.max(0, ...reads)
    if (rounds > 0) perPr.push({ pr, rounds })
  }
  if (perPr.length === 0) return { ...EMPTY_ROUNDS, unnamed, named }
  const mean = perPr.reduce((sum, one) => sum + one.rounds, 0) / perPr.length
  const worst = [...perPr].sort((a, b) => b.rounds - a.rounds || Number(a.pr) - Number(b.pr)).slice(0, 3)
  return { prs: perPr.length, mean: Math.round(mean * 100) / 100, worst, unnamed, named }
}

/** Whether rounds meet the bar for Review design 4: enough PRs, mean under target, at most one review in ten unnamed. */
export const roundsMet = (rounds: Rounds): boolean =>
  rounds.prs >= ROUNDS_MIN_PRS && rounds.mean !== null && rounds.mean < ROUNDS_TARGET && rounds.unnamed * 10 <= rounds.unnamed + rounds.named

/** One line for the evidence: the mean, the worst PRs and the unnamed reviews. */
export const roundsLine = (rounds: Rounds): string => {
  if (rounds.prs === 0) return rounds.unnamed > 0 ? `rounds not measured: ${rounds.unnamed} reviews named no PR` : 'rounds not measured yet'
  const worst = rounds.worst.filter(one => one.rounds > 1).map(one => `#${one.pr} (${one.rounds})`).join(', ')
  return [
    `rounds ${rounds.mean} over ${rounds.prs} PR${rounds.prs === 1 ? '' : 's'}`,
    worst === '' ? null : `worst ${worst}`,
    rounds.unnamed > 0 ? `${rounds.unnamed} review${rounds.unnamed === 1 ? '' : 's'} named no PR` : null,
  ]
    .filter(Boolean)
    .join(' · ')
}

const shiftedDay = (day: string, by: number): string => {
  const at = new Date(`${day}T12:00:00`)
  at.setDate(at.getDate() + by)
  const pad = (n: number): string => String(n).padStart(2, '0')
  return `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())}`
}
