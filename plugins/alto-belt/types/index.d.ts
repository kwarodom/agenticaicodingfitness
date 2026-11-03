/** The eight competencies of the rubric, in the order the pane lists them. */
export type Competency =
  | 'verification'
  | 'context'
  | 'instructions'
  | 'automation'
  | 'parallel'
  | 'multiagent'
  | 'review'
  | 'safety'

export type Belt = 'none' | 'white' | 'green' | 'blue' | 'brown' | 'black'

/** What the person says is true that the mod cannot see. */
export type Declaration = 'container' | 'rounds' | 'field' | 'cost'
export type Declared = Partial<Record<Declaration, { at: number; note: string }>>

/** Counts for one day in one repository. Counts only: no prompt or command text. */
export type Tally = {
  prompts: number
  verifierPrompts: number
  corrections: number
  /** A third (or later) correction in a row. */
  thirdStrikes: number
  /** A /clear that followed two or more corrections in a row. */
  strikeClears: number
  clears: number
  manualCompacts: number
  turns: number
  /** Turns that ended with the context over the high-water mark. */
  fullTurns: number
  testRuns: number
  commits: number
  testedCommits: number
  prs: number
  /** PRs whose body the command carried, so its sections could be read. */
  prsSeen: number
  prsEvidenced: number
  reviews: number
  agentSpawns: number
}

/** What a scan of the repository found. Null where it could not be read. */
export type RepoScan = {
  root: string
  /** The main checkout, the key the counts are kept under. */
  repo: string
  name: string
  claudeMdLines: number | null
  alwaysOnBytes: number
  rulesTotal: number
  rulesUnscoped: number
  hooks: number
  preToolUseHooks: number
  guardHooks: number
  hasHookTests: boolean
  skills: number
  agents: number
  /** Subagents whose name says they review: review, referee, security, trust, auditor. */
  reviewerAgents: number
  goldenSets: number
  hasReviewMd: boolean
  hasPrTemplate: boolean
  hasCodeowners: boolean
  broadApprovals: number
  isBypassDefault: boolean
  worktrees: number | null
  drafts: number | null
}

export type Feed<T> = { data: T | null; fetchedAt: number | null; error: string | null }

export type CompetencyScore = { id: Competency; score: number | null; evidence: string; next: string }

export type Assessment = {
  scores: CompetencyScore[]
  total: number
  measured: number
  belt: Belt
  /** Why the belt is not one higher, in a sentence. */
  gate: string
  next: string[]
}

export type ScorePoint = { day: string; total: number; measured: number; belt: Belt }

export type NudgeKind = 'strike2' | 'strike3' | 'context' | 'untested' | 'pr'
export type Nudge = { kind: NudgeKind; text: string; atPrompt: number }

/** This session's live numbers. */
export type Live = {
  prompts: number
  streak: number
  contextPercent: number | null
  testsSinceCommit: number
  commits: number
}

declare module 'claude-code' {
  interface PluginState {
    'alto-belt': {
      scan: Feed<RepoScan>
      days: Record<string, Tally>
      history: ScorePoint[]
      declared: Declared
      /** PR number → reviewer → the day of each read; `?` holds reviews that named no PR. */
      rounds: Record<string, Record<string, string[]>>
      live: Live
      nudge: Nudge | null
      note: string | null
      reveal: number
    }
  }
}
