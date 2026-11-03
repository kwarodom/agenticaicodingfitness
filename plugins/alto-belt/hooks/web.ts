// The belt web page: a local page the mod writes and the person opens in a browser. The page is
// static (web/index.html); the mod rewrites data.js beside it as the score changes, and the page
// reloads that file every few seconds. Nothing leaves the machine.

import type { Assessment, Live, Nudge, RepoScan, ScorePoint } from '../types'
import { nameOf } from './score'

export const BELT_IDS = ['white', 'green', 'blue', 'brown', 'black'] as const

export type WebData = {
  repo: string
  updatedAt: number
  belt: Assessment['belt']
  total: number
  measured: number
  gate: string
  next: string[]
  scores: { id: string; name: string; score: number | null; evidence: string; next: string }[]
  live: Live
  nudge: Nudge | null
  history: ScorePoint[]
}

export const webDataOf = (assessment: Assessment, scan: RepoScan, live: Live, nudge: Nudge | null, history: ScorePoint[], now: number): WebData => ({
  repo: scan.name,
  updatedAt: now,
  belt: assessment.belt,
  total: assessment.total,
  measured: assessment.measured,
  gate: assessment.gate,
  next: assessment.next,
  scores: assessment.scores.map(one => ({ id: one.id, name: nameOf(one.id), score: one.score, evidence: one.evidence, next: one.next })),
  live,
  nudge,
  history,
})

/** The script the page loads: the data as one assignment. `<` is escaped so no text can close the tag. */
export const dataScript = (data: WebData): string => `window.__BELT_DATA__ = ${JSON.stringify(data).replace(/</g, '\\u003c')};\n`

export const assetsScript = (art: Record<string, string>): string => `window.__BELT_ART__ = ${JSON.stringify(art)};\n`

export const webDirOf = (home: string, key: string): string => `${home.replace(/\/$/, '')}/.claude/alto-belt/${key}`
