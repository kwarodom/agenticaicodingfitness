// The /belt pane and the line above the prompt. Text is wrapped whole, never cut.

import type { Elements } from 'claude-code'

import type { Assessment, Belt, Competency, Feed, Live, Nudge, RepoScan, ScorePoint } from '../types'
import { BELT_NAMES } from './score'

/** What each competency means, in words a newcomer reads at a glance. */
export const PLAIN: Record<Competency, string> = {
  verification: 'Prove it works',
  context: 'Keep context clean',
  instructions: 'Lean instructions',
  automation: 'Rules that enforce themselves',
  parallel: 'Parallel, but capped',
  multiagent: 'Agents with evidence',
  review: 'Review that converges',
  safety: 'Safe autonomy',
}

const LADDER: Belt[] = ['white', 'green', 'blue', 'brown', 'black']
const ORDER: Belt[] = ['none', 'white', 'green', 'blue', 'brown', 'black']

/** The score the next belt needs from one competency. */
export const nextNeed = (belt: Belt, id: Competency): number =>
  belt === 'none' ? 1 : belt === 'white' ? 2 : belt === 'green' ? 3 : belt === 'blue' ? (id === 'review' ? 4 : 3) : 4

export type Ui = Pick<Elements['terminal'], 'Box' | 'Text' | 'Button'>

// Theme keys, one meaning each, never a colour alone.
const TONE = { brand: 'claude', good: 'success', caution: 'warning', bad: 'error', quiet: 'inactive' } as const
const BELT_TONE: Record<string, string> = { none: TONE.quiet, white: TONE.quiet, green: TONE.good, blue: 'suggestion', brown: TONE.caution, black: TONE.brand }
const LABEL = 24

export const wrap = (text: string, width: number): string[] => {
  const lines: string[] = []
  let line = ''
  for (const word of text.split(/\s+/).filter(one => one !== '')) {
    const longer = line === '' ? word : `${line} ${word}`
    if ([...longer].length <= width || line === '') {
      line = longer
      continue
    }
    lines.push(line)
    line = word
  }
  if (line !== '') lines.push(line)
  return lines
}

const Lines = (ui: Ui, words: string, width: number, style: { dim?: boolean; color?: string; indent?: string } = {}) => {
  const { Box, Text } = ui
  const indent = style.indent ?? ''
  return (
    <Box flexDirection="column">
      {wrap(words, Math.max(12, width - indent.length)).map(line => (
        <Text dimColor={style.dim === true} color={style.color}>{`${indent}${line}`}</Text>
      ))}
    </Box>
  )
}

const Heading = (ui: Ui, text: string) => {
  const { Box, Text } = ui
  return (
    <Box flexDirection="row">
      <Text color={TONE.brand}>▍</Text>
      <Text bold>{text}</Text>
    </Box>
  )
}

/** Four cells; `shown` cells of the score drawn so far, so the bar fills in when the pane opens. */
export const bar = (score: number | null, shown = 4): string => {
  if (score === null) return '····'
  const drawn = Math.max(0, Math.min(score, shown))
  return '█'.repeat(drawn) + '░'.repeat(4 - drawn)
}

const scoreTone = (score: number | null): string => (score === null ? TONE.quiet : score >= 3 ? TONE.good : score === 2 ? TONE.caution : TONE.bad)

export type PaneReads = {
  scan: Feed<RepoScan>
  assessment: Assessment | null
  live: Live
  history: ScorePoint[]
  note: string | null
  /** How many cells of each bar are drawn so far, 0 to 4. */
  reveal: number
}

export type PaneHandlers = { refresh: () => void; checkin: () => void; web: () => void; close: () => void }

export const trendText = (history: ScorePoint[]): string =>
  history.length === 0 ? 'No earlier days yet.' : history.slice(-8).map(point => `${point.day.slice(5)} ${point.total}`).join('  →  ')

export const BeltPane = (ui: Ui, reads: PaneReads, columns: number, on: PaneHandlers) => {
  const { Box, Text, Button } = ui
  const width = Math.max(30, columns - 2)
  const { assessment, scan } = reads
  const buttons = (
    <Box flexDirection="row" gap={2}>
      <Button key="web" plain hotkey="w" variant="primary" label="Open in browser" onPress={on.web} />
      <Button key="refresh" plain hotkey="r" label="Refresh" onPress={on.refresh} />
      <Button key="checkin" plain hotkey="c" label="Friday check-in" onPress={on.checkin} />
      <Button key="close" plain hotkey="q" role="dismiss" label="Close" onPress={on.close} />
    </Box>
  )
  if (assessment === null || scan.data === null) {
    return (
      <Box flexDirection="column" gap={1}>
        {Heading(ui, 'Alto Belt')}
        {Lines(ui, scan.error === null ? 'Reading this repository…' : `Could not read this repository: ${scan.error}`, width, { dim: true })}
        {buttons}
      </Box>
    )
  }
  const unmeasured = 8 - assessment.measured
  const live = reads.live
  const at = ORDER.indexOf(assessment.belt)
  const ladder = (
    <Box flexDirection="row" flexWrap="wrap">
      {LADDER.map((belt, i) => {
        const index = i + 1
        const isHere = index === at
        const isEarned = index < at
        const isNext = index === at + 1
        return (
          <Box flexDirection="row">
            {i > 0 && <Text dimColor>{isEarned || isHere ? ' ━ ' : ' ┄ '}</Text>}
            <Text bold={isHere || isNext} color={isHere ? BELT_TONE[belt] : isEarned ? TONE.good : isNext ? TONE.caution : TONE.quiet} dimColor={!isHere && !isEarned && !isNext}>
              {`${isHere ? '● ' : isEarned ? '✓ ' : isNext ? '→ ' : '○ '}${BELT_NAMES[belt]}`}
            </Text>
          </Box>
        )
      })}
    </Box>
  )
  return (
    <Box flexDirection="column" gap={1}>
      <Box flexDirection="column">
        {Heading(ui, `Alto Belt · ${scan.data.name}`)}
        <Box flexDirection="row" gap={1}>
          <Text bold color={BELT_TONE[assessment.belt]}>
            {assessment.belt === 'none' ? BELT_NAMES.none : `${BELT_NAMES[assessment.belt]} belt`}
          </Text>
          <Text>{`· ${assessment.total}/32`}</Text>
          {unmeasured > 0 && <Text dimColor>{`· ${unmeasured} not measured`}</Text>}
        </Box>
        {Lines(ui, assessment.gate, width, { dim: true })}
      </Box>

      {ladder}

      <Box flexDirection="column">
        {assessment.scores.map(one => {
          const need = nextNeed(assessment.belt, one.id)
          const verdict = one.score === null ? 'not measured yet' : one.score >= need ? 'on track' : `next belt needs ${need}`
          return (
            <Box flexDirection="column">
              <Box flexDirection="row">
                <Text>{PLAIN[one.id].padEnd(LABEL + 6)}</Text>
                <Text color={scoreTone(one.score)}>{bar(one.score, reads.reveal)}</Text>
                <Text bold>{` ${one.score === null ? '–' : one.score}`}</Text>
                <Text color={one.score !== null && one.score < need ? TONE.caution : TONE.quiet}>{`  ${verdict}`}</Text>
              </Box>
              {Lines(ui, one.evidence, width, { dim: true, indent: '  ' })}
            </Box>
          )
        })}
      </Box>

      <Box flexDirection="column">
        {Heading(ui, 'Your next three moves')}
        {assessment.next.map((line, i) => Lines(ui, `${i + 1}. ${line}`, width))}
      </Box>

      <Box flexDirection="column">
        {Heading(ui, 'This session')}
        {Lines(
          ui,
          `${live.prompts} prompts · correction streak ${live.streak} · context ${live.contextPercent === null ? 'not read yet' : `${live.contextPercent}%`} · ${live.commits} commits · ${live.testsSinceCommit} test runs since the last commit`,
          width,
          { dim: true },
        )}
        {Lines(ui, `Score by day: ${trendText(reads.history)}`, width, { dim: true })}
      </Box>

      {reads.note !== null && Lines(ui, reads.note, width, { color: TONE.caution })}
      {Lines(ui, 'Counts stay on this machine; no prompt text is kept. /belt declare <container|field|cost> for what the mod cannot see.', width, { dim: true })}
      {buttons}
    </Box>
  )
}

export const NudgeLine = (ui: Ui, nudge: Nudge, columns: number) => {
  const { Box, Text } = ui
  const tone = nudge.kind === 'strike3' ? TONE.bad : TONE.caution
  return (
    <Box flexDirection="row">
      <Text color={tone}>◆ </Text>
      <Box flexDirection="column">
        {wrap(`belt · ${nudge.text}`, Math.max(16, columns - 4)).map(part => (
          <Text color={tone}>{part}</Text>
        ))}
      </Box>
    </Box>
  )
}
