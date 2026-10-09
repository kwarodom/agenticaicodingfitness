export type GymTab = 'lesson' | 'events' | 'progress'

/** One line of a check's outcome, drawn under the lesson. */
export type CheckOutcome = { label: string; isPassed: boolean; detail: string }

/** One event the gym saw pass through its own hooks. */
export type TapeEntry = {
  at: number
  event: string
  detail: string
  pattern: 'observe' | 'wrap' | 'answer'
}

/** What lasts across sessions (kept in $.store under "progress"). */
export type GymProgress = {
  lesson: number
  passed: Record<string, string>
  days: string[]
}

declare module 'claude-code' {
  interface PluginState {
    'mod-gym': {
      lesson: number
      tab: GymTab
      checks: CheckOutcome[]
      isChecking: boolean
      tape: TapeEntry[]
      passed: Record<string, string>
      isBandHidden: boolean
      labDir: string
    }
  }
}
