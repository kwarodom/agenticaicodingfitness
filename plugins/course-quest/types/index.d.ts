export type QuestTab = 'map' | 'week' | 'labs' | 'quiz' | 'stats'

/** Kept in $.store under "progress": lasts across sessions on this machine. */
export type QuestProgress = {
  xp: number
  /** week id → first visit date */
  visited: Record<string, string>
  /** week id → indexes of labs marked done */
  labs: Record<string, number[]>
  /** week id → question index → answered right at least once */
  quiz: Record<string, Record<string, boolean>>
  days: string[]
}

declare module 'claude-code' {
  interface PluginState {
    'course-quest': {
      tab: QuestTab
      week: number
      lab: number
      quiz: number
      picked: number
      progress: QuestProgress
      isBandHidden: boolean
    }
  }
}
