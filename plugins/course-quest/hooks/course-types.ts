// The shape of one course week, as surveyed from each week's own docs.

export type Concept = { name: string; explain: string }
export type Lab = { name: string; path: string; run: string; port?: number; mode?: string; what: string }
export type QuizItem = { q: string; options: string[]; answer: number; why: string }

export type Week = {
  id: string
  title: string
  phase: number
  emoji: string
  tagline: string
  summary: string
  concepts: Concept[]
  flow: { title: string; nodes: string[]; loop: boolean; caption: string }
  needs: string[]
  start: string
  labs: Lab[]
  quiz: QuizItem[]
  funFact: string
  readme: string
  minutes: number
}

export const PHASES: Record<number, { label: string; name: string }> = {
  1: { label: '① Foundation', name: 'Foundation: talk → tools → agents' },
  2: { label: '② Strength', name: 'Strength: single-agent mastery, tools & knowledge' },
  3: { label: '③ Endurance', name: 'Endurance: systems that run reliably' },
  4: { label: '④ Performance', name: 'Performance: memory, GraphRAG, frameworks, fleets' },
  5: { label: '⑤ Sovereignty', name: 'Sovereignty & self-improvement' },
  6: { label: '⑥ Open stack', name: 'The open stack, typed decisions & your own hardware' },
  7: { label: '⑦ Robotics', name: 'Special track: agentic robotics (SO-ARM101)' },
}
