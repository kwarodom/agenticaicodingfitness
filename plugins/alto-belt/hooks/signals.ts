// Pure classifiers: what a prompt, a shell command or a tool call says about how someone works.
// The word lists are the one place to tune. Nothing here keeps any text.

import type { Tally } from '../types'

// A correction: the person telling Claude the last answer missed. English and Thai.
// Matched near the start of the prompt, where a correction says so.
const CORRECTION_EN = [
  /^(no|nope|nah)\b[\s,.!]/i,
  /^(that'?s|this is|it'?s) (not|wrong|incorrect|broken)/i,
  /^(wrong|incorrect)\b/i,
  /\bnot what i (asked|meant|wanted)\b/i,
  /\bstill (fail|failing|fails|broken|wrong|not working|doesn'?t|the same|red)\b/i,
  /\b(doesn'?t|didn'?t|does not|did not) work\b/i,
  /\byou (broke|missed|forgot|ignored)\b/i,
  /^(try again|undo|revert (that|it|this))\b/i,
]
const CORRECTION_TH = [
  /^ไม่ใช่/,
  /^ผิด/,
  /ยังไม่(ได้|ทำงาน|หาย|ผ่าน|ถูก)/,
  /ยัง(พัง|เหมือนเดิม|error|เออเร่อ|ผิด)/,
  /ไม่ถูก(ต้อง)?/,
  /^(ลอง|แก้)ใหม่/,
  /ทำไมยัง/,
]

// A prompt that names the check that proves the work is done.
const VERIFIER = [
  /\bdone when\b/i,
  /\bprove (it|to me|that)\b/i,
  /\bshow (me )?(the )?(output|test output|results?)\b/i,
  /\buntil .{1,60}\b(pass|passes|green)\b/i,
  /\bmake sure .{1,60}\b(pass|passes)\b/i,
  /\b(run|rerun) the (tests|suite)\b/i,
  /เสร็จเมื่อ/,
  /พิสูจน์/,
  /(รัน|ทดสอบ).{0,20}(ผ่าน|เทส)/,
]

const HEAD = 160

const head = (text: string): string => text.trim().slice(0, HEAD)

export const isCorrection = (text: string): boolean => {
  const start = head(text)
  if (start === '' || start.startsWith('/')) return false
  return CORRECTION_EN.some(re => re.test(start)) || CORRECTION_TH.some(re => re.test(start))
}

export const namesVerifier = (text: string): boolean => VERIFIER.some(re => re.test(text))

// --- shell commands ----------------------------------------------------------------------------

const TEST = [
  /\b(py\.?test|python3? -m pytest)\b/,
  /\bmanage\.py test\b/,
  /\b(vitest|jest|playwright test|mocha)\b/,
  /\b(npm|pnpm|yarn|bun) (run )?test\b/,
  /\bmake (test|verify|check)\b/,
  /\b(go|cargo|dotnet|mix) test\b/,
  /\btox\b/,
  /\bclaude plugin test\b/,
  /run_[a-z_]*tests?\.sh\b/,
]

export const isTestCommand = (command: string): boolean => TEST.some(re => re.test(command))
export const isCommit = (command: string): boolean => /\bgit( -C \S+)? commit(\s|$)/.test(command)
export const isPrCreate = (command: string): boolean => /\bgh pr create\b/.test(command)

export type PrBody = { isSeen: false } | { isSeen: true; hasTesting: boolean; hasRisk: boolean; hasRevert: boolean }

/**
 * What a `gh pr create` command says of its body. A body given inline or through a heredoc is in
 * the command; one read from a file named on disk is not, and is reported unseen, never missing.
 */
export const prBodyOf = (command: string): PrBody => {
  const fromFile = /--body-file[= ]+(?!-(\s|$))\S+/.test(command)
  const inline = /--body[= ]|--body-file[= ]+-(\s|$)|<<-?\s*'?\w+'?/.test(command)
  if (fromFile || !inline) return { isSeen: false }
  return {
    isSeen: true,
    hasTesting: /(^|\n)\s*#+\s*(testing|tests?|verification|test plan)\b/i.test(command),
    hasRisk: /(^|\n)\s*#+\s*risks?\b/i.test(command),
    hasRevert: /(how to )?(revert|roll ?back)/i.test(command),
  }
}

export const missingSections = (body: PrBody): string[] =>
  body.isSeen === false
    ? []
    : [body.hasTesting ? null : 'Testing', body.hasRisk ? null : 'Risk', body.hasRevert ? null : 'How to revert'].filter(
        (one): one is string => one !== null,
      )

// --- tool calls ------------------------------------------------------------------------------

const REVIEW_SKILLS = /^(code-review|review|security-review|simplify|cso|codex)$/
const REVIEW_AGENTS = /(review|referee|security|trust|auditor)/i

/** Whether a tool call asks for a review in a fresh context. */
export const isReview = (tool: string, input: Record<string, unknown>): boolean => {
  if (tool === 'Skill') return REVIEW_SKILLS.test(String(input.skill ?? '').split(':').pop() ?? '')
  if (tool === 'Agent' || tool === 'Task') return REVIEW_AGENTS.test(String(input.subagent_type ?? ''))
  return false
}

export const isAgentSpawn = (tool: string): boolean => tool === 'Agent' || tool === 'Task'

// --- permissions -----------------------------------------------------------------------------

// A pre-approved command wide enough to run anything: an interpreter, a shell, any GitHub call.
const BROAD = [
  /^Bash$/,
  /^Bash\(\*\)$/,
  /^Bash\((python3?|node|bash|sh|zsh|ruby|perl|deno|bun) (\*|-)\)$/,
  /^Bash\([^)]*python3? -c '? ?\*\)$/,
  /^Bash\(gh (api|repo) \*\)$/,
  /^Bash\((rm|sudo|curl|eval) \*\)$/,
]

export const isBroadApproval = (rule: string): boolean => BROAD.some(re => re.test(rule.trim()))

// --- counting --------------------------------------------------------------------------------

export const EMPTY_TALLY: Tally = {
  prompts: 0,
  verifierPrompts: 0,
  corrections: 0,
  thirdStrikes: 0,
  strikeClears: 0,
  clears: 0,
  manualCompacts: 0,
  turns: 0,
  fullTurns: 0,
  testRuns: 0,
  commits: 0,
  testedCommits: 0,
  prs: 0,
  prsSeen: 0,
  prsEvidenced: 0,
  reviews: 0,
  agentSpawns: 0,
}

export const added = (tally: Tally, change: Partial<Tally>): Tally => {
  const out: Tally = { ...tally }
  for (const [key, value] of Object.entries(change) as [keyof Tally, number][]) out[key] = (out[key] ?? 0) + value
  return out
}

export const summed = (tallies: Tally[]): Tally => tallies.reduce((sum, one) => added(sum, one), EMPTY_TALLY)

export const dayOf = (at: number): string => {
  const d = new Date(at)
  const pad = (n: number): string => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

/** The days kept: the last `keep` by date. */
export const trimmed = (days: Record<string, Tally>, keep: number): Record<string, Tally> =>
  Object.fromEntries(
    Object.entries(days)
      .sort(([a], [b]) => (a < b ? 1 : -1))
      .slice(0, keep),
  )

/** The tallies of the last `span` days up to `today`, inclusive. */
export const windowOf = (days: Record<string, Tally>, today: string, span: number): Tally => {
  const end = new Date(`${today}T12:00:00`)
  const start = new Date(end.getTime() - (span - 1) * 86_400_000)
  const from = dayOf(start.getTime())
  return summed(Object.entries(days).filter(([day]) => day >= from && day <= today).map(([, tally]) => tally))
}
