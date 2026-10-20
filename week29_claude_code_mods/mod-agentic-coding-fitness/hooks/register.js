// Agentic Coding Fitness — progress-tracking mod (hooks module)
//
// Written against the Claude Code mods API as documented for v2.1.287+ (Oct 2026).
// The API can change between releases: after the first `claude --plugin-dir`, trust the
// generated `.claude-plugin/types/claude-code/index.d.ts` over this file where they differ.
//
// What it does
//   /fitness                 open the progress pane (checklist by week, streak, context use)
//   /fitness-log W SKILL S   log a lab manually, e.g. `/fitness-log 05 agent-loops 0.86`
//   /fitness-grade TEXT      ask a small model to grade a 2–3 sentence kata reflection
//   /fitness-next            print the next unfinished lab
//   tool  log_lab            Claude can call mcp__agentic-coding-fitness__log_lab itself
//   auto-log                 a `pytest`/`uv run pytest` command mentioning weekNN that passes is logged
//   spinner suffix           "Thinking · W05 · 7/26 labs…"
//   first prompt             adds a one-line progress summary to Claude's context

// ---------- curriculum: edit to match the class week table ----------
const CURRICULUM = [
  ['02', 'llm-fundamentals'],
  ['03', 'models-and-patterns'],
  ['04', 'tool-use'],
  ['05', 'agent-loops'],
  ['06', 'agent-memory-graphs'],
  ['07', 'knowledge-graph-mastery'],
  ['08', 'rag-knowledge-agents'],
  ['09', 'multi-agent-systems'],
  ['10', 'mcp-and-skills'],
  ['11', 'a2a-protocol'],
  ['12', 'agent-evaluation'],
  ['13', 'production-and-observability'],
  ['14', 'long-running-and-distributed-agents'],
  ['15', 'skill-authoring'],
  ['16', 'self-evolving-agents'],
  ['17', 'sovereign-ai-edge'],
  ['18', 'nemo-agent-toolkit'],
  ['19', 'dgx-spark-sovereign-stack'],
  ['21', 'physical-ai-omniverse'],
  ['23', 'nvidia-open-stack'],
  ['24', 'jev-typed-decisions'],
  ['25', 'dgx-spark-playbooks'],
  ['26', 'nemoclaw-alto-reef'],
  ['R1', 'so-arm101-robotics'],
  ['X1', 'vibe-coding-and-security'],
  ['X2', 'curriculum-and-periodization'],
]
const PASS_THRESHOLD = 0.8
const PANE = 'fitness'
const PLUGIN = 'agentic-coding-fitness'
const LOG_TOOL = 'mcp__' + PLUGIN + '__log_lab'

// ---------- module state (reset on reload; durable copy lives in $.store) ----------
let progress = { labs: {}, sessions: [] } // labs: { "05/agent-loops": {score, at} }
let pendingLab = null // set by tool.call when a weekNN pytest run passes
let firstPromptSeen = false

// ---------- helpers ----------
const today = () => new Date().toISOString().slice(0, 10)

function streak(sessions) {
  const days = [...new Set(sessions)].sort().reverse()
  if (days.length === 0) return 0
  let n = 0
  let cursor = new Date(days[0] + 'T00:00:00Z')
  for (const d of days) {
    const diff = Math.round((cursor - new Date(d + 'T00:00:00Z')) / 86400000)
    if (diff > 1) break
    n += 1
    cursor = new Date(d + 'T00:00:00Z')
  }
  return n
}

function passedCount() {
  return Object.values(progress.labs).filter((l) => l.score >= PASS_THRESHOLD).length
}

function currentWeek() {
  const next = CURRICULUM.find(([w, s]) => !(progress.labs[w + '/' + s]?.score >= PASS_THRESHOLD))
  return next ? next[0] : 'done'
}

function summaryLine() {
  return 'W' + currentWeek() + ' · ' + passedCount() + '/' + CURRICULUM.length + ' labs · streak ' + streak(progress.sessions)
}

async function save($) {
  await $.store.set('progress', progress)
}

async function logLab($, week, skill, score) {
  const key = week + '/' + skill
  progress.labs[key] = { score, at: new Date().toISOString() }
  await save($)
  $.ui.invalidate('ui.render')
  $.ui.status(summaryLine())
  return key + ' → ' + score.toFixed(2) + (score >= PASS_THRESHOLD ? ' (pass)' : ' (below ' + PASS_THRESHOLD + ')')
}

function parseLogArgs(args) {
  const m = /^\s*(\S+)\s+(\S+)\s+([0-9.]+)\s*$/.exec(args || '')
  if (!m) return null
  const score = Number(m[3])
  if (!(score >= 0 && score <= 1)) return null
  return { week: m[1].padStart(2, '0'), skill: m[2], score }
}

// ---------- registration ----------
export function register(on) {
  // Runs before the first prompt and again after each reload of this module
  on('session.start', async ($, e, next) => {
    const saved = await $.store.get('progress')
    if (saved && typeof saved === 'object') progress = { labs: {}, sessions: [], ...saved }
    const d = today()
    if (!progress.sessions.includes(d)) {
      progress.sessions.push(d)
      await save($)
    }

    await $.command.register({ name: 'fitness', description: 'Open the Agentic Coding Fitness progress pane' })
    await $.command.register({
      name: 'fitness-log',
      description: 'Log a lab result',
      argumentHint: '<week> <skill> <score 0-1>',
    })
    await $.command.register({
      name: 'fitness-grade',
      description: 'Grade a short kata reflection with a small model',
      argumentHint: '<2-3 sentences>',
    })
    await $.command.register({ name: 'fitness-next', description: 'Show the next unfinished lab' })

    // A tool Claude can call when a guided-lab kata passes
    await $.tool.register({
      name: 'log_lab',
      description:
        'Record that the student completed a guided lab from the Agentic Coding Fitness curriculum. Call it once after a kata rubric is scored.',
      inputSchema: {
        type: 'object',
        properties: {
          week: { type: 'string', description: 'Two-digit week, e.g. "05"' },
          skill: { type: 'string', description: 'Skill directory name, e.g. "agent-loops"' },
          score: { type: 'number', description: 'Rubric score between 0 and 1' },
        },
        required: ['week', 'skill', 'score'],
      },
    })

    $.ui.status(summaryLine())
    return next(e)
  })

  // Claude called our tool
  on('tool.call', { tool: LOG_TOOL }, async ($, e) => {
    const score = Number(e.score)
    if (!(score >= 0 && score <= 1)) return { result: 'score must be between 0 and 1' }
    const line = await logLab($, String(e.week).padStart(2, '0'), String(e.skill), score)
    return { result: 'Logged ' + line }
  })

  // Auto-detect a passing weekNN pytest run. Observe → let the tool run → inspect the result.
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const cmd = e.command || ''
    const week = /week[_-]?(\d{1,2})/i.exec(cmd)
    const isTest = /\bpytest\b/.test(cmd)
    const r = await next(e)
    if (week && isTest) {
      const out = typeof r?.result === 'string' ? r.result : JSON.stringify(r?.result ?? '')
      const failed = /\b\d+ failed\b/.test(out)
      const passed = /\b\d+ passed\b/.test(out)
      if (passed && !failed) {
        const w = week[1].padStart(2, '0')
        const entry = CURRICULUM.find(([cw]) => cw === w)
        const skill = entry ? entry[1] : 'unknown'
        pendingLab = await logLab($, w, skill, 1.0)
      }
    }
    return r
  })

  // Show a line under Claude's answer when a lab was auto-logged during the turn
  on('turn.complete', async ($, e, next) => {
    const r = await next(e)
    if (pendingLab) {
      const text = 'Lab auto-logged: ' + pendingLab
      pendingLab = null
      return { text }
    }
    return r
  })

  // Give Claude a one-line view of the student's progress on the first prompt of the session
  on('prompt.submit', async ($, e, next) => {
    if (firstPromptSeen) return next(e)
    firstPromptSeen = true
    const ctx = 'Agentic Coding Fitness progress: ' + summaryLine() + '. Next lab: week ' + currentWeek() + '.'
    return next({ ...e, context: [...(e.context ?? []), ctx] })
  })

  // Commands
  on('command.run', { command: 'fitness' }, async ($) => {
    await $.ui.open({ id: PANE, title: 'Agentic Coding Fitness', focus: true, closeOnEscape: true })
    return {}
  })

  on('command.run', { command: 'fitness-log' }, async ($, e) => {
    const p = parseLogArgs(e.args)
    if (!p) return { text: 'Usage: /fitness-log <week> <skill> <score 0-1>, e.g. /fitness-log 05 agent-loops 0.86' }
    return { text: 'Logged ' + (await logLab($, p.week, p.skill, p.score)) }
  })

  on('command.run', { command: 'fitness-next' }, async () => {
    const w = currentWeek()
    if (w === 'done') return { text: 'All labs passed. Time to write a new skill.' }
    const [, skill] = CURRICULUM.find(([cw]) => cw === w)
    return { text: 'Next: week ' + w + ' → /' + PLUGIN + ':' + skill + ' (run its guided lab, then /fitness-log ' + w + ' ' + skill + ' <score>)' }
  })

  on('command.run', { command: 'fitness-grade' }, async ($, e) => {
    if (!e.args) return { text: 'Usage: /fitness-grade <your 2-3 sentence reflection on the lab>' }
    const r = await $.model.complete({
      model: 'haiku',
      system:
        'You grade a student reflection for an agentic-AI coding class. Reply on two lines: ' +
        'line 1 is PASS or REVISE; line 2 is one sentence of feedback. PASS requires a concrete mechanism ' +
        '(e.g. a loop, a tool, a memory, an eval) and one trade-off.',
      prompt: e.args,
      maxTokens: 80,
      timeoutMs: 15000,
    })
    if (!r.isAnswered) return { text: 'Grader unavailable (' + r.reason + '). Try again.' }
    return { text: r.text.trim() }
  })

  // Drawing: spinner suffix and the pane
  on('ui.render', { component: 'Spinner' }, async ($, e, next) => {
    return next({ ...e, props: { ...e.props, suffix: ' · ' + summaryLine() + '…' } })
  })

  on('ui.render', { component: 'Pane' }, async ($, e, next) => {
    if (e.requestId !== PANE) return next(e)
    const { Box, Text, Button } = $.ui.resolve(e)
    const usage = await $.session.usage()
    const ctxPct = usage?.context?.percent != null ? Math.round(usage.context.percent) + '%' : 'n/a'

    const rows = CURRICULUM.map(([w, s]) => {
      const lab = progress.labs[w + '/' + s]
      const mark = lab ? (lab.score >= PASS_THRESHOLD ? '✓' : '○') : '·'
      const score = lab ? ' ' + lab.score.toFixed(2) : ''
      return Text({ key: 'row-' + w, dimColor: !lab, children: [mark + ' W' + w + '  ' + s + score] })
    })

    return Box({
      flexDirection: 'column',
      children: [
        Text({ bold: true, children: [summaryLine() + ' · context ' + ctxPct] }),
        Text({ children: [' '] }),
        ...rows,
        Text({ children: [' '] }),
        Box({
          flexDirection: 'row',
          columnGap: 2,
          children: [
            Button({
              key: 'reset',
              label: 'Reset labs',
              hotkey: 'r',
              onPress: async () => {
                progress.labs = {}
                await save($)
                $.ui.invalidate('ui.render')
              },
            }),
            Button({ key: 'close', label: 'Close', hotkey: 'q', onPress: () => $.ui.close(PANE) }),
          ],
        }),
      ],
    })
  })
}
