// ⚡ Agentic Quest: the whole Agentic Coding Fitness course (Week 2 → Week 29 and the
// robotics track) as an animated, game-like panel inside Claude Code.
//
//   /quest                     open the panel: Map · Week · Labs · Quiz · Stats
//   /quest <id>                jump to a week, e.g. /quest 05 or /quest R1
//   /quest next|prev|map|labs|quiz|stats|band on|off|reset
//   tools course_week, course_overview   so Claude can answer and guide from the same data
//
// Animation: one clock ticks while the panel is open and repaints its Raster pictures
// (the banner, the metro map, a week's flow) with $.ui.blit, so nothing else redraws.

import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, RenderChildren, Timer } from 'claude-code'

import type { QuestProgress, QuestTab } from '../types'
import { WHITE, banner, bar, canvas, flow, hex, metro, mix, PHASE_COLOR, DIM } from './canvas'
import type { Canvas, MapLine } from './canvas'
import { WEEKS } from './course'
import { PHASES } from './course-types'
import type { Week } from './course-types'

const PANE = 'course-quest'
const TITLE = 'Agentic Quest'
const TICK_MS = 110

type $ = EngineInterface

const EMPTY: QuestProgress = { xp: 0, visited: {}, labs: {}, quiz: {}, days: [] }

const tabAtom = atom({ plugin: 'course-quest', key: 'tab' } as const, 'map')
const weekAtom = atom({ plugin: 'course-quest', key: 'week' } as const, 0)
const labAtom = atom({ plugin: 'course-quest', key: 'lab' } as const, 0)
const quizAtom = atom({ plugin: 'course-quest', key: 'quiz' } as const, 0)
const pickedAtom = atom({ plugin: 'course-quest', key: 'picked' } as const, -1)
const progressAtom = atom({ plugin: 'course-quest', key: 'progress' } as const, EMPTY)
const isBandHiddenAtom = atom({ plugin: 'course-quest', key: 'isBandHidden' } as const, false)

// ---------- animation (module state: it is fine for it to restart on a reload) ----------

let frame = 0
let party = 0
let partyFrom = 0
let cheer = ''

function celebrate(message: string) {
  party = 30
  partyFrom = frame
  cheer = message
}

/** An emoji only when it draws as one clean glyph (no text-style or variation-selector ones). */
function icon(emoji: string): string {
  const cp = emoji.codePointAt(0) ?? 0
  return cp > 0xffff && !emoji.includes('\ufe0f') ? emoji + ' ' : ''
}

/** What Bolt says on the banner, in turn, about 12 s each. */
function speeches(w: Week, p: QuestProgress, labsLeft: number, days: number): string[] {
  const list = [
    p.visited[w.id] ? `Welcome back to Week ${w.id}! Let's keep going.` : `Hi! Ready to explore Week ${w.id}?`,
    w.tagline,
    labsLeft > 0 ? `${labsLeft} lab${labsLeft === 1 ? '' : 's'} to go this week. You've got this!` : 'All labs done here. Try the quiz: press k',
    'Press t and Claude will teach you this week.',
    'Press g and Claude will walk you through the labs.',
  ]
  if (days > 1) list.push(`${days}-day streak! Keep the habit going.`)
  return list
}
const SPEECH_FRAMES = 110
let isOpen = false
let timer: Timer | undefined
const painters: Record<string, () => Canvas> = {}

function startAnimation($: $) {
  if (timer) return
  timer = $.clock.every(TICK_MS, () => {
    if (!isOpen) return
    frame += 1
    if (party > 0) party -= 1
    for (const [key, paint] of Object.entries(painters)) {
      const c = paint()
      void $.ui.blit({ requestId: PANE, key, cells: c.cells(), columns: c.columns, rows: c.rows }).catch(() => undefined)
    }
    if (frame % 40 === 0) {
      void $.ui.panes().then(
        panes => {
          isOpen = panes.some(p => p.id === PANE && p.isShown)
        },
        () => undefined,
      )
    }
  })
}

// ---------- course helpers ----------

const today = () => new Date().toISOString().slice(0, 10)

function weekAt(i: number): Week {
  return WEEKS[Math.max(0, Math.min(WEEKS.length - 1, i))] as Week
}

function findWeek(id: string): number {
  const want = id.trim().toUpperCase().replace(/^W(EEK)?\s*/, '')
  const padded = /^\d$/.test(want) ? '0' + want : want
  return WEEKS.findIndex(w => w.id === padded)
}

function level(xp: number) {
  const lvl = Math.floor(Math.sqrt(xp / 40)) + 1
  const floor = 40 * (lvl - 1) ** 2
  const next = 40 * lvl ** 2
  return { lvl, floor, next, fraction: (xp - floor) / (next - floor) }
}

function labsDone(p: QuestProgress, w: Week): number {
  return (p.labs[w.id] ?? []).filter(i => i < w.labs.length).length
}

function quizRight(p: QuestProgress, w: Week): number {
  return Object.values(p.quiz[w.id] ?? {}).filter(Boolean).length
}

function weekState(p: QuestProgress, w: Week): 'done' | 'started' | 'new' {
  const labsOk = w.labs.length === 0 || labsDone(p, w) * 2 >= w.labs.length
  if (labsOk && quizRight(p, w) >= w.quiz.length && w.quiz.length > 0) return 'done'
  if (p.visited[w.id] || labsDone(p, w) > 0 || quizRight(p, w) > 0) return 'started'
  return 'new'
}

function streak(days: readonly string[]): number {
  const sorted = [...new Set(days)].sort().reverse()
  let n = 0
  let cursor = sorted[0] ? Date.parse(sorted[0] + 'T00:00:00Z') : 0
  for (const d of sorted) {
    const t = Date.parse(d + 'T00:00:00Z')
    if (Math.round((cursor - t) / 86_400_000) > 1) break
    n += 1
    cursor = t
  }
  return n
}

function badges(p: QuestProgress): string[] {
  const out: string[] = []
  const labs = WEEKS.reduce((n, w) => n + labsDone(p, w), 0)
  const right = WEEKS.reduce((n, w) => n + quizRight(p, w), 0)
  if (Object.keys(p.visited).length > 0) out.push('👣 First steps')
  if (Object.keys(p.visited).length >= WEEKS.length) out.push('🧭 Explorer: visited every week')
  if (labs >= 1) out.push('🧪 First lab')
  if (labs >= 10) out.push('🔬 Lab rat: 10 labs')
  if (right >= 10) out.push('🧠 Quiz whiz: 10 right answers')
  if (streak(p.days) >= 3) out.push('🔥 3-day streak')
  for (const [phase, info] of Object.entries(PHASES)) {
    const weeks = WEEKS.filter(w => String(w.phase) === phase)
    if (weeks.length && weeks.every(w => weekState(p, w) === 'done')) out.push('🏅 ' + info.label + ' cleared')
  }
  return out
}

function nextUp(p: QuestProgress): Week | undefined {
  return WEEKS.find(w => weekState(p, w) !== 'done')
}

async function commit($: $, change: (p: QuestProgress) => QuestProgress): Promise<QuestProgress> {
  await update($, progressAtom, change)
  const p = await read($, progressAtom)
  await $.store.set('progress', p)
  const { lvl } = level(p.xp)
  $.ui.status('⚡ Quest Lv' + lvl + ' · W' + weekAt(await read($, weekAtom)).id)
  return p
}

async function goWeek($: $, i: number): Promise<Week> {
  const n = Math.max(0, Math.min(WEEKS.length - 1, i))
  const w = weekAt(n)
  await update($, weekAtom, () => n)
  await update($, labAtom, () => 0)
  await update($, quizAtom, () => 0)
  await update($, pickedAtom, () => -1)
  await commit($, p => (p.visited[w.id] ? p : { ...p, xp: p.xp + 5, visited: { ...p.visited, [w.id]: today() } }))
  return w
}

async function openPane($: $, tab?: QuestTab): Promise<void> {
  if (tab) await update($, tabAtom, () => tab)
  isOpen = true
  startAnimation($)
  await $.ui.open({ id: PANE, title: '⚡ ' + TITLE, focus: true, closeOnEscape: true })
}

function teachPrompt(w: Week): string {
  return [
    `Teach me Week ${w.id} of the Agentic Coding Fitness course: ${w.emoji} ${w.title}.`,
    `Read ${w.readme} (and the files it points to) before you answer.`,
    `Explain these ideas, each with one tiny concrete example from this repo: ${w.concepts.map(c => c.name).join('; ')}.`,
    `Then walk me through this flow: ${w.flow.nodes.join(' → ')}${w.flow.loop ? ' → (back to the start)' : ''}.`,
    'Keep it short and fun, use a small ASCII diagram, and end with one question for me.',
  ].join('\n')
}

function guidePrompt(w: Week, lab?: number): string {
  const labs = lab === undefined ? w.labs : w.labs.slice(lab, lab + 1)
  return [
    `Guide me through running ${lab === undefined ? 'the labs of' : 'one lab from'} Week ${w.id} (${w.title}) in this repo, step by step.`,
    w.needs.length ? `First check I have what it needs: ${w.needs.join(', ')}. Tell me how to fix anything missing.` : 'It has no special prerequisites; confirm that quickly.',
    lab === undefined && w.start ? `Start with: ${w.start}` : '',
    ...labs.map((l, i) => `${i + 1}. ${l.name}: ${l.run || 'read ' + l.path}${l.port ? ` (web UI on http://127.0.0.1:${l.port})` : ''} — ${l.what}`),
    'For each step, say what I should expect to see. Ask before you run anything that starts a server or costs money.',
  ]
    .filter(Boolean)
    .join('\n')
}

// ---------- registration ----------

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const saved = (await $.store.get('progress')) as Partial<QuestProgress> | undefined
    const p: QuestProgress = {
      xp: typeof saved?.xp === 'number' ? saved.xp : 0,
      visited: saved?.visited ?? {},
      labs: saved?.labs ?? {},
      quiz: saved?.quiz ?? {},
      days: Array.isArray(saved?.days) ? saved.days : [],
    }
    if (!p.days.includes(today())) p.days = [...p.days, today()].slice(-400)
    await update($, progressAtom, () => p)
    await $.store.set('progress', p)
    const savedWeek = await $.store.get('week')
    if (typeof savedWeek === 'number') await update($, weekAtom, () => Math.max(0, Math.min(WEEKS.length - 1, savedWeek)))
    if ((await $.store.get('isBandHidden')) === true) await update($, isBandHiddenAtom, () => true)

    await $.command.register({
      name: 'quest',
      description: 'Agentic Quest: explore every week of the course, run its labs, take its quiz',
      argumentHint: '[<week id>|map|labs|quiz|stats|next|prev|band on|off|reset]',
    })
    await $.tool.register({
      name: 'course_week',
      description:
        'Facts about one week of the Agentic Coding Fitness course in this repo: title, summary, concepts, prerequisites, the start command, every lab with its exact run command and port, quiz, and the learner\'s progress. Call it before explaining a week or guiding the learner through its labs. Week ids: "02".."29" (no 12, 13, 20, 22) and "R1" for robotics.',
      inputSchema: { type: 'object', properties: { week: { type: 'string', description: 'e.g. "05", "15", "R1"' } }, required: ['week'] },
    })
    await $.tool.register({
      name: 'course_overview',
      description: 'The whole Agentic Coding Fitness course at a glance: every week with its phase, title, tagline and the learner\'s state (new / started / done), plus XP and level.',
      inputSchema: { type: 'object', properties: {} },
    })
    $.ui.status('⚡ Quest Lv' + level(p.xp).lvl + ' · W' + weekAt(await read($, weekAtom)).id)
    startAnimation($)
    return next(e)
  })

  // ----- tools Claude calls -----

  on('tool.call', { tool: 'mcp__course-quest__course_week' }, async ($, e) => {
    const i = findWeek(String(e.week ?? ''))
    if (i < 0) return { result: 'No week "' + String(e.week) + '". Ids: ' + WEEKS.map(w => w.id).join(', ') }
    const w = weekAt(i)
    const p = await read($, progressAtom)
    const progress = {
      state: weekState(p, w),
      labsDone: (p.labs[w.id] ?? []).map(n => w.labs[n]?.name).filter(Boolean),
      quizRight: quizRight(p, w) + '/' + w.quiz.length,
    }
    return { result: JSON.stringify({ ...w, phaseName: PHASES[w.phase]?.name, progress }, null, 2) }
  }).catch(() => ({ result: 'Course data is unavailable right now.' }))

  on('tool.call', { tool: 'mcp__course-quest__course_overview' }, async $ => {
    const p = await read($, progressAtom)
    const { lvl } = level(p.xp)
    const lines = WEEKS.map(w => `${w.id} ${w.emoji} ${w.title} [${PHASES[w.phase]?.label ?? ''}] ${weekState(p, w)} — ${w.tagline}`)
    return { result: `XP ${p.xp} · level ${lvl} · next up: ${nextUp(p)?.id ?? 'all done'}\n` + lines.join('\n') }
  }).catch(() => ({ result: 'Course data is unavailable right now.' }))

  // ----- the command -----

  on('command.run', { command: 'quest' }, async ($, e) => {
    const [verb = '', ...rest] = e.args.trim().split(/\s+/)
    const v = verb.toLowerCase()
    const tabs: Record<string, QuestTab> = { map: 'map', week: 'week', labs: 'labs', quiz: 'quiz', stats: 'stats' }
    if (v === '' || v === 'open') {
      await openPane($)
      return {}
    }
    const tab = tabs[v]
    if (tab) {
      await openPane($, tab)
      return {}
    }
    if (v === 'next' || v === 'prev') {
      const w = await goWeek($, (await read($, weekAtom)) + (v === 'next' ? 1 : -1))
      await $.store.set('week', await read($, weekAtom))
      await openPane($, 'week')
      return { text: `Week ${w.id} ${w.emoji} ${w.title}` }
    }
    if (v === 'band') {
      const isHidden = rest[0] === 'off'
      await update($, isBandHiddenAtom, () => isHidden)
      await $.store.set('isBandHidden', isHidden)
      return { text: 'Quest band ' + (isHidden ? 'hidden' : 'shown') }
    }
    if (v === 'reset') {
      await commit($, () => ({ ...EMPTY, days: [today()] }))
      return { text: 'Quest progress reset. Fresh start! ⚡' }
    }
    const i = findWeek(verb)
    if (i >= 0) {
      const w = await goWeek($, i)
      await $.store.set('week', i)
      await openPane($, 'week')
      return { text: `Week ${w.id} ${w.emoji} ${w.title}` }
    }
    return { text: 'Usage: /quest [<week id e.g. 05 or R1>|map|labs|quiz|stats|next|prev|band on|off|reset]' }
  })

  // ----- the band above the prompt -----

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey || (await read($, isBandHiddenAtom))) return next(e)
    const { Box, Text, Button } = $.ui.resolve(e)
    const p = await read($, progressAtom)
    const w = weekAt(await read($, weekAtom))
    const { lvl } = level(p.xp)
    const below = await next(e)
    return (
      <Box flexDirection="column" width={e.props.bodyColumns}>
        {below}
        <Box flexDirection="row" columnGap={1}>
          <Box flexGrow={1} flexShrink={1} overflow="hidden">
            <Text wrap="truncate-end">
              <Text color={hex(PHASE_COLOR[w.phase] ?? WHITE)} bold>
                ⚡ Quest
              </Text>
              <Text dimColor>
                {' '}Lv {lvl} · {p.xp} xp · W{w.id} {icon(w.emoji)}{w.title}
              </Text>
            </Text>
          </Box>
          <Button key="quest-open" onPress={() => openPane($)}>
            Open
          </Button>
          <Button
            key="quest-hide"
            dimColor
            onPress={async () => {
              await update($, isBandHiddenAtom, () => true)
              await $.store.set('isBandHidden', true)
            }}
          >
            Hide
          </Button>
        </Box>
      </Box>
    )
  })

  // ----- the panel -----

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button, Raster, Select, Code, Link } = $.ui.resolve({ ...e, surface: 'terminal' as const })
    isOpen = true
    const width = Math.max(40, Math.min(e.props.bodyColumns, 140))
    const tab = (await read($, tabAtom)) as QuestTab
    const wi = await read($, weekAtom)
    const w = weekAt(wi)
    const p = await read($, progressAtom)
    const color = PHASE_COLOR[w.phase] ?? WHITE
    const { lvl, fraction } = level(p.xp)
    for (const key of Object.keys(painters)) delete painters[key]

    const paint = (key: string, make: () => Canvas, rows?: number) => {
      painters[key] = make
      const c = make()
      return <Raster key={key} columns={c.columns} rows={rows ?? c.rows} cells={c.cells()} />
    }

    const lines = speeches(w, p, w.labs.length - labsDone(p, w), streak(p.days))
    const head = paint('banner', () => {
      const turn = Math.floor(frame / SPEECH_FRAMES)
      return banner(width, {
        frame,
        phase: w.phase,
        label: `WEEK ${w.id} · ${(PHASES[w.phase]?.label ?? '').replace(/^\S+\s/, '').toUpperCase()}`,
        title: w.title,
        speech: party > 0 ? cheer : (lines[turn % lines.length] ?? ''),
        speechFrom: party > 0 ? partyFrom : turn * SPEECH_FRAMES,
        level: lvl,
        xp: p.xp,
        fraction,
        streak: streak(p.days),
        party,
      })
    })

    const tabButton = (id: QuestTab, hotkey: string, label: string) => (
      <Button key={'tab-' + id} hotkey={hotkey} variant={tab === id ? 'primary' : 'secondary'} onPress={() => update($, tabAtom, () => id)}>
        {label}
      </Button>
    )
    const tabs = (
      <Box flexDirection="row" columnGap={1} flexWrap="wrap">
        {tabButton('map', 'm', 'Map')}
        {tabButton('week', 'w', 'Week')}
        {tabButton('labs', 'l', 'Labs')}
        {tabButton('quiz', 'k', 'Quiz')}
        {tabButton('stats', 's', 'Stats')}
        <Button key="prev" hotkey="p" onPress={() => goWeek($, wi - 1)}>
          ◀
        </Button>
        <Button key="next" hotkey="n" onPress={() => goWeek($, wi + 1)}>
          ▶
        </Button>
      </Box>
    )
    const weekTitle = (
      <Text wrap="truncate-end">
        <Text color={hex(color)} bold>
          {icon(w.emoji)}Week {w.id} · {w.title}
        </Text>
        <Text dimColor> {PHASES[w.phase]?.label}</Text>
      </Text>
    )
    const footer = (
      <Box flexDirection="row" columnGap={1} flexWrap="wrap">
        <Button key="teach" hotkey="t" variant="primary" onPress={() => $.prompt.fill({ text: teachPrompt(w), mode: 'replace' }).then(() => $.ui.toast('📚 Lesson request is in your prompt: press Enter'))}>
          Teach me
        </Button>
        <Button key="guide" hotkey="g" onPress={() => $.prompt.fill({ text: guidePrompt(w), mode: 'replace' }).then(() => $.ui.toast('🧭 Lab guide request is in your prompt: press Enter'))}>
          Guide my labs
        </Button>
        <Button key="close" hotkey="x" role="dismiss" onPress={() => $.ui.close({ id: PANE })}>
          Close
        </Button>
      </Box>
    )
    const shell = (body: RenderChildren) => (
      <Box flexDirection="column" width={width} rowGap={1}>
        {head}
        {tabs}
        {body}
        {footer}
      </Box>
    )

    if (tab === 'map') {
      const lines: MapLine[] = Object.entries(PHASES)
        .map(([phase, info]) => ({
          phase: Number(phase),
          label: info.label,
          stations: WEEKS.filter(x => x.phase === Number(phase)).map(x => ({ id: x.id, state: weekState(p, x) })),
        }))
        .filter(l => l.stations.length > 0)
      const up = nextUp(p)
      return shell(
        <Box flexDirection="column" rowGap={1}>
          <Text bold>The course map</Text>
          {paint('map', () => metro(width, lines, w.id, frame))}
          <Text dimColor wrap="wrap">
            ● done  ◐ started  ○ new  ◉ you are here
          </Text>
          <Select
            key="pick-week"
            label="Jump to a week"
            value={w.id}
            options={WEEKS.map(x => ({ value: x.id, label: `${x.id} ${icon(x.emoji)}${x.title}` }))}
            onSelect={async value => {
              await goWeek($, findWeek(value))
              await $.store.set('week', findWeek(value))
              await update($, tabAtom, () => 'week')
            }}
          />
          {up && (
            <Text wrap="wrap">
              <Text color="success">Next up: </Text>
              {icon(up.emoji)}Week {up.id} · {up.title} <Text dimColor>({up.tagline})</Text>
            </Text>
          )}
        </Box>,
      )
    }

    if (tab === 'week') {
      return shell(
        <Box flexDirection="column" rowGap={1}>
          {weekTitle}
          <Text italic color={hex(mix(color, WHITE, 0.4))} wrap="wrap">
            “{w.tagline}”
          </Text>
          <Text wrap="wrap">{w.summary}</Text>
          <Box flexDirection="column">
            <Text bold>{w.flow.title}</Text>
            {paint('flow', () => flow(width - 2, w.flow.nodes, w.flow.loop, color, frame))}
            <Text dimColor wrap="wrap">
              {w.flow.caption}
            </Text>
          </Box>
          <Box flexDirection="row" flexWrap="wrap" columnGap={1} rowGap={0}>
            {w.concepts.map((c, n) => (
              <Box key={'concept-' + n} flexDirection="column" borderStyle="round" borderColor={hex(mix(color, 0x000000, 0.2))} paddingX={1} width={Math.max(28, Math.floor((width - 2) / 2) - 1)}>
                <Text bold color={hex(color)}>
                  {c.name}
                </Text>
                <Text wrap="wrap">{c.explain}</Text>
              </Box>
            ))}
          </Box>
          {w.needs.length > 0 && (
            <Text wrap="wrap">
              <Text bold>🧰 Needs: </Text>
              {w.needs.join(' · ')}
            </Text>
          )}
          <Text wrap="wrap">
            <Text bold>💡 Fun fact: </Text>
            {w.funFact}
          </Text>
          <Text dimColor wrap="wrap">
            📖 {w.readme} · ⏱ ~{w.minutes} min · {w.labs.length} labs · {w.quiz.length} quiz questions
          </Text>
        </Box>,
      )
    }

    if (tab === 'labs') {
      const li = Math.min(await read($, labAtom), Math.max(0, w.labs.length - 1))
      const lab = w.labs[li]
      const done = new Set(p.labs[w.id] ?? [])
      const doneCount = labsDone(p, w)
      return shell(
        <Box flexDirection="column" rowGap={1}>
          {weekTitle}
          {paint('labbar', () => {
            const c = canvas(Math.min(width, 60), 1)
            const n = c.text(0, 0, `${doneCount}/${w.labs.length} labs `, WHITE)
            bar(c, n + 1, 0, Math.max(5, c.columns - n - 2), w.labs.length ? doneCount / w.labs.length : 0, mix(color, 0x000000, 0.3), mix(color, WHITE, 0.4), frame)
            return c
          })}
          {w.start && (
            <Box flexDirection="column">
              <Text bold>▶ Start here</Text>
              <Code source={w.start} language="bash" />
            </Box>
          )}
          <Box flexDirection="column">
            {w.labs.map((l, n) => (
              <Text key={'lab-' + n} wrap="truncate-end" dimColor={n !== li}>
                <Text color={done.has(n) ? 'success' : n === li ? hex(color) : 'inactive'}>{done.has(n) ? '✓' : n === li ? '▸' : '○'}</Text> {n + 1}. {l.name}
                {l.port ? <Text color="suggestion"> :{l.port}</Text> : ''}
                {l.mode ? <Text dimColor> · {l.mode}</Text> : ''}
              </Text>
            ))}
          </Box>
          {w.labs.length > 0 && (
            <Select
              key="pick-lab"
              label="Lab"
              value={String(li)}
              options={w.labs.map((l, n) => ({ value: String(n), label: `${n + 1}. ${l.name}` }))}
              onSelect={value => update($, labAtom, () => Number(value))}
            />
          )}
          {lab && (
            <Box flexDirection="column" borderStyle="round" borderColor={hex(color)} paddingX={1}>
              <Text bold color={hex(color)}>
                {li + 1}. {lab.name}
              </Text>
              <Text wrap="wrap">{lab.what}</Text>
              <Text dimColor wrap="truncate-middle">
                📁 {lab.path}
              </Text>
              {lab.run ? <Code source={lab.run} language="bash" /> : <Text dimColor>Reading lab: open the file above.</Text>}
              {lab.port ? (
                <Link href={'http://127.0.0.1:' + lab.port} label={'🌐 open http://127.0.0.1:' + lab.port} />
              ) : (
                ''
              )}
            </Box>
          )}
          {lab && (
            <Box flexDirection="row" columnGap={1} flexWrap="wrap">
              <Button
                key="copy"
                hotkey="c"
                dimColor={!lab.run}
                onPress={async press => {
                  if (!lab.run) return
                  const r = await $.ui.copy({ text: lab.run, surface: press.surface })
                  $.ui.toast(r.isCopied ? '📋 Copied: ' + lab.run : 'Could not copy here')
                }}
              >
                Copy command
              </Button>
              <Button key="run" hotkey="r" onPress={() => $.prompt.fill({ text: guidePrompt(w, li), mode: 'replace' }).then(() => $.ui.toast('🚀 Ask is in your prompt: press Enter'))}>
                Run with Claude
              </Button>
              <Button
                key="done"
                hotkey="d"
                variant={done.has(li) ? 'secondary' : 'primary'}
                onPress={async () => {
                  const isDone = done.has(li)
                  await commit($, q => {
                    const set = new Set(q.labs[w.id] ?? [])
                    if (isDone) set.delete(li)
                    else set.add(li)
                    return { ...q, xp: Math.max(0, q.xp + (isDone ? -20 : 20)), labs: { ...q.labs, [w.id]: [...set].sort((a, b) => a - b) } }
                  })
                  if (!isDone) {
                    celebrate('Lab done! +20 xp. Nice work!')
                    $.ui.toast('🧪 Lab done! +20 xp')
                  }
                }}
              >
                {done.has(li) ? 'Undo done' : 'Mark done'}
              </Button>
            </Box>
          )}
        </Box>,
      )
    }

    if (tab === 'quiz') {
      const qi = Math.min(await read($, quizAtom), Math.max(0, w.quiz.length - 1))
      const item = w.quiz[qi]
      const picked = await read($, pickedAtom)
      const record = p.quiz[w.id] ?? {}
      if (!item) return shell(<Text dimColor>No quiz for this week yet.</Text>)
      const hasPicked = picked >= 0
      const isRight = picked === item.answer
      return shell(
        <Box flexDirection="column" rowGap={1}>
          {weekTitle}
          <Text dimColor>
            Question {qi + 1} of {w.quiz.length} · {quizRight(p, w)} right so far {quizRight(p, w) === w.quiz.length ? '🏆' : ''}
          </Text>
          <Text bold wrap="wrap">
            {item.q}
          </Text>
          <Box flexDirection="column">
            {item.options.map((o, n) => {
              const mark = !hasPicked ? '' : n === item.answer ? ' ✓' : n === picked ? ' ✗' : ''
              return (
                <Button
                  key={'opt-' + n}
                  hotkey={String(n + 1)}
                  plain
                  dimColor={hasPicked && n !== item.answer && n !== picked}
                  onPress={async () => {
                    if (hasPicked) return
                    await update($, pickedAtom, () => n)
                    const right = n === item.answer
                    const first = record[String(qi)] === undefined
                    await commit($, q => {
                      const prev = q.quiz[w.id] ?? {}
                      const wasRight = prev[String(qi)] === true
                      const gain = right && !wasRight ? 10 : 0
                      const now = { ...prev, [String(qi)]: wasRight || right }
                      const perfect = Object.values(now).filter(Boolean).length === w.quiz.length
                      const bonus = perfect && Object.values(prev).filter(Boolean).length < w.quiz.length ? 15 : 0
                      return { ...q, xp: q.xp + gain + bonus, quiz: { ...q.quiz, [w.id]: now } }
                    })
                    if (right) {
                      celebrate(first ? 'Correct! +10 xp. Woohoo!' : 'Correct again! You remember it.')
                      $.ui.toast(first ? '🎉 Correct! +10 xp' : '🎉 Correct!')
                    }
                  }}
                >
                  {o + mark}
                </Button>
              )
            })}
          </Box>
          {hasPicked && (
            <Box flexDirection="column" borderStyle="round" borderColor={isRight ? 'success' : 'warning'} paddingX={1}>
              <Text bold color={isRight ? 'success' : 'warning'}>
                {isRight ? '🎉 Nailed it!' : '🤔 Not quite.'}
              </Text>
              <Text wrap="wrap">{item.why}</Text>
            </Box>
          )}
          <Box flexDirection="row" columnGap={1}>
            <Button
              key="next-q"
              hotkey="j"
              variant={hasPicked ? 'primary' : 'secondary'}
              onPress={async () => {
                await update($, quizAtom, i => (i + 1) % Math.max(1, w.quiz.length))
                await update($, pickedAtom, () => -1)
              }}
            >
              Next question
            </Button>
          </Box>
        </Box>,
      )
    }

    // stats
    const totalLabs = WEEKS.reduce((n, x) => n + x.labs.length, 0)
    const doneLabs = WEEKS.reduce((n, x) => n + labsDone(p, x), 0)
    const totalQ = WEEKS.reduce((n, x) => n + x.quiz.length, 0)
    const rightQ = WEEKS.reduce((n, x) => n + quizRight(p, x), 0)
    const phaseRows = Object.entries(PHASES)
      .map(([phase, info]) => {
        const ws = WEEKS.filter(x => x.phase === Number(phase))
        const score = ws.reduce((n, x) => n + (weekState(p, x) === 'done' ? 1 : weekState(p, x) === 'started' ? 0.4 : 0), 0)
        return { phase: Number(phase), label: info.label, fraction: ws.length ? score / ws.length : 0, n: ws.length }
      })
      .filter(r => r.n > 0)
    const earned = badges(p)
    return shell(
      <Box flexDirection="column" rowGap={1}>
        {paint('stats', () => {
          const cols = Math.min(width, 90)
          const c = canvas(cols, phaseRows.length + 4)
          const lab = 16
          c.text(0, 0, `Level ${lvl}`, WHITE)
          bar(c, lab, 0, cols - lab - 8, fraction, 0x7c4dff, 0x00e5ff, frame)
          c.text(cols - 6, 0, `${Math.round(fraction * 100)}%`, WHITE)
          c.text(0, 1, 'Labs', WHITE)
          bar(c, lab, 1, cols - lab - 8, totalLabs ? doneLabs / totalLabs : 0, 0x00c853, 0xb2ff59, frame)
          c.text(cols - 6, 1, `${doneLabs}`, WHITE)
          c.text(0, 2, 'Quiz', WHITE)
          bar(c, lab, 2, cols - lab - 8, totalQ ? rightQ / totalQ : 0, 0xff6d00, 0xffd740, frame)
          c.text(cols - 6, 2, `${rightQ}`, WHITE)
          phaseRows.forEach((r, n) => {
            const pc = PHASE_COLOR[r.phase] ?? WHITE
            c.text(0, n + 4, r.label, pc)
            bar(c, lab, n + 4, cols - lab - 8, r.fraction, mix(pc, 0x000000, 0.4), mix(pc, WHITE, 0.3), frame)
            c.text(cols - 6, n + 4, `${Math.round(r.fraction * 100)}%`, r.fraction > 0 ? pc : DIM)
          })
          return c
        })}
        <Text>
          ⚡ {p.xp} xp · 🔥 {streak(p.days)}-day streak · 🧭 {Object.keys(p.visited).length}/{WEEKS.length} weeks visited · 🧪 {doneLabs}/{totalLabs} labs · 🧠 {rightQ}/{totalQ} answers
        </Text>
        <Box flexDirection="column">
          <Text bold>🏅 Badges</Text>
          {earned.length === 0 && <Text dimColor>None yet: visit a week, finish a lab, or ace a quiz.</Text>}
          {earned.map((b, n) => (
            <Text key={'badge-' + n}>{b}</Text>
          ))}
        </Box>
        <Text dimColor wrap="wrap">
          XP: +5 for visiting a week · +20 per lab you mark done · +10 per right answer · +15 for a perfect quiz.
        </Text>
      </Box>,
    )
  })
}
