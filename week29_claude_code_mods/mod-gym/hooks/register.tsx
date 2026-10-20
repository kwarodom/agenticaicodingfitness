// Mod Gym: Week 29 of Agentic Coding Fitness, taught inside Claude Code.
//
//   /mod-gym                 open the panel (Lesson · Events · Progress)
//   /mod-gym next|prev|N     move between the eight lessons
//   /mod-gym check           run the current lesson's checks on your lab mod
//   /mod-gym lab <path>      where your lab mods live (default ~/mod-gym-lab)
//   /mod-gym band on|off     the one-line band above the prompt
//   /mod-gym status|reset
//   tool gym_status          Claude reads where you are, so it can coach
//
// The panel is itself the course's running example: every hook below is one
// of the patterns a lesson teaches, and the Events tab shows this module's own
// hooks observing and wrapping the session's events as they happen.

import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { CheckOutcome, GymProgress, GymTab, TapeEntry } from '../types'
import { LESSONS } from './lessons'
import type { Check, Lesson } from './lessons'

const PANE = 'mod-gym'
const TOOL = 'mcp__mod-gym__gym_status'
const TAPE_MAX = 60

const lessonAtom = atom({ plugin: 'mod-gym', key: 'lesson' } as const, 0)
const tabAtom = atom({ plugin: 'mod-gym', key: 'tab' } as const, 'lesson')
const checksAtom = atom({ plugin: 'mod-gym', key: 'checks' } as const, [])
const isCheckingAtom = atom({ plugin: 'mod-gym', key: 'isChecking' } as const, false)
const tapeAtom = atom({ plugin: 'mod-gym', key: 'tape' } as const, [])
const passedAtom = atom({ plugin: 'mod-gym', key: 'passed' } as const, {})
const isBandHiddenAtom = atom({ plugin: 'mod-gym', key: 'isBandHidden' } as const, false)
const labDirAtom = atom({ plugin: 'mod-gym', key: 'labDir' } as const, '')

type $ = EngineInterface

// ---------- small helpers ----------

const today = () => new Date().toISOString().slice(0, 10)

function lessonAt(i: number): Lesson {
  return LESSONS[Math.max(0, Math.min(LESSONS.length - 1, i))] as Lesson
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

function clip(text: string, n: number): string {
  const one = text.replace(/\s+/g, ' ').trim()
  return one.length > n ? one.slice(0, n - 1) + '…' : one
}

function clock(ms: number): string {
  return new Date(ms).toISOString().slice(11, 19)
}

function compareVersions(a: string, b: string): number {
  const pa = a.split(/[.-]/).map(n => parseInt(n, 10) || 0)
  const pb = b.split(/[.-]/).map(n => parseInt(n, 10) || 0)
  for (let i = 0; i < 3; i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0)
    if (d !== 0) return d
  }
  return 0
}

async function home($: $): Promise<string> {
  return (await $.env.get('HOME')) ?? '~'
}

async function labDir($: $): Promise<string> {
  const set = await read($, labDirAtom)
  return set || (await home($)) + '/mod-gym-lab'
}

async function loadProgress($: $): Promise<GymProgress> {
  const saved = (await $.store.get('progress')) as Partial<GymProgress> | undefined
  return {
    lesson: typeof saved?.lesson === 'number' ? saved.lesson : 0,
    passed: saved?.passed && typeof saved.passed === 'object' ? saved.passed : {},
    days: Array.isArray(saved?.days) ? saved.days : [],
  }
}

async function saveProgress($: $, patch: Partial<GymProgress>): Promise<GymProgress> {
  const next = { ...(await loadProgress($)), ...patch }
  await $.store.set('progress', next)
  return next
}

async function refreshStatus($: $): Promise<void> {
  const i = await read($, lessonAtom)
  const passed = await read($, passedAtom)
  $.ui.status('Mod Gym L' + lessonAt(i).id + ' · ' + Object.keys(passed).length + '/' + LESSONS.length)
}

async function goTo($: $, i: number): Promise<Lesson> {
  const clamped = Math.max(0, Math.min(LESSONS.length - 1, i))
  await update($, lessonAtom, () => clamped)
  await update($, checksAtom, () => [])
  await saveProgress($, { lesson: clamped })
  await refreshStatus($)
  return lessonAt(clamped)
}

async function record($: $, entry: Omit<TapeEntry, 'at'>): Promise<void> {
  try {
    const at = await $.clock.now()
    await update($, tapeAtom, list => [...list, { ...entry, at }].slice(-TAPE_MAX))
  } catch {
    // The tape is a teaching aid: never let it break the event it watches.
  }
}

function briefToolInput(input: Record<string, unknown>): string {
  for (const k of ['command', 'file_path', 'pattern', 'url', 'description', 'prompt']) {
    const v = input[k]
    if (typeof v === 'string' && v) return clip(v, 48)
  }
  return ''
}

// ---------- checks ----------

async function runOne($: $, lab: string, check: Check): Promise<CheckOutcome> {
  const label = check.label
  const at = (p: string) => lab + '/' + p
  try {
    switch (check.kind) {
      case 'version': {
        const { version } = await $.session.version()
        const ok = compareVersions(version, check.min) >= 0
        return { label, isPassed: ok, detail: 'running ' + version + (ok ? '' : ', need ' + check.min) }
      }
      case 'exists': {
        const ok = await $.fs.exists(at(check.path))
        return { label, isPassed: ok, detail: ok ? check.path : 'missing ' + at(check.path) }
      }
      case 'contains': {
        // register.ts becomes register.tsx in lesson 5; earlier lessons accept either.
        let path = check.path
        if (!(await $.fs.exists(at(path))) && path.endsWith('.ts') && (await $.fs.exists(at(path + 'x')))) path += 'x'
        if (!(await $.fs.exists(at(path)))) {
          return { label, isPassed: false, detail: 'missing ' + at(path) }
        }
        const text = await $.fs.read(at(path))
        const ok = new RegExp(check.pattern).test(text)
        return { label, isPassed: ok, detail: ok ? path : 'not found in ' + path }
      }
      case 'anyFile': {
        if (!(await $.fs.exists(at(check.dir)))) {
          return { label, isPassed: false, detail: 'missing ' + at(check.dir) }
        }
        const hit = (await $.fs.list(at(check.dir))).find(f => f.name.endsWith(check.suffix))
        return { label, isPassed: !!hit, detail: hit ? check.dir + '/' + hit.name : 'no *' + check.suffix + ' in ' + check.dir }
      }
      case 'validate': {
        if (!(await $.fs.exists(at(check.mod)))) {
          return { label, isPassed: false, detail: 'missing ' + at(check.mod) }
        }
        const r = await $.process.run(['claude', 'plugin', 'validate', at(check.mod)], { timeoutMs: 60_000 })
        const out = r.stdout + '\n' + r.stderr
        const hasWanted = check.mustInclude ? new RegExp(check.mustInclude).test(out) : true
        const firstProblem = out.split('\n').find(l => /error|✘|refus|invalid|not /i.test(l)) ?? ''
        const ok = r.exitCode === 0 && hasWanted
        const detail = ok
          ? 'exit 0'
          : r.exitCode !== 0
            ? 'exit ' + r.exitCode + (firstProblem ? ': ' + clip(firstProblem, 70) : '')
            : 'output lacks /' + check.mustInclude + '/'
        return { label, isPassed: ok, detail }
      }
      case 'test': {
        if (!(await $.fs.exists(at(check.mod)))) {
          return { label, isPassed: false, detail: 'missing ' + at(check.mod) }
        }
        const r = await $.process.run(['claude', 'plugin', 'test', at(check.mod)], { cwd: at(check.mod), timeoutMs: 120_000 })
        const out = (r.stdout + '\n' + r.stderr).trim()
        const last = out.split('\n').filter(Boolean).slice(-1)[0] ?? ''
        return { label, isPassed: r.exitCode === 0, detail: 'exit ' + r.exitCode + (last ? ': ' + clip(last, 70) : '') }
      }
    }
  } catch (err) {
    return { label, isPassed: false, detail: 'check failed: ' + clip(String(err), 70) }
  }
}

async function runChecks($: $): Promise<{ lesson: Lesson; outcomes: CheckOutcome[]; isNewPass: boolean }> {
  const lesson = lessonAt(await read($, lessonAtom))
  const lab = await labDir($)
  await update($, isCheckingAtom, () => true)
  const outcomes: CheckOutcome[] = []
  try {
    for (const check of lesson.checks) {
      outcomes.push(await runOne($, lab, check))
      await update($, checksAtom, () => [...outcomes])
    }
  } finally {
    await update($, isCheckingAtom, () => false)
  }
  const isAll = outcomes.every(o => o.isPassed)
  const before = await read($, passedAtom)
  const isNewPass = isAll && !before[lesson.id]
  if (isNewPass) {
    const passed = { ...before, [lesson.id]: today() }
    await update($, passedAtom, () => passed)
    await saveProgress($, { passed })
    $.ui.toast('Mod Gym: lesson ' + lesson.id + ' passed 💪')
    await refreshStatus($)
  }
  return { lesson, outcomes, isNewPass }
}

function outcomeText(lesson: Lesson, outcomes: readonly CheckOutcome[]): string {
  const lines = outcomes.map(o => (o.isPassed ? '✓ ' : '✗ ') + o.label + ' (' + o.detail + ')')
  const n = outcomes.filter(o => o.isPassed).length
  return 'Lesson ' + lesson.id + ' · ' + lesson.title + ' · ' + n + '/' + outcomes.length + ' checks\n' + lines.join('\n')
}

function starterScript(lab: string): string {
  const dir = lab + '/hello-mod'
  return [
    "mkdir -p '" + dir + "/.claude-plugin' '" + dir + "/hooks'",
    "cat > '" + dir + "/.claude-plugin/plugin.json' <<'EOF'",
    '{ "name": "hello-mod", "version": "0.1.0", "description": "My first mod" }',
    'EOF',
    "cat > '" + dir + "/hooks/hooks.json' <<'EOF'",
    '{ "modules": ["./register.ts"] }',
    'EOF',
    "cat > '" + dir + "/hooks/register.ts' <<'EOF'",
    "import type { Register } from 'claude-code'",
    '',
    'export const register: Register = on => {',
    "  on('ui.render', { component: 'Spinner' }, ($, e, next) =>",
    "    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),",
    '  )',
    '}',
    'EOF',
    "claude plugin validate '" + dir + "'",
  ].join('\n')
}

async function askClaude($: $): Promise<void> {
  const lesson = lessonAt(await read($, lessonAtom))
  const lab = await labDir($)
  const text =
    '[Mod Gym · lesson ' + lesson.id + ': ' + lesson.title + ' · lab folder ' + lab + ' · tutorial ' + lesson.tutorial + ']\n' +
    lesson.ask
  await $.prompt.fill({ text, mode: 'replace' })
  $.ui.toast('Question is in your prompt: edit it or press Enter.')
}

// ---------- registration ----------

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const progress = await loadProgress($)
    const d = today()
    if (!progress.days.includes(d)) await saveProgress($, { days: [...progress.days, d].slice(-400) })
    await update($, lessonAtom, () => progress.lesson)
    await update($, passedAtom, () => progress.passed)
    const savedLab = await $.store.get('labDir')
    if (typeof savedLab === 'string' && savedLab) await update($, labDirAtom, () => savedLab)
    const savedBand = await $.store.get('isBandHidden')
    if (savedBand === true) await update($, isBandHiddenAtom, () => true)

    await $.command.register({
      name: 'mod-gym',
      description: 'Mod Gym: learn Claude Code mods in a panel (Week 29)',
      argumentHint: '[open|next|prev|<1-8>|check|status|lab <path>|band on|off|reset]',
    })
    await $.tool.register({
      name: 'gym_status',
      description:
        "Where the student is in Mod Gym (Week 29: Claude Code mods): current lesson, its tutorial path, the student's lab folder, last check results and passed lessons. Call it before coaching the student on a Mod Gym lesson; then guide with questions rather than writing their lab code.",
      inputSchema: { type: 'object', properties: {} },
    })
    await refreshStatus($)
    await record($, { event: 'session.start', detail: 'loaded lesson ' + lessonAt(progress.lesson).id, pattern: 'observe' })
    return next(e)
  })

  // ----- the patterns the lessons teach, running live -----

  // observe: look, then hand the prompt on unchanged
  on('prompt.submit', async ($, e, next) => {
    await record($, { event: 'prompt.submit', detail: clip(e.text, 48), pattern: 'observe' })
    return next(e)
  }) // no .catch: a hook that throws is skipped, so an observer already fails open

  // wrap: let the tool run, then act on its result
  on('tool.call', async ($, e, next) => {
    const started = await $.clock.now()
    const ran = await next(e)
    const ms = (await $.clock.now()) - started
    const outcome = ran.deny !== undefined ? 'denied' : ran.isError ? 'error' : 'ok'
    const arg = briefToolInput(e as unknown as Record<string, unknown>)
    await record($, {
      event: 'tool.call',
      detail: String(e.tool) + (arg ? ': ' + arg : '') + ' → ' + outcome + ' · ' + ms + 'ms',
      pattern: 'wrap',
    })
    return ran
  }) // no .catch on purpose: see lesson 04, a fail-open .catch here can let another hook's guarded call through

  on('turn.complete', async ($, e, next) => {
    await record($, {
      event: 'turn.complete',
      detail: e.reason + ' · ' + (e.durationMs / 1000).toFixed(1) + 's',
      pattern: 'observe',
    })
    return next(e)
  })

  // answer: our own tool, served without Claude Code's help
  on('tool.call', { tool: TOOL }, async $ => {
    const i = await read($, lessonAtom)
    const lesson = lessonAt(i)
    const status = {
      lesson: { number: i + 1, id: lesson.id, title: lesson.title, tutorial: lesson.tutorial },
      labDir: await labDir($),
      labMod: (await labDir($)) + '/hello-mod',
      checks: lesson.checks.map(c => c.label),
      lastResults: await read($, checksAtom),
      passed: await read($, passedAtom),
      lessons: LESSONS.map(l => l.id + ' ' + l.title),
      coaching: 'Ask one question at a time. Point at the lesson tutorial. Let the student type the code.',
    }
    return { result: JSON.stringify(status, null, 2) }
  }).catch(() => ({ result: 'Mod Gym status is unavailable right now.' }))

  on('command.run', { command: 'mod-gym' }, async ($, e) => {
    const [verb = '', ...rest] = e.args.trim().split(/\s+/)
    const arg = rest.join(' ')
    const v = verb.toLowerCase()

    if (v === '' || v === 'open') {
      await $.ui.open({ id: PANE, title: 'Mod Gym · Week 29', focus: true, closeOnEscape: true })
      return {}
    }
    if (v === 'next' || v === 'prev') {
      const i = await read($, lessonAtom)
      const l = await goTo($, i + (v === 'next' ? 1 : -1))
      return { text: 'Mod Gym → lesson ' + l.id + ': ' + l.title }
    }
    if (/^[1-9]$/.test(v)) {
      const l = await goTo($, Number(v) - 1)
      return { text: 'Mod Gym → lesson ' + l.id + ': ' + l.title }
    }
    if (v === 'check') {
      const { lesson, outcomes, isNewPass } = await runChecks($)
      return { text: outcomeText(lesson, outcomes) + (isNewPass ? '\nLesson passed. /mod-gym next when ready.' : '') }
    }
    if (v === 'lab') {
      if (!arg) return { text: 'Lab folder: ' + (await labDir($)) }
      const path = arg.startsWith('~') ? (await home($)) + arg.slice(1) : arg
      await update($, labDirAtom, () => path)
      await $.store.set('labDir', path)
      return { text: 'Mod Gym lab folder → ' + path + ' (your mod goes in ' + path + '/hello-mod)' }
    }
    if (v === 'band') {
      const isHidden = arg === 'off'
      await update($, isBandHiddenAtom, () => isHidden)
      await $.store.set('isBandHidden', isHidden)
      return { text: 'Mod Gym band ' + (isHidden ? 'hidden' : 'shown') }
    }
    if (v === 'reset') {
      await update($, passedAtom, () => ({}))
      await saveProgress($, { passed: {}, lesson: 0 })
      await goTo($, 0)
      return { text: 'Mod Gym progress reset to lesson 01.' }
    }
    if (v === 'status') {
      const i = await read($, lessonAtom)
      const passed = await read($, passedAtom)
      const days = (await loadProgress($)).days
      return {
        text:
          'Mod Gym · lesson ' + lessonAt(i).id + ': ' + lessonAt(i).title + ' · passed ' + Object.keys(passed).length + '/' +
          LESSONS.length + ' · streak ' + streak(days) + ' day(s) · lab ' + (await labDir($)),
      }
    }
    return { text: 'Usage: /mod-gym [open|next|prev|1-8|check|status|lab <path>|band on|off|reset]' }
  })

  // ----- drawing -----

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey || (await read($, isBandHiddenAtom))) return next(e)
    const { Box, Text, Button } = $.ui.resolve(e)
    const lesson = lessonAt(await read($, lessonAtom))
    const done = Object.keys(await read($, passedAtom)).length
    // Draw beneath whatever other plugins put in the band, rather than replacing it.
    const below = await next(e)
    return (
      <Box flexDirection="column" width={e.props.bodyColumns}>
        {below}
        <Box flexDirection="row" columnGap={1}>
          <Box flexGrow={1} flexShrink={1} overflow="hidden">
            <Text wrap="truncate-end">
              <Text color="claude" bold>
                🏋 Mod Gym
              </Text>
              <Text dimColor>
                {' '}L{lesson.id} {lesson.title} · {done}/{LESSONS.length} passed
              </Text>
            </Text>
          </Box>
          <Button key="gym-open" onPress={() => void $.ui.open({ id: PANE, title: 'Mod Gym · Week 29', focus: true, closeOnEscape: true })}>
            Open
          </Button>
          <Button
            key="gym-hide"
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

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button, Markdown } = $.ui.resolve(e)
    const i = await read($, lessonAtom)
    const lesson = lessonAt(i)
    const tab: GymTab = await read($, tabAtom)
    const passed = await read($, passedAtom)
    const done = Object.keys(passed).length
    const width = Math.max(30, e.props.bodyColumns)
    const rows = e.viewport?.rows ?? 30

    const tabButton = (id: GymTab, hotkey: string, label: string) => (
      <Button key={'tab-' + id} hotkey={hotkey} variant={tab === id ? 'primary' : 'secondary'} onPress={() => update($, tabAtom, () => id)}>
        {label}
      </Button>
    )

    const header = (
      <Box flexDirection="column" width={width}>
        <Text wrap="truncate-end">
          <Text color="claude" bold>
            🏋 Mod Gym
          </Text>
          <Text dimColor>
            {' '}Week 29 · Claude Code mods · {done}/{LESSONS.length} lessons passed
          </Text>
        </Text>
        <Box flexDirection="row" columnGap={1} flexWrap="wrap">
          {tabButton('lesson', '1', 'Lesson')}
          {tabButton('events', '2', 'Events')}
          {tabButton('progress', '3', 'Progress')}
        </Box>
      </Box>
    )

    if (tab === 'events') {
      const tape = await read($, tapeAtom)
      const room = Math.max(3, rows - 10)
      const shown = tape.slice(-room)
      const mark = { observe: '◦', wrap: '⟳', answer: '◆' } as const
      return (
        <Box flexDirection="column" width={width} rowGap={1}>
          {header}
          <Text dimColor wrap="wrap">
            Mod Gym's own hooks, live. ◦ observe = return next(e) · ⟳ wrap = await next(e), then act · ◆ answer = no next
          </Text>
          <Box flexDirection="column">
            {shown.length === 0 && <Text dimColor>Nothing yet. Send a prompt and watch.</Text>}
            {shown.map((t, n) => (
              <Text key={'t' + n} wrap="truncate-end">
                <Text dimColor>{clock(t.at)} </Text>
                <Text color={t.pattern === 'wrap' ? 'suggestion' : 'text'}>
                  {mark[t.pattern]} {t.event.padEnd(14)}
                </Text>
                <Text dimColor> {t.detail}</Text>
              </Text>
            ))}
          </Box>
          <Box flexDirection="row" columnGap={1}>
            <Button key="tape-clear" hotkey="x" onPress={() => update($, tapeAtom, () => [])}>
              Clear
            </Button>
            <Button key="close" hotkey="q" role="dismiss" onPress={() => $.ui.close({ id: PANE })}>
              Close
            </Button>
          </Box>
        </Box>
      )
    }

    if (tab === 'progress') {
      const days = (await loadProgress($)).days
      return (
        <Box flexDirection="column" width={width} rowGap={1}>
          {header}
          <Box flexDirection="column">
            {LESSONS.map((l, n) => (
              <Text key={'p' + l.id} wrap="truncate-end" dimColor={!passed[l.id] && n !== i}>
                {passed[l.id] ? '✓' : n === i ? '▶' : '·'} {l.id} {l.title}
                {passed[l.id] ? '  ' + passed[l.id] : ''}
              </Text>
            ))}
          </Box>
          <Text dimColor wrap="wrap">
            Streak {streak(days)} day(s) · lab folder {await labDir($)}
          </Text>
          <Box flexDirection="row" columnGap={1}>
            <Button
              key="reset"
              hotkey="r"
              onPress={async () => {
                const pick = await $.ui.ask('Reset Mod Gym progress?', ['Keep', 'Reset'])
                if (pick !== 'Reset') return
                await update($, passedAtom, () => ({}))
                await saveProgress($, { passed: {}, lesson: 0 })
                await goTo($, 0)
              }}
            >
              Reset
            </Button>
            <Button key="close" hotkey="q" role="dismiss" onPress={() => $.ui.close({ id: PANE })}>
              Close
            </Button>
          </Box>
        </Box>
      )
    }

    const checks = await read($, checksAtom)
    const isChecking = await read($, isCheckingAtom)
    return (
      <Box flexDirection="column" width={width} rowGap={1}>
        {header}
        <Text bold wrap="wrap">
          Lesson {lesson.id} of {String(LESSONS.length).padStart(2, '0')} · {lesson.title}
          {passed[lesson.id] ? '  ✓' : ''}
        </Text>
        <Markdown key={'lesson-' + lesson.id} text={lesson.body} />
        <Box flexDirection="column">
          <Text bold>Checks{isChecking ? ' (running…)' : ''}</Text>
          {lesson.checks.map((c, n) => {
            const o = checks[n]
            return (
              <Text key={'c' + n} wrap="truncate-end" dimColor={!o}>
                <Text color={o ? (o.isPassed ? 'success' : 'error') : 'inactive'}>{o ? (o.isPassed ? '✓' : '✗') : '·'}</Text> {c.label}
                {o ? <Text dimColor> ({o.detail})</Text> : ''}
              </Text>
            )
          })}
        </Box>
        <Text dimColor wrap="wrap">
          Long form: {lesson.tutorial} · lab: {await labDir($)}/hello-mod
        </Text>
        <Box flexDirection="row" columnGap={1} flexWrap="wrap">
          <Button key="prev" hotkey="p" onPress={() => void goTo($, i - 1)}>
            Prev
          </Button>
          <Button key="next" hotkey="n" onPress={() => void goTo($, i + 1)}>
            Next
          </Button>
          <Button key="check" hotkey="c" variant="primary" onPress={() => void runChecks($)}>
            Check
          </Button>
          <Button key="ask" hotkey="a" onPress={() => askClaude($)}>
            Ask Claude
          </Button>
          {lesson.id === '02' && (
            <Button
              key="starter"
              hotkey="f"
              onPress={async press => {
                const r = await $.ui.copy({ text: starterScript(await labDir($)), surface: press.surface })
                $.ui.toast(r.isCopied ? 'Starter script copied: paste it in a shell.' : 'Could not copy here: see ' + lesson.tutorial)
              }}
            >
              Copy starter
            </Button>
          )}
          <Button key="close" hotkey="q" role="dismiss" onPress={() => $.ui.close({ id: PANE })}>
            Close
          </Button>
        </Box>
      </Box>
    )
  })
}
