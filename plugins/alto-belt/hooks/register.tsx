// Alto Belt: a live Claude Code rubric score per repository, the next three changes, and one quiet
// nudge when a habit slips. Counts only, kept on this machine; no prompt or command text is stored.

import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { Assessment, Declaration, Declared, Feed, Live, Nudge, RepoScan, ScorePoint, Tally } from '../types'
import { BELT_NAMES, assess } from './score'
import { scanRepo } from './scan'
import type { Io } from './scan'
import {
  EMPTY_TALLY,
  added,
  dayOf,
  isAgentSpawn,
  isCommit,
  isCorrection,
  isPrCreate,
  isReview,
  isTestCommand,
  missingSections,
  namesVerifier,
  prBodyOf,
  trimmed,
  windowOf,
} from './signals'
import { prOfReview, reviewerOf, roundsOf, withRead } from './rounds'
import type { RoundsLog } from './rounds'
import { BeltPane, NudgeLine } from './views'
import { BELT_IDS, assetsScript, dataScript, webDataOf, webDirOf } from './web'
import type { Ui } from './views'

const PANE = 'belt'
const PANE_ROWS = 34
const WINDOW_DAYS = 14
const KEEP_DAYS = 60
const KEEP_HISTORY = 90
const FULL_PERCENT = 70
const PR_NUDGE_PROMPTS = 3
const SETTLE_MS = 2_000
const RUN_TIMEOUT_MS = 20_000
const DECLARATIONS: Declaration[] = ['container', 'rounds', 'field', 'cost']

const EMPTY_FEED: Feed<RepoScan> = { data: null, fetchedAt: null, error: null }
const EMPTY_LIVE: Live = { prompts: 0, streak: 0, contextPercent: null, testsSinceCommit: 0, commits: 0 }

const scanFeed = atom({ plugin: 'alto-belt', key: 'scan' } as const, EMPTY_FEED)
const daysAtom = atom({ plugin: 'alto-belt', key: 'days' } as const, {} as Record<string, Tally>)
const historyAtom = atom({ plugin: 'alto-belt', key: 'history' } as const, [] as ScorePoint[])
const declaredAtom = atom({ plugin: 'alto-belt', key: 'declared' } as const, {} as Declared)
const roundsAtom = atom({ plugin: 'alto-belt', key: 'rounds' } as const, {} as RoundsLog)
const liveAtom = atom({ plugin: 'alto-belt', key: 'live' } as const, EMPTY_LIVE)
const nudgeAtom = atom({ plugin: 'alto-belt', key: 'nudge' } as const, null as Nudge | null)
const noteAtom = atom({ plugin: 'alto-belt', key: 'note' } as const, null as string | null)
const revealAtom = atom({ plugin: 'alto-belt', key: 'reveal' } as const, 4)
const REVEAL_STEP_MS = 90

let hasBand = true
let hasStatus = true
let isCoachingModel = false
let reading: Promise<RepoScan | null> | null = null
// Whether the web page is on: set by /belt web, kept in the store so later sessions keep it fresh.
let isWebOn = false
const WEB_ON = 'alto-belt.web-on'

// --- the engine, as the scan reaches it ---------------------------------------------------------

const ioOf = ($: EngineInterface): Io => ({
  list: async path => (await $.fs.list(path)).map(entry => ({ name: entry.name, kind: entry.kind, size: entry.size })),
  read: path => $.fs.read(path),
  exists: path => $.fs.exists(path),
  run: async (argv, cwd) => {
    const ran = await $.process.run(argv, { cwd, timeoutMs: RUN_TIMEOUT_MS })
    return { exitCode: ran.exitCode, stdout: ran.stdout }
  },
})

const rootOf = async ($: EngineInterface): Promise<string> => $.session.root().catch(() => $.session.cwd())

const nowMs = async ($: EngineInterface): Promise<number> => {
  try {
    return await $.clock.now()
  } catch {
    return Date.now()
  }
}

const why = (problem: unknown): string => (problem instanceof Error ? problem.message : String(problem))

/** A short, stable key for a repository path, so store keys stay short and carry no path. */
export const repoKey = (repo: string): string => {
  let hash = 5381
  for (const char of repo) hash = ((hash * 33) ^ char.charCodeAt(0)) >>> 0
  const name = (repo.split('/').filter(Boolean).pop() ?? 'repo').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40)
  return `${name}-${hash.toString(16)}`
}

const storeKey = (scan: RepoScan | null, what: string): string | null => (scan === null ? null : `alto-belt.${repoKey(scan.repo)}.${what}`)

const loadStored = async ($: EngineInterface, scan: RepoScan): Promise<void> => {
  try {
    const [days, history, declared, rounds] = await Promise.all([
      $.store.get(storeKey(scan, 'days') ?? ''),
      $.store.get(storeKey(scan, 'history') ?? ''),
      $.store.get(storeKey(scan, 'declared') ?? ''),
      $.store.get(storeKey(scan, 'rounds') ?? ''),
    ])
    // Counts from before the scan finished stay: they are added to what was kept.
    if (typeof days === 'object' && days !== null) {
      await update($, daysAtom, now => {
        const merged: Record<string, Tally> = { ...(days as Record<string, Tally>) }
        for (const [day, tally] of Object.entries(now)) merged[day] = added(merged[day] ?? EMPTY_TALLY, tally)
        return merged
      })
    }
    if (Array.isArray(history)) await update($, historyAtom, () => history as ScorePoint[])
    if (typeof declared === 'object' && declared !== null) await update($, declaredAtom, () => declared as Declared)
    // Reads from before the scan finished stay: they are added to what was kept.
    if (typeof rounds === 'object' && rounds !== null) {
      await update($, roundsAtom, now => {
        const merged: RoundsLog = { ...(rounds as RoundsLog) }
        for (const [pr, byReviewer] of Object.entries(now)) {
          const into = { ...(merged[pr] ?? {}) }
          for (const [who, reads] of Object.entries(byReviewer)) into[who] = [...(into[who] ?? []), ...reads]
          merged[pr] = into
        }
        return merged
      })
    }
  } catch {
    // No store on this surface: the score covers this session only.
  }
}

const save = async ($: EngineInterface, what: 'days' | 'history' | 'declared' | 'rounds', value: unknown): Promise<void> => {
  const key = storeKey((await read($, scanFeed)).data, what)
  if (key === null) return
  try {
    await $.store.set(key, value)
  } catch {
    // No store: the value lives for this session.
  }
}

// --- counting ----------------------------------------------------------------------------------

const count = async ($: EngineInterface, change: Partial<Tally>): Promise<void> => {
  const day = dayOf(await nowMs($))
  const days = await update($, daysAtom, now => trimmed({ ...now, [day]: added(now[day] ?? EMPTY_TALLY, change) }, KEEP_DAYS))
  await save($, 'days', days)
}

/** Records one reviewer read of a PR (or of none, under `?`). Keeps the day only. */
const countRound = async ($: EngineInterface, pr: string, reviewer: string): Promise<void> => {
  const day = dayOf(await nowMs($))
  const rounds = await update($, roundsAtom, now => withRead(now, pr, reviewer, day))
  await save($, 'rounds', rounds)
}

const setNudge = async ($: EngineInterface, nudge: Nudge | null): Promise<void> => {
  await update($, nudgeAtom, () => nudge)
  await writeWebData($)
}

// --- the web page --------------------------------------------------------------------------------

const webDir = async ($: EngineInterface): Promise<string | null> => {
  const scan = (await read($, scanFeed)).data
  const home = await $.env.get('HOME').catch(() => undefined)
  if (scan === null || home === undefined || home === '') return null
  return webDirOf(home, repoKey(scan.repo))
}

/** Rewrites data.js beside the page, when the page is on. Never stands in anything's way. */
const writeWebData = async ($: EngineInterface): Promise<void> => {
  if (!isWebOn) return
  try {
    const dir = await webDir($)
    const assessment = await assessNow($)
    const scan = (await read($, scanFeed)).data
    if (dir === null || assessment === null || scan === null) return
    const data = webDataOf(assessment, scan, await read($, liveAtom), await read($, nudgeAtom), await read($, historyAtom), await nowMs($))
    await $.fs.write(`${dir}/data.js`, dataScript(data))
  } catch {
    // The page shows its last data; the next change tries again.
  }
}

/** Writes the page and its belt pictures from the plugin's own files. */
const installWeb = async ($: EngineInterface, dir: string): Promise<void> => {
  const root = $.plugin.root
  await $.fs.write(`${dir}/index.html`, await $.fs.read(`${root}/web/index.html`))
  const art: Record<string, string> = {}
  for (const id of BELT_IDS) {
    const { base64 } = (await $.fs.read(`${root}/web/belts/${id}.webp`, { as: 'bytes' })) as { base64: string }
    art[id] = `data:image/webp;base64,${base64}`
  }
  await $.fs.write(`${dir}/assets.js`, assetsScript(art))
}

const openWeb = async ($: EngineInterface): Promise<string> => {
  if ((await read($, scanFeed)).data === null) await readRepo($)
  const dir = await webDir($)
  if (dir === null) return 'The belt page could not be written: this repository has not been read, or HOME is not set.'
  try {
    await installWeb($, dir)
    isWebOn = true
    await $.store.set(WEB_ON, true).catch(() => undefined)
    await writeWebData($)
  } catch (problem) {
    return `The belt page could not be written: ${why(problem)}`
  }
  const page = `${dir}/index.html`
  for (const opener of [['open', page], ['xdg-open', page]]) {
    const ran = await $.process.run(opener, { timeoutMs: RUN_TIMEOUT_MS }).catch(() => null)
    if (ran !== null && ran.exitCode === 0) return `Opened your belt page in the browser. It updates by itself while you work.\n${page}`
  }
  return `Your belt page is ready. Open it in a browser:\n${page}`
}

// --- scoring -----------------------------------------------------------------------------------

const assessNow = async ($: EngineInterface): Promise<Assessment | null> => {
  const scan = (await read($, scanFeed)).data
  if (scan === null) return null
  const today = dayOf(await nowMs($))
  return assess(
    scan,
    windowOf(await read($, daysAtom), today, WINDOW_DAYS),
    await read($, declaredAtom),
    (await read($, liveAtom)).contextPercent,
    roundsOf(await read($, roundsAtom), today),
  )
}

const statusOf = (assessment: Assessment): string =>
  `belt ${assessment.belt === 'none' ? '–' : BELT_NAMES[assessment.belt]} ${assessment.total}/32${assessment.measured < 8 ? ` (${assessment.measured}/8)` : ''}`

/** Shows the belt in the status line and records today's score, at most one point a day. */
const publish = async ($: EngineInterface): Promise<void> => {
  const assessment = await assessNow($)
  if (assessment === null) return
  if (hasStatus) $.ui.status(statusOf(assessment))
  const day = dayOf(await nowMs($))
  const point: ScorePoint = { day, total: assessment.total, measured: assessment.measured, belt: assessment.belt }
  const history = await update($, historyAtom, now => [...now.filter(one => one.day !== day), point].slice(-KEEP_HISTORY))
  await save($, 'history', history)
  await writeWebData($)
}

const readOnce = async ($: EngineInterface): Promise<RepoScan | null> => {
  const at = await nowMs($)
  try {
    const scan = await scanRepo(ioOf($), await rootOf($))
    const was = (await read($, scanFeed)).data
    await update($, scanFeed, () => ({ data: scan, fetchedAt: at, error: null }))
    if (was === null || was.repo !== scan.repo) await loadStored($, scan)
    await publish($)
    return scan
  } catch (problem) {
    await update($, scanFeed, now => ({ ...now, error: why(problem) }))
    return null
  }
}

const readRepo = ($: EngineInterface): Promise<RepoScan | null> => {
  if (reading === null) reading = readOnce($).finally(() => (reading = null))
  return reading
}

// --- events ------------------------------------------------------------------------------------

/** What a submitted prompt says. The text is classified here and never kept. */
export const onPrompt = async ($: EngineInterface, text: string): Promise<void> => {
  const isFix = isCorrection(text)
  const live = await update($, liveAtom, now => ({ ...now, prompts: now.prompts + 1, streak: isFix ? now.streak + 1 : 0 }))
  await count($, {
    prompts: 1,
    verifierPrompts: namesVerifier(text) ? 1 : 0,
    corrections: isFix ? 1 : 0,
    thirdStrikes: live.streak >= 3 ? 1 : 0,
  })
  const was = await read($, nudgeAtom)
  if (live.streak >= 3) {
    await setNudge($, { kind: 'strike3', text: `${live.streak} corrections in a row. Two-strike rule: /clear and write a better first prompt with what you learned.`, atPrompt: live.prompts })
  } else if (live.streak === 2) {
    await setNudge($, { kind: 'strike2', text: 'Two corrections in a row. If the next answer misses too, /clear and rewrite the first prompt.', atPrompt: live.prompts })
  } else if (was !== null && (was.kind === 'strike2' || was.kind === 'strike3')) {
    await setNudge($, null)
  } else if (was !== null && was.kind === 'pr' && live.prompts - was.atPrompt >= PR_NUDGE_PROMPTS) {
    await setNudge($, null)
  } else {
    await writeWebData($)
  }
}

export const onShell = async ($: EngineInterface, command: string): Promise<void> => {
  const live = await read($, liveAtom)
  if (isTestCommand(command)) {
    await update($, liveAtom, now => ({ ...now, testsSinceCommit: now.testsSinceCommit + 1 }))
    await count($, { testRuns: 1 })
    if ((await read($, nudgeAtom))?.kind === 'untested') await setNudge($, null)
  }
  if (isCommit(command)) {
    const isTested = live.testsSinceCommit > 0 || isTestCommand(command)
    await update($, liveAtom, now => ({ ...now, testsSinceCommit: 0, commits: now.commits + 1 }))
    await count($, { commits: 1, testedCommits: isTested ? 1 : 0 })
    if (!isTested) await setNudge($, { kind: 'untested', text: 'That commit had no test run before it. Run the tests and show the output.', atPrompt: live.prompts })
  }
  if (isPrCreate(command)) {
    const body = prBodyOf(command)
    const missing = missingSections(body)
    await count($, { prs: 1, prsSeen: body.isSeen ? 1 : 0, prsEvidenced: body.isSeen && missing.length === 0 ? 1 : 0 })
    if (missing.length > 0) await setNudge($, { kind: 'pr', text: `That PR has no ${missing.join(', ')} section. Add it before review.`, atPrompt: live.prompts })
  }
}

export const onTurnEnd = async ($: EngineInterface): Promise<void> => {
  let percent: number | null = null
  try {
    const usage = await $.session.usage()
    percent = typeof usage.context.percent === 'number' ? Math.round(usage.context.percent) : null
  } catch {
    percent = null
  }
  await update($, liveAtom, now => ({ ...now, contextPercent: percent }))
  const isFull = percent !== null && percent >= FULL_PERCENT
  await count($, { turns: 1, fullTurns: isFull ? 1 : 0 })
  const was = await read($, nudgeAtom)
  if (isFull && (was === null || was.kind === 'context')) {
    await setNudge($, { kind: 'context', text: `Context is ${percent}% full. /compact with instructions on what to keep, or /clear if the task changed.`, atPrompt: (await read($, liveAtom)).prompts })
  } else if (!isFull && was?.kind === 'context') {
    await setNudge($, null)
  }
  await publish($)
}

export const onClear = async ($: EngineInterface): Promise<void> => {
  const live = await read($, liveAtom)
  await count($, { clears: 1, strikeClears: live.streak >= 2 ? 1 : 0 })
  await update($, liveAtom, now => ({ ...now, streak: 0, contextPercent: null }))
  const was = await read($, nudgeAtom)
  if (was !== null && was.kind !== 'pr') await setNudge($, null)
}

// --- the command -------------------------------------------------------------------------------

const HELP = [
  '/belt                      open the pane (and read the repository again)',
  '/belt status               your belt, score and next three changes, as text',
  '/belt refresh              read this repository again',
  '/belt web                  open your belt page in the browser; it updates live',
  '/belt checkin              fill the prompt with the Friday check-in',
  '/belt declare <what> [note] record what the mod cannot see: container, field, cost',
  '/belt undeclare <what>     take a declaration back',
  '/belt close                hide the pane',
].join('\n')

/** Fills the bars in, one cell at a time, as the pane opens. */
const revealBars = async ($: EngineInterface): Promise<void> => {
  await update($, revealAtom, () => 0)
  try {
    for (let step = 1; step <= 4; step += 1) {
      $.clock.after(step * REVEAL_STEP_MS, () => void update($, revealAtom, () => step).catch(() => undefined))
    }
  } catch {
    await update($, revealAtom, () => 4)
  }
}

const showPane = async ($: EngineInterface): Promise<void> => {
  await $.ui.open({ id: PANE, title: 'Alto Belt', focus: true, rows: PANE_ROWS })
  await revealBars($)
}

const closePane = async ($: EngineInterface): Promise<void> => {
  await $.ui.close({ id: PANE })
}

const CHECKIN = 'Run my Friday Claude Code belt check-in for this repository, following the belt-programme skill: run the scorecard, compare with last week\'s log entry, re-score the eight competencies from evidence, fill in the log template, and name next week\'s one change. Start from what /belt status shows.'

const fillCheckin = async ($: EngineInterface): Promise<void> => {
  await $.prompt.fill({ text: CHECKIN, mode: 'replace' })
}

export const statusText = (assessment: Assessment | null, scan: RepoScan | null): string => {
  if (assessment === null || scan === null) return 'Alto Belt has not read this repository yet. Try /belt refresh.'
  const lines = [
    `${scan.name}: ${assessment.belt === 'none' ? BELT_NAMES.none : `${BELT_NAMES[assessment.belt]} belt`}, ${assessment.total}/32, ${assessment.measured} of 8 measured.`,
    assessment.gate,
    '',
    ...assessment.scores.map(one => `${one.id.padEnd(13)} ${one.score === null ? '–' : one.score}  ${one.evidence}`),
    '',
    'Next:',
    ...assessment.next.map((line, i) => `${i + 1}. ${line}`),
  ]
  return lines.join('\n')
}

const declare = async ($: EngineInterface, words: string[], isOn: boolean): Promise<string> => {
  const what = (words[0] ?? '').toLowerCase() as Declaration
  if (what === 'rounds') return 'Review rounds are measured now: name the PR (#123) in each reviewer\'s description and the mod counts its rounds. /belt status shows the mean.'
  if (!DECLARATIONS.includes(what)) return `Declare one of: ${DECLARATIONS.filter(one => one !== 'rounds').join(', ')}.`
  const at = await nowMs($)
  const declared = await update($, declaredAtom, now => {
    const next: Declared = { ...now }
    if (isOn) next[what] = { at, note: words.slice(1).join(' ').slice(0, 200) }
    else delete next[what]
    return next
  })
  await save($, 'declared', declared)
  await publish($)
  return isOn ? `Declared ${what}. The pane marks it as declared, not measured.` : `Took back ${what}.`
}

const runCommand = async ($: EngineInterface, args: string): Promise<{ text?: string }> => {
  const words = args.trim().split(/\s+/).filter(word => word !== '')
  const first = (words[0] ?? '').toLowerCase()
  if (first === 'help' || first === '?') return { text: HELP }
  if (first === 'close') {
    await closePane($)
    return { text: 'Alto Belt pane hidden. /belt shows it again.' }
  }
  if (first === 'refresh') {
    await readRepo($)
    return { text: 'Read this repository again.' }
  }
  if (first === 'status') {
    if ((await read($, scanFeed)).data === null) await readRepo($)
    return { text: statusText(await assessNow($), (await read($, scanFeed)).data) }
  }
  if (first === 'web' || first === 'open') return { text: await openWeb($) }
  if (first === 'checkin') {
    await fillCheckin($)
    return {}
  }
  if (first === 'declare') return { text: await declare($, words.slice(1), true) }
  if (first === 'undeclare') return { text: await declare($, words.slice(1), false) }
  if (first === '') {
    // Always opens (or brings back) the pane and reads the repository again; /belt close hides it.
    await showPane($)
    void readRepo($).catch(() => undefined)
    return { text: 'Alto Belt is open. Press w for the belt page in your browser, or type /belt close to hide it.' }
  }
  return { text: `"${args.trim()}" is not something /belt knows.\n${HELP}` }
}

// --- the model's instructions, only when asked for ---------------------------------------------

const COACH_SECTION = {
  id: 'alto-belt:coach',
  scope: 'session' as const,
  text: [
    'The person is training their Claude Code habits. Before a non-trivial task, if they have not named how the result will be checked, ask for the check in one line (a test command, a screen to look at).',
    'Before saying a task is done, run that check and show its output.',
    'If the person has corrected you twice in a row on the same task, suggest they /clear and restate the task with what was learned.',
  ].join(' '),
}

// --- hooks -------------------------------------------------------------------------------------

const readOptions = (options: Record<string, unknown>): void => {
  hasBand = options.band !== false
  hasStatus = options.status !== false
  isCoachingModel = options.coach_model === true
}

export const register: Register = (on, options) => {
  readOptions(options as Record<string, unknown>)

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'belt',
      description: 'Alto Belt: your Claude Code belt and score in this repository, and what to change next. /belt [web|status|refresh|checkin|declare|help|close]',
      argumentHint: '[web|status|refresh|checkin|declare <container|field|cost> [note]|undeclare <what>|help|close]',
    })
    isWebOn = (await $.store.get(WEB_ON).catch(() => false)) === true
    try {
      $.clock.after(SETTLE_MS, () => void readRepo($).catch(() => undefined))
    } catch {
      // No clock here: the repository is read when /belt is typed.
    }
    return next(e)
  })

  on('command.run', { command: 'belt' }, ($, e) => runCommand($, e.args))

  // Classifies the prompt and passes it on untouched. Never stands in the prompt's way.
  on('prompt.submit', async ($, e, next) => {
    if (e.origin?.kind === 'composer') await onPrompt($, e.text).catch(() => undefined)
    return next(e)
  }).catch((_$, e, next) => next(e))

  // Looks at what a tool call did and returns its result untouched.
  on('tool.call', async ($, e, next) => {
    const result = await next(e)
    if (result.deny !== undefined || result.isError === true) return result
    // A call's input fields sit on the event itself, beside `tool`.
    const input = e as unknown as Record<string, unknown>
    const tool = String(e.tool)
    if (tool === 'Bash') await onShell($, String(input.command ?? '')).catch(() => undefined)
    if (isReview(tool, input)) await count($, { reviews: 1 }).catch(() => undefined)
    const reviewer = reviewerOf(tool, input)
    const pr = reviewer === null ? null : prOfReview(input)
    if (reviewer !== null && pr !== null) await countRound($, pr, reviewer).catch(() => undefined)
    if (isAgentSpawn(tool)) await count($, { agentSpawns: 1 }).catch(() => undefined)
    return result
  }).catch((_$, e, next) => next(e))

  on('turn.complete', async ($, e, next) => {
    if (e.agentId === undefined) await onTurnEnd($).catch(() => undefined)
    return next(e)
  }).catch((_$, e, next) => next(e))

  on('session.compact', async ($, e, next) => {
    if (e.trigger === 'manual' && e.agentId === undefined) await count($, { manualCompacts: 1 }).catch(() => undefined)
    return next(e)
  }).catch((_$, e, next) => next(e))

  on('session.end', async ($, e, next) => {
    if (e.reason === 'clear') await onClear($).catch(() => undefined)
    return next(e)
  }).catch((_$, e, next) => next(e))

  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    if (!isCoachingModel) return composed
    return { sections: [...composed.sections.filter(one => one.id !== COACH_SECTION.id), COACH_SECTION] }
  }).catch((_$, e, next) => next(e))

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (!hasBand || e.props.hasSurvey) return next(e)
    const nudge = await read($, nudgeAtom)
    if (nudge === null) return next(e)
    const table = $.ui.resolve(e)
    const ui: Ui = { Box: table.Box, Text: table.Text, Button: table.Button }
    const beneath = await next(e).then(
      tree => tree,
      () => null,
    )
    const { Box } = ui
    return (
      <Box flexDirection="column">
        {beneath}
        {NudgeLine(ui, nudge, e.props.bodyColumns)}
      </Box>
    )
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const table = $.ui.resolve(e)
    const ui: Ui = { Box: table.Box, Text: table.Text, Button: table.Button }
    return BeltPane(
      ui,
      {
        scan: await read($, scanFeed),
        assessment: await assessNow($),
        live: await read($, liveAtom),
        history: await read($, historyAtom),
        note: await read($, noteAtom),
        reveal: await read($, revealAtom),
      },
      e.props.bodyColumns,
      {
        refresh: () => void readRepo($).catch(() => undefined),
        checkin: () => void fillCheckin($).catch(() => undefined),
        web: () => void openWeb($).then(text => update($, noteAtom, () => text.split('\n')[0] ?? null)).catch(() => undefined),
        close: () => void closePane($).catch(() => undefined),
      },
    )
  })
}
