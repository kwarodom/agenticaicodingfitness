// A made-up repository served to the mod through stand-ins for `$.fs` and `$.process`.
// Every file, name and number here is made up for the tests.

import type { On } from 'claude-code'

import type { RepoScan } from '../types'

export const NOW = new Date(2026, 9, 29, 12, 0, 0).getTime()
export const ROOT = '/work/test-repo'
export const ASKED = { origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 160 } } as const
export const TYPED = { wait: false, origin: { kind: 'composer' } } as const
export const BAND = {
  plugin: 'alto-belt',
  surface: 'terminal',
  component: 'AbovePrompt',
  props: { hasSurvey: false, isWorking: false, maxRows: 10, bodyColumns: 100, scroll: { offset: 0, bodyRows: 10 }, view: {} },
} as const
export const PANE = {
  plugin: 'alto-belt',
  surface: 'terminal',
  component: 'Pane',
  requestId: 'belt',
  props: { title: 'Alto Belt', isFocused: true, bodyColumns: 110, placement: 'dock', scroll: { offset: 0, bodyRows: 60 }, view: {} },
} as const

const lines = (n: number): string => Array.from({ length: n }, (_, i) => `line ${i + 1}`).join('\n')

/** A repository at roughly Green: a long CLAUDE.md, one unscoped rule, two hooks, no review files. */
export const greenRepo = (): Map<string, string> =>
  new Map([
    [`${ROOT}/CLAUDE.md`, lines(222)],
    [`${ROOT}/.claude/rules/always.md`, '# Always\n'.repeat(1500)],
    [`${ROOT}/.claude/rules/python.md`, '---\npaths:\n  - "**/*.py"\n---\n# Python\n'],
    [
      `${ROOT}/.claude/settings.json`,
      JSON.stringify({
        hooks: {
          PreToolUse: [{ matcher: 'Bash', hooks: [{ type: 'command', command: '$CLAUDE_PROJECT_DIR/.claude/hooks/env-guard.sh' }] }],
          Stop: [{ hooks: [{ type: 'command', command: '$CLAUDE_PROJECT_DIR/.claude/hooks/todo.sh' }] }],
        },
      }),
    ],
    [`${ROOT}/.claude/settings.local.json`, JSON.stringify({ permissions: { allow: ['Bash(python3 *)', 'Bash(git status)'] } })],
    [`${ROOT}/.claude/hooks/env-guard.sh`, '#!/bin/sh\n'],
    [`${ROOT}/.claude/hooks/tests/run.sh`, '#!/bin/sh\n'],
    [`${ROOT}/.claude/agents/reviewer.md`, '---\nname: reviewer\n---\n'],
    [`${ROOT}/.claude/skills/fix-issue/SKILL.md`, '---\nname: fix-issue\n---\n'],
  ])

const isDir = (files: Map<string, string>, path: string): boolean => [...files.keys()].some(one => one.startsWith(`${path.replace(/\/$/, '')}/`))

const entriesOf = (files: Map<string, string>, dir: string) => {
  const base = `${dir.replace(/\/$/, '')}/`
  const names = new Map<string, { name: string; kind: 'file' | 'dir'; size: number; mtimeMs: number; isLink: boolean }>()
  for (const [path, text] of files) {
    if (!path.startsWith(base)) continue
    const [name = '', ...rest] = path.slice(base.length).split('/')
    names.set(name, rest.length === 0 ? { name, kind: 'file', size: new TextEncoder().encode(text).length, mtimeMs: NOW, isLink: false } : { name, kind: 'dir', size: 0, mtimeMs: 0, isLink: false })
  }
  if (names.size === 0) throw new Error(`ENOENT: ${dir}`)
  return [...names.values()]
}

/** Stand-ins for the filesystem, git and gh. `worktrees` and `drafts` set what git and gh answer. */
export const serve = (on: On, files: Map<string, string>, world: { worktrees?: number; drafts?: number; opened?: string[] } = {}): void => {
  const worktrees = world.worktrees ?? 2
  const drafts = world.drafts ?? 1
  on('session.root', () => ({ value: ROOT }))
  on('session.cwd', () => ({ value: ROOT }))
  on('fs.list', (_$, e) => ({ value: entriesOf(files, e.path) }))
  on('fs.read', (_$, e) => {
    const text = files.get(e.path)
    if (text === undefined) throw new Error(`ENOENT: ${e.path}`)
    return { value: text }
  })
  on('fs.exists', (_$, e) => ({ value: files.has(e.path) || isDir(files, e.path) }))
  on('process.run', (_$, e) => {
    const argv = e.argv.join(' ')
    let stdout = ''
    if (argv.startsWith('git rev-parse')) stdout = `${ROOT}/.git\n`
    else if (argv.startsWith('git worktree list')) stdout = Array.from({ length: worktrees }, (_, i) => `worktree ${ROOT}${i === 0 ? '' : `-wt${i}`}\nHEAD abc\n`).join('\n')
    else if (argv.startsWith('gh pr list')) stdout = JSON.stringify(Array.from({ length: drafts }, (_, i) => ({ number: i + 1 })))
    return { value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('ui.open', (_$, e) => {
    world.opened?.push(e.id)
    return { value: { isPlaced: true } } as never
  })
}

/** A scan with everything at its weakest, for scoring tests to change one field at a time. */
export const bareScan = (change: Partial<RepoScan> = {}): RepoScan => ({
  root: ROOT,
  repo: ROOT,
  name: 'test-repo',
  claudeMdLines: null,
  alwaysOnBytes: 0,
  rulesTotal: 0,
  rulesUnscoped: 0,
  hooks: 0,
  preToolUseHooks: 0,
  guardHooks: 0,
  hasHookTests: false,
  skills: 0,
  agents: 0,
  reviewerAgents: 0,
  goldenSets: 0,
  hasReviewMd: false,
  hasPrTemplate: false,
  hasCodeowners: false,
  broadApprovals: 0,
  isBypassDefault: false,
  worktrees: 1,
  drafts: 0,
  ...change,
})

type Found = { key: string | undefined; text?: string }
export const words = async (ui: { findAll: (query: { type: string }) => Promise<Found[]> }): Promise<string> =>
  (await ui.findAll({ type: 'Text' })).map(one => one.text ?? '').join('\n')

/** A store in memory the test can read directly. */
export const memoryStore = (on: On): Map<string, unknown> => {
  const held = new Map<string, unknown>()
  on('store.get', (_$, e) => ({ value: held.get(e.key) }) as never)
  on('store.set', (_$, e) => {
    held.set(e.key, e.value)
    return { value: undefined } as never
  })
  on('store.delete', (_$, e) => {
    held.delete(e.key)
    return { value: undefined } as never
  })
  on('store.keys', () => ({ value: [...held.keys()] }) as never)
  return held
}
