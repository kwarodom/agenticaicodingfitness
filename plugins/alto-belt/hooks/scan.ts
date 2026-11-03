// Reads a repository for the evidence the rubric needs. Read-only; through `Io` so tests can stand
// in for the engine's `$.fs` and `$.process`.

import type { RepoScan } from '../types'
import { isBroadApproval } from './signals'

export type Entry = { name: string; kind: string; size: number }
export type Ran = { exitCode: number; stdout: string }
export type Io = {
  list: (path: string) => Promise<Entry[]>
  read: (path: string) => Promise<string>
  exists: (path: string) => Promise<boolean>
  run: (argv: string[], cwd: string) => Promise<Ran>
}

const MAX_READ = 400_000
const GOLDEN_MIN = 10
const REVIEWER = /(review|referee|security|trust|auditor)/i
const GUARD = /(guard|env|secret|deny|protect|block|d9|killswitch)/i

const join = (...parts: string[]): string => parts.join('/').replace(/\/+/g, '/')

const listed = async (io: Io, path: string): Promise<Entry[]> => {
  try {
    return await io.list(path)
  } catch {
    return []
  }
}

const readText = async (io: Io, path: string): Promise<string | null> => {
  try {
    const text = await io.read(path)
    return text.length > MAX_READ ? text.slice(0, MAX_READ) : text
  } catch {
    return null
  }
}

const firstLine = (text: string): string => text.split('\n')[0]?.trim() ?? ''

/** Whether a rule's front matter names `paths:`, so it loads only for matching files. */
export const isPathScoped = (text: string): boolean => {
  if (!text.startsWith('---')) return false
  const end = text.indexOf('\n---', 3)
  if (end === -1) return false
  return /^paths\s*:/m.test(text.slice(3, end))
}

/** Tasks in a golden set: list items, numbered items, `##` headings, or JSON lines. */
export const goldenTasks = (name: string, text: string): number => {
  const lines = text.split('\n').map(line => line.trim())
  if (name.endsWith('.jsonl')) return lines.filter(line => line.startsWith('{')).length
  return lines.filter(line => /^(-|\*|\d+[.)]|##)\s+\S/.test(line)).length
}

type Settings = { permissions?: { allow?: unknown; defaultMode?: unknown }; hooks?: Record<string, unknown> }

const parsed = (text: string | null): Settings => {
  if (text === null) return {}
  try {
    const value = JSON.parse(text) as unknown
    return typeof value === 'object' && value !== null ? (value as Settings) : {}
  } catch {
    return {}
  }
}

const hookCommands = (settings: Settings): { event: string; command: string }[] => {
  const out: { event: string; command: string }[] = []
  for (const [event, groups] of Object.entries(settings.hooks ?? {})) {
    if (!Array.isArray(groups)) continue
    for (const group of groups as { hooks?: { command?: unknown }[] }[]) {
      for (const hook of group.hooks ?? []) out.push({ event, command: String(hook.command ?? '') })
    }
  }
  return out
}

const allowOf = (settings: Settings): string[] => (Array.isArray(settings.permissions?.allow) ? (settings.permissions.allow as unknown[]).map(String) : [])

/** The main checkout of the repository that holds `root`, so every worktree counts as one repository. */
export const mainCheckout = async (io: Io, root: string): Promise<string> => {
  try {
    const ran = await io.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], root)
    const common = firstLine(ran.stdout)
    if (ran.exitCode !== 0 || common === '') return root
    return common.replace(/\/$/, '').replace(/\/\.git$/, '') || root
  } catch {
    return root
  }
}

const countOf = async (io: Io, argv: string[], cwd: string, count: (stdout: string) => number): Promise<number | null> => {
  try {
    const ran = await io.run(argv, cwd)
    return ran.exitCode === 0 ? count(ran.stdout) : null
  } catch {
    return null
  }
}

export const scanRepo = async (io: Io, root: string): Promise<RepoScan> => {
  const repo = await mainCheckout(io, root)
  const claudeMd = await readText(io, join(root, 'CLAUDE.md'))
  let alwaysOnBytes = claudeMd === null ? 0 : new TextEncoder().encode(claudeMd).length

  const rules = (await listed(io, join(root, '.claude/rules'))).filter(entry => entry.kind === 'file' && entry.name.endsWith('.md'))
  let rulesUnscoped = 0
  for (const rule of rules) {
    const text = (await readText(io, join(root, '.claude/rules', rule.name))) ?? ''
    if (!isPathScoped(text)) {
      rulesUnscoped += 1
      alwaysOnBytes += rule.size
    }
  }

  const shared = parsed(await readText(io, join(root, '.claude/settings.json')))
  const local = parsed(await readText(io, join(root, '.claude/settings.local.json')))
  const hooks = [...hookCommands(shared), ...hookCommands(local)]
  const hookDir = await listed(io, join(root, '.claude/hooks'))
  const hasHookTests = hookDir.some(entry => /tests?$/.test(entry.name) || /\.test\.|_test\./.test(entry.name))

  const skillDirs = (await listed(io, join(root, '.claude/skills'))).filter(entry => entry.kind === 'dir')
  let skills = 0
  for (const dir of skillDirs) if (await io.exists(join(root, '.claude/skills', dir.name, 'SKILL.md')).catch(() => false)) skills += 1

  const agentFiles = (await listed(io, join(root, '.claude/agents'))).filter(entry => entry.kind === 'file' && entry.name.endsWith('.md'))
  const agents = agentFiles.length
  const reviewerAgents = agentFiles.filter(entry => REVIEWER.test(entry.name)).length

  let goldenSets = 0
  for (const entry of await listed(io, join(root, '.claude/golden'))) {
    if (entry.kind !== 'file' || !/\.(md|jsonl)$/.test(entry.name)) continue
    if (goldenTasks(entry.name, (await readText(io, join(root, '.claude/golden', entry.name))) ?? '') >= GOLDEN_MIN) goldenSets += 1
  }

  const anyOf = async (paths: string[]): Promise<boolean> => {
    for (const path of paths) if (await io.exists(join(root, path)).catch(() => false)) return true
    return false
  }

  const worktrees = await countOf(io, ['git', 'worktree', 'list', '--porcelain'], root, out => out.split('\n').filter(line => line.startsWith('worktree ')).length)
  const drafts = await countOf(io, ['gh', 'pr', 'list', '--author', '@me', '--state', 'open', '--draft', '--limit', '1000', '--json', 'number'], root, out => {
    const value = JSON.parse(out) as unknown
    return Array.isArray(value) ? value.length : 0
  })

  const name = repo.split('/').filter(Boolean).pop() ?? repo
  return {
    root,
    repo,
    name,
    claudeMdLines: claudeMd === null ? null : claudeMd.replace(/\n$/, '').split('\n').length,
    alwaysOnBytes,
    rulesTotal: rules.length,
    rulesUnscoped,
    hooks: hooks.length,
    preToolUseHooks: hooks.filter(hook => hook.event === 'PreToolUse').length,
    guardHooks: hooks.filter(hook => hook.event === 'PreToolUse' && GUARD.test(hook.command)).length,
    hasHookTests,
    skills,
    agents,
    reviewerAgents,
    goldenSets,
    hasReviewMd: await anyOf(['REVIEW.md', '.github/REVIEW.md']),
    hasPrTemplate: await anyOf(['.github/pull_request_template.md', '.github/PULL_REQUEST_TEMPLATE.md', '.github/PULL_REQUEST_TEMPLATE', 'pull_request_template.md']),
    hasCodeowners: await anyOf(['.github/CODEOWNERS', 'CODEOWNERS', 'docs/CODEOWNERS']),
    broadApprovals: [...allowOf(shared), ...allowOf(local)].filter(isBroadApproval).length,
    isBypassDefault: shared.permissions?.defaultMode === 'bypassPermissions' || local.permissions?.defaultMode === 'bypassPermissions',
    worktrees,
    drafts,
  }
}
