import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// The engine fills these fields in a real session; a test spells them out.
const START = { cwd: '/home/student', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

// Run with: claude plugin test week29/reference/hello-mod
// Every hook `on` adds here sits beneath the plugin and answers in Claude Code's place.

function engine(on: On, saved: Record<string, unknown> = {}) {
  mock.store(on, saved)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
}

test('/reps starts at zero and log_reps adds to it', async ($, on) => {
  engine(on)
  await $.session.start(START)
  expect((await $.command.run(cmd('reps', ''))).text).toContain('0 reps')
  const r = await $.tool.call({ tool: 'mcp__hello-mod__log_reps', n: 10 })
  expect(String(r.result)).toContain('Total: 10')
  expect((await $.command.run(cmd('reps', ''))).text).toContain('10 reps')
})

test('reps saved by an earlier session are loaded from $.store', async ($, on) => {
  engine(on, { reps: 7 })
  await $.session.start(START)
  expect((await $.command.run(cmd('reps', ''))).text).toContain('7 reps')
})

test('a force push is refused when no one can answer the question (the guard fails closed)', async ($, on) => {
  engine(on)
  // $.ui.ask is a tool.call of AskUserQuestion; with nothing here to answer it, it rejects
  on('tool.call', { tool: 'Bash' }, () => ({ result: 'pushed' }))
  await $.session.start(START)
  const r = await $.tool.call({ tool: 'Bash', command: 'git push --force origin main' })
  expect(r.deny).toContain('force-push guard failed')
  const ok = await $.tool.call({ tool: 'Bash', command: 'git push origin feature' })
  expect(ok.result).toBe('pushed')
})

test('the prompt gets the coach context', async ($, on) => {
  engine(on)
  on('prompt.submit', ($, e) => ({ text: e.text, context: e.context }))
  await $.session.start(START)
  const r = await $.prompt.submit({ text: 'hi', wait: false, origin: { kind: 'composer' } })
  expect(r.context).toContain('hello-mod: reply like an encouraging fitness coach.')
})

test('the band\'s +1 rep button saves to $.store', async ($, on) => {
  const saved: Record<string, unknown> = {}
  on('store.get', ($, e) => ({ value: saved[e.key] }))
  on('store.set', ($, e) => {
    saved[e.key] = e.value
    return { value: undefined }
  })
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Box({})) // the empty band beneath
  await $.session.start(START)
  const band = {
    component: 'AbovePrompt',
    props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns: 80, scroll: { offset: 0, bodyRows: 4 }, view: {} },
  } as const
  const ui = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...band })
  await ui.press({ key: 'rep' })
  await ui.press({ key: 'rep' })
  expect(saved.reps).toBe(2)
  await ui.unmount()
})
