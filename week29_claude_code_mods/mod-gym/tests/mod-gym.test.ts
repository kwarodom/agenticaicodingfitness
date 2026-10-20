import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// The engine fills these fields in a real session; a test spells them out.
const START = { cwd: '/home/student', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

// `claude plugin test` runs these with no session, sign-in or network. The hooks
// `on` registers here sit beneath the plugin and answer in Claude Code's place.

const PANE = { component: 'Pane', requestId: 'mod-gym', props: { title: 'Mod Gym', isFocused: true, bodyColumns: 80, placement: 'dock', scroll: { offset: 0, bodyRows: 40 }, view: {} } } as const

function engine(on: On, files: Record<string, string> = {}, run = { exitCode: 0, stdout: '✔ Validation passed', stderr: '' }) {
  mock.store(on)
  mock.clock(on)
  mock.env(on, { HOME: '/home/student' })
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('fs.exists', ($, e) => ({ value: Object.keys(files).some(p => p === e.path || p.startsWith(e.path + '/')) }))
  on('fs.read', ($, e) => ({ value: files[e.path] ?? '' }))
  on('fs.list', ($, e) => ({
    value: Object.keys(files)
      .filter(p => p.startsWith(e.path + '/'))
      .map(p => ({ name: p.slice(e.path.length + 1), kind: 'file' as const, size: 1, mtimeMs: 0, isLink: false })),
  }))
  on('process.run', () => ({ value: { ...run, isStdoutTruncated: false, isStderrTruncated: false } }))
}

const LAB = '/home/student/mod-gym-lab/hello-mod'

test('/mod-gym moves between lessons and reports status', async ($, on) => {
  engine(on)
  await $.session.start(START)
  expect((await $.command.run(cmd('mod-gym', 'next'))).text).toContain('lesson 02')
  expect((await $.command.run(cmd('mod-gym', '5'))).text).toContain('lesson 05: Draw')
  expect((await $.command.run(cmd('mod-gym', 'prev'))).text).toContain('lesson 04: Guards')
  expect((await $.command.run(cmd('mod-gym', 'status'))).text).toContain('/home/student/mod-gym-lab')
  expect((await $.command.run(cmd('mod-gym', 'nonsense'))).text).toContain('Usage:')
})

test('lesson 02 checks fail on an empty lab and pass on the three files', async ($, on) => {
  const files: Record<string, string> = {}
  engine(on, files)
  await $.session.start(START)
  await $.command.run(cmd('mod-gym', '2'))

  const empty = (await $.command.run(cmd('mod-gym', 'check'))).text ?? ''
  expect(empty).toContain('0/4 checks')
  expect(empty).toContain('✗ plugin.json exists')

  files[LAB + '/.claude-plugin/plugin.json'] = '{ "name": "hello-mod", "version": "0.1.0" }'
  files[LAB + '/hooks/hooks.json'] = '{ "modules": ["./register.ts"] }'
  files[LAB + '/hooks/register.ts'] = "on('ui.render', { component: 'Spinner' }, ($, e, next) => next(e))"
  const full = (await $.command.run(cmd('mod-gym', 'check'))).text ?? ''
  expect(full).toContain('4/4 checks')
  expect(full).toContain('Lesson passed')
  expect((await $.command.run(cmd('mod-gym', 'status'))).text).toContain('passed 1/8')
})

test('a validate check reads the output it asks for', async ($, on) => {
  const files = { [LAB + '/hooks/register.tsx']: 'x', [LAB + '/hooks/register.ts']: "on('tool.call', async ($, e, next) => { await next(e) }); on('prompt.submit', f)" }
  engine(on, files, { exitCode: 0, stdout: 'hooks: session.start', stderr: '' })
  await $.session.start(START)
  await $.command.run(cmd('mod-gym', '3'))
  const text = (await $.command.run(cmd('mod-gym', 'check'))).text ?? ''
  expect(text).toContain('✓ hooks tool.call')
  expect(text).toContain('✗ validate lists tool.call (output lacks')
})

test('the tool.call wrap records to the tape and hands the result back unchanged', async ($, on) => {
  engine(on)
  on('tool.call', { tool: 'Bash' }, () => ({ result: 'file1\nfile2' }))
  await $.session.start(START)
  const r = await $.tool.call({ tool: 'Bash', command: 'ls' })
  expect(r.result).toBe('file1\nfile2')
  const status = await $.tool.call({ tool: 'mcp__mod-gym__gym_status' })
  expect(String(status.result)).toContain('"title": "What a mod is"')
  const ui = await $.ui.mount({ plugin: 'mod-gym', surface: 'terminal', ...PANE })
  await ui.press({ key: 'tab-events' })
  expect(await ui.find({ type: 'Text', text: /Bash: ls → ok/ })).toBeDefined()
  await ui.unmount()
})

test('the pane draws the lesson and its buttons move it', async ($, on) => {
  engine(on)
  await $.session.start(START)
  for (const surface of ['terminal', 'desktop'] as const) {
    await $.command.run(cmd('mod-gym', '1'))
    const ui = await $.ui.mount({ plugin: 'mod-gym', surface, ...PANE })
    expect(await ui.find({ type: 'Text', text: /Lesson 01 of 08/ })).toBeDefined()
    await ui.press({ key: 'next' })
    expect(await ui.find({ type: 'Text', text: /Lesson 02 of 08 · Your first mod/ })).toBeDefined()
    expect(await ui.find({ key: 'starter' })).toBeDefined()
    await ui.press({ key: 'tab-progress' })
    expect(await ui.find({ type: 'Text', text: /▶ 02 Your first mod/ })).toBeDefined()
    await ui.press({ key: 'tab-lesson' })
    await ui.unmount()
  }
})
