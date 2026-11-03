import type { On } from 'claude-code'
import { expect, mock, test } from 'claude-code/testing'

import { ASKED, BAND, NOW, PANE, ROOT, TYPED, greenRepo, memoryStore, serve, words } from './kit'

type Engine = Parameters<Extract<Parameters<typeof test>[1], (...args: never[]) => unknown>>[0]

const band = async ($: Engine): Promise<string> => {
  try {
    const ui = await $.ui.mount(BAND)
    const text = await words(ui)
    await ui.unmount()
    return text
  } catch {
    // The mod drew nothing, and nothing is beneath it in a test.
    return ''
  }
}

const submit = async ($: Engine, text: string): Promise<void> => {
  await $.prompt.submit({ text, ...TYPED }).catch(() => undefined)
}

const answerPrompts = (on: On): void => {
  on('prompt.submit', (_$, e) => ({ text: e.text }) as never)
}

test('/belt status scores the repository from what is on disk', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())

  // Act
  const text = String((await $.command.run({ command: 'belt', args: 'status', ...ASKED })).text)

  // Assert
  expect(text).toContain('test-repo:')
  expect(text).toContain('CLAUDE.md 222 lines')
  expect(text).toContain('1 broad pre-approved commands')
  expect(text).toContain('Next:')
  expect(text).not.toContain('undefined')
})

test('two corrections in a row put one line above the prompt; a normal prompt clears it', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())
  answerPrompts(on)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await submit($, 'No, that is the wrong file')
  await submit($, 'still failing on the same test')
  const after2 = await band($)
  await submit($, 'Thanks. Now add the Thai label.')
  const afterPlain = await band($)

  // Assert
  expect(after2).toContain('Two corrections in a row')
  expect(afterPlain).toBe('')
})

test('a third correction names the two-strike rule, and /clear takes it down', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())
  answerPrompts(on)
  on('session.end', () => ({ sessionId: 's1' }) as never)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  for (const text of ['No, wrong file', 'still failing', "that's not what I asked"]) await submit($, text)
  const struck = await band($)
  await $.session.end({ reason: 'clear', sessionId: 's1', resume: {} } as never)
  const cleared = await band($)

  // Assert
  expect(struck).toContain('3 corrections in a row')
  expect(cleared).toBe('')
})

test('a commit with no test run before it is flagged; one after a test run is not', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())
  on('tool.call', () => ({ result: { stdout: 'ok' } }) as never)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await $.tool.call({ tool: 'Bash', command: 'git commit -m "x"' } as never)
  const flagged = await band($)
  await $.tool.call({ tool: 'Bash', command: 'npx vitest run' } as never)
  const afterTests = await band($)
  await $.tool.call({ tool: 'Bash', command: 'git commit -m "y"' } as never)
  const second = await band($)

  // Assert
  expect(flagged).toContain('no test run before it')
  expect(afterTests).toBe('')
  expect(second).toBe('')
})

test('a refused tool call counts for nothing', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())
  on('tool.call', () => ({ deny: 'held by a test hook' }) as never)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await $.tool.call({ tool: 'Bash', command: 'git commit -m "x"' } as never)

  // Assert
  expect(await band($)).toBe('')
})

test('a PR created without Testing, Risk or How to revert names what is missing', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())
  on('tool.call', () => ({ result: { stdout: 'https://example.test/pr/1' } }) as never)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await $.tool.call({ tool: 'Bash', command: 'gh pr create --title t --body "## Summary\nok\n## Testing\npassed"' } as never)

  // Assert
  expect(await band($)).toContain('no Risk, How to revert section')
})

test('a turn that ends with the context over 70% says so above the prompt', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())
  on('session.usage', () => ({ value: { startedAt: NOW, context: { window: 200_000, tokens: 160_000, percent: 80 }, rateLimits: [] } }) as never)
  on('turn.complete', () => ({ text: 'done' }) as never)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await $.turn.complete({ answer: 'done', durationMs: 10, isAborted: false, turnId: 't1' } as never)

  // Assert
  expect(await band($)).toContain('Context is 80% full')
})

test('declarations are kept across sessions and shown as declared', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  const held = memoryStore(on)
  serve(on, greenRepo())
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  const said = String((await $.command.run({ command: 'belt', args: 'declare container docker + STOP file', ...ASKED })).text)
  const wrong = String((await $.command.run({ command: 'belt', args: 'declare everything', ...ASKED })).text)
  const stored = [...held.keys()]

  // Assert
  expect(said).toContain('Declared container')
  expect(wrong).toContain('Declare one of')
  expect(stored.some(key => key.endsWith('.declared'))).toBe(true)
})

test('no prompt text reaches the store', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  const held = memoryStore(on)
  serve(on, greenRepo())
  answerPrompts(on)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await submit($, 'No, the secret project codename is BLUEHERON')
  const values = [...held.values()]

  // Assert
  expect(JSON.stringify(values)).not.toContain('BLUEHERON')
  expect(JSON.stringify(values)).toContain('"corrections":1')
})

test('the pane shows the belt, eight competencies and the next changes, on terminal and desktop', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  serve(on, greenRepo())
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  for (const surface of ['terminal', 'desktop'] as const) {
    // Act
    const ui = await $.ui.mount({ ...PANE, surface } as never)
    const text = await words(ui)
    await ui.unmount()

    // Assert
    expect(text).toContain('Alto Belt · test-repo')
    expect(text).toContain('Accountability and safety')
    expect(text).toContain('Your next three moves')
    expect(text).toContain('Prove it works')
    expect(text).not.toContain('undefined')
  }
})

test('with coaching off, Claude\'s instructions are unchanged; with it on, one section is added', async ($, on) => {
  // Arrange
  on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'hi', scope: 'shared' }] }) as never)
  const input = { model: 'm', promptModel: 'm', surfaces: ['terminal'], tools: [], outputStyle: null, traits: [] } as never

  // Act
  const off = await $.prompt.compose(input)

  // Assert
  expect(off.sections.map(one => one.id)).toEqual(['intro'])
})

test('with coaching on, one coaching section is added last', { options: { coach_model: true } }, async ($, on) => {
  // Arrange
  on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'hi', scope: 'shared' }] }) as never)
  const input = { model: 'm', promptModel: 'm', surfaces: ['terminal'], tools: [], outputStyle: null, traits: [] } as never

  // Act
  const composed = await $.prompt.compose(input)

  // Assert
  expect(composed.sections.map(one => one.id)).toEqual(['intro', 'alto-belt:coach'])
})

test('the repository is keyed by its main checkout, so a worktree shares its counts', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  const held = memoryStore(on)
  serve(on, greenRepo())
  answerPrompts(on)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await submit($, 'Add the seat meter')
  const keys = [...held.keys()]

  // Assert
  expect(keys.length).toBeGreaterThan(0)
  expect(keys.every(key => key.startsWith('alto-belt.test-repo-'))).toBe(true)
  expect(keys.some(key => key.includes(ROOT))).toBe(false)
})

test('a bare /belt always opens the pane, even when the engine thinks it is already shown', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  mock.store(on)
  const opened: string[] = []
  serve(on, greenRepo(), { opened })
  let closed = 0
  on('ui.panes', () => ({ value: [{ id: 'belt', isShown: true }] }) as never)
  on('ui.close', () => {
    closed += 1
    return { value: undefined } as never
  })

  // Act
  const first = String((await $.command.run({ command: 'belt', args: '', ...ASKED })).text)
  await $.command.run({ command: 'belt', args: '', ...ASKED })

  // Assert
  expect(opened).toEqual(['belt', 'belt'])
  expect(closed).toBe(0)
  expect(first).toContain('/belt close')
})

test('a reviewer run that names its PR is kept as a round, with the day only and none of its prompt', async ($, on) => {
  // Arrange
  mock.clock(on, { now: NOW })
  const held = memoryStore(on)
  serve(on, greenRepo())
  on('tool.call', () => ({ result: { stdout: 'Nothing to fix' } }) as never)
  await $.command.run({ command: 'belt', args: 'status', ...ASKED })

  // Act
  await $.tool.call({ tool: 'Agent', subagent_type: 'trust-security', description: 'Review PR 869 safety plane', prompt: 'The diff is in BLUEHERON' } as never)
  await $.tool.call({ tool: 'Agent', subagent_type: 'trust-security', description: 'Golden run 2 task 4', prompt: 'Replay #683' } as never)
  const rounds = [...held.entries()].find(([key]) => key.endsWith('.rounds'))?.[1]
  const status = String((await $.command.run({ command: 'belt', args: 'status', ...ASKED })).text)
  const declared = String((await $.command.run({ command: 'belt', args: 'declare rounds by hand', ...ASKED })).text)

  // Assert
  expect(rounds).toEqual({ '869': { 'trust-security': ['2026-10-29'] } })
  expect(JSON.stringify([...held.values()])).not.toContain('BLUEHERON')
  expect(status).toContain('rounds 1 over 1 PR')
  expect(declared).toContain('measured now')
})
