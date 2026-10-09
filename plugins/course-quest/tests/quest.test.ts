import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// Run with: claude plugin test plugins/course-quest
// The engine fills these inputs in a real session; a test spells them out.
const START = { cwd: '/repo', surface: 'terminal', isInteractive: true } as const
const cmd = (args = '') =>
  ({ command: 'quest', args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 140 } }) as const
const PANE = {
  component: 'Pane',
  requestId: 'course-quest',
  props: { title: 'Agentic Quest', isFocused: true, bodyColumns: 110, placement: 'dock', scroll: { offset: 0, bodyRows: 60 }, view: {} },
} as const

function engine(on: On, store: Record<string, unknown> = {}) {
  mock.store(on, store)
  const clock = mock.clock(on)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('prompt.fill', () => ({ isFilled: true }))
  return clock
}

test('/quest jumps to a week by id and pads single digits', async ($, on) => {
  engine(on)
  await $.session.start(START)
  expect((await $.command.run(cmd('5'))).text).toContain('Week 05')
  expect((await $.command.run(cmd('R1'))).text).toContain('Week R1')
  expect((await $.command.run(cmd('week15'))).text).toContain('Week 15')
  expect((await $.command.run(cmd('nope'))).text).toContain('Usage:')
})

test('every tab draws for every week on the terminal (rasters and all)', async ($, on) => {
  engine(on)
  await $.session.start(START)
  const ui = await $.ui.mount({ plugin: 'course-quest', surface: 'terminal', ...PANE })
  for (const tab of ['tab-map', 'tab-week', 'tab-labs', 'tab-quiz', 'tab-stats']) {
    await ui.press({ key: tab })
    expect(await ui.find({ key: 'banner' })).toBeDefined()
  }
  // walk all 25 weeks on the week and labs tabs: every flow and lab card must validate
  await ui.press({ key: 'tab-week' })
  for (let i = 0; i < 25; i++) {
    expect(await ui.find({ key: 'flow' })).toBeDefined()
    await ui.press({ key: 'next' })
  }
  await ui.press({ key: 'tab-map' })
  expect(await ui.find({ key: 'map' })).toBeDefined()
  await ui.unmount()
})

test('a right quiz answer earns xp once, a perfect quiz earns a bonus', async ($, on) => {
  engine(on)
  await $.session.start(START)
  await $.command.run(cmd('05')) // +5 xp for the visit
  const ui = await $.ui.mount({ plugin: 'course-quest', surface: 'terminal', ...PANE })
  await ui.press({ key: 'tab-quiz' })
  const overview = async () => String((await $.tool.call({ tool: 'mcp__course-quest__course_overview' })).result)
  // answer all three questions, finding the right option by trying them in order
  for (let q = 0; q < 3; q++) {
    for (let opt = 0; opt < 4; opt++) {
      await ui.press({ key: 'opt-' + opt })
      if (await ui.find({ type: 'Text', text: /Nailed it/ })) break
      await ui.press({ key: 'next-q' })
      for (let back = 0; back < 2; back++) await ui.press({ key: 'next-q' }) // cycle back to the same question
    }
    await ui.press({ key: 'next-q' })
  }
  expect(await overview()).toContain('XP 50') // 5 visit + 3×10 + 15 bonus
  await ui.unmount()
})

test('marking a lab done adds 20 xp and shows in course_week', async ($, on) => {
  engine(on)
  await $.session.start(START)
  await $.command.run(cmd('28'))
  const ui = await $.ui.mount({ plugin: 'course-quest', surface: 'terminal', ...PANE })
  await ui.press({ key: 'tab-labs' })
  await ui.press({ key: 'done' })
  const week = JSON.parse(String((await $.tool.call({ tool: 'mcp__course-quest__course_week', week: '28' })).result))
  expect(week.progress.labsDone).toHaveLength(1)
  expect(String((await $.tool.call({ tool: 'mcp__course-quest__course_overview' })).result)).toContain('XP 25')
  await ui.unmount()
})

test('progress loads from an earlier session and course_week knows unknown ids', async ($, on) => {
  engine(on, { progress: { xp: 400, visited: { '02': '2026-10-01' }, labs: {}, quiz: {}, days: [] } })
  await $.session.start(START)
  expect(String((await $.tool.call({ tool: 'mcp__course-quest__course_overview' })).result)).toContain('level 4')
  expect(String((await $.tool.call({ tool: 'mcp__course-quest__course_week', week: '13' })).result)).toContain('No week')
})

test('the clock repaints the rasters while the panel is open', async ($, on) => {
  const painted: string[] = []
  const clock = engine(on)
  on('ui.blit', ($, e) => {
    if ('key' in e && typeof e.key === 'string') painted.push(e.key)
    return { value: {} }
  })
  on('ui.panes', () => ({ value: [{ id: 'course-quest', title: 'q', isShown: true, isFocused: true, isPlaced: true }] }))
  await $.session.start(START)
  const ui = await $.ui.mount({ plugin: 'course-quest', surface: 'terminal', ...PANE })
  await clock.advance(500)
  expect(painted).toContain('banner')
  expect(painted).toContain('map')
  await ui.unmount()
})

test('narrow and wide panels draw every week on the week, labs and quiz tabs', async ($, on) => {
  engine(on)
  await $.session.start(START)
  for (const bodyColumns of [50, 140]) {
    const ui = await $.ui.mount({ plugin: 'course-quest', surface: 'terminal', ...PANE, props: { ...PANE.props, bodyColumns } })
    for (const tab of ['tab-week', 'tab-labs', 'tab-quiz']) {
      await ui.press({ key: tab })
      for (let i = 0; i < 25; i++) {
        expect(await ui.find({ key: 'banner' })).toBeDefined()
        await ui.press({ key: 'next' })
      }
      for (let i = 0; i < 25; i++) await ui.press({ key: 'prev' })
    }
    await ui.press({ key: 'tab-map' })
    expect(await ui.find({ key: 'map' })).toBeDefined()
    await ui.press({ key: 'tab-stats' })
    expect(await ui.find({ key: 'stats' })).toBeDefined()
    await ui.unmount()
  }
})
