// hello-mod: the Week 29 lab mod as it stands after lesson 08.
// Each block is labelled with the lesson that adds it.

import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

// L06 · values the session keeps across reloads, declared in types/index.d.ts
const reps = atom({ plugin: 'hello-mod', key: 'reps' } as const, 0)
const calls = atom({ plugin: 'hello-mod', key: 'calls' } as const, 0)
const isBandHidden = atom({ plugin: 'hello-mod', key: 'isBandHidden' } as const, false)

const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/

export const register: Register = on => {
  // L06 + L07 · load what lasts across sessions; add a command and a tool
  on('session.start', async ($, e, next) => {
    const saved = await $.store.get('reps')
    if (typeof saved === 'number') await update($, reps, () => saved)
    await $.command.register({ name: 'reps', description: 'Show the reps hello-mod has logged' })
    await $.tool.register({
      name: 'log_reps',
      description: 'Record reps the student finished. Call it when the student says they did reps.',
      inputSchema: { type: 'object', properties: { n: { type: 'number' } }, required: ['n'] },
    })
    return next(e)
  })

  // L02 · rewrite what Claude Code draws: a suffix on the spinner
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )

  // L04 · guard: hold a force push and ask; refuse if the guard itself fails.
  // Registered before the catch-all observers below, so it is the outer hook on Bash.
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (FORCE_PUSH.test(e.command)) {
      const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
      if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    }
    return next(e)
  }).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))

  // L03 · wrap: let every tool run, then count it
  on('tool.call', async ($, e, next) => {
    const ran = await next(e)
    await update($, calls, n => n + 1)
    $.ui.status('hello-mod · ' + (await read($, calls)) + ' tool calls')
    return ran
  })

  // L03 · rewrite: add a line of context Claude reads with the prompt
  on('prompt.submit', ($, e, next) =>
    next({ ...e, context: [...(e.context ?? []), 'hello-mod: reply like an encouraging fitness coach.'] }),
  )

  // L05 · draw: a band above the prompt, beneath what other plugins draw there
  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey || (await read($, isBandHidden))) return next(e)
    const { Box, Text, Button } = $.ui.resolve(e)
    const below = await next(e)
    return (
      <Box flexDirection="column">
        {below}
        <Box flexDirection="row" columnGap={1}>
          <Text color="success">hello-mod</Text>
          <Text dimColor>{await read($, reps)} reps</Text>
          <Button
            key="rep"
            onPress={async () => {
              // A write from a Button's own handler did not reach the state.set hook below
              // in our tests, so the handler saves to the store itself (lesson 06).
              await update($, reps, n => n + 1)
              await $.store.set('reps', await read($, reps))
            }}
          >
            +1 rep
          </Button>
          <Button key="hide" dimColor onPress={() => update($, isBandHidden, () => true)}>
            Hide
          </Button>
        </Box>
      </Box>
    )
  })

  // L06 · keep reps past the session: whenever the atom changes, write it to the store
  on('state.set', { plugin: 'hello-mod', key: 'reps' }, async ($, e, next) => {
    const r = await next(e)
    await $.store.set('reps', e.value)
    return r
  })

  // L07 · a command that answers without a Claude turn
  on('command.run', { command: 'reps' }, async $ => ({ text: (await read($, reps)) + ' reps logged' }))

  // L07 · serve the tool Claude sees as mcp__hello-mod__log_reps
  on('tool.call', { tool: 'mcp__hello-mod__log_reps' }, async ($, e) => {
    const n = Number(e.n)
    if (!Number.isFinite(n) || n <= 0) return { result: 'n must be a positive number' }
    await update($, reps, r => r + n)
    return { result: 'Logged ' + n + ' reps. Total: ' + (await read($, reps)) }
  })
}
