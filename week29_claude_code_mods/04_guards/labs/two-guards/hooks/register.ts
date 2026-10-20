import type { Register } from 'claude-code'

const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/
const WIDE_DELETE = /\brm\s+-[a-z]*r[a-z]*\s+(\/|~|\$HOME|\.\.)/

export const register: Register = on => {
  // a SECOND guard on Bash, stacked above the force-push guard
  on('tool.call', { tool: 'Bash' }, ($, e, next) =>
    WIDE_DELETE.test(e.command)
      ? { deny: 'hello-mod: recursive delete outside the lab folder refused: ' + e.command }
      : next(e),
  ).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the delete guard failed.' }))

  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (FORCE_PUSH.test(e.command)) {
      const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
      if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    }
    return next(e)
  }).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))

  // L03 · the observer, second, with no .catch
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    return r
  })
}
