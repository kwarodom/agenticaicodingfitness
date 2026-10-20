import type { Register } from 'claude-code'

const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/
const WIDE_DELETE = /\brm\s+-[a-z]*r[a-z]*\s+(\/|~|\$HOME|\.\.)/

export const register: Register = on => {
  // L04 · one guard for Bash, registered first; every Bash rule lives here
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (WIDE_DELETE.test(e.command)) {
      return { deny: 'hello-mod: recursive delete outside the lab folder refused: ' + e.command }
    }
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
