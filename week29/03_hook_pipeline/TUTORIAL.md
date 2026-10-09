# ▶ Mod Gym Lesson 03 — The hook pipeline: observe, rewrite, wrap, answer

> Part of Week 29 · Claude Code mods. You learn to change how Claude Code itself looks and behaves, from inside the terminal you code in every day. These tutorials are the long form; **🏋 Mod Gym**, the panel inside Claude Code, is the short form with checks. Through the lessons you grow one lab mod, `hello-mod`; the finished version is in `week29/reference/hello-mod`.

**What you'll actually do**
- Read every hook as `($, e, next)` and learn the four things a hook can do with `next`: **observe**, **rewrite**, **wrap**, **answer**.
- Add three hooks to `hello-mod`: a tool-call counter (wrap), a line of context on every prompt (rewrite), and a check at the end of each turn (observe).
- Prove what they do with `claude plugin test`, with no session and no network.
- Find out which hook runs first when several hooks share an event, by measuring it.
- Know why a module variable like `calls` starts again from zero every time you save.
- Read the **Events** tab of Mod Gym as a live example: it is the panel's own hooks watching your session.

**Time** ~45 min · **Difficulty** beginner–intermediate · **Hardware** this laptop only

**Sources:** [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [React to events](https://code.claude.com/docs/en/plugins/mods/events) · [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · the API declarations Claude Code 2.1.292 writes (`.claude-plugin/types/claude-code/index.d.ts`, the authority whenever the docs and the code differ)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Lesson 02 | `claude plugin validate ~/mod-gym-lab/hello-mod` passes | this lesson adds hooks to that mod |
| Claude Code 2.1.287 or newer | `claude --version` | mods arrived in 2.1.287 |
| Mod Gym installed | `/mod-gym` opens the panel | Section 6 reads its Events tab |

The labs for this lesson are already built in the repo, so you can compare your mod with them:

```text
week29/03_hook_pipeline/
  labs/hello-mod-l03/              hello-mod as it stands after this lesson, with tests
  labs/order-demo/                 three hooks on one event, to measure the order
  exercises/solutions/heavy-set/   the exercise solution
```

## 1 · One shape for every hook

Every hook in every mod has the same shape:

```ts
on('<event>', { /* optional matcher */ }, async ($, e, next) => { … })
```

| name | what it is |
|---|---|
| `$` | the mods API: `$.ui`, `$.tool`, `$.store`, `$.process` … everything outside your module |
| `e` | the event's input, a plain frozen value (for `tool.call`: `e.tool`, `e.command`, …) |
| `next` | the rest of the chain: the plugins beneath you, then Claude Code's own behaviour |

`next(e)` resolves to the event's **result**. What you do around that one call is the whole pattern:

| you write | pattern | effect |
|---|---|---|
| `return next(e)` | **observe** | look, change nothing |
| `return next({ ...e, x })` | **rewrite** | everything beneath sees your version of `e` |
| `const r = await next(e); …; return r` | **wrap** | let it happen, then act on the result |
| `return { … }` and never call `next` | **answer** | you are the result; nothing beneath runs |

Lesson 02's spinner suffix was already a **rewrite**: it passed `next` a copy of `e` with a new `props.suffix`, and Claude Code drew that copy.

✓ Checkpoint: you can say which of the four patterns the Spinner hook from Lesson 02 is, and why.

## 2 · Add three hooks to hello-mod

Open `~/mod-gym-lab/hello-mod/hooks/register.ts` and make it look like this (the Spinner hook is from Lesson 02):

```ts
import type { Register } from 'claude-code'

export const register: Register = on => {
  // L02 · rewrite what Claude Code draws
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )

  // L03 · wrap: let every tool run, then count it
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    $.ui.status('hello-mod · ' + calls + ' tool calls')
    return r
  })

  // L03 · rewrite: add a line of context Claude reads with the prompt
  on('prompt.submit', ($, e, next) =>
    next({ ...e, context: [...(e.context ?? []), 'Reply like a fitness coach.'] }),
  )

  // L03 · observe: look at the finished turn, change nothing
  on('turn.complete', ($, e, next) => {
    if (e.durationMs > 120_000) $.ui.toast('Long set: ' + Math.round(e.durationMs / 1000) + 's')
    return next(e)
  })
}
```

Read each hook against the table in Section 1:

- **`tool.call` (no matcher)** runs for every tool Claude uses. It awaits `next(e)`, so the tool really runs first; then it counts and updates the status line. It returns `r`, the tool's own result, untouched. Forget the `return r` and Claude gets no result for its tool.
- **`prompt.submit`** passes on a copy of the prompt with one more entry in `context`. Context is text Claude reads with your prompt but you never see in the transcript. Your typed `text` is unchanged.
- **`turn.complete`** reads `e.durationMs` (it also has `e.answer`, `e.reason`, `e.isAborted`, `e.usage`) and returns `next(e)` unchanged.

Validate it. The `hooks:` line is what the engine will hook, and `calls:` is every `$` method the module can reach:

```bash
# on: laptop
claude plugin validate week29/03_hook_pipeline/labs/hello-mod-l03
```

**Expected output** (captured on this Mac, Claude Code 2.1.292; the manifest lines are trimmed)

```
…
  ❯ ./register.ts hooks: ui.render{component=Spinner}, tool.call, prompt.submit, turn.complete
  ❯ ./register.ts gating hook without .catch: tool.call
  ❯ ./register.ts gating hook without .catch: prompt.submit
  ❯ ./register.ts calls: $.ui.status, $.ui.toast
✔ Validation passed with warnings
```

The *warnings* are only the missing `author` field in `plugin.json`. A **gating hook** is a hook at a site whose answer could refuse what was asked: a tool call, a prompt. Validate states it as a fact, not a problem. Whether a gating hook needs `.catch` is Lesson 04's question.

✓ Checkpoint: your own `claude plugin validate ~/mod-gym-lab/hello-mod` shows `tool.call` and `prompt.submit` on the `hooks:` line.

## 3 · Prove it with the test kit

`claude plugin test` runs `tests/*.test.ts` against the real engine, with no session, sign-in or network. In a test, the hooks you add with the test's own `on` sit **beneath** your plugin and stand in for Claude Code:

```ts
import { expect, test } from 'claude-code/testing'

// The engine fills these fields in a real session; a test spells them out.
const ORIGIN = { kind: 'composer' } as const

test('wrap: the tool result comes back unchanged and the status counts it', async ($, on) => {
  const status: string[] = []
  on('ui.status', ($, e) => { status.push(String(e.text)); return { value: undefined } })
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran: ' + e.command }))
  const r = await $.tool.call({ tool: 'Bash', command: 'ls' })
  await $.tool.call({ tool: 'Bash', command: 'pwd' })
  expect(r.result).toBe('ran: ls')
  expect(status).toEqual(['hello-mod · 1 tool calls', 'hello-mod · 2 tool calls'])
})

test('rewrite: the prompt reaches Claude Code with one more line of context', async ($, on) => {
  on('prompt.submit', ($, e) => ({ text: e.text, context: e.context }))
  const r = await $.prompt.submit({ text: 'plan my leg day', wait: false, origin: ORIGIN })
  expect(r.text).toBe('plan my leg day')
  expect(r.context).toEqual(['Reply like a fitness coach.'])
})
```

Two things that trip people up:

- Stub hooks take `($, e)` like any hook. Writing `e => …` gives you `$` under the name `e`.
- A `$` call the plugin makes (here `$.ui.status`) is answered with `{ value }`. An event (`tool.call`, `prompt.submit`) is answered with that event's own result.

```bash
# on: laptop
claude plugin test week29/03_hook_pipeline/labs/hello-mod-l03
```

**Expected output** (captured on this Mac)

```
tests/pipeline.test.ts:
(pass) wrap: the tool result comes back unchanged and the status counts it [16.30ms]
(pass) rewrite: the prompt reaches Claude Code with one more line of context [6.55ms]

 2 pass
 0 fail
```

Copy `tests/pipeline.test.ts` into your own `hello-mod/tests/` and run it there too.

✓ Checkpoint: both tests pass against **your** mod.

## 4 · Who runs first? Measure it

When several hooks share an event, they form a stack. `labs/order-demo` puts three hooks on `tool.call` for Bash. A and B rewrite the command on the way in and tag the result on the way out. C answers `whoami` itself and never calls `next`:

```ts
export const register: Register = on => {
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const r = await next({ ...e, command: e.command + ' >A' })
    return { result: String(r.result) + ' <A' }
  })
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const r = await next({ ...e, command: e.command + ' >B' })
    return { result: String(r.result) + ' <B' }
  })
  // answer: C never calls next, so nothing beneath it runs
  on('tool.call', { tool: 'Bash' }, ($, e, next) =>
    e.command.startsWith('whoami') ? { result: 'coach (answered by C)' } : next(e),
  )
}
```

The test's own Bash stub stands in for the real tool and echoes the command it finally received:

```ts
test('the first hook registered is the outermost', async ($, on) => {
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran [' + e.command + ']' }))
  const r = await $.tool.call({ tool: 'Bash', command: 'ls' })
  expect(r.result).toBe('ran [ls >A >B] <B <A')
  const w = await $.tool.call({ tool: 'Bash', command: 'whoami' })
  expect(w.result).toBe('coach (answered by C) <B <A')
})
```

```bash
# on: laptop
claude plugin test week29/03_hook_pipeline/labs/order-demo
```

**Expected output** (captured on this Mac)

```
tests/order.test.ts:
(pass) the first hook registered is the outermost [15.87ms]

 1 pass
 0 fail
Ran 1 test across 1 file. [0.15s]
```

What the passing assertions tell you:

```text
call ──► A ──► B ──► C ──► (Claude Code's Bash)
            >A     >B        ran [ls >A >B]
result ◄── <A ◄── <B ◄───────────┘
```

- Inside one plugin, **the first hook registered is the outermost**. It sees the call first and the result last.
- A rewrite (`>A`) is seen by everything beneath it, never by anything above it.
- An answer (C on `whoami`) stops the call going down, but the hooks **above** still wrap its result (`<B <A`).
- **Between plugins**, the engine stacks them in tiers: `prepend` (where an organisation puts its policy plugin), `user`, `append`, `builtin`, `core`. Do not build on the order between two ordinary user plugins. Write each hook so it is correct wherever it sits.

✓ Checkpoint: you can predict, before running it, what `cat` would return through `order-demo`.

## 5 · Why `calls` goes back to zero

Save `register.ts` while a session has your mod loaded (or run `/reload-plugins`). The status line starts again at `1 tool calls`. A reload is a **fresh load of the module**: `register` runs again in a new environment, so every module-level or closure variable (`let calls = 0`) starts over, and `session.start` fires again.

| where a value lives | survives a save / reload | survives the session |
|---|---|---|
| a `let` in your module | no | no |
| `$.state` (an atom) | yes | no |
| `$.store` | yes | yes (per machine) |

Lesson 06 moves `calls` into an atom. Until then, treat the counter as "since the last save".

✓ Checkpoint: you can explain why the count drops to zero after a save, and name the place that would keep it.

## 6 · In the Mod Gym panel

Mod Gym is built from the same four patterns. Its `claude plugin validate` output shows the hooks it puts on your session:

```bash
# on: laptop
claude plugin validate week29/mod-gym
```

**Expected output** (captured on this Mac; the first two `❯ ./` lines, and the later ones trimmed)

```
…
  ❯ ./register.tsx hooks: session.start, prompt.submit, tool.call, turn.complete, tool.call{tool=mcp__mod-gym__gym_status}, command.run{command=mod-gym}, ui.render{component=AbovePrompt}, ui.render{component=Pane, requestId=mod-gym}
  ❯ ./register.tsx answers its own command: command.run{command=mod-gym}
…
```

- `prompt.submit` and `turn.complete` **observe**: they write one row to the Events tape and `return next(e)`.
- `tool.call` (no matcher) **wraps**: it times `await next(e)` and records the tool, a short argument, `ok` / `error` / `denied`, and the milliseconds.
- `tool.call{tool=mcp__mod-gym__gym_status}` **answers**: Mod Gym serves its own tool and never calls `next`.

Now watch it live:

1. Type `/mod-gym 3` (jumps to this lesson), then `/mod-gym` to open the panel.
2. Press **2** for the *Events* tab and send any prompt that makes Claude use a tool, e.g. "list the files in week29".
3. Read the rows: `◦` is observe, `⟳` is wrap, `◆` is answer. Each `tool.call` row shows the time the tool took **inside** Mod Gym's `await next(e)`.
4. Press **1**, then **c**. Mod Gym checks that your `register.ts` hooks `tool.call`, awaits `next(`, rewrites `prompt.submit`, and that `claude plugin validate` lists `tool.call`.
5. Press **a** to put a coaching question in your prompt. It asks Claude to label each of your hooks with its pattern.

✓ Checkpoint: the four Lesson 03 checks are ✓ in the panel.

## Try it yourself

Add a **heavy-set warning**: when one turn uses more than 5 tool calls, show a toast at the end of the turn. You need three hooks: one that starts a new count when a prompt is submitted, one that counts tool calls, one that reads the count when the turn completes. Then write a test that submits a prompt, makes 6 tool calls and completes the turn.

<details><summary>Solution — <code>exercises/solutions/heavy-set</code></summary>

```ts
import type { Register } from 'claude-code'

export const register: Register = on => {
  let turnCalls = 0

  // observe: a new prompt starts a new set
  on('prompt.submit', ($, e, next) => {
    turnCalls = 0
    return next(e)
  })

  // wrap: count each tool call of the turn
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    turnCalls += 1
    return r
  })

  // observe the end of the turn; toast after a heavy one
  on('turn.complete', ($, e, next) => {
    if (turnCalls > 5) $.ui.toast('Heavy set: ' + turnCalls + ' tool calls in one turn')
    return next(e)
  })
}
```

```ts
import { expect, test } from 'claude-code/testing'

test('a turn with 6 tool calls raises one toast', async ($, on) => {
  const toasts: string[] = []
  on('ui.toast', ($, e) => { toasts.push(String(e.text)); return { value: undefined } })
  on('prompt.submit', ($, e) => ({ text: e.text }))
  on('tool.call', () => ({ result: 'ok' }))
  on('turn.complete', ($, e) => ({ text: e.answer }))

  await $.prompt.submit({ text: 'refactor everything', wait: false, origin: { kind: 'composer' } })
  for (let i = 0; i < 6; i++) await $.tool.call({ tool: 'Read', file_path: '/lab/f' + i })
  await $.turn.complete({ turnId: 't1', answer: 'done', durationMs: 900, isAborted: false, reason: 'answer' })
  expect(toasts).toEqual(['Heavy set: 6 tool calls in one turn'])
})
```

```bash
# on: laptop
claude plugin test week29/03_hook_pipeline/exercises/solutions/heavy-set
```

**Expected output** (captured on this Mac)

```
tests/heavy.test.ts:
(pass) a turn with 6 tool calls raises one toast [17.14ms]

 1 pass
 0 fail
```

Why the reset lives in `prompt.submit` and not in `turn.complete`: a turn can end by being aborted, and the next prompt is the one moment you know a new set has started.

</details>

✓ Checkpoint: your heavy-set test passes.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Claude says a tool returned nothing after you added `tool.call` | the hook does not `return` the result of `next(e)` | `const r = await next(e); …; return r` |
| `hook was skipped: … no implementation for …` in a test | the plugin made a call nothing beneath it answers | add a stub with `on('<event>', ($, e) => …)` |
| A stub reads `undefined` from `e.text` | it was written `e => …`, so `e` is really `$` | write `($, e) => …` |
| The status count drops to 1 after a save | a reload starts the module again | expected; Lesson 06 keeps it in `$.state` |

## Next

[Lesson 04 — Guards: deny, hold and catch](../04_guards/TUTORIAL.md): a hook that **answers** `tool.call` with `{ deny }` stops a tool from running. Lesson 04 builds a force-push guard and shows, with a measured failure, why the order of the hooks you just learned about decides whether a guard holds.
