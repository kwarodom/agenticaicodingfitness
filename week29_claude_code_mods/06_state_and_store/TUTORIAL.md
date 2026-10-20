# ▶ Mod Gym Lesson 06 — State that survives a reload

> Part of Week 29 · Claude Code mods. You learn to change how Claude Code itself looks and behaves, from inside the terminal you code in every day. The course has two halves that teach the same eight lessons: these tutorials (the long form) and **🏋 Mod Gym**, a panel inside Claude Code (the short form, with checks). Mod Gym is itself a mod, so everything it does on screen is something you will learn to build. Through the lessons you grow one small lab mod, `hello-mod`; the finished version is in `week29_claude_code_mods/reference/hello-mod`.

**What you'll actually do**
- Choose where a value lives: a module variable, a `$.state` atom or `$.store`, by what happens to it on a reload, a `/clear` and tomorrow's session.
- Move `hello-mod`'s counters into atoms with `atom`, `read` and `update`, and watch the band redraw with no `$.ui.invalidate`.
- Write the `PluginState` contract in `types/index.d.ts`, name it in `plugin.json`, and meet the validator gotcha we measured: the shape must be written inline.
- See the rule "drawing is pure" enforced: a render hook that writes state is skipped.
- Keep `reps` across sessions in `$.store`, and test it with a store you can look inside.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** this laptop only

**Sources:** [Draw in the interface](https://code.claude.com/docs/en/plugins/mods/interface) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Use the mods API](https://code.claude.com/docs/en/plugins/mods/api) · [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · the type declarations Claude Code 2.1.292 writes for every mod (`.claude-plugin/types/claude-code/index.d.ts`), which win wherever a page and the types disagree

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Lesson 05 | `hello-mod` has `hooks/register.tsx`, a band and a pane | this lesson changes what the band reads |
| Claude Code 2.1.287+ | `claude --version` → `2.1.292 (Claude Code)` here | mods need it |
| Your lab folder | `claude plugin validate ~/mod-gym-lab/hello-mod` passes | Mod Gym checks this folder |

✓ Checkpoint: your band shows a tool-call count that comes from a `let calls` module variable.

## 1 · Three places to keep a value

| Where | Lives until | Shared with | Redraws its readers | Use it for |
|---|---|---|---|---|
| module `let` | the next save or reload of the module | this module only | no (`$.ui.invalidate` by hand) | caches you can lose |
| `$.state` atom | the session ends; also reset by `/clear`, `/resume`, `/branch` | the session (other plugins can read) | **yes** | what a drawing shows |
| `$.store` | forever, on this machine | every session of this plugin on this machine (up to 4 MiB) | no | progress, settings, history |

Why this matters with mods in particular: **a reload is a fresh load of the module**. When you save `register.tsx` during development, Claude Code runs `register` again in a new environment, fires `session.start` again, and drops the old timers. Every `let` starts over. Values in `$.state` (held by the host for the session) and in `$.store` (held on disk) stay.

So `let calls = 0` from Lesson 03 resets to zero each time you save. That is the bug this lesson fixes.

✓ Checkpoint: for each of `calls`, `reps` and "is the band hidden", you can say which row of the table it belongs in and why.

## 2 · Atoms: `atom`, `read`, `update`

An atom is a typed name for one value in `$.state`, with a starting value:

```ts
import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

// L06 · values the session keeps across reloads, declared in types/index.d.ts
const reps = atom({ plugin: 'hello-mod', key: 'reps' } as const, 0)
const calls = atom({ plugin: 'hello-mod', key: 'calls' } as const, 0)
const isBandHidden = atom({ plugin: 'hello-mod', key: 'isBandHidden' } as const, false)
```

- `plugin` and `key` must be **literals** (`as const`), so `claude plugin validate` can list what the module reads and writes. With `key: K` (a variable) validate fails: `the state library's atom takes a source the scan can read: a reference whose plugin and key are string literals …`.
- `await read($, reps)` reads it. Inside a render hook, that read also **subscribes** the drawing: when the value changes, exactly the drawings that read it are drawn again.
- `await update($, reps, n => n + 1)` reads, applies your function and writes, retrying if someone wrote in between, so two quick presses both count.

The counter becomes:

```ts
// L03 · wrap: let every tool run, then count it (now in an atom, not a let)
on('tool.call', async ($, e, next) => {
  const ran = await next(e)
  await update($, calls, n => n + 1)
  $.ui.status('hello-mod · ' + (await read($, calls)) + ' tool calls')
  return ran
})
```

No `$.ui.invalidate`: the band read `calls`, so writing `calls` redraws it.

And the band shows reps, with a button to add one and a button to hide the band:

```tsx
// L05 · a band above the prompt; reading atoms subscribes it to their changes
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
        <Button key="rep" onPress={() => update($, reps, n => n + 1)}>
          +1 rep
        </Button>
        <Button key="hide" dimColor onPress={() => update($, isBandHidden, () => true)}>
          Hide
        </Button>
      </Box>
    </Box>
  )
})
```

✓ Checkpoint: `register.tsx` has no `let calls` left, and no `$.ui.invalidate`.

## 3 · The contract: `types/index.d.ts`

Every atom must be declared in a **contract**: a `.d.ts` file that tells Claude Code (and TypeScript) the type of each value. Create `types/index.d.ts`:

```ts
declare module 'claude-code' {
  interface PluginState {
    'hello-mod': { reps: number; calls: number; isBandHidden: boolean }
  }
}
```

and name it in `.claude-plugin/plugin.json`:

```json
{
  "name": "hello-mod",
  "version": "0.6.0",
  "description": "My first mod",
  "types": "./types/index.d.ts"
}
```

Validate:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (captured on this Mac; paths shortened)

```
Validating plugin manifest: …/hello-mod/.claude-plugin/plugin.json

⚠ Found 1 warning:

  ❯ author: No author information provided. Consider adding author details for plugin attribution

  ❯ types ./types/index.d.ts declares on $: nothing (no EngineInterface member)
  ❯ types ./types/index.d.ts declares state: hello-mod.reps, hello-mod.calls, hello-mod.isBandHidden

Validating hooks: …/hello-mod/hooks/hooks.json

  ❯ ./register.tsx hooks: session.start, ui.render{component=Spinner}, tool.call{tool=Bash}, tool.call, ui.render{component=AbovePrompt}, state.set{plugin=hello-mod, key=reps}
  ❯ ./register.tsx gating hook with .catch: tool.call{tool=Bash}
  ❯ ./register.tsx gating hook without .catch: tool.call
  ❯ ./register.tsx gating hook without .catch: state.set{plugin=hello-mod, key=reps}
  ❯ ./register.tsx calls: $.state.get, $.state.set, $.store.get, $.store.set, $.ui.ask, $.ui.resolve, $.ui.status
  ❯ ./register.tsx state writes: hello-mod.calls, hello-mod.isBandHidden, hello-mod.reps
  ❯ ./register.tsx state reads: hello-mod.calls, hello-mod.isBandHidden, hello-mod.reps

✔ Validation passed with warnings
```

(This output already includes the `session.start` and `state.set` hooks from section 5.) Read the two `types` lines and the two `state` lines together: every key the module reads or writes is one the contract declares.

**The gotcha we measured: write the shape inline.** A tidy-looking contract that names the shape through a type alias is *not* read by the validator:

```ts
export type HelloModState = { reps: number; calls: number; isBandHidden: boolean }

declare module 'claude-code' {
  interface PluginState {
    'hello-mod': HelloModState
  }
}
```

```
✘ Found 3 errors:

  ❯ modules../register.tsx: hello-mod.calls is not declared: the manifest's types contract must name it in interface PluginState { hello-mod: { calls: ... } }
  ❯ modules../register.tsx: hello-mod.isBandHidden is not declared: the manifest's types contract must name it in interface PluginState { hello-mod: { isBandHidden: ... } }
  ❯ modules../register.tsx: hello-mod.reps is not declared: the manifest's types contract must name it in interface PluginState { hello-mod: { reps: ... } }


✘ Validation failed
```

You get the same three errors if you forget `"types"` in `plugin.json`. Aliases for the **value** types are fine (Mod Gym's contract uses `checks: CheckOutcome[]`); the object of keys under your plugin's name must be written out.

✓ Checkpoint: validate prints `declares state: hello-mod.reps, hello-mod.calls, hello-mod.isBandHidden`.

## 4 · Drawing is pure

A render hook **reads** state and returns a tree. It never writes. Writes belong in a handler (`onPress`) or in another event's hook (`tool.call`, `session.start`, …).

The validator cannot see this, so try it once to know the message. We put one wrong line in the band:

```tsx
await update($, calls, n => n + 1) // wrong: a render hook writes state
```

`claude plugin validate` still passes. Mounting the band in a test does not:

```
hello-mod's ui.render hook was skipped: hello-mod: HooksError: hello-mod: state.set: denied: it was made while ui.render is being dispatched, and drawing is pure (a render hook reads state and returns a tree; write from a handler such as onPress, or from another event) (host check)
```

A skipped band hook means no band at all. Take the line out.

One more rule from the docs: in a handler, write from what you read **now**, not from a value the closure captured while drawing. `update($, reps, n => n + 1)` does that for you; `$.state.set(reps, shownValue + 1)` with a value captured at draw time can lose a press.

✓ Checkpoint: you can say where each write in your mod happens (which handler or which event).

## 5 · Keep it past the session with `$.store`

`$.state` ends with the session. For reps that should still be there tomorrow, also write them to `$.store`, and load them in `session.start`:

```ts
// L06 · load what lasts across sessions
on('session.start', async ($, e, next) => {
  const saved = await $.store.get('reps')
  if (typeof saved === 'number') await update($, reps, () => saved)
  return next(e)
})

// L06 · keep reps past the session: whenever the atom changes, write it to the store
on('state.set', { plugin: 'hello-mod', key: 'reps' }, async ($, e, next) => {
  const r = await next(e)
  await $.store.set('reps', e.value)
  return r
})
```

`state.set` is the event every write to an atom goes through. Hooking your own key there gives you one place that mirrors the atom into the store, which is what `week29_claude_code_mods/reference/hello-mod` does; Lesson 07's `log_reps` tool writes `reps` from a `tool.call` hook, and that write reaches this hook.

**What we measured about button presses.** In `claude plugin test`, a write made by the band's own `onPress` handler redrew the band but did **not** reach the plugin's `state.set` hook: the store stayed empty. Writes from a `tool.call` hook did reach it. Until you have checked this in a live session, write the store in the handler too:

```tsx
<Button
  key="rep"
  onPress={async () => {
    await update($, reps, n => n + 1)
    await $.store.set('reps', await read($, reps))
  }}
>
  +1 rep
</Button>
```

Facts about `$.store` worth knowing:

- It lives on this machine: every session of `hello-mod` on this laptop sees the same `reps`, and two terminals open at once share it.
- Values must be plain JSON. Keep it small (the cap is 4 MiB).
- It survives reloads, `/clear` and restarts. To start over you delete the key yourself (`$.store.delete('reps')`).

✓ Checkpoint: `register.tsx` contains `$.store.set(` and a `session.start` hook that reads `$.store.get('reps')`.

## 6 · Test it with a store you can look inside

`mock.store(on)` gives your plugin a working store, but you cannot see inside it. To check what was written, answer `store.get` and `store.set` yourself. Save this as `tests/state.test.ts`:

```ts
import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

const START = { cwd: '/lab', surface: 'terminal', isInteractive: true } as const
const BAND = {
  component: 'AbovePrompt',
  props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns: 80, scroll: { offset: 0, bodyRows: 40 }, view: {} },
} as const

// A store you can look inside: the plugin's $.store calls land here.
function engine(on: On, store: Record<string, unknown>) {
  on('store.get', ($, e) => ({ value: store[e.key] }))
  on('store.set', ($, e) => { store[e.key] = e.value; return { value: undefined } })
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Box({}))
}

test('+1 rep redraws the band and is written to $.store', async ($, on) => {
  const store: Record<string, unknown> = {}
  engine(on, store)
  await $.session.start(START)
  const band = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  expect(await band.find({ type: 'Text', text: '0 reps' })).toBeDefined()
  await band.press({ key: 'rep' })
  await band.press({ key: 'rep' })
  expect(await band.find({ type: 'Text', text: '2 reps' })).toBeDefined()
  expect(store.reps).toBe(2)
  await band.unmount()
})

test('reps saved by an earlier session are loaded at session.start', async ($, on) => {
  engine(on, { reps: 7 })
  await $.session.start(START)
  const band = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  expect(await band.find({ type: 'Text', text: '7 reps' })).toBeDefined()
  await band.unmount()
})

test('Hide removes the band until the session ends', async ($, on) => {
  engine(on, {})
  await $.session.start(START)
  const band = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  await band.press({ key: 'hide' })
  expect(await band.find({ key: 'rep' })).toBeUndefined()
  await band.unmount()
})
```

`session.start` needs a full input in a test (`cwd`, `surface`, `isInteractive`); in a real session Claude Code fills it in.

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/hello-mod
```

**Expected output** with the `+1 rep` button that only calls `update` (captured on this Mac): the band redrew to `2 reps`, but the store stayed empty.

```
tests/state.test.ts:
(fail) +1 rep redraws the band and is written to $.store [24.41ms]
  AssertionError: expect(received).toBe()
  
  Expected: 2
  Received: undefined
(pass) reps saved by an earlier session are loaded at session.start [11.34ms]
(pass) Hide removes the band until the session ends [10.39ms]

 2 pass
 1 fail
Ran 3 tests across 1 file. [0.17s]
```

**Expected output** with the handler that also writes the store (section 5):

```
tests/state.test.ts:
(pass) +1 rep redraws the band and is written to $.store [26.36ms]
(pass) reps saved by an earlier session are loaded at session.start [11.77ms]
(pass) Hide removes the band until the session ends [10.04ms]

 3 pass
 0 fail
Ran 3 tests across 1 file. [0.19s]
```

This is the kind of thing a test is for: the screen looked right and the data was not saved.

✓ Checkpoint: three tests pass.

## In the Mod Gym panel

Type `/mod-gym 6`, then `/mod-gym` and press **c**. The five checks are:

| Check | Passes when |
|---|---|
| types/index.d.ts exists | `hello-mod/types/index.d.ts` is there |
| plugin.json names the types | `plugin.json` has a `"types"` field |
| uses an atom | `register.tsx` calls `atom(` |
| persists to $.store | `register.tsx` calls `$.store.set(` |
| validate passes (contract checked) | `claude plugin validate` exits 0, so every atom is declared |

Mod Gym itself is built this way: its lesson, tab, checks and Events tape are atoms (they survive a reload of the panel), while your passed lessons, lab folder and streak are in `$.store` (they survive tomorrow). Press **3** for the Progress tab to see the `$.store` side.

## Try it yourself — make Hide stick

Right now **Hide** lasts until the session ends. Make it last until you show the band again: store the choice in `$.store`, load it in `session.start`, and add a test that presses **Hide**, starts a new session, and finds no band.

Checks for yourself: the new test passes, and so do the three from section 6.

<details><summary>Solution</summary>

In `session.start`, after loading `reps`:

```ts
if ((await $.store.get('isBandHidden')) === true) await update($, isBandHidden, () => true)
```

The Hide button writes both:

```tsx
<Button
  key="hide"
  dimColor
  onPress={async () => {
    await update($, isBandHidden, () => true)
    await $.store.set('isBandHidden', true)
  }}
>
  Hide
</Button>
```

Add to `tests/state.test.ts`:

```ts
test('Hide is remembered by the next session', async ($, on) => {
  const store: Record<string, unknown> = {}
  engine(on, store)
  await $.session.start(START)
  const first = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  await first.press({ key: 'hide' })
  await first.unmount()
  expect(store.isBandHidden).toBe(true)

  await $.session.start(START) // what a reload or tomorrow's session runs
  const again = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  expect(await again.find({ key: 'rep' })).toBeUndefined()
  await again.unmount()
})
```

**Expected output** (captured on this Mac)

```
tests/state.test.ts:
(pass) +1 rep redraws the band and is written to $.store [24.44ms]
(pass) reps saved by an earlier session are loaded at session.start [10.60ms]
(pass) Hide removes the band until the session ends [10.54ms]
(pass) Hide is remembered by the next session [12.57ms]

 4 pass
 0 fail
Ran 4 tests across 1 file. [0.19s]
```

A way to show the band again is a command, which is Lesson 07: Mod Gym's `/mod-gym band on` does exactly this.

</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `hello-mod.reps is not declared` | no contract, no `"types"` in `plugin.json`, or the shape is a type alias | write `'hello-mod': { reps: number; … }` inline and name the file in `plugin.json` |
| The count resets to 0 when you save the file | it is still a module `let` | move it into an atom |
| `state.set: denied: … drawing is pure` | a render hook writes state | write from `onPress` or another event |
| The band shows the right number but tomorrow it is 0 | the write never reached `$.store` | write the store in the handler (section 5) and test it with an inspectable store |
| `the state library's atom takes a source the scan can read …` | `plugin` or `key` is a variable, not a literal | `atom({ plugin: 'hello-mod', key: 'reps' } as const, 0)` |
| A test fails with `no implementation for store.get` | the plugin reads the store and nothing answers | `mock.store(on)` or your own `store.get`/`store.set` stubs |

## Next

[Lesson 07 — Commands, tools and models](../07_commands_tools_models/TUTORIAL.md): add `/reps`, a command that answers without a Claude turn, and `log_reps`, a tool Claude can call to add to the same atom.
