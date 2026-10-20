# ▶ Mod Gym Lesson 05 — Draw: panes, bands, status, toast

> Part of Week 29 · Claude Code mods. You learn to change how Claude Code itself looks and behaves, from inside the terminal you code in every day. The course has two halves that teach the same eight lessons: these tutorials (the long form) and **🏋 Mod Gym**, a panel inside Claude Code (the short form, with checks). Mod Gym is itself a mod, so everything it does on screen is something you will learn to build. Through the lessons you grow one small lab mod, `hello-mod`; the finished version is in `week29_claude_code_mods/reference/hello-mod`.

**What you'll actually do**
- Pick the right place to show something: the status line, a toast, a band above the prompt, or a pane.
- Draw with elements from `$.ui.resolve(e)` (not imports), and see how the element table changes per surface: terminal, desktop, VS Code, mobile.
- Add a band to `hello-mod` that draws **beneath** whatever other plugins already put there, instead of erasing it.
- Open a pane from a button with `$.ui.open`, and draw it with a `ui.render` hook on `Pane`.
- Read the dim *refused … the engine drew its own* line, and write a UI test that mounts your band and pane on two surfaces and presses their buttons.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** this laptop only

**Sources:** [Draw in the interface](https://code.claude.com/docs/en/plugins/mods/interface) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Use the mods API](https://code.claude.com/docs/en/plugins/mods/api) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · the type declarations Claude Code 2.1.292 writes for every mod (`.claude-plugin/types/claude-code/index.d.ts`), which win wherever a page and the types disagree

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Lessons 02–04 | your `hello-mod` has the spinner suffix, the tool counter and the force-push guard | this lesson adds drawing on top of them |
| Claude Code 2.1.287+ | `claude --version` → `2.1.292 (Claude Code)` here | mods need it |
| Your lab folder | `ls ~/mod-gym-lab/hello-mod/hooks` shows `hooks.json register.ts` | Mod Gym checks this folder (change it with `/mod-gym lab <path>`) |

✓ Checkpoint: `claude plugin validate ~/mod-gym-lab/hello-mod` ends with `✔ Validation passed`.

## 1 · Four ways to put something on screen

| You want | Call | Stays | Starts a Claude turn? |
|---|---|---|---|
| a short fact, always visible | `$.ui.status(text)` (`undefined` clears it) | until you change it | no |
| a heads-up that goes away | `$.ui.toast(text)` | a few seconds | no |
| a row above the prompt | a `ui.render` hook on `{ component: 'AbovePrompt' }` | every redraw | no |
| a side panel, like Mod Gym | `$.ui.open({ id, title })` + a `ui.render` hook on `{ component: 'Pane', requestId: id }` | until closed | no |

The first two are plain calls you can make from any hook. The last two are **render hooks**: Claude Code asks your hook "what goes here?" every time it draws that spot, and your hook answers with a tree of elements, or with `next(e)` to leave the spot to the rest of the chain.

You have already used `$.ui.status` in Lesson 03. Mod Gym uses all four: the `Mod Gym L01 · 0/8` status entry, the *lesson passed 💪* toast, the `🏋 Mod Gym` band and the panel itself.

✓ Checkpoint: you can say which of the four Mod Gym used when you passed Lesson 04.

## 2 · Elements come from the surface, not from imports

A mod has no React, no DOM and no element globals. The elements a hook may draw with come from the surface that asked:

```tsx
const { Box, Text, Button, Markdown } = $.ui.resolve(e)
```

`e.surface` says where the drawing goes, and each surface has its own table:

| Surface | Elements |
|---|---|
| `terminal` | `Box` `Text` `Button` `Input` `Select` `Link` `Code` `Markdown` `Client` `Raster` `Image` |
| `desktop` | `Box` `Text` `Button` `Input` `Select` `Svg` `Link` `Code` `Markdown` `Client` |
| `vscode` | `Box` `Text` `Button` `Input` `Select` `Svg` `Link` `Code` `Markdown` |
| `mobile` | `Box` `Text` `Button` `Svg` `Link` `Code` `Markdown` (no input fields yet) |

So a band you draw with `Box`, `Text` and `Button` works on every surface; one that draws an `Svg` would not work in the terminal. Write JSX: it compiles against the global `h`, and the file must end in `.tsx`.

A few props you will use all the time:

| Element | Props worth knowing |
|---|---|
| `Box` | `flexDirection`, `columnGap`, `rowGap`, `width`, `flexGrow`, `overflow` |
| `Text` | `color` (a theme key such as `success`, `error`, `claude`, or a color), `bold`, `dimColor`, `wrap="truncate-end"` |
| `Button` | `key`, `onPress`, `hotkey` (one digit or lowercase letter), `variant="primary"`, `role="dismiss"`, `dimColor`; the label is its one string child |
| `Markdown` | `text`: headings, lists, tables and code drawn as Claude's own replies are |

`e.props.bodyColumns` (on `AbovePrompt` and `Pane`) is the width of the box you draw into. Size your tree to it, not to the whole terminal: a docked pane is narrower than the screen.

✓ Checkpoint: you can say why a band that uses `Svg` would be drawn on desktop but not in the terminal.

## 3 · Rename to `.tsx` first

JSX in a `.ts` file does not parse. If you add the band below and forget to rename, validate says so:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (captured on this Mac; paths shortened)

```
Validating hooks: …/hello-mod/hooks/hooks.json

✘ Found 1 error:

  ❯ modules../register.ts: jsx: …/hello-mod/hooks/register.ts does not parse: Expected ">" but found "flexDirection" (line 38, column 12); Expected ")" but found "width" (line 38, column 35); Expected "}" but found "." (line 38, column 43); Expected ";" but found "props" (line 38, column 44); Unexpected .… [+21 chars]

✘ Validation failed
```

Rename the file and point `hooks.json` at the new name:

```bash
# on: laptop
cd ~/mod-gym-lab/hello-mod
mv hooks/register.ts hooks/register.tsx
echo '{ "modules": ["./register.tsx"] }' > hooks/hooks.json
```

Rename without updating `hooks.json`, and validate tells you that too:

```
  ❯ modules../register.ts: stale: …/hello-mod/hooks/register.ts: no such file
```

✓ Checkpoint: `hooks/hooks.json` names `./register.tsx`, and that file exists.

## 4 · A band that shares the row

Add this hook to `register.tsx`. It draws one row: the mod's name, the tool-call count from Lesson 03, and two buttons.

```tsx
// L05 · a band above the prompt, drawn beneath what other plugins put there
on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
  if (e.props.hasSurvey) return next(e)
  const { Box, Text, Button } = $.ui.resolve(e)
  const below = await next(e)
  return (
    <Box flexDirection="column" width={e.props.bodyColumns}>
      {below}
      <Box flexDirection="row" columnGap={1}>
        <Text color="success">hello-mod</Text>
        <Text dimColor>{calls} tool calls</Text>
        <Button key="hi" onPress={() => $.ui.toast('Hi from hello-mod!')}>
          Say hi
        </Button>
        <Button key="stats" variant="primary" onPress={() => void $.ui.open({ id: PANE, title: 'hello-mod stats' })}>
          Stats
        </Button>
      </Box>
    </Box>
  )
})
```

Three things in it are on purpose:

1. **`await next(e)` and `{below}`.** The band is one spot shared by every plugin. A hook that returns its own tree **without** calling `next` answers for the whole chain, so the plugins beneath it never draw. Mod Gym draws its `🏋 Mod Gym` row there; Clean View draws its plan there. Calling `next(e)` first and putting its result in your column keeps their rows and adds yours.
2. **`if (e.props.hasSurvey) return next(e)`.** When Claude Code shows its own survey in that spot, step aside.
3. **`key` on every `Button`.** Keys are how presses find their handler, and how tests find the button.

The handlers run in your plugin. `onPress` may be `async`, and it is the right place to change things (Lesson 06 shows why a render hook must not).

**Redrawing.** `calls` is still the module variable from Lesson 03. A render hook that reads a plain variable does not know when it changes, so the counter hook asks for a redraw itself:

```ts
on('tool.call', async ($, e, next) => {
  const ran = await next(e)
  calls += 1
  $.ui.status('hello-mod · ' + calls + ' tool calls')
  $.ui.invalidate('ui.render') // a module variable changed: ask for a redraw
  return ran
})
```

Lesson 06 replaces this with `$.state`, which redraws its readers by itself.

✓ Checkpoint: you can explain what would happen to Mod Gym's band if your hook returned its tree without calling `next(e)`.

## 5 · A pane opened by a button

The *Stats* button above calls `$.ui.open({ id: PANE, title: 'hello-mod stats' })`. Opening only reserves the space; a render hook matched on that `requestId` draws it:

```tsx
const PANE = 'hello-stats'

// L05 · a pane: opened by $.ui.open, drawn by this hook
on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
  const { Box, Text, Button, Markdown } = $.ui.resolve(e)
  return (
    <Box flexDirection="column" rowGap={1} width={e.props.bodyColumns}>
      <Text bold>hello-mod stats</Text>
      <Markdown key="about" text={'**' + calls + '** tool calls this session.\n\n- the band above the prompt\n- this pane\n- the status line'} />
      <Button key="close" hotkey="q" role="dismiss" onPress={() => $.ui.close({ id: PANE })}>
        Close
      </Button>
    </Box>
  )
})
```

Notes from the types:

- A pane opened by something the person did (a button press, a command) seats at any width. One opened unasked, from `session.start` or a timer, only seats from 144 terminal columns, so open panes from a press or a command.
- `focus`, `closeOnEscape` and `holdToasts` accept only `true`. `focus: true, closeOnEscape: true` makes the pane take the keys, and **Esc** closes it; Mod Gym opens its panel this way.
- `hotkey="q"` presses the button while the pane holds the keyboard. `role="dismiss"` marks it as the close control: the terminal draws it as a normal button, a desktop as its native close mark.
- This hook does not call `next`: the pane is yours alone, so there is nothing beneath to keep.

Validate the whole mod:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (captured on this Mac; paths shortened)

```
Validating hooks: …/hello-mod/hooks/hooks.json

  ❯ ./register.tsx hooks: ui.render{component=Spinner}, tool.call{tool=Bash}, tool.call, ui.render{component=AbovePrompt}, ui.render{component=Pane, requestId=hello-stats}
  ❯ ./register.tsx gating hook with .catch: tool.call{tool=Bash}
  ❯ ./register.tsx gating hook without .catch: tool.call
  ❯ ./register.tsx calls: $.ui.ask, $.ui.close, $.ui.invalidate, $.ui.open, $.ui.resolve, $.ui.status, $.ui.toast

✔ Validation passed with warnings
```

(The one warning is `author: No author information provided`: add an `author` to `plugin.json` if you like.)

Load it in a second terminal and try it: `claude --plugin-dir ~/mod-gym-lab/hello-mod`. The band appears under the prompt's other rows; **Say hi** shows a toast; **Stats** opens the pane; **q** or **Esc** closes it.

✓ Checkpoint: validate lists `ui.render{component=AbovePrompt}` and `ui.render{component=Pane, requestId=hello-stats}`.

## 6 · When a tree is refused

Claude Code checks every tree against the surface's element table before it draws. A tree that does not fit (an element the surface lacks, a prop an element does not take) is **not drawn**: the engine draws its own instead and says why. While a session hot-reloads your folder the transcript shows it as one dim line; otherwise it goes to the debug log (`claude --debug`).

Here is a real one. We gave the band's `Box` a prop it does not have, `sparkle="yes"`, and mounted it in a test:

```
hooks: ui.render (AbovePrompt) refused: Box prop "sparkle" is not allowed; the engine drew its own
```

And a `Button` whose child is an element rather than a string:

```
hello-mod's ui.render hook was skipped: hello-mod: HooksError: JSX element <Button> needs a label: the label prop, or one string child
```

Two habits catch these before you load the mod:

- **Type-check.** Once Claude Code has loaded your mod, `.claude-plugin/types/` holds the API and a `tsconfig.json`; `npx -p typescript tsc -p ~/mod-gym-lab/hello-mod` then flags the bad prop: `Property 'sparkle' does not exist on type 'IntrinsicAttributes & BoxProps & ElementChildren'`.
- **Mount it in a test** (next section). A refused tree fails the test with the same reason.

✓ Checkpoint: you know where the *refused* line appears and what "the engine drew its own" means for your band.

## 7 · Test the drawing on two surfaces

`claude plugin test` can mount a component through your plugin on a surface you name, press its buttons by key and find elements. Save this as `tests/draw.test.ts`:

```ts
import { expect, test } from 'claude-code/testing'

// Mount props a real session fills in; a test spells them out.
const SITE = { scroll: { offset: 0, bodyRows: 40 }, view: {} } as const
const BAND = { component: 'AbovePrompt', props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns: 80, ...SITE } } as const
const PANE = { component: 'Pane', requestId: 'hello-stats', props: { title: 'hello-mod stats', isFocused: true, bodyColumns: 60, placement: 'dock', ...SITE } } as const

test('the band draws on terminal and desktop, and its buttons act', async ($, on) => {
  const toasts: string[] = []
  const opened: string[] = []
  on('ui.toast', ($, e) => { toasts.push(e.text); return { value: undefined } })
  on('ui.open', ($, e) => { opened.push(e.id); return { value: { isPlaced: true } } })
  // Beneath the band stands Claude Code's own (empty) band: draw it as an empty Box.
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Box({}))

  for (const surface of ['terminal', 'desktop'] as const) {
    const band = await $.ui.mount({ plugin: 'hello-mod', surface, ...BAND })
    expect(await band.find({ type: 'Text', text: 'hello-mod' })).toBeDefined()
    await band.press({ key: 'hi' })
    await band.press({ key: 'stats' })
    await band.unmount()

    const pane = await $.ui.mount({ plugin: 'hello-mod', surface, ...PANE })
    expect(await pane.find({ type: 'Text', text: 'hello-mod stats' })).toBeDefined()
    expect(await pane.find({ key: 'close' })).toBeDefined()
    await pane.unmount()
  }
  expect(toasts).toEqual(['Hi from hello-mod!', 'Hi from hello-mod!'])
  expect(opened).toEqual(['hello-stats', 'hello-stats'])
})
```

What to notice:

- The hooks the test adds with `on(...)` sit **beneath** your plugin and stand in for Claude Code. They take `($, e)` like any hook. The `ui.toast` and `ui.open` stubs record what your buttons asked for.
- Because your band calls `next(e)`, something must answer beneath it. Without the `ui.render` stub the mount fails with `no implementation for ui.render`.
- The loop over `['terminal', 'desktop']` proves the band does not depend on one surface.

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/hello-mod
```

**Expected output** (captured on this Mac)

```
tests/draw.test.ts:
(pass) the band draws on terminal and desktop, and its buttons act [27.81ms]

 1 pass
 0 fail
Ran 1 test across 1 file. [0.15s]
```

✓ Checkpoint: one test passes, and it ran on both surfaces.

## In the Mod Gym panel

Type `/mod-gym 5` to jump to this lesson, then `/mod-gym` to open the panel and press **c**. The four checks are:

| Check | Passes when |
|---|---|
| hooks.json points at the JSX module | `hooks/hooks.json` names `register.tsx` |
| draws with ui.render | `register.tsx` has `on('ui.render', { component: 'AbovePrompt' …` or `'Pane'` |
| gets elements from $.ui.resolve(e) | `register.tsx` calls `$.ui.resolve(` |
| validate passes | `claude plugin validate` exits 0 |

Press **a** to put this lesson's question in your prompt for Claude, and **2** to watch the Events tape while you try your band.

## Try it yourself — a band that fits a narrow terminal

At 40 columns your band's row is too long. Change the band so that when `e.props.bodyColumns` is under 50 it keeps only the name and the **Stats** button, then add a test that mounts the band at 100 and at 40 columns on both surfaces.

Checks for yourself: at 100 columns `find({ key: 'hi' })` is defined; at 40 it is `undefined` and `find({ key: 'stats' })` is defined.

<details><summary>Solution</summary>

In the band hook, decide once, then leave out what does not fit:

```tsx
const isNarrow = e.props.bodyColumns < 50
const below = await next(e)
return (
  <Box flexDirection="column" width={e.props.bodyColumns}>
    {below}
    <Box flexDirection="row" columnGap={1}>
      <Text color="success">hello-mod</Text>
      {!isNarrow && <Text dimColor>{calls} tool calls</Text>}
      {!isNarrow && (
        <Button key="hi" onPress={() => $.ui.toast('Hi from hello-mod!')}>
          Say hi
        </Button>
      )}
      <Button key="stats" variant="primary" onPress={() => void $.ui.open({ id: PANE, title: 'hello-mod stats' })}>
        Stats
      </Button>
    </Box>
  </Box>
)
```

`tests/narrow.test.ts`:

```ts
import { expect, test } from 'claude-code/testing'

const SITE = { scroll: { offset: 0, bodyRows: 40 }, view: {} } as const
const band = (bodyColumns: number) =>
  ({ component: 'AbovePrompt', props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns, ...SITE } }) as const

test('a narrow band keeps only the name and Stats', async ($, on) => {
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Box({}))
  for (const surface of ['terminal', 'desktop'] as const) {
    const wide = await $.ui.mount({ plugin: 'hello-mod', surface, ...band(100) })
    expect(await wide.find({ key: 'hi' })).toBeDefined()
    await wide.unmount()

    const narrow = await $.ui.mount({ plugin: 'hello-mod', surface, ...band(40) })
    expect(await narrow.find({ key: 'hi' })).toBeUndefined()
    expect(await narrow.find({ key: 'stats' })).toBeDefined()
    await narrow.unmount()
  }
})
```

**Expected output** (captured on this Mac)

```
tests/narrow.test.ts:
(pass) a narrow band keeps only the name and Stats [27.21ms]

 2 pass
 0 fail
Ran 2 tests across 2 files. [0.17s]
```

</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `does not parse: Expected ">" but found …` | JSX in a `.ts` file | rename to `.tsx` and update `hooks.json` |
| `stale: …/register.ts: no such file` | you renamed the file but not `hooks.json` | `{ "modules": ["./register.tsx"] }` |
| Mod Gym's band (or Clean View's) vanished | your band returns a tree without `next(e)` | `const below = await next(e)` and put `{below}` in your column |
| `refused: Box prop "…" is not allowed` | a prop the element does not take | check the element's props in the types; type-check with `tsc` |
| `<Button> needs a label` | a Button with an element child | give it one string child or a `label` prop |
| The count in the band does not move | it reads a module variable and nothing asked for a redraw | `$.ui.invalidate('ui.render')`, or use `$.state` (Lesson 06) |
| A test fails with `no implementation for ui.render` | your hook calls `next(e)` and nothing answers beneath it | stub `ui.render` for that component, as in section 7 |
| A pane opened from `session.start` never shows | unasked panes seat only from 144 columns | open it from a button press or a command |

## Next

[Lesson 06 — State that survives a reload](../06_state_and_store/TUTORIAL.md): move `calls` out of a module variable into `$.state`, so your band redraws by itself and keeps its count when you save the file.
