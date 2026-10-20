// The eight Mod Gym lessons. Each one is the short, in-terminal form of a
// week29/NN_*/TUTORIAL.md module: what to know, what to do, and the checks the
// gym runs against the student's lab mod (`<labDir>/hello-mod`).

export type Check =
  | { kind: 'version'; label: string; min: string }
  | { kind: 'exists'; label: string; path: string }
  | { kind: 'contains'; label: string; path: string; pattern: string }
  | { kind: 'anyFile'; label: string; dir: string; suffix: string }
  | { kind: 'validate'; label: string; mod: string; mustInclude?: string }
  | { kind: 'test'; label: string; mod: string }

export type Lesson = {
  id: string
  title: string
  tutorial: string
  body: string
  checks: Check[]
  ask: string
}

const MOD = 'hello-mod'
const REG = MOD + '/hooks/register.ts'

export const LESSONS: Lesson[] = [
  {
    id: '01',
    title: 'What a mod is',
    tutorial: 'week29/01_what_is_a_mod/TUTORIAL.md',
    body: `A **mod** is a plugin whose \`hooks/hooks.json\` names a \`modules\` file: TypeScript functions that run *inside* Claude Code. Every hook is \`($, e, next)\` and can **observe**, **rewrite** or **answer** an event: a tool call, a prompt, a piece of the screen being drawn.

| piece | runs as | draws UI |
|---|---|---|
| skill | SKILL.md that Claude reads | no |
| settings hook | shell / HTTP on an event | no |
| MCP server | a separate process | no |
| **mod** | functions in Claude Code | **yes** |

A mod is **not sandboxed**: it runs with your permissions. Read \`claude plugin validate <dir>\` (its *hooks* and *calls* lines) before you load anyone's mod.

**Do this**
1. Press **c** to check your Claude Code version (mods need 2.1.287+).
2. Press **2** for the *Events* tab, then send any prompt. Each row is this panel's own hooks watching the event go by.`,
    checks: [{ kind: 'version', label: 'Claude Code is 2.1.287 or newer', min: '2.1.287' }],
    ask: 'Using the Mod Gym panel open in my terminal as the example, explain in 5 short bullets how a Claude Code mod differs from a skill, a settings hook and an MCP server. End with one risk of installing a stranger\'s mod.',
  },
  {
    id: '02',
    title: 'Your first mod: three files',
    tutorial: 'week29/02_first_mod/TUTORIAL.md',
    body: `Make a folder \`hello-mod\` in your lab folder with three files:

\`\`\`text
hello-mod/.claude-plugin/plugin.json
hello-mod/hooks/hooks.json
hello-mod/hooks/register.ts
\`\`\`

\`plugin.json\`: \`{ "name": "hello-mod", "version": "0.1.0", "description": "My first mod" }\`
\`hooks.json\`: \`{ "modules": ["./register.ts"] }\`. The \`modules\` key is what makes a plugin a mod.

\`\`\`ts
import type { Register } from 'claude-code'

export const register: Register = on => {
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )
}
\`\`\`

**Do this**
1. Write the three files. (Press **f** to copy the starter command.)
2. \`claude plugin validate <lab>/hello-mod\`
3. In a second terminal: \`claude --plugin-dir <lab>/hello-mod\`, ask anything, watch the spinner.`,
    checks: [
      { kind: 'exists', label: 'plugin.json exists', path: MOD + '/.claude-plugin/plugin.json' },
      { kind: 'contains', label: 'hooks.json names a module', path: MOD + '/hooks/hooks.json', pattern: '"modules"\\s*:\\s*\\[' },
      { kind: 'contains', label: 'register.ts hooks the Spinner', path: REG, pattern: "component:\\s*['\"]Spinner" },
      { kind: 'validate', label: 'claude plugin validate passes', mod: MOD },
    ],
    ask: 'Help me write my first Claude Code mod in my Mod Gym lab folder (hello-mod) that adds a suffix to the spinner. Explain each of the three files, then show me the validate command. Do not write the files for me unless I ask.',
  },
  {
    id: '03',
    title: 'The hook pipeline: observe, rewrite, answer',
    tutorial: 'week29/03_hook_pipeline/TUTORIAL.md',
    body: `\`next(e)\` runs the plugins beneath you and then Claude Code itself. What you do around it is the whole pattern:

| you write | effect |
|---|---|
| \`return next(e)\` | **observe** |
| \`return next({ ...e, x })\` | **rewrite** what the rest sees |
| \`const r = await next(e); …; return r\` | **wrap**: act on the result |
| \`return { … }\` with no \`next\` | **answer** for Claude Code |

\`\`\`ts
let calls = 0
on('tool.call', async ($, e, next) => {
  const r = await next(e)
  calls += 1
  $.ui.status('hello-mod · ' + calls + ' tool calls')
  return r
})
on('prompt.submit', ($, e, next) =>
  next({ ...e, context: [...(e.context ?? []), 'Reply like a fitness coach.'] }),
)
\`\`\`

**Do this**
1. Add both hooks to \`hello-mod\`. It reloads on save.
2. Open the *Events* tab (**2**): the gym wraps the same events, so you see each one and its timing.
3. Why does a module variable like \`calls\` reset when you save? (Lesson 6.)`,
    checks: [
      { kind: 'contains', label: 'hooks tool.call', path: REG, pattern: "on\\(\\s*['\"]tool\\.call" },
      { kind: 'contains', label: 'awaits next(e) to wrap a result', path: REG, pattern: 'await\\s+next\\(' },
      { kind: 'contains', label: 'rewrites prompt.submit', path: REG, pattern: "on\\(\\s*['\"]prompt\\.submit" },
      { kind: 'validate', label: 'validate lists tool.call', mod: MOD, mustInclude: 'tool\\.call' },
    ],
    ask: 'Look at the Events tab of my Mod Gym panel and my hello-mod register.ts. For each hook I wrote, tell me if it observes, rewrites, wraps or answers, and what would break if I forgot to call next(e).',
  },
  {
    id: '04',
    title: 'Guards: deny, hold and catch',
    tutorial: 'week29/04_guards/TUTORIAL.md',
    body: `A hook that returns \`{ deny }\` from \`tool.call\` stops the tool: it never runs, and Claude reads your reason as the result.

\`\`\`ts
on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
  if (/git\\s+push\\b.*(--force|-f\\b)/.test(e.command)) {
    const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
    if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod.' }
  }
  return next(e)
}).catch(($, e, next) =>
  next.called ? next(e) : { deny: 'hello-mod: the guard failed.' },
)
\`\`\`

A hook that throws is **skipped**. For a guard that means the dangerous call goes through. \`.catch\` decides what happens instead: refuse unless \`next\` already ran.

Register guards **before** catch-all observers, and do not give an observer \`.catch(($, e, next) => next(e))\`. In our tests that combination let a force push run when the question could not be asked (as in \`claude -p\`).

**Do this**
1. Add the guard. Ask Claude to run \`git push --force\` in a scratch repo and watch the question.
2. Validate: look for **gating hook with .catch** in the output.`,
    checks: [
      { kind: 'contains', label: 'returns a deny', path: REG, pattern: 'deny\\s*:' },
      { kind: 'contains', label: 'has a .catch handler', path: REG, pattern: '\\.catch\\(' },
      { kind: 'validate', label: 'validate reports a gating hook with .catch', mod: MOD, mustInclude: 'gating hook with \\.catch' },
    ],
    ask: 'Review the guard in my hello-mod register.ts like a security reviewer: which dangerous commands slip past my regex, what happens if the hook throws, and how would the blast-radius sample mod handle it?',
  },
  {
    id: '05',
    title: 'Draw: panes, bands, status, toast',
    tutorial: 'week29/05_drawing/TUTORIAL.md',
    body: `Four ways to put something on screen:

| want | call |
|---|---|
| a line in the status bar | \`$.ui.status(text)\` |
| a short popup | \`$.ui.toast(text)\` |
| a row above the prompt | \`ui.render\` on \`AbovePrompt\` |
| a side panel (like this one) | \`$.ui.open({ id })\` + \`ui.render\` on \`Pane\` |

Elements come from the surface, not from imports:

\`\`\`tsx
on('ui.render', { component: 'AbovePrompt' }, ($, e, next) => {
  const { Box, Text, Button } = $.ui.resolve(e)
  return (
    <Box columnGap={1}>
      <Text color="success">hello-mod</Text>
      <Button key="hi" onPress={() => $.ui.toast('Hi!')}>Say hi</Button>
    </Box>
  )
})
\`\`\`

Rename \`register.ts\` to \`register.tsx\` for JSX, and update \`hooks.json\`.

**Do this**
1. Add a band (above) or a pane opened by a command.
2. Watch the transcript for a dim *refused* line if a tree does not validate.`,
    checks: [
      { kind: 'contains', label: 'hooks.json points at the JSX module', path: MOD + '/hooks/hooks.json', pattern: 'register\\.tsx' },
      { kind: 'contains', label: 'draws with ui.render', path: MOD + '/hooks/register.tsx', pattern: "on\\(\\s*['\"]ui\\.render['\"]\\s*,\\s*\\{\\s*component:\\s*['\"](AbovePrompt|Pane)" },
      { kind: 'contains', label: 'gets elements from $.ui.resolve(e)', path: MOD + '/hooks/register.tsx', pattern: '\\$\\.ui\\.resolve\\(' },
      { kind: 'validate', label: 'validate passes', mod: MOD },
    ],
    ask: 'I want my hello-mod to show a small band above the prompt with my workout streak and a Hide button. Sketch the ui.render hook with $.ui.resolve(e) and explain why the elements are not imported.',
  },
  {
    id: '06',
    title: 'State that survives a reload',
    tutorial: 'week29/06_state_and_store/TUTORIAL.md',
    body: `Three places to keep a value:

| where | lives until | redraws |
|---|---|---|
| module \`let\` | the next save / reload | no |
| \`$.state\` atom | the session ends | **yes**, readers redraw |
| \`$.store\` | forever (per machine) | no |

\`\`\`ts
import { atom, read, update } from 'claude-code'
const reps = atom({ plugin: 'hello-mod', key: 'reps' } as const, 0)
// while drawing:   const n = await read($, reps)
// in a handler:    onPress: () => update($, reps, n => n + 1)
\`\`\`

Declare it in \`types/index.d.ts\`, and name that file in \`plugin.json\` as \`"types": "./types/index.d.ts"\`:

\`\`\`ts
declare module 'claude-code' {
  interface PluginState { 'hello-mod': { reps: number } }
}
\`\`\`

**Do this**
1. Move your counter to an atom; save the file; the count stays.
2. Also write it to \`$.store\` so tomorrow's session sees it.`,
    checks: [
      { kind: 'exists', label: 'types/index.d.ts exists', path: MOD + '/types/index.d.ts' },
      { kind: 'contains', label: 'plugin.json names the types', path: MOD + '/.claude-plugin/plugin.json', pattern: '"types"\\s*:' },
      { kind: 'contains', label: 'uses an atom', path: MOD + '/hooks/register.tsx', pattern: 'atom\\(' },
      { kind: 'contains', label: 'persists to $.store', path: MOD + '/hooks/register.tsx', pattern: '\\$\\.store\\.set\\(' },
      { kind: 'validate', label: 'validate passes (contract checked)', mod: MOD },
    ],
    ask: 'Explain module variables vs $.state vs $.store in my hello-mod with a concrete reload scenario, then check my types/index.d.ts contract matches every atom I use.',
  },
  {
    id: '07',
    title: 'Commands, tools and models',
    tutorial: 'week29/07_commands_tools_models/TUTORIAL.md',
    body: `A mod can add things that Claude and you can call:

\`\`\`ts
on('session.start', async ($, e, next) => {
  await $.command.register({ name: 'reps', description: 'Show my reps' })
  await $.tool.register({
    name: 'log_reps',
    description: 'Record reps the student finished.',
    inputSchema: { type: 'object', properties: { n: { type: 'number' } }, required: ['n'] },
  })
  return next(e)
})
on('command.run', { command: 'reps' }, async $ => ({ text: (await read($, reps)) + ' reps logged' }))
on('tool.call', { tool: 'mcp__hello-mod__log_reps' }, async ($, e) => {
  await update($, reps, n => n + Number(e.n))
  return { result: 'logged ' + e.n }
})
\`\`\`

A command runs **without a Claude turn**. A tool is listed to Claude as \`mcp__<plugin>__<name>\`. Use \`$.model.complete({ model: 'haiku', prompt })\` for a cheap side question, \`$.process.run([...])\` for a program, \`$.http.fetch\` for a local model on your own box.

**Do this**
1. Add \`/reps\` and the \`log_reps\` tool. Ask Claude: "log 10 reps".`,
    checks: [
      { kind: 'contains', label: 'registers a slash command', path: MOD + '/hooks/register.tsx', pattern: '\\$\\.command\\.register\\(' },
      { kind: 'contains', label: 'registers a tool for Claude', path: MOD + '/hooks/register.tsx', pattern: '\\$\\.tool\\.register\\(' },
      { kind: 'contains', label: 'serves mcp__hello-mod__…', path: MOD + '/hooks/register.tsx', pattern: 'mcp__hello-mod__' },
      { kind: 'validate', label: 'validate lists command.run', mod: MOD, mustInclude: 'command\\.run' },
    ],
    ask: 'In my hello-mod, when should something be a slash command, a tool Claude calls, or a $.model.complete call? Give one example of each from a hotel HVAC copilot.',
  },
  {
    id: '08',
    title: 'Test, validate, ship',
    tutorial: 'week29/08_test_and_ship/TUTORIAL.md',
    body: `\`claude plugin test <dir>\` runs \`tests/*.test.ts\` with no session, sign-in or network. Your stubs stand in for Claude Code:

\`\`\`ts
import { expect, mock, test } from 'claude-code/testing'

test('/reps starts at zero', async ($, on) => {
  mock.store(on)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  await $.session.start({ cwd: '/lab', surface: 'terminal', isInteractive: true })
  const r = await $.command.run({ command: 'reps', args: '',
    origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } })
  expect(r.text).toContain('0 reps')
})
\`\`\`

Stub hooks take \`($, e)\` like any hook. Nothing beneath them: a call you did not stub fails with *no implementation for …*.

Ship it: add \`.claude-plugin/marketplace.json\` next to \`plugin.json\`, push the folder to GitHub, and a classmate types:

\`\`\`text
/plugin install hello-mod --marketplace <you>/<repo>
\`\`\`

Bump \`version\` every time you ship: an installed copy is cached by version.

**Do this**
1. Write one test, run \`claude plugin test <lab>/hello-mod\` until it is green.
2. Add the marketplace file and validate again.`,
    checks: [
      { kind: 'anyFile', label: 'has a tests/*.test.ts', dir: MOD + '/tests', suffix: '.test.ts' },
      { kind: 'test', label: 'claude plugin test passes', mod: MOD },
      { kind: 'exists', label: 'marketplace.json exists', path: MOD + '/.claude-plugin/marketplace.json' },
      { kind: 'validate', label: 'validate passes with the marketplace', mod: MOD },
    ],
    ask: 'Read my hello-mod tests and suggest the one missing test that would have caught a real bug. Then walk me through publishing it so a classmate can /plugin install it.',
  },
]
