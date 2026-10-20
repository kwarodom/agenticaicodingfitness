# Week 29 · authoring guide (for maintainers and agents)

Week 29 teaches **Claude Code mods** inside Claude Code. Each lesson exists twice, kept in step:

| where | what | edit when |
|---|---|---|
| `mod-gym/hooks/lessons.ts` | the short form shown in the panel: `body` (Markdown), `checks`, `ask` | the lesson's code or checks change |
| `NN_*/TUTORIAL.md` + `.th.md` | the long form, with captured outputs, labs and an exercise | the lesson's content changes |
| `reference/hello-mod/` | the lab mod after lesson 08; the source of truth for every snippet | an API or lesson change |

## Rules

- **The types file is the authority.** Code is checked against the `claude-code.d.ts` that Claude Code writes for the
  running build (the `plugin-authoring` skill names it; a loaded mod gets a copy in `.claude-plugin/types/`). The
  research document in this folder was written from the docs and not run; where they differ, the types win.
- **Every "Expected output" is captured, never written by hand.** Rebuild the example in a scratch folder and run it.
- **Every check must pass on `reference/hello-mod`.** After changing `lessons.ts` or the reference, point a session at
  it and run every lesson's checks. `/mod-gym check` records passes in your real `$.store`, so reset afterwards:

  ```bash
  claude -p "/mod-gym lab $PWD/week29_claude_code_mods/reference"
  for n in 1 2 3 4 5 6 7 8; do claude -p "/mod-gym $n" >/dev/null; claude -p "/mod-gym check"; done
  claude -p "/mod-gym reset"; claude -p "/mod-gym lab ~/mod-gym-lab"
  ```

- **Keep both plugins green:** `claude plugin validate` and `claude plugin test` on `mod-gym` and
  `reference/hello-mod`, and a strict `tsc` with the tsconfig from the types file's header.
- **Thai files keep every code block identical to the English.** Translate prose only.
- Labs in `04_guards/labs/` that fail on purpose (`guard-before`, `guard-without-catch`, `two-guards`) must keep
  failing: they show the bypass. Their tutorial says so.
- Check kinds (`hooks/lessons.ts`): `version`, `exists`, `contains` (regex; `register.ts` also matches
  `register.tsx`), `anyFile`, `validate` (exit 0, optional `mustInclude` regex on the output), `test`.

## Things we learned building it (2.1.292)

- Test-kit stubs are hooks: `on('fs.read', ($, e) => ({ value: … }))`, not `(e) => …`. `session.start` needs a
  `cwd` and something beneath the plugin answering it: `on('session.start', ($, e) => ({ cwd: e.cwd }))`.
- A plugin's refusal reaches a test's `$.tool.call` as `{ deny }`, not `isError`.
- Inside one plugin, the hook registered first is the outermost.
- `$.ui.ask` is a `tool.call` of `AskUserQuestion`; it rejects in `claude -p` and in tests with nothing to answer it.
- A catch-all `tool.call` observer with `.catch(($, e, next) => next(e))`, registered above a guard that calls
  `$.ui.ask`, let the guarded command run when the question failed. Guards go first; observers get no `.catch`
  (a hook that throws is skipped anyway). Covered in module 04 and by `reference/hello-mod/tests`.
- The validator reads a `PluginState` contract only when the shape is written inline
  (`'hello-mod': { reps: number }`); through a type alias it reported the keys as not declared.
- In the test kit, an atom written from a Button's own `onPress` redraws its readers but does not raise the plugin's
  `state.set` hook; writes from other events do. Persist to `$.store` in the handler itself (module 06).
- `claude plugin validate` passed a module containing `require('fs')`; it fails only when loaded.
- A matcher built by joining strings shows as `tool=?` in the validate report: write it as a literal.
- `$.ui.close` takes `{ id }`, not a bare string.
- Command output is already prefixed with the plugin's name; don't repeat it in `text`.
- A band (`AbovePrompt`) shared with other plugins: call `await next(e)` and draw it inside your own `Box`, or you
  erase theirs.
