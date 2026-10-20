# Week 29 · Claude Code mods, learned inside Claude Code

On 1 October 2026 Anthropic added **mods** to Claude Code: small TypeScript functions, shipped in a plugin, that run
*inside* Claude Code and can watch, rewrite or take over what it does — a tool call, a prompt, a piece of the screen.
This week you learn to write them, and you learn it in the place they run: the course comes with **🏋 Mod Gym**, a
mod that opens a panel in your terminal, teaches one lesson at a time, checks the mod you are building, and shows its
own hooks working on your session live.

You build one mod across the week, `hello-mod`, from three files to a tested, shippable plugin with a guard, a band
above the prompt, state that survives reloads, a slash command and a tool Claude can call. Eight modules, each with a
tutorial in English and Thai and a lesson in the panel.

## Start

You need **Claude Code 2.1.287 or newer** (`claude --version`; this course was built and checked on 2.1.292).

```bash
# on: laptop, from the repo root — install Mod Gym from this folder
claude plugin marketplace add ./week29
claude plugin install mod-gym@fitness-week29
```

Then, inside Claude Code (start a new session or run `/reload-plugins`):

```text
/mod-gym
```

Once this week is pushed to GitHub, students who already use the class marketplace install it the usual way:

```text
/plugin marketplace update agentic-coding-fitness
/plugin install mod-gym@agentic-coding-fitness
```

### The panel

| key | does |
|---|---|
| **1 2 3** | tabs: *Lesson* · *Events* (live hook tape) · *Progress* |
| **p / n** | previous / next lesson |
| **c** | run the lesson's checks on your lab mod (`<lab>/hello-mod`) |
| **a** | put a coaching question about this lesson in your prompt, for you to edit and send |
| **f** | lesson 02 only: copy a shell script that writes the three starter files |
| **q** / Esc | close |

Commands: `/mod-gym [open|next|prev|1-8|check|status|lab <path>|band on|off|reset]`. Your lab folder defaults to
`~/mod-gym-lab`; change it with `/mod-gym lab <path>`. A one-line band above the prompt shows where you are
(`/mod-gym band off` hides it), and the status line shows `Mod Gym L03 · 2/8`. Claude can call the mod's
`gym_status` tool to see your lesson, lab folder and last check results, so it can coach without guessing.

## Modules

| # | module | you learn | you add to `hello-mod` |
|---|---|---|---|
| 01 | [What a mod is](01_what_is_a_mod/TUTORIAL.md) | mods vs skills, settings hooks and MCP; the trust model; auditing with `claude plugin validate`; touring Mod Gym | — |
| 02 | [Your first mod: three files](02_first_mod/TUTORIAL.md) | `plugin.json`, `hooks.json` `modules`, `register`; validate; `--plugin-dir`; hot reload; type-checking | a spinner suffix |
| 03 | [The hook pipeline](03_hook_pipeline/TUTORIAL.md) | `($, e, next)`: observe, rewrite, wrap, answer; reading the Events tape | a tool-call counter, prompt context |
| 04 | [Guards: deny, hold and catch](04_guards/TUTORIAL.md) | `{ deny }`, `$.ui.ask`, `.catch`, fail-closed guards, and a hook-order bug we measured | a force-push guard |
| 05 | [Draw: panes, bands, status, toast](05_drawing/TUTORIAL.md) | `$.ui.resolve(e)`, surfaces, panes, bands, sharing the band with other plugins, UI tests | a band with a button |
| 06 | [State that survives a reload](06_state_and_store/TUTORIAL.md) | module variables vs `$.state` atoms vs `$.store`; the `PluginState` contract | reps that persist |
| 07 | [Commands, tools and models](07_commands_tools_models/TUTORIAL.md) | slash commands with no Claude turn, tools Claude calls, `$.model`, `$.process`, `$.http` | `/reps`, `log_reps` |
| 08 | [Test, validate, ship](08_test_and_ship/TUTORIAL.md) | `claude plugin test`, tsc, marketplaces, versions; capstone: fix the untested research starter | tests, `marketplace.json` |

## What is where

```text
week29/
  mod-gym/                        the Mod Gym plugin (a mod): panel, band, status, /mod-gym, gym_status tool
    hooks/register.tsx            the hooks module — itself the course's running example
    hooks/lessons.ts              the 8 lessons as data: body, checks, the "Ask Claude" question
    types/index.d.ts              its $.state contract
    tests/mod-gym.test.ts         5 tests: commands, checks, the tool.call wrap, the pane on two surfaces
  reference/hello-mod/            the lab mod as it stands after lesson 08 (all 29 Mod Gym checks pass)
  .claude-plugin/marketplace.json folder marketplace "fitness-week29" (mod-gym + reference hello-mod)
  NN_*/TUTORIAL.md (+ .th.md)     the long form of each lesson
  mod-agentic-coding-fitness/     the research's untested v3 starter — module 08's capstone audits and fixes it
  Claude Code Mods for the Agentic AI Class.md   the research this week is built from
  AUTHORING.md                    for maintainers: how lessons, checks and tutorials fit together
```

## Check it yourself

```bash
# on: laptop, from the repo root
claude plugin validate week29/mod-gym
claude plugin test week29/mod-gym                 # 5 pass
claude plugin validate week29/reference/hello-mod
claude plugin test week29/reference/hello-mod     # 5 pass
```

## Safety

A mod is **not sandboxed**. It runs with your permissions: it can read your files, approve tool calls and spend your
usage. Read a mod's `claude plugin validate` report — the `hooks:`, `calls:` and `env reads:` lines — before you
load it, including this one. Mod Gym reads `HOME`, reads files under your lab folder, and runs only
`claude plugin validate` and `claude plugin test` on your lab mod.
