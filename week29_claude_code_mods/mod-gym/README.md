# 🏋 Mod Gym

A Claude Code mod that teaches Claude Code mods. `/mod-gym` opens a panel with eight lessons (Week 29 of Agentic
Coding Fitness), runs checks on the mod you are building, and shows its own hooks observing your session live.

Needs Claude Code 2.1.287+ (built and tested on 2.1.292).

## Install

From a clone of this repo:

```bash
claude plugin marketplace add ./week29_claude_code_mods
claude plugin install mod-gym@fitness-week29
```

From GitHub, at the Claude Code prompt:

```text
/plugin marketplace add kwarodom/agenticaicodingfitness
/plugin install mod-gym@agentic-coding-fitness
```

A folder marketplace is read from the folder itself: edit the files, then `/reload-plugins`.

## Use

`/mod-gym` · keys **1/2/3** tabs, **p/n** lessons, **c** check, **a** ask Claude, **q** close.
`/mod-gym [next|prev|1-8|check|status|lab <path>|band on|off|reset]`.

## What it does to your session

From `claude plugin validate`: it hooks `session.start`, `prompt.submit`, `tool.call` (observes every call and its
timing for the Events tab; never changes or blocks one), `turn.complete`, its own command and tool, and draws a band
above the prompt and its pane. It reads `HOME`, reads files under your lab folder, keeps progress in `$.store`, and
runs `claude plugin validate` / `claude plugin test` on your lab mod when you press Check.

## Develop

```bash
claude plugin validate .
claude plugin test .
```
