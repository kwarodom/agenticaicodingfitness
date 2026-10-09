---
name: fitness-whats-new
description: Weekly "what changed in Claude Code and coding agents" digest for the class. Use when the student asks what is new or wants the week's five-minute update.
disable-model-invocation: true
allowed-tools: WebFetch
argument-hint: "[weeks back, default 2]"
---

Produce a five-minute class digest.

1. Fetch https://code.claude.com/docs/en/whats-new and read the last $0 weeks (default 2).
2. For each new feature, write one line: what it is, which curriculum week it touches, and one 10-minute drill.
3. Fetch https://code.claude.com/docs/en/plugins/mods/overview and note any mods API change that affects the `agentic-coding-fitness` mod (events, methods, elements).
4. Finish with a "try tonight" item for Tuesday's class.

Format: Markdown, max 25 lines, every item with its source URL.
