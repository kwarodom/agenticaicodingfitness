---
name: commit-message
description: "Write a clear, conventional git commit message from the staged diff. Use when the user asks for a commit message, or to summarize their staged changes."
---

# Commit Message Writer

A tiny example skill so you can see the format. The frontmatter `description`
is the **trigger** — write it as *what it does + when to use it*. The body is
the know-how Claude follows when the skill activates.

## Steps

1. Run `git diff --staged` to see what changed. If nothing is staged, say so
   and stop.
2. Run `git log -5 --oneline` to match the repo's existing message style.
3. Write a commit message:
   - First line: `<type>: <summary>` (≤ 72 chars). Types: feat, fix, docs,
     refactor, test, chore.
   - Blank line, then 1–3 bullet points on *what* changed and *why* (not a
     line-by-line diff).
4. Output the message in a fenced block so it's easy to copy.

## Try it
Stage a change, then ask: *"write me a commit message."*
This skill should activate automatically.
