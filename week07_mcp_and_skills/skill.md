---
name: pr-description
description: Writes pull request descriptions. Use when creating a PR, writing a PR, or when the user asks to summarize changes for a pull request.
allowed-tools: Bash(git diff*), Bash(git log*), Bash(git status*), Read
model: sonnet
---

When writing a PR description:

1. Run git diff main...HEAD (or the appropriate base branch) to see all changes
2. Analyze the diff to understand what changed and why
3. Write a description using the **What / Why / Changes** format

## Default Format

## What
One sentence explaining what this PR does.

## Why
Brief context on why this change is needed.

## Changes
- Bullet points of specific changes made
- Group related changes together
- Mention any files deleted or renamed

## Progressive References

Load these files only when the situation calls for them:

- **For the full template with Testing and Notes sections:** Read references/format-template.md