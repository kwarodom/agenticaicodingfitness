#!/usr/bin/env python3
"""Block `git worktree add` outside .claude/worktrees/. PreToolUse(Bash)."""
import re, sys, os; sys.path.insert(0, os.path.dirname(__file__)); from _common import read, cmd, block
c = cmd(read())
m = re.search(r"git\s+worktree\s+add\s+(.+)", c)
if m:
    args = [a for a in m.group(1).split() if not a.startswith("-")]
    path = args[0] if args else ""
    if not path.startswith(".claude/worktrees/"):
        block(f"worktree path '{path}' must be under .claude/worktrees/")
sys.exit(0)
