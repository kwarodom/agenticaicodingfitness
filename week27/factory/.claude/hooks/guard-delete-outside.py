#!/usr/bin/env python3
"""Block rm/find -delete/git clean aimed at ~, / or paths outside the project. PreToolUse(Bash)."""
import re, sys, os; sys.path.insert(0, os.path.dirname(__file__)); from _common import read, cmd, block
c = cmd(read())
if re.search(r"(^|[\s;|&])(rm\s+-[a-zA-Z]*r|find\s.*-delete|git\s+clean\s+-[a-zA-Z]*f)", c):
    if re.search(r"(\s|=)(~|\$HOME|/)(/)?(\*)?(\s|$)", c) or re.search(r"\s\.\.(/|\s|$)", c):
        block("recursive delete outside the project (~, / or ..). Delete inside the worktree only.")
sys.exit(0)
