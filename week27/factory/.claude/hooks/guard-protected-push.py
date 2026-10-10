#!/usr/bin/env python3
"""Block direct or force pushes to main/master. PreToolUse(Bash).
Catches: `git push origin main`, `git push origin HEAD:main`, `git push origin feat:refs/heads/main`,
`git push --force …`, `git push origin +main`. Allows: `git push -u origin issue-7-x`, `git push origin main:issue-7-x`."""
import re, sys, os; sys.path.insert(0, os.path.dirname(__file__)); from _common import read, cmd, block
c = cmd(read())
for push in re.findall(r"\bgit\s+push\b([^|;&]*)", c):
    if re.search(r"(--force(-with-lease)?\b|(^|\s)-f\b|\s\+\S+)", push):
        block("force push is not allowed")
    args = [a for a in push.split() if not a.startswith("-")]
    remote, refspecs = (args[0], args[1:]) if args else (None, [])
    for spec in refspecs:
        dst = spec.split(":", 1)[1] if ":" in spec else spec
        dst = dst.replace("refs/heads/", "")
        if dst in ("main", "master"):
            block("direct push to main/master; push the issue branch and open a PR")
    if remote and not refspecs:
        block("bare `git push` is ambiguous here; push the issue branch explicitly (git push -u origin <branch>)")
sys.exit(0)
