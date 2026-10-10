#!/usr/bin/env python3
"""Block a live-looking API key literal written into a file. PreToolUse(Write|Edit)."""
import re, sys, os, json; sys.path.insert(0, os.path.dirname(__file__)); from _common import read, block
d = read(); ti = d.get("tool_input") or {}
text = " ".join(str(ti.get(k, "")) for k in ("content", "new_string"))
pats = [r"sk-ant-[A-Za-z0-9_-]{20,}", r"sk-[A-Za-z0-9]{32,}", r"AKIA[0-9A-Z]{16}", r"ghp_[A-Za-z0-9]{36}", r"xox[baprs]-[A-Za-z0-9-]{10,}", r"AIza[0-9A-Za-z_-]{35}"]
for p in pats:
    if re.search(p, text):
        block("a live-looking API key literal; read it from the environment instead")
sys.exit(0)
