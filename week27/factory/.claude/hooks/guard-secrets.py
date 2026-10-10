#!/usr/bin/env python3
"""Block reading or piping credential files. PreToolUse(Bash). Exit 2 = block."""
import re, sys, os; sys.path.insert(0, os.path.dirname(__file__)); from _common import read, cmd, block
d = read(); c = cmd(d)
files = [r"(^|[\s/'\"])\.env(?!\.(example|sample|template)\b)(\.[\w.-]+)?\b", r"id_rsa|id_ed25519|id_ecdsa", r"\.aws/credentials", r"\.netrc", r"\.npmrc", r"\.pypirc", r"\.docker/config\.json"]
reads = r"(^|[\s;|&])(cat|less|more|head|tail|sed|awk|grep|rg|xxd|base64|curl|scp|cp|python3?|node|source|\.)\b"
if re.search(reads, c) and any(re.search(p, c) for p in files):
    block("reading or piping a credential file. List key NAMES only, e.g. `grep -o '^[A-Z_]*' .env.example`.")
sys.exit(0)
