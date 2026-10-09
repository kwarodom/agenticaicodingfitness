#!/usr/bin/env bash
# Classic settings hook (runs as a shell command, JSON on stdin).
# Appends one line per session to a local CSV so students can chart attendance later.
mkdir -p "$HOME/.agentic-fitness"
reason=$(jq -r '.reason // "unknown"' 2>/dev/null)
printf '%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${reason:-unknown}" >> "$HOME/.agentic-fitness/sessions.csv"
exit 0
