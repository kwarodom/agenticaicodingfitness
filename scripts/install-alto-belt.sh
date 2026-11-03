#!/usr/bin/env bash
# Install Alto Belt for your user, so /belt works in every repository you open.
# Usage: bash scripts/install-alto-belt.sh
set -euo pipefail

command -v claude >/dev/null || { echo "Claude Code is not installed: https://code.claude.com" >&2; exit 1; }

version=$(claude --version | awk '{print $1}')
min=2.1.287
if [ "$(printf '%s\n%s\n' "$min" "$version" | sort -V | head -1)" != "$min" ]; then
  echo "Claude Code $version is too old for Alto Belt (needs $min+). Run: claude update" >&2
  exit 1
fi

claude plugin marketplace add kwarodom/agenticaicodingfitness || claude plugin marketplace update agentic-coding-fitness
claude plugin install alto-belt@agentic-coding-fitness

echo
echo "Done. In Claude Code, run /reload-plugins (or start a new session), then type /belt."
