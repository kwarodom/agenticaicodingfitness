#!/usr/bin/env bash
# Drain the Ready column with headless Claude Code workers, one per card, max $MAX_WORKERS in parallel.
# Usage: PROJECT_NUM=3 OWNER=me scripts/factory-run.sh [build|qa|review]
set -euo pipefail
PROJECT_NUM=${PROJECT_NUM:?set PROJECT_NUM}; OWNER=${OWNER:?set OWNER}
LANE=${1:-build}; MAX=${MAX_WORKERS:-3}; MODEL=${MODEL:-claude-sonnet-5}
case "$LANE" in build) COL=Ready;; qa) COL=QA;; review) COL=Review;; *) echo "lane must be build|qa|review"; exit 64;; esac
mkdir -p .claude/factory/logs
cards=$(gh project item-list "$PROJECT_NUM" --owner "$OWNER" --format json --limit 200 \
  | jq -r --arg col "$COL" '.items[] | select(.status==$col) | .content.number' | head -n "$MAX")
[ -z "$cards" ] && { echo "no cards in $COL"; exit 0; }
for N in $cards; do
  echo "→ $LANE #$N"
  ( claude -p "Use the fitness-$LANE skill on issue #$N. Do not ask questions. Stop when the handoff comment is posted." \
      --model "$MODEL" --max-turns 80 --output-format json \
      > ".claude/factory/logs/issue-$N-$LANE.json" 2> ".claude/factory/logs/issue-$N-$LANE.err" \
    && echo "✓ #$N" || echo "✗ #$N (see logs)" ) &
done
wait
