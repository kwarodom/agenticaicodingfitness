#!/usr/bin/env bash
# Merge PR only if: head == reviewed SHA, no human-only label, size ≤ cap, verify commands pass on base+branch.
# Usage: scripts/merge-gate.sh <PR> <reviewed-sha>
set -euo pipefail
PR=${1:?PR}; EXPECT=${2:?reviewed sha}; CAP=${AUTO_MAX_LINES:-400}; VERIFY=${VERIFY_CMD:-make test}
head=$(gh pr view "$PR" --json headRefOid --jq .headRefOid)
[ "$head" = "$EXPECT" ] || { echo "head moved since review ($head != $EXPECT)"; exit 2; }
labels=$(gh pr view "$PR" --json labels --jq '[.labels[].name] | join(",")')
for l in money auth schema control tenant pdpa; do
  case ",$labels," in *",$l,"*) gh pr comment "$PR" -b "🙋 needs human merge: label \`$l\`"; echo "label $l → human"; exit 3;; esac
done
changed=$(gh pr view "$PR" --json additions,deletions --jq '.additions+.deletions')
[ "$changed" -le "$CAP" ] || { gh pr comment "$PR" -b "🙋 big PR ($changed lines > $CAP) — please review"; exit 3; }
branch=$(gh pr view "$PR" --json headRefName --jq .headRefName)
tmp=$(mktemp -d); git worktree add "$tmp" origin/main >/dev/null
( cd "$tmp" && git merge --no-commit --no-ff "origin/$branch" >/dev/null && eval "$VERIFY" ) || { git worktree remove --force "$tmp"; echo "verify failed on base+branch"; exit 4; }
git worktree remove --force "$tmp"
gh pr merge "$PR" --squash --match-head-commit "$EXPECT"
echo "merged #$PR at $EXPECT"
