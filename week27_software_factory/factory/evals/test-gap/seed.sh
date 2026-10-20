#!/usr/bin/env bash
set -euo pipefail
git init -q . && mkdir -p app tests .git/gh-fixture
cat > app/rollup.py << 'PY'
def kwh_rollup(readings):
    """Sum kWh for a day; ACs: (1) returns 0.0 for an empty list, (2) ignores None entries, (3) rounds to 2 dp."""
    return round(sum(r for r in readings if r is not None), 2)
PY
cat > tests/test_rollup.py << 'PY'
from app.rollup import kwh_rollup
def test_rounds(): assert kwh_rollup([1.234, 2.0]) == 3.23
def test_ignores_none(): assert kwh_rollup([1.0, None]) == 1.0
PY
touch app/__init__.py
cat > .git/gh-fixture/issue.json << 'J'
{"number":7,"title":"🐛 [bug] alerts: kwh_rollup crashes on empty day","body":"## Acceptance Criteria\n- [ ] kwh_rollup([]) returns 0.0\n- [ ] None entries are ignored\n- [ ] result rounded to 2 dp","labels":[],"comments":[]}
J
cat > .git/gh-fixture/pr.json << 'J'
{"number":7,"headRefOid":"abc123","headRefName":"issue-7-rollup","additions":4,"deletions":1,"labels":[],"comments":[{"url":"https://x/issues/7#issuecomment-1","body":"[builder] [report] ACs covered: empty list → test_rounds; None → test_ignores_none; rounding → test_rounds."}]}
J
printf 'diff --git a/app/rollup.py b/app/rollup.py\n+    return round(sum(r for r in readings if r is not None), 2)\n' > .git/gh-fixture/diff.patch
git add -A && git -c user.email=e@x -c user.name=eval commit -qm seed
echo 'echo "$@" >> .cmd-log' > /dev/null
