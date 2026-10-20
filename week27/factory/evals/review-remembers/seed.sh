#!/usr/bin/env bash
set -euo pipefail
git init -q . && mkdir -p app .git/gh-fixture
cat > app/setpoint.py << 'PY'
LIMITS = (16, 30)
def clamp(v):
    lo, hi = LIMITS
    return max(lo, min(hi, v))
def apply(room, v):
    room["setpoint"] = clamp(v)   # R1 fixed: now clamps
    return room                    # R2 not fixed: still no audit log entry
PY
cat > .git/gh-fixture/pr.json << 'J'
{"number":9,"headRefOid":"def456","headRefName":"issue-9-setpoint","additions":6,"deletions":2,"labels":["control"],"comments":[
 {"url":"https://x/issues/9#issuecomment-21","body":"<!-- fitness-review:report -->\n| id | finding | file:line | status |\n|---|---|---|---|\n| R1 | setpoint not clamped to 16–30 | app/setpoint.py:5 | ❌ |\n| R2 | no audit log entry on setpoint change | app/setpoint.py:7 | ❌ |"},
 {"url":"https://x/issues/9#issuecomment-22","body":"[builder] [report] Fixed R1 and R2, threads resolved."}]}
J
cat > .git/gh-fixture/issue.json << 'J'
{"number":9,"title":"✨ [feat] rooms: clamp setpoint and log changes","body":"## Acceptance Criteria\n- [ ] setpoint clamped to 16–30\n- [ ] every change appends an audit entry","labels":["control"],"comments":[]}
J
printf 'diff --git a/app/setpoint.py b/app/setpoint.py\n+    room["setpoint"] = clamp(v)\n' > .git/gh-fixture/diff.patch
git add -A && git -c user.email=e@x -c user.name=eval commit -qm seed
