#!/usr/bin/env bash
set -euo pipefail
git init -q . && mkdir -p app tests .git/gh-fixture && touch app/__init__.py
cat > app/format.py << 'PY'
import json, pathlib
class FormatterStrategy:
    def format(self, kwh): raise NotImplementedError
class TwoDecimalStrategy(FormatterStrategy):
    def format(self, kwh): return f"{kwh:.2f} kWh"
REGISTRY = {"two_decimal": TwoDecimalStrategy}
class FormatterFactory:
    def __init__(self, cfg_path="formatter.json"):
        self.cfg = json.loads(pathlib.Path(cfg_path).read_text()) if pathlib.Path(cfg_path).exists() else {"strategy": "two_decimal"}
    def build(self): return REGISTRY[self.cfg["strategy"]]()
def format_kwh(kwh): return FormatterFactory().build().format(kwh)
PY
cat > tests/test_format.py << 'PY'
from app.format import format_kwh
def test_format(): assert format_kwh(3.14159) == "3.14 kWh"
PY
cat > .git/gh-fixture/pr.json << 'J'
{"number":11,"headRefOid":"789abc","headRefName":"issue-11-format","additions":16,"deletions":1,"labels":[],"comments":[{"url":"https://x/issues/11#issuecomment-31","body":"[builder] [report] AC covered by tests/test_format.py::test_format. Local tests: python3 -m pytest -q"}]}
J
cat > .git/gh-fixture/issue.json << 'J'
{"number":11,"title":"✨ [feat] dashboard: show kWh with two decimals","body":"## Acceptance Criteria\n- [ ] kWh values render as '<n>.<dd> kWh'","labels":[],"comments":[]}
J
git diff --no-index /dev/null app/format.py > .git/gh-fixture/diff.patch || true
git add -A && git -c user.email=e@x -c user.name=eval commit -qm seed
