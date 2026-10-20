import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex02_ticket_lint import lint
GOOD = """## Problem
x
## Context
- **Where:** /api/alerts
## Fix
y
## Acceptance Criteria
- [ ] returns 0.0 for []
- [ ] ignores None entries
## Risk
🟢 Low
## Blocked by
- None.
"""
def test_good(): assert all(ok for _, ok, _ in lint(GOOD))
def test_vague(): assert not dict((r, ok) for r, ok, _ in lint(GOOD.replace("ignores None entries", "handles None gracefully")))["R3 no vague words"]
def test_missing_section(): assert not dict((r, ok) for r, ok, _ in lint(GOOD.replace("## Blocked by\n- None.\n", "")))["R1 sections"]
