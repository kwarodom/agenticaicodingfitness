"""ex02 · Ticket linter. Usage: python ex02_ticket_lint.py < ticket.md
Implement lint(body) -> list[(rule, ok, detail)]. Rules (from docs/agents/issue-tracker.md):
  R1 sections Problem, Context, Fix, Acceptance Criteria, Risk, Blocked by all present
  R2 2–5 acceptance criteria, each a `- [ ]` checkbox
  R3 no banned vague words in ACs: gracefully, properly, correctly, works, handles
  R4 Risk line contains one of 🟢 🟡 🔴
  R5 Context has a **Where:** line
Run the solution's tests with: python -m pytest -q exercises/solutions/test_ex02.py
"""
import re, sys
BANNED = ("gracefully", "properly", "correctly", "works", "handles")
def lint(body: str):
    out = []
    # TODO: implement R1–R5; append (rule, ok, detail) tuples
    return out
if __name__ == "__main__":
    for rule, ok, detail in lint(sys.stdin.read()):
        print(f"{'PASS' if ok else 'FAIL'} {rule}  {detail}")
