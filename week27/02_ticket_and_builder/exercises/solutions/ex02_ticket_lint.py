import re, sys
BANNED = ("gracefully", "properly", "correctly", "works", "handles")
SECTIONS = ("Problem", "Context", "Fix", "Acceptance Criteria", "Risk", "Blocked by")
def _section(body, name):
    m = re.search(rf"^##\s*{re.escape(name)}\s*$(.*?)(?=^##\s|\Z)", body, re.M | re.S)
    return m.group(1) if m else None
def lint(body: str):
    out = []
    missing = [s for s in SECTIONS if _section(body, s) is None]
    out.append(("R1 sections", not missing, "missing: " + ", ".join(missing) if missing else "all present"))
    ac = _section(body, "Acceptance Criteria") or ""
    boxes = re.findall(r"^\s*- \[ \] (.+)$", ac, re.M)
    out.append(("R2 2–5 ACs", 2 <= len(boxes) <= 5, f"{len(boxes)} checkbox ACs"))
    vague = [w for b in boxes for w in BANNED if re.search(rf"\b{w}\b", b, re.I)]
    out.append(("R3 no vague words", not vague, "found: " + ", ".join(sorted(set(vague))) if vague else "ok"))
    risk = _section(body, "Risk") or ""
    out.append(("R4 risk emoji", any(e in risk for e in "🟢🟡🔴"), risk.strip().splitlines()[0] if risk.strip() else "empty"))
    ctx = _section(body, "Context") or ""
    out.append(("R5 Where line", "**Where:**" in ctx, "ok" if "**Where:**" in ctx else "no **Where:**"))
    return out
if __name__ == "__main__":
    for rule, ok, detail in lint(sys.stdin.read()):
        print(f"{'PASS' if ok else 'FAIL'} {rule}  {detail}")
