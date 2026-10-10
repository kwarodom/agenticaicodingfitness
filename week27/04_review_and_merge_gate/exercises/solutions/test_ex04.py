import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex04_min_confidence import aggregate
def test_min_blocks():
    r = aggregate([{"agent": "code-grounder", "confidence": 55, "severity": "Should fix", "kind": "verification-miss", "text": "claimed test not found"},
                   {"agent": "historian", "confidence": 95, "severity": "Nit", "kind": "style", "text": "ok"}])
    assert r["confidence"] == 55 and r["approved"] is False
def test_overengineering_not_blocker():
    r = aggregate([{"agent": "code-grounder", "confidence": 90, "severity": "Blocker", "kind": "over-engineering", "text": "strategy registry for a formatter"}])
    assert r["approved"] is True and r["findings"][0]["severity"] == "Should fix"
