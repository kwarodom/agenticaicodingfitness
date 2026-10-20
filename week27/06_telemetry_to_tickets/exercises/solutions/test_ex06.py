import json, pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex06_verifier import verify
SINCE = "2026-09-26"
BOARD = [{"number": 31, "title": "energy: kwh_rollup crashes on None", "fingerprint": "fp-rollup-none", "route": "/api/alerts"}]
PRS = [{"number": 41, "fingerprint": "fp-room-name-none", "merged_at": "2026-09-30"}]
def c(**k):
    base = {"id": "S", "title": "t", "fingerprint": "fp-x", "route": "/r", "count": 10, "first_seen": "2026-10-01", "last_seen": "2026-10-09"}
    base.update(k); return base
def test_file(): assert verify(c(), BOARD, PRS, SINCE)["verdict"] == "file"
def test_fixed(): assert verify(c(fingerprint="fp-room-name-none", last_seen="2026-09-29"), BOARD, PRS, SINCE)["verdict"] == "drop-fixed"
def test_fixed_but_recurring(): assert verify(c(fingerprint="fp-room-name-none", last_seen="2026-10-05"), BOARD, PRS, SINCE)["verdict"] == "file"
def test_noise(): assert verify(c(count=3, deploy_window=True), BOARD, PRS, SINCE)["verdict"] == "drop-noise"
def test_stale(): assert verify(c(last_seen="2026-09-01"), BOARD, PRS, SINCE)["verdict"] == "drop-stale"
def test_duplicate(): assert verify(c(fingerprint="fp-rollup-none"), BOARD, PRS, SINCE) == {"verdict": "duplicate", "why": "same fingerprint/route as open #31", "ref": "#31"}
def test_triage(): assert verify(c(count=0), BOARD, PRS, SINCE)["verdict"] == "needs-triage"
