import os
os.environ["REEF_MODE"] = "auto"
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)

def test_health_mock_watermark():
    h = c.get("/api/health").json()
    assert h["mode"] in ("mock", "real") and "landmarks" in h

def test_assign_and_history():
    sbs = c.get("/api/sandboxes").json(); pid = c.get("/api/profiles").json()[0]["id"]
    sid = sbs[0]["id"]
    r = c.post(f"/api/sandboxes/{sid}/members", json={"profileId": pid}); assert r.status_code == 200
    assert pid in r.json()["members"]
    hist = c.get("/api/events?kind=history").json(); assert any(e["profileId"] == pid for e in hist)
    r = c.delete(f"/api/sandboxes/{sid}/members/{pid}"); assert pid not in r.json()["members"]

def test_ask_is_honest_in_mock():
    r = c.post("/api/ask", json={"text": "hello", "profileIds": []}).json()
    assert r["queued"] is False
    assert any("MOCK" in e["text"] or "not implemented" in e["text"] for e in c.get("/api/events?kind=system").json())
