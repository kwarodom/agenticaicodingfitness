def test_health(client): assert client.get("/api/health").json() == {"ok": True}
def test_rooms(client):
    rs = client.get("/api/rooms").json(); assert len(rs) == 4 and rs[0]["id"] == 1201
def test_setpoint_ok(client):
    r = client.post("/api/rooms/1201/setpoint", json={"setpoint": 23.5}); assert r.status_code == 200 and r.json()["setpoint"] == 23.5
def test_alerts_list(client):
    al = client.get("/api/alerts").json(); assert len(al) == 3 and {"kind", "message", "yesterday_kwh"} <= set(al[0])
def test_history_with_tz(client):
    h = client.get("/api/rooms/1201/history", params={"tz": "Asia/Bangkok"}).json(); assert len(h) >= 4
