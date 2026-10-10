import datetime as dt
from fastapi import APIRouter, HTTPException
from ..data import connect
from ..energy.rollup import kwh_rollup
router = APIRouter(prefix="/api/alerts", tags=["alerts"])
@router.get("")
def list_alerts():
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    out = []
    with connect() as c:
        for a in c.execute("SELECT * FROM alerts ORDER BY ts DESC"):
            a = dict(a); a["yesterday_kwh"] = None
            if a["room_id"]:
                vals = [r["kwh"] for r in c.execute("SELECT kwh FROM readings WHERE room_id=? AND ts LIKE ?", (a["room_id"], yesterday + "%"))]
                a["yesterday_kwh"] = kwh_rollup(vals) if vals else None   # None renders as '—' in the UI (Lab 01 ticket)
            out.append(a)
    return out
@router.post("/{alert_id}/ack")
def ack(alert_id: int):
    with connect() as c:
        if c.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,)).rowcount == 0: raise HTTPException(404, "alert not found")
    return {"id": alert_id, "acknowledged": True}
