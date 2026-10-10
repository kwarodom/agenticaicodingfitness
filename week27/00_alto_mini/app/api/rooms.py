import datetime as dt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..data import connect
from ..hvac.setpoint import validate
from ..energy.daily import daily_kwh
router = APIRouter(prefix="/api/rooms", tags=["rooms"])
class SetpointIn(BaseModel):
    setpoint: float
    actor: str = "qa@example.local"
@router.get("")
def list_rooms():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM rooms ORDER BY id")]
@router.get("/{room_id}")
def get_room(room_id: int):
    with connect() as c:
        r = c.execute("SELECT * FROM rooms WHERE id=?", (room_id,)).fetchone()
    if not r: raise HTTPException(404, "room not found")
    return dict(r)
@router.post("/{room_id}/setpoint")
def set_setpoint(room_id: int, body: SetpointIn):
    v = validate(body.setpoint)   # ValueError → 500 today; see Lab 02
    with connect() as c:
        if not c.execute("SELECT 1 FROM rooms WHERE id=?", (room_id,)).fetchone(): raise HTTPException(404, "room not found")
        c.execute("UPDATE rooms SET setpoint=? WHERE id=?", (v, room_id))
        c.execute("INSERT INTO audit(ts, actor, action, detail) VALUES (?,?,?,?)", (dt.datetime.now(dt.timezone.utc).isoformat(), body.actor, "setpoint", f"room {room_id} → {v}"))
    return {"room_id": room_id, "setpoint": v}
@router.get("/{room_id}/history")
def history(room_id: int, tz: str | None = None):
    with connect() as c:
        rows = [(r["ts"], r["kwh"]) for r in c.execute("SELECT ts, kwh FROM readings WHERE room_id=? ORDER BY ts", (room_id,))]
    cfg = {"tz": tz} if tz else {}
    return daily_kwh(rows, cfg)
