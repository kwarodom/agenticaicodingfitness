"""Alto Mini — the Week 27 factory target: a tiny hotel-energy app with known, ticketable gaps."""
import pathlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api import rooms, alerts
STATIC = pathlib.Path(__file__).resolve().parents[1] / "static"
app = FastAPI(title="Alto Mini", version="0.1.0")
app.include_router(rooms.router); app.include_router(alerts.router)
@app.get("/api/health")
def health(): return {"ok": True}
@app.get("/")
def index(): return FileResponse(STATIC / "index.html")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
