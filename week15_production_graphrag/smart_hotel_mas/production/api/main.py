#!/usr/bin/env python3
"""FastAPI Backend. (smart_hotel_mas.pdf §"FastAPI Backend")

The production deployment wraps the MAS memory tiers behind a small REST API so
external systems (BMS gateways, dashboards, mobile apps) can push sensor batches
and pull alerts / floor summaries / optimization setpoints.

Shared, process-level memory instances are created once at import:
    L1  WorkingMemory()                  in-process scratchpad
    L2  EpisodicMemory("hotel_prod.db")  durable SQLite event log
    L4  HotelKGMemory("ProductionMAS")   Neo4j knowledge graph
    (L3 SemanticMemory would point at a managed Chroma Cloud endpoint here.)

Run:  uvicorn main:app --reload        # from production/api/
Production cost estimate (per the PDF): ~$200–250/month
    Neo4j Aura + Chroma Cloud + a small API host + Anthropic API usage.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # production/api/ -> workshop root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "checkpoints"))

# ── Guarded optional dependency: FastAPI ─────────────────────────────────────
try:
    from fastapi import FastAPI, BackgroundTasks, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("The production API needs FastAPI. Install it with:")
    print("    pip install fastapi uvicorn")
    sys.exit(1)

from config import check_neo4j  # noqa: F401
from checkpoint2_memory import WorkingMemory, EpisodicMemory

# L4 wrapper lives in checkpoint6; guard so the module still imports without it.
try:
    from checkpoint6_full_mas import HotelKGMemory
except ImportError:
    HotelKGMemory = None  # type: ignore

# ── App + CORS ───────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Hotel MAS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared, process-level memory instances ───────────────────────────────────
wm = WorkingMemory()
em = EpisodicMemory("hotel_prod.db")
# In production L3 would be a managed Chroma Cloud collection, e.g.
#   sm = SemanticMemory(host=os.getenv("CHROMA_HOST"), port=443, ...)
kg = None
if HotelKGMemory is not None:
    try:
        kg = HotelKGMemory("ProductionMAS")
    except Exception as exc:  # noqa: BLE001
        print(f"(L4 HotelKGMemory unavailable at startup: {type(exc).__name__})")


# ── Schemas ──────────────────────────────────────────────────────────────────
class SensorBatch(BaseModel):
    room_ids: list[str]
    readings: dict


# ── Background helper ─────────────────────────────────────────────────────────
def persist_to_memory_layers(room_ids: list[str], readings: dict, em, kg):
    """Persist an ingested batch to the durable tiers (L2, and L4 if present)."""
    em.store(
        "APIIngest", "sensor_batch_read",
        {"rooms": room_ids, "count": len(room_ids), "readings": readings},
        tags=["api", "batch"],
    )
    if kg is not None:
        try:
            kg.record_event("sensor_batch", {"rooms": room_ids, "n": len(room_ids)})
        except Exception as exc:  # noqa: BLE001
            print(f"(L4 persist skipped: {type(exc).__name__})")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/api/v1/sensor_batch")
def ingest_sensor_batch(batch: SensorBatch, background: BackgroundTasks):
    """Ingest a batch of sensor readings: write to L1 now, persist in background."""
    ts = datetime.now().isoformat()
    for room_id in batch.room_ids:
        wm.set(f"reading:{room_id}", batch.readings.get(room_id, batch.readings))
    background.add_task(
        persist_to_memory_layers, batch.room_ids, batch.readings, em, kg
    )
    return {"status": "accepted", "rooms": batch.room_ids, "ts": ts}


@app.get("/api/v1/alerts")
def get_alerts(hours: int = Query(4, ge=1, le=168)):
    """Return active alerts from L4 over the trailing ``hours`` window."""
    if kg is None:
        return {"alerts": [], "note": "L4 (HotelKGMemory) not connected"}
    try:
        return {"alerts": kg.query_active_alerts(hours=hours)}
    except TypeError:
        # signature without an hours kwarg
        return {"alerts": kg.query_active_alerts()}
    except Exception as exc:  # noqa: BLE001
        return {"alerts": [], "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/v1/floor/{floor_num}/summary")
def get_floor_summary(floor_num: int):
    """Return an aggregate summary for a floor from L4."""
    if kg is None:
        return {"floor": floor_num, "summary": None,
                "note": "L4 (HotelKGMemory) not connected"}
    try:
        return {"floor": floor_num, "summary": kg.query_floor_summary(floor_num)}
    except Exception as exc:  # noqa: BLE001
        return {"floor": floor_num, "summary": None,
                "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/v1/optimize")
def optimize(floor_num: int = Query(1, ge=1, le=5)):
    """Compute HVAC setpoints. Imports the heavy CP4 optimizers lazily/guarded."""
    try:
        # checkpoint4 pulls in PuLP/Prophet — only import on demand.
        from checkpoint4_optimizer import HVACOptimizer, OccupancyForecaster
    except ImportError:
        return {"setpoints": None,
                "error": "Optimization deps unavailable. "
                         "pip install pulp prophet pandas numpy"}
    try:
        forecaster = OccupancyForecaster()
        optimizer = HVACOptimizer()
        forecast = forecaster.forecast(floor_num)
        setpoints = optimizer.optimize(floor_num, forecast)
        return {"floor": floor_num, "setpoints": setpoints}
    except Exception as exc:  # noqa: BLE001
        return {"setpoints": None, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    # Convenience local launch; production uses the uvicorn command above.
    try:
        import uvicorn
    except ImportError:
        print("Install uvicorn to run locally:  pip install uvicorn")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000)
