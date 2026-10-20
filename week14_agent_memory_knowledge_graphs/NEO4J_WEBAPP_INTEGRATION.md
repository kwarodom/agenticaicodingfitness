# Neo4j Web App Integration Tutorial — Smart Hotel Backend

Learn how to consume the hotel knowledge graph from a real web application.
This tutorial uses the graph built in `6.hotel_kg_builder.py` and shows both
**Node.js (Express)** and **Python (Django/FastAPI)** patterns side-by-side.

> **Prerequisite:** Complete `6.hotel_kg_builder.py` first so the graph exists.
> Neo4j running at `bolt://localhost:7687` · credentials `neo4j / mas_memory_2024`.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Driver Setup](#2-driver-setup)
   - [Node.js / Express](#21-nodejs--express)
   - [Python / Django](#22-python--django)
   - [Python / FastAPI](#23-python--fastapi)
3. [Connection Pooling and Config](#3-connection-pooling-and-config)
4. [Reading Data — GET Endpoints](#4-reading-data--get-endpoints)
5. [Writing Data — POST / PATCH Endpoints](#5-writing-data--post--patch-endpoints)
6. [Transactions](#6-transactions)
7. [Parameterised Queries (Security)](#7-parameterised-queries-security)
8. [Error Handling](#8-error-handling)
9. [Pagination and Filtering](#9-pagination-and-filtering)
10. [GraphRAG — Natural Language to Cypher](#10-graphrag--natural-language-to-cypher)
11. [Caching Strategy](#11-caching-strategy)
12. [Full Working Example — Hotel Dashboard API](#12-full-working-example--hotel-dashboard-api)
13. [Production Checklist](#13-production-checklist)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
│   React / Next.js   ◄──── REST / GraphQL / WebSocket ────► │
└─────────────────────────────┬───────────────────────────────┘
                               │ HTTP / WS
┌─────────────────────────────▼───────────────────────────────┐
│                    Web App Backend                          │
│                                                             │
│   Express (Node.js)       Django / FastAPI (Python)        │
│   ┌──────────────┐        ┌────────────────────────────┐   │
│   │ Route Handler│        │ View / APIView / Router    │   │
│   │ Service Layer│        │ Service / Repository Layer │   │
│   │ neo4j-driver │        │ neo4j-python-driver        │   │
│   └──────┬───────┘        └──────────────┬─────────────┘   │
└──────────┼──────────────────────────────┼─────────────────┘
           │ Bolt protocol (port 7687)     │
┌──────────▼──────────────────────────────▼─────────────────┐
│                        Neo4j                               │
│   Hotel · Floors · Zones · Rooms · HVAC · Agents · ...    │
└────────────────────────────────────────────────────────────┘
```

**Key idea:** Neo4j is accessed over the **Bolt protocol** (similar to how
PostgreSQL uses TCP port 5432). Your backend holds a **driver** (a connection
pool) and executes Cypher queries, mapping results to JSON you return to clients.

**When to use a graph DB over SQL:**
- Queries that traverse relationships (e.g., "which agent controls the HVAC in the zone that is over-temp?")
- Highly connected data with many relationship types
- Paths, recommendations, impact analysis

---

## 2. Driver Setup

### 2.1 Node.js / Express

```bash
npm install neo4j-driver express dotenv
```

**`db/neo4j.js`** — singleton driver module
```js
const neo4j = require('neo4j-driver');

const driver = neo4j.driver(
  process.env.NEO4J_URI   || 'bolt://localhost:7687',
  neo4j.auth.basic(
    process.env.NEO4J_USER || 'neo4j',
    process.env.NEO4J_PASS || 'mas_memory_2024'
  ),
  {
    maxConnectionPoolSize: 50,
    connectionAcquisitionTimeout: 5000,
  }
);

// Verify connectivity on startup
driver.verifyConnectivity()
  .then(() => console.log('Neo4j connected'))
  .catch(err => { console.error('Neo4j connection failed', err); process.exit(1); });

module.exports = driver;
```

**`server.js`**
```js
const express = require('express');
const driver  = require('./db/neo4j');
const hotelRoutes = require('./routes/hotel');

const app = express();
app.use(express.json());
app.use('/api/hotel', hotelRoutes);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Hotel API listening on :${PORT}`));

// Graceful shutdown
process.on('SIGINT', async () => {
  await driver.close();
  process.exit(0);
});
```

---

### 2.2 Python / Django

```bash
pip install neo4j django djangorestframework python-dotenv
```

**`hotel/neo4j_client.py`** — module-level driver (created once on import)
```python
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASS", "mas_memory_2024"),
    ),
    max_connection_pool_size=50,
)

def get_driver():
    return _driver

def run_query(cypher: str, **params) -> list[dict]:
    with _driver.session() as session:
        return [dict(r) for r in session.run(cypher, **params)]
```

**`hotel/apps.py`** — verify on startup
```python
from django.apps import AppConfig

class HotelConfig(AppConfig):
    name = 'hotel'

    def ready(self):
        from .neo4j_client import get_driver
        get_driver().verify_connectivity()
        print("Neo4j connected")
```

**`settings.py`** — register app
```python
INSTALLED_APPS = [
    ...
    'rest_framework',
    'hotel.apps.HotelConfig',   # <-- use AppConfig so ready() fires
]
```

---

### 2.3 Python / FastAPI

FastAPI makes it easy to inject the driver as a dependency:

```bash
pip install neo4j fastapi uvicorn python-dotenv
```

**`app/database.py`**
```python
from contextlib import asynccontextmanager
from neo4j import AsyncGraphDatabase
import os

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"),
              os.getenv("NEO4J_PASS", "mas_memory_2024"))

driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

@asynccontextmanager
async def lifespan(app):
    await driver.verify_connectivity()
    print("Neo4j async driver ready")
    yield
    await driver.close()

async def get_session():
    async with driver.session() as session:
        yield session
```

**`app/main.py`**
```python
from fastapi import FastAPI
from .database import lifespan
from .routers import hotel

app = FastAPI(lifespan=lifespan)
app.include_router(hotel.router, prefix="/api/hotel")
```

---

## 3. Connection Pooling and Config

The Bolt driver manages a **connection pool** internally — you do not open
a new connection per request. The pattern is:

```
driver (singleton, lives for app lifetime)
  └── session (one per request or operation, auto-returned to pool)
        └── transaction (explicit or auto-commit)
```

| Setting | Default | Recommendation |
|---|---|---|
| `maxConnectionPoolSize` | 100 | 50–100 for web apps |
| `connectionAcquisitionTimeout` | 60 s | 5 s (fail fast) |
| `maxTransactionRetryTime` | 30 s | 15 s |

**.env** (shared between both stacks)
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=mas_memory_2024
```

> **Production:** use `neo4j+s://` (encrypted Bolt) or `neo4j+ssc://` (self-signed)
> instead of plain `bolt://`.

---

## 4. Reading Data — GET Endpoints

All examples query the hotel graph from `6.hotel_kg_builder.py`.

### 4.1 Hotel Overview

**Node.js route**
```js
// routes/hotel.js
const express = require('express');
const driver  = require('../db/neo4j');
const router  = express.Router();

// GET /api/hotel/overview
router.get('/overview', async (req, res) => {
  const session = driver.session();
  try {
    const result = await session.run(`
      MATCH (h:Hotel {id: 'hotel-grandvista'})
      MATCH (h)-[:HAS_BUDGET]->(b:EnergyBudget)
      MATCH (h)-[:HAS_FLOOR]->(f:Floor)
      RETURN h.name            AS name,
             h.total_rooms     AS totalRooms,
             h.location        AS location,
             b.consumed_kwh    AS consumedKwh,
             b.total_kwh       AS totalKwh,
             b.on_track        AS onTrack,
             count(f)          AS floorCount
    `);
    // neo4j integers must be converted
    const r = result.records[0];
    res.json({
      name:        r.get('name'),
      totalRooms:  r.get('totalRooms').toNumber(),
      location:    r.get('location'),
      consumedKwh: r.get('consumedKwh'),
      totalKwh:    r.get('totalKwh'),
      onTrack:     r.get('onTrack'),
      floorCount:  r.get('floorCount').toNumber(),
    });
  } finally {
    await session.close();
  }
});
```

**Django view**
```python
# hotel/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from .neo4j_client import run_query

class HotelOverviewView(APIView):
    def get(self, request):
        rows = run_query("""
            MATCH (h:Hotel {id: 'hotel-grandvista'})
            MATCH (h)-[:HAS_BUDGET]->(b:EnergyBudget)
            RETURN h.name            AS name,
                   h.total_rooms     AS totalRooms,
                   h.location        AS location,
                   b.consumed_kwh    AS consumedKwh,
                   b.total_kwh       AS totalKwh,
                   b.on_track        AS onTrack
        """)
        return Response(rows[0] if rows else {})
```

**URLs** (`hotel/urls.py`)
```python
from django.urls import path
from . import views

urlpatterns = [
    path('overview/', views.HotelOverviewView.as_view()),
    path('floors/',   views.FloorListView.as_view()),
    path('rooms/',    views.RoomListView.as_view()),
    path('zones/',    views.ZoneStatusView.as_view()),
    path('energy/',   views.EnergyDashboardView.as_view()),
]
```

---

### 4.2 Room Availability

```python
class RoomListView(APIView):
    def get(self, request):
        floor  = request.query_params.get('floor')
        status = request.query_params.get('status')   # 'occupied' | 'vacant'

        cypher = """
            MATCH (r:Room)
            WHERE ($floor  IS NULL OR r.floor_number   = $floor)
              AND ($status IS NULL OR r.status          = $status)
            RETURN r.room_number    AS roomNumber,
                   r.floor_number   AS floor,
                   r.type           AS type,
                   r.status         AS status,
                   r.occupancy_count AS guests
            ORDER BY r.room_number
        """
        rows = run_query(
            cypher,
            floor=int(floor) if floor else None,
            status=status,
        )
        return Response(rows)
```

**Example calls:**
```
GET /api/hotel/rooms/                     → all 300 rooms
GET /api/hotel/rooms/?status=occupied     → occupied rooms only
GET /api/hotel/rooms/?floor=7             → floor 7 rooms
GET /api/hotel/rooms/?floor=7&status=vacant
```

---

### 4.3 Zone Energy & Temperature Status

```python
class ZoneStatusView(APIView):
    def get(self, request):
        rows = run_query("""
            MATCH (z:Zone)
            OPTIONAL MATCH (z)-[:HAS_HVAC]->(u:HVACUnit)
            OPTIONAL MATCH (s:Sensor {type: 'temperature'})-[:MONITORS]->(z)
            RETURN z.id              AS zoneId,
                   z.type            AS zoneType,
                   z.current_temp_c  AS currentTemp,
                   z.setpoint_celsius AS setpoint,
                   round(z.current_temp_c - z.setpoint_celsius, 2) AS tempDeviation,
                   u.power_kw        AS hvacPowerKw,
                   u.status          AS hvacStatus,
                   CASE
                     WHEN z.current_temp_c > z.setpoint_celsius + 1 THEN 'COOLING_NEEDED'
                     WHEN z.current_temp_c < z.setpoint_celsius - 1 THEN 'HEATING_NEEDED'
                     ELSE 'OK'
                   END AS comfortStatus
            ORDER BY z.type, z.id
        """)
        return Response(rows)
```

---

### 4.4 Graph Traversal — Agent → HVAC chain

This is where Neo4j shines over SQL: multi-hop traversal in a single query.

```python
class AgentControlChainView(APIView):
    """Return the full control chain for a given agent."""
    def get(self, request, agent_id):
        rows = run_query("""
            MATCH (a:Agent {id: $agent_id})-[:CONTROLS]->(u)
            OPTIONAL MATCH (z:Zone)-[:HAS_HVAC]->(u)
            OPTIONAL MATCH (f:Floor)-[:HAS_ZONE]->(z)
            RETURN a.id           AS agent,
                   labels(u)[0]  AS controlsType,
                   u.id           AS unitId,
                   u.status       AS unitStatus,
                   z.id           AS zoneId,
                   z.type         AS zoneType,
                   f.floor_number AS floor
            ORDER BY f.floor_number
        """, agent_id=agent_id)
        return Response(rows)
```

```
GET /api/hotel/agents/HVACControlAgent/chain/
→ [{agent, controlsType, unitId, zoneId, floor}, ...]
```

---

## 5. Writing Data — POST / PATCH Endpoints

### 5.1 Check In a Guest (Node.js)

```js
// POST /api/hotel/guests/checkin
router.post('/guests/checkin', async (req, res) => {
  const { guestId, name, roomNumber } = req.body;

  // Input validation
  if (!guestId || !name || !roomNumber) {
    return res.status(400).json({ error: 'guestId, name, roomNumber required' });
  }

  const session = driver.session();
  try {
    const result = await session.executeWrite(async tx => {
      return tx.run(`
        // MERGE is idempotent — safe to retry
        MERGE (g:Guest {id: $guestId})
        ON CREATE SET g.name     = $name,
                      g.check_in = date()
        ON MATCH  SET g.name     = $name
        WITH g
        MATCH (r:Room {room_number: $roomNumber})
        MERGE (g)-[:STAYS_IN]->(r)
        SET r.status          = 'occupied',
            r.occupancy_count = r.occupancy_count + 1
        RETURN g.id AS guestId, g.name AS name,
               r.room_number AS room, r.status AS status
      `, { guestId, name, roomNumber: neo4j.int(roomNumber) });
    });

    const r = result.records[0];
    res.status(201).json({
      guestId: r.get('guestId'),
      name:    r.get('name'),
      room:    r.get('room').toNumber(),
      status:  r.get('status'),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  } finally {
    await session.close();
  }
});
```

### 5.2 Update HVAC Setpoint (Django)

```python
# hotel/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .neo4j_client import get_driver

@api_view(['PATCH'])
def update_hvac_setpoint(request, zone_id):
    new_setpoint = request.data.get('setpoint_celsius')
    if new_setpoint is None:
        return Response({'error': 'setpoint_celsius required'},
                        status=status.HTTP_400_BAD_REQUEST)

    driver = get_driver()
    with driver.session() as session:
        result = session.execute_write(
            lambda tx: tx.run("""
                MATCH (z:Zone {id: $zone_id})-[:HAS_HVAC]->(u:HVACUnit)
                SET u.setpoint_celsius = $setpoint,
                    u.last_adjusted    = datetime()
                RETURN z.id AS zone, u.id AS unit,
                       u.setpoint_celsius AS newSetpoint
            """, zone_id=zone_id, setpoint=float(new_setpoint))
            .single()
        )

    if result is None:
        return Response({'error': 'zone not found'}, status=status.HTTP_404_NOT_FOUND)

    return Response(dict(result))
```

```
PATCH /api/hotel/zones/zone-FLOOR-07/hvac/
{"setpoint_celsius": 22.5}
```

---

## 6. Transactions

Use **explicit transactions** when you need multiple Cypher statements to
succeed or fail together (atomically).

### Node.js — Explicit Write Transaction

```js
// Transfer guest from one room to another atomically
router.post('/guests/:guestId/transfer', async (req, res) => {
  const { guestId } = req.params;
  const { newRoomNumber } = req.body;

  const session = driver.session();
  try {
    const result = await session.executeWrite(async tx => {
      // Step 1: remove old relationship and vacate old room
      await tx.run(`
        MATCH (g:Guest {id: $guestId})-[r:STAYS_IN]->(oldRoom:Room)
        DELETE r
        SET oldRoom.occupancy_count = oldRoom.occupancy_count - 1,
            oldRoom.status = CASE WHEN oldRoom.occupancy_count - 1 = 0
                                  THEN 'vacant' ELSE 'occupied' END
      `, { guestId });

      // Step 2: link to new room
      return tx.run(`
        MATCH (g:Guest {id: $guestId})
        MATCH (r:Room {room_number: $newRoomNumber})
        MERGE (g)-[:STAYS_IN]->(r)
        SET r.status          = 'occupied',
            r.occupancy_count = r.occupancy_count + 1
        RETURN g.id AS guest, r.room_number AS newRoom
      `, { guestId, newRoomNumber: neo4j.int(newRoomNumber) });
    });

    const r = result.records[0];
    res.json({ guest: r.get('guest'), newRoom: r.get('newRoom').toNumber() });
  } finally {
    await session.close();
  }
});
```

### Python — Explicit Transaction

```python
def transfer_guest(guest_id: str, new_room: int) -> dict:
    driver = get_driver()
    with driver.session() as session:
        def _txn(tx):
            # Both statements inside one ACID transaction
            tx.run("""
                MATCH (g:Guest {id: $gid})-[r:STAYS_IN]->(old:Room)
                DELETE r
                SET old.occupancy_count = old.occupancy_count - 1,
                    old.status = CASE WHEN old.occupancy_count - 1 = 0
                                      THEN 'vacant' ELSE 'occupied' END
            """, gid=guest_id)

            return tx.run("""
                MATCH (g:Guest {id: $gid})
                MATCH (r:Room {room_number: $room})
                MERGE (g)-[:STAYS_IN]->(r)
                SET r.status = 'occupied',
                    r.occupancy_count = r.occupancy_count + 1
                RETURN g.id AS guest, r.room_number AS newRoom
            """, gid=guest_id, room=new_room).single()

        result = session.execute_write(_txn)
        return dict(result) if result else {}
```

> **Rule:** `execute_write` automatically retries on transient errors
> (e.g., deadlocks). Avoid side-effects (API calls, emails) inside the
> transaction function — it may run more than once.

---

## 7. Parameterised Queries (Security)

**Never** interpolate user input into Cypher strings — that is a **Cypher
injection** vulnerability analogous to SQL injection.

```python
# DANGEROUS — never do this
zone_id = request.GET['zone']
run_query(f"MATCH (z:Zone {{id: '{zone_id}'}}) RETURN z")

# SAFE — always use parameters
run_query("MATCH (z:Zone {id: $zone_id}) RETURN z", zone_id=zone_id)
```

```js
// DANGEROUS
const id = req.query.zoneId;
session.run(`MATCH (z:Zone {id: '${id}'}) RETURN z`)

// SAFE
session.run('MATCH (z:Zone {id: $id}) RETURN z', { id: req.query.zoneId })
```

Parameter placeholders (`$name`) are handled by the Bolt protocol —
the query and data travel separately, so injection is impossible.

---

## 8. Error Handling

```python
# hotel/neo4j_client.py
from neo4j.exceptions import (
    ServiceUnavailable,
    AuthError,
    ConstraintError,
    TransactionError,
)

def run_query(cypher: str, **params) -> list[dict]:
    try:
        with _driver.session() as session:
            return [dict(r) for r in session.run(cypher, **params)]
    except ServiceUnavailable as e:
        raise RuntimeError(f"Neo4j unavailable: {e}") from e
    except AuthError as e:
        raise PermissionError(f"Neo4j auth failed: {e}") from e
    except ConstraintError as e:
        raise ValueError(f"Constraint violation: {e}") from e
```

**Django view with consistent error responses:**
```python
from neo4j.exceptions import ServiceUnavailable

class RoomListView(APIView):
    def get(self, request):
        try:
            rows = run_query("MATCH (r:Room) RETURN r.room_number AS num LIMIT 20")
            return Response(rows)
        except ServiceUnavailable:
            return Response({'error': 'database unavailable'},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            return Response({'error': str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
```

**Node.js middleware:**
```js
// Handle neo4j-specific errors uniformly
app.use((err, req, res, next) => {
  if (err.code === 'Neo.ClientError.Schema.ConstraintValidationFailed') {
    return res.status(409).json({ error: 'Duplicate entry', detail: err.message });
  }
  if (err.code === 'Neo.TransientError.Network.CommunicationError') {
    return res.status(503).json({ error: 'Graph database temporarily unavailable' });
  }
  res.status(500).json({ error: err.message });
});
```

---

## 9. Pagination and Filtering

```python
class RoomListView(APIView):
    def get(self, request):
        # Parse query parameters with safe defaults
        page     = max(1, int(request.query_params.get('page',  1)))
        per_page = min(100, int(request.query_params.get('limit', 20)))
        skip     = (page - 1) * per_page
        floor    = request.query_params.get('floor')
        status   = request.query_params.get('status')
        room_type = request.query_params.get('type')

        rows = run_query("""
            MATCH (r:Room)
            WHERE ($floor     IS NULL OR r.floor_number = $floor)
              AND ($status    IS NULL OR r.status       = $status)
              AND ($room_type IS NULL OR r.type         = $room_type)
            RETURN r.room_number    AS roomNumber,
                   r.floor_number   AS floor,
                   r.type           AS type,
                   r.status         AS status,
                   r.occupancy_count AS guests
            ORDER BY r.room_number
            SKIP $skip LIMIT $limit
        """,
            floor=int(floor) if floor else None,
            status=status,
            room_type=room_type,
            skip=skip,
            limit=per_page,
        )

        # Total count (separate query — Neo4j has no COUNT(*) OVER())
        total = run_query("""
            MATCH (r:Room)
            WHERE ($floor  IS NULL OR r.floor_number = $floor)
              AND ($status IS NULL OR r.status       = $status)
            RETURN count(r) AS total
        """, floor=int(floor) if floor else None, status=status)[0]['total']

        return Response({
            'data': rows,
            'pagination': {
                'page': page,
                'perPage': per_page,
                'total': total,
                'pages': -(-total // per_page),  # ceiling division
            }
        })
```

---

## 10. GraphRAG — Natural Language to Cypher

The hotel builder already includes a `graphrag_demo()` function. Here is how
to expose that as an API endpoint.

### FastAPI + LangChain endpoint

```python
# app/routers/graphrag.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import os

router = APIRouter(prefix="/graphrag", tags=["graphrag"])

# Initialize once (expensive)
_graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASS", "mas_memory_2024"),
)
_llm   = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
_chain = GraphCypherQAChain.from_llm(_llm, graph=_graph, verbose=False)

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_graph(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        result = _chain.invoke({"query": body.question})
        return {"question": body.question, "answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Example requests:**
```bash
curl -X POST http://localhost:8000/api/graphrag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which zones need more cooling right now?"}'

curl -X POST http://localhost:8000/api/graphrag/ask \
  -d '{"question": "What is today energy budget status?"}'

curl -X POST http://localhost:8000/api/graphrag/ask \
  -d '{"question": "How many rooms are occupied on floor 7?"}'
```

---

## 11. Caching Strategy

Graph queries can be expensive on large datasets. Apply caching at the
**service layer**, not inside the Cypher runner.

### Node.js — in-memory cache with TTL

```js
// services/cache.js
const cache = new Map();

function cached(key, ttlMs, fetchFn) {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.ts < ttlMs) return Promise.resolve(hit.value);

  return fetchFn().then(value => {
    cache.set(key, { value, ts: Date.now() });
    return value;
  });
}

module.exports = { cached };
```

```js
// In route handler — cache energy dashboard for 30 seconds
router.get('/energy', (req, res) => {
  cached('energy-dashboard', 30_000, () =>
    session.run(`
      MATCH (b:EnergyBudget)
      RETURN b.consumed_kwh, b.remaining_kwh, b.on_track
    `).then(r => r.records.map(rec => rec.toObject()))
  ).then(data => res.json(data));
});
```

### Django — Redis cache via django-redis

```bash
pip install django-redis
```

```python
# settings.py
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
    }
}
```

```python
# hotel/views.py
from django.core.cache import cache

class EnergyDashboardView(APIView):
    def get(self, request):
        cached = cache.get('energy_dashboard')
        if cached:
            return Response(cached)

        rows = run_query("""
            MATCH (h:Hotel {id: 'hotel-grandvista'})-[:HAS_BUDGET]->(b:EnergyBudget)
            RETURN b.date, b.consumed_kwh AS consumed,
                   b.remaining_kwh AS remaining, b.on_track AS onTrack,
                   round(b.consumed_kwh / b.total_kwh * 100, 1) AS pctUsed
        """)
        cache.set('energy_dashboard', rows, timeout=30)  # 30 seconds
        return Response(rows)
```

---

## 12. Full Working Example — Hotel Dashboard API

A self-contained Node.js server with four endpoints that mirror the MAS agents.

```js
// hotel-api/server.js
require('dotenv').config();
const express = require('express');
const neo4j   = require('neo4j-driver');

const driver = neo4j.driver(
  process.env.NEO4J_URI || 'bolt://localhost:7687',
  neo4j.auth.basic('neo4j', 'mas_memory_2024')
);

const app = express();
app.use(express.json());

// Helper — run a read query, convert integer values
async function runRead(cypher, params = {}) {
  const session = driver.session({ defaultAccessMode: neo4j.session.READ });
  try {
    const result = await session.run(cypher, params);
    return result.records.map(r => {
      const obj = r.toObject();
      // Convert Neo4j Integer → JS number
      Object.keys(obj).forEach(k => {
        if (neo4j.isInt(obj[k])) obj[k] = obj[k].toNumber();
      });
      return obj;
    });
  } finally {
    await session.close();
  }
}

// ── GET /api/occupancy ──────────────────────────────────────────────────────
app.get('/api/occupancy', async (req, res) => {
  const rows = await runRead(`
    MATCH (f:Floor)-[:HAS_ZONE]->(z:Zone)-[:CONTAINS_ROOM]->(r:Room)
    WITH f.floor_number AS floor,
         count(r) AS totalRooms,
         sum(CASE r.status WHEN 'occupied' THEN 1 ELSE 0 END) AS occupied,
         sum(r.occupancy_count) AS totalGuests
    RETURN floor, totalRooms, occupied, totalGuests,
           round(occupied * 100.0 / totalRooms, 1) AS occupancyPct
    ORDER BY floor
  `);
  res.json(rows);
});

// ── GET /api/hvac ───────────────────────────────────────────────────────────
app.get('/api/hvac', async (req, res) => {
  const rows = await runRead(`
    MATCH (z:Zone)-[:HAS_HVAC]->(u:HVACUnit)
    RETURN z.id AS zoneId, z.type AS zoneType,
           z.current_temp_c AS currentTemp, z.setpoint_celsius AS setpoint,
           u.id AS hvacId, u.type AS hvacType,
           u.power_kw AS powerKw, u.efficiency_pct AS efficiencyPct,
           CASE
             WHEN z.current_temp_c > z.setpoint_celsius + 1 THEN 'COOLING_NEEDED'
             WHEN z.current_temp_c < z.setpoint_celsius - 1 THEN 'HEATING_NEEDED'
             ELSE 'OK'
           END AS action
    ORDER BY z.id
  `);
  res.json(rows);
});

// ── GET /api/energy ─────────────────────────────────────────────────────────
app.get('/api/energy', async (req, res) => {
  const rows = await runRead(`
    MATCH (h:Hotel {id: 'hotel-grandvista'})-[:HAS_BUDGET]->(b:EnergyBudget)
    RETURN h.name AS hotel, b.date AS date,
           b.total_kwh AS budgetKwh, b.consumed_kwh AS consumedKwh,
           b.remaining_kwh AS remainingKwh,
           b.hvac_kwh AS hvacKwh, b.lighting_kwh AS lightingKwh,
           b.pool_kwh AS poolKwh, b.elevators_kwh AS elevatorsKwh,
           round(b.consumed_kwh / b.total_kwh * 100, 1) AS pctUsed,
           b.on_track AS onTrack
  `);
  res.json(rows[0] || {});
});

// ── GET /api/agents ─────────────────────────────────────────────────────────
app.get('/api/agents', async (req, res) => {
  const rows = await runRead(`
    MATCH (a:Agent)
    OPTIONAL MATCH (a)-[:PERFORMED]->(e:Event)
    RETURN a.id AS agentId, a.role AS role, a.status AS status,
           a.loop_interval_s AS intervalS, count(e) AS eventCount
    ORDER BY a.id
  `);
  res.json(rows);
});

// ── GET /api/agents/:id/events ──────────────────────────────────────────────
app.get('/api/agents/:id/events', async (req, res) => {
  const rows = await runRead(`
    MATCH (a:Agent {id: $agentId})-[:PERFORMED]->(e:Event)
    RETURN e.type AS eventType, e.data AS data, e.ts AS timestamp
    ORDER BY e.ts DESC
    LIMIT 20
  `, { agentId: req.params.id });
  res.json(rows);
});

// ── POST /api/rooms/:roomNumber/checkin ─────────────────────────────────────
app.post('/api/rooms/:roomNumber/checkin', async (req, res) => {
  const { guestId, name } = req.body;
  const roomNumber = parseInt(req.params.roomNumber, 10);

  if (!guestId || !name) {
    return res.status(400).json({ error: 'guestId and name are required' });
  }

  const session = driver.session();
  try {
    const result = await session.executeWrite(tx => tx.run(`
      MERGE (g:Guest {id: $guestId})
      ON CREATE SET g.name = $name, g.check_in = date()
      WITH g
      MATCH (r:Room {room_number: $roomNumber})
      MERGE (g)-[:STAYS_IN]->(r)
      SET r.status = 'occupied',
          r.occupancy_count = r.occupancy_count + 1
      RETURN g.id AS guestId, g.name AS name, r.room_number AS room
    `, { guestId, name, roomNumber: neo4j.int(roomNumber) }));

    const rec = result.records[0];
    res.status(201).json({
      guestId: rec.get('guestId'),
      name:    rec.get('name'),
      room:    rec.get('room').toNumber(),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  } finally {
    await session.close();
  }
});

driver.verifyConnectivity()
  .then(() => {
    app.listen(3000, () => console.log('Hotel Dashboard API → http://localhost:3000'));
  })
  .catch(err => {
    console.error('Cannot connect to Neo4j:', err.message);
    process.exit(1);
  });
```

**Run it:**
```bash
node hotel-api/server.js

# Test endpoints
curl http://localhost:3000/api/occupancy
curl http://localhost:3000/api/energy
curl http://localhost:3000/api/hvac
curl http://localhost:3000/api/agents
curl http://localhost:3000/api/agents/HVACControlAgent/events

curl -X POST http://localhost:3000/api/rooms/701/checkin \
  -H "Content-Type: application/json" \
  -d '{"guestId":"guest-demo","name":"Somchai Srisuk"}'
```

---

## 13. Production Checklist

### Security
- [ ] Use `neo4j+s://` (TLS) for all non-localhost connections
- [ ] Store credentials in environment variables / secrets manager — never hardcode
- [ ] Always use parameterised queries (prevent Cypher injection)
- [ ] Create a **read-only** Neo4j user for GET-only services
- [ ] Restrict Bolt port 7687 to backend VPC — not publicly accessible

### Performance
- [ ] Create indexes for frequently queried properties:
  ```cypher
  CREATE INDEX room_number IF NOT EXISTS FOR (r:Room) ON (r.room_number);
  CREATE INDEX zone_id     IF NOT EXISTS FOR (z:Zone) ON (z.id);
  CREATE INDEX agent_id    IF NOT EXISTS FOR (a:Agent) ON (a.id);
  ```
- [ ] Tune `maxConnectionPoolSize` to match your concurrency (start at 50)
- [ ] Cache read-heavy, slowly-changing data (energy budget, weather) in Redis
- [ ] Use `LIMIT` on all paginated queries — never return unbounded results
- [ ] Prefer `MERGE` over `CREATE` for idempotent writes

### Reliability
- [ ] Implement health-check endpoint that calls `driver.verifyConnectivity()`
- [ ] Use `execute_write` / `executeWrite` so the driver retries transient errors
- [ ] Set `connectionAcquisitionTimeout` to fail fast under saturation
- [ ] Log slow queries using Neo4j's `slow query log` (configurable in `neo4j.conf`)

### Schema
- [ ] Add `CONSTRAINT` for unique IDs:
  ```cypher
  CREATE CONSTRAINT guest_id IF NOT EXISTS
    FOR (g:Guest) REQUIRE g.id IS UNIQUE;
  ```
- [ ] Document relationship types and their directionality
- [ ] Store `created` / `updated` timestamps on mutable nodes

---

## Quick Reference — Driver Cheat Sheet

| Task | Node.js | Python |
|---|---|---|
| Read query | `session.run(cypher, params)` | `session.run(cypher, **params)` |
| Write (auto-retry) | `session.executeWrite(tx => tx.run(...))` | `session.execute_write(lambda tx: tx.run(...))` |
| Read (auto-retry) | `session.executeRead(tx => tx.run(...))` | `session.execute_read(lambda tx: tx.run(...))` |
| Single result | `result.records[0]` | `result.single()` |
| All results | `result.records.map(r => r.toObject())` | `[dict(r) for r in result]` |
| Integer conversion | `r.get('field').toNumber()` | auto (Python int) |
| Close driver | `await driver.close()` | `driver.close()` |

---

*Tutorial uses the graph built by `6.hotel_kg_builder.py`.
Run that script first, then start your web server — the graph is ready to query.*
