# Alto Mini — the Week 27 factory target

A deliberately small hotel-energy app (FastAPI + one static page) with **known, ticketable gaps**, so every lab has a
real bug to drain through the factory. Rooms with setpoints, 14 days of kWh readings, an alerts page, a daily-kWh history.

```bash
cd week27/00_alto_mini
make seed      # alto_mini.db — 4 rooms, 14 days of readings, 3 alerts (incl. one empty day and one None reading)
make dev       # http://127.0.0.1:8127  (API docs at /docs)
make test      # 6 pytest tests pass on the shipped code
```

## The seeded gaps (instructor key — students discover these via telemetry/*.json and the UI)

| Where | Symptom | Telemetry id | Lab |
|---|---|---|---|
| `/api/alerts` → UI | "Yesterday kWh" shows `—` when a room had no readings yesterday (room 1205) | — | 01 (your first ticket) |
| `app/energy/rollup.py` | `kwh_rollup` raises `TypeError` on a `None` reading (room 1410, day −3); empty list should be `0.0` | S-101 / S-106 | 02 |
| `app/hvac/setpoint.py` + UI slider | slider allows 31 °C, API answers **500** instead of 422 or clamping | S-102 (+ rage clicks) | 02 (`control` label → human merge) |
| `app/energy/daily.py` | `cfg["tz"]` → `KeyError` when `?tz=` is omitted; should default to the hotel timezone | S-103 | 02 |
| alerts funnel | view → acknowledge drops; "Ack" has no confirmation or undo | PostHog funnel | 06 |

Test user for lab write-ups: `qa@example.local` (no real auth in this app — add it as a Lab 07 stretch goal).

## Layout
```
app/main.py           FastAPI app, serves static/index.html at /
app/api/rooms.py      GET /api/rooms, GET /api/rooms/{id}, POST /api/rooms/{id}/setpoint, GET /api/rooms/{id}/history?tz=
app/api/alerts.py     GET /api/alerts, POST /api/alerts/{id}/ack
app/energy/           rollup.py (kwh_rollup), daily.py (daily_kwh)
app/hvac/setpoint.py  validate()
app/data.py           SQLite schema + `make seed`
tests/                pytest (unit + TestClient API)
e2e/smoke.spec.ts     optional Playwright smoke; the QA lane adds forensics guards
```
