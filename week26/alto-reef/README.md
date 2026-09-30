# Alto Reef — web lab runner for the NemoClaw / DGX Spark tutorial

Milestone status: **Reef map (M6 visual + M0 skeleton) implemented.** Sandbox Manager, Build-a-Claw, Workbench, Task Monitor and Lab Mode are routed placeholders.

```
alto-reef/
  docs/                 spec.md, tutorial.md, reference-screenshots/ (10 JPGs + README)
  runner/               FastAPI runner (:4455) — adapters wrap fixed openshell/nemoclaw argv, never a shell
    app/security.py     name validation + run_argv (no shell, missing binary => exit 127)
    app/adapters/       openshell.py, nemoclaw.py, http_probes.py   (// VERIFY comments mark unconfirmed flags)
    app/world.py        in-memory world; REEF_MODE=auto|mock|real
    app/main.py         /api/health /api/sandboxes /api/profiles /api/events /api/stream (SSE) /api/ask
    tests/              pytest
  frontend/             Vite + React 18 + TS + Tailwind (:4454), hash routing, no localStorage
    src/components/reef ReefCanvas (SVG island), Enclosure, Avatar, Landmarks, geometry
    src/components/Hud  Toolbar, HealthTiles, StatusBanner, ProfileCard, CommsStream, AskBar
  fixtures/             recorded gateway outputs (M1)
```

## Run on the DGX Spark

```bash
cd alto-reef/runner && pip install -r requirements.txt
REEF_MODE=auto uvicorn app.main:app --host 127.0.0.1 --port 4455   # picks real when `openshell` is on PATH
cd ../frontend && npm install && npm run dev -- --port 4454          # Vite proxies /api -> :4455
open http://localhost:4454
```
`make dev` runs both. `make test` runs the runner tests. Set `REEF_TOKEN=...` to require a bearer token
(`VITE_REEF_TOKEN` on the frontend build).

## Honesty rules implemented
- `mock` mode only when no `openshell` binary is found; refused otherwise. UI shows an amber banner and a "MOCK DATA" watermark.
- Real mode: gateway/inference/sandbox state from `openshell status|inference get|sandbox list`; vLLM/NAT/Phoenix/OTel via HTTP probes; unreachable → red chip with probe time.
- Assigning a claw to a sandbox in real mode records runner metadata only (history event says so). Sub-agent provisioning lands in M3.
- "Reset booth" shows the exact `openshell sandbox delete` argv it would run; confirm is disabled until M1.

## Next milestones (spec §8)
M1 real-gateway fixtures + lifecycle mutations · M2 Sandbox Manager + Workbench (Run/Policies/Chat/Console) · M3 Build a Claw · M4 lab catalog + Lab Mode · M5 Traces/Bench tabs.
