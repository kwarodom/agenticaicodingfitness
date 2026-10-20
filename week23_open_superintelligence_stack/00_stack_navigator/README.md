# 00 · Stack Navigator — the Week 23 hub

An interactive map of NVIDIA's **Open Superintelligence Stack**, bottom to top:
hardware → runtime → model → harness → flywheel. 16 nodes, each with what/why,
the real DGX Spark commands, the build.nvidia.com cloud path, a live demo prompt,
and a link to the Week 23 app (ports 8100–8111) that teaches it in depth.

Two modes, auto-detected: **REAL** streams each node's demo prompt against a live
OpenAI-compatible endpoint; **SIM** streams a faithful canned answer at plausible
DGX-Spark tok/s. The real commands are part of the content — shown either way.

## Launch

```bash
uv pip install -r week23/00_stack_navigator/requirements.txt
.venv/bin/python week23/00_stack_navigator/tutorial_server.py
# → http://127.0.0.1:8112   (NT_GUIDE_PORT overrides; auto-picks the next free port)
```

## Point it at a real model

```bash
# DGX Spark on your LAN / this laptop (Ollama)
export DGX_CONN=local                          # http://localhost:11434/v1

# DGX Spark over Tailscale
export DGX_CONN=tunnel
export DGX_TUNNEL_URL=http://your-spark.your-tailnet.ts.net:11434/v1

# build.nvidia.com hosted NIMs (the cloud on-ramp — usage-billed, not sovereign)
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...                   # free key at build.nvidia.com

# also honored: DGX_BASE_URL (explicit, always wins), DGX_MODEL, DGX_MODE=sim|real
```

## API

`GET /api/stack` (the full content) · `GET /api/status` · `POST /api/probe` ·
`GET /api/apps` (12 apps + running dots) · `POST /api/run` (NDJSON stream).
