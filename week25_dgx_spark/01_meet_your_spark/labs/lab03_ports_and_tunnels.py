#!/usr/bin/env python3
"""Lab 01-3 · Ports and tunnels: reach every service this week will start on the Spark.

Prints the course's port map, probes each port on your Spark's HTTP host (and on
localhost, in case you tunnelled), and writes ONE `ssh -L` command that forwards
all of them — the manual version of what NVIDIA Sync does for DGX Dashboard.
Read-only: it only sends GET /v1/models (or GET /) to each port.

Run: .venv/bin/python week25/01_meet_your_spark/labs/lab03_ports_and_tunnels.py
"""
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import PORTS, api_host, banner, host, models, note, result, step, table, up  # noqa: E402

WHO = {"ollama": "Module 03", "openwebui": "Module 03", "llamacpp": "Module 04", "lmstudio": "Module 04",
       "vllm": "Modules 05, 13", "sglang": "Module 06", "trtllm": "Modules 06-07", "nim": "Module 06 (same port as vLLM)",
       "litellm": "Module 08", "dashboard": "this module"}


def http_up(u: str) -> bool:
    try:
        with urlopen(u, timeout=1.5):              # noqa: S310 — probing your own Spark
            return True
    except Exception as e:                          # 401/404 still means "something is listening"
        return getattr(e, "code", None) in (401, 403, 404)


banner("Lab 01-3 · ports and tunnels", "one ssh -L command for every service you will start this week")

step(1, "probe each port on the Spark and on localhost")
spark_h = api_host("a")
rows = []
for kind, port in PORTS.items():
    if kind == "nim":
        continue                                    # NIM reuses :8000 — same probe as vLLM
    web = kind in ("openwebui", "dashboard")
    probe = (lambda h: http_up(f"http://{h}:{port}/")) if web else (lambda h: up(f"http://{h}:{port}/v1", 1.5))
    on_spark = probe(spark_h) if spark_h else False
    on_local = probe("localhost")
    ids = models(f"http://{spark_h}:{port}/v1", timeout=1.5) if (on_spark and not web) else []
    rows.append([kind, port, "● up" if on_spark else "○", "● up" if on_local else "○",
                 ", ".join(ids[:2]) or "—", WHO.get(kind, "")])
table(rows, ["service", "port", f"on {spark_h or 'Spark (no host)'}", "on localhost", "models", "used in"])
note("'on localhost' is THIS laptop. Ollama on your laptop shows up here too — it is not the Spark.")

step(2, "one tunnel for all of them")
ssh_h = host("a") or "<you>@<spark-hostname>"
fwd = " ".join(f"-L {p}:localhost:{p}" for k, p in PORTS.items() if k != "nim" and k != "ollama")
print(f"$ ssh -N {fwd} {ssh_h}")
note("Ollama (11434) is left out on purpose: your laptop may already use that port. Forward it to another "
     "local port instead: -L 21434:localhost:11434, then set SPARK_URL_OLLAMA=http://localhost:21434/v1")
note("With the tunnel open, set SPARK_API_HOST=localhost in 🖥 Spark setup so labs call the forwarded ports.")

step(3, "why tunnels: most servers bind to localhost on the Spark")
print("│ Ollama listens on 127.0.0.1:11434 unless you set OLLAMA_HOST=0.0.0.0 on the Spark.")
print("│ DGX Dashboard listens on localhost:11000 only — reach it through a tunnel or NVIDIA Sync.")
print("│ Docker servers started with -p 8000:8000 listen on every interface, including your tailnet.")
result("Tailscale makes the Spark reachable; a tunnel makes a localhost-only service reachable. "
       "Prefer tunnels for anything without authentication.")
