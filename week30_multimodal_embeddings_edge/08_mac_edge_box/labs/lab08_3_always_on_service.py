#!/usr/bin/env python3
"""Lab 08-3 · The Mac mini as the hotel's embedding server: a local service, a sustained-load test, power.

Tina Huang's home-server tier, for embeddings. This lab starts common/embed_service.py (MLX 4-bit) on this
Mac's loopback address, then plays a small device on the network: plain HTTP + JSON, the way an ESP32 or a
BMS gateway would call it. It measures round-trip latency and payload sizes, runs a 60-second sustained load
to check for slowdown, reads the power source (and the battery draw when unplugged — no sudo), and writes —
but does NOT install — a launchd file that would keep the service running and restart it if it stops.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_3_always_on_service.py
"""
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import mackit  # noqa: E402

mackit.ensure_mlx()                                      # → week30_multimodal_embeddings_edge/.venv-mlx

import egress  # noqa: E402
import embedkit as ek  # noqa: E402

RUNS = ek.runs(__file__)


def free_port(start: int = 8765) -> int:
    import socket
    for p in range(start, start + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def call(port: int, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST" if body else "GET",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(raw), (time.perf_counter() - t0) * 1000, len(data or b""), len(raw)


ek.banner("Lab 08-3 · the Mac as the hotel's embedding server", "local HTTP service · sustained load · power · launchd")

ek.step(1, "start the service on loopback (MLX 4-bit, hotel notes pre-embedded)")
import uvicorn  # noqa: E402
import embed_service  # noqa: E402

port = free_port()
server = uvicorn.Server(uvicorn.Config(embed_service.build_app("4bit"), host="127.0.0.1", port=port, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(200):
    if server.started:
        break
    time.sleep(0.1)
health, _, _, _ = call(port, "/health")
ek.ok(f"listening on http://127.0.0.1:{port} · {health['model']} · {health['docs']} notes indexed")

with egress.guard(allow_ports={port}) as log:
    ek.step(2, "be the small device: plain JSON over HTTP")
    r, ms, up, down = call(port, "/search", {"q": "someone is stuck in the lift", "k": 2})
    print(f"  POST /search {{\"q\": \"someone is stuck in the lift\"}}  →  {r['hits']}")
    rows = []
    for dim in (768, 256, 128):
        r, ms, up, down = call(port, "/embed", {"texts": ["water under the air handler"], "dim": dim})
        rows.append([dim, f"{up} B", f"{down:,} B", f"{ms:.1f} ms"])
    ek.table(rows, ["dim", "request", "response (JSON)", "round trip"])
    ek.note("a microcontroller with 8 MB of RAM can parse a 128-d answer (≈ 2.7 KB of JSON); send /search instead and "
            "it never handles a vector at all")

    ek.step(3, "round-trip latency: 30 searches")
    times = [call(port, "/search", {"q": q["q"], "k": 1})[1] for q in ek.load_kb()["queries"] + ek.load_kb()["queries"][:12]]
    ek.ok(f"p50 {statistics.median(times):.1f} ms · p95 {ek.percentile(times, 95):.1f} ms · over loopback HTTP + JSON")

    ek.step(4, "sustained load: 60 s of 32-note batches — does the Mac slow down?")
    batch = [f"{d['title']}: {d['text']}" for d in ek.load_kb()["docs"]] * 2
    batch = batch[:32]
    src, w0 = mackit.power_source(), mackit.battery_watts()
    windows, t_end, done, watts = [], time.time() + 60, 0, []
    win_start, win_done = time.time(), 0
    while time.time() < t_end:
        call(port, "/embed", {"texts": batch, "task": "document", "dim": 256})
        done += 32
        win_done += 32
        if time.time() - win_start >= 15:
            windows.append(win_done / (time.time() - win_start))
            w = mackit.battery_watts()
            if w:
                watts.append(w)
            win_start, win_done = time.time(), 0
    for i, rate in enumerate(windows, 1):
        print(f"  window {i} (15 s): {rate:6.0f} notes/s  {ek.bar(rate, max(windows), 24)}")
    drop = (1 - windows[-1] / windows[0]) * 100 if len(windows) > 1 else 0.0
    (ek.ok if drop < 10 else ek.warn)(f"{done:,} notes in 60 s · last window vs first: {-drop:+.1f} %")
    ek.step(5, "power — no sudo needed")
    if src == "battery" and watts:
        avg = statistics.mean(watts)
        ek.ok(f"on battery: whole-Mac draw ≈ {avg:.1f} W under load (idle before: {w0 or 0:.1f} W) → "
              f"≈ {avg * 1000 / statistics.mean(windows):.1f} J per 1,000 notes")
    else:
        ek.note(f"power source: {src} — the battery is not discharging, so its draw cannot be read without sudo")
        ek.note("unplug and rerun for whole-Mac watts, or run Apple's tool yourself in the ⌨ terminal:")
        print("  $ sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 10")
s = egress.summary(log)
ek.ok(f"left this machine: {s['outbound_allowed']} connections (attempts {s['attempts']}, all to 127.0.0.1:{port})")
server.should_exit = True

ek.step(6, "keep it running: a launchd file (written, NOT installed)")
py = mackit.MLX_PY
svc = ek.COMMON / "embed_service.py"
plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.alto.edge-memory</string>
  <key>ProgramArguments</key>
  <array><string>{py}</string><string>{svc}</string><string>--host</string><string>127.0.0.1</string><string>--port</string><string>8765</string></array>
  <key>EnvironmentVariables</key><dict><key>HF_HUB_OFFLINE</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/alto-edge-memory.log</string>
  <key>StandardErrorPath</key><string>/tmp/alto-edge-memory.log</string>
</dict>
</plist>
"""
out = RUNS / "com.alto.edge-memory.plist"
out.write_text(plist)
lint = subprocess.run(["plutil", "-lint", str(out)], capture_output=True, text=True)
ek.ok(f"wrote week30_multimodal_embeddings_edge/08_mac_edge_box/.runs/{out.name} · plutil: {lint.stdout.strip().split(': ')[-1]}")
ek.note("to install it yourself (persistent — starts at every login):")
print(f"  $ cp {out} ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/{out.name}")
print("  $ launchctl unload ~/Library/LaunchAgents/com.alto.edge-memory.plist   # to stop it")
ek.result("an always-on Mac is a sovereign embedding appliance: small devices send words, the Mac sends back "
          "meaning, and nothing leaves the building.")
