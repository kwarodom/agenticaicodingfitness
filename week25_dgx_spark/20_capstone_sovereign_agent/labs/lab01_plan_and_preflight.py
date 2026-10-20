#!/usr/bin/env python3
"""Lab 20-1 · The plan and the preflight: what runs on which Spark, does it fit, are both Sparks ready?

Step 1 is arithmetic (runs anywhere): the capstone's services, their ports and the memory each vLLM
reserves with --gpu-memory-utilization. Step 2 runs read-only checks on Spark A and Spark B over ssh
(GB10, free memory, free disk, Docker, and whether the ports the plan needs are free). DRY shows the
commands with EXAMPLE output.

Run: .venv/bin/python week25/20_capstone_sovereign_agent/labs/lab01_plan_and_preflight.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _capstone as C  # noqa: E402
from sparkkit import SPEC, banner, bar, host, note, result, sh, step, table, where  # noqa: E402

banner("Lab 20-1 · plan and preflight", "two Sparks · four services · one gateway")

step(1, "the plan: what runs where (arithmetic, not a measurement)")
table([[f"Spark {s}", svc, port or "—", model[:44], f"{gb:.1f} GB", m] for s, svc, port, model, gb, m in C.PLAN],
      ["where", "service", "port", "model / role", "weights", "module"])
RESERVE = {"A": 0.4, "B": 0.3}          # --gpu-memory-utilization in lab 20-2 (A: Module 05's agent recipe)
for s, frac in RESERVE.items():
    gb = frac * SPEC["memory_gb"]
    print(f"│ Spark {s}: vLLM reserves {frac:.0%} of {SPEC['memory_gb']} GB = {gb:5.1f} GB  {bar(gb, SPEC['memory_gb'])}")
note("Spark A keeps ~77 GB free for the gateway, the sandbox and the OS. Spark B keeps ~90 GB free: enough to "
     "retrain the next hotel adapter (Module 09, LoRA on 4B) while the current one serves.")
note("Why two Sparks: the brain and the router scale and fail separately, and retraining never touches the agent's "
     "Spark. One Spark also works: run both vLLMs on A with --gpu-memory-utilization 0.4 + 0.3 on ports 8000/8001.")

step(2, "preflight on both Sparks (read-only)")
CHECK = ("nvidia-smi --query-gpu=name --format=csv,noheader; free -g | awk '/Mem:/{print \"free_gb\",$7}'; "
         "df -BG / | awk 'NR==2{print \"disk_free\",$4}'; docker ps --format '{{.Names}}' | head -5; "
         "for p in 8000 4000; do (ss -ltn | grep -q \":$p \") && echo \"port $p BUSY\" || echo \"port $p free\"; done")
EXAMPLE = "NVIDIA GB10\nfree_gb 112\ndisk_free 3100G\nport 8000 free\nport 4000 free"
rows = []
for which in ("a", "b"):
    print(f"\n→ Spark {which.upper()} ({host(which) or 'not configured'})")
    r = sh(CHECK, which, example=EXAMPLE, timeout=60)
    out = r.out or ""
    free = re.search(r"free_gb\s+(\d+)", out)
    disk = re.search(r"disk_free\s+(\d+)G", out)
    need_port = "8000" if which == "b" else "8000 + 4000"
    busy = re.findall(r"port (\d+) BUSY", out)
    blocking = [p for p in busy if p in need_port.split(" + ")]   # Spark B only serves :8000; its :4000 is not ours
    status = "◈ " + r.source if r.source != "live" else ("✓" if "GB10" in out and not blocking else "✕")
    rows.append([f"Spark {which.upper()}", "GB10" if "GB10" in out else "?", free.group(1) + " GB" if free else "?",
                 disk.group(1) + " GB" if disk else "?", need_port, ", ".join(busy) or "none", status])
print()
table(rows, ["Spark", "GPU", "free mem", "free disk", "needs ports", "busy", "ready"])
if where("a") == "dry":
    note("DRY: the rows above are EXAMPLE values. With both Sparks connected, a ✕ names what to stop first "
         "(Module 19's port-clash list: txt2kg's Ollama on :11434, the chatbot backend on :8000).")
result("Plan fits: 20 GB brain + 8 GB router on two 128 GB Sparks, each well under its reservation. Next: lab 20-2.")
