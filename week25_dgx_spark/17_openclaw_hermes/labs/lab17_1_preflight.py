#!/usr/bin/env python3
"""Lab 17-1 · Agent preflight: is the Spark ready for OpenClaw and Hermes on a local model?

Read-only checks on your Spark (locally, or over `ssh $SPARK_HOST`). The first three are the
Hermes playbook's Step 1; the next ones check what both playbooks depend on: a vLLM server
answering on :8000 with a model list, WHO can reach that port (the playbooks say keep it
bound to the Spark), and whether OpenClaw or Hermes is already installed (the Hermes
installer offers to migrate an existing OpenClaw; this course says no).

Nothing is installed or changed.

Run: .venv/bin/python week25/17_openclaw_hermes/labs/lab17_1_preflight.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, result, sh, step, table, warn  # noqa: E402

AGENT_READY = "nvidia/Qwen3.6-35B-A3B-NVFP4"          # both playbooks' DGX Spark recommendation

CHECKS = [
    ("Linux", "uname -a", "Linux spark-abcd 6.11.0-1016-nvidia #16-Ubuntu SMP … aarch64 GNU/Linux",
     lambda o: "Linux" in o, "Both playbooks target Linux (DGX OS). Check SPARK_HOST points at the Spark."),
    ("curl", "curl --version | head -1", "curl 8.5.0 (aarch64-unknown-linux-gnu) …",
     lambda o: o.strip().startswith("curl "), "Install curl: sudo apt install -y curl"),
    ("git", "git --version", "git version 2.43.0",
     lambda o: o.strip().startswith("git version"), "Install git: sudo apt install -y git"),
    ("vLLM model list", "curl -sS --max-time 5 http://localhost:8000/v1/models || echo 'no server on :8000'",
     '{"object":"list","data":[{"id":"nvidia/Qwen3.6-35B-A3B-NVFP4","object":"model","max_model_len":262144}]}',
     lambda o: '"data"' in o, "Start vLLM first (Section 2). Both installers read the model list from :8000."),
    ("Who can reach :8000", "ss -ltn 2>/dev/null | grep -E ':8000\\b' || echo 'nothing listening on :8000'",
     "LISTEN 0      4096         0.0.0.0:8000       0.0.0.0:*",
     lambda o: True, ""),
    ("Existing installs", "command -v openclaw hermes 2>/dev/null; ls -d ~/.openclaw ~/.hermes 2>/dev/null; echo '(end of list)'",
     "(end of list)", lambda o: True, ""),
    ("OpenClaw gateway process", "pgrep -fa '[o]penclaw' | grep -v 17_openclaw_hermes | head -3 | grep . || echo 'no openclaw process'",
     "no openclaw process", lambda o: True, ""),
]

banner("Lab 17-1 · agent preflight", "read-only checks before installing OpenClaw or Hermes on the Spark")
note("Both playbooks: run agents on a dedicated or isolated system, with dedicated accounts, and only the data they need.")

rows, failures, outs = [], [], {}
for i, (title, cmd, ex, test, fix) in enumerate(CHECKS, 1):
    step(i, title)
    r = sh(cmd, timeout=30, example=ex)
    outs[title] = r
    passed = bool(test(r.out or ""))
    status = ("✓" if passed else "✕") if r.live else "◈ " + r.source
    last = (r.out or "").strip().splitlines()
    rows.append([title, status, last[-1][:52] if last else "—"])
    if r.live and not passed and fix:
        failures.append((title, fix))

print()
table(rows, ["check", "result", "last line of output"])

step(len(CHECKS) + 1, "what the answers mean")
served = []
m = outs["vLLM model list"]
try:
    served = [d.get("id") for d in json.loads(m.out).get("data", [])] if '"data"' in (m.out or "") else []
except (json.JSONDecodeError, AttributeError):
    served = re.findall(r'"id"\s*:\s*"([^"]+)"', m.out or "")
if m.live:
    if AGENT_READY in served:
        note(f"{AGENT_READY} is served — the handle both playbooks recommend for DGX Spark.")
    elif served:
        note(f"served: {', '.join(served)} — use this exact handle as the OpenClaw model id / Hermes model.default.")
bind = outs["Who can reach :8000"]
if bind.live and re.search(r"(0\.0\.0\.0|\[::\]|\*):8000", bind.out or ""):
    warn(":8000 listens on ALL interfaces (docker -p 8000:8000 does this). Anyone on your LAN or tailnet can use "
         "the model. The playbooks: do not expose it without authentication — bind -p 127.0.0.1:8000:8000.")
elif not bind.live:
    note("DRY: the EXAMPLE shows 0.0.0.0:8000 — what `docker run -p 8000:8000` gives you. See Section 2 for the fix.")
inst = outs["Existing installs"]
if inst.live and ".openclaw" in (inst.out or "") and ".hermes" not in (inst.out or ""):
    note("OpenClaw is installed. When the Hermes installer asks to import/migrate from OpenClaw, answer n.")

if failures:
    print()
    for title, fix in failures:
        check(False, "", f"{title}: {fix}")
    result(f"{len(failures)} check(s) need attention before installing the agents.")
elif all(r[1] == "✓" for r in rows):
    result("ready: vLLM answers on :8000 — continue with OpenClaw (Section 4) and Hermes (Section 5).")
else:
    result("DRY run: EXAMPLE rows are illustrative shapes, not your Spark. Connect a Spark (🖥 Spark setup) to check yours.")
