#!/usr/bin/env python3
"""Lab 03-1 · Ollama on the Spark: is it installed, which models does it hold, and who can reach it?

Four read-only checks on the Spark (over ssh, or locally): the Ollama version, the models
it has pulled, whether the Open WebUI container carries its own Ollama, and which address
port 11434 is bound to. Then it asks Ollama's native API (GET /api/tags, POST /api/show)
what each model really is: parameters, quantization, context length and capabilities.

Pulling a model is a big download, so it only starts with --pull (in the background, with
a log you can tail). Everything else only reads.

Run: .venv/bin/python week25/03_ollama_open_webui/labs/lab01_ollama_models.py [--pull]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (LAPTOP_OLLAMA, banner, http_json, note, result, sh, step, table, up,  # noqa: E402
                      url, where)

FIRST_MODEL = "gpt-oss:20b"         # the Open WebUI playbook's first model (~15 GB)
AGENT_MODEL = "qwen3.6:35b-a3b"     # the playbook's "agent-ready" pick for DGX Spark
PULL = "--pull" in sys.argv


def native(base_v1: str) -> str:
    """Ollama's native API lives at the server root; the OpenAI one under /v1."""
    return base_v1[:-3] if base_v1.endswith("/v1") else base_v1


def describe(base_v1: str) -> list[list]:
    """One row per model from /api/tags + /api/show (the native API)."""
    root = native(base_v1)
    rows = []
    for m in http_json("GET", f"{root}/api/tags", timeout=10).get("models", []):
        name = m.get("name", "")
        if name.endswith(":cloud") or "-cloud" in name:
            continue                                            # cloud models do not run locally
        d = m.get("details") or {}
        try:
            show = http_json("POST", f"{root}/api/show", {"model": name}, timeout=20)
        except Exception:  # noqa: BLE001
            show = {}
        info = show.get("model_info") or {}
        ctx = next((v for k, v in info.items() if k.endswith(".context_length")), "?")
        experts = next((v for k, v in info.items() if k.endswith(".expert_count")), 0)
        rows.append([name, f"{m.get('size', 0) / 1e9:.1f} GB", d.get("parameter_size", "?"),
                     d.get("quantization_level", "?"), f"{ctx:,}" if isinstance(ctx, int) else ctx,
                     f"MoE ×{experts}" if experts else "dense",
                     ", ".join(c for c in show.get("capabilities", []) if c != "completion") or "—"])
    return rows


banner("Lab 03-1 · Ollama on the Spark", "installed? which models? who can reach port 11434?")

step(1, "is Ollama installed on the Spark?")
sh("ollama --version", example="ollama version is 0.x.y")
note("Not installed? The Ollama playbook installs it with the official script (Section 1 of the tutorial).")

step(2, "which models has it pulled?")
sh("ollama list",
   example="NAME           ID              SIZE     MODIFIED\n"
           "gpt-oss:20b    <12-hex-id>     <size>   <when>")
if PULL:
    sh(f"mkdir -p ~/w25/logs && {{ nohup ollama pull {FIRST_MODEL} > ~/w25/logs/ollama-pull.log 2>&1 < /dev/null & }}; "
       f"echo started pull of {FIRST_MODEL}, pid $!",
       example=f"started pull of {FIRST_MODEL}, pid 12345")
    note("Follow it with:  ssh spark-a tail -f ~/w25/logs/ollama-pull.log   (Ctrl+C stops the tail, not the pull)")
else:
    print(f"→ add --pull to start `ollama pull {FIRST_MODEL}` in the background (~15 GB download).")
    print(f"→ the playbook's agent-ready model is `{AGENT_MODEL}`; pull it the same way when you need tools.")

step(3, "does the Open WebUI container carry its own Ollama?")
r = sh("docker ps --filter name=open-webui --format '{{.Names}}  {{.Status}}  {{.Ports}}'",
       example="open-webui  Up 2 hours  0.0.0.0:12000->8080/tcp")
if r.live and "open-webui" in r.out:
    sh("docker exec open-webui ollama list",
       example="NAME           ID              SIZE     MODIFIED")
note("The :ollama image bundles its own Ollama, with models in the open-webui-ollama volume. Only port 8080 "
     "is published, so that Ollama is NOT on host port 11434, and `ollama list` on the host does not see it.")

step(4, "which address is port 11434 bound to?")
sh("ss -tln | grep 11434 || echo 'nothing listening on 11434'",
   example="LISTEN 0      4096       127.0.0.1:11434      0.0.0.0:*")
note("127.0.0.1:11434 → only the Spark itself (and ssh -L tunnels) can reach it. "
     "*:11434 or 0.0.0.0:11434 → every device on your LAN and tailnet can. Section 2 shows how to switch.")

step(5, "ask the native API what each model really is")
spark_base = url("ollama") if where("a") != "dry" else ""
targets = []
if spark_base and up(spark_base):
    targets.append((spark_base, "Spark A — LIVE"))
else:
    note(f"the Spark's Ollama is not reachable over HTTP ({spark_base or 'no host'}); "
         "open a tunnel or set OLLAMA_HOST on the Spark (Section 2).")
if up(LAPTOP_OLLAMA):
    targets.append((LAPTOP_OLLAMA, "THIS laptop — LAPTOP STAND-IN, not the Spark"))
for base, who in targets:
    print(f"\n→ GET {native(base)}/api/tags  +  POST /api/show for each model   [{who}]")
    rows = describe(base)
    if rows:
        table(rows, ["model", "on disk", "params", "quant", "context", "kind", "capabilities"])
    else:
        print("│ (no local models)")
if not targets:
    print("◈ no Ollama answered on the Spark or on this laptop — nothing to describe.")

note("'tools' in capabilities → the model can return tool_calls (lab 04). 'thinking' → it can emit a separate "
     "reasoning stream (lab 02).")
result("Ollama is one binary, one port (11434) and one model store. The native API describes models; "
       "the OpenAI-compatible /v1 API is what every other tool this week talks to.")
