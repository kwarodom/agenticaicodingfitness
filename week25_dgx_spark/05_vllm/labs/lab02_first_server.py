#!/usr/bin/env python3
"""Lab 05-2 · Your first vLLM server: start it (opt-in), wait for /health, list models, send the test request.

On your Spark (over ssh, or locally) this lab:
  1. checks whether a `vllm-server` container is already running and what answers on :8000;
  2. starts the playbook's base-configuration container — ONLY with --yes (or SPARK_APPLY=1),
     because the first start downloads the model (~8 GB for the default) and reserves 80% of memory;
  3. waits for GET /health, like the playbook's `until curl … /health` loop;
  4. lists GET /v1/models and sends the playbook's "12*17" chat request through the OpenAI API.

Without a Spark, steps 1–3 are DRY (command + labelled output) and step 4 uses Ollama on this laptop as a
labelled LAPTOP STAND-IN so you still see a real OpenAI-style response.

Run: .venv/bin/python week25/05_vllm/labs/lab02_first_server.py [--yes] [--model nvidia/Llama-3.1-8B-Instruct-FP8]
Stop it later on the Spark: docker rm -f vllm-server
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, chat_any, mode, models, note, ok, result, sh, show_chat, step, up,  # noqa: E402
                      url, warn)

REF_ANSWER = 'Expected response should contain `"content": "204"` or similar mathematical calculation.'

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/Llama-3.1-8B-Instruct-FP8",
                help="HF handle from the vLLM playbook's model support matrix")
ap.add_argument("--ctx", type=int, default=32_768, help="--max-model-len (the playbook's example uses 131072)")
ap.add_argument("--yes", action="store_true", help="really start the container on the Spark")
args = ap.parse_args()
apply = args.yes or os.environ.get("SPARK_APPLY") == "1"

banner("Lab 05-2 · your first vLLM server", "base configuration from the vLLM playbook · OpenAI API on :8000")

step(1, "is anything already serving on :8000?")
sh("docker ps --filter name=vllm-server --format '{{.Names}}  {{.Image}}  {{.Status}}'; "
   "curl -sf http://localhost:8000/health && echo 'health: OK' || echo 'health: nothing on :8000'",
   example="health: nothing on :8000", timeout=30)
note("NIM also uses :8000 (Module 06). Run one of them at a time, or map the second to -p 8001:8000.")

step(2, "start the container (the playbook's base configuration)")
cmd = f"""docker run -d \\
  --name vllm-server \\
  --gpus all \\
  --ipc host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  --entrypoint "" \\
  -p 8000:8000 \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  vllm/vllm-openai:latest \\
  vllm serve {args.model} \\
    --max-model-len {args.ctx} \\
    --gpu-memory-utilization 0.8"""
if apply and mode() == "live":
    sh(cmd, timeout=900)
else:
    first, *rest = cmd.splitlines()
    print(f"$ {first}   [not run]")
    for ln in rest:
        print(f"  {ln}")
    print("→ add --yes (or SPARK_APPLY=1) to start it on your Spark. It pulls vllm/vllm-openai:latest and the "
          "model on first run, then keeps 80% of the unified memory until you remove the container.")
note("Course change from the playbook: the whole ~/.cache/huggingface is mounted so the token from "
     "`hf auth login` on the Spark is found. The playbook mounts only …/hub and passes -e HF_TOKEN.")

step(3, "wait for /health (model loading can take several minutes)")
sh("timeout 600 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done'; "
   "docker logs vllm-server 2>&1 | grep -E 'Application startup complete|Uvicorn running' | tail -2",
   reference="INFO:     Application startup complete.\n"
             "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)", timeout=660)
note("No lines printed? The model is still loading: `docker logs -f vllm-server` shows the download and "
     "load progress. Re-run this lab when it finishes.")

step(4, "the OpenAI API: GET /v1/models, then POST /v1/chat/completions")
base = url("vllm")
if mode() == "live" and up(base):
    ids = models(base)
    ok(f"{base} serves: {', '.join(ids) or '(none)'}")
    served = ids[0] if ids else args.model
else:
    note(f"no vLLM answering at {base or '(no Spark host)'} — the chat below falls back to the laptop stand-in")
    served = args.model
r = chat_any("vllm", served, [{"role": "user", "content": "12*17"}], max_tokens=150)
show_chat(r)
if r["source"] == "reference":
    print("◈ REFERENCE — expected output from the NVIDIA playbook (not your machine):")
    print(REF_ANSWER)
elif "204" in r.get("text", ""):
    ok("the answer contains 204, as the playbook expects")
else:
    warn("no '204' in the answer — a thinking model may have spent its budget; the playbook uses max_tokens 500")

result("Every client that speaks the OpenAI API — curl, the openai SDK, LiteLLM, NAT, Open WebUI — can now "
       "use this server by changing one base URL: http://<spark>:8000/v1.")
