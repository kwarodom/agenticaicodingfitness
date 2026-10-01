#!/usr/bin/env python3
"""Lab 20-2 · Bring it up: the router on Spark B, the brain on Spark A, the gateway in front.

Prints the three launch sequences with the right host names, and writes the gateway config the Spark
will use. With --launch (or SPARK_APPLY=1) it starts them in order and waits for each /v1/models:
    1. Spark B · vLLM: Qwen3-4B base + the hotel-ft LoRA adapter   (Module 13's path A)
    2. Spark A · vLLM: the agent-ready Qwen3.6 recipe             (Module 05, quoted from the older playbook)
    3. Spark A · LiteLLM gateway with two aliases                   (Module 08's on-Spark install)
Deviations from the playbooks are marked in the printed commands.

Run: .venv/bin/python week25/20_capstone_sovereign_agent/labs/lab02_bring_up.py [--launch]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _capstone as C  # noqa: E402
from sparkkit import api_host, banner, check, note, put, result, sh, step, table, warn, where  # noqa: E402

APPLY = "--launch" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
B_IP = api_host("b") or "spark-b"

ROUTER_B = f"""mkdir -p ~/w25/adapters && cp -r ~/w25/m09/saves/qwen3-4b-hotel/lora/sft ~/w25/adapters/hotel-ft.new \\
  && rm -rf ~/w25/adapters/hotel-ft && mv ~/w25/adapters/hotel-ft.new ~/w25/adapters/hotel-ft
docker rm -f w25-router 2>/dev/null
docker run -d --name w25-router --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/w25/adapters:/adapters" \\
  --entrypoint '' vllm/vllm-openai:latest \\
  vllm serve {C.ROUTER_BASE} --max-model-len 8192 --gpu-memory-utilization 0.3 \\
    --enable-lora --lora-modules hotel-ft=/adapters/hotel-ft --max-lora-rank 16"""

BRAIN_A = f"""docker rm -f w25-brain 2>/dev/null
docker run -d --name w25-brain --gpus all -p 127.0.0.1:8000:8000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  vllm/vllm-openai:latest \\
  {C.BRAIN_MODEL} \\
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --trust-remote-code --kv-cache-dtype fp8 \\
  --attention-backend flashinfer --moe-backend marlin --gpu-memory-utilization 0.4 --max-model-len 262144 \\
  --max-num-seqs 4 --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \\
  --enable-prefix-caching --speculative-config '{{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}}' \\
  --load-format fastsafetensors --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice"""

GATEWAY_A = """[ -x ~/w25/litellm-venv/bin/litellm ] || { python3 -m venv ~/w25/litellm-venv && ~/w25/litellm-venv/bin/pip install 'litellm[proxy]==1.89.0'; }
mkdir -p ~/w25/litellm ~/w25/logs
[ -f ~/w25/litellm/master.key ] || ( umask 077; echo "sk-$(openssl rand -hex 16)" > ~/w25/litellm/master.key )
pkill -u "$(id -un)" -x litellm 2>/dev/null   # -x on the process name: -f would match this shell's own command line
LITELLM_MASTER_KEY="$(cat ~/w25/litellm/master.key)" LITELLM_LOCAL_MODEL_COST_MAP=True \\
  nohup ~/w25/litellm-venv/bin/litellm --config ~/w25/litellm/capstone.yaml \\
  --host 0.0.0.0 --port 4000 --telemetry False > ~/w25/logs/litellm.log 2>&1 < /dev/null &"""

banner("Lab 20-2 · bring it up", "router on Spark B · brain on Spark A · gateway on Spark A")

step(1, "the gateway config the Spark will run (no laptop fallbacks: the sandbox must stay on the Sparks)")
cfg = C.gateway_config()
cfg["model_list"] = [d for d in cfg["model_list"] if not d["model_name"].endswith("-laptop")]
cfg["model_list"][0]["litellm_params"]["api_base"] = "http://localhost:8000/v1"          # brain is local to A
cfg["model_list"][1]["litellm_params"]["api_base"] = f"http://{B_IP}:8000/v1"             # router on B
cfg["router_settings"]["fallbacks"] = []
spark_cfg = C.G.write_config(cfg, C.RUNS / "capstone-gateway.spark.yaml")
table([[d["model_name"], d["litellm_params"]["model"], d["litellm_params"]["api_base"]] for d in cfg["model_list"]],
      ["alias", "backend", "api_base (as seen from Spark A)"])
note(f"written to {spark_cfg.relative_to(C.WEEK)} — uploaded as ~/w25/litellm/capstone.yaml on Spark A")

for title, which, cmd, deviation in (
        ("1 · Spark B — the fine-tuned router (Module 13, path A)", "b", ROUTER_B,
         "LoRA flags are vLLM's own CLI (course addition, Module 05 §7)"),
        ("2 · Spark A — the agent brain (Module 05 §4, older vLLM playbook)", "a", BRAIN_A,
         "course deviations: -d --name instead of -it; -p 127.0.0.1:8000 so only the gateway on A can reach it; "
         "no -e HF_TOKEN (the model is not gated; use `hf auth login` if yours is)"),
        ("3 · Spark A — the LiteLLM gateway (Module 08 §7)", "a", GATEWAY_A,
         "binds 0.0.0.0:4000 on purpose: OpenShell's provider must use the Spark's IP, not localhost "
         "(openshell playbook troubleshooting); the master key is the protection")):
    step(title[0], title[4:])
    for line in cmd.splitlines():
        print(f"  {line}")
    print(f"◆ {deviation}")

step(4, "launch in order and wait for each server (opt-in)")
if not APPLY:
    note("Not launched. Re-run with --launch once Module 09's adapter exists on Spark B "
         "(train there, or copy ~/w25/m09 across) and both Sparks answer ssh.")
    raise SystemExit(0)
if where("a") == "dry" or where("b") == "dry":
    warn("--launch needs BOTH Sparks reachable (see lab 20-1).")
    raise SystemExit(0)
sh(ROUTER_B, "b", timeout=300)
put(spark_cfg, "~/w25/litellm/capstone.yaml", "a")
sh(BRAIN_A, "a", timeout=300)
# Poll each server from its own Spark: the brain is bound to 127.0.0.1 on A, so a laptop cannot reach it directly.
PROBE = ("curl -s -m 3 http://localhost:8000/v1/models | python3 -c "
         "'import sys,json; print(\" \".join(m[\"id\"] for m in json.load(sys.stdin)[\"data\"]))' 2>/dev/null")
for label_, which, want in (("router on B", "b", "hotel-ft"), ("brain on A", "a", C.BRAIN_MODEL)):
    print(f"│ waiting for {label_}: localhost:8000/v1/models on Spark {which.upper()} lists {want} "
          "(first start downloads weights) …")
    ids: list[str] = []
    for _ in range(120):
        ids = sh(PROBE, which, timeout=30, quiet=True).out.split()
        if want in ids:
            break
        time.sleep(10)
    check(want in ids, f"{label_} is up: {', '.join(ids)}", f"{label_} not up after 20 min — docker logs on that Spark")
sh(GATEWAY_A, "a", timeout=600)
GW_PROBE = ('curl -s -m 3 -H "Authorization: Bearer $(cat ~/w25/litellm/master.key)" http://localhost:4000/v1/models '
            "| python3 -c 'import sys,json; print(\" \".join(m[\"id\"] for m in json.load(sys.stdin)[\"data\"]))' 2>/dev/null")
print("│ waiting for the gateway on A: localhost:4000/v1/models lists agent-brain and hotel-router …")
ids = []
for _ in range(30):
    ids = sh(GW_PROBE, "a", timeout=30, quiet=True).out.split()
    if {"agent-brain", "hotel-router"} <= set(ids):
        break
    time.sleep(5)
if not check({"agent-brain", "hotel-router"} <= set(ids), f"gateway on A is up: {', '.join(ids)}",
             "gateway not up after 2.5 min: read ~/w25/logs/litellm.log on Spark A"):
    raise SystemExit(1)
result("All three up. Lab 20-3 runs the acceptance scenarios; point it at the Spark gateway with "
       "SPARK_URL_LITELLM or run the NAT agent on Spark A.")
