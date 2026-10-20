#!/usr/bin/env bash
# Open WebUI on Spark A, chatting through the capstone's LiteLLM gateway (START_HERE.md 5c).
#   week25/spark_host/openwebui_gateway.sh start   # after lab 20-2 --launch: models agent-brain + hotel-router
#   week25/spark_host/openwebui_gateway.sh stop
# Runs on Spark A as the instructor. The container runs as SPARK_HOST's user (sparklab), binds 127.0.0.1:12000
# only (so it can run without a login), and gets the gateway's master key on the Spark; the key is never printed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cfg() { local v=${!1:-}; [[ -n $v ]] && { echo "$v"; return; }; sed -n "s/^$1=//p" "$HERE/../.env.local" 2>/dev/null | tail -1 || true; }
A=$(cfg SPARK_HOST); A=${A:-sparklab@localhost}
URL=http://127.0.0.1:12000
DATA="$HERE/../09_llama_factory/data/hotel_ops.json"

case "${1:-}" in
  stop) ssh -o BatchMode=yes "$A" 'docker rm -f w25-openwebui >/dev/null 2>&1; echo "✓ Open WebUI stopped (chats stay in the w25-openwebui-data volume)"' ;;
  start)
    [[ -f $DATA ]] || { echo "✕ $DATA missing: run Module 09 lab 02 once (it writes the dataset)"; exit 1; }
    # ENABLE_PERSISTENT_CONFIG=False: Open WebUI otherwise keeps the connection and default model from its
    # first start in the volume and ignores these variables on every later start.
    ssh -o BatchMode=yes "$A" 'set -e
      K=$(cat ~/w25/litellm/master.key 2>/dev/null) || { echo "✕ no gateway key: run lab 20-2 --launch first"; exit 1; }
      curl -s -m 5 -o /dev/null -H "Authorization: Bearer $K" http://localhost:4000/v1/models || { echo "✕ the gateway on :4000 is not up: run lab 20-2 --launch"; exit 1; }
      docker rm -f w25-openwebui >/dev/null 2>&1 || true
      docker run -d --name w25-openwebui -p 127.0.0.1:12000:8080 --add-host host.docker.internal:host-gateway \
        -e ENABLE_PERSISTENT_CONFIG=False \
        -e OPENAI_API_BASE_URL=http://host.docker.internal:4000/v1 -e OPENAI_API_KEY="$K" \
        -e ENABLE_OLLAMA_API=False -e WEBUI_AUTH=False -e DEFAULT_MODELS=agent-brain \
        -v w25-openwebui-data:/app/backend/data ghcr.io/open-webui/open-webui:ollama >/dev/null
      echo "▶ Open WebUI started, pointed at the gateway"'
    for _ in $(seq 60); do curl -s -o /dev/null -m 3 "$URL/health" && break; sleep 5; done
    # hotel-router was fine-tuned with one fixed system prompt (Module 09): attach it, or it answers in prose.
    # And switch off Open WebUI's built-in tools for it: in a browser session Open WebUI offers every model its
    # tools (tool_choice "auto"), and the router's vLLM (no tool parser, by design) rejects that with a 400.
    python3 - "$URL" "$DATA" <<'EOF'
import json, sys, urllib.request
url, data = sys.argv[1], sys.argv[2]
def call(path, body, tok=None):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + tok} if tok else {})}
    return json.load(urllib.request.urlopen(urllib.request.Request(url + path, json.dumps(body).encode(), h), timeout=60))
tok = call("/api/v1/auths/signin", {"email": "", "password": ""})["token"]   # WEBUI_AUTH=False: the built-in admin
system = json.load(open(data, encoding="utf-8"))[0]["system"]
model = {"id": "hotel-router", "base_model_id": None, "name": "hotel-router", "is_active": True, "params": {"system": system},
         "meta": {"description": "Module 09 fine-tune (Qwen3-4B + hotel-ft LoRA) on Spark B: one line of routing JSON",
                  "capabilities": {"builtin_tools": False}}}
try:
    call("/api/v1/models/create", model, tok)
except urllib.error.HTTPError:                                                 # already there (kept in the volume)
    call("/api/v1/models/model/update?id=hotel-router", model, tok)
req = urllib.request.Request(url + "/api/models", headers={"Authorization": "Bearer " + tok})
ids = [m["id"] for m in json.load(urllib.request.urlopen(req, timeout=60)).get("data", [])]
ok = {"agent-brain", "hotel-router"} <= set(ids)
print(("✓" if ok else "✕") + " models: " + ", ".join(i for i in ids if i != "arena-model"))
req = urllib.request.Request(url + "/api/v1/models/model?id=hotel-router", headers={"Authorization": "Bearer " + tok})
info = json.load(urllib.request.urlopen(req, timeout=60))
has = (info.get("params") or {}).get("system") == system
print("✓ hotel-router has its training system prompt" if has else "✕ hotel-router has no system prompt: it will answer in prose")
notools = ((info.get("meta") or {}).get("capabilities") or {}).get("builtin_tools") is False
print("✓ hotel-router gets no built-in tools" if notools else "✕ hotel-router still gets built-in tools: chats fail with \"auto\" tool choice")
ok = ok and has and notools
if not ok:
    print("  check lab 20-2 and the gateway log (~/w25/logs/litellm.log on Spark A)")
print(f"═ open {url} in a browser on Spark A (no login); from a laptop: ssh -N -L 12000:localhost:12000 <SPARK_HOST>")
sys.exit(0 if ok else 1)
EOF
    ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
