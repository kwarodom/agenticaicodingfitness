#!/usr/bin/env python3
"""Lab 17-3 · Agent config generator: one served model, two agent configs, both validated.

Runs on your laptop (real). It finds the model your Spark's vLLM serves (or uses the
playbooks' DGX Spark default, nvidia/Qwen3.6-35B-A3B-NVFP4), then writes:

  • openclaw.models.json — the `models` section for ~/.openclaw/openclaw.json, in the
    exact shape the OpenClaw playbook shows (provider `vllm`, baseUrl, apiKey placeholder,
    api, models[] with id/name/contextWindow/maxTokens);
  • hermes-config.sh     — the Hermes playbook's non-interactive fallback
    (`hermes config set model.provider/base_url/default` + a `hermes -z` check).

Both are validated: the model id must be one the endpoint serves, baseUrl must end in /v1
and point at the Spark itself, contextWindow must be at least the playbooks' 32K minimum,
and apiKey must be a placeholder, never a real key. Then it proves the (endpoint, model)
pair answers with one real chat: on the Spark's vLLM, or on the laptop stand-in (labelled).

It never edits ~/.openclaw/openclaw.json. With a Spark, the two files are copied to
~/w25/agents/ for you to merge by hand.

Run: .venv/bin/python week25/17_openclaw_hermes/labs/lab17_3_agent_configs.py [--context 262144]
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, chat_any, check, models, note, pick_laptop_model, put, resolve, result,  # noqa: E402
                      show_chat, step, table, where)

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / ".runs"
DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"          # both playbooks' DGX Spark agent-ready model
SPARK_BASE = "http://localhost:8000/v1"                 # the agents run ON the Spark, next to vLLM
MIN_CONTEXT = 32_768                                    # OpenClaw playbook: "at least 32K tokens"
SECRET = re.compile(r"^(sk-|nvapi-|hf_|xox[bap]-|ghp_)|^[A-Za-z0-9_\-]{32,}$")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def openclaw_models(base_url: str, model_id: str, context: int, max_tokens: int = 8192) -> dict:
    """The OpenClaw playbook's example `models` section, with your values."""
    return {"models": {"mode": "merge", "providers": {"vllm": {
        "baseUrl": base_url, "apiKey": "vllm", "api": "openai-responses",
        "models": [{"id": model_id, "name": model_id, "reasoning": True, "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": context, "maxTokens": max_tokens}]}}}}


def hermes_script(base_url: str, model_id: str) -> str:
    """The Hermes playbook's non-interactive SSH fallback, with your values."""
    return "\n".join([
        "#!/usr/bin/env bash",
        "# Hermes: point the agent at the local vLLM endpoint (Hermes playbook, non-interactive fallback)",
        "set -e",
        'export PATH="$HOME/.local/bin:$PATH"',
        "hermes config set model.provider custom",
        f"hermes config set model.base_url {shlex.quote(base_url)}",
        f"hermes config set model.default {shlex.quote(model_id)}",
        'hermes -z "Reply exactly HERMES_OK"',
    ]) + "\n"


def problems(cfg: dict, served: list[str]) -> list[str]:
    p = (cfg.get("models") or {}).get("providers", {}).get("vllm", {})
    out = []
    base = p.get("baseUrl", "")
    u = urlparse(base)
    if not base.rstrip("/").endswith("/v1"):
        out.append(f"baseUrl {base!r} must end in /v1")
    if u.hostname not in LOCAL_HOSTS:
        out.append(f"baseUrl host {u.hostname!r} is not local — the agent should talk to vLLM on the same Spark")
    key = str(p.get("apiKey", ""))
    if not key or SECRET.search(key):
        out.append("apiKey must be a non-empty placeholder (vLLM needs none) — never a real key in this file")
    if not str(p.get("api", "")).startswith("openai-"):              # playbook: openai-responses, or the
        out.append(f"api {p.get('api')!r} is not an OpenAI-style value")   # chat-completions variant of your version
    for m in p.get("models", []):
        if m.get("id") != m.get("name"):
            out.append("id and name must both be the served handle")
        if served and m.get("id") not in served:
            out.append(f"model {m.get('id')!r} is not served by the endpoint (served: {', '.join(served[:3])})")
        if int(m.get("contextWindow", 0)) < MIN_CONTEXT:
            out.append(f"contextWindow {m.get('contextWindow')} < {MIN_CONTEXT} (the playbook's minimum for agents)")
        if int(m.get("maxTokens", 0)) >= int(m.get("contextWindow", 1)):
            out.append("maxTokens must be smaller than contextWindow")
    return out


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--context", type=int, default=262_144, help="the server's --max-model-len (default: the recipe's 262144)")
ap.add_argument("--model", default="", help="override the model handle")
args = ap.parse_args()

banner("Lab 17-3 · agent config generator", "one served model → OpenClaw + Hermes configs, validated and smoke-tested")

step(1, "which model does the endpoint serve?")
base, src = resolve("vllm")
spark_served = models(base) if src == "spark" else []
model_id = args.model or (spark_served[0] if spark_served else DEFAULT_MODEL)
if src == "spark":
    note(f"Spark vLLM serves: {', '.join(spark_served) or '(nothing)'}")
else:
    note(f"no Spark vLLM reachable → using the playbooks' DGX Spark default: {model_id}")
    note("the validation below cannot check 'is it served?' without the Spark; the smoke test uses the laptop stand-in")

step(2, "generate the two configs")
oc = openclaw_models(SPARK_BASE, model_id, args.context)
hs = hermes_script(SPARK_BASE, model_id)
OUT.mkdir(exist_ok=True)
(OUT / "openclaw.models.json").write_text(json.dumps(oc, indent=2) + "\n", encoding="utf-8")
(OUT / "hermes-config.sh").write_text(hs, encoding="utf-8")
print("▣ openclaw.models.json  (merge into ~/.openclaw/openclaw.json)")
for line in json.dumps(oc, indent=2).splitlines():
    print("│ " + line)
print("\n▣ hermes-config.sh")
for line in hs.splitlines():
    print("│ " + line)
note(f"written to {OUT.relative_to(HERE.parents[1])}/")

step(3, "validate")
found = problems(json.loads((OUT / "openclaw.models.json").read_text()), spark_served)
negatives = [
    ("real-looking key", {**oc, "models": {**oc["models"], "providers": {"vllm": {
        **oc["models"]["providers"]["vllm"], "apiKey": "sk-FAKE-not-a-real-key-0000"}}}}),
    ("LAN address", openclaw_models("http://192.168.1.42:8000/v1", model_id, args.context)),
    ("8K context", openclaw_models(SPARK_BASE, model_id, 8192, 4096)),
    ("missing /v1", openclaw_models("http://localhost:8000", model_id, args.context)),
]
for p in found:
    print(f"✕ {p}")
if not found:
    print("✓ openclaw.models.json: local /v1 endpoint, placeholder key, ≥32K context")
rows = [[label, "✓ caught" if problems(c, spark_served) else "✕ missed", (problems(c, spark_served) or ["—"])[0][:60]]
        for label, c in negatives]
table(rows, ["broken config", "validator", "finding"])

step(4, "smoke test: does this (endpoint, model) pair answer?")
laptop_model = pick_laptop_model(["gemma4:12b", "nemotron-3-nano"]) if src == "laptop" else None
r = chat_any("vllm", model_id, [{"role": "user", "content": "Reply exactly HERMES_OK"}], max_tokens=256,
             laptop_model=laptop_model, reference="HERMES_OK")
show_chat(r)
answered = "HERMES_OK" in (r.get("text") or "")
if r.get("source") == "laptop":
    note(f"LAPTOP STAND-IN: answered by {laptop_model} on this laptop, not by {model_id} on the Spark. It proves the "
         "request shape; on the Spark, `hermes -z` sends this user prompt to your real model.")

step(5, "copy to the Spark (never over ~/.openclaw/openclaw.json)")
for name in ("openclaw.models.json", "hermes-config.sh"):
    put(OUT / name, f"~/w25/agents/{name}")
if where() == "dry":
    note("DRY: on the Spark, run `bash ~/w25/agents/hermes-config.sh` and merge the JSON into ~/.openclaw/openclaw.json "
         "by hand, then restart the OpenClaw gateway.")

print()
ok1 = check(not found, "configs are valid for a local vLLM", "configs have problems — see step 3")
ok2 = check(all(r_[1] == "✓ caught" for r_ in rows), "every broken config was caught", "the validator missed a case")
ok3 = check(answered or r.get("source") == "reference", f"smoke test answered ({r.get('source')})",
            "smoke test did not return HERMES_OK")
result("Same endpoint, same model handle, two agents: OpenClaw reads it from JSON, Hermes from `hermes config set`.")
sys.exit(0 if ok1 and ok2 and ok3 else 1)
