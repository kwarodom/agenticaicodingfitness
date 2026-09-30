#!/usr/bin/env python3
"""Lab 02-4 · Prove inference is local: the route, the models, one hello, and a call that must fail.

The Spark half (read-only): `openshell inference get` must name a LOCAL provider (ollama / local-vllm / vllm);
`openshell sandbox exec -n <s> -- curl -s https://inference.local/v1/models` must list a model from inside the
sandbox; one short chat completion goes through inference.local; and `curl https://api.openai.com/v1/models` from
inside the sandbox must NOT get through (Part 1, exercise 3). Then the sandbox log is asked where the denial shows.

The laptop half (for real): the OpenShell CLI parses every command above; the course's policykit model explains
each decision (inference.local is intercepted, api.openai.com has no network_policies entry, the Spark's own IP is
private, the metadata IP is always blocked); and a LAPTOP STAND-IN shows what a /v1/models answer looks like.

Pass criterion (Reef spec §6, L1.5): the provider is local AND the models list comes back. DRY never passes.

Run: .venv/bin/python week26/02_first_claw/labs/lab02_4_inference_is_local.py
"""
import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
from clawkit import (LAPTOP_OLLAMA, banner, http_json, note, ok, openshell_offline, parsed_ok, result,  # noqa: E402
                     sandbox, sh, step, table, warn)

S = sandbox()
INFERENCE_REF = "Expected output should show `provider: local-vllm` and your chosen `model`."
LOCAL_HINTS = ("ollama", "vllm")                     # local-vllm, vllm, install-vllm, ollama …
CLOUD_HINTS = ("build", "nvidia", "openai", "anthropic", "openrouter", "gemini", "routed")


def classify(provider: str) -> str:
    p = provider.lower()
    if any(h in p for h in LOCAL_HINTS):
        return "local"
    if any(h in p for h in CLOUD_HINTS):
        return "cloud"
    return "unknown"


def model_ids(text: str) -> list[str]:
    try:
        d = json.loads(text[text.index("{"):])
        return [m.get("id", "") for m in d.get("data", []) if m.get("id")]
    except (ValueError, AttributeError):
        return []


banner("Lab 02-4 · prove inference is local", f"sandbox `{S}` · read-only · the deny test is expected to fail")

step(1, "which provider does inference.local route to? (on the Spark host)")
ig = sh("openshell inference get", timeout=30, reference=INFERENCE_REF)
prov = (re.search(r"provider:\s*(\S+)", ig.out, re.I) or [None, ""])[1] if ig.live else ""
sh("openshell provider list", timeout=30, example="""NAME         TYPE     BASE URL
<provider>   openai   http://<spark-ip>:8000/v1        ← EXAMPLE shape""")
note("A provider's NAME is only a label. `provider list` shows where it points: local means this Spark's own IP "
     "(not localhost — the gateway runs in Docker).")

step(2, "from INSIDE the sandbox: list the models behind inference.local")
EX_MODELS = '{"object":"list","data":[{"id":"<MODEL_HANDLE>","object":"model","owned_by":"<server>"}]}'
ms = sh(f"openshell sandbox exec -n {S} -- curl -s https://inference.local/v1/models", timeout=60,
        example=EX_MODELS + "        ← EXAMPLE shape")
ids = model_ids(ms.out) if ms.live else []

step(3, "one short hello through inference.local (inside the sandbox)")
model = ids[0] if ids else "<MODEL_HANDLE>"
body = json.dumps({"model": model, "max_tokens": 32,
                   "messages": [{"role": "user", "content": "Say hello from the Spark."}]})
cmd = (f"openshell sandbox exec -n {S} -- curl -s https://inference.local/v1/chat/completions "
       f"-H 'Content-Type: application/json' -d {shlex.quote(body)}")
if ms.live and not ids:
    warn("no model id came back in step 2 — skipping the hello (fix the route first)")
else:
    sh(cmd, timeout=120, example='{"choices":[{"message":{"role":"assistant","content":"<the answer>"}}], …}'
                                 "        ← EXAMPLE shape")

step(4, "the call that must fail: api.openai.com from inside the sandbox (Part 1, exercise 3)")
deny = sh(f"openshell sandbox exec -n {S} -- curl -sS -o /dev/null -w '%{{http_code}}\\n' --max-time 15 "
          "https://api.openai.com/v1/models", timeout=60,
          example="curl: (<n>) <the proxy refused the CONNECT>\n000        ← EXAMPLE shape — no HTTP answer from OpenAI")
code = (re.findall(r"^(\d{3})\s*$", deny.out, re.M) or ["?"])[-1] if deny.live else "?"
reached = code in ("200", "401")                     # 401 = OpenAI itself answered "no key" → egress is OPEN
sh(f"openshell logs {S} --source sandbox -n 50 --since 5m", timeout=60,
   example="<time> <level> sandbox … deny … api.openai.com:443 …        ← EXAMPLE shape — look for your host")
note("Where you see it: `openshell term` on the host (live, `f` follow · `s` source · `q` quit) and "
     f"`openshell logs {S} --source sandbox`. The fix is NOT to add api.openai.com to the policy.")

step(5, "the laptop half — why each call went the way it did (real, offline)")
rows = []
for label, args in (("step 1 · inference get", ["inference", "get"]),
                    ("step 2 · exec … inference.local/v1/models",
                     ["sandbox", "exec", "-n", S, "--", "curl", "-s", "https://inference.local/v1/models"]),
                    ("step 4 · exec … api.openai.com/v1/models",
                     ["sandbox", "exec", "-n", S, "--", "curl", "https://api.openai.com/v1/models"]),
                    ("step 4 · logs --source sandbox -n 50", ["logs", S, "--source", "sandbox", "-n", "50", "--since", "5m"])):
    r = openshell_offline(args)
    rows.append([label, "✓ parsed · needs the gateway" if parsed_ok(r) else f"✕ exit {r.code}"])
table(rows, ["command", "laptop CLI 0.0.111"])

POLICY = """
version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /proc, /app, /etc, /var/log]
  read_write: [/sandbox, /tmp]
landlock:
  compatibility: best_effort
process:
  run_as_user: sandbox
  run_as_group: sandbox
network_policies: {}
"""
policy = pk.load(POLICY)
errs, _ = pk.validate(policy)
CURL, PY = "/usr/bin/curl", "/usr/bin/python3"
CASES = [
    {"op": "http", "host": "inference.local", "port": 443, "binary": CURL, "method": "GET", "path": "/v1/models"},
    {"op": "http", "host": "api.openai.com", "port": 443, "binary": CURL, "method": "GET", "path": "/v1/models"},
    {"op": "connect", "host": "integrate.api.nvidia.com", "port": 443, "binary": PY},
    {"op": "connect", "host": "192.168.1.42", "port": 8000, "binary": PY},
    {"op": "connect", "host": "169.254.169.254", "port": 80, "binary": CURL},
]
for e in errs:
    print(f"✕ {e}")
for a in CASES:
    decision, why = pk.decide(policy, a)
    glyph = {"allow": "✓ allow", "deny": "✕ deny", "inspect_for_inference": "◆ inspect_for_inference"}[decision]
    print(f"│ {pk.fmt_action(a):42s} {glyph}\n│ {'':42s}   {why}")
note("policykit is the course's TEACHING MODEL, not OpenShell. The policy above is a stand-in for the Restricted "
     "tier (no presets); read your real one with `nemoclaw <s> policy get` in Module 03. 192.168.1.42 is the "
     "playbook's example Spark IP: even the local vLLM is reachable only through inference.local.")
bad = pk.load(POLICY.replace("network_policies: {}", """network_policies:
  openai:
    endpoints: [{host: api.openai.com, port: 443, protocol: rest, enforcement: enforce, access: read-only}]
    binaries: [{path: /usr/bin/curl}]"""))
for sev, where_, finding in pk.harden(bad):
    if sev == "HIGH":
        print(f"✕ {sev} {where_}: {finding}")
note("That is the tempting 'fix' for step 4 — and the checklist refuses it: inference goes through inference.local.")

try:
    d = http_json("GET", LAPTOP_OLLAMA + "/models", timeout=3)
    local_ids = [m["id"] for m in d.get("data", []) if "cloud" not in m.get("id", "")]
    print(f"→ GET {LAPTOP_OLLAMA}/models · Ollama on THIS laptop (LAPTOP STAND-IN, not the Spark)")
    print(f"◆ LAPTOP STAND-IN · object={d.get('object')} · {len(local_ids)} local models · first: "
          f"{', '.join(local_ids[:3])}")
    note("Same OpenAI-compatible shape the sandbox gets from inference.local: {\"object\":\"list\",\"data\":[{\"id\":…}]}")
except Exception as e:  # noqa: BLE001
    note(f"no laptop Ollama answered ({type(e).__name__}) — skip the stand-in")

step(6, "the verdict (L1.5)")
if ig.live:
    kind = classify(prov) if prov else "unknown"
    denied = code in ("000", "403")                  # curl ran inside, and no HTTP answer came from OpenAI
    deny_v = "✕ REACHED OpenAI" if reached else ("✓" if denied else f"⚠ inconclusive (exit {deny.code})")
    table([
        ["provider is local", prov or "no `provider:` line", {"local": "✓", "cloud": "✕ cloud", "unknown": "⚠ check"}[kind]],
        ["models list from inside", ", ".join(ids[:3]) or "none", "✓" if ids else "✕"],
        ["api.openai.com blocked", f"http_code {code}, exit {deny.code}", deny_v],
    ], ["proof", "seen", "verdict"])
    if kind == "local" and ids:
        ok("L1.5 passed: inference.local routes to a local provider and answers from inside the sandbox")
    else:
        print("✕ L1.5 not passed — see the ✕ rows above")
else:
    table([[p, "◈ DRY — not proven"] for p in ("provider is local (ollama / local-vllm / vllm)",
                                             "models list returned from inside the sandbox",
                                             "api.openai.com denied from inside the sandbox")], ["proof", "verdict"])
    warn("DRY: the laptop half is real; the Spark half is not your machine. Connect a Spark and run it LIVE.")
result("Local route + models from inside + the cloud call denied = prompts and data stay on the Spark.")
