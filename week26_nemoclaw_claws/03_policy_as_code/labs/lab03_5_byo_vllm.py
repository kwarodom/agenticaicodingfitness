#!/usr/bin/env python3
"""Lab 03-5 · Raw OpenShell without NemoClaw: bring your own vLLM (L2.7).

This is the DGX Spark OpenShell playbook as ordered steps — what `nemoclaw onboard` does under the hood.
  • Read-only checks run with sh(): openshell status, the gateway service, provider list, inference get, docker ps,
    the vLLM /health and /v1/models endpoints.
  • Every change goes through clawkit.change(): the vLLM container, the provider, the inference route, the port
    forward, the cleanup. The installer and `openshell sandbox create --from openclaw` (an interactive wizard)
    are never run by the lab — you run them in the ⌨ terminal.
  • On THIS laptop, for real: the LAN-IP-not-localhost rule computed and checked, policykit's view of why the
    agent must call inference.local, and the real CLI 0.0.111 parsing the exact command lines (plus a guard
    for the one mistake the parser does not catch: `--policy` with `--from openclaw`).

Run: .venv/bin/python week26/03_policy_as_code/labs/lab03_5_byo_vllm.py
"""
import ipaddress
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import parsekit  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import banner, cfg, change, note, ok, result, sh, step, table, warn  # noqa: E402

MOD = HERE.parents[1]
SB = "openshell-demo"                                           # the playbook's SANDBOX_NAME
MODEL = cfg("MODEL_HANDLE", "nvidia/Qwen3.6-35B-A3B-NVFP4")     # the OpenShell playbook's DGX Spark model
VLLM_IMAGE, MAX_LEN = "vllm/vllm-openai:latest", 131072

REF_HELP = ("Expected output should show the `openshell` command tree with subcommands like `gateway`, "
            "`sandbox`, `provider`, and `inference`.")
REF_STATUS = "`openshell status` should report the gateway as **Connected**."
REF_MODELS = ("Expected: a JSON `\"data\"` array listing your model handle. Note the exact `id` — you will reuse "
              "it in Steps 6–7 and the OpenClaw wizard.")
REF_INFER = "Expected output should show `provider: local-vllm` and your chosen `model`."
REF_POLICY = ("Do not pass `--policy` with a local file path when using `--from openclaw`. The policy is bundled "
              "with the community sandbox; a local file path can cause \"file not found.\"")


def provider_url_problem(host: str) -> str:
    """'' if http://<host>:8000/v1 is a URL the gateway (inside Docker) can reach, else why not."""
    if host in ("localhost", "ip6-localhost"):
        return "localhost inside the gateway's container is the container itself"
    if host == "host.docker.internal":
        return "works only where it resolves — check on your unit, else use the LAN IP"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "" if host and not host.startswith("<") else "not an address yet"
    if ip.is_loopback:
        return "loopback: the gateway cannot reach host services on 127.0.0.1"
    if ip.is_unspecified:
        return "0.0.0.0 is a bind address for vLLM, not a destination"
    return ""


def create_problems(cmd: str) -> list[str]:
    """The playbook's sandbox-create gotchas, checked on a command line."""
    argv, out = shlex.split(cmd), []
    if "--from" in argv and argv[argv.index("--from") + 1] == "openclaw" and "--policy" in argv:
        out.append("--policy with --from openclaw: the policy is bundled with the community sandbox")
    if "--" not in argv:
        out.append("no `-- <command>`: the trailing command is the health-defining main process")
    return out


banner("Lab 03-5 · raw OpenShell + your own vLLM", "the OpenShell playbook as steps · read-only + change() on the "
       "Spark · the real CLI parser on this laptop")

step(1, "is OpenShell installed, and is its gateway up? (read-only)")
print("→ not installed yet? run the installer yourself in the ⌨ terminal (a lab never pipes curl into sh):")
print("    curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh")
sh("command -v openshell && openshell --help | head -20", reference=REF_HELP, timeout=60)
sh("systemctl --user status --no-pager openshell-gateway | head -5", timeout=60,
   example="● openshell-gateway.service - OpenShell gateway\n     Active: active (running) since <time>")
sh("openshell status", reference=REF_STATUS, timeout=60)
print("→ keep the gateway alive after logout (asks for sudo — run it in the ⌨ terminal):  "
      "sudo loginctl enable-linger $USER")

step(2, "serve a model with vLLM — bound to 0.0.0.0:8000")
change(f"docker run -d --name vllm-server --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 "
       f"--entrypoint \"\" -p 8000:8000 -e HF_TOKEN=\"$HF_TOKEN\" "
       f"-v \"$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub\" {VLLM_IMAGE} "
       f"vllm serve \"{MODEL}\" --max-model-len {MAX_LEN} --gpu-memory-utilization 0.8",
       preview="docker ps -a --filter name=vllm-server --format '{{.Names}} {{.Status}}'",
       example="vllm-server Up 3 minutes   (or nothing yet)")
note("HF_TOKEN must already be exported in the Spark's shell (`export HF_TOKEN=...` in the ⌨ terminal). The lab "
     "passes the literal `$HF_TOKEN`, so the token never appears on a command line the runner echoes.")
sh("curl -sf http://localhost:8000/health && echo healthy || echo 'not up yet — the first load takes minutes; "
   "re-run this lab'", example="healthy", timeout=60)
sh("curl -s http://0.0.0.0:8000/v1/models | head -c 400", reference=REF_MODELS, timeout=60)

step(3, "the provider URL — the LAN IP, never localhost")
ipres = sh("hostname -I | awk '{print $1}'", example="192.168.1.42", timeout=30)
ip = ipres.out.strip().splitlines()[0] if ipres.live and ipres.ok and ipres.out.strip() else "<spark-ip>"
rows = []
for h in ["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal", ip]:
    why = provider_url_problem(h)
    verdict = ("· placeholder — DRY" if h.startswith("<") else "⚠ " + why if "resolves" in why else
               "✕ " + why if why else "✓ reachable from the gateway's container")
    rows.append([f"http://{h}:8000/v1", verdict, "← your Spark" if h == ip else ""])
table(rows, ["OPENAI_BASE_URL", "verdict (course rule, from the playbook)", "note"])
if ip == "<spark-ip>":
    warn("DRY: `hostname -I` did not run — the last row is a placeholder, not your Spark's address")
note("Why: the gateway runs inside Docker. Inside its container, 127.0.0.1 is the container, not the Spark, so "
     "bind vLLM to 0.0.0.0 and give the provider the machine's IP.")

base = pk.load(MOD / "policies" / "anatomy.yaml")
probe_ip = ip if not ip.startswith("<") else "192.168.1.42"
rows = []
for host, port in [("inference.local", 443), (probe_ip, 8000), ("127.0.0.1", 8000)]:
    d, why = pk.decide(base, {"op": "connect", "host": host, "port": port, "binary": "/usr/local/bin/openclaw"})
    rows.append([f"{host}:{port}", {"allow": "✓ allow", "deny": "✕ deny"}.get(d, "◆ " + d), why])
table(rows, ["the AGENT (inside the sandbox) calls", "decision", "why (policykit, teaching model)"])
note("Two different callers. The GATEWAY needs the LAN IP to reach vLLM. The AGENT never calls vLLM directly: it "
     "calls https://inference.local/v1 and the gateway forwards it. So you add no network_policies entry for "
     "vLLM at all.")

step(4, "register vLLM as an OpenShell provider, then route inference.local to it")
change("IP=$(hostname -I | awk '{print $1}') && openshell provider create --name local-vllm --type openai "
       "--credential OPENAI_API_KEY=not-needed --config OPENAI_BASE_URL=http://$IP:8000/v1",
       preview="openshell provider list", example="  NAME        TYPE\n  (your providers)")
change(f"openshell inference set --provider local-vllm --model \"{MODEL}\"", preview="openshell inference get",
       reference=REF_INFER)
note("`failed to verify inference endpoint`? Warm the server up with one chat completion first; `--no-verify` "
     "only after you confirmed the host API works (the playbook's advice).")

step(5, "create the sandbox from the community OpenClaw image — in the ⌨ terminal")
GOOD = (f"openshell sandbox create --keep --tty --forward 18789 --name {SB} --from openclaw -- openclaw-start")
BAD = (f"openshell sandbox create --keep --forward 18789 --name {SB} --from openclaw --policy ./my-policy.yaml "
       "-- openclaw-start")
print(f"→ {GOOD}")
print("  the OpenClaw wizard is fully interactive (arrow keys + Enter): Custom Provider · base URL "
      f"https://inference.local/v1 · any non-empty key (\"not-needed\") · OpenAI-compatible · model {MODEL}")
rows = []
for label, cmd in [("the playbook's command", GOOD), ("with --policy added", BAD)]:
    probs = create_problems(cmd)
    argv = shlex.split(cmd)[1:]
    argv = [a for a in argv if a not in ("--tty",)]
    if "--forward" in argv:                                 # the CLI checks the local port before the gateway
        i = argv.index("--forward")
        del argv[i:i + 2]
    v, msg = parsekit.cli(argv)
    rows.append([label, parsekit.glyph(v, "parses" if v else msg, 30), "✕ " + probs[0] if probs else "✓ ok"])
v, msg = parsekit.cli(["sandbox", "create", "--kep", "--name", SB, "--from", "openclaw"])
rows.append(["a typo: --kep", parsekit.glyph(v, msg, 30), "—"])
table(rows, ["sandbox create …", "CLI 0.0.111 (laptop)", "course guard"])
print(f"◆ the playbook: {REF_POLICY}")
note("The laptop parser accepts `--policy` next to `--from openclaw` — this is a runtime failure, not a syntax "
     "error, which is why the guard exists. `--keep` parses although 0.0.111's --help lists only `--no-keep`. "
     "(`--tty` and `--forward` were left out of the probe: `--forward` checks the local port first.)")
ok("captured on this Mac: the real OpenShell 0.0.111 parser, no gateway")

step(6, "reach the dashboard from your workstation, then verify isolation")
change(f"openshell forward start --background 18789 {SB}", preview="openshell forward list",
       example="  (no forwards yet — after the change: openshell-demo  18789)")
print("→ in the ⌨ terminal: `openshell term` — watch allow / deny / inspect_for_inference while the agent works")
print(f"→ files: `openshell sandbox upload {SB} ./local-file /sandbox/destination` · "
      f"`openshell sandbox download {SB} /sandbox/file ./local-destination`")

step(7, "clean up — gateway-dependent steps first, while the gateway still answers")
change(f"openshell sandbox delete {SB} && openshell provider delete local-vllm",
       preview="openshell sandbox list", example=f"  NAME            PHASE\n  {SB}  Ready")
change("docker rm -f vllm-server 2>/dev/null || true",
       preview="docker ps -a --filter name=vllm-server --format '{{.Names}} {{.Status}}'",
       example="vllm-server Up 20 minutes")
result("You wired a sandbox to your own vLLM by hand: provider on the LAN IP, inference.local routed, no provider "
       "host in the policy, and no --policy next to --from openclaw.")
