# NemoClaw on DGX Spark — Beginner to Expert

## Building "claws" with OpenShell sandboxing, NeMo Agent Toolkit (NAT), tracing and performance benchmarking

Prepared for Warodom Khamphanchai / AltoTech — 29 September 2026

---

## How to use this tutorial

This is a hands-on, six-part course. Each part has a concept section, guided labs with exact commands, a set of exercises, and worked solutions. It is designed to be run on a single NVIDIA DGX Spark (GB10, 128 GB unified memory) but almost everything also works on a Linux x86 workstation with an NVIDIA GPU, and Parts 3–5 can be done on any Linux machine with Docker.

| Part | Level | What you will be able to do afterwards |
|---|---|---|
| 0 | Orientation | Explain what a "claw" is, what NemoClaw, OpenShell, OpenClaw, Hermes, Deep Agents and NAT each do, and why the sandbox matters |
| 1 | Beginner | Prepare a DGX Spark, install NemoClaw with one command, onboard a sandboxed assistant on a local Nemotron model, chat via Web UI / TUI / Telegram |
| 2 | Intermediate | Read and write OpenShell policy YAML, apply and audit NemoClaw presets, approve egress in the TUI, snapshot and rebuild sandboxes, point a sandbox at your own vLLM |
| 3 | Intermediate–Advanced | Install NAT on the Spark, write YAML workflows and custom Python tools, run ReAct/tool-calling agents against local vLLM, expose them over MCP, consume MCP tools, run NAT inside an OpenShell sandbox |
| 4 | Advanced | Trace agent runs with Phoenix, OTel collector, Langfuse (Hermes) and OpenShell's own policy/inference logs |
| 5 | Advanced | Benchmark model throughput on the Spark, profile NAT workflows, evaluate accuracy with RAGAS/trajectory evaluators, and size GPU capacity with the sizing calculator |
| 6 | Expert | Threat-model a claw, harden it against real OpenClaw CVE classes, write L7/MCP-aware policies, build custom blueprints, run remote/multi-node gateways, and map the whole thing onto AltoTech's sovereign tier |

### Two ways to run the labs

Every lab can be run from a terminal exactly as written. The companion document "Alto Reef — Web Lab Runner" specifies a browser-based runner (a reef-style sandbox map, a Build-a-Claw profile builder, a per-sandbox Workbench with Run / Policies / Chat / Console / Traces / Bench tabs, and a Lab Mode overlay) that executes the same commands and auto-checks the evidence. Each lab in this tutorial carries a runner id in its heading (for example `L2.3`); Section 6 of the companion maps those ids to the runner's actions and pass criteria. The runner must always display the commands it is about to execute, so the terminal path and the web path teach the same thing.

### Status warnings you must keep in mind

- NemoClaw is an alpha, Apache-2.0 open-source reference stack; the GitHub repository is explicit about alpha status and routes security reports to psirt@nvidia.com ([NVIDIA/NemoClaw on GitHub](https://github.com/NVIDIA/NemoClaw)).
- NVIDIA's own Build-a-Claw page warns that always-on agents can have broad system access and expose data, and recommends running them in a clean environment ([NVIDIA Build a Claw](https://www.nvidia.com/en-us/ai/build-a-claw/)).
- The DGX Spark playbook says to run the demo on a fresh device or VM, not one containing personal or confidential data ([DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview)).
- OpenShell is moving fast: the repository's latest release at time of writing is v0.1.2 (28 Sep 2026), while the NemoClaw-managed OpenShell version is pinned at 0.0.116 ([OpenShell on GitHub](https://github.com/NVIDIA/openshell), [NemoClaw architecture](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/architecture)). Commands in this tutorial are taken from the current docs; when a flag differs on your machine, `--help` wins.
- All throughput numbers quoted in Part 5 come from third-party write-ups and community forum posts. They are useful for orientation, not for procurement decisions, and they should be reproduced on your own unit.

---

## Part 0 — Orientation: what is a "claw"?

### 0.1 The vocabulary

NVIDIA's Build-a-Claw hub describes NemoClaw as an open-source reference stack that deploys with a single command and bundles an agent harness (Hermes, LangChain Deep Agents or OpenClaw), the NVIDIA OpenShell secure runtime, and Nemotron models; it installs on DGX Spark and OEM GB10 systems and Jetson Orin Nano, and can also be tried for free in the cloud via Brev ([NVIDIA Build a Claw](https://www.nvidia.com/en-us/ai/build-a-claw/)).

So a "claw" is: **an always-on, tool-using agent (the harness) + a local or routed model (Nemotron by default) + a kernel-enforced sandbox and policy boundary (OpenShell), assembled by an installer and CLI (NemoClaw).**

| Component | Role | Where it runs |
|---|---|---|
| **NemoClaw CLI** (`nemoclaw`, `nemohermes`, `nemo-deepagents`) | Host-side installer and lifecycle tool: onboard, connect, status, logs, policy add/remove, snapshot, rebuild, inference set | Host |
| **OpenShell gateway** | Control plane: sandbox lifecycle, credentials, policy revisions, inference routes; runs as a Docker/Podman-driven service (default port 8080, auto range 8990–9005) | Host |
| **OpenShell sandbox + supervisor** | Data plane: Landlock filesystem isolation, seccomp, network namespace with a policy proxy, credential injection, `inference.local` interception | Container |
| **Harness** | OpenClaw (default), Hermes, or LangChain Deep Agents Code — the actual agent loop, tools, channels, dashboard | Inside sandbox |
| **Blueprint** | Versioned YAML describing image, agent manifest, network policy and inference profile, digest-verified before apply | Repo (`nemoclaw-blueprint/`) |
| **Inference provider** | NVIDIA Endpoints, OpenAI, Anthropic, Gemini, OpenRouter, local Ollama, existing or managed vLLM, Model Router | Host or cloud |
| **NeMo Agent Toolkit (NAT)** | Framework-agnostic toolkit for building, profiling, evaluating and serving agent workflows (YAML + Python); MCP client/server | Anywhere — in this course, on the Spark host and inside a sandbox |

The how-it-works page describes the flow: the host CLI talks to the OpenShell gateway, which manages the sandbox; agents call `https://inference.local`, the gateway injects the real credential and forwards to the configured provider; policies are layered (network and inference are hot-reloadable, filesystem and process are locked at creation); blocked egress surfaces in a TUI for approval, and literal credentials in policy files are refused ([NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works)).

### 0.2 Why the sandbox is the whole point

OpenClaw, the default harness, has had a rough security year. Cyera disclosed four vulnerabilities in September 2026 — a TOCTOU filesystem write escape (CVE-2026-44112, CVSS 9.6), an execution-allowlist environment-variable disclosure (CVE-2026-44115, 8.8), an MCP loopback privilege escalation (CVE-2026-44118, 7.8) and a TOCTOU read escape (CVE-2026-44113, 7.7), three of which are exploitable from a single prompt-injection foothold ([Cyera research](https://www.cyera.com/research/four-new-openclaw-vulnerabilities-when-ai-agents-become-the-attackers-execution-layer)). The Cloud Security Alliance's "Claw Chain" note explains how these chain into full agent compromise ([CSA research note](https://labs.cloudsecurityalliance.org/research/csa-research-note-openclaw-claw-chain-cve-20260517-csa-style/)).

NemoClaw's answer is defence in depth outside the harness: five deny-by-default layers (network, filesystem, process, gateway authentication, inference), with the OpenShell policy and credential providers treated as the enforcement boundary and the agent's own mutable config (`/sandbox/.openclaw`, `/sandbox/.hermes`, `/sandbox/.deepagents`) explicitly not trusted as isolation ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).

Keep this mental model for the whole course: **the harness is the untrusted thing you are containing; OpenShell is what you are actually configuring; NAT is how you build the agent logic you want to run inside.**

### 0.3 The three harnesses at a glance

| Harness | Default model on NVIDIA Endpoints | State dir | CLI alias | Notes |
|---|---|---|---|---|
| OpenClaw | `nvidia/nemotron-3-super-120b-a12b` (shared default with Hermes) | `/sandbox/.openclaw` | `nemoclaw` | Web dashboard on 18789, `openclaw tui`, Telegram/Discord/Slack channels, Brave or Tavily search |
| Hermes | Nemotron 3 Super | `/sandbox/.hermes` | `nemohermes` | Dashboard 18789, OpenAI-compatible API on 8642, Tavily only, Langfuse plugin |
| LangChain Deep Agents Code | `Nemotron 3 Ultra` | `/sandbox/.deepagents` | `nemo-deepagents` | Planner that spawns sub-agents; coding-oriented |

Model defaults and the model-fit table are documented on the choose-a-model page ([NemoClaw choose a model](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/inference/learn-and-choose/choose-model)); Hermes ports and state paths come from the Hermes quickstart ([NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart)); Deep Agents commands from its quickstart ([NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart)).

### 0.4 Lab environment used throughout

- DGX Spark, DGX OS (Ubuntu 24.04 arm64), Docker 28.x preinstalled, GB10 GPU. The spark-install note says no pre-setup is needed on Spark and the standard quickstart applies ([spark-install.md](https://github.com/NVIDIA/NemoClaw/blob/main/spark-install.md)).
- A second laptop on the same LAN for remote dashboard access (optional).
- Running example: **"Alto Ops Claw"** — a hotel-operations assistant that reads chiller-plant CSV exports, answers energy questions, and calls a mock BMS MCP server. We build it in Part 3, trace it in Part 4, benchmark it in Part 5 and harden it in Part 6.

### Exercises — Part 0

1. In one paragraph, explain to a hotel GM why "the agent runs in a sandbox" is different from "the agent is safe".
2. List which of the four OpenShell/NemoClaw policy layers are hot-reloadable and which require sandbox recreation.
3. Which harness would you pick for (a) a Telegram concierge bot, (b) a code-refactoring agent on a private repo, (c) an agent whose traces must land in Langfuse with zero raw keys inside the sandbox?

**Solutions**

1. Model answer: the sandbox limits what damage a compromised or confused agent can do (which files, hosts, syscalls and credentials it can touch) — it does not make the agent's reasoning correct or immune to prompt injection. NemoClaw's docs state that mutable agent configuration can change independently of host policy and that OpenShell policy and credential providers are the enforcement boundary ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).
2. Network and inference are hot-reloadable; filesystem and process are locked at sandbox creation ([DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview)).
3. (a) OpenClaw — it has the Telegram channel flow (`nemoclaw <sandbox> channels add telegram`); (b) LangChain Deep Agents Code — planner/sub-agent coding harness with a Nemotron 3 Ultra default; (c) Hermes — the `langfuse-hermes-v1` credential type keeps Langfuse keys as OpenShell placeholders (Part 4).

---

## Part 1 — Beginner: your first claw on DGX Spark

### 1.1 Concepts

**Express install vs interactive onboarding.** The installer (`nemoclaw.sh`) pulls Node.js, OpenShell and the NemoClaw CLI as needed; the `onboard` wizard (or Express Install when offered) creates a sandboxed agent, optional Brave Search, optional Telegram/Discord/Slack channels and a policy tier with network presets ([DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview)). On supported platforms the installer asks `Run express install with these settings? [Y/n]:`; answering `n` lets you pick Hermes or Deep Agents, a sandbox name, a provider and a model ([NVIDIA/NemoClaw on GitHub](https://github.com/NVIDIA/NemoClaw)).

**Local inference on Spark.** The Spark playbook routes inference to local vLLM on the device, and the quickstart notes that DGX Spark can automatically select local vLLM when `NEMOCLAW_PROVIDER` is omitted; managed vLLM is Docker-backed and requires a large download ([NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart)). NVIDIA's earlier Spark walkthrough used local Ollama with `nemotron-3-super:120b` instead ([NVIDIA Technical Blog — NemoClaw + OpenClaw on Spark](https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/)), and the June 2026 follow-up shows Express auto-downloading Qwen3.6-35B via Ollama ([NVIDIA Technical Blog — faster models and multi-node](https://developer.nvidia.com/blog/run-local-ai-agents-with-faster-models-and-multi-node-clustering-on-nvidia-dgx-spark/)). Either path keeps prompts and data on the device — the provider trust table lists local Ollama as "no data leaves the machine" ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).

**Policy tiers.** During onboarding you choose a tier: `Restricted` (baseline only), `Balanced` (default — `npm`, `pypi`, `huggingface`, `brew`, `brave`, plus Tavily when selected), `Open` (adds messaging, `jira`, `outlook`, `weather`, `public-reference`), or `Personal` (mandatory `personal-open-internet`, any binary may reach ports 80/443 at L4). Non-interactive runs set `NEMOCLAW_POLICY_TIER` ([NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies)).

### 1.2 Lab 1.1 (L1.1) — verify the Spark

```bash
head -n 2 /etc/os-release           # Ubuntu 24.04 / DGX OS
nvidia-smi                          # NVIDIA GB10
docker info --format '{{.ServerVersion}}'   # 28.x+
docker ps                           # if "permission denied": see below
```

If Docker refuses, add yourself to the group and make sure the NVIDIA runtime is configured:

```bash
sudo usermod -aG docker $USER && newgrp docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

These verification and Docker steps are the ones in the Spark playbook and the NVIDIA blog ([DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview), [NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/)).

Also make sure you have plenty of free disk: large Express models can require hundreds of GB for weights plus the vLLM container ([DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview)).

### 1.3 Lab 1.2 (L1.2) — one-command install (interactive)

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

What happens, in order:

1. A third-party software notice (accept with `--yes-i-accept-third-party-software` or `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` — placed on the `bash` side of the pipe, never before `curl`).
2. Node.js, OpenShell and the NemoClaw CLI are installed.
3. `nemoclaw onboard` starts automatically when preflight passes. If the installer prints `To finish setup, run:`, run the shown `nemoclaw onboard` yourself before connecting.
4. On Spark you are offered Express Install. Take it the first time.

All of these behaviours are documented in the quickstart ([NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart)).

Important rule from the same page: the sandbox only exists after `nemoclaw onboard` completes — do not run `launch`, `connect` or `openclaw tui` before that.

### 1.4 Lab 1.3 (L1.3) — scripted (non-interactive) install variants

Reproducible installs matter once you have more than one Spark. The quickstart documents a fully non-interactive first run against NVIDIA Endpoints:

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | \
  NEMOCLAW_NON_INTERACTIVE=1 \
  NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
  NEMOCLAW_AGENT=openclaw \
  NEMOCLAW_PROVIDER=build \
  NVIDIA_INFERENCE_API_KEY=<your-key> \
  NEMOCLAW_SANDBOX_NAME=my-gpt-claw \
  bash
```

Provider values you can substitute for `NEMOCLAW_PROVIDER`: `build` (NVIDIA Endpoints), `openrouter`, `openai`, `anthropic`, `gemini`, `routed` (Model Router), `custom` (any `/v1/chat/completions` endpoint with `COMPATIBLE_API_KEY`), `anthropicCompatible`, `ollama` (optional `NEMOCLAW_MODEL`), `vllm` (an already-running server on `localhost:${NEMOCLAW_VLLM_PORT:-8000}`), `install-vllm` (managed, Docker-backed vLLM), and `hermes-provider` (Hermes only) ([NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart)).

Pin a release when you need repeatability:

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | NEMOCLAW_INSTALL_REF= NEMOCLAW_INSTALL_TAG=vX.Y.Z bash
```

Other useful switches: `NEMOCLAW_AGENT=hermes|langchain-deepagents-code`, `NEMOCLAW_NO_EXPRESS=1` to force the full wizard, `NEMOCLAW_WEB_SEARCH_PROVIDER=tavily|none`, `NEMOCLAW_GATEWAY_RUNTIME=podman`, `--defer-onboarding` to install the CLI without creating a provider or sandbox ([NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart), [NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart)).

### 1.5 Lab 1.4 (L1.4) — the lifecycle commands you will use every day

```bash
nemoclaw my-assistant status                 # sandbox phase, provider, model
nemoclaw launch my-assistant                 # full preflight, then openclaw tui
nemoclaw my-assistant connect                # shell inside the sandbox
nemoclaw my-assistant logs --follow          # agent + supervisor logs
nemoclaw my-assistant dashboard-url --quiet  # tokenised Web UI URL
nemoclaw my-assistant gateway-token --quiet  # token only
nemoclaw inference set --model <model> --provider <provider> --sandbox my-assistant
nemoclaw my-assistant policy list
nemoclaw my-assistant policy add <preset> --dry-run
nemoclaw my-assistant snapshot create --name before-change
nemoclaw my-assistant rebuild
nemoclaw onboard --recreate-sandbox          # when changing web search etc.
nemoclaw onboard --fresh --gpu               # destroy + recreate, pick a new model
nemoclaw upgrade-sandboxes --auto
nemoclaw credentials reset <PROVIDER> && nemoclaw onboard
```

Sources: quickstart command list ([NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart)), `--fresh --gpu` and `gateway-token` ([NVIDIA Technical Blog — faster models](https://developer.nvidia.com/blog/run-local-ai-agents-with-faster-models-and-multi-node-clustering-on-nvidia-dgx-spark/)), policy/snapshot/rebuild verbs ([NemoClaw integration policy examples](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/integration-policy-examples), [NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart)).

The dashboard is on `http://127.0.0.1:18789/#token=<token>`; if you are on a remote laptop, forward it from the Spark:

```bash
openshell forward start 18789 my-assistant --background
```

([NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/)).

### 1.6 Lab 1.5 (L1.5) — first conversation, and proving inference is local

Inside the sandbox (`nemoclaw my-assistant connect`), prove the route with `curl`:

```bash
curl https://inference.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<MODEL_HANDLE>", "messages": [{"role":"user","content":"Say hello from the Spark."}]}'
```

This is the exact test the OpenShell playbook uses; the agent never sees a provider key because the supervisor intercepts `inference.local` and the gateway injects credentials ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions), [OpenShell how it works](https://docs.nvidia.com/openshell/about/how-it-works)).

On the host, open the TUI in a second terminal and watch:

```bash
openshell term      # f = follow, s = filter by source, q = quit
```

You should see `inspect_for_inference` decisions for the model calls and `deny` for anything unlisted ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions)).

Non-interactive smoke test from inside the sandbox (from the NVIDIA blog):

```bash
openclaw agent --agent main --local -m "hello" --session-id test
```

### 1.7 Lab 1.6 (L1.6) — add a Telegram channel

1. Create a bot with `@BotFather` → `/newbot`, copy the token.
2. Either paste it during onboarding, or later:

```bash
nemoclaw my-assistant channels add telegram
```

3. Approve the pairing code the bot sends you, inside the sandbox: `openclaw pairing approve telegram <CODE>`.

Telegram setup via `channels add telegram` is in the Spark playbook ([DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview)); the pairing approval command appears in the NVIDIA blog walkthrough ([NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/)). The `telegram` preset only opens the Telegram Bot API, but note the documented risk: the agent can then message any chat accessible to the bot token ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).

### 1.8 Lab 1.7 (L1.7) — Hermes and Deep Agents variants

```bash
# Hermes, sandbox named my-hermes
curl -fsSL https://www.nvidia.com/nemoclaw.sh | NEMOCLAW_AGENT=hermes NEMOCLAW_SANDBOX_NAME=my-hermes bash
nemohermes my-hermes status
nemohermes my-hermes connect

# LangChain Deep Agents Code
curl -fsSL https://www.nvidia.com/nemoclaw.sh | NEMOCLAW_AGENT=langchain-deepagents-code NEMOCLAW_SANDBOX_NAME=my-deepagents bash
nemo-deepagents my-deepagents status
nemo-deepagents my-deepagents connect
```

([NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart), [NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart)).

### Exercises — Part 1

1. Install with Express, then answer: which provider and model did Express choose on your Spark, and which policy tier? (Use `nemoclaw <sandbox> status` and `policy list`.)
2. Write a non-interactive install line for a Hermes claw named `alto-hermes` that uses an already-running vLLM on port 8000, Tavily search, and the `restricted` tier.
3. From inside the sandbox, try `curl https://api.openai.com/v1/models`. What happens, and where do you see it?
4. Change the model without destroying the sandbox. Then change it in a way that does destroy the sandbox. Explain the difference.
5. Explain why the docs say to add `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` on the `bash` side of the pipe.

**Solutions**

1. Your answer depends on the unit; expect a local vLLM (or Ollama) provider with a Nemotron or Qwen3.6 model and the `Balanced` tier plus `openclaw-pricing`, because OpenClaw onboarding adds that preset on top of Balanced and Open defaults ([NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies)).
2.
   ```bash
   curl -fsSL https://www.nvidia.com/nemoclaw.sh | \
     NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
     NEMOCLAW_AGENT=hermes NEMOCLAW_SANDBOX_NAME=alto-hermes \
     NEMOCLAW_PROVIDER=vllm NEMOCLAW_VLLM_PORT=8000 \
     NEMOCLAW_WEB_SEARCH_PROVIDER=tavily TAVILY_API_KEY=<key> \
     NEMOCLAW_POLICY_TIER=restricted bash
   ```
   Each variable is documented individually (provider table and `NEMOCLAW_VLLM_PORT` in the quickstart, Tavily variables in the Hermes quickstart, `NEMOCLAW_POLICY_TIER` in the network-policies reference). Verify the combination on your unit; the docs do not show this exact combination.
3. The connection is denied: no `network_policies` entry matches, so the CONNECT proxy denies; the denial shows in `openshell term` and in `openshell logs <name> --source sandbox`. The docs explicitly say never to add `api.openai.com` or `api.anthropic.com` to policy — inference must go through `inference.local` ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices), [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
4. `nemoclaw inference set --model <m> --provider <p> --sandbox <s>` changes the inference route, which is a dynamic control; `nemoclaw onboard --fresh --gpu` destroys and recreates the sandbox to pick a new model in the wizard. Inference routing is hot-reloadable; the sandbox image/blueprint is not ([OpenShell how it works](https://docs.nvidia.com/openshell/about/how-it-works), [NVIDIA Technical Blog — faster models](https://developer.nvidia.com/blog/run-local-ai-agents-with-faster-models-and-multi-node-clustering-on-nvidia-dgx-spark/)).
5. Because the variable must be in the environment of the `bash` process that runs the installer script, not the `curl` process that merely downloads it; the quickstart states this directly ([NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart)).

---
## Part 2 — Intermediate: OpenShell sandboxes and policy as code

### 2.1 Concepts

**Two enforcement points.** OpenShell applies static controls (filesystem via Landlock LSM, process via seccomp BPF and privilege drop) that are locked at sandbox creation, and dynamic controls (network via a CONNECT proxy plus an OPA policy engine, provider credentials via proxy substitution) that you can change on a running sandbox with `openshell policy update` or `openshell policy set` ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

**Deny by default, bound to binaries.** Every outbound connection goes through the proxy; if no `network_policies` entry matches host, port and calling binary, it is denied. Every entry requires a `binaries` list, and OpenShell pins each authorised executable path to its first-observed SHA256 (trust on first use), failing closed on mismatch ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

**Network namespace, not environment variables.** The sandbox lives in its own Linux netns with a veth pair; all traffic routes through the host-side veth IP `10.200.0.1` where the proxy listens, so a process that ignores `HTTP_PROXY` can still only reach the proxy ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

**L4 vs L7.** Endpoints without a `protocol` field are checked only on host/port/binary (the proxy still terminates TLS and requires the request authority to match). Adding `protocol: rest|websocket|graphql|mcp|json-rpc|tcp` turns on request-level inspection, paired with `rules`/`deny_rules` or an `access` preset (`full`, `read-only`, `read-write`). `enforcement` defaults to `audit` (log violations, forward traffic) and can be set to `enforce` (403 with a JSON body) ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

**SSRF protection.** Loopback `127.0.0.0/8`, link-local `169.254.0.0/16` and `0.0.0.0` are always blocked and cannot be overridden even by `allowed_ips`; private RFC 1918 ranges are blocked for wildcard/hostless entries unless declared as exact hosts or via a narrow `allowed_ips` CIDR ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

**Filesystem baseline.** The NemoClaw OpenClaw baseline grants read-write to `/sandbox`, `/tmp`, `/dev/null`, `/dev/pts` and read-only to `/usr`, `/lib`, `/proc`, `/dev/urandom`, `/app`, `/etc`, `/var/log`, `/var/lib/dpkg`; the process runs as a dedicated `sandbox` user ([NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies)). OpenShell's Landlock baseline requires ABI 3 (Linux 6.2+) and by default uses `compatibility: best_effort`, emitting a High-severity OCSF `DetectionFinding` when a policy cannot be fully applied; `hard_requirement` refuses to start instead ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

### 2.2 Lab 2.1 (L2.1) — read the policy that NemoClaw created for you

```bash
nemoclaw my-assistant policy list
nemoclaw my-assistant policy get > current-policy.yaml      # strips metadata, replaces literal creds with [STRIPPED_BY_MIGRATION]
openshell policy get <sandbox> --base > base.yaml            # OpenShell's view of the base policy
openshell policy get <sandbox> --full                        # effective policy incl. provider-composed entries
openshell policy list <sandbox>                              # revisions
```

The NemoClaw export behaviour (`[STRIPPED_BY_MIGRATION]` markers, requires OpenShell 0.0.72+) is documented in the network-policies reference ([NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies)); the `--base/--full/--rev` flags are in the OpenShell sandbox-policy guide ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).

You should recognise the baseline entries the docs describe: `nvidia` (`integrate.api.nvidia.com:443`, binary `/usr/local/bin/openclaw`, POST to inference/embedding paths, GET model listings), `clawhub`, `openclaw_api`, `openclaw_docs`, `npm_registry` (GET only, `openclaw` binary only) and the required `managed_inference` route, all TLS-terminated on 443 ([NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies)).

### 2.3 Lab 2.2 (L2.2) — anatomy of a policy file

The OpenShell schema page gives the top-level structure and a complete example. Annotated:

```yaml
version: 1

# STATIC — locked at creation
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /etc]
  read_write: [/tmp]
landlock:
  compatibility: best_effort        # or hard_requirement
# process:
#   run_as_user: "1500"
#   run_as_group: "1500"

# DYNAMIC — hot-reloadable
network_policies:
  github_rest_api:
    endpoints:
      - host: api.github.com
        port: 443
        protocol: rest
        enforcement: enforce
        access: read-only            # GET/HEAD/OPTIONS
    binaries:
      - path: /usr/bin/gh
  npm_registry:
    endpoints:
      - host: registry.npmjs.org
        port: 443
        protocol: rest
        enforcement: enforce
        access: read-only
        allow_encoded_slash: true
    binaries:
      - path: /usr/bin/node

network_middlewares:
  regex-redactor:
    name: Redact API tokens
    middleware: openshell/regex
    order: 10
    config: { mode: redact }
    on_error: fail_closed
    endpoints:
      include: ["*.example.com"]
      exclude: ["trusted.example.com"]
```

Field semantics (static/dynamic split, presets, `allow_encoded_slash`, middleware) are from the schema reference and the sandbox-policy guide ([OpenShell policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema), [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).

Rule forms you will use:

```yaml
# REST: allow wraps matchers; deny_rules list matchers directly
rules:
  - allow: { method: GET, path: /repos/** }
  - allow:
      method: GET
      path: /api/v1/download
      query:
        platform: { any: ["linux-*", "darwin-*"] }
deny_rules:
  - { method: "*", path: "/repos/*/*/rulesets" }

# WebSocket
rules:
  - allow: { method: GET, path: /v1/realtime }
  - allow: { method: WEBSOCKET_TEXT, path: /v1/realtime }
deny_rules:
  - { method: WEBSOCKET_TEXT, path: /v1/admin/** }

# GraphQL
rules:
  - allow: { operation_type: query }
  - allow: { operation_type: mutation, fields: [createIssue] }
deny_rules:
  - { operation_type: mutation, fields: [deleteRepository] }

# MCP (we return to this in Part 6)
rules:
  - allow: { method: initialize }
  - allow: { method: notifications/initialized }
  - allow: { method: tools/call, tool: { any: [search_web, list_issues] } }
deny_rules:
  - { method: tools/call, tool: send_email }

# Native TCP (databases)
network_policies:
  postgres:
    endpoints:
      - { host: db.internal.example, port: 5432, protocol: tcp }
    binaries:
      - { path: /usr/bin/psql }
```

([OpenShell policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema), [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).

### 2.4 Lab 2.3 (L2.3) — the iterate loop: deny → observe → allow → verify

The documented workflow is: create with an initial policy, watch denials, pull, edit, push, verify ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).

```bash
# 1. watch denials
openshell logs my-assistant --tail --source sandbox

# 2. additive fixes without rewriting YAML
openshell policy update my-assistant \
  --add-endpoint api.github.com:443:read-only:rest:enforce \
  --binary /usr/bin/gh --wait
openshell policy update my-assistant \
  --add-allow 'api.github.com:443:POST:/repos/*/issues' --wait
openshell policy update my-assistant \
  --add-deny 'api.github.com:443:POST:/admin/**' --wait
openshell policy update my-assistant --add-endpoint pypi.org:443 \
  --add-endpoint files.pythonhosted.org:443 \
  --binary /usr/bin/pip --binary /usr/local/bin/uv --wait

# 3. preview a merge before sending it
openshell policy update my-assistant --add-allow 'api.github.com:443:GET:/repos/**' --dry-run

# 4. remove
openshell policy update my-assistant --remove-endpoint pypi.org:443 --wait
openshell policy update my-assistant --remove-rule github_repos --wait

# 5. full replacement
openshell policy set my-assistant --policy current-policy.yaml --wait
openshell policy list my-assistant
```

Endpoint spec grammar is `host:port[:access[:protocol[:enforcement[:options]]]]`; rule spec is `host:port:METHOD:path_glob`. Note the trap: `api.github.com:443::rest` is rejected — an L7 endpoint with a protocol but no access or rules does not mean "allow all" ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).

Where the docs recommend it, prefer the NemoClaw wrapper for presets because it understands the blueprint's baseline (for example, it refuses an `npm` change if the live baseline drifted from the reviewed GET-only entry) ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)):

```bash
nemoclaw my-assistant policy add github --dry-run
nemoclaw my-assistant policy add github --yes
nemoclaw my-assistant policy remove github --yes
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml     # custom preset
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml --trusted-private-host 10.20.0.15 --dry-run
```

([NemoClaw integration policy examples](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/integration-policy-examples), [NemoClaw customize network policy](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/customize-network-policy)).

### 2.5 Lab 2.4 (L2.4) — operator approval in the TUI

When the agent hits an unlisted endpoint, OpenShell blocks it and surfaces the request in `openshell term` for review; approved endpoints become a new durable policy revision that persists across restarts of the same sandbox instance but resets when it is destroyed and recreated ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)). Before approval, OpenShell's prover flags risky new access (a new host with credentials, a new API method) and waits for a human ([OpenShell on GitHub](https://github.com/NVIDIA/openshell)).

Exercise this deliberately: ask the assistant to "fetch https://httpbin.org/get and show me the headers", watch the TUI, approve once, then check `openshell policy list` to see the new revision.

### 2.6 Lab 2.5 (L2.5) — presets and posture profiles

Maintained presets live in `nemoclaw-blueprint/policies/presets/` and include `brave`, `brew`, `claude-code`, `discord`, `github`, `gmail`, `googlechat`, `huggingface`, `jira`, `local-inference`, `npm`, `nous-*` (Hermes), `openclaw-pricing`, `outlook`, `public-reference`, `pypi`, `slack`, `tavily`, `teams`, `telegram`, `weather`, `wechat`, `whatsapp` ([NemoClaw integration policy examples](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/integration-policy-examples)).

Read each preset's risk column before applying: `pypi` is GET/HEAD only but allows installing arbitrary packages; `github` gives read/write to repos via `git` only (binary-scoped to `/usr/bin/git`); `slack`/`discord` WebSocket legs use `access: full` with no inspection; `personal-open-internet` removes hostname, method, path and body restrictions on ports 80/443 ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).

Posture profiles from the same page, which we will map to AltoTech deployments in Part 6:

| Profile | Tier | Presets | Inference | Notes |
|---|---|---|---|---|
| Locked-Down | Restricted | none (no web search) | NVIDIA Endpoints or local Ollama | operator approval for everything else; watch TUI |
| Development | Balanced | `pypi`, `npm` | any | keep binary restrictions; review with `openshell term` |
| Personal | Personal | `personal-open-internet` | any | trusted single-user only; recreate as Balanced when done |
| Integration Testing | custom | tight method/path entries, `protocol: rest` | any | clean up baseline after tests |

### 2.7 Lab 2.6 (L2.6) — snapshots, rebuild, recovery

```bash
nemoclaw my-assistant snapshot create --name before-change
# ...make changes...
nemoclaw my-assistant rebuild
```

The snapshot/rebuild verbs are shown in the Deep Agents quickstart for `nemo-deepagents` and behave the same for the other CLIs ([NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart)). The security guide's rule for suspected compromise is simple: recreate the sandbox from trusted inputs rather than trying to clean it, because the agent can mutate its own config tree ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).

### 2.8 Lab 2.7 (L2.7) — raw OpenShell without NemoClaw: bring your own vLLM

This lab is the OpenShell playbook, condensed. It teaches you what `nemoclaw onboard` does under the hood and gives you full control of the model server ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions)).

```bash
# 1. install OpenShell CLI + gateway service
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
source ~/.bashrc && openshell --help
systemctl --user status --no-pager openshell-gateway
openshell status                      # expect: Connected
sudo loginctl enable-linger $USER     # keep gateway alive after logout
journalctl --user -u openshell-gateway -f

# 2. serve a model with vLLM (host 0.0.0.0, port 8000)
export HF_TOKEN=...; export MODEL_HANDLE="<HF handle from recipes.vllm.ai for DGX Spark>"
export VLLM_IMAGE=vllm/vllm-openai:latest; export MAX_MODEL_LEN=131072
docker run -d --name vllm-server --gpus all --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --entrypoint "" \
  -p 8000:8000 -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  "$VLLM_IMAGE" vllm serve "$MODEL_HANDLE" --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.8
timeout 900 bash -c 'until curl -sf http://localhost:8000/health >/dev/null; do sleep 10; done'
curl -s http://0.0.0.0:8000/v1/models

# 3. register it as an OpenShell provider — use the LAN IP, not localhost
IP=$(hostname -I | awk '{print $1}')
openshell provider create --name local-vllm --type openai \
  --credential OPENAI_API_KEY=not-needed --config OPENAI_BASE_URL=http://$IP:8000/v1
openshell provider list

# 4. route inference.local to it
openshell inference set --provider local-vllm --model "$MODEL_HANDLE"
openshell inference get                 # provider: local-vllm

# 5. create a sandbox from the community OpenClaw image
export SANDBOX_NAME=openshell-demo
openshell sandbox create --keep --forward 18789 --name "$SANDBOX_NAME" --from openclaw -- openclaw-start
openshell forward start --background 18789 "$SANDBOX_NAME"
openshell forward list

# 6. inside the OpenClaw wizard: Custom Provider, base URL https://inference.local/v1,
#    any non-empty key ("not-needed"), OpenAI-compatible, model = $MODEL_HANDLE

# 7. verify isolation, transfer files, clean up
openshell term
openshell sandbox upload "$SANDBOX_NAME" ./local-file /sandbox/destination
openshell sandbox download "$SANDBOX_NAME" /sandbox/file ./local-destination
openshell sandbox ssh-config "$SANDBOX_NAME"    # append to ~/.ssh/config for VS Code
openshell sandbox delete "$SANDBOX_NAME"; openshell provider delete local-vllm
```

Two gotchas the playbook calls out: the gateway runs in Docker and cannot reach host services on `127.0.0.1`, so bind vLLM to `0.0.0.0` and use the machine IP (or `host.docker.internal` where it resolves); and do not pass `--policy` with a local file when using `--from openclaw`, because the policy is bundled ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions)). The vLLM launch block is the DGX Spark vLLM playbook's base configuration ([DGX Spark vLLM instructions](https://build.nvidia.com/spark/vllm/instructions)).

Other sandbox flavours worth knowing: `openshell sandbox create --from base` (minimal Ubuntu, no agent) and `--from sdg`; `--from ./dir` or a Dockerfile for custom images; `--gpu`, `--cpu 2 --memory 4Gi`, `--forward`, `--upload`, `--env`, `--label`; `openshell sandbox exec -n <name> -- <cmd>` for one-shot commands (the trailing command in `create` is the health-defining main process — if it exits the sandbox goes to `Error`) ([OpenShell on GitHub](https://github.com/NVIDIA/openshell), [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)). Set `OPENSHELL_SANDBOX_POLICY=./my-policy.yaml` to avoid passing `--policy` every time ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).

### Exercises — Part 2

1. Write a custom preset `alto-bms.yaml` that lets only `/usr/bin/python3` call a private BMS REST API at `bms.alto.local:8443` with `GET /api/v1/points/**` and `POST /api/v1/setpoints/*` allowed, `POST /api/v1/admin/**` denied, enforced. The host resolves to `10.20.0.15`.
2. Explain why `enforcement: audit` is the default and when to flip to `enforce`.
3. A colleague wants `read_write: [/]` so the agent can "install anything". What does OpenShell do, and what is the correct alternative?
4. Your agent needs `psql` to reach a Postgres at `timescale.alto.local:5432`. Write the endpoint spec both as YAML and as an `--add-endpoint` argument.
5. Why must you never put `tls: skip` on a provider-credentialed endpoint without also thinking about `allow_uninspected_credentials`?
6. You approved three endpoints in the TUI during a debugging session. What is the fastest way to guarantee they are gone?

**Solutions**

1.
   ```yaml
   version: 1
   network_policies:
     alto_bms:
       name: alto_bms
       endpoints:
         - host: bms.alto.local
           port: 8443
           protocol: rest
           enforcement: enforce
           rules:
             - allow: { method: GET,  path: "/api/v1/points/**" }
             - allow: { method: POST, path: "/api/v1/setpoints/*" }
           deny_rules:
             - { method: POST, path: "/api/v1/admin/**" }
       binaries:
         - { path: /usr/bin/python3 }
   ```
   Apply with `nemoclaw my-assistant policy add --from-file ./alto-bms.yaml --trusted-private-host bms.alto.local --dry-run`, review the generated address pins, then rerun with `--yes`. An exact declared hostname may resolve to a private RFC 1918 address; wildcard/hostless entries may not ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices), [NemoClaw customize network policy](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/customize-network-policy)).
2. `audit` logs rule violations but forwards traffic so you can learn the real access pattern; switch to `enforce` once rules are validated, after which non-matching requests get `403 Forbidden` with a JSON body ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
3. Rejected with `INVALID_ARGUMENT` — overly broad read-write paths such as `/` are refused. Add a specific writable subdirectory (for example `/sandbox/tools`) and install binaries at image build time via `nemoclaw onboard --from <Dockerfile>` instead of runtime egress ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices), [NemoClaw customize network policy](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/customize-network-policy)).
4. YAML: `endpoints: [{host: timescale.alto.local, port: 5432, protocol: tcp}]` with `binaries: [{path: /usr/bin/psql}]`. CLI: `--add-endpoint timescale.alto.local:5432::tcp --binary /usr/bin/psql` — the empty access segment before `tcp` is required ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).
5. `tls: skip` disables placeholder credential rewriting, token injection and L7 inspection; the proxy relays ciphertext blind. A provider-credentialed endpoint additionally requires `allow_uninspected_credentials: true` as an explicit acknowledgement that OpenShell cannot see or rewrite the traffic ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
6. Destroy and recreate the sandbox: approvals persist only within a sandbox instance and reset on recreation ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

---
## Part 3 — Building claws with NeMo Agent Toolkit (NAT)

### 3.1 Concepts

NAT (`nvidia-nat`, formerly AIQ Toolkit / Agent Intelligence Toolkit) is a framework-agnostic layer that sits beside LangChain/LangGraph, CrewAI, Semantic Kernel, Google ADK, Strands and AutoGen; every agent, tool and workflow is a composable function; a YAML file with `functions`, `llms`, `embedders` and `workflow` sections wires them together, and `_type` switches implementations ([NeMo Agent Toolkit docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/), [Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/)). It ships a profiler (tokens, timings, bottlenecks), evaluators, tracing exporters (LangSmith, Phoenix, Weave, Langfuse, OpenTelemetry), a chat UI, and MCP and A2A client/server support ([NeMo Agent Toolkit docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/)). The current documentation version is 1.8 ([NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html)).

Why NAT in a NemoClaw course? Because the harnesses (OpenClaw/Hermes/Deep Agents) are general assistants; when you want a purpose-built claw — Alto Ops Claw — with explicit tools, deterministic evaluation and profiling, NAT gives you that, and it can run either on the Spark host (calling vLLM directly) or inside an OpenShell sandbox (calling `inference.local`).

Built-in agent types: ReAct, Reasoning, ReWOO, Responses API, Router, Tool Calling, plus Parallel and Sequential executors and an Automatic Memory Wrapper ([NAT agents](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/index.html)).

### 3.2 Lab 3.1 (L3.1) — install NAT on the Spark (arm64)

`nvidia-nat` is a pure-Python wheel and installs on arm64 without compilation; the Classmethod write-up used Python 3.12 in a uv venv on DGX OS with CUDA 13.0 ([Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/)).

```bash
mkdir -p ~/works/alto-ops-claw && cd ~/works/alto-ops-claw
uv venv --python 3.12 && source .venv/bin/activate
uv pip install 'nvidia-nat[langchain,mcp,profiler,phoenix,opentelemetry]'
nat --version
```

Extras you will meet: `langchain` (ReAct/tool-calling agents), `mcp` (client + server), `profiler` (needed by `nat eval` profiling and the sizing calculator), `phoenix`, `opentelemetry`, `weave`, `data-flywheel`, `async_endpoints` ([NAT installing](https://docs.nvidia.com/nemo/agent-toolkit/latest/quick-start/installing.html), [NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html), [NAT API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html)).

### 3.3 Lab 3.2 (L3.2) — a local vLLM for NAT

Reuse the `vllm-server` container from Lab 2.7, or start the smaller Nemotron 3 Nano that Classmethod validated on Spark:

```bash
docker run -d --name vllm-nat --gpus all --shm-size=16g -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Nemotron 3 Nano uses the hybrid Mamba–Transformer `nemotron_h` architecture and needs `--trust-remote-code`; without it the server crashes on a pydantic `ValidationError`. Loading 30.5 GiB took about four minutes plus two minutes of `torch.compile`, with 34.79 GiB allocated to KV cache at 8K context ([Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/)).

### 3.4 Lab 3.3 (L3.3) — hello, local workflow

Scaffold a package, then point it at vLLM:

```bash
nat workflow create --no-install --workflow-dir ./workflows alto_ops --description "Alto Ops Claw"
# creates workflows/alto_ops/{pyproject.toml, src/alto_ops/{alto_ops.py, register.py, configs/config.yml}}
```

Minimal `workflow.yml` for local inference (the `api_key` field is required by validation even for keyless servers — use a dummy like `EMPTY`):

```yaml
functions:
  current_datetime:
    _type: current_datetime
llms:
  local_vllm:
    _type: openai
    base_url: http://localhost:8000/v1
    api_key: EMPTY
    model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
    temperature: 0.0
workflow:
  _type: react_agent
  llm_name: local_vllm
  tool_names: [current_datetime]
  verbose: true
```

```bash
nat run --config_file workflow.yml --input "What time is it in Bangkok right now?"
```

The scaffold command, the `_type: openai` + `base_url` pattern, the `EMPTY` key requirement and the observed ~13 s latency per ReAct cycle on Spark are all from the Classmethod article ([Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/)); the ReAct hello-world shape (`wiki_search`, `react_agent`, `parse_agent_response_max_retries`) is NAT's own front page ([NeMo Agent Toolkit docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/)).

### 3.5 Lab 3.4 (L3.4) — write a custom tool: chiller-plant CSV analytics

Functions are registered with `@register_function(config_type=...)`, take a `FunctionBaseConfig` subclass and a `Builder`, and `yield` either a callable, a `FunctionInfo`, or a `Function` subclass; the registration coroutine may do async setup and cleanup around the `yield` ([NAT writing custom functions](https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/functions.html)).

`workflows/alto_ops/src/alto_ops/chiller_tool.py`:

```python
from pydantic import Field
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ChillerKpiConfig(FunctionBaseConfig, name="chiller_kpi"):
    """Compute plant kW/RT and flag anomalies from a chiller CSV export."""
    csv_path: str = Field("/sandbox/data/chiller_plant.csv", description="CSV with columns ts,kw,rt")
    kw_per_rt_alarm: float = Field(0.85, description="Alarm threshold for plant efficiency", gt=0)


@register_function(config_type=ChillerKpiConfig)
async def chiller_kpi(config: ChillerKpiConfig, builder: Builder):
    import csv

    async def _kpi(hours: int = 24) -> str:
        rows = list(csv.DictReader(open(config.csv_path)))[-hours * 4:]   # 15-min data
        kw = sum(float(r["kw"]) for r in rows) / len(rows)
        rt = sum(float(r["rt"]) for r in rows) / len(rows)
        eff = kw / rt if rt else float("nan")
        flag = "ALARM" if eff > config.kw_per_rt_alarm else "OK"
        return f"window={hours}h avg_kw={kw:.1f} avg_rt={rt:.1f} kw_per_rt={eff:.3f} status={flag}"

    yield FunctionInfo.from_fn(
        _kpi,
        description="Average chiller plant kW, RT and kW/RT over the last N hours; flags efficiency alarms.",
    )
```

Register it in `register.py` (`from . import chiller_tool  # noqa`), install the package (`uv pip install -e workflows/alto_ops`) and reference it in YAML:

```yaml
functions:
  chiller_kpi:
    _type: chiller_kpi
    csv_path: ./data/chiller_plant.csv
    kw_per_rt_alarm: 0.80
  current_datetime:
    _type: current_datetime
llms:
  local_vllm:
    _type: openai
    base_url: http://localhost:8000/v1
    api_key: EMPTY
    model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
    temperature: 0.0
workflow:
  _type: tool_calling_agent
  llm_name: local_vllm
  tool_names: [chiller_kpi, current_datetime]
  verbose: true
  handle_tool_errors: true
```

The `tool_calling_agent` YAML keys (`tool_names`, `llm_name`, `verbose`, `handle_tool_errors`) follow the tool-calling agent reference ([NAT tool calling agent](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/tool-calling-agent/tool-calling-agent.html)). The import paths above follow the NAT 1.x package layout shown in NAT's docs and examples; confirm them against `nat workflow create`'s generated `register.py` on your installed version, since module paths have moved between releases.

Composition pattern — an agent as a tool of another agent — is also YAML-only:

```yaml
functions:
  math_agent:
    _type: tool_calling_agent
    tool_names: [calculator]
    llm_name: local_vllm
    description: 'Useful for performing simple mathematical calculations.'
```

([NAT tool calling agent](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/tool-calling-agent/tool-calling-agent.html)).

### 3.6 Lab 3.5 (L3.5) — serve it: REST and OpenAI-compatible endpoints

```bash
nat serve --config_file workflow.yml           # FastAPI on :8000 by default; pick --port if vLLM is on 8000
curl -X POST http://localhost:8001/v1/workflow -H 'Content-Type: application/json' \
  -d '{"input_message":"Is the chiller plant efficient over the last 6 hours?"}'
curl -X POST http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"alto-ops","messages":[{"role":"user","content":"Plant status?"}],"stream":false}'
curl -X POST 'http://localhost:8001/v1/workflow/full?filter_steps=LLM_END,TOOL_END' \
  -H 'Content-Type: application/json' -d '{"input_message":"Plant status?"}'
```

Endpoints: `/v1/workflow` (+`/stream`, `/full`, `/async`), `/v1/chat` (+`/stream`), `/v1/chat/completions` (OpenAI-compatible, so the OpenAI Python client works with `base_url=http://localhost:8000/v1` and a dummy key), `/feedback`, and `/monitor/users` when `general.enable_per_user_monitoring: true`; legacy `/generate` and `/chat` remain unless `disable_legacy_routes: true` ([NAT API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html)). `/v1/workflow/full` returns intermediate steps and accepts `filter_steps` — you will use this in Part 4.

### 3.7 Lab 3.6 (L3.6) — MCP both ways

**NAT as an MCP server** (publish Alto Ops tools to OpenClaw/Hermes or Claude Code):

```bash
nat mcp serve --config_file workflow.yml --name "Alto Ops MCP" --host 0.0.0.0 --port 9901
nat mcp serve --config_file workflow.yml --tool_names chiller_kpi     # publish one tool only
curl -s http://localhost:9901/debug/tools/list | jq
nat mcp client tool list --url http://localhost:9901/mcp
nat mcp client tool call chiller_kpi --url http://localhost:9901/mcp --json-args '{"hours": 6}'
```

Default transport is streamable-HTTP on `/mcp`, port 9901; `--transport sse` is available for legacy clients; base path is configurable under `general.front_end._type: mcp` ([NAT MCP server](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/mcp-server.html)).

**NAT as an MCP client** (consume a BMS MCP server as tools):

```yaml
function_groups:
  bms_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://bms.alto.local:8443/mcp"
      # auth_provider: mcp_oauth2      # for protected servers
    include: [read_point, list_alarms, write_setpoint]
    tool_call_timeout: 60
    reconnect_enabled: true
    reconnect_max_attempts: 3
    tool_overrides:
      write_setpoint:
        description: "Write a setpoint. Requires an approved work order id."
workflow:
  _type: tool_calling_agent
  llm_name: local_vllm
  tool_names: [bms_tools, chiller_kpi]
```

`mcp_client` discovers server tools, filters with `include`/`exclude`, renames with `tool_overrides`, supports `stdio`, `sse` and `streamable-http`, and exposes timeout, reconnect and session knobs; individual tools are referenced as `<group>__<tool>` ([NAT MCP client](https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html)). Inspect any server first with `nat mcp client tool list --url ... [--auth]`.

### 3.8 Lab 3.7 (L3.7) — sandboxed code execution for the agent

If Alto Ops Claw should write and run Python for ad-hoc analysis, do not let it exec on the host. NAT's `code_execution` function sends code to a remote sandbox — a local Docker `local_sandbox` (default `http://127.0.0.1:6000`, 10 s timeout, 1000 output chars) or a Piston server ([NAT code execution](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/functions/code-execution.html)).

```bash
cd packages/nvidia_nat_core/src/nat/tool/code_execution/local_sandbox && source start_local_sandbox.sh
```

```yaml
functions:
  code_execution_tool:
    _type: code_execution
    uri: "http://127.0.0.1:6000"
    timeout: 30
    max_output_characters: 3000
```

### 3.9 Lab 3.8 (L3.8) — run NAT inside an OpenShell sandbox

This is where the two halves of the course meet. Build a sandbox image with NAT preinstalled, keep the host filesystem out, and make the only egress `inference.local`.

`Dockerfile.alto-ops`:

```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y python3.12 python3-pip curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages uv && uv pip install --system 'nvidia-nat[langchain,mcp,profiler,opentelemetry]'
COPY workflows/alto_ops /app/alto_ops
RUN uv pip install --system -e /app/alto_ops
COPY workflow.sandbox.yml /app/workflow.yml
USER 1500
WORKDIR /sandbox
```

`workflow.sandbox.yml` differs in one line — the LLM points at the managed route:

```yaml
llms:
  routed:
    _type: openai
    base_url: https://inference.local/v1
    api_key: EMPTY
    model_name: <MODEL_HANDLE configured with openshell inference set>
```

Create the sandbox with a policy that has **no** network entries beyond what NemoClaw/OpenShell composes for inference (inference is intercepted by the supervisor, it is not a `network_policies` entry you add):

```bash
openshell sandbox create --name alto-ops --from ./ --policy ./alto-ops-policy.yaml \
  --upload ./data:/sandbox/data --forward 8001 --keep \
  -- nat serve --config_file /app/workflow.yml --host 0.0.0.0 --port 8001
openshell forward start --background 8001 alto-ops
openshell logs alto-ops --tail --source sandbox
```

`--from ./dir`/Dockerfile, `--upload`, `--forward`, `--policy`, `--keep` and the trailing main-process command are the documented `sandbox create` options ([OpenShell on GitHub](https://github.com/NVIDIA/openshell), [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)); the NemoClaw equivalent for custom images is `nemoclaw onboard --from <Dockerfile>` ([NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart)). The Dockerfile above is this tutorial's own composition — verify the NAT install line and non-root `USER` against your base image; OpenShell requires a non-root identity and rejects root ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

Because NAT's Python process is the caller, the TLS trust for the intercepting proxy is already injected via `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)), so `https://inference.local/v1` works out of the box for `openai`-type clients.

### Exercises — Part 3

1. Convert Lab 3.4 from `tool_calling_agent` to `react_agent` and to `rewoo_agent`. Which one makes fewer LLM calls for "give me the 6-hour and 24-hour kW/RT"? Verify with `/v1/workflow/full`.
2. Add a `write_setpoint` tool and make it refuse unless the input carries a `work_order_id`. Where do you enforce this — in the tool, in the prompt, or in OpenShell policy? Argue all three.
3. Publish Alto Ops as MCP and connect it to your OpenClaw sandbox from Part 1. What policy entry does the OpenClaw sandbox need?
4. Write a `network_policies` entry that lets the sandboxed NAT process from Lab 3.8 call the BMS MCP server but only `tools/call` on `read_point` and `list_alarms`.
5. The sandboxed NAT calls `https://inference.local/v1` but you see `deny` in the TUI for `inference.local`. List three things to check.

**Solutions**

1. Expect ReWOO to plan both tool calls up front and execute them, typically with fewer LLM round-trips than ReAct's think→act→observe loop; tool-calling agents rely on the model's native function-calling and usually need one planning call plus one summarisation call. Measure rather than assume: `curl ... /v1/workflow/full?filter_steps=LLM_END,TOOL_END` shows the exact sequence ([NAT API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html)). Agent type names are listed in the agents index ([NAT agents](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/index.html)).
2. Do all three, in layers: (a) the tool validates `work_order_id` with a Pydantic `input_schema` override so a missing id raises `ValidationError` before any network call ([NAT writing custom functions](https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/functions.html)); (b) the `tool_overrides.description` tells the model the precondition; (c) OpenShell policy with `protocol: mcp` denies `tools/call` for `write_setpoint` entirely in sandboxes that should never write. Only (c) survives prompt injection.
3. The OpenClaw sandbox needs an endpoint entry for the NAT MCP host and port bound to `/usr/local/bin/openclaw` (and `/usr/local/bin/node` if OpenClaw's MCP client runs under node), ideally `protocol: mcp` with `initialize`, `notifications/initialized`, `tools/list` and `tools/call` on `chiller_kpi` allowed. If the MCP server runs on the Spark host, it is a private address: declare the exact host or use `--trusted-private-host` with the NemoClaw preset flow ([NemoClaw customize network policy](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/customize-network-policy)).
4.
   ```yaml
   network_policies:
     bms_mcp:
       name: bms_mcp
       endpoints:
         - host: bms.alto.local
           port: 8443
           path: /mcp
           protocol: mcp
           enforcement: enforce
           mcp: { max_body_bytes: 131072 }
           rules:
             - allow: { method: initialize }
             - allow: { method: notifications/initialized }
             - allow: { method: tools/list }
             - allow: { method: tools/call, tool: { any: [read_point, list_alarms] } }
           deny_rules:
             - { method: tools/call, tool: write_setpoint }
       binaries:
         - { path: /usr/bin/python3.12 }
   ```
   MCP rule fields (`method`, `tool`, `mcp.max_body_bytes`, `strict_tool_names` default true) are from the sandbox-policy guide ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).
5. (i) `openshell inference get` — is a provider and model actually set? (ii) Was the sandbox created through the managed gateway path? Policy and inference auth are not enforced when a runtime launches outside it ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)); (iii) the upstream is unhealthy — `openshell inference set` reports `failed to verify inference endpoint` if vLLM is not warm; warm it with one chat completion, then retry (or `--no-verify` once reachability is confirmed) ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions)).

---

## Part 4 — Tracing and observability

### 4.1 Concepts

You have three telemetry planes in a claw, and they answer different questions:

| Plane | Source | Answers |
|---|---|---|
| Agent traces | NAT telemetry exporters (Phoenix, OTel collector, Langfuse, Weave, file); Hermes Langfuse plugin | What did the model think, which tools ran, how many tokens, how long |
| Policy/inference logs | OpenShell supervisor and gateway: `openshell logs`, `openshell term`, OCSF findings | What did the agent try to reach, what was allowed/denied/inspected, did Landlock apply |
| Harness logs | `nemoclaw <sandbox> logs --follow`, `/tmp/gateway.log` inside OpenClaw | Channel events, pairing, crashes |

NAT's observability runs off the hot path: an `IntermediateStepManager` publishes `IntermediateStep` events (function boundaries, LLM calls, tool usage) to a reactive stream; exporters consume them asynchronously; raw, span, OpenTelemetry and custom exporter types exist and several can run at once ([NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html)).

### 4.2 Lab 4.1 (L4.1) — Phoenix on the Spark

```bash
docker run -d --name phoenix-nat -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
```

Add to `workflow.yml`:

```yaml
general:
  telemetry:
    logging:
      console: { _type: console, level: WARN }
      file:    { _type: file, path: ./.tmp/alto_ops.log, level: DEBUG }
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: alto-ops-claw
        # api_key: ${PHOENIX_API_KEY}     # if Phoenix auth is enabled
      file_backup:
        _type: file
        # path etc.
```

```bash
nat run --config_file workflow.yml --input "Plant status last 6 hours?"
# open http://localhost:6006 → trace tree: <workflow> → tool_calling_agent → chiller_kpi, LLM spans with prompt/completion tokens
```

Phoenix docker command, the `_type: phoenix` block, and the multi-exporter layout are from the NAT observability guide ([NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html)). One documented quirk: when using the generic `otelcollector` exporter against Phoenix, NAT sends `project` as `service.name` and Phoenix files traces under `default`; the native `phoenix` exporter is the cleaner route ([Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/)).

Discover every registered exporter on your install:

```bash
nat info components -t tracing
nat info components -t logging
```

### 4.3 Lab 4.2 (L4.2) — vendor-neutral: OTel collector to files (or Dynatrace)

```bash
cat > otelcollectorconfig.yaml <<'YML'
receivers:
  otlp:
    protocols:
      http: { endpoint: 0.0.0.0:4318 }
exporters:
  file:
    path: /otellogs/llm_spans.json
service:
  pipelines:
    traces: { receivers: [otlp], exporters: [file] }
YML
docker run -d -v $(pwd)/otelcollectorconfig.yaml:/etc/otelcol-contrib/config.yaml \
  -p 4318:4318 -v $(pwd)/otellogs:/otellogs/ otel/opentelemetry-collector-contrib:0.128.0
```

```yaml
general:
  telemetry:
    tracing:
      otelcollector:
        _type: otelcollector
        endpoint: http://0.0.0.0:4318/v1/traces
        project: alto-ops-claw
```

Then `cat otellogs/llm_spans.json`. The collector image tag, mount paths and the `otelcollector` exporter YAML are from the NAT guide, which also shows the same exporter pointed at a Dynatrace collector ([NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html)). Because the collector is a normal OTLP endpoint, a sandboxed NAT (Lab 3.8) can export to it if — and only if — you add an endpoint entry for the collector host bound to the Python binary; that is a good moment to use `protocol: rest`, `access: read-write` in `audit` mode first.

### 4.4 Lab 4.3 (L4.3) — Hermes claw traces to Langfuse without leaking keys

The Hermes quickstart shows how NemoClaw registers Langfuse keys as an OpenShell credential of type `langfuse-hermes-v1`; the sandbox receives only placeholders and OpenShell substitutes the real values at egress ([NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart)):

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
nemohermes credentials add my-hermes-langfuse \
  --type langfuse-hermes-v1 \
  --credential LANGFUSE_PUBLIC_KEY \
  --credential LANGFUSE_SECRET_KEY
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
nemohermes my-hermes rebuild

nemohermes my-hermes connect
hermes plugins enable observability/langfuse
exit
nemohermes my-hermes gateway restart
```

Do not put raw Langfuse keys or copied placeholders into `~/.hermes/.env`; only the non-secret `HERMES_LANGFUSE_BASE_URL` belongs there ([NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart)). This is the pattern to copy for any SaaS observability backend: credential handle in OpenShell, placeholder in the sandbox, substitution at the proxy.

For an OpenClaw claw, the network-policies reference notes that enabling OpenClaw OTEL diagnostics with a local endpoint adds the `openclaw-diagnostics-otel-local` preset on Balanced/Open/Personal tiers ([NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies)) — so an OTel collector on the Spark host is the natural sink for both harness and NAT traces.

### 4.5 Lab 4.4 (L4.4) — the policy plane: reading OpenShell like a trace

```bash
openshell logs alto-ops --tail --source sandbox     # denied host, path, binary
openshell term                                      # live: allow / deny / inspect_for_inference
docker logs $(docker ps --filter name=openshell-alto-ops --format '{{.Names}}') --tail 50
#   look for "OpenShell Sandbox Supervisor success" and "Applying Landlock filesystem sandbox"
openshell settings get alto-ops                     # effective policy source
openshell policy get alto-ops --full                # what is really enforced now
```

Log sources and TUI semantics are from the OpenShell playbook and sandbox-policy guide ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions), [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)). Two facts to remember when correlating: L7 `enforcement: audit` logs violations but forwards traffic — so a "violation" in audit mode is a finding, not a block; and a skipped Landlock path under `best_effort` shows up as a High-severity OCSF `DetectionFinding` ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).

### 4.6 Lab 4.5 (L4.5) — correlate: one request, three planes

1. Send one request through `nat serve` (`/v1/workflow/full`) and save the intermediate steps.
2. Find the same run in Phoenix by timestamp and input text; note LLM span count and total tokens.
3. In `openshell term`, filter (`s`) to the sandbox and count `inspect_for_inference` events — it should equal the LLM span count.
4. Trigger a denied call (ask the agent to "download the latest weather from open-meteo") and confirm you see a tool span fail in Phoenix while OpenShell shows `deny`.

Write the mapping down; it is the basis of the audit story you will tell a hotel owner or a Thai regulator in Part 6.

### Exercises — Part 4

1. Configure Phoenix and file exporters simultaneously and prove both receive the same run.
2. NAT's `otelcollector` exporter sends `project` as `service.name`. Design a collector `processors` block that routes traces to different Phoenix projects per hotel property (hint: resource attribute → header). Note what you would need to verify.
3. A sandboxed NAT exporter to `otel.alto.local:4318` shows `403` in the collector logs. Which OpenShell setting did you forget?
4. Explain why storing `LANGFUSE_SECRET_KEY` in `/sandbox/.hermes/config.yaml` is worse than useless.
5. Name the NAT decorator for tracing an arbitrary Python function that is not a registered NAT function, and the three event types it emits.

**Solutions**

1. Two entries under `general.telemetry.tracing` (`phoenix:` and `file_backup:`); multiple exporters run simultaneously by design ([NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html)). Compare the file's span count with the Phoenix trace.
2. Use an OTel collector `routing` or `attributes` processor keyed on `service.name` to set the Phoenix project header per pipeline. Phoenix's project-routing header name and the exact collector processor configuration are not covered by the sources in this tutorial — verify against Phoenix docs before relying on it; the Classmethod author flagged this as future work ([Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/)).
3. You set `protocol: rest` with `enforcement: enforce` and a `read-only` access preset (GET/HEAD/OPTIONS); OTLP/HTTP is a POST. Use `access: read-write` or an explicit `allow: {method: POST, path: /v1/traces}` ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies), [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
4. The agent can read and rewrite its own config tree — the docs treat `/sandbox/.hermes` as mutable, agent-controlled state and not an isolation boundary — so a real key there is exfiltrable by prompt injection, and the sandbox cannot even use it correctly because OpenShell expects the placeholder ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices), [NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart)).
5. `@track_function` from `nat.plugins.profiler.decorators.function_tracking`; it emits `SPAN_START`, `SPAN_CHUNK` (generators) and `SPAN_END` ([NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html)).

---
## Part 5 — Performance benchmarking on DGX Spark

### 5.1 Concepts: the four layers you must benchmark separately

1. **Engine / model** — tokens per second, single-stream and aggregate, per quantisation. Bounded by the Spark's LPDDR5X bandwidth.
2. **Workflow** — end-to-end latency per agent turn, LLM calls per turn, tokens per turn (NAT profiler).
3. **Quality** — does the claw finish the task correctly (NAT evaluators, Exxact-style task suites).
4. **Sandbox overhead** — the delta added by OpenShell's proxy/TLS interception and Landlock.

A fast engine paired with a model that fails tool calls is slower in practice than a slow reliable one; Exxact's agent benchmark found reliability and structured-output discipline mattered more than raw tok/s ([Exxact — local agents on DGX Spark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark)).

> All third-party figures below are single-machine measurements published by their authors on specific software versions; treat them as reference points to reproduce, not as specifications.

### 5.2 Reference numbers for the Spark (third-party)

| Model / engine | Metric | Value | Source |
|---|---|---|---|
| Nemotron 3 Nano 30B-A3B, vLLM, W4A16 NVFP4 | single-stream decode | 74.75 tok/s (vs ~67 public baseline, 47.6 FP8 Triton) | [ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nemotron-3-nano-w4a16-74-toks) |
| same, W4A4 NVFP4 | single / aggregate c=16 | 58.27 / 786 tok/s | [ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nemotron-3-nano-w4a16-74-toks) |
| same, W4A16 | aggregate | ~400 tok/s | [ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nemotron-3-nano-w4a16-74-toks) |
| Nemotron 3 Nano NVFP4, vLLM (forum) | single-stream | 65+ tok/s | [NVIDIA developer forum](https://forums.developer.nvidia.com/t/dgx-spark-nemotron3-and-nvfp4-getting-to-65-tps/355261) |
| nemotron-3-nano:30b, Ollama | avg tok/s over agent tasks | 64.7 | [Exxact benchmark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark) |
| nemotron-3-super:120b-a12b, Ollama | avg tok/s; task pass | 16.4; 17/17 (strongest agent profile) | [Exxact benchmark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark) |
| qwen3.5:35b-a3b / qwen3.5:122b-a10b, Ollama | avg tok/s | 48.2 / 20.1 | [Exxact benchmark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark) |
| gemma4:26b, Ollama vs vLLM | single-stream | ~64 vs ~30 tok/s | [Exxact engines](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark) |
| gemma4:26b, vLLM | aggregate at 10+ concurrent | >300 tok/s | [Exxact engines](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark) |
| gemma4:26b, Ollama `OLLAMA_NUM_PARALLEL=4` | aggregate | ~122 tok/s | [Exxact engines](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark) |
| NAT ReAct + 1 tool on Nemotron 3 Nano FP8 | end-to-end per query | ~13 s | [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) |

Interpretation guidance from those sources: decode on Spark is bandwidth-bound at roughly 273 GB/s, so the single-stream ceiling for a 3B-active MoE sits in the low 80s tok/s regardless of engine tricks ([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nemotron-3-nano-w4a16-74-toks)); above ~20 tok/s an agent feels usable and 40+ feels responsive ([Exxact benchmark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark)); Ollama wins single-user latency while vLLM wins multi-user throughput, and Mamba-hybrid Nemotron models did not gain from Ollama parallelism ([Exxact engines](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark)). Exxact also recommends a memory watchdog because unified memory pressure can hard-reset the Spark ([Exxact engines](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark)).

### 5.3 Lab 5.1 (L5.1) — engine benchmark you can reproduce in 10 minutes

```bash
# vLLM built-in benchmark, single stream then concurrency sweep
docker exec vllm-nat vllm bench serve --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --backend openai-chat --endpoint /v1/chat/completions --host 127.0.0.1 --port 8000 \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --num-prompts 32 --max-concurrency 1
for c in 2 4 8 16; do docker exec vllm-nat vllm bench serve ... --num-prompts $((c*8)) --max-concurrency $c; done
```

Record TTFT, output tok/s per request and aggregate. Then repeat against Ollama (`ollama run nemotron-3-nano:30b --verbose`) so you have both engines on the same prompt set. Exxact's harness is open if you prefer a ready-made agent task suite ([Exxact local-agent-benchmark on GitHub](https://github.com/Exxact-Software/local-agent-benchmark)).

### 5.4 Lab 5.2 (L5.2) — NAT profiler and evaluation

Create `data/alto_ops_eval.jsonl` with 20 questions and reference answers derived from your CSV (for example `{"id": 1, "question": "Average kW/RT over the last 6 hours?", "answer": "0.78"}`); NAT datasets treat `question` as input and `answer` as the reference, with `structure` overrides available for other column names ([NAT evaluate](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html)).

`eval_config.yml`:

```yaml
# ...functions/llms/workflow from Part 3...
eval:
  general:
    output_dir: ./.tmp/eval/alto_ops/
    max_concurrency: 4
    dataset:
      _type: jsonl
      file_path: ./data/alto_ops_eval.jsonl
    profiler:
      token_uniqueness_forecast: true
      workflow_runtime_forecast: true
      compute_llm_metrics: true
      csv_exclude_io_text: true
      prompt_caching_prefixes:
        enable: true
        min_frequency: 0.5
      bottleneck_analysis:
        enable_nested_stack: true
      concurrency_spike_analysis:
        enable: true
        spike_threshold: 7
  evaluators:
    accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: local_vllm
    trajectory:
      _type: trajectory
      llm_name: local_vllm
```

```bash
nat eval --config_file eval_config.yml
nat eval --config_file eval_config.yml --override eval.general.max_concurrency 1   # serial baseline
ls .tmp/eval/alto_ops/
#  workflow_output.json  accuracy_output.json  trajectory_accuracy_output.json  config_effective.yml
#  all_requests_profiler_traces.json  inference_optimization.json  standardized_data_all.csv  workflow_profiling_report.txt
```

Profiler options, output files (p90/p95/p99 confidence intervals in `inference_optimization.json`, bottleneck report in `workflow_profiling_report.txt`, per-call token CSV) and the `--override` mechanism are from the profiler and evaluation guides ([NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html), [NAT evaluate](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html)). Using the same local model as judge (`llm_name: local_vllm`) is fine for a smoke test but biased; for a real report use a stronger judge (a Super/Ultra model via NVIDIA endpoints, or a second Spark).

Quick analysis of the CSV:

```python
import pandas as pd
df = pd.read_csv(".tmp/eval/alto_ops/standardized_data_all.csv")
print(df.groupby("llm_name")[["prompt_tokens","completion_tokens"]].describe())
```

### 5.5 Lab 5.3 (L5.3) — sizing: how many Sparks for a hotel portfolio?

```bash
export CONFIG_FILE=eval_config.yml CALC_OUTPUT_DIR=./.tmp/sizing/alto_ops
nat sizing calc --config_file $CONFIG_FILE --calc_output_dir $CALC_OUTPUT_DIR \
  --concurrencies 1,2,3,4,6,8,12,16,24,32 --num_passes 2 \
  --test_gpu_count 1 --target_workflow_runtime 15 --target_users 40
# later, re-fit without re-running:
nat sizing calc --offline_mode --calc_output_dir $CALC_OUTPUT_DIR --test_gpu_count 1 --target_workflow_runtime 10 --target_users 100
```

The calculator runs the workflow at each concurrency, records p95 LLM latency and p95 workflow runtime, fits a linear model, and estimates GPUs needed for the target users; ten or more concurrency values are recommended for a robust fit, the calculator output directory should be separate from evaluation output, and the GPU estimate is explicitly "rough — not for production" ([NAT sizing calculator](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/sizing-calc.html)). Treat one Spark as one GPU; a Spark-class estimate for 40 concurrent GM/engineer users of Alto Ops Claw is exactly the artefact you need for a sovereign-tier quote.

### 5.6 Lab 5.4 (L5.4) — measure the sandbox tax

Run the identical `nat eval` (a) on the host against `http://localhost:8000/v1` and (b) inside the OpenShell sandbox from Lab 3.8 against `https://inference.local/v1`, both at `max_concurrency 1` and `4`. Compare p95 workflow runtime from `inference_optimization.json`. The delta is TLS interception plus policy evaluation plus the extra hop over the veth pair. Publish the number with the OpenShell version (`openshell --version`) — nothing in the sources gives an official overhead figure, so your measurement is the reference.

### 5.7 Lab 5.5 (L5.5) — harness-level benchmark for a claw

For OpenClaw/Hermes claws, there is no NAT profiler. Use the API each harness exposes:

```bash
# Hermes REST API (port 8642) — send 20 identical tasks, time them
for i in $(seq 1 20); do
  /usr/bin/time -f "%e" curl -s -X POST http://localhost:8642/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"hermes","messages":[{"role":"user","content":"Summarise /sandbox/data/chiller_plant.csv in 3 bullets"}]}' >/dev/null
done 2>&1 | sort -n | awk '{a[NR]=$1} END {print "p50",a[int(NR*0.5)],"p95",a[int(NR*0.95)]}'
```

Pair each run with the `inspect_for_inference` count from `openshell term` to get LLM calls per task, and with Phoenix/Langfuse spans (Part 4) for tokens per task. Report the four layers side by side.

### Exercises — Part 5

1. Your Spark shows 64 tok/s single-stream on Nemotron 3 Nano with Ollama but the claw feels slow. Give three non-engine causes and the tool that exposes each.
2. Design an A/B eval between Nemotron 3 Nano (30B-A3B) and Nemotron 3 Super (120B-A12B) for Alto Ops Claw. Which NAT files do you compare and on which columns?
3. Why must the sizing calculator's output directory differ from the eval output directory, and why ≥10 concurrency values?
4. Estimate the aggregate tok/s you could expect from a Spark serving 16 concurrent Alto Ops sessions on Nemotron 3 Nano W4A4, and explain why per-session speed drops.
5. You see a `concurrency_spike_analysis` spike of 9 at t=42 s. What does it mean and what would you change?

**Solutions**

1. (a) Too many LLM round-trips per turn — ReAct loops or retries; see `workflow_profiling_report.txt` and LLM span count in Phoenix. (b) Long prompts — check `prompt_tokens` in `standardized_data_all.csv` and enable `prompt_caching_prefixes` to see repeated prefixes. (c) Slow tools or policy denials with retries — tool spans in Phoenix and `deny` lines in `openshell logs`. Profiler outputs from ([NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html)).
2. Run `nat eval` twice with the same dataset and different `llms.local_vllm.model_name`; compare `accuracy_output.json` (average score), `trajectory_accuracy_output.json`, and `inference_optimization.json` p95 workflow runtime; also completion tokens in the CSV. Expect Super to score higher on agent reliability at ~4× lower tok/s, consistent with Exxact's 17/17 result at 16.4 tok/s ([Exxact benchmark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark)).
3. The calculator writes per-concurrency sub-runs and a fit; mixing them with eval output confuses the offline mode, which re-reads the directory. Ten values give the linear regression enough points for a robust fit ([NAT sizing calculator](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/sizing-calc.html)).
4. Around 700–800 tok/s aggregate (ai-muninn measured 786 at c=16 for W4A4), i.e. ~45–50 tok/s per session — the total is capped by memory bandwidth, so batching raises throughput but each stream gets a smaller share ([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nemotron-3-nano-w4a16-74-toks)).
5. At that moment nine NAT functions were running concurrently, at or above the `spike_threshold` of 7; the profiler surfaces which functions they were ([NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html)). Either the agent fanned out too many tool calls or eval `max_concurrency` is too high for one Spark; lower concurrency or gate parallel tool execution.

---
## Part 6 — Expert: threat model, hardening, custom blueprints, fleet operations

### 6.1 Threat model for a claw

Assets: the sandbox filesystem (`/sandbox`, the agent's own config tree), provider credentials, channel tokens (Telegram/Slack), your CSV/BMS data, and — most valuable — the authority to call tools such as `write_setpoint`. Adversaries: indirect prompt injection via anything the agent reads (web pages, emails, MCP tool results, files), malicious skills/plugins from hubs, a compromised dependency pulled through `npm`/`pypi` presets, and an insider with host access.

NemoClaw's documented limitations shape the model ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)):

| Limitation | Why it matters | Mitigation |
|---|---|---|
| Bypassing managed gateway paths | Policy and inference auth are not enforced for runtimes launched outside the NemoClaw-managed path | Only start agents via the managed entrypoints; never `docker exec` a second agent into the sandbox |
| Same-UID native lifecycle | Supervisor, gateway and agent share the sandbox UID; a same-user agent can signal or imitate peers | OpenShell contains the sandbox; do not put anything in it you would not give the agent |
| Raw filesystem writes bypass scanners | Application-layer scanners see tool calls, not `echo secret > file` | Landlock write scoping; keep secrets out of files |
| Encoded secrets undetected | Regex redaction misses Base64/hex | Use OpenShell credential handles, not file-borne secrets |

Add the OpenClaw CVE classes from Part 0 (TOCTOU file escapes, allowlist env disclosure, MCP loopback privilege escalation) and the CSA analysis of indirect prompt injection ([CSA — OpenClaw indirect prompt injection](https://labs.cloudsecurityalliance.org/research/csa-research-note-openclaw-indirect-prompt-injection-2026061/)): every one of these starts with the model reading attacker-controlled text. Your control is therefore not "a better prompt" but what the process is physically able to do afterwards — which is exactly what OpenShell constrains.

### 6.2 Hardening checklist (Alto sovereign profile)

Each line is a concrete setting with its source.

- **Network:** every endpoint `protocol: rest` (or `mcp`) with `enforcement: enforce`; explicit `rules` rather than `access: full`; no wildcard hosts; `allowed_ips` narrow CIDR for private hosts ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
- **Never add inference-provider hosts** (`api.openai.com`, `integrate.api.nvidia.com`) to the policy; route all inference via OpenShell so credentials stay outside the sandbox and usage is tracked ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).
- **Binaries:** one binary per endpoint; rely on SHA256 TOFU pinning; install tools at image build, not at runtime ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
- **Metadata SSRF:** NemoClaw already injects `AWS_EC2_METADATA_DISABLED=true` into images, gateway processes, cron jobs and shells; OpenShell blocks `169.254.0.0/16` regardless ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices), [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
- **Filesystem:** `landlock.compatibility: hard_requirement` for production; minimal `read_write`; kernel ≥ 6.2 (Landlock ABI 3) ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
- **Kernel/process:** rely on seccomp (mount, pivot_root, bpf, perf_event_open, userfaultfd, kexec, memfd_create blocked; AF_PACKET/AF_BLUETOOTH/AF_VSOCK blocked), `no_new_privs`, `RLIMIT_CORE=0`, non-root user; on Kubernetes use user namespaces (`hostUsers: false`) ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
- **Gateway:** keep `[openshell.gateway] policy_validation_failure_mode = "fail_closed"` (the default) rather than `retain_last_valid`, so an invalid policy can never leave a stale or permissive generation in force ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)); the gateway's policy engine is host-side and out of reach of any sandbox ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
- **Tier:** Restricted for kiosk/always-on claws; Balanced only where `pypi`/`npm` are genuinely needed; never Personal on shared hardware ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).
- **Channels:** treat every channel token as a live outbound path; pair devices explicitly — the retired `NEMOCLAW_DISABLE_DEVICE_AUTH` is ignored by OpenClaw 2026.9.1 ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).
- **Recovery:** snapshot before changes; on suspicion, destroy and recreate from trusted inputs ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).
- **Formal check:** let OpenShell's prover evaluate what a change would newly allow (new host with credentials, new API method) and wait for human approval ([OpenShell on GitHub](https://github.com/NVIDIA/openshell)).

### 6.3 Lab 6.1 (L6.1) — a complete production policy for Alto Ops Claw

```yaml
version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /etc, /app]
  read_write: [/tmp, /sandbox/data/out]
landlock:
  compatibility: hard_requirement
process:
  run_as_user: "1500"
  run_as_group: "1500"

network_policies:
  bms_mcp:
    name: bms_mcp
    endpoints:
      - host: bms.alto.local
        port: 8443
        path: /mcp
        protocol: mcp
        enforcement: enforce
        allowed_ips: ["10.20.0.15/32"]
        mcp: { max_body_bytes: 131072, strict_tool_names: true }
        rules:
          - allow: { method: initialize }
          - allow: { method: notifications/initialized }
          - allow: { method: tools/list }
          - allow: { method: tools/call, tool: { any: [read_point, list_alarms, get_trend] } }
        deny_rules:
          - { method: tools/call, tool: { any: [write_setpoint, override_schedule] } }
    binaries:
      - { path: /usr/bin/python3.12 }
  otel_collector:
    name: otel_collector
    endpoints:
      - host: otel.alto.local
        port: 4318
        protocol: rest
        enforcement: enforce
        rules:
          - allow: { method: POST, path: /v1/traces }
    binaries:
      - { path: /usr/bin/python3.12 }

network_middlewares:
  redact-secrets:
    name: Redact tokens in tool results
    middleware: openshell/regex
    order: 10
    config: { mode: redact }
    on_error: fail_closed
    endpoints:
      include: ["bms.alto.local"]
```

Schema elements (MCP rules, `allowed_ips`, middleware `on_error: fail_closed`, `hard_requirement`, `process`) are from the OpenShell schema and policy guides ([OpenShell policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema), [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)). Apply with `openshell policy set alto-ops --policy prod.yaml --wait`, then prove the deny path: ask the agent to "set chiller 2 setpoint to 6.5°C" and confirm the 403 JSON in `openshell logs` and a failed tool span in Phoenix.

Design note for AltoTech: writes stay out of the sandbox. The write path is a separate, human-approved service (the Alto Copilot approvals flow) that the claw can only request via a read-only ticket tool. That maps the "evidence-gated" pattern you already use in Copilot onto the OpenShell boundary.

### 6.4 Lab 6.2 (L6.2) — custom blueprints and images

- **Custom Dockerfile image:** `nemoclaw onboard --from ./Dockerfile` (or `--from <image>`) bakes your tools in so no runtime egress is needed; NemoClaw documents this on the Deep Agents quickstart and the customisation guide ([NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart), [NemoClaw customize network policy](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/customize-network-policy)).
- **Choice of harness:** Hermes (`nemohermes`, API on 8642, ecosystem plugins such as Langfuse) and Deep Agents (`nemo-deepagents`, Nemotron 3 Ultra default) are first-class alternatives to OpenClaw with the same OpenShell boundary ([NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart), [NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart)).
- **Model Router:** `NEMOCLAW_PROVIDER=routed` with `NVIDIA_INFERENCE_API_KEY` lets NVIDIA's router pick a model per request; `custom`/`anthropicCompatible` cover any OpenAI- or Anthropic-compatible endpoint; `ollama` and `vllm`/`install-vllm` keep inference on the Spark ([NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works)).
- **Podman hosts:** `NEMOCLAW_GATEWAY_RUNTIME=podman` selects Podman for the gateway container ([NemoClaw architecture](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/architecture)).

### 6.5 Lab 6.3 (L6.3) — remote and external gateways

**Remote gateway with the OpenShell CLI:**

```bash
openshell gateway start --remote user@spark-01.alto.local        # provisions on the remote host over SSH
openshell gateway add https://openshell:8080 --remote               # register an existing one
# map the hostname for TLS: add "<ip> openshell" to /etc/hosts on the client
```

The `--remote` forms and the `openshell` hostname requirement are from the playbook and the OpenShell README ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions), [OpenShell on GitHub](https://github.com/NVIDIA/openshell)).

**External gateway owned by infra (experimental):** for a fleet where platform engineering already runs OpenShell (Kubernetes/Helm), NemoClaw's `nemoclaw-blueprint-runner` targets it via `blueprint.yaml`:

```yaml
version: 1.0.0
min_openshell_version: 0.0.116
max_openshell_version: 0.0.116
openshell_target:
  endpoint: https://openshell.alto.local:8443
  workspace: default
  expected_release: 0.0.116
  lifecycle: external
  trust:
    ca_file: /var/run/openshell-target/ca.pem
  authentication:
    credential_file: /var/run/openshell-target/authentication
```

```bash
NEMOCLAW_BLUEPRINT_PATH=/abs/path/blueprint nemoclaw-blueprint-runner plan                     # validates, fingerprints CA, no network
NEMOCLAW_BLUEPRINT_PATH=/abs/path/blueprint nemoclaw-blueprint-runner status --external-target   # one credential-free health request
```

The target must be a bare HTTPS origin with an exact OpenShell release, a PEM-only CA bundle ≤ 1 MiB (regular file, not a symlink) and an absolute authentication file path; the SDK verifies the certificate and hostname against that bundle and uses platform DNS, so you must control the target hostname's resolution ([NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works)).

**Multi-node inference:** NemoClaw's multi-node tab is scoped to DGX Station (multi-node-capable hardware with validated fabric); a Hugging Face token is needed for gated models, and first download and start can exceed an hour ([NemoClaw on DGX Spark overview](https://build.nvidia.com/spark/nemoclaw/overview)). For two Sparks, NVIDIA's blog covers clustering at the vLLM level rather than via NemoClaw ([NVIDIA blog — faster models and multi-node on DGX Spark](https://developer.nvidia.com/blog/run-local-ai-agents-with-faster-models-and-multi-node-clustering-on-nvidia-dgx-spark/)) — verify before promising it to a customer.

**Cloud sandboxes for training or burst:** Brev hosts NemoClaw launchables and OpenShell agent sandboxes ([Brev agent sandboxes](https://docs.nvidia.com/brev/guides/ai-agents/agent-sandboxes)); the DLI course "Securing Agents with OpenShell and NemoClaw" (Modules 1–2 agent loop/ReAct/tools and coordination/retrieval; Module 3 OpenClaw gateway, workspace, skills, scheduled runs on a Brev launchable; Module 4 OpenShell policy boundary and modern CLI agents including Hermes and Deep Agents) is the closest official curriculum to this tutorial ([DLI — NemoClaw course](https://nvdli.github.io/NemoClawDLI/nemoclaw/)). NVIDIA also publishes the NemoClaw docs as an MCP server and `llms.txt` so your claw can read its own manual ([NemoClaw docs MCP server](https://docs.nvidia.com/nemoclaw/_mcp/server)).

### 6.6 Mapping to AltoTech's Alto Copilot tiers

| Alto Copilot tier | Claw pattern | Inference | Posture | Telemetry |
|---|---|---|---|---|
| Cloud | NAT workflows behind Copilot's gateway; optional OpenShell sandbox per tenant on Brev/K8s | NVIDIA endpoints or Model Router | Development/Integration Testing during build; Locked-Down in prod with tenant-scoped MCP entries | Phoenix/OTel per tenant project |
| Sovereign (on-prem) | NemoClaw on DGX Spark/Station in the hotel or HQ; NAT inside OpenShell | local vLLM/Ollama via `inference.local`; no cloud provider hosts | Restricted + custom presets (`alto-bms`, `otel_collector`); `hard_requirement` | OTel collector on-prem; audit trail from `openshell` logs |
| Edge | Restricted OpenClaw/Hermes claw on the Spark or N1x next to the BMS, read-only tools only | local NVFP4 model | Locked-Down; writes only through approvals service | Local file exporter + periodic sync |

Talking points for owners and regulators: data never leaves the property in the sovereign tier because the only inference route is the on-prem provider; every network decision is logged; write authority is structurally absent from the agent.

### Capstone exercise

Build and document "Alto Ops Claw v1":

1. Custom image with NAT + your chiller tool (Lab 3.8), created via `nemoclaw onboard --from` or `openshell sandbox create --from`.
2. Production policy from Lab 6.1, applied with `hard_requirement`; prove the write-deny path.
3. Phoenix + OTel exporters; one correlated request across three planes (Lab 4.5).
4. `nat eval` with 20 questions on Nano and Super; report accuracy, p95 runtime and tokens per task; sizing estimate for 40 users.
5. Sandbox-tax measurement (Lab 5.4).
6. One-page runbook: rebuild, snapshot, rotate credential handles, upgrade OpenShell (`openshell --version`, NemoClaw-pinned 0.0.116 vs latest 0.1.2).

**Marking scheme:** each item 15 points; 10 bonus points for a policy that OpenShell's prover accepts with no new credentialed hosts.

### Exercises — Part 6

1. A vendor asks you to add `api.openai.com:443` to the sandbox policy "so the agent can use GPT for summaries". Respond.
2. Why does `hard_requirement` matter more on a fleet of heterogeneous edge boxes than on your own Spark?
3. Explain the difference in trust boundary between `openshell gateway start --remote` and `nemoclaw-blueprint-runner` with an external target.
4. Your Copilot approvals service must be reachable by the claw to request a write. Write the endpoint entry so the claw can only create tickets, never approve them.
5. List the evidence you would hand to a hotel owner to prove "no data left the building last month".

**Solutions**

1. Decline: adding an inference host directly bypasses credential isolation and usage tracking; instead create an OpenShell provider (`openshell provider create --type openai ...`) and route via `inference.local`, or use `NEMOCLAW_PROVIDER=routed`/`custom` ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)). In the sovereign tier the answer is no regardless — the model must be on-prem.
2. On mixed kernels, `best_effort` silently downgrades Landlock and only emits a finding; `hard_requirement` refuses to start, converting a hidden weakening into a loud failure you will notice during rollout ([OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)).
3. `gateway start --remote` provisions and controls the gateway from your CLI over SSH — you own it. The external-target path connects to a gateway someone else runs, authenticated with a credential file and a pinned CA bundle, at a pinned OpenShell release; you get planning and health only and the platform team owns lifecycle ([NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works), [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions)).
4.
   ```yaml
   approvals:
     endpoints:
       - host: approvals.alto.local
         port: 443
         protocol: rest
         enforcement: enforce
         rules:
           - allow: { method: POST, path: /api/v1/tickets }
           - allow: { method: GET,  path: "/api/v1/tickets/*" }
         deny_rules:
           - { method: "", path: "/api/v1/tickets//approve" }
     binaries:
       - { path: /usr/bin/python3.12 }
   ```
   ([OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies)).
5. `openshell policy get --full` history (`policy list` revisions) showing no external hosts; `openshell inference get` showing the on-prem provider; exported `openshell logs` with zero `allow` decisions to non-local hosts; OTel traces stored on-prem; the sandbox's Landlock findings log empty under `hard_requirement`; and the NemoClaw tier (`Restricted`) in the onboarding record.

---

## Appendix

### A. Cheat sheet

```bash
# NemoClaw
nemoclaw onboard [--express] ; nemoclaw onboard --from <Dockerfile>
nemoclaw <s> status|connect|logs --follow|restart|stop|start|delete|rebuild
nemoclaw <s> policy list|get|add <preset> [--dry-run|--yes]|remove <preset>|add --from-file f.yaml [--trusted-private-host h]
nemoclaw <s> snapshot create --name <n>
NEMOCLAW_PROVIDER=ollama|vllm|install-vllm|build|routed|custom ; NEMOCLAW_POLICY_TIER=restricted|balanced|open|personal
nemohermes ... ; nemo-deepagents ...

# OpenShell
openshell status ; openshell gateway start|stop|destroy|start --remote u@h|add URL --remote
openshell provider create --name X --type openai --credential OPENAI_API_KEY=k --config OPENAI_BASE_URL=http://IP:8000/v1
openshell inference set --provider X --model M [--no-verify] ; openshell inference get
openshell sandbox create --name N --from openclaw|base|sdg|./dir [--policy p.yaml] [--forward P] [--upload src:dst] [--keep] -- CMD
openshell sandbox list|connect|exec -n N -- CMD|upload|download|ssh-config|delete
openshell forward start --background P N ; openshell forward list
openshell policy get N [--base|--full|--rev R] ; policy set N --policy f.yaml --wait ; policy list N
openshell policy update N --add-endpoint host:port[:access[:proto[:enf]]] --binary /path --add-allow 'h:p:METHOD:/glob' --add-deny ... --remove-endpoint ... --dry-run --wait
openshell logs N --tail --source sandbox ; openshell term ; openshell settings get N
OPENSHELL_SANDBOX_POLICY=./p.yaml

# NAT
uv pip install 'nvidia-nat[langchain,mcp,profiler,phoenix,opentelemetry]'
nat workflow create --no-install --workflow-dir ./workflows NAME
nat run --config_file c.yml --input "..." | --input_file f
nat serve --config_file c.yml [--port 8001]          # /v1/workflow, /v1/chat/completions, /v1/workflow/full?filter_steps=
nat mcp serve --config_file c.yml --host 0.0.0.0 --port 9901 --name "..." [--tool_names t] [--transport sse]
nat mcp client tool list --url http://h:9901/mcp [--tool t] ; nat mcp client tool call t --url ... --json-args '{}'
nat eval --config_file eval.yml [--override eval.general.max_concurrency 1]
nat sizing calc --config_file c.yml --calc_output_dir D --concurrencies 1,2,4,8,16,32 --num_passes 2 [--offline_mode]
nat info components -t tracing|logging
```

### B. Troubleshooting

| Symptom | Cause | Fix | Source |
|---|---|---|---|
| `Phase: Unspecified` after onboarding | Wizard not completed inside the sandbox | Finish the wizard via `nemoclaw <s> connect` | [NemoClaw troubleshooting](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/troubleshooting) |
| `failed to verify inference endpoint` on `inference set` | vLLM still loading / compiling | Wait for `/health`, warm with one request, retry (or `--no-verify` once reachable) | [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) |
| Gateway cannot reach vLLM | vLLM bound to 127.0.0.1 or provider URL uses localhost | Bind `0.0.0.0`; use machine IP in `OPENAI_BASE_URL` | [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) |
| `--policy` rejected with `--from openclaw` | Community image bundles its own policy | Drop `--policy`; use `policy update/set` after creation | [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) |
| `api.github.com:443::rest` rejected | L7 protocol without access or rules | Add `:read-only` or `--add-allow` rules | [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) |
| Sandbox `Error` shortly after create | Main process exited | Check `openshell logs`; make CMD long-running | [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) |
| Landlock not applied | Kernel < 6.2 / ABI < 3 | Upgrade kernel; check `docker logs openshell-<s>` for "Applying Landlock filesystem sandbox" | [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices), [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) |
| vLLM crash on Nemotron 3 Nano | Missing `--trust-remote-code` (nemotron_h) | Add the flag | [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) |
| NAT config validation error on `api_key` | Field required even for keyless servers | Set `api_key: EMPTY` | [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) |
| Spark hard-resets under load | Unified memory exhaustion | Lower `--gpu-memory-utilization`, run a memory watchdog, drop page cache between runs | [Exxact engines](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark) |
| Pairing "bypass" env var has no effect | `NEMOCLAW_DISABLE_DEVICE_AUTH` retired in OpenClaw 2026.9.1 | Pair devices properly | [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) |
| Phoenix traces land in `default` project | `otelcollector` exporter sends project as `service.name` | Use the native `phoenix` exporter | [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) |

### C. Glossary

- **Claw** — an always-on personal/operational AI agent (OpenClaw, Hermes, Deep Agents) with tools, memory and channels.
- **NemoClaw** — NVIDIA's alpha blueprint that installs a claw inside an OpenShell sandbox with routed inference and policy presets.
- **OpenShell** — NVIDIA's open-source agent sandbox runtime: gateway + supervisor, Landlock/seccomp static controls, OPA-backed network proxy, credential substitution.
- **Policy tier / preset** — NemoClaw's Restricted/Balanced/Open/Personal baselines and named per-integration endpoint bundles.
- **L4 / L7 endpoint** — host:port:binary check vs request-level inspection (`protocol`, `rules`, `access`, `enforcement`).
- **`inference.local`** — the in-sandbox HTTPS route that OpenShell intercepts and forwards to the configured provider with real credentials.
- **NAT** — NeMo Agent Toolkit: YAML-configured, framework-agnostic agent functions with profiler, evaluators, tracing and MCP.
- **IntermediateStep** — NAT's event unit for LLM/tool/function boundaries consumed by exporters and the profiler.
- **Prover** — OpenShell's formal-verification step that flags what a policy change would newly allow.
- **TOFU** — trust-on-first-use SHA256 pinning of sandbox binaries.

### D. Web Lab Runner (Alto Reef) quick reference

The companion spec `alto_reef_web_runner_spec.md` defines the runner. Summary of what each tutorial part looks like in the web interface:

| Part | Runner surface | Lab ids |
|---|---|---|
| 1 | Sandbox Manager tiles (gateway, inference), Build a Claw, Workbench Chat tab (embedded OpenClaw Control UI on the forwarded 18789 port) | L1.1–L1.7 |
| 2 | Workbench Policies tab (effective policy table, revisions, dry-run/apply, live decision stream), Task Monitor preset toggles, embedded `openshell term` | L2.1–L2.7 |
| 3 | NAT-mode Chat tab (`/v1/chat/completions`, tool-call cards from `/v1/workflow/full`), file editor for custom tools, MCP tool list | L3.1–L3.8 |
| 4 | Traces tab (Phoenix embed / OTel file fallback) and Correlation panel (LLM spans vs `inspect_for_inference`) | L4.1–L4.5 |
| 5 | Bench tab (engine sweep, NAT eval/profiler, sizing, sandbox tax) with provenance on every number | L5.1–L5.5 |
| 6 | Policy editor with schema validation, deny test, custom blueprint onboarding, remote/external gateway forms | L6.1–L6.3, Capstone |

Ports the runner assumes: 4454 UI, 4455 runner API, 8080 OpenShell gateway, 18789 OpenClaw Control UI (forwarded), 8642 Hermes, 8000 vLLM, 8001 NAT REST, 9901 NAT MCP, 6006 Phoenix, 4318 OTel ([OpenClaw Control UI docs](https://docs.openclaw.ai/web/control-ui), [NAT MCP server](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/mcp-server.html), [NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html)).

### E. Open questions and unverified items

- Exact NAT 1.8 import paths for `register_function`/`FunctionBaseConfig` — confirm against the scaffold your version generates.
- Official OpenShell proxy overhead — no published figure; Lab 5.4 produces your own.
- Two-Spark multi-node under NemoClaw — docs scope multi-node to DGX Station; the NVIDIA blog describes vLLM-level clustering on Spark; test before committing.
- Phoenix per-project header routing through an OTel collector — flagged as future work by the only source that tried it.
- All tok/s figures are third-party single-machine measurements on specific engine builds; NVFP4 and engine versions move monthly.
- NemoClaw is alpha; command names and preset lists may change between releases ([NemoClaw on GitHub](https://github.com/NVIDIA/NemoClaw)).
