# ▶ Reef Lab 01 — What is a claw? The stack, the sandbox, and the three harnesses

> Part of Week 26 · NemoClaw on DGX Spark, beginner to expert. You type the commands, you see the real output. Laptop labs (the NAT and OpenShell CLIs, the policy model) run for real everywhere. Spark labs also run in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's docs or playbooks, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn the one-sentence definition of a claw, and the seven parts of the stack you will configure all week.
- See why "the agent runs in a sandbox" is not the same as "the agent is safe".
- Walk the five deny-by-default layers, and sort them into the two you can change live and the two that are locked at creation.
- Compare the three harnesses (OpenClaw, Hermes, Deep Agents) and pick one per job.
- Run lab 01: find out which parts of the stack already exist on this laptop and on your Spark.

**Time** ~30 min · **Difficulty** beginner · **Hardware** none (DRY + laptop) · 1 DGX Spark optional

**Sources:** the course's research tutorial *NemoClaw on DGX Spark — Beginner to Expert* (Part 0), which cites [NVIDIA Build a Claw](https://www.nvidia.com/en-us/ai/build-a-claw/) · [NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works) · [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs the labs and the Reef Lab Runner |
| The Week 26 laptop tools | `week26/.venv-nat/bin/nat --version` and `week26/.venv-openshell/bin/openshell --version` | NAT agents and the OpenShell policy parser run on your laptop in Modules 03–08 |
| A DGX Spark (optional) | `ssh -o BatchMode=yes <spark> true` | Modules 02–03 install and drive a real claw; without one they run DRY |

If the two laptop tools are missing, `week26/README.md` has the two `uv` commands that create them.

```bash
# on: laptop
week26/.venv-nat/bin/nat --version
week26/.venv-openshell/bin/openshell --version
```

**Expected output** (captured on this Mac)

```
nat, version 1.9.0
openshell 0.0.111
```

> 📌 **Versions move fast.** The research tutorial was written against the NAT 1.8 docs; this course installs NAT **1.9.0**. NemoClaw pins OpenShell **0.0.116** on the Spark, and the latest OpenShell release is **0.1.2**. The laptop CLI is **0.0.111**, the last PyPI release that ships a macOS binary, so the course uses it only to *parse* policies offline. When a flag differs on your machine, `--help` wins.

✓ Checkpoint: both commands print a version, or you know which of them you still need to install.

## 1 · A claw, in one sentence

NVIDIA's Build-a-Claw hub describes NemoClaw as an open-source reference stack that deploys with one command. It bundles an agent harness, the NVIDIA OpenShell secure runtime, and Nemotron models. From that, the research tutorial defines a claw as:

> **an always-on, tool-using agent (the harness) + a local or routed model (Nemotron by default) + a kernel-enforced sandbox and policy boundary (OpenShell), assembled by an installer and CLI (NemoClaw).**

Seven parts make up the stack. You will touch every one of them this week.

| Component | Role | Where it runs | Module |
|---|---|---|---|
| **NemoClaw CLI** (`nemoclaw`, `nemohermes`, `nemo-deepagents`) | installer and lifecycle: onboard, status, logs, policy add/remove, snapshot, rebuild, inference set | Spark host | 02 |
| **OpenShell gateway** | control plane: sandbox lifecycle, credentials, policy revisions, inference routes (port 8080) | Spark host (Docker) | 02–03 |
| **OpenShell sandbox + supervisor** | data plane: Landlock, seccomp, a network namespace with a policy proxy, `inference.local` interception | container | 03 |
| **Harness** | OpenClaw (default), Hermes or LangChain Deep Agents — the agent loop, tools, channels | inside the sandbox | 02 |
| **Blueprint** | versioned YAML: image, agent manifest, network policy, inference profile | repo (`nemoclaw-blueprint/`) | 07 |
| **Inference provider** | local vLLM or Ollama on the Spark, or NVIDIA / OpenAI / Anthropic endpoints | Spark host or cloud | 02, 04 |
| **NeMo Agent Toolkit (NAT)** | build, profile, evaluate and serve agent workflows (YAML + Python); MCP client and server | Spark host, inside a sandbox, or your laptop | 04–06 |

```text
 Spark host                                          OpenShell sandbox (container)
┌─────────────────────────────────────┐  manages   ┌──────────────────────────────────────────┐
│ nemoclaw CLI ──► OpenShell gateway ─┼───────────►│ supervisor: Landlock · seccomp · netns    │
│                  :8080  credentials │            │   policy proxy (deny by default)          │
│                  policy revisions   │            │ harness: OpenClaw / Hermes / Deep Agents  │
│ vLLM :8000  /  Ollama :11434  ◄─────┼─ inference │   or a NAT workflow (Module 04)           │
│   (the model — local, on the GB10)  │   .local   │   calls https://inference.local/v1        │
└─────────────────────────────────────┘            └──────────────────────────────────────────┘
```

The flow, from the NemoClaw how-it-works page: the host CLI talks to the gateway, which manages the sandbox. Agents call `https://inference.local`. The gateway injects the real credential and forwards the call to the configured provider, so the agent never holds a provider key.

✓ Checkpoint: you can name which part holds the provider credential (the gateway), and which part you would change to swap OpenClaw for a NAT workflow (the harness).

## 2 · Why the sandbox is the whole point

OpenClaw, the default harness, had a rough security year. The research tutorial cites Cyera's September 2026 disclosure of four vulnerabilities, three of which are exploitable from a single prompt-injection foothold. It also cites the Cloud Security Alliance's "Claw Chain" note on how they chain together.

| CVE (per the research tutorial, citing Cyera) | Class | CVSS |
|---|---|---|
| CVE-2026-44112 | TOCTOU filesystem **write** escape | 9.6 |
| CVE-2026-44115 | execution-allowlist environment-variable disclosure | 8.8 |
| CVE-2026-44118 | MCP loopback privilege escalation | 7.8 |
| CVE-2026-44113 | TOCTOU filesystem **read** escape | 7.7 |

Every one of these starts the same way: the model reads text an attacker controls, such as a web page, an email, a tool result or a file. A better prompt does not fix that. What protects you is **what the process is physically able to do afterwards**, and that is what OpenShell constrains.

NemoClaw's security guide makes the same point. The OpenShell policy and the credential providers are the enforcement boundary. The agent's own mutable config (`/sandbox/.openclaw`, `/sandbox/.hermes`, `/sandbox/.deepagents`) is explicitly **not** trusted as isolation, because the agent can rewrite it.

Keep this mental model for the whole week:

> **The harness is the untrusted thing you are containing. OpenShell is what you are actually configuring. NAT is how you build the agent logic you want to run inside.**

✓ Checkpoint: in one sentence, explain to a hotel GM why "the agent runs in a sandbox" does not mean "the agent is safe" (Exercise 01 asks for this).

## 3 · Five layers, two speeds

NemoClaw's defence in depth has five deny-by-default layers. Four of them are **policy layers** you write in YAML. They change at two different speeds:

| Layer | Enforced by | Policy section | Changeable on a running sandbox? |
|---|---|---|---|
| Filesystem | Landlock LSM (Linux 6.2+, ABI 3) | `filesystem_policy`, `landlock` | **no** — locked at creation |
| Process | seccomp BPF, privilege drop, non-root user | `process` | **no** — locked at creation |
| Network | CONNECT proxy + OPA policy engine, in its own network namespace | `network_policies` | **yes** — `openshell policy update / set` |
| Inference | the supervisor intercepts `inference.local`; the gateway routes it | set with `openshell inference set` | **yes** — hot-reloadable |
| Gateway authentication | the gateway itself (tokens, device pairing) | not in the sandbox policy | — |

The DGX Spark NemoClaw playbook states the split directly: network and inference are hot-reloadable, filesystem and process are locked at sandbox creation.

This split has a practical meaning:

- To **open a new host** for the agent, you update a running sandbox (Module 03, lab 03-3).
- To **give the agent a new writable directory**, you recreate the sandbox. There is no live fix.
- To **switch the model**, you change the inference route. The sandbox keeps running.

Lab 01-2 below walks through one decision per layer, using the course's `policykit` teaching model on the annotated policy from the research tutorial's Lab 2.2.

✓ Checkpoint: without looking, say which two layers are hot-reloadable (network, inference) and which two are locked at creation (filesystem, process).

## 4 · The three harnesses

A harness is the agent loop inside the sandbox. NemoClaw ships three, and all three sit behind the same OpenShell boundary:

| Harness | Default model (NVIDIA Endpoints) | State dir | CLI | Good at |
|---|---|---|---|---|
| **OpenClaw** | `nvidia/nemotron-3-super-120b-a12b` | `/sandbox/.openclaw` | `nemoclaw` | web dashboard on 18789, `openclaw tui`, Telegram / Discord / Slack channels, Brave or Tavily search |
| **Hermes** | Nemotron 3 Super | `/sandbox/.hermes` | `nemohermes` | OpenAI-compatible API on 8642, Tavily, a Langfuse plugin whose keys stay outside the sandbox |
| **LangChain Deep Agents Code** | Nemotron 3 Ultra | `/sandbox/.deepagents` | `nemo-deepagents` | a planner that spawns sub-agents, coding work |

On a DGX Spark you will usually not use the NVIDIA Endpoints defaults. Module 02 routes inference to a **local** vLLM or Ollama model, so prompts and data stay on the device.

The fourth option this week is **your own claw**. A NAT workflow you write in Module 04 runs inside the same kind of sandbox and calls the same `inference.local`.

✓ Checkpoint: you can pick a harness for (a) a Telegram concierge bot, (b) a refactoring agent on a private repo, (c) an agent whose traces go to Langfuse with no raw keys in the sandbox.

## 5 · The course map, and the running example

| Module | Research tutorial part | You build |
|---|---|---|
| 01 · this one | Part 0 | the mental model |
| 02 · first claw | Part 1 | a verified Spark, a NemoClaw install, proof that inference is local |
| 03 · policy as code | Part 2 | policies you can read, write, iterate on and roll back; bring-your-own vLLM |
| 04 · NAT claws | Part 3 | **Alto Ops Claw**: a NAT agent with a chiller-plant tool, served over REST and MCP, inside a sandbox |
| 05 · tracing | Part 4 | agent traces, policy logs and harness logs for one request, correlated |
| 06 · benchmarking | Part 5 | engine, workflow, quality and sandbox-overhead numbers you measured yourself |
| 07 · hardening | Part 6 | a threat model, a production policy, custom blueprints, remote gateways |
| 08 · capstone | Capstone | Alto Ops Claw v1, documented and graded |

The running example is **Alto Ops Claw**: a hotel-operations assistant that reads chiller-plant CSV exports, answers energy questions, and calls a (mock) BMS over MCP. Its data lives in `week26/common/data/chiller_plant.csv`. The data is **synthetic**: seven days of 15-minute rows, with a degraded last six hours so the alarm path has something to find.

> 🪸 **The Reef.** The runner's **🪸 Reef** button shows the claw stack at a glance: the gateway as a lighthouse, one enclosure per sandbox, and the services as landmarks. Every value comes from a real command (`openshell status`, `openshell sandbox list`, `openshell inference get`) or an HTTP probe, shown with its exit code and time. With no Spark connected, the island stays empty and says why. It never shows invented state. The full visual app it grew from is in `week26/alto-reef/`.

✓ Checkpoint: you opened **🪸 Reef** once and can say what it shows when no Spark is connected.

## Labs — run them here

**labs/lab01_1_claw_map.py** — Which parts of the claw stack exist on this laptop and on your Spark, each found with a real read-only command.

**labs/lab01_2_five_layers.py** — One allow/deny decision per policy layer on the research tutorial's annotated policy, sorted into hot-reloadable and locked.

## Try it yourself

`exercises/ex01_claw_basics.py` has three TODOs:

1. Sort the four policy layers into `HOT` (changeable on a running sandbox) and `LOCKED` (fixed at creation).
2. Pick a harness for each of the three jobs in Section 4.
3. Write your one-paragraph "sandboxed is not safe" answer for a hotel GM. The checker looks for the two ideas it must contain.

```bash
# on: laptop
.venv/bin/python week26/01_what_is_a_claw/exercises/ex01_claw_basics.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ layers: network + inference are hot-reloadable · filesystem + process are locked at creation
✓ harnesses: concierge → openclaw · refactor → deepagents · langfuse → hermes
✓ your GM paragraph says what the sandbox limits AND what it does not fix
```

<details><summary>Hint — the GM paragraph</summary>

A sandbox limits the **blast radius**: which files, hosts, system calls and credentials a confused or compromised agent can touch. It does **not** make the agent's reasoning correct, and it does not make the agent immune to prompt injection. Say both halves.

</details>

<details><summary>Hint — the Langfuse job</summary>

Look for the harness whose observability plugin gets its keys as OpenShell credential placeholders (`langfuse-hermes-v1`) rather than as files inside the sandbox.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `week26/.venv-nat/bin/nat: No such file or directory` | the NAT venv was not created | run the two `uv` lines in `week26/README.md` |
| `openshell --version` prints nothing on the laptop | a newer `openshell` wheel (0.0.116 / 0.1.2) is the Python SDK only, with no CLI | install `openshell==0.0.111` into `week26/.venv-openshell` |
| 🪸 Reef says "no Spark configured" | `SPARK_HOST` is empty | open 🖥 Spark setup, or keep going in DRY mode |
| Lab 01-1 shows `◈ EXAMPLE` for every Spark row | DRY mode: nothing ran on the Spark | connect a Spark and switch to ⚡ Live |

## Next

[Lab 02 — Your first claw on DGX Spark](../02_first_claw/TUTORIAL.md): verify the Spark, install NemoClaw with one command, onboard a sandboxed assistant on a local model, and prove that inference never leaves the box.
