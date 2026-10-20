# ▶ Spark Lab 16 — NemoClaw: always-on sandboxed agents

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn who does what: **OpenClaw** is the agent, **OpenShell** is the sandbox, **NemoClaw** is the installer and CLI that puts one inside the other and wires it to local vLLM.
- Run a ten-point **preflight** on your Spark, then install NemoClaw and onboard your first sandboxed agent (`my-assistant`).
- Reach the agent's Web UI through an SSH tunnel, and inspect its status, policy and logs without changing anything.
- Build a **least-privilege network preset** on your laptop, validate it against every rule the playbook documents, and apply it only when you choose to.
- Check that the model endpoint answers the way an agent needs: a model list, a plain reply, and a well-formed **tool call**.
- Set up three of NVIDIA's example agents (developer, deck reviewer, news digest) with their exact policy steps.

**Time** ~60 min · **Difficulty** intermediate · **Hardware** 1 Spark (or none: DRY mode + laptop labs)

**Official playbooks covered:** [Run NemoClaw with a Local LLM](https://build.nvidia.com/spark/nemoclaw) · [Set Up Example NemoClaw Agents](https://build.nvidia.com/spark/nemoclaw-applications)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done: `ssh spark-a` works with a key | `ssh -o BatchMode=yes spark-a true` | labs 16-1 and 16-2 run over SSH |
| A **clean** Spark: no personal data, no real accounts, no production credentials | you decide | the playbook's first safety rule: "Use only a clean environment" |
| `sudo` on the Spark (interactive is fine) | `sudo -v` on the Spark | the installer needs root for some steps |
| Docker 28.x+ without sudo | `docker ps` on the Spark | NemoClaw runs the gateway and sandbox as containers |
| Disk for the model | `df -h ~` on the Spark | the playbook warns large Express models can need **hundreds of GB** |
| This repo's Python + laptop Ollama (optional) | `.venv/bin/python --version`, `curl -s localhost:11434/api/tags` | labs 16-3 and 16-4 run on the laptop |

Useful background: [Module 05](../05_vllm/TUTORIAL.md) (vLLM and tool calling) and [Module 15](../15_openshell_sandbox/TUTORIAL.md) (OpenShell on its own). Week 23 Lab 06 explained NemoClaw's ideas with a simulated policy gate; this module is the real install on a Spark.

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
curl -s localhost:11434/api/tags | head -c 120; echo
```

> 🔐 **Security first, all module long.** An always-on agent with tools can read files, run commands and send messages while you are not watching. The playbook lists four risks: **data leakage, malicious code execution, unintended actions, prompt injection**. The sandbox reduces them; it does not remove them. In this module no lab installs a messaging integration, grants a permission, or changes a policy unless you pass `--apply`. Never paste a real token into chat, a prompt, or a command line.

✓ Checkpoint: you have a Spark you are willing to treat as a lab machine, and `ssh -o BatchMode=yes spark-a true` returns without a prompt (or you have decided to follow along DRY).

## 1 · Who does what: OpenClaw, OpenShell, NemoClaw

Three names, three jobs. The playbook describes NemoClaw as "an open-source reference stack that simplifies running OpenClaw always-on assistants more safely".

| Piece | What it is | You meet it as |
|---|---|---|
| **OpenClaw** | the agent: chat, memory, tools, skills, a built-in scheduler (cron) | the Web UI on `127.0.0.1:18789`, `openclaw tui`, `openclaw cron …` inside the sandbox |
| **OpenShell** | the runtime that sandboxes agents: filesystem, network, process and inference isolation | `openshell policy get`, `openshell forward …`, `openshell term` on the host |
| **NemoClaw** | the installer (`nemoclaw.sh`), the onboard wizard and the `nemoclaw` CLI that create the sandbox, run OpenClaw in it, and route its model calls to local vLLM | `nemoclaw onboard`, `nemoclaw <name> status`, `policy-add`, `logs`, `uninstall` |

NemoClaw can also run two other agents in the same kind of sandbox: **Hermes** (`NEMOCLAW_AGENT=hermes` or `nemohermes onboard`) and **LangChain Deep Agents Code**. This module uses the default, OpenClaw. Module 17 runs OpenClaw and Hermes *without* a sandbox, so you can compare.

OpenShell enforces four isolation layers. Two are fixed when the sandbox is created; two can change while it runs:

| Layer | What it protects | When it applies |
|---|---|---|
| Filesystem | reads/writes outside allowed paths | locked at sandbox creation (change = `nemoclaw <name> rebuild`) |
| Network | unauthorized outbound connections | hot-reloadable at runtime (`policy-add` / `policy-remove`) |
| Process | privilege escalation and dangerous syscalls | locked at sandbox creation |
| Inference | reroutes model API calls to controlled backends | hot-reloadable at runtime |

(REFERENCE — the table is quoted from the playbook.) The inference layer is why the agent never sees your vLLM port directly: inside the sandbox the model lives at **`inference.local`**.

✓ Checkpoint: you can say which of the three pieces you would blame if (a) the agent answers badly, (b) the agent reaches a website it should not, (c) `nemoclaw` is "command not found".

## 2 · Preflight: lab 16-1

The playbook asks you to verify three things before starting:

```bash
# on: spark
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
```

"Expected: Ubuntu 24.04 (or your platform's supported OS), a detected NVIDIA GPU, Docker 28.x+." (REFERENCE — quoted from the playbook.) The playbook's starter prompt goes further: it wants Node.js, disk, existing NemoClaw, vLLM, relevant ports and administrator access checked too. Lab 16-1 runs all ten checks, read-only, and tells you which onboarding path fits:

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_1_preflight.py
```

**Expected output** (DRY mode, captured on this Mac; with a Spark you get your own values and ✓ / ✕)

```
│ check                              result     last line of output
│ ─────────────────────────────────  ─────────  ──────────────────────────────────────────────────
│ Operating system                   ◈ example  NAME="Ubuntu"
│ GPU visible                        ◈ example  NVIDIA GB10, 580.95.05
│ Docker 28.x+                       ◈ example  28.3.3
│ Node.js                            ◈ example  node not installed
│ Unified memory                     ◈ example  Mem:             119           9         102
│ Free disk                          ◈ example  /dev/nvme0n1p2  3.7T  412G  3.1T  12% /
│ Non-interactive sudo               ◈ example  passwordless sudo: no — the installer will ask for
│ Existing NemoClaw / OpenShell      ◈ example  (end of list)
│ Ports 8000 · 8080 · 18789 · 18790  ◈ example  none of 8000 8080 18789 18790 is listening
│ vLLM already on :8000              ◈ example  no server on :8000

▣ STEP 11 · which onboarding path fits this Spark?
◆ DRY: with a Spark you would see whether :8000 is free (Express Install) or already serving (Existing vLLM).
```

Why these ports: **8000** is vLLM (if something already serves there, onboarding can reuse it as "Existing vLLM"), **8080** is the OpenShell gateway (a stale container holding it is a known failure), **18789/18790** is the OpenClaw dashboard.

> 💡 Node.js missing is fine: the installer installs Node.js **22.16+** if needed. A Node older than 22.16 that the installer cannot upgrade is a failure (see Troubleshooting).

✓ Checkpoint: in LIVE mode every row is ✓ (or you know the fix for each ✕), and you know whether you will take Express Install or reuse an existing vLLM.

## 3 · Install NemoClaw

The playbook's install is one command, run **on the Spark**. It installs Node.js if needed, installs OpenShell, clones the last known good (LKG) NemoClaw release, builds the CLI, and creates a sandbox:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

Piping a script into `bash` runs code you have not read. The playbook's own starter prompt says to "download and inspect the official installer" when in doubt. This course recommends that variant; it is the same script, just saved first:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh -o ~/nemoclaw.sh
less ~/nemoclaw.sh            # read what it will do; q to quit
bash ~/nemoclaw.sh
```

What happens next:

1. You are shown a **third-party software notice**. Read it; accepting makes you responsible for those components.
2. On a DGX Spark the installer may offer **Express Install**: managed local vLLM, a maintained Express model, sandbox name `my-assistant`, and the **Balanced** policy. Press **Enter** (or `Y`) to accept, or `n` for custom onboarding (Section 4).
3. The model download starts. It can be large; this is the long part.

If `nemoclaw` is "command not found" afterwards, reload your shell:

```bash
# on: spark
source ~/.bashrc
nemoclaw list
```

To skip the Express prompt entirely, set `NEMOCLAW_NO_EXPRESS=1` before running the installer. Setting `NEMOCLAW_PROVIDER` also bypasses Express and uses that provider.

✓ Checkpoint: `nemoclaw list` runs on the Spark and shows `my-assistant` (Express), or the installer is waiting at the onboard wizard (custom).

## 4 · Onboard: Express or custom

If you accepted Express Install, onboarding already ran; skip to Section 5. For a custom setup (or later, `nemoclaw onboard` to change things), the wizard asks nine questions:

| # | Prompt | This course's choice | Why |
|---|---|---|---|
| 1 | Select your agent | `1` OpenClaw | the default; Module 17 covers Hermes |
| 2 | Configuring inference | a local option | keep data on the Spark |
| 3 | Inference models | the installer's pick, or your running vLLM model | must support tool calling (Section 7) |
| 4 | Sandbox name | `my-assistant` | every command below uses it |
| 5 | Apply this configuration | `Y` | |
| 6 | Enable Brave Web Search | **No** for now | it adds an API key and egress; add it later with a rebuild |
| 7 | Messaging channels | **No** | the playbook: "Prefer No for a first local install" |
| 8 | Resource profiles | Enter (`No profile`) | |
| 9 | Policy presets | **Balanced**, accept the suggested presets | the recommended tier |

The playbook's starter prompt documents environment variables for scripted (non-interactive) runs. The useful ones for a Spark:

| Variable | Meaning |
|---|---|
| `NEMOCLAW_PROVIDER=vllm` | use an **existing** vLLM server on `localhost:8000` |
| `NEMOCLAW_PROVIDER=install-vllm` | NemoClaw installs and manages vLLM |
| `NEMOCLAW_PROVIDER=ollama` (+ optional `NEMOCLAW_MODEL`) | local Ollama |
| `NEMOCLAW_AGENT=hermes` | Hermes instead of OpenClaw |
| `NEMOCLAW_NO_EXPRESS=1` | skip the Express prompt |
| `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1`, `NEMOCLAW_YES=1` | pre-accept, **only after** you have read what you accept |

When onboarding finishes you see this summary:

**Expected output** (REFERENCE — quoted from the playbook)

```text
 ──────────────────────────────────────────────────
  OpenClaw is ready

  Sandbox:  my-assistant
  Model:    <your-selected-model> (Local vLLM)

  Start chatting

    Browser:
      http://127.0.0.1:18789/

    Terminal:
      nemoclaw my-assistant connect
      then run: openclaw tui

  Authenticated dashboard URL, if needed:
    nemoclaw my-assistant dashboard-url --quiet

  Remote access (SSH session detected):
    On your workstation, run:
      ssh -L 18789:127.0.0.1:18789 lab@<host>
    Then open the dashboard URL above in your local browser.

  Manage later

    Status:      nemoclaw my-assistant status
    Logs:        nemoclaw my-assistant logs --follow
    Model:       nemoclaw inference set --model <model> --provider <provider> --sandbox my-assistant
    Policies:    nemoclaw my-assistant policy-add
    Credentials: nemoclaw credentials reset <KEY> && nemoclaw onboard
  ──────────────────────────────────────────────────
```

You can run onboarding again to add more sandboxes. Three flags look similar and are not:

| Flag | Effect | Use it when |
|---|---|---|
| `--name <new-name>` | a **new** sandbox next to the existing ones | a second agent for a different job |
| `--recreate-sandbox` | rebuilds an existing sandbox to bake in a feature (e.g. Brave Search) | adding a feature |
| `--fresh` | **destroys and recreates** a same-named sandbox and discards wizard state | only to recover from a failed onboarding |

```bash
# on: spark
nemoclaw onboard --gpu --name <new-name>
```

✓ Checkpoint: you have seen "OpenClaw is ready", and `nemoclaw my-assistant status` reports the sandbox.

## 5 · Talk to your agent: Web UI and terminal

**Web UI.** The dashboard URL contains a login token. Print it on the Spark:

```bash
# on: spark
nemoclaw my-assistant dashboard-url --quiet
```

It prints a URL like `http://127.0.0.1:18790/#token=<token>` (REFERENCE — the playbook's example). The port is auto-assigned, commonly 18789 or 18790. **Treat the whole URL as a password**: do not paste it into chat or a ticket.

From your laptop, forward that port (use the one from your URL) and open the URL in your browser:

```bash
# on: laptop
ssh -N -L 18789:127.0.0.1:18789 spark-a
```

> ⚠ Use `127.0.0.1`, not `localhost`, in the browser. The gateway's origin check needs an exact match, otherwise you get `origin not allowed`.

**Terminal UI.** Connect into the sandbox and start the TUI; **Ctrl+C** leaves the TUI, `exit` leaves the sandbox:

```bash
# on: spark
nemoclaw my-assistant connect
openclaw tui
```

Try one harmless first prompt, such as "What model are you running on, and which tools can you use?".

> 💡 The playbook also documents `nemoclaw tunnel start`, a cloudflared tunnel that gives the Web UI a **public** URL. This course does not use it: an SSH tunnel keeps the dashboard private to you.

✓ Checkpoint: the agent answers you in the Web UI (through the tunnel) or in `openclaw tui`.

## 6 · The sandbox and its policies

The network layer is an **allow-list**. A tier (Balanced, Restricted, Open) is chosen at onboarding, and **presets** add groups of hosts on top. Five commands cover it:

```bash
# on: spark
nemoclaw my-assistant policy-list                                # which presets are applied
openshell policy get my-assistant --full | grep -E "host:|port:" # every host the sandbox may reach
nemoclaw my-assistant policy-add                                 # interactive: add a maintained preset
nemoclaw my-assistant policy-add --from-file ./my-preset.yaml --yes   # add your own preset (hot-reload)
nemoclaw my-assistant policy-remove <preset> --yes                # take one away (hot-reload)
```

Lab 16-2 runs the read-only half of this, plus the two **boundary tests** the applications playbook uses: the public internet must be refused, and `inference.local` must answer. It masks the dashboard token and anything token-shaped in the logs before printing.

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_2_status_and_logs.py
```

**Expected output** (DRY mode, captured on this Mac; the 403 line is the playbook's, the rest are EXAMPLE shapes)

```
▣ STEP 7 · boundary test 1 — the public internet is refused
$ nemoclaw my-assistant exec -- bash -lc 'curl -sS --max-time 5 https://example.com' 2>&1 | head -3   [DRY]
◈ REFERENCE — expected output from the NVIDIA playbook (not your machine):
curl: (56) CONNECT tunnel failed, response 403

▣ STEP 8 · boundary test 2 — the model route is allowed
$ nemoclaw my-assistant exec -- bash -lc 'curl -sf https://inference.local/v1/models' | head -c 300   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
{"object":"list","data":[{"id":"<your-selected-model>","object":"model"}]}

│ check                 source     what it shows
│ ────────────────────  ─────────  ────────────────────────────────────
│ nemoclaw list         example    — (dry: shape only)
│ status                example    — (dry: shape only)
│ policy-list           example    — (dry: shape only)
│ policy get --full     example    — (dry: shape only)
│ dashboard-url         reference  token masked: never paste it in chat
│ logs (tail 40)        example    — (dry: shape only)
│ curl example.com      reference  — (dry: shape only)
│ curl inference.local  example    — (dry: shape only)
```

If `example.com` is **not** refused, the sandbox has more egress than you think: run `policy-list` and remove what you do not need.

**Writing your own preset.** The News Digest agent (Section 8) needs to read three news sites. The playbook's preset file has a precise shape, and it lists the mistakes that make it fail:

| Mistake | What you see |
|---|---|
| `network_policies` written as a list of `{host, port}` | `invalid type: sequence, expected a map` |
| `preset.name` with an underscore (`news_sources`) | `Preset must declare preset.name (lowercase, hyphenated RFC 1123 label)` |
| an endpoint with only `host` and `port` (no `access: full` + `tls: skip`) | `curl: (56) CONNECT tunnel failed, response 403` |
| no `binaries` list naming which programs may use the egress | every fetch returns 403 |

(REFERENCE — the error texts are quoted from the applications playbook. One detail is inconsistent in the playbook: one note says the group key must also be hyphenated, the Troubleshooting table says the group key accepts underscores. Use hyphens everywhere and both are satisfied.)

Lab 16-3 generates the file on your laptop, validates it, and proves the validator catches each mistake:

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_3_policy_preset.py
```

**Expected output** (captured on this Mac, no Spark configured)

```
▣ STEP 1 · generate news-sources.yaml (the playbook's shape)
preset:
  name: news-sources
  description: "Daily news digest source allowlist"

network_policies:
  news-sources:
    name: news-sources
    endpoints:
      - host: developer.nvidia.com
        port: 443
        access: full
        tls: skip
      - host: blogs.nvidia.com
        port: 443
        access: full
        tls: skip
      - host: news.ycombinator.com
        port: 443
        access: full
        tls: skip
    binaries:
      - { path: /usr/local/bin/openclaw }
      - { path: /usr/local/bin/node }
      - { path: /usr/bin/node }
      - { path: /usr/bin/curl }
◆ written to week25/16_nemoclaw/.runs/news-sources.yaml

▣ STEP 2 · parse it back and validate
✓ no findings: valid shape, least-privilege checks pass

▣ STEP 3 · the validator must catch every documented mistake (rows 1-4: playbook · 5-6: this course)
│ broken preset                       validator  first finding
│ ──────────────────────────────────  ─────────  ────────────────────────────────────────────────────
│ underscore in preset.name           ✓ caught   preset.name 'news_sources': must be a lowercase, hy…
│ list instead of a map               ✓ caught   network_policies is a list: 'invalid type: sequence…
│ bare {host, port} (no access mode)  ✓ caught   blogs.nvidia.com: no access mode → the proxy answer…
│ no binaries allow-list              ✓ caught   group 'news-sources' has no binaries allow-list → n…
│ a token pasted into the file        ✓ caught   a credential-shaped string is in the file — never p…
│ wildcard host + a shell binary      ✓ caught   *.example.com: wildcard host — name each site inste…

▣ STEP 4 · put it on the Spark and (only with --apply) add it to the sandbox
$ scp news-sources.yaml <spark>:~/w25/nemoclaw/news-sources.yaml   [DRY]
→ not applied (DRY — no Spark). On the Spark, run:
$ nemoclaw my-assistant policy-add --from-file ~/w25/nemoclaw/news-sources.yaml --yes
$ openshell policy get my-assistant --full | grep -E "host:|port:"
◆ Undo later with: nemoclaw <sandbox> policy-remove news-sources --yes  (network changes hot-reload, no rebuild)

✓ news-sources.yaml is valid and least-privilege
✓ every documented mistake was caught
```

The `binaries` list is part of least privilege: it names **which programs** may use the tunnel. The playbook includes `/usr/bin/curl` so shell-based fetches work; the lab warns if you add a shell such as `/bin/bash`, because then any script in the sandbox could use that egress.

**Filesystem changes are different.** Network presets hot-reload. The filesystem policy is locked at creation: to make a directory read-only at the kernel level you add it to `read_only` in the sandbox's `filesystem_policy` and run `nemoclaw my-assistant rebuild` (workspace state is preserved). To approve or deny a blocked network request live, use `openshell term` on the host.

✓ Checkpoint: lab 16-3 ends with two ✓ lines, and you can explain why a bare `{host, port}` entry "applies cleanly" but still gets a 403.

## 7 · Check the model route: lab 16-4

An agent is only as good as its model's **tool calls**. If the model never returns `tool_calls`, OpenClaw's tools and skills never fire and you have a chatbot. Lab 16-4 checks the endpoint NemoClaw routes to: vLLM on the Spark's `:8000` when it answers, otherwise Ollama on your laptop as a labelled stand-in. It uses the playbook's own smoke-test prompt, "Reply with exactly: READY", then one tool.

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_4_inference_route.py
```

**Expected output** (captured on this Mac: LAPTOP STAND-IN, laptop speed, not Spark numbers)

```
◆ endpoint: http://localhost:11434/v1 · Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark

▣ STEP 1 · GET /v1/models
→ GET http://localhost:11434/v1/models
· nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b, kimi-k2.7-code:cloud, gemma4:12b, gemma4:latest …

▣ STEP 2 · a plain answer (the playbook's smoke test prompt)
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  READY
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 9169 ms · 2 tok in 9.2s · 77.0 tok/s

▣ STEP 3 · one tool, one question — does the model emit a tool call?
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER
→ tool_call get_sandbox_status({"sandbox":"my-assistant"})
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 17379 ms · 19 tok in 17.4s · 1.1 tok/s

│ check          result  detail
│ ─────────────  ──────  ──────────────────────────────────────────────
│ models listed  ✓       7 model(s)
│ plain answer   ✓       READY
│ tool call      ✓       get_sandbox_status({"sandbox":"my-assistant"})

✓ this endpoint can drive an agent (LAPTOP STAND-IN)
```

On the Spark with a vLLM server up, the same lab reports `vLLM on your Spark (:8000)`. If you run your own vLLM for NemoClaw ("Existing vLLM", `NEMOCLAW_PROVIDER=vllm`), start it with tool calling enabled: for Qwen3.6 the vLLM playbook uses `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` (Module 05 and Module 17 Section 2 show the full command). Express Install's managed vLLM handles this for you. Inside the sandbox the playbook's check is `curl -sf https://inference.local/v1/models`.

✓ Checkpoint: all three rows are ✓, and you can say why the laptop's TTFT and tok/s tell you nothing about the Spark.

## 8 · Three example agents from the applications playbook

NVIDIA's companion playbook gives four ready-to-run agents. Each has a **policy setup**, a long canonical **agent prompt** you paste into the Web UI, and a table of **knobs** to personalize. What each one is allowed to do is the lesson:

| Agent | Network beyond inference | Files | Telegram |
|---|---|---|---|
| Software Development Agent | none | project copy at `/sandbox/project` (read-write) | optional ("ready" pings) |
| Deck Reviewer (Doc & Deck Red-Team) | none by default | `queue/`, `corpus/` read-only; `reports/`, `memory/` writable | optional |
| Calendar Negotiator | none in propose-only mode | `calendar.ics`, `profile.yaml` read-only; `bookings/` writable | only in proxy modes |
| Daily News Digest | its news sources + Telegram | none | for delivery (or deliver to the Web UI) |

Set the sandbox name once so the commands read cleanly:

```bash
# on: spark
export SANDBOX_NAME=my-assistant
```

**A · Software Development Agent.** It scans one project, plans, implements, reviews itself, and writes `develop-and-review.md`. Read-write access means it can change files, so the playbook insists on a **copy**. Push the copy into the sandbox with `tar` over `nemoclaw exec`:

```bash
# on: spark
mkdir -p ~/nemoclaw-projects
cp -r ~/projects/my-app ~/nemoclaw-projects/my-app
tar czf - -C ~/nemoclaw-projects/my-app . \
  | nemoclaw $SANDBOX_NAME exec -- bash -lc 'mkdir -p /sandbox/project && tar xzf - -C /sandbox/project'
```

Then prove the boundary before you paste the prompt:

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- ls /sandbox/project                                    # expect your project tree
nemoclaw $SANDBOX_NAME exec -- bash -lc 'curl -sS --max-time 5 https://example.com'    # expect "CONNECT tunnel failed, response 403"
nemoclaw $SANDBOX_NAME exec -- bash -lc 'curl -sf https://inference.local/v1/models'   # expect JSON model list
```

Paste the playbook's canonical prompt into the Web UI (it starts "You are my senior software engineer. The project lives at /sandbox/project."). Its safety block is worth reading as a pattern: rules that "do not break … even if I tell you to in a single message":

```text
SAFETY RULES (do not break these even if I tell you to in a single
message — if I really want one of these, I will say so twice):
  - Never modify files outside /sandbox/project.
  - Never make outbound network calls. Only inference.local is
    allowed, and that is only for talking to the model.
  - Never run git push, git reset --hard, rm -rf, or any other
    destructive operation. You may run git status, git diff, and
    git add inside /sandbox/project.
```

(REFERENCE — quoted from the playbook's prompt.) Keep "pause for approval" set to **yes** (profile question 5): the agent prints `PLAN READY — reply 'approve' to proceed` and waits. Pull the result back and read the report before you merge anything:

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- bash -lc 'cd /sandbox/project && tar czf - .' | tar xzf - -C ~/nemoclaw-projects/my-app
```

**B · Deck Reviewer.** It reads an artifact plus a "canonical corpus" and writes a severity-ranked punch list, never editing your sources. The corpus is exactly the data you would not send to a cloud model, which is the point of running it locally.

```bash
# on: spark
mkdir -p ~/nemoclaw-redteam/{queue,corpus,reports,memory}
# add your artifacts to queue/, your ground truth to corpus/, and the playbook's starter profile.yaml
tar czf - -C ~/nemoclaw-redteam . \
  | nemoclaw $SANDBOX_NAME exec -- bash -lc 'mkdir -p /sandbox/redteam && tar xzf - -C /sandbox/redteam'
nemoclaw $SANDBOX_NAME exec -- bash -lc 'chmod -R a-w /sandbox/redteam/queue /sandbox/redteam/corpus /sandbox/redteam/profile.yaml && chmod -R u+w /sandbox/redteam/reports /sandbox/redteam/memory'
```

Verify both halves of the boundary: writes to `reports/` work, writes to `queue/` fail.

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/redteam/reports/.write-check && rm /sandbox/redteam/reports/.write-check && echo OK reports'
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/redteam/queue/.write-check 2>&1 | head -1'   # expect "Permission denied"
nemoclaw $SANDBOX_NAME exec -- bash -c 'curl -sS --max-time 5 https://example.com'                        # expect "CONNECT tunnel failed, response 403"
```

> ⚠ The playbook is explicit that this `chmod` is a **soft** boundary: the agent runs as the `sandbox` user, which owns the files and could `chmod` them back. For a kernel-enforced boundary, add the paths to `read_only` in `filesystem_policy` and `nemoclaw $SANDBOX_NAME rebuild`.

Pull the punch lists back with `nemoclaw $SANDBOX_NAME exec -- bash -lc 'cd /sandbox/redteam && tar czf - reports memory' | tar xzf - -C ~/nemoclaw-redteam`.

**C · Daily News Digest.** A scheduled briefing. It is the only one of the three that needs internet egress, which is why you built and validated its preset in lab 16-3. Copy the file from the lab (or write it by hand), apply it, and check:

```bash
# on: spark
nemoclaw $SANDBOX_NAME policy-add --from-file ~/w25/nemoclaw/news-sources.yaml --yes
openshell policy get $SANDBOX_NAME --full | grep -E "host:|port:"
```

Paste the playbook's prompt (it starts "You are my personal news intelligence analyst."). Without Telegram, replace its delivery line with `Deliver each briefing to the web UI (this session). Do not use any messaging channel.`

The playbook warns that the agent's own scheduler call can be rejected for missing scope, and recommends registering the job **from the operator side**:

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- openclaw cron add \
  --name news-digest --cron "0 8 * * 1-5" --tz America/Los_Angeles \
  --agent default --session-key agent:default:news-digest \
  --message "Run my daily news briefing now and write it to this session." \
  --no-deliver --token ""
nemoclaw $SANDBOX_NAME exec -- openclaw cron list
```

> ⚠ Known limitation (from the playbook): **scheduled** runs on a local model can be `skipped` because a pre-flight DNS lookup of `inference.local` fails (`getaddrinfo EAI_AGAIN inference.local`). Live turns are unaffected. The playbook's workaround is to pass a DNS-resolvable endpoint (`host.openshell.internal:8000`, allowed by the `local-inference` preset) to `cron add` with `--model`. Test with a one-off first: "Run the digest task now as a one-off, then keep the schedule for tomorrow."

The fourth agent, **Calendar Negotiator**, follows the Deck Reviewer pattern (`~/nemoclaw-calendar/` with `calendar.ics` read-only and `bookings/` writable); run it in **propose-only** mode so it never sends a message itself.

✓ Checkpoint: for each agent you set up, the `example.com` test is refused and the write test matches the table above.

## 9 · Optional: a messaging channel (Telegram)

Skip this unless you need the agent on your phone. A channel is a second way in: anyone who can message the bot can drive your agent, so treat it as a permission you grant.

1. Create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`). The **bot token** it returns is a credential.
2. Register the channel. NemoClaw prompts for the token (do not put it on the command line) and rebuilds the sandbox:

```bash
# on: spark
nemoclaw my-assistant channels add telegram
```

3. The wizard asks for an optional **Telegram User ID**. Give yours (send `/start` to `@userinfobot` to find it). If you skip it, the bot requires device pairing before it responds.
4. If messages fail with network or policy errors, add the egress preset:

```bash
# on: spark
nemoclaw my-assistant policy-list
nemoclaw my-assistant policy-add telegram
```

Telegram uses **long-polling**: the sandbox pulls messages from Telegram's servers, so **no public URL and no cloudflared tunnel** are needed. `policy-add telegram` alone does not register the channel; without `channels add` the bot replies `Error: Channel is unavailable: telegram`.

✓ Checkpoint: you can explain why the allowed User ID matters, and why a Telegram bot needs no inbound port on your Spark.

## 10 · Update, stop, uninstall

```bash
# on: spark
nemoclaw update --check              # is a newer LKG release available?
nemoclaw update --yes                # update the host CLI (does not rebuild sandboxes)
nemoclaw upgrade-sandboxes --check   # which sandboxes are stale after an update
```

Stop public access and forwards:

```bash
# on: spark
nemoclaw tunnel stop
openshell forward list
openshell forward stop <port>
```

The built-in uninstaller removes sandboxes, the OpenShell gateway, NemoClaw's Docker containers/images/volumes, the CLI and state directories. It keeps Docker, Node.js, npm and the vLLM image, and it keeps `~/.nemoclaw/` user data unless you ask:

```bash
# on: spark
nemoclaw uninstall --yes
```

| Flag | Effect |
|---|---|
| `--yes` | skip the confirmation prompt |
| `--keep-openshell` | leave the `openshell` binary in place |
| `--delete-models` | also remove model weights pulled by NemoClaw |
| `--destroy-user-data` | also remove `~/.nemoclaw/` (`rebuild-backups/`, `backups/`, `sandboxes.json`) |

(REFERENCE — flags quoted from the playbook.) Uninstall is hard to undo; no lab in this course runs it.

✓ Checkpoint: you know which command updates the CLI and which one tells you a sandbox needs a rebuild.

## Labs — run them here

**labs/lab16_1_preflight.py** — Ten read-only readiness checks on your Spark, and which onboarding path (Express or Existing vLLM) fits.

**labs/lab16_2_status_and_logs.py** — Read-only status, presets, live policy hosts, forwards, a bounded log slice and the two boundary tests, with tokens masked.

**labs/lab16_3_policy_preset.py** — Generate the News Digest network preset on your laptop, validate it against the playbook's rules, and apply it only with `--apply`.

**labs/lab16_4_inference_route.py** — Check that the model endpoint lists models, answers, and returns a well-formed tool call (Spark vLLM, or the laptop stand-in).

Labs 16-1 and 16-2 run LIVE on your Spark or DRY. Labs 16-3 and 16-4 run for real on the laptop; 16-3 copies its file to the Spark when one is configured.

## Try it yourself

**Exercise 16 — a least-privilege preset and its checker.** Open `week25/16_nemoclaw/exercises/ex16_policy_preset.py`. It has four `TODO`s:

1. `is_rfc1123(name)`: is the preset name a lowercase, hyphenated RFC 1123 label?
2. `endpoint(host)`: one HTTPS endpoint with the access mode the proxy needs.
3. `build_preset(name, description, hosts, binaries)`: the whole preset as a dict.
4. `errors(doc)`: catch the four failures the playbook documents.

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/exercises/ex16_policy_preset.py
```

**Expected output** (once all four TODOs are done; captured on this Mac with the reference solution)

```
✓ is_rfc1123: news-sources ✓ · news_sources ✕ · News ✕ · -news ✕ · 64 chars ✕
✓ endpoint: host + port 443 + access full + tls skip
✓ build_preset: a map keyed by group, 3 endpoints, 4 binaries
✓ errors: valid → [] · catches name · map · access · binaries

▣ your preset, as the file you would pass to `nemoclaw <sandbox> policy-add --from-file`
│ preset:
│   name: news-sources
│   description: Daily news digest source allowlist
│ network_policies:
│   news-sources:
│     name: news-sources
│     endpoints:
│     - host: developer.nvidia.com
│       port: 443
│       access: full
│       tls: skip
…
│     binaries:
│     - path: /usr/local/bin/openclaw
│     - path: /usr/local/bin/node
│     - path: /usr/bin/node
│     - path: /usr/bin/curl
```

<details><summary>Hint — why does a bare {host, port} entry get a 403?</summary>

The egress proxy needs to know *how* to pass the traffic. `access: full` with `tls: skip` is a raw pass-through tunnel. Without an access mode (or the L7 alternative `protocol: rest` + `enforcement: enforce` + `rules`), the proxy has no rule to apply, so it refuses the CONNECT even though the host is listed.

</details>

<details><summary>Stretch — a URL checker for the Deck Reviewer</summary>

The playbook's Deck Reviewer knobs include an optional "URL verification" preset that lets the agent HEAD-check links on a few hosts. Build it with your `build_preset("url-check", …, ["build.nvidia.com"], ["/usr/local/bin/openclaw", "/usr/local/bin/node"])`. Why might you leave `/usr/bin/curl` out this time?

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nemoclaw: command not found` after install | `source ~/.bashrc`, or open a new terminal |
| Installer fails with a Node.js version error | Node.js must be 22.16+. The playbook's fix: `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt-get install -y nodejs`, then re-run the installer |
| npm fails with `EACCES` | `mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global && export PATH=~/.npm-global/bin:$PATH`, then re-run |
| Gateway: "port 8080 is held by container…" or sandbox creation fails | Run `nemoclaw onboard` (or `nemoclaw onboard --resume`) again; it reuses a healthy gateway and recreates stale state when safe |
| Gateway fails with cgroup / "Failed to start ContainerManager" | Re-run the installer to get a newer OpenShell first; the playbook's fallback is a `daemon.json` `default-cgroupns-mode: host` fix (see the playbook) |
| "No GPU detected" during onboard | Expected on some platforms with unified memory; the wizard still uses vLLM |
| Inference hangs or times out | `curl http://127.0.0.1:8000/v1/models` on the Spark should list your model; wait for `Application startup complete`, then check `nemoclaw my-assistant status` |
| Web UI shows `origin not allowed` | Open `http://127.0.0.1:<port>/#token=…`, not `localhost` |
| Web UI forward died | `openshell forward stop 18789 my-assistant`, then `openshell forward start 18789 my-assistant --background` |
| `policy-add --from-file` fails with `Preset must declare preset.name …` | Hyphens, not underscores, in `preset.name` (lab 16-3 catches this) |
| A host is in the policy but fetches still 403 | The endpoint has no access mode or the group has no `binaries` (lab 16-3 catches both) |
| `openshell policy set` rejects `unknown field 'Version'` | Prefer the additive `policy-add --from-file` flow; or `sed -i 's/^Version:/version:/' policy.yaml` and retry |
| Scheduled digest never fires / shows `skipped` | Register the job with `openclaw cron add` from the operator side (Section 8); for local models see the `inference.local` DNS note |
| Telegram bot says `Error: Channel is unavailable: telegram` | `policy-add telegram` is not enough: run `nemoclaw <name> channels add telegram` |
| Memory pressure within capacity | The playbooks flush the page cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |

## Next

Continue to [Lab 17 — OpenClaw and Hermes Agent with a local LLM](../17_openclaw_hermes/TUTORIAL.md): run the two agents directly on the Spark against local vLLM, compare how each is configured and calls tools, and probe your endpoint's tool-calling ability.
