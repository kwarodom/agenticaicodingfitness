# ▶ Spark Lab 17 — OpenClaw and Hermes Agent with a local LLM

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Serve the agent-ready model the playbooks recommend for DGX Spark (`nvidia/Qwen3.6-35B-A3B-NVFP4`) with vLLM, and keep it private to the Spark.
- Run a read-only **preflight** on the Spark: tools, the model list on `:8000`, who can reach that port, and existing installs.
- **Probe tool calling** before you install anything: five small tests that separate a chat model from an agent model.
- Install **OpenClaw** and **Hermes Agent** the way their playbooks say, and point both at the same local model.
- Generate and validate both agents' configs on your laptop, and compare how the two agents are configured and call tools.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** 1 Spark (or none: DRY mode + laptop labs)

**Official playbooks covered:** [Run OpenClaw with a Local LLM](https://build.nvidia.com/spark/openclaw) · [Run Hermes Agent with a Local LLM](https://build.nvidia.com/spark/hermes-agent)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done: `ssh spark-a` works with a key | `ssh -o BatchMode=yes spark-a true` | lab 17-1 runs over SSH |
| Module 16 read | — | it ran OpenClaw *inside* a sandbox; here it runs **without** one |
| A Spark you treat as a lab machine | you decide | both playbooks: run the agent on a dedicated or isolated system |
| A Hugging Face login **on the Spark** | `hf auth whoami` on the Spark | vLLM downloads the model |
| This repo's Python + laptop Ollama | `.venv/bin/python --version`, `curl -s localhost:11434/api/tags` | labs 17-2 and 17-3 run on the laptop |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
curl -s localhost:11434/api/tags | head -c 120; echo
```

> 🔐 **No sandbox in this module.** In Module 16, OpenShell stood between the agent and your machine. Here OpenClaw and Hermes run as **your user** on the Spark: a tool that runs a shell command runs it for real. The OpenClaw playbook rates the risk **Medium to High**; the Hermes playbook rates it **Medium**. Labs in this module never install a messaging integration, enable a tool, or edit an agent's config file. Never paste a real token into chat, a prompt, a config file you share, or a command line.

✓ Checkpoint: you can say, in one sentence, what is different about running an agent here compared with Module 16.

## 1 · Two agents, one local model

**OpenClaw** is, in its playbook's words, a **local-first** agent: it "remembers conversations, adapts to your usage, runs continuously, uses context from your files and apps, and can be extended with community **skills**". It is the agent NemoClaw put in a sandbox in Module 16.

**Hermes Agent**, by Nous Research, is a **self-improving** agent. It runs as a terminal TUI, "creates skills from experience, improves them during use, persists memory across sessions, and can run scheduled tasks via its built-in cron". A built-in gateway can make it reachable from Telegram, Discord or Slack.

Both need the same thing from the Spark: an **OpenAI-compatible endpoint** whose model can call tools. Both playbooks recommend the same model and server for DGX Spark:

| Hardware | Recommended agent-ready model | Serve with |
|---|---|---|
| DGX Spark | Agent-ready Qwen3.6-35B-A3B (NVFP4) — `nvidia/Qwen3.6-35B-A3B-NVFP4` | vLLM, at `http://localhost:8000/v1` |

(REFERENCE — from both playbooks' Agent-ready Models tab.) Qwen3.6-35B-A3B is a Mixture-of-Experts model: about 3B parameters are active per token, the Module 01 recipe for staying fast on 273 GB/s.

You can also run either agent *inside* NemoClaw's sandbox: Module 16's `NEMOCLAW_AGENT=hermes` (or `nemohermes onboard`) does that for Hermes. This module installs both directly, so you see what each agent is like on its own.

✓ Checkpoint: you can name the one model handle and the one URL both agents will use.

## 2 · Serve the agent-ready model on the Spark

Both agent playbooks send you to the [vLLM playbook](https://build.nvidia.com/spark/vllm) for the launch command, and Module 05 covers vLLM in depth. The newest vLLM playbook only names the model and links a recipe on `recipes.vllm.ai`. The full command, including the three flags an agent needs, is in the **older** vLLM playbook's section "Run Agent Ready Qwen3.6 35B Model with vLLM" (`dgx-spark-playbooks/nvidia/vllm/README.md`). Module 05 uses the same source. This is that command, with three changes explained below it:

```bash
# on: spark
docker pull vllm/vllm-openai:latest
read -rs HF_TOKEN && export HF_TOKEN        # paste your token; it is not echoed or kept in shell history
docker run -d --name vllm-server --gpus all -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.4 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
```

The last three lines are what make this an **agent** server. `--enable-auto-tool-choice` lets the model decide when to call a tool. `--tool-call-parser qwen3_xml` turns Qwen3.6's tool-call output into OpenAI `tool_calls`. `--reasoning-parser qwen3` moves its thinking into a separate `reasoning` field. Without the tool flags the server still chats, but it returns no `tool_calls`, and lab 17-2 scores it "chat only". `--max-model-len 262144` is the context size you give both agents in Sections 5 and 6.

Changes from the playbook's command, all deliberate:

- **`-p 127.0.0.1:8000:8000`** instead of `-p 8000:8000`. The playbook form listens on every interface, so anyone on your LAN or tailnet could use your model. Both agent playbooks say to keep the endpoint bound to the Spark, and these agents run on the Spark, so loopback is enough. `--host 0.0.0.0` stays: it is the address *inside* the container. (If you also use Module 16's "Existing vLLM" option, re-run its `inference.local` check after changing the binding.)
- **`read -rs HF_TOKEN`** instead of `export HF_TOKEN="your_huggingface_token"`, so the token never lands in `~/.bash_history`.
- **`-d --name vllm-server`** instead of `-it`, so the server keeps running after you close the terminal and you can read its logs by name.

Wait for `Application startup complete` (`docker logs -f vllm-server`), then check the model list:

```bash
# on: spark
curl -sS http://localhost:8000/v1/models
```

"You should see your served model handle (for example `nvidia/Qwen3.6-35B-A3B-NVFP4`) in the returned list." (REFERENCE — quoted from the Hermes playbook.)

For agents, context matters: the OpenClaw playbook asks for **at least 32K tokens**, and 64K or more when memory allows. Set the agent's context window to match the server's `--max-model-len`.

✓ Checkpoint: `curl -sS http://localhost:8000/v1/models` on the Spark returns JSON with a `"data"` array that contains your model handle.

## 3 · Preflight: lab 17-1

Lab 17-1 runs the Hermes playbook's Step 1 (`uname -a`, `curl --version`, `git --version`) plus four checks both agents depend on. It is read-only.

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/labs/lab17_1_preflight.py
```

**Expected output** (DRY mode, captured on this Mac; with a Spark you get your own values and ✓ / ✕)

```
│ check                     result     last line of output
│ ────────────────────────  ─────────  ────────────────────────────────────────────────────
│ Linux                     ◈ example  Linux spark-abcd 6.11.0-1016-nvidia #16-Ubuntu SMP …
│ curl                      ◈ example  curl 8.5.0 (aarch64-unknown-linux-gnu) …
│ git                       ◈ example  git version 2.43.0
│ vLLM model list           ◈ example  {"object":"list","data":[{"id":"nvidia/Qwen3.6-35B-A
│ Who can reach :8000       ◈ example  LISTEN 0      4096         0.0.0.0:8000       0.0.0.
│ Existing installs         ◈ example  (end of list)
│ OpenClaw gateway process  ◈ example  no openclaw process

▣ STEP 8 · what the answers mean
◆ DRY: the EXAMPLE shows 0.0.0.0:8000 — what `docker run -p 8000:8000` gives you. See Section 2 for the fix.
```

What each line protects you from:

| Check | Fails later as… |
|---|---|
| vLLM model list | the Hermes installer "can't list any models at the model-selection prompt"; OpenClaw says "no model available" |
| Who can reach `:8000` | a model endpoint anyone on the network can use; LIVE mode warns when it sees `0.0.0.0:8000` |
| Existing installs | the Hermes installer offers to **import/migrate** from OpenClaw; the playbook says answer `n` |
| OpenClaw gateway process | config changes that never apply, because the running gateway was not restarted |

✓ Checkpoint: in LIVE mode every row is ✓, and `:8000` listens on `127.0.0.1` only.

## 4 · Probe tool calling before you install: lab 17-2

An agent works in a loop: send the conversation plus tool definitions, get back a `tool_call` (a name and JSON arguments), **run the tool itself**, send the result back as a `tool` message, get the final answer. The model only proposes; the agent executes. If the model never proposes, the agent's skills and tools never run.

Lab 17-2 tests that loop with five small probes. It runs against vLLM on the Spark when `:8000` answers, otherwise for real against Ollama on your laptop, labelled.

| Probe | Question | Why an agent needs it |
|---|---|---|
| P1 call | asked something only a tool can answer, does it emit a tool call? | no call, no tools |
| P2 choose | with three tools offered, does it pick the right one? | agents offer many tools |
| P3 schema | valid JSON, required keys, right types (`amount` is a number)? | the agent parses the arguments |
| P4 round trip | given the tool result, does the answer use it? | multi-step tasks |
| P5 restraint | asked "2 + 2", does it answer without a tool? | fewer needless (and risky) calls |

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/labs/lab17_2_tool_probe.py
```

**Expected output** (captured on this Mac: LAPTOP STAND-IN, these are the laptop's models, not the Spark's)

```
◆ probing Ollama on THIS laptop · http://localhost:11434/v1 · LAPTOP STAND-IN — tool behaviour is the model's, speed is not the Spark's

▣ STEP 1 · probe gemma4:12b
│ P1 call ✓ · P2 choose ✓ · P3 schema ✓ · P4 round trip ✓ · P5 restraint ✓
· first tool call: convert_currency({"amount":120,"from_ccy":"USD","to_ccy":"THB"})

▣ STEP 2 · probe nemotron-3-nano:latest
│ P1 call ✓ · P2 choose ✓ · P3 schema ✓ · P4 round trip ✓ · P5 restraint ✓
· first tool call: convert_currency({"from_ccy":"USD","to_ccy":"THB","amount":120})

▣ STEP 3 · probe gemma3:4b
│ P1 call ✕ · P2 choose ✕ · P3 schema ✕ · P4 round trip ✕ · P5 restraint ✕
· why: HTTP 400: registry.ollama.ai/library/gemma3:4b does not support tools

│ model                   P1  P2  P3  P4  P5  verdict      wall time
│ ──────────────────────  ──  ──  ──  ──  ──  ───────────  ─────────
│ gemma4:12b              ✓   ✓   ✓   ✓   ✓   agent-ready  52s
│ nemotron-3-nano:latest  ✓   ✓   ✓   ✓   ✓   agent-ready  26s
│ gemma3:4b               ✕   ✕   ✕   ✕   ✕   chat only    0s
◆ P1–P4 must pass for OpenClaw/Hermes to use tools. P5 failing means the agent calls tools when it should not (slower, and riskier once the tools can run commands).
```

`gemma3:4b` is the lesson: it answers chat questions well, and Ollama refuses the request as soon as tools are attached. Point an agent at it and you get errors, not an assistant. On the Spark, vLLM needs `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (Section 2) for the same reason: without them the server does not return `tool_calls`. The wall times are the laptop's and say nothing about the Spark.

✓ Checkpoint: you have run lab 17-2 against the model you plan to use, and P1–P4 are ✓.

## 5 · Install and run OpenClaw

Run the official install script on the Spark (the same "read it first" advice as Module 16 applies: you can save it with `-o ~/openclaw-install.sh`, read it, then `bash` it):

```bash
# on: spark
curl -fsSL https://openclaw.ai/install.sh | bash
```

After downloading dependencies, OpenClaw shows a **security warning**. Read it; select **Yes** only if you accept the risks. Then answer the onboarding prompts:

| Prompt | Choice | Why |
|---|---|---|
| Quickstart vs Manual | **Quickstart** | |
| Model provider | **Skip for now** (bottom of the list), or your backend if it is listed | you configure the model in `openclaw.json` next |
| Filtering models by provider | **All Providers**, then **Keep Current** | |
| Communication channel | **Skip for Now** | add channels only after the basics work |
| Skills | **No** | skills add capability *and* risk |
| Homebrew | **No** | macOS only; not needed on the Spark |
| Hooks | the playbook recommends all three | note: hooks may log data locally |
| Dashboard URL | **save the URL and token** | you need both to open the web UI; treat the token as a password |
| Finish | **Yes** | |

**Point OpenClaw at vLLM.** If you skipped the provider, open `~/.openclaw/openclaw.json` (`nano ~/.openclaw/openclaw.json`) and add the `models` section. This is the playbook's example for a vLLM server:

```json
"models": {
  "mode": "merge",
  "providers": {
    "vllm": {
      "baseUrl": "http://localhost:8000/v1",
      "apiKey": "vllm",
      "api": "openai-responses",
      "models": [
        {
          "id": "nvidia/Qwen3.6-35B-A3B-NVFP4",
          "name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
          "reasoning": true,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

`id` and `name` must match the served handle exactly; `contextWindow` must match the server's `--max-model-len`. `apiKey` is a placeholder: vLLM needs no key, "any non-empty placeholder works". If OpenClaw reports an unsupported endpoint for the Responses API, the playbook says to change `"api"` to the chat-completions variant for your OpenClaw version (vLLM always exposes `/v1/chat/completions`).

Restart the OpenClaw gateway so it reloads the file, then verify:

1. Forward the dashboard's port to your laptop. Use the port in the URL the installer printed (the playbook does not fix one):

```bash
# on: laptop
ssh -N -L <port>:127.0.0.1:<port> spark-a
```

2. Open the dashboard URL (with its token) in your browser, start a **new** conversation, and send a short message. A reply means the setup works.
3. Ask which model it is using. In the chat UI, `/model MODEL_NAME` switches models.

> ⚠ The playbook's critical rule: the OpenClaw web UI and any messaging channel must **never be exposed** to the public internet without strong authentication. Use an SSH tunnel or a VPN. Enable only **skills you trust**; skills with terminal or file-system access "increase risk significantly".

✓ Checkpoint: the OpenClaw dashboard answers through your tunnel, and it reports `nvidia/Qwen3.6-35B-A3B-NVFP4` (or your handle) as its model.

## 6 · Install and run Hermes Agent

The Hermes playbook was verified against **Hermes Agent v0.18.0 (2026.7.1)**; newer installers may word prompts differently. Run the installer from an **interactive** SSH session (`ssh -t spark-a`), with vLLM already serving:

```bash
# on: spark
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

The playbook uses the **Blank Slate** path, which keeps the wizard short:

| Prompt | Choice | Why |
|---|---|---|
| Install ripgrep … ffmpeg? | **Enter** (yes) | faster file search, TTS; needs sudo |
| Import/migrate from OpenClaw? | **`n`** | mixing migrations can leave gateway or messaging state inconsistent |
| How would you like to set up Hermes? | **Blank Slate** | Quick Setup signs in to Nous Portal instead of your local model |
| Select provider | **Custom endpoint (enter URL manually)** | your Spark, not a hosted provider |
| API base URL | `http://localhost:8000/v1` | the vLLM from Section 2 |
| API key [optional] | leave blank | vLLM needs no key |
| Model selection | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Hermes lists what `/v1/models` returns |
| Context length | Enter (auto-detect) | the recipe serves 262144 |
| Display name | Enter | |
| Select terminal backend | **Keep current (local)** / **Local** | see the warning below |
| What next? | **Start with everything disabled — finish now** | enable tools one at a time later |

Reload your shell and check the command:

```bash
# on: spark
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
which hermes
```

Run `hermes`, type `hello`, and press **Enter**. When you `/exit`, Hermes prints `hermes --resume <sessionId>`; save it to continue that conversation later.

**No terminal? Use the config commands.** If the installer printed "Setup wizard skipped (no terminal available)", the playbook's fallback configures the endpoint without prompts:

```bash
# on: spark
export PATH="$HOME/.local/bin:$PATH"
hermes config set model.provider custom
hermes config set model.base_url http://localhost:8000/v1
hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4
hermes -z "Reply exactly HERMES_OK"
```

"The last command should return `HERMES_OK`, confirming that Hermes can call the local vLLM model without opening the TUI." (REFERENCE — quoted from the playbook.)

**Tools come later, one at a time.** Blank Slate starts with the agent's tools disabled. `hermes tools` opens the list (web search, browser automation, terminal, file operations, code execution, and more; **SPACE** toggles, **ENTER** confirms).

> ⚠ The **local** terminal backend "runs them directly on the hardware platform": when you enable Hermes' terminal tool, the model's commands run as your user on the Spark. Enable it only on a Spark with nothing you would mind the agent reading or deleting.

Other day-to-day commands from the playbook: `hermes model` (switch model), `hermes --resume <sessionId>`, `hermes update`, and `/reasoning show` inside the TUI to see the model's intermediate reasoning.

✓ Checkpoint: `hermes -z "Reply exactly HERMES_OK"` prints `HERMES_OK` on the Spark.

## 7 · Configs side by side: lab 17-3

The two agents store the same three facts (endpoint, model handle, context size) in different places. Lab 17-3 writes both configs from one served model, validates them, and smoke-tests the endpoint. It never edits `~/.openclaw/openclaw.json`; with a Spark it copies the two files to `~/w25/agents/` for you to merge by hand.

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/labs/lab17_3_agent_configs.py
```

**Expected output** (captured on this Mac, no Spark configured; the smoke test is a LAPTOP STAND-IN)

```
▣ STEP 1 · which model does the endpoint serve?
◆ no Spark vLLM reachable → using the playbooks' DGX Spark default: nvidia/Qwen3.6-35B-A3B-NVFP4
◆ the validation below cannot check 'is it served?' without the Spark; the smoke test uses the laptop stand-in

▣ STEP 2 · generate the two configs
▣ openclaw.models.json  (merge into ~/.openclaw/openclaw.json)
│ {
│   "models": {
│     "mode": "merge",
│     "providers": {
│       "vllm": {
│         "baseUrl": "http://localhost:8000/v1",
│         "apiKey": "vllm",
│         "api": "openai-responses",
…
▣ hermes-config.sh
│ #!/usr/bin/env bash
│ # Hermes: point the agent at the local vLLM endpoint (Hermes playbook, non-interactive fallback)
│ set -e
│ export PATH="$HOME/.local/bin:$PATH"
│ hermes config set model.provider custom
│ hermes config set model.base_url http://localhost:8000/v1
│ hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4
│ hermes -z "Reply exactly HERMES_OK"
◆ written to week25/17_openclaw_hermes/.runs/

▣ STEP 3 · validate
✓ openclaw.models.json: local /v1 endpoint, placeholder key, ≥32K context
│ broken config     validator  finding
│ ────────────────  ─────────  ────────────────────────────────────────────────────
│ real-looking key  ✓ caught   apiKey must be a non-empty placeholder (vLLM needs …
│ LAN address       ✓ caught   baseUrl host '192.168.1.42' is not local — the agen…
│ 8K context        ✓ caught   contextWindow 8192 < 32768 (the playbook's minimum …
│ missing /v1       ✓ caught   baseUrl 'http://localhost:8000' must end in /v1

▣ STEP 4 · smoke test: does this (endpoint, model) pair answer?
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  HERMES_OK
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 10847 ms · 5 tok in 11.2s · 15.4 tok/s

✓ configs are valid for a local vLLM
✓ every broken config was caught
✓ smoke test answered (laptop)
```

With both agents running, here is how they compare:

| | OpenClaw | Hermes Agent |
|---|---|---|
| Made by | OpenClaw project ([openclaw.ai](https://openclaw.ai)) | Nous Research |
| Built for | an always-on, local-first assistant; extended with community skills | a self-improving agent that writes and refines its own skills |
| Main interface | web dashboard (URL + token), gateway chat, `/model` | terminal TUI `hermes`; one-shot `hermes -z "…"`; `--resume` |
| Install | `curl -fsSL https://openclaw.ai/install.sh \| bash` | `curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \| bash` |
| Model config | `~/.openclaw/openclaw.json` → `models.providers.vllm` {`baseUrl`, `apiKey`, `api`, `models[]`} | `hermes config set model.provider custom` / `model.base_url` / `model.default`; or `hermes model` |
| API it uses | `"api": "openai-responses"` (Responses API); a chat-completions variant if unsupported | an OpenAI-compatible custom endpoint (`…/v1`) |
| Applying a change | restart the gateway so it reloads `openclaw.json` | takes effect for the next session |
| Where tools come from | skills: web-UI sidebar, [Clawhub](https://docs.openclaw.ai/tools/clawhub), or ask OpenClaw; plus hooks | `hermes tools` (web search, browser, terminal, file ops, code execution); skills it creates from experience |
| Where tools run | on the host, as your user (in a sandbox only via NemoClaw) | the terminal backend: `local` = on the host; Docker, Modal, SSH, Daytona, Singularity are out of the playbook's scope |
| Memory and schedules | remembers conversations; OpenClaw's cron (Module 16 used `openclaw cron add`) | persistent memory across sessions; built-in cron |
| Messaging | a channel during onboarding, or later | `hermes gateway setup` (Telegram, Discord, Slack) + a systemd service |
| Update / remove | re-run the install script; remove its directory; stop the gateway | `hermes update`; `hermes uninstall`; `~/.hermes` may remain |

The tool loop itself is the same in both, and it is what lab 17-2 probed: the agent sends tool definitions, the model returns a `tool_call`, the agent runs it, the result goes back as a `tool` message.

✓ Checkpoint: you can point to where each agent stores the model handle, and say what you would change in each to switch models.

## 8 · Harden, update, remove

Both playbooks end their risk section the same way: you cannot remove all risk, so reduce it.

| Measure | OpenClaw playbook | Hermes playbook | This course |
|---|---|---|---|
| Isolated machine, only the data the agent needs | strongly recommended | recommended | a lab Spark |
| Dedicated accounts, minimum access | yes | — | never your main accounts |
| Web UI / endpoint never public | critical | keep vLLM bound to the Spark | SSH tunnel; `-p 127.0.0.1:8000:8000` |
| Only trusted skills / tools | yes | enable tools deliberately (`hermes tools`) | Blank Slate, one tool at a time |
| Limit the agent's internet access | firewall or network isolation | — | or run it in NemoClaw (Module 16) |
| Messaging restricted to you | — | enter your numeric user ID at "Allowed user IDs" | leaving it blank lets anyone who finds the bot use it |
| Monitor activity | review logs and executed commands | review sessions; `sudo journalctl -u <hermes-gateway-unit> -e` | check after every new tool |

**Optional messaging for Hermes.** If you want Hermes on your phone, the playbook first checks that the Spark can reach Telegram, then runs the gateway wizard, which asks for the bot token (hidden as you paste) and your allowed user IDs:

```bash
# on: spark
curl -sS --connect-timeout 10 -o /dev/null -w "HTTP %{http_code}\n" https://api.telegram.org/
hermes gateway setup
```

Any HTTP status line (`HTTP 404`, `HTTP 200`, `HTTP 302`) means the connection completed over TLS; a timeout means your network blocks Telegram. No lab runs these commands.

**Update and remove.**

```bash
# on: spark
hermes update
hermes uninstall             # interactive; add sudo "$(which hermes)" if you installed a system gateway service
ls -la ~/.hermes             # configuration, sessions and skills may still be here
```

For OpenClaw the playbook's rollback is to stop the gateway and uninstall it with the install script or by removing its directory. Stop vLLM separately with `docker stop vllm-server && docker rm vllm-server`. Deleting `~/.hermes` (`rm -rf ~/.hermes`) cannot be undone; do it only when you want a full reset.

✓ Checkpoint: your vLLM port is loopback-only, no messaging channel is open without an allowed-user list, and you know where each agent keeps its data.

## Labs — run them here

**labs/lab17_1_preflight.py** — Read-only checks on the Spark: Linux, curl and git, the vLLM model list, who can reach `:8000`, and existing installs.

**labs/lab17_2_tool_probe.py** — Five tool-calling probes (call, choose, schema, round trip, restraint) against the Spark's vLLM or the laptop stand-in.

**labs/lab17_3_agent_configs.py** — Generate OpenClaw's `models` section and Hermes' config commands from one served model, validate both, and smoke-test the endpoint.

Lab 17-1 runs LIVE on your Spark or DRY. Labs 17-2 and 17-3 run for real on the laptop (Spark vLLM when it answers); 17-3 copies its files to the Spark when one is configured.

## Try it yourself

**Exercise 17 — one model, two agents.** Open `week25/17_openclaw_hermes/exercises/ex17_agent_config.py`. It has three `TODO`s:

1. `openclaw_provider(base_url, model_id, context_window)`: the `models.providers.vllm` entry in the playbook's shape.
2. `hermes_commands(base_url, model_id)`: the four commands of the Hermes playbook's non-interactive setup.
3. `review(provider, served)`: flag a missing `/v1`, a non-local host, a real-looking key, an unserved model, and a context under 32K.

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/exercises/ex17_agent_config.py
```

**Expected output** (once all three TODOs are done; captured on this Mac with the reference solution)

```
✓ openclaw_provider: baseUrl · placeholder key · openai-responses · id = name · 262144 context
✓ hermes_commands: provider custom · base_url · default · hermes -z check
✓ review: good → [] · catches /v1 · local · key · served · context

▣ ~/.openclaw/openclaw.json → models.providers.vllm
│ {
│   "baseUrl": "http://localhost:8000/v1",
│   "apiKey": "vllm",
│   "api": "openai-responses",
│   "models": [
│     {
│       "id": "nvidia/Qwen3.6-35B-A3B-NVFP4",
│       "name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
│ …

▣ Hermes, on the Spark
$ hermes config set model.provider custom
$ hermes config set model.base_url http://localhost:8000/v1
$ hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4
$ hermes -z "Reply exactly HERMES_OK"
```

<details><summary>Hint — why does a non-local baseUrl count as a problem?</summary>

These agents run on the Spark, next to vLLM, so `localhost` is always right for them. A LAN address in the config usually means the model port is open to the LAN too, which both playbooks warn against. And if the config ever points at someone else's machine, your prompts and files go there.

</details>

<details><summary>Stretch — probe before you pick</summary>

Run `lab17_2_tool_probe.py --models gemma4:12b,gemma3:4b`, then use the exercise's `openclaw_provider()` with the model that passed. What would happen in OpenClaw's chat if you configured the one that failed?

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| OpenClaw dashboard URL does not load | Restart the gateway so it reloads `~/.openclaw/openclaw.json`; check it runs with `pgrep -f openclaw`; find the URL and token in the installer output or `~/.openclaw/logs/` |
| "Connection refused" to `localhost:8000` | vLLM is not running, still loading, or on another port: `docker ps`, then `curl http://localhost:8000/v1/models` |
| OpenClaw says no model available | Add the provider to `openclaw.json`; `id`/`name` must match the served handle exactly |
| OpenClaw reports an unsupported endpoint for the Responses API | Change `"api": "openai-responses"` to the chat-completions variant for your OpenClaw version |
| Config changes not applied | Restart the OpenClaw gateway |
| `hermes: command not found` | `source ~/.bashrc`; in scripts, `export PATH="$HOME/.local/bin:$PATH"` or call `~/.local/bin/hermes` |
| `sudo: hermes: command not found` | `sudo "$(which hermes)" …` (sudo resets PATH) |
| "Setup wizard skipped (no terminal available)" | Re-run `hermes setup` in an interactive terminal, or use the `hermes config set` fallback (Section 6) |
| The Hermes installer lists no models | vLLM is not up yet: wait for `Application startup complete`, check `curl http://localhost:8000/v1/models`, re-run the installer |
| Installer asks about OpenClaw import/migration | Answer `n`; if you already migrated, uninstall, `rm -rf ~/.hermes`, reinstall |
| Agent chats but never uses a tool | The model or server does not return `tool_calls`: run lab 17-2; on the Spark start vLLM with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (Section 2) |
| Out of memory or very slow | Check `nvidia-smi`; lower `--gpu-memory-utilization` / `--max-model-len` or use a smaller model; the playbooks flush the page cache with `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| Telegram bot never answers | Run the `api.telegram.org` HTTPS check (Section 8); check the gateway unit with `systemctl` / `journalctl`; confirm your user ID is in the allowed list |

## Next

Continue to [Lab 18 — coding agents on local inference](../18_coding_agents/TUTORIAL.md): point command-line coding agents at the same local models, and see which of this module's probes a coding agent needs most.
