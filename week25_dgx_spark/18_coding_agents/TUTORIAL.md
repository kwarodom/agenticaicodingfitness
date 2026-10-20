# ▶ Spark Lab 18 — Coding agents on local inference

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Serve a coding model on your Spark with Ollama, the way NVIDIA's three coding playbooks do it.
- Start Claude Code, OpenCode or Codex against it with one command: `ollama launch <agent> --model …`.
- Learn what changes when the model is local: context length, tool-call reliability, and speed.
- Preflight the endpoint: the four APIs a coding agent speaks, tool support, and the context window.
- Measure a model the honest way: real tool calls, real edits, real tests, in a temp dir.
- Generate settings for four agents (Claude Code, OpenCode, Codex, Continue) without touching your own global config.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** 1 DGX Spark (or none: labs run against your laptop's Ollama as a labelled stand-in)

**Official playbooks covered:** [CLI coding agents with local inference](https://build.nvidia.com/spark/cli-coding-agent) · [Vibe coding in VS Code](https://build.nvidia.com/spark/vibe-coding) · plus Claude Code with local inference (`nvidia/playbook-local-coding-agent` in the GitHub repo; not in the build.nvidia.com/spark index, see Section 2)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done | `ssh -o BatchMode=yes spark-a true` | labs run Spark commands over SSH |
| Module 03 (Ollama) helps | `ollama --version` on the Spark | every playbook here serves through Ollama |
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs the labs, with PyYAML for lab 03 |
| Ollama on the laptop (optional) | `ollama list` on the laptop | the labelled stand-in when no Spark is reachable |
| A coding agent to try (optional) | `claude --version`, `opencode --version` or `codex --version` | Section 4 launches one |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
ollama --version
```

**Expected output** (captured on this Mac)

```
Python 3.13.13
ollama version is 0.34.4
```

> 🔐 The labs never write to `~/.claude`, `~/.codex`, `~/.config/opencode` or `~/.continue`. Lab 03 writes every generated file under `week25/18_coding_agents/.runs/` (gitignored), then checks that your global config files were not changed.

✓ Checkpoint: you can reach your Spark over SSH, or you have Ollama running on the laptop to use as the stand-in.

## 1 · What changes when the coding model is local

A coding agent (Claude Code, OpenCode, Codex, Continue) is a loop. It sends the model your request, the files it has read and a list of **tools** (read a file, write a file, run a command). The model replies with a **tool call**. The agent runs it, sends back the result, and asks again. The model never touches your disk. The agent does, but only when the model returns a well-formed tool call.

When you swap a cloud model for one on your Spark, four things change:

| | Cloud coding model | Local model on a Spark | What to check |
|---|---|---|---|
| Where your code goes | to a provider | stays on your network | nothing, that is the point |
| Cost | per token | electricity | nothing |
| **Context length** | large and fixed | whatever the *server* allocates, which can be far less than the model supports | lab 01: allocated context vs model maximum |
| **Tool-call reliability** | tuned for agents | depends on the model, the quantization and the server's chat template | lab 02: does it call the tool, with valid JSON, and do the tests pass? |
| **Speed** | fast | bounded by 273 GB/s of memory bandwidth (Module 01) | TTFT and tok/s on your Spark |

Tool-call reliability is the one that surprises people. A model can write perfect code and still fail as an agent: it answers in prose instead of calling `write_file`, it prints the tool call as text the server does not parse, or it sends broken JSON. Every one of those is a failed edit. The CLI playbook's own Troubleshooting table has this row: a direct Claude Code setup "produces prose but does not edit files" because "some model/server combinations do not emit tool calls reliably".

**Why the playbooks pick an MoE model.** Decode speed is bounded by the bytes of *active* weights read per token (Module 01, Section 7). The CLI playbook's Spark default, `qwen3.6:35b-a3b-mtp-q4_K_M`, is a 35B Mixture-of-Experts model. "a3b" in the tag means about 3B parameters active per token. The Claude Code playbook uses the dense `qwen3.6:27b`. Here is the arithmetic, using `sparkkit.decode_ceiling_tok_s` at Q4_K_M (4.85 bits per weight):

| Model | Active per token | Read per token | Single-stream ceiling at 273 GB/s |
|---|---|---|---|
| qwen3.6 35B-A3B | ~3B | 1.8 GB | ~150 tok/s |
| qwen3.6 27B (dense) | 27B | 16.4 GB | ~17 tok/s |
| gpt-oss-120b (vibe-coding) | 5.1B | 2.7 GB (MXFP4) | ~101 tok/s |

These are upper bounds from arithmetic, not measurements. An agent makes many model calls per task, so a 9× gap in the ceiling decides whether an edit takes seconds or minutes.

✓ Checkpoint: you can name the three ways a model fails as an agent even when its code is correct, and you can say why the Spark playbook's default is an MoE model.

## 2 · Three playbooks, one idea

All three playbooks serve the model with Ollama on port `11434` and differ in the client:

| Playbook | Client | Model (as printed) | Hardware row in its matrix |
|---|---|---|---|
| [CLI coding agents](https://build.nvidia.com/spark/cli-coding-agent) | Claude Code, OpenCode, Codex CLI, each via `ollama launch` | `qwen3.6:35b-a3b-mtp-q4_K_M` (~23 GB) · optional `q8_0` (~39 GB), `bf16` (~71 GB) | **DGX Spark** |
| Claude Code with local inference ([GitHub README](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/playbook-local-coding-agent)) | Claude Code via `ollama launch` | `qwen3.6:27b` | **DGX Station** (see note) |
| [Vibe coding in VS Code](https://build.nvidia.com/spark/vibe-coding) | Continue extension in VS Code | `gpt-oss:120b` | **DGX Spark** |

> ⚠ The current `playbook-local-coding-agent/README.md` lists only **DGX Station** in its hardware matrix, and it is not listed in the build.nvidia.com/spark index (checked 2026-09-29), so there is no Spark page to link to. For the Spark, follow the [CLI coding agents](https://build.nvidia.com/spark/cli-coding-agent) playbook. Its Claude Code steps are the same commands as the CLI playbook. This module takes one extra step from it, the 64K context setting (Section 3). For the Spark, use the CLI playbook's model.

The memory sizes come from the CLI playbook's prerequisites. All three variants fit in the Spark's 128 GB. The `q4_K_M` default leaves about 100 GB for the KV cache, a second model, or a Module 19 stack.

✓ Checkpoint: you know which playbook to follow for a terminal agent (CLI coding agents) and which one for an editor assistant (vibe coding).

## 3 · Serve the coding model on the Spark

These are the CLI playbook's Steps 1–4, run **on the Spark**. The playbook installs Ollama with its official script. If you did Module 03, Ollama is already there, so check the version and skip the install.

```bash
# on: spark
cat /etc/os-release | head -n 2
nvidia-smi
ollama --version                       # already installed (Module 03)? skip the next line
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.6:35b-a3b-mtp-q4_K_M
ollama list
```

The playbook says only that `ollama --version` "should show a current Ollama release". `ollama launch` and the MTP Q4_K_M tag need a recent build. An old Ollama answers `unknown command` or fails the pull with HTTP 412, and the fix in both cases is to reinstall Ollama. Optional, larger variants from the same step:

```bash
# on: spark
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

**Context length.** The Claude Code playbook (Step 6) warns that "Ollama defaults to a 4096 token context length" and raises it to 64K for coding agents. You can set it for one session, or for the whole server:

```bash
# on: spark
ollama run qwen3.6:35b-a3b-mtp-q4_K_M
# at the >>> prompt:   /set parameter num_ctx 64000     then /bye
sudo systemctl stop ollama
OLLAMA_CONTEXT_LENGTH=64000 ollama serve        # keep this terminal open
```

> 💡 Defaults change between Ollama versions. The CLI playbook says Qwen3.6 "ships with a 256K context window by default", and the Ollama 0.34.4 on this Mac allocated 262,144 tokens for `nemotron-3-nano` (lab 01 below). Don't assume a number. Lab 01 reads what the server actually allocated from `/api/ps`. A bigger context costs KV-cache memory (Module 01's formula), so set it to what your repo needs.

Test it with the playbook's own prompt, or as an agent would: with a tool. The ⚡ block below goes to the Spark's Ollama, or to the laptop stand-in, clearly labelled:

```spark
{"target": "ollama", "which": "a", "model": "qwen3.6:35b-a3b-mtp-q4_K_M",
 "messages": [{"role": "user", "content": "Create math_utils.py with a function add(a, b) that returns a + b. Use the write_file tool."}],
 "max_tokens": 256,
 "tools": [{"type": "function", "function": {"name": "write_file", "description": "Write a text file in the workspace.",
   "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]}
```

A good answer is a `→ tool_call write_file({...})` line, not a paragraph of code.

✓ Checkpoint: `ollama list` on the Spark shows `qwen3.6:35b-a3b-mtp-q4_K_M`, and the ⚡ block returns a `write_file` tool call.

## 4 · Launch an agent with one command

`ollama launch` starts a supported agent already wired to your local model, so there are no environment variables and no provider files to write. These are the CLI playbook's Step 5 commands for each agent (install the agent once, then launch):

```bash
# on: spark
# Claude Code
curl -fsSL https://claude.ai/install.sh | bash
claude --version
ollama launch claude --model qwen3.6:35b-a3b-mtp-q4_K_M

# OpenCode
curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"
opencode --version
ollama launch opencode --model qwen3.6:35b-a3b-mtp-q4_K_M

# Codex CLI (needs Node.js / npm)
npm install -g @openai/codex
codex --version
ollama launch codex --model qwen3.6:35b-a3b-mtp-q4_K_M
```

`ollama launch --help` on Ollama 0.34.4 lists more integrations than the playbook covers (Copilot CLI, Cline, Qwen Code, Hermes, OpenClaw and others; Module 17 uses the last two). It also lists `--config` (configure without launching) and `--restore` (restore an integration to its default profile). So `launch` **edits that agent's own profile** on the machine where you run it. That is fine on the Spark. Know it before you run it on a laptop you depend on.

Then the playbook's end-to-end task (Step 6): a stub, a test, and one instruction.

```bash
# on: spark
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo
printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pytest
```

In the agent, type `Please implement add() in math_utils.py and make sure the test passes.`, exit (`/exit` or Ctrl+C), then:

```bash
# on: spark
cd ~/cli-agent-demo && source .venv/bin/activate
python3 -m pytest -q
deactivate
```

The playbook's expected result: "Expected output should show the test passing." If the agent printed the code but `math_utils.py` still ends in `pass`, you have met the tool-call failure from Section 1. Lab 02 measures how often it happens.

✓ Checkpoint: `pytest -q` passes in `~/cli-agent-demo` after an agent edited the file, not you.

## 5 · Where does the agent run: on the Spark, or on your laptop?

The playbooks run the agent **on the Spark** (in an SSH session), next to the model. That is the simplest setup: `localhost:11434`, no network exposure. If you want the agent or your editor on the laptop instead, the laptop must reach the Spark's Ollama. There are two ways:

| | SSH tunnel (course default) | Open Ollama to the network (vibe-coding Step 2) |
|---|---|---|
| On the Spark | nothing to change | systemd drop-in `OLLAMA_HOST=0.0.0.0:11434`, `OLLAMA_ORIGINS=*`, `ufw allow 11434/tcp` |
| URL on the laptop | `http://localhost:21434` | `http://<spark-ip>:11434` |
| Who else can call it | nobody | everyone on the LAN or tailnet, **with no authentication** |
| Good for | one developer | a team, behind a gateway with keys (Module 08) |

The tunnel uses local port 21434 because your laptop may already run Ollama on 11434 (Module 01, lab 03):

```bash
# on: laptop
ssh -N -L 21434:localhost:11434 spark-a
curl -s http://localhost:21434/api/version
```

For the vibe-coding path, the playbook's remote-access steps on the Spark are:

```bash
# on: spark
sudo systemctl edit ollama          # add the [Service] lines lab 03 generates (ollama-override.conf)
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo ufw allow 11434/tcp
```

Then, from the laptop, `curl -v http://YOUR_HARDWARE_IP:11434/api/version`. In VS Code, install **Continue**, choose **Ollama** as the provider and **Autodetect** as the model. For a remote Spark, replace Continue's `config.yaml` with the block lab 03 generates, which is the playbook's own YAML with your address filled in.

✓ Checkpoint: you have picked tunnel or open port, and you can say what `OLLAMA_ORIGINS=*` on `0.0.0.0` exposes.

## 6 · Preflight the endpoint: lab 01

Before you blame the agent, check the server. A coding agent needs one of four APIs from Ollama, depending on the client:

| Endpoint | Spoken by |
|---|---|
| `/api/*` (native Ollama) | `ollama launch`, Continue's `ollama` provider |
| `/v1/chat/completions` (OpenAI chat) | OpenCode, most OpenAI-compatible clients |
| `/v1/messages` (Anthropic Messages) | Claude Code |
| `/v1/responses` (OpenAI Responses) | Codex |

Lab 01 checks all of them, plus the model's `tools` capability from `/api/show`, a real `write_file` tool call, and the context the server allocated. On a Spark it also runs `ollama --version`, checks for `ollama launch` and lists the pulled models over SSH.

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab01_endpoint_preflight.py
```

**Expected output** (captured on this Mac with no Spark configured: LAPTOP STAND-IN, so the Spark rows are EXAMPLE shapes and the model is the laptop's)

```
▣ STEP 2 · which endpoint are we testing?
→ http://localhost:11434 · model=nemotron-3-nano:latest · Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark

▣ STEP 3 · native API: version, model details, capabilities, context
│ nemotron-3-nano:latest: 31.6B params · Q4_K_M · capabilities=['completion', 'tools', 'thinking'] · model max context=1,048,576

▣ STEP 4 · the three wire formats: OpenAI chat (OpenCode), Anthropic messages (Claude Code), Responses (Codex)
· /v1/chat/completions → 'READY' in 10.1s
· /v1/messages         → blocks=['thinking'] text='' in 12.6s (stop_reason=max_tokens)
  ~ a thinking model spent the whole 24-token budget on a `thinking` block. Retry with thinking off:
· /v1/messages + thinking:disabled → 'READY.' (stop_reason=end_turn)
· /v1/responses        → 'READY' (status=completed)

▣ STEP 5 · a real tool call, and the context the server actually allocated
→ tool_call write_file({"path":"hello.py","content":"print(\"hi\")"})

│ check                                ok  detail
│ ───────────────────────────────────  ──  ────────────────────────────────────
│ Ollama native API  /api/version      ✓   0.34.4
│ model reports `tools` capability     ✓   completion, tools, thinking
│ OpenAI API  /v1/chat/completions     ✓   READY
│ Anthropic API  /v1/messages          ✓   READY. (only with thinking disabled)
│ OpenAI Responses API  /v1/responses  ✓   READY
│ tool call with valid JSON args       ✓   write_file · 1.8s
│ allocated context ≥ 64,000           ✓   262,144 tokens (model max 1,048,576)
◆ LAPTOP STAND-IN (these are not Spark numbers). Speeds above are this endpoint's, for a ≤ 300-token reply.
✓ every check passed: this endpoint can drive a CLI coding agent.
═ ready for `ollama launch claude --model …`
```

The first call took 10 s because it loaded the model, which is cold-start time and not a measure of speed. The Anthropic line shows a real trap. A **thinking** model on `/v1/messages` answered with a `thinking` block only and ran out of its 24-token budget before writing any text. With `"thinking": {"type": "disabled"}` it answered straight away. On a Spark with Qwen3.6 (also a thinking-capable family), an agent that feels slow may be spending tokens on hidden reasoning.

✓ Checkpoint: every row is ✓ against your Spark (`SPARK_HOST` set), or against the laptop stand-in, and you know which endpoint your chosen agent uses.

## 7 · Measure reliability, not vibes: lab 02

"It wrote nice code in the chat" tells you nothing about agent work. Lab 02 does what an agent does, three times: it gives the model a stub, a failing test and one `write_file` tool. It writes what the model sends into a fresh temp dir, runs the test with a stdlib runner (no pytest needed), and on failure gives the model the real test output for one repair turn. Task 1 is the playbook's own `add()` task. Tasks 2 and 3 (`slugify`, `parse_duration`) need a little more care.

> ⚠ Lab 02 runs code a model wrote. It runs it in a temp dir, in a subprocess with a 20 s timeout and a bare environment, after refusing code that mentions `os`, `subprocess`, `socket`, `open(` or `eval(`. That is a guard rail, not a sandbox. Module 15 (OpenShell) is the sandbox.

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab02_tool_call_probe.py            # --trials 5 for a real verdict
```

**Expected output** (captured on this Mac, LAPTOP STAND-IN: three laptop models, one trial each. Your run will differ, because sampling is random.)

```
▣ STEP 1 · does each model advertise tool support? (GET /api/show → capabilities)
│ nemotron-3-nano:latest       completion, tools, thinking
│ gemma4:12b                   completion, vision, audio, tools, thinking
│ gemma3:4b                    completion, vision   ← no tools: an agent cannot edit files

▣ STEP 2 · run 3 tasks × 3 models × 1 trial(s)
✓ nemotron-3-nano:latest   add              44.5s  passed first try
✕ nemotron-3-nano:latest   slugify          15.1s  tests failed again after repair: AssertionError
✕ nemotron-3-nano:latest   parse_duration   11.4s  tool call printed as TEXT — the server's parser missed it, so no edit happened
✓ gemma4:12b               add              11.5s  passed first try
✓ gemma4:12b               slugify          23.6s  passed first try
✓ gemma4:12b               parse_duration   13.2s  passed first try
✕ gemma3:4b                add               0.0s  server refused: registry.ollama.ai/library/gemma3:4b does not support tools

▣ STEP 3 · scorecard
│ model                   tool call  valid args  pass 1st try  pass ≤1 repair  median time
│ ──────────────────────  ─────────  ──────────  ────────────  ──────────────  ───────────
│ nemotron-3-nano:latest  2/3        2/3         1/3           1/3             15.1s
│ gemma4:12b              3/3        3/3         3/3           3/3             13.2s
│ gemma3:4b               0/3        0/3         0/3           0/3             —
→ wrote week25/18_coding_agents/.runs/lab02_scorecard.json
```

Three runs on this Mac gave the same shape. `gemma4:12b` passed 9 of 9. `nemotron-3-nano` passed `add` every time, fixed `slugify` after the repair turn once in three, and printed its `parse_duration` tool call as text (`<tool_call><function=write_file>…`) all three times. `gemma3:4b` has no `tools` capability, so the server rejects the request outright. That gives you three lessons:

1. **Capability is a hard gate.** Check `/api/show → capabilities` before anything else. No `tools`, no agent.
2. **"Printed as text" is a server/template problem, not a coding problem.** The model tried to call the tool, but in a format the server did not parse. This is the playbook's "produces prose but does not edit files", and why it recommends `ollama launch` with the tested Qwen3.6 model.
3. **The bigger model is not automatically the better agent.** The 31.6B `nemotron-3-nano` lost to the 12B `gemma4` on this probe. Measure the model you plan to use, on the Spark, with `--trials 5`.

On a Spark, lab 02 probes `qwen3.6:35b-a3b-mtp-q4_K_M` by default (`--models a,b` to compare). Nobody has recorded that run for this course yet, so there is no Spark output to show here. Yours will be the first.

✓ Checkpoint: you have a scorecard for at least one tool-capable model, and you can explain each ✕ line's cause.

## 8 · Settings for every agent, safely: lab 03

`ollama launch` needs no files, but only on the machine that runs Ollama and only for agents it supports. When the agent runs on your laptop, or you want settings under version control, you need config files. Lab 03 writes them for one endpoint and one model, validates each (YAML, JSON, TOML parsers), and prints how to apply it. It **never** applies anything.

| File | Source | For |
|---|---|---|
| `on_spark_launch.sh` | CLI playbook, Steps 3 and 5 | pull + `ollama launch claude\|opencode\|codex` on the Spark |
| `ollama-override.conf` | vibe-coding playbook, Step 2 | opening the Spark's Ollama to the network (and an optional, commented `OLLAMA_CONTEXT_LENGTH`) |
| `continue-config.yaml` | vibe-coding playbook, Step 6 | Continue in VS Code on your laptop |
| `claude-code-direct.env` | **course**, not the playbook | Claude Code on the laptop → Ollama's `/v1/messages`, one shell only |
| `opencode.json` | **course**, not the playbook | OpenCode *project* config → `/v1` |
| `codex-home/config.toml` | **course**, not the playbook | Codex with `CODEX_HOME`, so `~/.codex` is never touched |

The three COURSE files are wiring for when `ollama launch` is not an option. Lab 01 showed that Ollama answers `/v1/messages` and `/v1/responses`, but the playbook's warning still applies: a direct setup can produce prose instead of edits. Run lab 02 against the same endpoint first. Field names in the Codex and OpenCode files come from those tools' docs, not NVIDIA's, so check them against your installed version.

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab03_agent_config_generator.py            # placeholder host
.venv/bin/python week25/18_coding_agents/labs/lab03_agent_config_generator.py --tunnel   # http://localhost:21434
```

**Expected output** (captured on this Mac, no Spark configured, so the playbook's `YOUR_HARDWARE_IP` placeholder stays)

```
▣ STEP 2 · write and validate each file
│ file                    source    size     check
│ ──────────────────────  ────────  ───────  ───────────────────────────────
│ on_spark_launch.sh      PLAYBOOK    539 B  ✓ bash script
│ ollama-override.conf    PLAYBOOK    537 B  ✓ systemd [Service] block
│ continue-config.yaml    PLAYBOOK    324 B  ✓ valid YAML · provider ollama
│ claude-code-direct.env  COURSE      387 B  ✓ 3 variables
│ opencode.json           COURSE      386 B  ✓ valid JSON
│ codex-home/config.toml  COURSE      407 B  ✓ valid TOML · provider defined
→ all files in week25/18_coding_agents/.runs/agent-config/
…
▣ STEP 4 · prove your global agent config was not touched
✓ ~/.claude/settings.json            unchanged
✓ ~/.codex/config.toml               unchanged
✓ ~/.config/opencode/opencode.json   absent (not created)
✓ ~/.continue/config.yaml            absent (not created)
═ Generated and validated; nothing outside .runs/ was written.
```

Applying them is a choice you make, one agent at a time. For example, Claude Code on the laptop through the tunnel, in a subshell so nothing persists:

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab03_agent_config_generator.py --tunnel
( set -a; . week25/18_coding_agents/.runs/agent-config/claude-code-direct.env; set +a; \
  cd ~/cli-agent-demo && claude --model qwen3.6:35b-a3b-mtp-q4_K_M )
```

✓ Checkpoint: lab 03 ends with `nothing outside .runs/ was written`, and you can say which files come from the playbooks and which are the course's.

## Labs — run them here

**labs/lab01_endpoint_preflight.py** — Preflight an Ollama endpoint for coding agents: version, tool capability, all four APIs, a real tool call and the allocated context.

**labs/lab02_tool_call_probe.py** — Reliability probe: three coding tasks through a write_file tool, graded by running the real tests, with one repair turn.

**labs/lab03_agent_config_generator.py** — Generate and validate Claude Code, OpenCode, Codex and Continue settings into .runs/, without touching global config.

Labs 01 and 02 use the Spark's Ollama when it answers, else the laptop's (labelled LAPTOP STAND-IN). Lab 03 is offline.

## Try it yourself

**Exercise 18 — grade an agent's reply.** An eval harness has to turn a raw model reply into a verdict. Open `week25/18_coding_agents/exercises/ex18_grade_agent_reply.py`. It has three `TODO`s:

1. `tool_args(tool_call)`: the arguments as a dict, whether the server sent a JSON string or a dict.
2. `safe_path(path)`: refuse absolute paths, `~`, `..` and backslashes, so an agent can't write outside the workspace.
3. `grade(reply, tests_passed)`: one of `no-tool-support`, `text-tool-call`, `prose-only`, `bad-args`, `unsafe-path`, `edit+pass`, `edit+fail`.

The fixtures follow the reply shapes lab 02 met on this Mac, plus three hand-made edge cases.

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/exercises/ex18_grade_agent_reply.py
```

**Expected output** (once all three TODOs are done)

```
✓ tool_args: JSON string → dict · dict passes through · bad JSON and lists → None
✓ safe_path: 2 allowed · 6 refused (empty, absolute, ~, .., nested .., backslash)
✓ grade: all 7 fixtures get the right verdict

▣ your grader, applied to the fixtures
│ lab 02 shape · clean tool call       → edit+pass
│ lab 02 shape · edit, tests fail      → edit+fail
│ lab 02 · tool call printed as text   → text-tool-call
│ lab 02 · gemma3:4b refuses tools     → no-tool-support
│ hand-made · explains instead         → prose-only
│ hand-made · truncated JSON           → bad-args
│ hand-made · escapes workspace        → unsafe-path
```

<details><summary>Hint — why is "a/../../b.py" unsafe when it starts with a normal folder?</summary>

Split on `/` and look for a `..` segment anywhere. `a/../../b.py` resolves to `../b.py`, one level above the workspace. Checking only `startswith("..")` misses it.

</details>

<details><summary>Stretch — grade lab 02's real output</summary>

Lab 02 writes `.runs/lab02_scorecard.json`. Map each record's `note` to your verdicts and print a pass rate per model. Then run lab 02 with `--trials 5` on your Spark and compare Qwen3.6 with the laptop models.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ollama launch` reports unknown command, or the pull fails with HTTP 412 | Ollama is too old for `launch` or the MTP Q4_K_M tag. Reinstall from ollama.com/download (CLI playbook) |
| The agent prints code but never edits the file | The tool call is missing or unparsed (lab 02: "printed as TEXT"). Use `ollama launch <agent>` with the playbook's Qwen3.6 model, as its Troubleshooting says |
| `does not support tools` (HTTP 400) | The model has no `tools` capability (`gemma3:4b` here). Pick one whose `/api/show` capabilities include `tools` |
| Claude Code answers are empty or cut off | A thinking model spent the budget on reasoning (lab 01, Step 4). Give it more output tokens, or disable thinking where your client allows it |
| The agent forgets files it read a minute ago | Context too small. Check the allocated context with lab 01, and set `num_ctx` / `OLLAMA_CONTEXT_LENGTH=64000` (Section 3) |
| `connection refused` to `localhost:11434` | Ollama is not running: `ollama serve` or `sudo systemctl start ollama` |
| Continue on the laptop can't connect | Port closed or bound to localhost: `ss -tuln \| grep 11434` on the Spark, then vibe-coding Step 2, or use the tunnel |
| `externally-managed-environment` when installing pytest | Make a venv first: `python3 -m venv .venv && source .venv/bin/activate` |
| Slow responses or OOM | Close other GPU work, keep the `q4_K_M` default, set `OLLAMA_MAX_LOADED_MODELS=1`. On unified memory the playbooks flush the cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |

## Next

Continue to [Lab 19 — multi-agent chatbot, RAG and knowledge graphs](../19_multi_agent_rag_kg/TUTORIAL.md): run a supervisor agent that delegates to coding, retrieval and vision specialists, turn text into a knowledge graph with txt2kg, and build a small retriever with citations.
