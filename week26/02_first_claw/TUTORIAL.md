# ▶ Reef Lab 02 — Your first claw on DGX Spark: verify, install, onboard, prove inference is local

> Part of Week 26 · NemoClaw on DGX Spark, beginner to expert. You type the commands, you see the real output. Laptop labs (the NAT and OpenShell CLIs, the policy model) run for real everywhere. Spark labs also run in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's docs or playbooks, or a clearly marked EXAMPLE.

**What you'll actually do**
- Check that your Spark is ready: DGX OS, a GB10, Docker without sudo, a kernel new enough for Landlock, and enough disk.
- Install NemoClaw with one command, and learn what Express Install picks for you.
- Write a scripted, non-interactive install line that you could hand to a second Spark — and check it before anyone pastes it.
- Learn the lifecycle commands, and which of them a lab may run for you, which wait for 🔓, and which are secrets.
- Prove, with three pieces of evidence, that your claw's inference never leaves the box.
- Add a Telegram channel, and install the Hermes and Deep Agents variants.

**Time** ~75 min · **Difficulty** beginner · **Hardware** 1 DGX Spark (without one: DRY + laptop)

**Sources:** the course's research tutorial *NemoClaw on DGX Spark — Beginner to Expert* (Part 1, labs L1.1–L1.7), which cites the [DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview) · [NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart) · [NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart) · [NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart) · [NemoClaw network policies](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies) · [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [NVIDIA Technical Blog — NemoClaw + OpenClaw on Spark](https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| A DGX Spark you can wipe | a fresh DGX OS, no personal data | the playbook says to run the demo on a fresh device or VM with no personal or confidential data |
| SSH to it | `ssh -o BatchMode=yes <spark> true` | the labs run read-only commands there; set it in 🖥 Spark setup |
| `sudo` on the Spark | you know the password | the installer and the Docker fixes need root — **you** type those, never a lab |
| The laptop OpenShell parser | `week26/.venv-openshell/bin/openshell --version` | Labs 02-3 and 02-4 check every OpenShell command offline |

```bash
# on: laptop
week26/.venv-openshell/bin/openshell --version
bash --version | head -n 1
```

**Expected output** (captured on this Mac)

```
openshell 0.0.111
GNU bash, version 3.2.57(1)-release (arm64-apple-darwin26)
```

> ⚠ **NemoClaw is alpha.** The research tutorial is explicit: an Apache-2.0 reference stack, alpha status, security reports to psirt@nvidia.com. An always-on agent can have broad access. The sandbox reduces that risk; it does not remove it (Module 01).

Three machines appear in this module. Keep them apart in your head:

| Where | What runs there | How you reach it |
|---|---|---|
| **this laptop** | the Reef runner, the labs, the OpenShell *parser*, `bash -n` | you are here |
| **the Spark host** | `nemoclaw`, `openshell`, Docker, the gateway, vLLM or Ollama | the ⌨ terminal with `# on: spark`, or a lab over SSH |
| **inside the sandbox** | the harness (OpenClaw / Hermes / Deep Agents), `inference.local` | `nemoclaw <s> connect`, or `openshell sandbox exec -n <s> -- …` |

✓ Checkpoint: the laptop parser prints its version, and you know which of the three places each command in this module runs in.

## 1 · L1.1 — Verify the Spark

Before you install anything, confirm the box is what you think it is. The playbook's check is three commands:

```bash
# on: spark
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
```

**Expected output** (REFERENCE — quoted from the DGX Spark NemoClaw playbook)

```
Expected: Ubuntu 24.04 (or your platform's supported OS), a detected NVIDIA GPU, Docker 28.x+.
```

The rest of the course needs four more facts, so Lab 02-1 checks them too:

| Check | Command | Pass | Why |
|---|---|---|---|
| Docker without sudo | `docker ps` | no `permission denied` | NemoClaw drives Docker as your user |
| NVIDIA runtime | `docker info` runtimes | `nvidia` is listed | vLLM containers run with `--runtime=nvidia` |
| Kernel | `uname -sr` | Linux **≥ 6.2** | Landlock ABI 3 — the filesystem layer you meet in Module 03 |
| Free disk | `df -BG` on `$HOME` | ≥ 200 GB (course rule of thumb) | large Express models can need hundreds of GB, plus the vLLM image |

The Reef runner's pass criterion for L1.1 is the short version: **GB10 detected, Docker present, kernel ≥ 6.2**.

If `docker ps` says `permission denied`, fix it yourself in the ⌨ terminal. The lab prints these lines; it never runs `sudo`:

```bash
# on: spark
sudo usermod -aG docker $USER && newgrp docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The last line is your proof that a container can see the GPU.

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_1_verify_spark.py`)

```
◈ DRY · SPARK_HOST not set · commands are shown, not run; outputs are RECORDED, REFERENCE or EXAMPLE (labelled)

▣ STEP 1 · the playbook's three checks — OS, GPU, Docker server version
$ head -n 2 /etc/os-release   [DRY]
  nvidia-smi
  docker info --format '{{.ServerVersion}}'
◈ REFERENCE — quoted from NVIDIA's playbook / docs (not your machine):
Expected: Ubuntu 24.04 (or your platform's supported OS), a detected NVIDIA GPU, Docker 28.x+.

▣ STEP 2 · the course's extra checks (one read-only command each)
$ docker ps   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
$ docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}'   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
io.containerd.runc.v2 nvidia runc
$ uname -sr   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
Linux 6.X.Y-NNNN-nvidia        ← EXAMPLE: your kernel release
$ free -g   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
               total        used        free      shared  buff/cache   available
Mem:            <T>         <U>         <F>         <S>         <B>         <A>
Swap:           <T>         <U>         <F>
$ df -BG --output=avail,target "$HOME" | tail -n 1   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
   <N>G /
$ node --version 2>/dev/null || echo "node: not installed"   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
v22.X.Y

▣ STEP 3 · the verdict
│ check                verdict              what LIVE mode looks for
│ ───────────────────  ───────────────────  ──────────────────────────────────
│ OS                   ◈ DRY — not checked  Ubuntu 24.04 / DGX OS
│ GPU                  ◈ DRY — not checked  NVIDIA GB10 in nvidia-smi
│ Docker server        ◈ DRY — not checked  28.x+
│ docker ps (no sudo)  ◈ DRY — not checked  no 'permission denied'
│ NVIDIA runtime       ◈ DRY — not checked  `nvidia` in docker info's runtimes
│ kernel ≥ 6.2         ◈ DRY — not checked  Landlock ABI 3 (Module 03)
│ memory (free -g)     ◈ DRY — not checked  info only
│ free disk ≥ 200 GB   ◈ DRY — not checked  course rule of thumb
│ Node.js              ◈ DRY — not checked  info only — the installer adds it
⚠ DRY: nothing above ran on a Spark. The EXAMPLE lines are shapes, not your machine.

▣ STEP 4 · the same kernel question, asked of THIS laptop (for real)
$ uname -sr   [this laptop]
│ this laptop: Darwin 27.0.0
◆ not Linux → no Landlock, no seccomp, no network namespaces here. That is why the sandbox runs on the Spark and this laptop only builds, parses and checks things.
═ Green on GB10, Docker and kernel ≥ 6.2 means the Spark is ready for the one-command installer (Lab 02-2).
```

Step 4 is real: this Mac runs Darwin, which has no Landlock, no seccomp and no network namespaces. That is why the sandbox lives on the Spark.

✓ Checkpoint: in LIVE mode the table shows ✓ for GPU, Docker server and kernel ≥ 6.2 — or you know which fix to type.

## 2 · L1.2 — One-command install (interactive)

One line installs everything. Run it yourself, on the Spark, in the ⌨ terminal:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

What happens, in order:

1. **A third-party software notice.** You accept it. (Scripted runs accept it with `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` or `--yes-i-accept-third-party-software` — see Section 3 for *where* that goes.)
2. **Node.js, OpenShell and the NemoClaw CLI** are installed. The installer needs Node.js 22.16+ and adds it if it is missing.
3. **`nemoclaw onboard` starts** when preflight passes. If the installer prints `To finish setup, run:`, run the `nemoclaw onboard` it shows before you connect.
4. **Express Install.** On a Spark you are asked `Run express install with these settings? [Y/n]:`. Express means managed local vLLM, a maintained Express model, the sandbox name `my-assistant`, and the Balanced policy. Take it the first time. Answer `n` to pick the agent, provider, model and name yourself.

> 📌 **The sandbox only exists after `nemoclaw onboard` completes.** Do not run `launch`, `connect` or `openclaw tui` before that. If `nemoclaw` is "not found" right after the install, run `source ~/.bashrc`.

When onboarding finishes, the playbook shows this summary:

**Expected output** (REFERENCE — quoted from the DGX Spark NemoClaw playbook)

```
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

Three things to notice. The model line says **(Local vLLM)**. The dashboard URL needs a token, and you get it with a separate command (Section 4 explains why a lab never prints it). And this summary spells the policy verb `policy-add`, while the research tutorial writes `policy add` — your release decides, and `--help` wins.

L1.2 passes when `nemoclaw --version` answers on the Spark. Lab 02-2 ends with that check.

✓ Checkpoint: you ran the one-liner in the ⌨ terminal (or you can say, step by step, what it would do), and `nemoclaw --version` answers.

## 3 · L1.3 — Scripted installs: the non-interactive line

With one Spark, the wizard is fine. With five, you want the same install every time. The quickstart documents a fully non-interactive first run. This one uses NVIDIA Endpoints (cloud):

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh | \
  NEMOCLAW_NON_INTERACTIVE=1 \
  NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
  NEMOCLAW_AGENT=openclaw \
  NEMOCLAW_PROVIDER=build \
  NVIDIA_INFERENCE_API_KEY=<your-key> \
  NEMOCLAW_SANDBOX_NAME=my-gpt-claw \
  bash
```

The rules, in one table:

| Rule | Why |
|---|---|
| Every `VAR=value` goes **after the `\|`**, in front of `bash` | an assignment in front of a command reaches only that command. In front of `curl`, only the downloader sees it; the installer never does. |
| Non-interactive needs `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` | nobody is there to accept the notice |
| `NEMOCLAW_AGENT` = `openclaw` \| `hermes` \| `langchain-deepagents-code` | picks the harness (and the CLI: `nemoclaw`, `nemohermes`, `nemo-deepagents`) |
| `NEMOCLAW_POLICY_TIER` = `restricted` \| `balanced` \| `open` \| `personal` | the policy tier (below) |
| `NEMOCLAW_SANDBOX_NAME` — lowercase letters, digits, `-` | the course runner accepts `^[a-z0-9-]{1,40}$`; stay inside it |
| Setting `NEMOCLAW_PROVIDER` (or `NEMOCLAW_NO_EXPRESS=1`) skips Express | the playbook says so; you chose instead |
| Keys stay `<placeholders>` in anything you share | and replace them before you run: unquoted, bash reads `<` and `>` as redirections |

`NEMOCLAW_PROVIDER` values, from the quickstart and the playbook's provider mapping:

| Value | What it is | Inference runs | Key variable |
|---|---|---|---|
| `vllm` | an already-running vLLM on `localhost:${NEMOCLAW_VLLM_PORT:-8000}` | **on the Spark** | — |
| `install-vllm` | managed, Docker-backed vLLM (large download) | **on the Spark** | — |
| `ollama` | local Ollama, optional `NEMOCLAW_MODEL` | **on the Spark** | — |
| `build` | NVIDIA Endpoints | cloud | `NVIDIA_INFERENCE_API_KEY` |
| `routed` | Model Router | cloud | `NVIDIA_INFERENCE_API_KEY` |
| `openrouter` · `openai` · `anthropic` · `gemini` | hosted APIs | cloud | `OPENROUTER_API_KEY` · `OPENAI_API_KEY` · `ANTHROPIC_API_KEY` · `GEMINI_API_KEY` |
| `custom` · `anthropicCompatible` | any OpenAI- / Anthropic-compatible endpoint | wherever it points | `COMPATIBLE_API_KEY` · `COMPATIBLE_ANTHROPIC_API_KEY` |
| `hermes-provider` | Hermes Provider | — | Hermes only |

Note that `local-vllm` is **not** in that list. It is the *OpenShell provider name* the OpenShell playbook creates (you will see it in `openshell inference get`), not a `NEMOCLAW_PROVIDER` value.

The four policy tiers, from the network-policies reference:

| Tier | What it allows |
|---|---|
| `restricted` | the baseline only |
| `balanced` (default) | `npm`, `pypi`, `huggingface`, `brew`, `brave`, plus Tavily when you select it |
| `open` | adds messaging, `jira`, `outlook`, `weather`, `public-reference` |
| `personal` | mandatory `personal-open-internet`: any binary may reach ports 80/443 at L4. Never on shared hardware. |

More switches from the quickstarts: `NEMOCLAW_WEB_SEARCH_PROVIDER=tavily|none` (Tavily also needs `TAVILY_API_KEY`), `NEMOCLAW_GATEWAY_RUNTIME=podman`, `--defer-onboarding` (install the CLI without a provider or sandbox), and a pinned release when you need repeatability:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh | NEMOCLAW_INSTALL_REF= NEMOCLAW_INSTALL_TAG=vX.Y.Z bash
```

Lab 02-2 builds these lines from choices, checks them, and asks the real bash on this laptop to parse them (`bash -n` reads the grammar and runs nothing). Step 2 proves the pipe rule with `echo` standing in for `curl`:

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_2_install_line.py`, steps 2–5)

```
▣ STEP 2 · why the variables go on the bash side of the pipe — a real demo on this laptop
◆ `echo` stands in for curl: it prints a one-line 'installer' that reports what IT can see.
$ NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 echo 'echo "installer sees ACCEPT=${NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE:-<unset>}"' | bash   [this laptop]
installer sees ACCEPT=<unset>
$ echo 'echo "installer sees ACCEPT=${NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE:-<unset>}"' | NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 bash   [this laptop]
installer sees ACCEPT=1
✓ an assignment in front of a command reaches only THAT command: curl gets it, the installer (bash) does not

▣ STEP 3 · build the lines from choices
◆ wrote 6 lines to week26/02_first_claw/.runs/install_*.sh — bash -n parses them, it never runs them
$ for f in .runs/install_*.sh; do bash -n "$f" && echo "syntax ok  $f"; done   [this laptop]
syntax ok  install_existing_vllm__8000.sh
syntax ok  install_hermes__locked_down.sh
syntax ok  install_local_ollama.sh
syntax ok  install_managed_vllm.sh
syntax ok  install_nvidia_endpoints__cloud.sh
syntax ok  install_pinned_release.sh
│ variant                   NEMOCLAW_PROVIDER  inference runs      Express  lint  bash -n
│ ────────────────────────  ─────────────────  ──────────────────  ───────  ────  ───────
│ existing vLLM :8000       vllm               ◆ on this Spark     skipped  ✓     ✓
│ managed vLLM              install-vllm       ◆ on this Spark     skipped  ✓     ✓
│ local Ollama              ollama             ◆ on this Spark     skipped  ✓     ✓
│ Hermes, locked down       vllm               ◆ on this Spark     skipped  ✓     ✓
│ NVIDIA Endpoints (cloud)  build              ⚠ leaves the Spark  skipped  ✓     ✓
│ pinned release            (Express/wizard)   wizard decides      offered  ✓     ✓

$ curl -fsSL https://www.nvidia.com/nemoclaw.sh | \   [NOT RUN — paste it in the ⌨ terminal, # on: spark]
    NEMOCLAW_NON_INTERACTIVE=1 \
    NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
    NEMOCLAW_AGENT=openclaw \
    NEMOCLAW_SANDBOX_NAME=my-assistant \
    NEMOCLAW_PROVIDER=vllm \
    NEMOCLAW_VLLM_PORT=8000 \
    NEMOCLAW_POLICY_TIER=balanced \
    bash

$ curl -fsSL https://www.nvidia.com/nemoclaw.sh | \   [NOT RUN — paste it in the ⌨ terminal, # on: spark]
    NEMOCLAW_NON_INTERACTIVE=1 \
    NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
    NEMOCLAW_AGENT=hermes \
    NEMOCLAW_SANDBOX_NAME=my-hermes \
    NEMOCLAW_PROVIDER=vllm \
    NEMOCLAW_WEB_SEARCH_PROVIDER=none \
    NEMOCLAW_POLICY_TIER=restricted \
    bash
◆ Keys are written as '<your-key>' placeholders on purpose. For the NVIDIA Endpoints line, prompts leave the Spark — the provider trust table lists local Ollama as 'no data leaves the machine', not cloud endpoints.

▣ STEP 4 · the mistakes the checker catches
│ vars before curl             → NEMOCLAW_NON_INTERACTIVE, NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE set on the curl side of the pipe — curl only downloads the script; the bash process that runs it never sees these. Move them after the `|`.
│ notice not accepted          → NEMOCLAW_NON_INTERACTIVE=1 without NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 — nobody is there to accept the third-party software notice
│ provider typo                → NEMOCLAW_PROVIDER='local-vllm' is not a documented value — one of build, openrouter, openai, anthropic, gemini, routed, custom, anthropicCompatible, ollama, vllm, install-vllm, hermes-provider
│ tier typo                    → NEMOCLAW_POLICY_TIER='strict' — use one of restricted, balanced, open, personal
│ bad sandbox name             → NEMOCLAW_SANDBOX_NAME='My_Claw' — use lowercase letters, digits and - (course rule ^[a-z0-9-]{1,40}$, the runner's)
│ hermes-provider on OpenClaw  → hermes-provider works with NEMOCLAW_AGENT=hermes only
│ tavily with no key           → tavily web search needs TAVILY_API_KEY in a non-interactive run (as a placeholder here)
│ a real-looking key           → NVIDIA_INFERENCE_API_KEY looks like a REAL key — never type one into a line you share, paste or record (course rule); keep a <placeholder> here
✓ 8/8 broken lines refused before anyone pasted them

▣ STEP 5 · the placeholder trap — what bash does with an unedited <key> (real, in .runs/)
$ bash -c 'TAVILY_API_KEY=<key> NEMOCLAW_POLICY_TIER=restricted env'   [this laptop]
bash: key: No such file or directory
⚠ bash read `<key>` as 'take input from a file called key' (and `>` would have written a file named after the next word). Replace every <placeholder> before you paste a line.
```

Step 5 is the trap in the documented lines: an unedited `<key>` makes bash look for a file called `key`. Here it fails loudly. In the exercise line, the `>` would also turn the next word into a file name, so the tier would silently not be set.

✓ Checkpoint: you can explain why `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 curl … | bash` does not accept anything — and Lab 02-2 shows 8/8 broken lines refused.

## 4 · L1.4 — The lifecycle commands you will use every day

These are the verbs from the quickstart, the NVIDIA blog and the Deep Agents quickstart. The course sorts them by what a lab may do with them:

| Command | Kind | How the course runs it |
|---|---|---|
| `nemoclaw my-assistant status` | read-only | a lab runs it (LIVE) |
| `nemoclaw my-assistant policy list` | read-only | a lab runs it (LIVE) |
| `nemoclaw list` · `openshell sandbox list` · `openshell forward list` | read-only | a lab runs it (LIVE) |
| `nemoclaw my-assistant policy add <preset> --dry-run` | preview | shows the merge, changes nothing |
| `nemoclaw my-assistant snapshot create --name before-change` | change | `change()` — only with 🔓 |
| `nemoclaw my-assistant restart` · `stop` · `start` | change | `change()` — only with 🔓 |
| `nemoclaw inference set --model <model> --provider <provider> --sandbox my-assistant` | change, hot | the sandbox keeps running |
| `nemoclaw my-assistant rebuild` · `nemoclaw onboard --recreate-sandbox` | change, recreates | `change()` — only with 🔓 |
| `nemoclaw onboard --fresh --gpu` | **destroys** and recreates | ⌨ terminal only |
| `nemoclaw launch my-assistant` · `nemoclaw my-assistant connect` | interactive | ⌨ terminal only |
| `nemoclaw my-assistant logs --follow` · `openshell term` | streaming | ⌨ terminal only |
| `nemoclaw my-assistant dashboard-url --quiet` · `gateway-token --quiet` | **secret** | ⌨ terminal only |
| `nemoclaw upgrade-sandboxes --auto` · `nemoclaw credentials reset <PROVIDER> && nemoclaw onboard` | change | ⌨ terminal only |

The everyday read-only set:

```bash
# on: spark
nemoclaw list
nemoclaw my-assistant status
nemoclaw my-assistant policy list
openshell sandbox list
openshell forward list
```

**Why the token commands stay in your terminal.** `dashboard-url --quiet` prints `http://127.0.0.1:18789/#token=<token>`. That token is a bearer credential for the agent's Control UI: whoever has it can chat with, and steer, your always-on agent. A lab's output is streamed into the runner, kept in its run history, and can be RECORDED. So you type these two yourself, on the Spark:

```bash
# on: spark
nemoclaw my-assistant dashboard-url --quiet
```

To open the dashboard from your laptop, forward the port. Use `127.0.0.1`, not `localhost` — the gateway's origin check wants an exact match:

```bash
# on: laptop
ssh -L 18789:127.0.0.1:18789 <you>@<spark>
```

On the Spark itself, `openshell forward start 18789 my-assistant --background` starts the forward, and `openshell forward list` shows it. The playbook notes the port is auto-assigned (commonly 18789 or 18790), so read it from the URL.

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_3_lifecycle.py`, steps 2–5)

```
▣ STEP 2 · the same OpenShell commands, parsed by the real CLI on THIS laptop
$ openshell sandbox list   [this laptop]
$ openshell sandbox get my-assistant   [this laptop]
$ openshell forward list   [this laptop]
$ openshell logs my-assistant --source sandbox -n 20   [this laptop]
$ openshell status   [this laptop]
│ command                                             laptop CLI 0.0.111            last line it printed
│ ──────────────────────────────────────────────────  ────────────────────────────  ────────────────────────────────────
│ openshell sandbox list                              ✓ parsed · needs the gateway  ╰─▶ Connection refused (os error 61)
│ openshell sandbox get my-assistant                  ✓ parsed · needs the gateway  ╰─▶ Connection refused (os error 61)
│ openshell forward list                              ✓ ran locally                 No active forwards.
│ openshell logs my-assistant --source sandbox -n 20  ✓ parsed · needs the gateway  ╰─▶ Connection refused (os error 61)
│ openshell status                                    ✓ parsed · needs the gateway  ╰─▶ Connection refused (os error 61)
◆ `forward list` is local state (forwards are processes on the machine you type on), so it answers even here. The laptop CLI is 0.0.111; NemoClaw pins 0.0.116 on the Spark — `--help` wins there.

▣ STEP 3 · every L1.4 verb, sorted — what may a lab run for you? (<s> = my-assistant)
│ command                                            kind                  how the course runs it
│ ─────────────────────────────────────────────────  ────────────────────  ───────────────────────────────────────
│ nemoclaw <s> status                                read-only             sh() — runs in LIVE
│ nemoclaw <s> policy list                           read-only             sh() — runs in LIVE
│ nemoclaw list · openshell sandbox|forward list     read-only             sh() — runs in LIVE
│ nemoclaw <s> policy add <preset> --dry-run         preview               sh() — shows the merge, changes nothing
│ nemoclaw <s> snapshot create --name before-change  change                change() — needs 🔓
│ nemoclaw <s> restart · stop · start                change                change() — needs 🔓
│ nemoclaw inference set --model … --sandbox <s>     change (hot)          change() — the sandbox keeps running
│ nemoclaw <s> rebuild · onboard --recreate-sandbox  change (recreates)    change() — needs 🔓
│ nemoclaw onboard --fresh --gpu                     DESTROYS + recreates  never from a lab — ⌨ terminal only
│ nemoclaw launch <s> · nemoclaw <s> connect         interactive           ⌨ terminal only (needs a TTY)
│ nemoclaw <s> logs --follow · openshell term        streaming             ⌨ terminal only (never ends)
│ nemoclaw <s> dashboard-url · gateway-token         SECRET                ⌨ terminal only — never printed here

▣ STEP 4 · the change gate — snapshot first, then restart (L1.4: watch the status change)
$ nemoclaw my-assistant status   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
(the status block from step 1 — read-only preview)
$ nemoclaw my-assistant snapshot create --name before-change   [DRY]
◈ (dry run — the change above would be applied here)
$ nemoclaw my-assistant restart   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
<restart output>        ← EXAMPLE
$ nemoclaw my-assistant status   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
Phase:     <phase after restart>        ← EXAMPLE shape
⚠ DRY: nothing was snapshotted or restarted. In LIVE mode with 🔓 on, the two status blocks are your evidence.

▣ STEP 5 · why this lab never runs dashboard-url or gateway-token
│ `nemoclaw <s> dashboard-url --quiet` prints http://127.0.0.1:18789/#token=<token>. That token is a bearer
│ credential for the agent's Control UI: whoever holds it can chat with, and steer, your always-on agent.
│ A lab's output is streamed into the runner's console, kept in its run history, and can be RECORDED.
│ clawkit redacts the shape it knows:  http://127.0.0.1:18789/#token=EXAMPLE-not-a-real-token  →  http://127.0.0.1:18789/#token=•••
◆ …but a redactor is a safety net, not a plan. Run those two commands yourself in the ⌨ terminal, on the Spark, and open the URL there (or through `ssh -L 18789:127.0.0.1:18789 <you>@<spark>` — use 127.0.0.1, not localhost).
═ Read-only verbs run for you; changes wait for 🔓; interactive, streaming and secret verbs are yours to type.
```

`forward list` answers even on the laptop, because forwards are local processes. Everything else parsed and then stopped at the missing gateway, which is exactly what the laptop CLI can prove.

✓ Checkpoint: for any verb in the table you can say whether a lab may run it, and why `dashboard-url` is never one of them.

## 5 · L1.5 — First conversation, and proving inference is local

Say hello first. Connect to the sandbox, and send one request to `inference.local` — the same test the OpenShell playbook uses:

```bash
# on: spark
# inside the sandbox (nemoclaw my-assistant connect)
curl https://inference.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<MODEL_HANDLE>", "messages": [{"role":"user","content":"Say hello from the Spark."}]}'
```

There is no API key in that request. The supervisor intercepts `inference.local`, and the gateway adds the real credential and forwards the call to the provider you set. The NVIDIA blog also gives a non-interactive smoke test:

```bash
# on: spark
# inside the sandbox (nemoclaw my-assistant connect)
openclaw agent --agent main --local -m "hello" --session-id test
```

A friendly answer is not proof that the answer came from *your* GPU. For that you need three pieces of evidence:

| # | Evidence | Command | Pass |
|---|---|---|---|
| 1 | the route | `openshell inference get` (on the host) | the provider is local: `ollama`, `local-vllm`, `vllm` |
| 2 | the models, seen from inside | `openshell sandbox exec -n my-assistant -- curl -s https://inference.local/v1/models` | a model list comes back |
| 3 | the cloud, blocked | inside: `curl https://api.openai.com/v1/models` | it does **not** get through |

1 + 2 are the Reef runner's L1.5 pass criterion. 3 is Part 1's exercise 3: no `network_policies` entry matches, so the proxy denies the connection. You see it in `openshell term` (host; `f` follow, `s` filter by source, `q` quit) and in `openshell logs my-assistant --source sandbox`. The docs are explicit that the fix is **never** to add `api.openai.com` or `api.anthropic.com` to a policy.

```bash
# on: spark
openshell inference get
openshell sandbox exec -n my-assistant -- curl -s https://inference.local/v1/models
openshell sandbox exec -n my-assistant -- curl -sS -o /dev/null -w '%{http_code}\n' --max-time 15 https://api.openai.com/v1/models
openshell logs my-assistant --source sandbox -n 50 --since 5m
```

**Expected output** (REFERENCE — quoted from the OpenShell playbook, for `openshell inference get`)

```
Expected output should show `provider: local-vllm` and your chosen `model`.
```

A provider *name* is only a label. `openshell provider list` shows where it points; local means the Spark's own IP, not `localhost`, because the gateway runs inside Docker.

Lab 02-4 runs the Spark half read-only and the laptop half for real: the OpenShell parser accepts each command, the course's policykit model explains each decision, and a LAPTOP STAND-IN shows the shape of a `/v1/models` answer.

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_4_inference_is_local.py`, steps 4–6)

```
▣ STEP 4 · the call that must fail: api.openai.com from inside the sandbox (Part 1, exercise 3)
$ openshell sandbox exec -n my-assistant -- curl -sS -o /dev/null -w '%{http_code}\n' --max-time 15 https://api.openai.com/v1/models   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
curl: (<n>) <the proxy refused the CONNECT>
000        ← EXAMPLE shape — no HTTP answer from OpenAI
$ openshell logs my-assistant --source sandbox -n 50 --since 5m   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
<time> <level> sandbox … deny … api.openai.com:443 …        ← EXAMPLE shape — look for your host
◆ Where you see it: `openshell term` on the host (live, `f` follow · `s` source · `q` quit) and `openshell logs my-assistant --source sandbox`. The fix is NOT to add api.openai.com to the policy.

▣ STEP 5 · the laptop half — why each call went the way it did (real, offline)
$ openshell inference get   [this laptop]
$ openshell sandbox exec -n my-assistant -- curl -s https://inference.local/v1/models   [this laptop]
$ openshell sandbox exec -n my-assistant -- curl https://api.openai.com/v1/models   [this laptop]
$ openshell logs my-assistant --source sandbox -n 50 --since 5m   [this laptop]
│ command                                    laptop CLI 0.0.111
│ ─────────────────────────────────────────  ────────────────────────────
│ step 1 · inference get                     ✓ parsed · needs the gateway
│ step 2 · exec … inference.local/v1/models  ✓ parsed · needs the gateway
│ step 4 · exec … api.openai.com/v1/models   ✓ parsed · needs the gateway
│ step 4 · logs --source sandbox -n 50       ✓ parsed · needs the gateway
│ curl → inference.local:443 GET /v1/models  ◆ inspect_for_inference
│                                              inference.local is handled by the proxy's inference routing, forwarded to the provider set with `openshell inference set`
│ curl → api.openai.com:443 GET /v1/models   ✕ deny
│                                              api.openai.com:443 is not in network_policies (default deny egress)
│ python3 → integrate.api.nvidia.com:443     ✕ deny
│                                              integrate.api.nvidia.com:443 is not in network_policies (default deny egress)
│ python3 → 192.168.1.42:8000                ✕ deny
│                                              private RFC 1918 IP: blocked unless declared as an exact host or opened with a narrow allowed_ips CIDR
│ curl → 169.254.169.254:80                  ✕ deny
│                                              169.254.169.254 is loopback / link-local / 0.0.0.0 — always blocked, even with allowed_ips (SSRF)
◆ policykit is the course's TEACHING MODEL, not OpenShell. The policy above is a stand-in for the Restricted tier (no presets); read your real one with `nemoclaw <s> policy get` in Module 03. 192.168.1.42 is the playbook's example Spark IP: even the local vLLM is reachable only through inference.local.
✕ HIGH openai.endpoints[0]: api.openai.com is an inference provider — never in policy; route via inference.local
◆ That is the tempting 'fix' for step 4 — and the checklist refuses it: inference goes through inference.local.
→ GET http://localhost:11434/v1/models · Ollama on THIS laptop (LAPTOP STAND-IN, not the Spark)
◆ LAPTOP STAND-IN · object=list · 5 local models · first: nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b
◆ Same OpenAI-compatible shape the sandbox gets from inference.local: {"object":"list","data":[{"id":…}]}

▣ STEP 6 · the verdict (L1.5)
│ proof                                           verdict
│ ──────────────────────────────────────────────  ──────────────────
│ provider is local (ollama / local-vllm / vllm)  ◈ DRY — not proven
│ models list returned from inside the sandbox    ◈ DRY — not proven
│ api.openai.com denied from inside the sandbox   ◈ DRY — not proven
⚠ DRY: the laptop half is real; the Spark half is not your machine. Connect a Spark and run it LIVE.
═ Local route + models from inside + the cloud call denied = prompts and data stay on the Spark.
```

Note the fourth decision. Even your own vLLM on the Spark's LAN IP is denied from inside the sandbox. The only road to a model is `inference.local`.

✓ Checkpoint: in LIVE mode the verdict shows a local provider, a model list and a blocked OpenAI call — or you can say which of the three is missing.

## 6 · L1.6 — Add a Telegram channel

Telegram is optional. Skip it for a first install; the Web UI and `openclaw tui` are enough.

1. In Telegram, open `@BotFather`, send `/newbot`, and copy the bot token.
2. Register the channel. Paste the token when the wizard asks — never on the command line:

```bash
# on: spark
nemoclaw my-assistant channels add telegram
```

NemoClaw stores the credential and **rebuilds** the sandbox so OpenClaw can use the channel. That is a change, so it is yours to type.

3. The wizard asks for an optional Telegram user ID to restrict who can DM the bot. Skip it and the bot asks for pairing first. Approve the code it sends, inside the sandbox:

```bash
# on: spark
# inside the sandbox (nemoclaw my-assistant connect)
openclaw pairing approve telegram <CODE>
```

4. If messages fail with policy errors, check that the `telegram` preset is applied (the playbook's spelling):

```bash
# on: spark
nemoclaw my-assistant policy-list
nemoclaw my-assistant policy-add telegram
```

Telegram uses long-polling, so no public URL or cloudflared tunnel is needed. Know the documented risk: the `telegram` preset only opens the Telegram Bot API, but the agent can then message **any chat the bot token can reach**.

| Symptom | First look |
|---|---|
| the bot never answers | `nemoclaw my-assistant status`, then `nemoclaw my-assistant logs` (no `--follow`) |
| `409 Conflict` after a rebuild | another process uses the same bot token |
| it receives but does not reply | an inference failure, a policy denial, or the allowlist / mention gate — the logs say which |

✓ Checkpoint: a message sent to your bot comes back from the agent (the runner asks you to confirm it; it cannot see your phone).

## 7 · L1.7 — Hermes and Deep Agents variants

The same installer builds the other two harnesses. Only `NEMOCLAW_AGENT` and the CLI name change:

```bash
# on: spark
# Hermes, sandbox named my-hermes
curl -fsSL https://www.nvidia.com/nemoclaw.sh | NEMOCLAW_AGENT=hermes NEMOCLAW_SANDBOX_NAME=my-hermes bash
nemohermes my-hermes status
nemohermes my-hermes connect

# LangChain Deep Agents Code
curl -fsSL https://www.nvidia.com/nemoclaw.sh | NEMOCLAW_AGENT=langchain-deepagents-code NEMOCLAW_SANDBOX_NAME=my-deepagents bash
nemo-deepagents my-deepagents status
nemo-deepagents my-deepagents connect
```

The playbook's starter prompt names the equivalent onboarding commands too: `nemohermes onboard` for Hermes and `nemo-deepagents onboard` for Deep Agents.

| | Hermes (`nemohermes`) | Deep Agents (`nemo-deepagents`) |
|---|---|---|
| state dir | `/sandbox/.hermes` | `/sandbox/.deepagents` |
| good at | OpenAI-compatible API on 8642, Tavily, Langfuse (Module 05) | planning with sub-agents, coding |
| messaging during onboarding | yes | skipped |
| local Ollama | offered | not offered unless the docs add support (playbook starter prompt) |

The L1.7 pass criterion is that both sandboxes appear in `openshell sandbox list`. Every sandbox is a separate claw with its own policy, so three sandboxes means three policies to read in Module 03.

✓ Checkpoint: `openshell sandbox list` shows `my-hermes` and `my-deepagents` next to `my-assistant` — or you can explain why you skipped them.

## Labs — run them here

**labs/lab02_1_verify_spark.py** — Verify the Spark: OS, GB10, Docker, kernel for Landlock, and free disk.

**labs/lab02_2_install_line.py** — Build and check the one-command NemoClaw installer line, without running it.

**labs/lab02_3_lifecycle.py** — The read-only lifecycle commands, the change gate, and why tokens are never printed.

**labs/lab02_4_inference_is_local.py** — Prove inference is local: the route, the models, one hello, and a call that must fail.

## Try it yourself

`exercises/ex02_install_line.py` has three TODOs:

1. **(Part 1, exercise 2)** Write the non-interactive install line for a Hermes claw named `alto-hermes` that uses an already-running vLLM on port 8000, Tavily search, and the `restricted` tier.
2. **(Part 1, exercise 5)** Which process must see `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1`: `curl` or `bash`?
3. **(Part 1, exercise 4)** Change the model of `my-assistant` once without destroying the sandbox, and once in a way that does.

The checker parses your line (it never runs it), runs `bash -n` on it, and checks the two answers.

```bash
# on: laptop
.venv/bin/python week26/02_first_claw/exercises/ex02_install_line.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ install line: hermes · alto-hermes · vllm on 8000 · tavily (key as a placeholder) · restricted · all on the bash side · bash -n OK
✓ the bash side: the variable must be in the environment of the bash process that RUNS the script
✓ model change: `inference set` is hot (route only) · `onboard --fresh` destroys and recreates

⚠ Verify the combination on your unit: each variable is documented on its own (quickstart, Hermes quickstart, network-policies reference), but the docs do not show this exact combination.
═ Done. Paste the line into the ⌨ terminal on a Spark (# on: spark) when you are ready — the course never will.
```

Keep that last warning. The research tutorial says it plainly: each variable is documented on its own, but the docs do not show this exact combination. Verify it on your unit.

<details><summary>Hint — TODO 1</summary>

Start from the documented quickstart line in Section 3. Swap the agent, the name and the provider, then add four variables: the vLLM port, the web-search provider, the Tavily key (as a placeholder) and the tier. Every one of them goes after the `|`.

</details>

<details><summary>Hint — TODO 3</summary>

Section 4's verb table has one line marked "change, hot" and one marked "destroys and recreates". The hot one changes the inference route, which is a dynamic control; the other recreates the sandbox, whose image and blueprint are not.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nemoclaw: command not found` after the install | the shell PATH was not reloaded | `source ~/.bashrc`, or open a new terminal |
| `docker ps` → `permission denied` | your user is not in the `docker` group | Section 1's four lines, in the ⌨ terminal |
| installer fails on Node.js | Node.js older than 22.16 | install Node.js 22.16+, then re-run the installer |
| `connect` or `openclaw tui` fails right after install | onboarding has not completed yet | run the `nemoclaw onboard` the installer printed, and wait for "OpenClaw is ready" |
| gateway: "port 8080 is held by container…" | another OpenShell gateway is running | `nemoclaw onboard` (or `nemoclaw onboard --resume`) reuses or recreates it |
| inference hangs | vLLM is still loading | on the host, `curl http://127.0.0.1:8000/v1/models`; wait for `Application startup complete` |
| Web UI says `origin not allowed` | you used `localhost` | use `http://127.0.0.1:18789/#token=…` |
| `policy list` is "unknown command" | your release spells it `policy-list` | Lab 02-3 tries both; `--help` wins |
| Lab 02-4 says `⚠ inconclusive` for the OpenAI test | `sandbox exec` itself failed (wrong sandbox name, no gateway) | check `openshell sandbox list` and `CLAW_SANDBOX` in 🖥 Spark setup |

## Next

[Lab 03 — OpenShell sandboxes and policy as code](../03_policy_as_code/TUTORIAL.md): read the policy NemoClaw created for you, write your own, run the deny → observe → allow → verify loop, and bring your own vLLM.
