# ▶ Spark Lab 01 — Meet your DGX Spark: connect, check, and budget memory

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn the handful of hardware numbers that decide everything this week: 128 GB unified memory, 273 GB/s bandwidth, FP4.
- Connect to your Spark the official ways (NVIDIA Sync, manual SSH, Tailscale), then point the Lab Runner at it.
- Run the **Spark doctor**: eight read-only checks that your box is ready.
- Open **DGX Dashboard** through an SSH tunnel, and build one tunnel for every port this week uses.
- Work out, with arithmetic, which models fit on one Spark and which need two.

**Time** ~40 min · **Difficulty** beginner · **Hardware** 1 DGX Spark (or none: DRY mode + arithmetic labs)

**Official playbooks covered:** [Connect to your Spark](https://build.nvidia.com/spark/connect-to-your-spark) · [Tailscale](https://build.nvidia.com/spark/tailscale) · [DGX Dashboard](https://build.nvidia.com/spark/dgx-dashboard) · [VS Code](https://build.nvidia.com/spark/vscode)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs the labs and the Lab Runner |
| An SSH client | `ssh -V` | every Spark lab drives the Spark over SSH |
| A DGX Spark on your network or tailnet | `ping <spark>.local` or `tailscale status` | optional: without one, everything runs DRY |
| Your Spark username and password | from first boot | for the first SSH login |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
ssh -V
```

**Expected output**

```
Python 3.13.13
OpenSSH_10.2p1, LibreSSL 3.3.6
```

> 🔐 Nothing in this course stores a password. SSH uses a **key**. Tokens you add later (Hugging Face, NGC) are saved server-side in `week25/.env.local`, which is gitignored and never sent to the browser.

✓ Checkpoint: `ssh -V` prints a version, and you know your Spark's hostname (it is printed on the quick-start card, e.g. `spark-abcd`).

## 1 · What a DGX Spark is, in one picture

A DGX Spark is a small desktop computer built around one **NVIDIA GB10 Grace Blackwell Superchip**: an Arm CPU and a Blackwell GPU in the same package, sharing one pool of memory.

| Spec | Value | What it means for you this week |
|---|---|---|
| Memory | **128 GB LPDDR5x, unified** | CPU and GPU share it. Models up to ~200B parameters fit at 4-bit on one Spark |
| Memory bandwidth | **273 GB/s** | caps how fast *one* conversation can generate tokens (Section 7) |
| AI compute | **1 PFLOP at FP4** (sparse) | why the NVFP4 format in Module 07 matters |
| CPU | 20 Arm cores (10× Cortex-X925 + 10× Cortex-A725) | everything is **aarch64**: pick ARM64 containers and wheels |
| Network | ConnectX-7, 2× QSFP, 200 Gb/s | cable two Sparks together in Module 02 |
| Storage | up to 4 TB NVMe | models are big: plan ~200 GB for this week |
| OS | DGX OS (Ubuntu 24.04) with CUDA, Docker, and the NVIDIA Container Toolkit | the playbooks assume it |

```text
   your laptop                                   DGX Spark (GB10)
 ┌──────────────┐    ssh · tailnet · tunnels    ┌──────────────────────────────────────┐
 │ Lab Runner   │ ─────────────────────────────►│ Grace CPU (20 Arm cores)             │
 │ :8125        │                               │        ▲                             │
 │ labs/*.py    │ ◄──── HTTP :8000 :11434 … ────│        │  one 128 GB unified pool    │
 └──────────────┘                               │        ▼                             │
                                                │ Blackwell GPU  ·  273 GB/s  ·  FP4    │
                                                └──────────────────────────────────────┘
```

**Unified memory changes one habit.** On a normal PC you ask "does the model fit in the GPU's VRAM?". On a Spark there is no separate VRAM: the model, the KV cache, your Python process and the OS all share 128 GB. So `free -g` is the honest memory meter, and closing a forgotten Jupyter kernel can free memory for your model.

✓ Checkpoint: you can say why a Spark can load a 70B model that would not fit on a 24 GB gaming GPU, and which number limits how fast it generates.

## 2 · Connect: NVIDIA Sync or plain SSH

NVIDIA's [Connect to your Spark](https://build.nvidia.com/spark/connect-to-your-spark) playbook offers two paths:

- **NVIDIA Sync** (a desktop app for macOS, Windows, Linux) adds the Spark once, then opens a terminal, VS Code, Cursor or DGX Dashboard in one click. It creates the SSH key and the tunnels for you.
- **Manual SSH** is what this course automates, so do it once by hand.

First check that the Spark's mDNS name resolves on your network:

```bash
# on: laptop
ping -c 3 spark-abcd.local
```

**Expected output**

```
PING spark-abcd.local (10.9.1.9): 56 data bytes
64 bytes from 10.9.1.9: icmp_seq=0 ttl=64 time=6.902 ms
64 bytes from 10.9.1.9: icmp_seq=1 ttl=64 time=116.335 ms
64 bytes from 10.9.1.9: icmp_seq=2 ttl=64 time=33.301 ms
```

(That block is quoted from the playbook, so the name and IP are the playbook's.) If you see `cannot resolve … Unknown host`, mDNS is blocked, which is common on corporate Wi-Fi. Use the IP address from your router, or use Tailscale (next section).

Log in once with your password, and confirm you are on the Spark:

```bash
# on: laptop
ssh <you>@spark-abcd.local
hostname && uname -m
exit
```

Now switch to key-based login. The Lab Runner runs commands with `ssh -o BatchMode=yes`, which **never asks for a password**, so a key is required:

```bash
# on: laptop
ssh-keygen -t ed25519 -f ~/.ssh/spark -N ""        # skip if you already have a key you want to use
ssh-copy-id -i ~/.ssh/spark.pub <you>@spark-abcd.local
cat >> ~/.ssh/config <<'EOF'
Host spark-a
  HostName spark-abcd.local
  User <you>
  IdentityFile ~/.ssh/spark
EOF
ssh -o BatchMode=yes spark-a 'echo key login works on $(hostname)'
```

The `Host spark-a` alias means every later command can say `ssh spark-a`. If you have a second Spark, add `Host spark-b` the same way now; Module 02 uses it.

✓ Checkpoint: `ssh -o BatchMode=yes spark-a true` returns straight away with no password prompt.

## 3 · Reach it from anywhere: Tailscale

mDNS works only on the same local network. The [Tailscale playbook](https://build.nvidia.com/spark/tailscale) puts the Spark and your laptop on a private encrypted network (a *tailnet*), so `ssh spark-a` works from home, the office, or a café.

On the **Spark**, add Tailscale's apt repository and install it (these are the playbook's commands for Ubuntu 24.04 "noble"):

```bash
# on: spark
sudo apt update && sudo apt install -y curl gnupg
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | \
  sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | \
  sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale
sudo tailscale up          # opens a login URL: sign in with the SAME account you use on the laptop
```

On the **laptop**, install the Tailscale app, log in with the same account, then:

```bash
# on: laptop
tailscale status
tailscale ping spark-abcd
```

Point your SSH alias at the tailnet name (for example `spark-abcd` or `spark-abcd.<tailnet>.ts.net`) instead of `.local`, and you can reach the Spark from anywhere. The Lab Runner shows the tailnet status in its header: **Spark A: ● spark-a** when ssh works.

> ⚠ Tailscale makes *every* port that listens on all interfaces reachable from your tailnet. Servers you start with `docker run -p 8000:8000` are reachable by every device on the tailnet. That is convenient for a lab, but put authentication in front before sharing the tailnet (Module 08 adds LiteLLM keys).

✓ Checkpoint: `tailscale status` lists both your laptop and the Spark, and `ssh -o BatchMode=yes spark-a true` works over the tailnet.

## 4 · Point the Lab Runner at your Spark

The Lab Runner never guesses a host. Open **🖥 Spark setup** in the header and set:

| Setting | Example | Used by |
|---|---|---|
| `SPARK_HOST` | `spark-a` (your SSH alias) | every lab: `sh()` runs commands here |
| `SPARK_HOST2` | `spark-b` | the two-Spark modules (02, 05, 11) |
| `SPARK_API_HOST` | leave empty, or `localhost` when you tunnel ports (Section 6) | HTTP labs (serving, gateways, agents) |

Settings are saved to `week25/.env.local` (gitignored, mode 0600). You can also export them in a shell. Then ask `sparkkit` where commands would run:

```bash
# on: laptop
.venv/bin/python week25/common/sparkkit.py
```

**Expected output** (a laptop with no Spark configured, captured on this Mac)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━ sparkkit self-check
   where would commands run, and which endpoints answer?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◈ DRY · SPARK_HOST not set · commands are shown, not run; outputs are RECORDED, REFERENCE or EXAMPLE (labelled)
◆ on a Spark: False · SPARK_HOST=— · SPARK_HOST2=—
│ ollama    (no host)                                  ○ down
│ vllm      (no host)                                  ○ down
│ sglang    (no host)                                  ○ down
│ trtllm    (no host)                                  ○ down
│ llamacpp  (no host)                                  ○ down
│ litellm   (no host)                                  ○ down
│ laptop    http://localhost:11434/v1                  nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b, …
```

With `SPARK_HOST` set and reachable, the second line becomes `▣ LIVE · Spark A · spark-a` and every lab runs for real.

Three places a lab can run, always printed:

- **`[ssh spark-a]`**: the lab runs on your laptop and sends each command to the Spark over SSH.
- **`[Spark A (this machine)]`**: you started the Lab Runner *on* the Spark (then open it through a tunnel).
- **`[DRY]`**: nothing runs. You see the command and a labelled RECORDED, REFERENCE or EXAMPLE output.

The **💻 laptop stand-in** switch is for HTTP labs only. When a Spark endpoint is down, the ⚡ blocks and serving labs may use Ollama on your laptop instead. The answer is real, but it is labelled `LAPTOP STAND-IN` because the speed is your laptop's, not the Spark's.

✓ Checkpoint: the header shows **Spark A: ●** with your host, or you have decided to follow along in DRY mode.

## 5 · The Spark doctor: lab 01

Before you download 200 GB of models, check the basics. Lab 01 runs eight **read-only** commands on the Spark. Each one prints the exact command first, so you can copy any of them into the ⌨ terminal (set it to 🟩 Spark A).

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/labs/lab01_spark_doctor.py
```

**Expected output** (DRY mode, captured on a laptop; with a Spark you get your own values and ✓ / ✕)

```
▣ STEP 3 · GPU and driver
$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
NVIDIA GB10, 580.95.05
…
│ check                 result       last line of output
│ ────────────────────  ───────────  ──────────────────────────────────────────────
│ CPU architecture      ◈ reference  aarch64
│ Operating system      ◈ example    Ubuntu 24.04.3 LTS
│ GPU and driver        ◈ example    NVIDIA GB10, 580.95.05
│ CUDA toolkit          ◈ example    Cuda compilation tools, release 13.0
│ Unified memory        ◈ example    Mem:             119           6         105
│ Free disk for models  ◈ example    /dev/nvme0n1p2  3.7T  412G  3.1T  12% /
│ Docker without sudo   ◈ example    28.3.3
│ Hugging Face cache    ◈ example    empty (no models downloaded yet)
═ DRY run: REFERENCE rows are what the playbooks state, EXAMPLE rows are illustrative. Connect a Spark (🖥 Spark setup) to check yours.
```

The playbooks state the minimums (driver 580.95.05 or newer, CUDA 13.0) but do not print these exact lines, so the lab shows them as EXAMPLE. Only text quoted word-for-word from a playbook is labelled REFERENCE; `week25/common/audit_references.py` checks that for every lab.

What each check protects you from later:

| Check | Fails later as… |
|---|---|
| `aarch64` | "exec format error" when you pull an x86-only container |
| GB10 + driver ≥ 580.95.05 | containers start but see no GPU |
| CUDA 13 | fine-tuning wheels (`cu130`) fail to import |
| ~119 GiB free | out-of-memory when vLLM reserves its KV cache |
| ≥ 200 GB disk | a model download dies at 97% |
| Docker without sudo | every `docker run` in Modules 03–07 needs `sudo` |

> 💡 128 GB shows as about **119 GiB** in `free -g`: 128 × 10⁹ bytes ÷ 2³⁰ ≈ 119. Nothing is missing.

✓ Checkpoint: in LIVE mode every row is ✓, or every ✕ row has a fix printed under it. In DRY mode you can explain what a REFERENCE row is and how it differs from an EXAMPLE row.

## 6 · DGX Dashboard, and one tunnel for every port

**DGX Dashboard** is the Spark's built-in web app: GPU and memory telemetry, a one-click JupyterLab, and system updates. It listens on `localhost:11000` **on the Spark only**, so reach it through an SSH tunnel (or NVIDIA Sync, which makes the tunnel for you):

```bash
# on: laptop
ssh -N -L 11000:localhost:11000 spark-a
# now open http://localhost:11000 and log in with your Spark username + password
```

> ℹ `spark-a` is the `Host spark-a` alias from §3 in your laptop's `~/.ssh/config`. Without it you get `Could not resolve hostname spark-a`: add the alias, or use the full target instead (e.g. `<you>@spark-abcd.<tailnet>.ts.net` or its `100.x` Tailscale IP). Run this in your laptop's own terminal, not the lab app's ⌨ terminal: when the lab app runs on a Spark, its "laptop" target is that Spark, and the tunnel won't reach your browser.

The [DGX Dashboard playbook](https://build.nvidia.com/spark/dgx-dashboard) then has you start JupyterLab from the dashboard, run a Stable Diffusion XL cell, and watch GPU utilisation climb in the telemetry panel. Each user gets their own JupyterLab port. Find yours with `cat /opt/nvidia/dgx-dashboard-service/jupyterlab_ports.yaml` on the Spark, and add a second `-L` for it.

> ⚠ Install system updates from **Dashboard → Settings → Updates**, not with a bare `apt upgrade`. The dashboard update also updates firmware and reboots the Spark. Do it *before* you start a long fine-tuning run.

This week starts nine services. Lab 03 probes every port and writes one `ssh -L` command for all of them:

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/labs/lab03_ports_and_tunnels.py
```

**Expected output** (captured on this Mac with no Spark configured: only the laptop's own Ollama answers)

```
▣ STEP 1 · probe each port on the Spark and on localhost
│ service    port   on Spark (no host)  on localhost  models  used in
│ ─────────  ─────  ──────────────────  ────────────  ──────  ──────────────
│ ollama     11434  ○                   ● up          —       Module 03
│ vllm       8000   ○                   ○             —       Modules 05, 13
│ sglang     30000  ○                   ○             —       Module 06
│ trtllm     8355   ○                   ○             —       Modules 06-07
│ llamacpp   30080  ○                   ○             —       Module 04
│ lmstudio   1234   ○                   ○             —       Module 04
│ litellm    4000   ○                   ○             —       Module 08
│ openwebui  12000  ○                   ○             —       Module 03
│ dashboard  11000  ○                   ○             —       this module
◆ 'on localhost' is THIS laptop. Ollama on your laptop shows up here too — it is not the Spark.

▣ STEP 2 · one tunnel for all of them
$ ssh -N -L 8000:localhost:8000 -L 30000:localhost:30000 -L 8355:localhost:8355 -L 30080:localhost:30080 -L 1234:localhost:1234 -L 4000:localhost:4000 -L 12000:localhost:12000 -L 11000:localhost:11000 <you>@<spark-hostname>
```

Two ways to reach a Spark service, and when to use each:

| | Direct over the tailnet | SSH tunnel |
|---|---|---|
| URL in labs | `http://spark-a:8000/v1` | `http://localhost:8000/v1` (set `SPARK_API_HOST=localhost`) |
| Works for a service bound to `127.0.0.1` on the Spark | ✕ | ✓ |
| Anyone on the tailnet can call it | yes | no, only your laptop |
| Good for | quick lab work | Dashboard, Ollama, anything without auth |

> 💡 **VS Code** ([playbook](https://build.nvidia.com/spark/vscode)): install the *Remote - SSH* extension and connect to `spark-a`. Your editor then runs on the Spark's files with the Spark's Python. NVIDIA Sync can launch this in one click.

✓ Checkpoint: DGX Dashboard opens at `http://localhost:11000` through your tunnel, and you have the one-line tunnel command from lab 03 saved.

## 7 · Will it fit? Memory arithmetic for the whole week

Two formulas decide which model you can run, on how many Sparks, and roughly how fast:

```text
memory needed ≈ weights + KV cache + overhead
  weights   = parameters × bits per weight ÷ 8
  KV cache  = 2 (K and V) × layers × KV heads × head dim × context tokens × users × 2 bytes

single-stream speed ≤ memory bandwidth ÷ bytes of ACTIVE weights read per token
```

Lab 02 applies them to six open models. It is pure arithmetic, so the output is the same on every machine:

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/labs/lab02_memory_budget.py
```

**Expected output**

```
▣ STEP 1 · weights + KV cache (32K context, batch 1) + 10 GB overhead
│ model                  bf16                  fp8                   nvfp4
│ ─────────────────────  ────────────────────  ────────────────────  ───────────────────
│ Llama 3.1 8B               30 GB  1 Spark        22 GB  1 Spark        19 GB  1 Spark
│ Qwen3 32B                  84 GB  1 Spark        51 GB  1 Spark        37 GB  1 Spark
│ Llama 3.3 70B             162 GB  2 Sparks       91 GB  1 Spark        60 GB  1 Spark
│ gpt-oss-120b (MoE)        246 GB  2 Sparks      129 GB  2 Sparks       78 GB  1 Spark
│ Qwen3 235B-A22B (MoE)     486 GB  ✕ too big     251 GB  2 Sparks      148 GB  2 Sparks
│ Llama 3.1 405B            837 GB  ✕ too big     432 GB  ✕ too big     255 GB  2 Sparks

▣ STEP 2 · why the KV cache matters — the same 70B model at longer contexts
│ Llama 3.3 70B ·   8K context  KV cache   2.7 GB  ██░░░░░░░░░░░░░░░░░░░░░░░░░░
│ Llama 3.3 70B ·  32K context  KV cache  10.7 GB  ██████░░░░░░░░░░░░░░░░░░░░░░
│ Llama 3.3 70B · 128K context  KV cache  42.9 GB  ████████████████████████░░░░

▣ STEP 3 · decode speed ceiling — 273 GB/s of memory bandwidth, one stream
│ model                  read per token  bf16 ceiling  nvfp4 ceiling
│ ─────────────────────  ──────────────  ────────────  ─────────────
│ Llama 3.1 8B           8B active         17.1 tok/s    60.7 tok/s
│ Qwen3 32B              32.8B active       4.2 tok/s    14.8 tok/s
│ Llama 3.3 70B          70.6B active       1.9 tok/s     6.9 tok/s
│ gpt-oss-120b (MoE)     5.1B active       26.8 tok/s    95.2 tok/s
│ Qwen3 235B-A22B (MoE)  22B active         6.2 tok/s    22.1 tok/s
│ Llama 3.1 405B         405B active        0.3 tok/s     1.2 tok/s
═ MoE models (gpt-oss-120b, Qwen3 235B-A22B) are the Spark's sweet spot: big total size for quality, few active weights for speed. Llama 405B needs two Sparks and 4-bit weights — Module 02.
```

Three lessons that come back all week:

1. **Precision is the biggest lever.** Going from bf16 to NVFP4 cuts weights by ~3.5×. Llama 3.3 70B moves from "two Sparks" to "one Spark with room to spare" (Module 07 does this for real).
2. **Mixture-of-Experts suits the Spark.** gpt-oss-120b stores 117B parameters but reads only ~5B per token. The 128 GB pays for quality and the 273 GB/s stays fast.
3. **The ceiling is per stream.** Serving engines (vLLM, SGLang in Modules 05–06) batch many users together, so *total* throughput can be many times one stream's speed.

These are upper bounds from arithmetic, not benchmarks. Module 03 onward measures real tok/s on your Spark and compares it with this ceiling.

✓ Checkpoint: you can explain why Llama 3.3 70B needs two Sparks at bf16 but only one at NVFP4, and why gpt-oss-120b can be faster than Qwen3 32B despite being bigger.

## Labs — run them here

**labs/lab01_spark_doctor.py** — Eight read-only readiness checks on your Spark, each printed as a real command, summarised as a pass/fail table.

**labs/lab02_memory_budget.py** — Weights + KV cache for six models at three precisions, placed on one or two Sparks, plus the bandwidth speed ceiling.

**labs/lab03_ports_and_tunnels.py** — Probe every port this week uses, on the Spark and on localhost, and print one `ssh -L` command for all of them.

Lab 01 runs LIVE on your Spark or DRY. Labs 02 and 03 run on the laptop. Lab 03 probes the Spark when a host is set.

## Try it yourself

**Exercise 01 — the "will it fit?" calculator.** Open `week25/01_meet_your_spark/exercises/ex01_will_it_fit.py`. It has three `TODO`s:

1. `weights_gb(params_b, bits)`: the size of the weights in GB.
2. `kv_cache_gb(layers, kv_heads, head_dim, ctx, batch, bytes_per)`: the KV cache in GB.
3. `placement(need_gb)`: return `"1 Spark"`, `"2 Sparks"`, or `"too big"`.

The checker is offline and free. It compares your functions with known answers, then places three models.

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/exercises/ex01_will_it_fit.py
```

**Expected output** (once all three TODOs are done)

```
✓ weights_gb: 8B bf16 = 16 GB · 70B nvfp4 ≈ 39.4 GB
✓ kv_cache_gb: Llama 8B @32K ≈ 4.3 GB · Llama 70B @8K × 4 users ≈ 10.7 GB
✓ placement: 100→1 · 128→1 · 129→2 · 256→2 · 300→too big

▣ your calculator, applied (32K context, one user, 10 GB overhead)
│ Qwen3 32B · fp8          needs   51.4 GB → 1 Spark
│ Llama 3.3 70B · bf16     needs  161.9 GB → 2 Sparks
│ Llama 3.1 405B · nvfp4   needs  254.7 GB → 2 Sparks
```

<details><summary>Hint — where does the "× 2" in the KV cache come from?</summary>

Every attention layer stores two tensors per token: the **K**eys and the **V**alues. So the cache is `2 × layers × kv_heads × head_dim` numbers per token. Multiply by tokens, users, and bytes per number.

</details>

<details><summary>Stretch — 20 users at 32K context</summary>

Change the applied examples to `batch=20`. How much KV cache does Llama 3.3 70B need now, and does it still fit on one Spark at NVFP4? This is the calculation behind vLLM's `--max-num-seqs`.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ssh: Could not resolve hostname spark-xxxx.local` | mDNS is blocked on this network. Use the Spark's IP, or Tailscale (Section 3) |
| `Permission denied (publickey)` in the Lab Runner but a password login works | BatchMode never asks for a password. Run `ssh-copy-id` (Section 2) and check the `IdentityFile` in `~/.ssh/config` |
| Header says **Spark A: ○ spark-a** | The Spark is off, asleep, or off the tailnet. Try `tailscale ping spark-abcd`, then press ↻ |
| `docker: permission denied … docker.sock` | `sudo usermod -aG docker $USER`, then log out and back in |
| `free -g` shows far less than 119 GB free | Another process holds memory (a notebook kernel, a stopped-but-cached container). If memory stays low after the job has ended, the playbooks flush the page cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| DGX Dashboard tunnel opens but the page is blank | The dashboard binds to localhost on the Spark: tunnel `-L 11000:localhost:11000`, not `-L 11000:spark-a:11000` |

## Next

Continue to [Lab 02 — two Sparks, one cluster](../02_two_sparks_nccl/TUTORIAL.md): cable two Sparks with QSFP, configure the 200 Gb/s link, and measure it with NCCL.
