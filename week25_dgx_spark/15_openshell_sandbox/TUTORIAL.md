# ▶ Spark Lab 15 — OpenShell: sandbox and govern AI agents

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn OpenShell's parts: a gateway on the Spark, sandboxes, providers, the `inference.local` route, and a YAML policy.
- Install OpenShell on the Spark exactly as the playbook does, and the CLI on your laptop from a week25-local venv.
- Route a sandbox's model calls to vLLM on your Spark, with no API key leaving the machine.
- Write a policy for the Module 14 NAT hotel agent, and check it two ways: the real OpenShell parser and a course-made validator.
- Ask "what would this policy allow?" for a list of agent actions, and preview what a policy change opens up.
- Put the NAT agent in a sandbox on the Spark, watch its decisions, and prove it cannot reach the internet.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** 1 Spark (or none: labs 15-2 and 15-3 and the exercise are fully offline)

**Official playbooks covered:** [OpenShell](https://build.nvidia.com/spark/openshell) (`nvidia/playbook-openshell`, and the older `nvidia/openshell`). The policy layout is the one NVIDIA ships in the healthcare-agent playbook (`assets/sandbox-policy.yaml`); the policy rules come from the troubleshooting tables of the OpenShell and NemoClaw-applications playbooks.

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 14 done | `week25/.venv-nat/bin/nat --version` | the agent you will sandbox |
| Module 05 done on the Spark | vLLM answers on `:8000`, started with `--host 0.0.0.0` | the model the sandbox routes to |
| Docker without sudo on the Spark, with the NVIDIA runtime | `docker ps` on the Spark | OpenShell runs a k3s cluster inside Docker |
| Python ≥ 3.12 on the Spark | `python3 --version` | a playbook prerequisite |
| This repo cloned on the Spark | `ls ~/agenticaicodingfitness/week25` | lab 15-4 uploads the NAT module into the sandbox from there |

> ⚠ The playbook's first instruction: **use a clean environment**. Run agents on a Spark (or VM) without personal data, real accounts or production credentials. OpenShell lowers the risk of running an agent; it does not remove it.

✓ Checkpoint: vLLM on the Spark answers `curl -s http://localhost:8000/v1/models`, and `docker ps` works without sudo.

## 1 · What OpenShell is, in one picture

An agent that can read files, run programs and call APIs is useful and dangerous for the same reason. **OpenShell** is NVIDIA's open-source sandbox runtime for agents: it runs the agent in an isolated sandbox and enforces a declarative YAML policy on what it can read, write, run and reach.

| Part | What it is | Command |
|---|---|---|
| **Gateway** | the control plane; on a Spark a systemd user service running k3s inside Docker | `openshell status` |
| **Sandbox** | one isolated agent environment, built from an image (`--from base`, `--from openclaw`, a Dockerfile) | `openshell sandbox create / exec / connect / delete` |
| **Policy** | YAML: `filesystem_policy`, `landlock`, `process`, `network_policies` | `openshell policy get / set / update` |
| **Provider** | a named credential + config the sandbox can use without seeing the secret | `openshell provider create` |
| **inference.local** | a hostname inside every sandbox that the proxy routes to your model | `openshell inference set` |

```text
  DGX Spark
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │  openshell-gateway (systemd user service · k3s in Docker)                      │
 │    ┌─ sandbox "hotel-agent" ───────────────────────────────┐                   │
 │    │  NAT agent (python) ── https://inference.local/v1 ──┐  │   provider        │
 │    │   Landlock: write only /sandbox /tmp               │  │   local-vllm ──►  vLLM :8000
 │    │   runs as user "sandbox"                    egress proxy ──────────────►  (host IP, not localhost)
 │    │   curl https://api.openai.com  ──────────►  ✕ deny (not in policy)         │
 │    └────────────────────────────────────────────────────────┘                   │
 └───────────────────────────────────────────────────────────────────────────────┘
```

Three kinds of control, three mechanisms. **Filesystem** rules are enforced by the Linux kernel (Landlock) and fixed when the sandbox is created. **Process** rules pick an unprivileged user. **Network** rules are enforced by an egress proxy: everything is denied unless a policy group names the host, the port *and* the program allowed to use it. Network rules hot-reload; filesystem rules need a new sandbox.

✓ Checkpoint: you can say which of the three kinds of rule you can change on a running sandbox.

## 2 · Install OpenShell on the Spark (playbook Steps 1–4)

Check the environment (Step 1) and Docker (Step 2). If `docker ps` says permission denied, the playbook adds you to the group with `sudo usermod -aG docker $USER` (then log out and in). Configuring the NVIDIA runtime also needs `sudo`, so type these yourself:

```bash
# on: spark
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Install the CLI with the official installer (Step 3). It installs `openshell` and registers the `openshell-gateway` systemd user service. Then check the gateway (Step 4):

```bash
# on: spark
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
source ~/.bashrc
openshell --help
systemctl --user status --no-pager openshell-gateway
openshell status
sudo loginctl enable-linger $USER
```

`openshell status` should report the gateway as **Connected**. The first start can take a few minutes while Docker pulls images and k3s boots; follow it with `journalctl --user -u openshell-gateway -f`.

> ⚠ **The playbook's "alternative install" (`uv pip install openshell`) no longer gives you the CLI.** On PyPI, `openshell` 0.1.x (0.1.2 on 2026-09-28) is the Python SDK only: one `py3-none-any` wheel with no `openshell` command. The last releases that ship the CLI binary are the 0.0.x wheels (0.0.111 has `macosx_13_0_arm64` and `manylinux_2_39_aarch64` builds). On the Spark, use the official installer as above.

**On your laptop** the course installs that 0.0.111 CLI into a module-local venv — no installer script, no sudo — to read its help, check your policy files with its parser (Section 4), and optionally manage the Spark's gateway remotely (`openshell gateway add https://openshell:8080 --remote <user>@<spark-ip>`, playbook Step 9). A 0.0.x CLI talking to a 0.1.x gateway is not guaranteed to work; for remote management, prefer matching versions.

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
uv venv -p 3.12 week25/15_openshell_sandbox/.venv-openshell
uv pip install --python week25/15_openshell_sandbox/.venv-openshell/bin/python "openshell==0.0.111"
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_1_install_and_gateway.py
```

**Expected output** (the laptop half, captured on this Mac; the Spark half is DRY here)

```
▣ STEP B1 · the OpenShell CLI on this laptop (openshell==0.0.111 in week25/15_openshell_sandbox/.venv-openshell)
$ openshell --version   [this laptop]
openshell 0.0.111
…
✓ 16 top-level commands, including sandbox, logs, policy, provider, gateway, inference, term

▣ STEP B2 · status and prerequisites on the laptop (no gateway here — the output says so)
$ openshell status   [this laptop]
Gateway Status
  Status: No gateway configured.
Register a gateway with: openshell gateway add <endpoint>
$ openshell doctor check   [this laptop]
Checking system prerequisites...
  Docker ............. FAILED
Error:   × docker info failed: Cannot connect to the Docker daemon at unix:///var/
  │ run/docker.sock. Is the docker daemon running?
```

"No gateway configured" is the same message as the playbook's troubleshooting row for a gateway that never started. On the Spark you want **Connected**.

✓ Checkpoint: on the Spark, `openshell status` says Connected. On the laptop, `openshell --version` prints 0.0.111 (optional).

## 3 · Route the sandbox's model calls to your Spark (playbook Steps 5–7)

Inside a sandbox, the agent calls `https://inference.local/v1`. OpenShell's proxy forwards that to a **provider** you define, so the agent never holds a real key and never needs an internet route to a model.

Start vLLM with the playbook's agent-ready recipe from Module 05, keeping `--host 0.0.0.0` and port 8000. The gateway runs inside Docker, so it reaches vLLM through the Spark's IP address, not `localhost`:

```bash
# on: spark
curl -sf http://localhost:8000/v1/models
export HARDWARE_IP="$(hostname -I | awk '{print $1}')"
test -n "$HARDWARE_IP"
curl -sf "http://${HARDWARE_IP}:8000/v1/models"
```

Create the provider (vLLM needs no key, so any non-empty placeholder works) and point `inference.local` at the model:

```bash
# on: spark
openshell provider create \
    --name local-vllm \
    --type openai \
    --credential OPENAI_API_KEY=not-needed \
    --config OPENAI_BASE_URL="http://${HARDWARE_IP}:8000/v1"
openshell provider list
openshell inference set \
    --provider local-vllm \
    --model nvidia/Qwen3.6-35B-A3B-NVFP4
openshell inference get
```

`openshell inference get` should show `provider: local-vllm` and your model. If `inference set` says `failed to verify inference endpoint`, warm vLLM up with one chat request first; `--no-verify` skips the check once you know the host API works.

✓ Checkpoint: `openshell inference get` names `local-vllm` and `nvidia/Qwen3.6-35B-A3B-NVFP4`.

## 4 · The policy file, and two ways to check it

The course's policy for the NAT hotel agent is `policies/hotel_agent_policy.yaml`. It copies the layout of the policy NVIDIA ships with the healthcare-agent playbook:

```yaml
version: 1
filesystem_policy:
  include_workdir: true
  read_only:  [/usr, /lib, /proc, /dev/urandom, /etc]      # the OS: readable, never writable
  read_write: [/sandbox, /tmp, /dev/null]                  # the agent's own files
landlock:
  compatibility: best_effort                               # the shipped policy's choice (see its comment)
process:
  run_as_user: sandbox
  run_as_group: sandbox
network_policies:                                          # a MAP of groups, never a list
  inference:
    name: inference
    endpoints:
      - host: inference.local
        port: 443
    binaries:                                              # which programs may use this egress
      - { path: "/sandbox/.venv-nat/bin/python*" }
      - { path: "/usr/bin/python3*" }
      - { path: /usr/bin/curl }
  pypi:                                                    # setup-only: removed after NAT is installed
    name: pypi
    endpoints:
      - { host: pypi.org, port: 443, access: full, tls: skip }
      - { host: files.pythonhosted.org, port: 443, access: full, tls: skip }
    binaries:
      - { path: "/sandbox/.venv-nat/bin/python*" }
      - { path: "/sandbox/.venv-nat/bin/pip*" }
      - { path: "/usr/bin/python3*" }
```

The rules a policy must follow are spread over three playbooks. Collected in one place:

| Rule | Source | Checked by |
|---|---|---|
| Top-level keys are `version`, `filesystem_policy`, `landlock`, `process`, `network_policies` (lower-case `version`) | OpenShell + NemoClaw-applications troubleshooting (`unknown field 'Version'`) | OpenShell CLI parser |
| `network_policies` is a map keyed by group name | NemoClaw-applications (`invalid type: sequence, expected a map`) | OpenShell CLI parser |
| Paths start with `/` and contain no `..` | OpenShell troubleshooting ("Policy push returns exit code 1") | the gateway, on push |
| `run_as_user` is not `root` | same row | the gateway, on push |
| Every endpoint has `host` and `port` | same row | the gateway, on push |
| Each group needs `binaries`, and each endpoint an access mode, or calls fail with `CONNECT tunnel failed, response 403` | NemoClaw-applications | nobody — it applies cleanly and then denies |

Lab 15-2 generates the policy from a short Python spec, then checks it and eight broken variants with **policykit** (`week25/15_openshell_sandbox/policykit.py`, a course-made validator for the rules above) and with the **real OpenShell parser**. The trick: `openshell policy set` parses the file on your laptop before it contacts a gateway, so pointing it at a closed port (`OPENSHELL_GATEWAY_ENDPOINT=http://127.0.0.1:9`) tells "parse error" apart from "parsed, then could not connect" — no gateway needed.

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_2_policy_builder.py
```

**Expected output** (captured on this Mac, trimmed)

```
✓ generated policy == policies/hotel_agent_policy.yaml (same keys, same values)

▣ STEP 2 · eight broken variants — each one is a row in the playbooks' troubleshooting tables
│ policy                            policykit.validate  openshell CLI parser
│ ────────────────────────────────  ──────────────────  ────────────────────────────────────────────────────
│ policies/hotel_agent_policy.yaml  ✓ ok                ✓ parsed
│ healthcare playbook (shipped)     ⚠ 3 warning(s)      ✓ parsed
│ Version (capital V)               ✕ 2 error(s)        ✕ unknown field `Version`, expected one of `version…
│ network_policies as a list        ✕ 1 error(s)        ✕ network_policies: invalid type: sequence, expecte…
│ relative / .. path                ✕ 2 error(s)        ✓ parsed
│ run_as_user: root                 ✕ 1 error(s)        ✓ parsed
│ endpoint without port             ✕ 1 error(s)        ✓ parsed
│ group without binaries            ⚠ 1 warning(s)      ✓ parsed
│ endpoint without access mode      ⚠ 1 warning(s)      ✓ parsed
│ endpoint with description:        ✕ 1 error(s)        ✕ network_policies.pypi.endpoints.\[0\]: unknown fi…
```

Read the columns together. The parser catches **structure** (unknown fields, list vs map, wrong types) on your laptop; **semantic** rules come back from the gateway only when you push, which is why the course checks them first. Three honest notes from building this table:

- **The two NVIDIA policies disagree.** The shipped healthcare policy has bare `{host, port}` entries for npm and PyPI; the NemoClaw-applications playbook says a bare entry is "the single most common reason" for 403s. policykit warns, and the course's policy follows the stricter rule (`access: full, tls: skip`).
- **A runbook example does not parse with CLI 0.0.111.** The healthcare RUNBOOK's "Modifying the Policy" example puts `description:` on an endpoint; the 0.0.111 parser rejects it (`unknown field`). The newer gateway may differ — check with `openshell policy set --wait` on your Spark.
- **Private IPs.** The shipped policy's comment says raw private/loopback IPs are blocked "regardless of policy entries", yet the same file allows a Docker-bridge IP for one service, and the CLI has `allowed_ips`. Route model calls through `inference.local` and the question never arises.

✓ Checkpoint: your policy is `✓ ok` in both columns, and you can say why "relative / .. path" parses but would still be rejected.

## 5 · What would this policy allow? Watching and denying

Before you push a policy, ask what it lets the agent do. Lab 15-3 replays thirteen actions — the hotel agent doing its job, and a prompt-injected agent trying to leak — through **`policykit.decide`**. This is a **course-made teaching model** of the semantics the playbooks describe (Landlock paths, default-deny egress, per-group binaries, inference routing, L7 method rules), **not OpenShell itself**; its reasons say "(course assumption)" wherever the playbooks are silent.

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_3_what_would_it_allow.py
```

**Expected output** (captured on this Mac; policykit model, not OpenShell)

```
│ who     action                                                decision     why (the rule that decided)
│ ──────  ────────────────────────────────────────────────────  ───────────  ────────────────────────────────────────────────────
│ agent   write /sandbox/.runs/tickets.jsonl                    ✓ allow      under read_write /sandbox
│ agent   python3.12 → inference.local:443                      → inference  inference.local is handled by the proxy's inference…
│ setup   pip → pypi.org:443                                    ✓ allow      network_policies.pypi: pypi.org:443 for binary /san…
│ hijack  python3.12 → api.openai.com:443                       ✕ deny       api.openai.com:443 is not in network_policies (defa…
│ hijack  curl → pypi.org:443                                   ✕ deny       pypi.org:443 is listed, but not for binary /usr/bin…
│ hijack  read  /home/sandbox/.ssh/id_ed25519                   ✕ deny       not in read_only or read_write (Landlock: Permissio…
│ hijack  write /etc/cron.d/backdoor                            ✕ deny       /etc is read_only (Landlock: Permission denied)
│ hijack  run as root                                           ✕ deny       the sandbox runs as 'sandbox'; it cannot become 'ro…
◆ 7 of 13 actions denied · hijack attempts that got through: none · legitimate actions blocked: none

▣ STEP 2 · preview a change: add GitHub read-only for curl (like `openshell policy update --add-endpoint … --dry-run`)
│ action                                                before  after
│ ────────────────────────────────────────────────────  ──────  ───────  ─────────
│ curl → api.github.com:443 GET /repos/NVIDIA/OpenShe…  ✕ deny  ✓ allow  ◆ CHANGED
│ curl → api.github.com:443 POST /repos/NVIDIA/OpenSh…  ✕ deny  ✕ deny
│ python3.12 → api.github.com:443 GET /                 ✕ deny  ✕ deny
```

Note `curl → pypi.org`: the host is allowed, but only for pip and Python. **The binaries list is what stops a hijacked agent from using an allowed host with a different tool.**

On the Spark, the real decisions are visible live. The playbook's Step 11 uses the terminal UI; its troubleshooting uses the log stream:

```bash
# on: spark
openshell term
openshell logs hotel-agent --tail --source sandbox
```

`openshell term` shows each sandbox's status and a live log of outbound connections with the decision: `allow`, `deny` or `inspect_for_inference` (press `f` to follow, `s` to filter by source, `q` to quit). To change what is allowed:

| Change | Command | Takes effect |
|---|---|---|
| add one endpoint | `openshell policy update hotel-agent --add-endpoint api.github.com:443:read-only:rest:enforce --binary /usr/bin/curl --wait` | hot-reload |
| preview first | the same with `--dry-run` | nothing sent |
| remove a group | `openshell policy update hotel-agent --remove-rule pypi --wait` | hot-reload |
| replace the whole policy | `openshell policy get hotel-agent > p.yaml` (not `--full`), edit, `openshell policy set hotel-agent --policy p.yaml --wait` | hot-reload |
| change filesystem rules | delete and recreate the sandbox | new sandbox only |

(The `policy update` flags come from the 0.0.111 CLI help; `policy get` without `--full` avoids the `unknown field 'Version'` error in the playbook's troubleshooting table.)

✓ Checkpoint: you can explain why `curl → pypi.org` is denied even though `pypi.org:443` is in the policy.

## 6 · Put the NAT hotel agent in a sandbox

Now combine Modules 14 and 15. Lab 15-4 prints — and with `--yes` runs — the whole sequence on the Spark. It is **course-designed**: the provider and inference steps are the playbook's; the sandbox for a NAT agent (instead of the playbook's OpenClaw image) is ours, and **it has not been run on a Spark yet**.

```bash
# on: spark
cd ~/agenticaicodingfitness/week25
mkdir -p ~/w25 && cp 15_openshell_sandbox/policies/hotel_agent_policy.yaml ~/w25/
openshell sandbox create --name hotel-agent --from base --policy ~/w25/hotel_agent_policy.yaml \
  --upload ~/agenticaicodingfitness/week25/14_nat_agents:/sandbox/14_nat_agents --keep --no-tty -- true
openshell sandbox exec -n hotel-agent -- python3 -m venv /sandbox/.venv-nat
openshell sandbox exec -n hotel-agent -- /sandbox/.venv-nat/bin/pip install -q 'nvidia-nat[langchain]~=1.9' greenlet
openshell sandbox exec -n hotel-agent -- /sandbox/.venv-nat/bin/pip install -q --no-deps -e /sandbox/14_nat_agents/hotel_ops_nat
openshell policy update hotel-agent --remove-rule pypi --wait
openshell sandbox exec -n hotel-agent --workdir /sandbox/14_nat_agents \
  --env HOTEL_LLM_BASE_URL=https://inference.local/v1 --env HOTEL_LLM_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --env HOTEL_TICKET_LOG=/sandbox/tickets.jsonl \
  -- /sandbox/.venv-nat/bin/nat run --config_file configs/hotel_agent_spark.yml --input "Room 808 feels hot. Check it and open a ticket if something is wrong."
openshell sandbox exec -n hotel-agent -- curl -s --max-time 5 https://api.openai.com || echo 'blocked ✓'
```

From the laptop, the same with the lab (DRY prints each command with an EXAMPLE output; with a Spark connected it runs read-only checks, and changes only with `--yes`):

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_4_sandbox_the_nat_agent.py
```

**Expected output** (EXAMPLE — illustrative shape, not a measurement)

```
▣ STEP 6 · run the agent inside the sandbox — its only way out is inference.local
$ openshell sandbox exec -n hotel-agent --workdir /sandbox/14_nat_agents --env HOTEL_LLM_BASE_URL=https://inference.local/v1 … nat run --config_file configs/hotel_agent_spark.yml --input "Room 808 feels hot. …"   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
Workflow Result:
Room 808 is too warm … Created maintenance ticket MT-A83AD2 (high priority).

▣ STEP 7 · prove the fence: the same sandbox cannot reach the internet (runbook pattern: these should fail)
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
curl: (56) CONNECT tunnel failed, response 403
blocked ✓
```

**The first time you run this on a Spark, check these six things** — each is an assumption the course could not test without one:

1. `--from base` has `python3` with `venv` and `pip`. If not, build a small image (`--from <dir with a Dockerfile>`) that has them.
2. `--upload <path>:<path>` copies the folder where the lab expects it (`/sandbox/14_nat_agents`).
3. Python reaches `inference.local` through the sandbox's proxy settings, and trusts its certificate. The playbook's own test is `curl https://inference.local/v1/chat/completions` inside the sandbox (Step 10); if TLS fails for Python, set `--env HOTEL_LLM_VERIFY_SSL=false` for that one call and report it.
4. `--remove-rule pypi` removes the group by its key name.
5. `openshell logs hotel-agent --source sandbox` shows `deny` for api.openai.com and `inspect_for_inference` for the model calls.
6. `--keep --no-tty -- true` leaves the sandbox running after `true` exits (the playbook uses `--keep` with an interactive command; the 0.0.111 help lists only `--no-keep`, and accepts `--keep`).

✓ Checkpoint: the agent's answer quotes a ticket id, and the `curl` to api.openai.com fails from inside the same sandbox.

## 7 · The playbook's own sandbox: OpenClaw (Steps 8–13)

The playbook itself sandboxes **OpenClaw**, a local-first agent, from a community image with its policy bundled. Modules 16 and 17 build on it, so here is the core of it. Do not pass `--policy` with `--from openclaw`; the policy comes with the image:

```bash
# on: spark
export SANDBOX_NAME=openshell-demo
openshell sandbox create \
  --keep \
  --tty \
  --forward 18789 \
  --name "$SANDBOX_NAME" \
  --from openclaw \
  -- openclaw-start
```

The onboarding wizard is interactive (arrow keys): choose **Custom Provider**, API base URL `https://inference.local/v1`, any placeholder key, **OpenAI-compatible**, and the same model handle as `openshell inference set`. After a minute or two it prints the dashboard URL:

**Expected output** (REFERENCE — quoted from the playbook)

```
OpenClaw gateway starting in background.
  Logs: /tmp/gateway.log
  UI:   http://127.0.0.1:18789/?token=<unique-token>
```

From a laptop, forward the port with `openshell forward start --background 18789 "$SANDBOX_NAME"` and open `http://127.0.0.1:18789/#token=<your-token>`. When you are done, clean up while the gateway is still running (Step 13):

```bash
# on: spark
openshell sandbox delete "$SANDBOX_NAME"
openshell sandbox delete hotel-agent
openshell provider delete local-vllm
```

✓ Checkpoint: `openshell sandbox list` shows no sandboxes you do not intend to keep.

## Labs — run them here

**labs/lab15_1_install_and_gateway.py** — The playbook's read-only environment and gateway checks on the Spark, plus the real OpenShell CLI on your laptop.

**labs/lab15_2_policy_builder.py** — Generate the hotel-agent policy, then validate it and eight broken variants with policykit and the real OpenShell parser.

**labs/lab15_3_what_would_it_allow.py** — Replay the agent's and a hijacked agent's actions through a teaching model of the policy, and preview a policy change.

**labs/lab15_4_sandbox_the_nat_agent.py** — The full Spark sequence to run the Module 14 NAT agent in an OpenShell sandbox (changes only with `--yes`).

Labs 15-2 and 15-3 are offline and identical on every machine. Lab 15-1's Spark half and lab 15-4 are DRY until a Spark is connected.

## Try it yourself

**Exercise 15 — lock down the agent.** After setup, the NAT agent needs no package index: only its files and its model. Open `week25/15_openshell_sandbox/exercises/ex15_lock_down_the_agent.py` and write the **runtime** policy in three `TODO`s:

1. `filesystem()`: the OS readable, only `/sandbox`, `/tmp` and `/dev/null` writable.
2. `network_policies()`: exactly one group, `inference`, for `inference.local:443` and the NAT venv's Python.
3. `process()`: run as `sandbox:sandbox`.

The checker validates your policy with policykit, then replays twelve actions and compares each decision with a locked-down agent's.

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/exercises/ex15_lock_down_the_agent.py
```

**Expected output** (once all three TODOs are done, captured on this Mac, trimmed)

```
✓ filesystem: /sandbox writable · /etc read-only
✓ network_policies: exactly one group, 'inference'
✓ process: runs as sandbox:sandbox
✓ policykit.validate: 0 errors, 0 warnings

▣ twelve actions, replayed through policykit.decide
│ action                                                should be              your policy              why
│ write /sandbox/tickets.jsonl                          allow                  ✓ allow                  under read_write /sandbox
│ python3.12 → inference.local:443                      inspect_for_inference  ✓ inspect_for_inference  inference.local is handled by the proxy's infere
│ pip → pypi.org:443                                    deny                   ✓ deny                   pypi.org:443 is not in network_policies (default
│ read  /home/sandbox/.ssh/id_ed25519                   deny                   ✓ deny                   not in read_only or read_write (Landlock: Permis
│ run as root                                           deny                   ✓ deny                   the sandbox runs as 'sandbox'; it cannot become
…
✓ every decision matches a locked-down agent
```

<details><summary>Hint — why is /home/sandbox/.ssh denied when nothing says "deny"?</summary>

Filesystem policy is an allow-list. Landlock denies every path that is not under a `read_only` or `read_write` entry, so you never list what to block — only what to allow. `/home` is in neither list.

</details>

<details><summary>Stretch — let the agent read a building API</summary>

Add a group `bms` for `bms.example.hotel:443` with `protocol: rest`, `enforcement: enforce` and a `rules` list that allows only `GET /rooms/**`, for the NAT venv's Python. Add two actions to the checker: `GET /rooms/808` should be allowed, `POST /rooms/808/setpoint` denied. This is the shape of the `openfold3` group in NVIDIA's healthcare policy.

</details>

✓ Checkpoint: all checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `openshell status` shows "Connection refused" or "No gateway configured" | `systemctl --user start openshell-gateway`, then `journalctl --user -u openshell-gateway --no-pager -n 50`. If the service cannot reach Docker: `sudo setfacl -m u:$USER:rw /var/run/docker.sock` and restart the service |
| `uv pip install openshell` gave no `openshell` command | PyPI 0.1.x is the SDK only. Use the official installer on the Spark; on the laptop pin `openshell==0.0.111` (Section 2) |
| `failed to verify inference endpoint` on `inference set` | vLLM still loading, or the provider URL uses `localhost`. Use the host IP from `hostname -I`, warm up with one chat request, then retry |
| Everything outbound is denied, even hosts in the policy | The group has no `binaries`, or the endpoint has no access mode (NemoClaw-applications: `CONNECT tunnel failed, response 403`). Watch with `openshell logs <name> --tail --source sandbox` |
| `policy set` fails with `unknown field 'Version'` | You exported with `--full`. Use `openshell policy get <name>` without `--full`, or lowercase the key |
| `failed to parse sandbox policy YAML … invalid type: sequence, expected a map` | `network_policies` must be a map of named groups, not a list of endpoints |
| Policy push exits 1, "validation failed" | A path without a leading `/`, `..` in a path, `run_as_user: root`, or an endpoint without host/port. Run lab 15-2's checks on your file |
| "Permission denied" / Landlock errors inside the sandbox | The path is not in `read_only`/`read_write`. Filesystem policy is static: fix the YAML and recreate the sandbox |
| Sandbox stuck in `Error` phase | `openshell logs <name>`: invalid policy YAML, missing provider credentials, or a port conflict |

## Next

Continue to [Lab 16 — NemoClaw: always-on sandboxed agents](../16_nemoclaw/TUTORIAL.md): NVIDIA's reference stack that runs OpenClaw inside OpenShell around the clock, with policy presets you add and remove without a rebuild.
