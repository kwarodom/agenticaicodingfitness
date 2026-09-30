# ▶ Reef Lab 07 — Expert: threat model, hardening, custom blueprints, remote gateways

> Part of Week 26 · NemoClaw on DGX Spark, beginner to expert. You type the commands, you see the real output. Laptop labs (the NAT and OpenShell CLIs, the policy model) run for real everywhere. Spark labs also run in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's docs or playbooks, or a clearly marked EXAMPLE.

**What you'll actually do**
- Attack your own claw on paper: twelve steps a prompt-injected agent would try, decided against a first-draft policy and against the production policy.
- Walk the Alto sovereign hardening checklist line by line, and learn which lines a policy lint can check and which it never can.
- Run the research tutorial's production policy through three checkers (the real OpenShell CLI parser, `policykit.validate`, `policykit.harden`), then break it twelve ways and see which checker catches what.
- Check the policy against the **real** mock BMS tool list, and apply it on the Spark with a preview first.
- Generate a custom sandbox image, validate an external-gateway `blueprint.yaml` (plus eleven broken copies), and see where the tutorial's remote-gateway commands need fixing.
- Build the evidence pack for "no data left the building last month", and fix a broken approvals rule in the exercise.

**Time** ~90 min · **Difficulty** expert · **Hardware** none (DRY + laptop) · 1 DGX Spark optional · a second Spark optional (remote gateway)

**Sources:** the course's research tutorial *NemoClaw on DGX Spark — Beginner to Expert* (Part 6: §6.1–6.6, Labs 6.1–6.3, Part 6 exercises), which cites [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema) · [NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [OpenShell on GitHub](https://github.com/NVIDIA/openshell) · [CSA — OpenClaw indirect prompt injection](https://labs.cloudsecurityalliance.org/research/csa-research-note-openclaw-indirect-prompt-injection-2026061/)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Modules 03 and 04 | you can read a policy and you know the Alto Ops Claw | this module hardens both |
| The laptop OpenShell CLI | `week26/.venv-openshell/bin/openshell --version` → 0.0.111 | the real policy parser (no gateway behind it) |
| The NAT venv | `week26/.venv-nat/bin/python -c "import mcp"` | runs the mock BMS in lab 07-2 |
| `openssl` | `openssl version` | lab 07-3 makes a throwaway CA for the blueprint |
| A DGX Spark (optional) | `ssh -o BatchMode=yes <spark> true` | apply the policy, run the blueprint runner, collect real evidence |

```bash
# on: laptop
week26/.venv-openshell/bin/openshell --version
openssl version
week26/.venv-nat/bin/python -c "import mcp; print('mcp ok')"
```

**Expected output** (captured on this Mac)

```
openshell 0.0.111
OpenSSL 3.6.4 25 Aug 2026 (Library: OpenSSL 3.6.4 25 Aug 2026)
mcp ok
```

> 📌 **Two things this module is not.** `policykit` (`decide`, `validate`, `harden`) is the course's **teaching model and lint**. It is not OpenShell and not OpenShell's **prover**. The real enforcement is Landlock, seccomp and the egress proxy on the Spark. The real "what does this change newly allow?" check is the prover, which waits for a human in `openshell term`. Use policykit to think *before* you push; trust the Spark's answer *after*.

✓ Checkpoint: the three commands print a version or `mcp ok`, and you can say in one sentence why policykit is not the prover.

## 1 · The threat model: what a prompt-injected claw can reach

Start from what an attacker wants, not from the YAML.

| Asset | Example | Most valuable because |
|---|---|---|
| sandbox filesystem | `/sandbox`, the agent's own config tree | secrets and state live here |
| provider credentials | model API keys | reusable anywhere |
| channel tokens | Telegram / Slack bot tokens | each one is a live outbound path |
| CSV / BMS data | chiller exports, live points | the customer's data |
| **tool authority** | `write_setpoint` | it changes a real building |

The adversaries are indirect prompt injection (anything the agent reads: web pages, emails, MCP tool results, files), malicious skills or plugins from hubs, a compromised dependency pulled through the `npm` / `pypi` presets, and an insider with host access. All but the insider start the same way: **the model reads text an attacker controls.** The OpenClaw CVE classes from Module 01 (TOCTOU file escapes, allow-list env disclosure, MCP loopback privilege escalation; per the research tutorial, citing Cyera) and the CSA analysis of indirect prompt injection all start there too. So your control is not a better prompt. It is what the process can physically do next.

Lab 07-1 turns that into twelve concrete attacker steps. It decides each one against a **course-made first draft** (`policies/balanced_draft.yaml`: the pypi preset's shape, the BMS still at L4, OTel still in audit, `best_effort`) and against the tutorial's **production policy** (`policies/prod.yaml`).

**Expected output** (captured on this Mac)

```
│ #    attacker step                             action                                                draft              prod
│ ───  ────────────────────────────────────────  ────────────────────────────────────────────────────  ─────────────────  ─────────
│ A1   phone home to a cloud model               curl → api.openai.com:443                             ✕ denied           ✕ denied
│ A2   read its own secrets file                 read  /sandbox/.hermes/.env                           ✓ allowed          ✓ allowed
│ A3   POST the secrets to an attacker host      python3.12 → paste.attacker.example:443 POST /upload  ✕ denied           ✕ denied
│ A4   hide the secrets in a GET URL             pip → pypi.org:443 GET /simple/c2stbGl2ZS0xMjM0/      ✓ allowed          ✕ denied
│ A5   cloud metadata (SSRF)                     curl → 169.254.169.254:80                             ✕ denied           ✕ denied
│ A6   write_setpoint over MCP                   python3.12 → bms.alto.local:8443 MCP tools/call wri…  ✓ allowed          ✕ denied
│ A7   pip install a package                     pip → files.pythonhosted.org:443 GET /packages/evil…  ✓ allowed          ✕ denied
│ A8   write to /usr (plant a sitecustomize.py)  write /usr/lib/python3/sitecustomize.py               ✕ denied           ✕ denied
│ A9   rewrite its own harness config            write /sandbox/.openclaw/openclaw.json                ✓ allowed          ✓ allowed
│ A10  send data out as 'telemetry'              python3.12 → otel.alto.local:4318 POST /v1/logs       ⚠ allowed (audit)  ✕ denied
│ A11  become root                               run as root                                           ✕ denied           ✕ denied
│ A12  call the BMS with curl instead            curl → bms.alto.local:8443 MCP tools/call read_point  ✕ denied           ✕ denied
  (for the attacker, ✓ allowed means the step worked)
```

```
│ policy  steps that worked
│ ──────  ─────────────────  ────────────
│ draft   6 / 12             ██████░░░░░░
│ prod    2 / 12             ██░░░░░░░░░░
```

Read the two ✓ rows under **prod**. `include_workdir: true` makes `/sandbox` writable, and the agent owns its own config tree, so it can read `/sandbox/.hermes/.env` and rewrite `/sandbox/.openclaw`. NemoClaw's docs say the same thing: that tree is not an isolation boundary. There is no filesystem rule that fixes this. There are two real fixes. Keep secrets out of files (OpenShell credential handles, Module 05). And make sure nothing the agent reads can leave: under prod, steps A3 and A4 have no host to send it to.

Some limits sit outside any policy file. NemoClaw documents four:

| Limitation | Why it matters | Mitigation |
|---|---|---|
| bypassing managed gateway paths | policy and inference auth are not enforced for runtimes launched outside the NemoClaw-managed path | start agents only via the managed entrypoints; never `docker exec` a second agent into the sandbox |
| same-UID native lifecycle | supervisor, gateway and agent share the sandbox UID; a same-user agent can signal or imitate peers | put nothing in the sandbox you would not give the agent |
| raw filesystem writes bypass scanners | scanners see tool calls, not `echo secret > file` | Landlock write scoping; keep secrets out of files |
| encoded secrets undetected | regex redaction misses Base64 / hex | OpenShell credential handles, not file-borne secrets |

✓ Checkpoint: you can name the two attacker steps that still work under prod, and say why the answer is "nowhere to send it" rather than another filesystem rule.

## 2 · The hardening checklist (Alto sovereign profile)

The research tutorial's §6.2 gives one concrete setting per line, each with its source. The last column says how this module checks it.

| Area | Setting | Checked here by |
|---|---|---|
| Network | every endpoint `protocol: rest` (or `mcp`) with `enforcement: enforce`; explicit `rules`, not `access: full`; no wildcard hosts; a narrow `allowed_ips` CIDR for private hosts | `harden()` (lab 07-2) |
| Inference hosts | never put `api.openai.com` or `integrate.api.nvidia.com` in the policy; route inference through OpenShell | `harden()` → HIGH |
| Binaries | one binary per endpoint; SHA256 TOFU pinning; install tools at image build, not at runtime | `harden()` (count) · Dockerfile lint (lab 07-3) · TOFU: OpenShell only |
| Metadata SSRF | NemoClaw injects `AWS_EC2_METADATA_DISABLED=true`; OpenShell blocks `169.254.0.0/16` regardless | `decide()` (always blocked) |
| Filesystem | `landlock.compatibility: hard_requirement`; minimal `read_write`; kernel ≥ 6.2 (Landlock ABI 3) | `harden()` · `validate()` (`/` refused) |
| Kernel / process | seccomp blocks mount, pivot_root, bpf, perf_event_open, userfaultfd, kexec, memfd_create and AF_PACKET / AF_BLUETOOTH / AF_VSOCK; `no_new_privs`; `RLIMIT_CORE=0`; non-root user | `validate()` (root) · the rest: OpenShell only |
| Gateway | keep `policy_validation_failure_mode = "fail_closed"` (the default), not `retain_last_valid` | not in the policy file |
| Tier | Restricted for kiosk / always-on claws; Balanced only where `pypi` / `npm` are really needed; never Personal on shared hardware | onboarding record |
| Channels | treat every token as a live outbound path; pair devices explicitly (`NEMOCLAW_DISABLE_DEVICE_AUTH` is retired and ignored by OpenClaw 2026.9.1) | not in the policy file |
| Recovery | snapshot before changes; on suspicion, destroy and recreate from trusted inputs | your runbook (Module 08) |
| Formal check | let OpenShell's prover evaluate what a change newly allows, then wait for human approval | the prover, not this course |

**A vendor asks you to add `api.openai.com:443` "so the agent can use GPT for summaries"** (Part 6, exercise 1). Say no to the policy entry. A provider host in `network_policies` puts the key inside the sandbox and bypasses usage tracking; lab 07-2 shows `harden()` flagging it HIGH and `decide()` then allowing the call. If a cloud model really is allowed, register it as an OpenShell provider and keep calling `inference.local`, so the gateway holds the key:

```bash
# on: spark
# --credential KEY with no =VALUE reads the key from your environment, so it never sits on the command line
openshell provider create --name openai-summaries --type openai --credential OPENAI_API_KEY
openshell inference set --provider openai-summaries --model <model>
```

The `KEY`-only form is in the laptop CLI 0.0.111 help; check `openshell provider create --help` on the Spark. Or use `NEMOCLAW_PROVIDER=routed` / `custom`. In the **sovereign** tier the answer is no either way: the model must run on-prem.

**Why `hard_requirement` matters more on a fleet** (Part 6, exercise 2). On a mix of edge boxes with different kernels, `best_effort` quietly downgrades Landlock and only emits a High-severity `DetectionFinding`. `hard_requirement` refuses to start. A hidden weakening becomes a loud failure you notice during rollout, not a gap you find in an audit.

✓ Checkpoint: for each checklist line, you can say whether a lint can check it, and what you say to the vendor.

## 3 · L6.1 — A production policy for Alto Ops Claw

This is the research tutorial's Lab 6.1 policy, verbatim, in `policies/prod.yaml`:

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

What each part buys you:

- **MCP rules are an allow-list.** Three read tools are allowed by name. `deny_rules` name the two write tools as a second line. A tool nobody named (say `reboot_controller`) is still denied.
- **`allowed_ips: ["10.20.0.15/32"]`** pins the private BMS host to one address.
- **`otel_collector` allows exactly one call:** `POST /v1/traces`. Not `/v1/logs`, not a GET.
- **`on_error: fail_closed`** means a broken redactor blocks traffic instead of passing it through unredacted.
- **`hard_requirement`** and a non-root uid lock the static layers, which are fixed when the sandbox is created.

Lab 07-2 runs it through three checkers. The real CLI parses it offline and stops only because no gateway answers:

**Expected output** (captured on this Mac, DRY mode)

```
$ openshell policy set alto-ops --policy week26/07_hardening/policies/prod.yaml --wait   [this laptop]
Error:   × transport error
  ├─▶ tcp connect error
  ├─▶ tcp connect error
  ╰─▶ Connection refused (os error 61)

✓ real CLI 0.0.111: the YAML and every field name parsed; it stopped only because no gateway answers
✓ policykit.validate: 0 errors, 0 warnings
│ severity  where                        finding (course §6.2 lint)
│ ────────  ───────────────────────────  ────────────────────────────────────────────────────
│ OK        landlock                     hard_requirement — refuses to start if Landlock can…
│ LOW       otel_collector.endpoints[0]  otel.alto.local: private host without allowed_ips —…
```

The one LOW is the course lint being stricter than the docs (an exact private host is allowed; only wildcard or hostless entries are blocked). Pin `otel.alto.local` to its `/32` when you know the IP.

Then twelve weakened copies, one checklist line each. Read the columns: each checker sees something different.

**Expected output** (captured on this Mac)

```
│ weakened copy                 §6.2 line                real CLI 0.0.111                          policykit.validate                    policykit.harden (new)
│ ────────────────────────────  ───────────────────────  ────────────────────────────────────────  ────────────────────────────────────  ───────────────────────────────────────────
│ bms enforcement: audit        network: enforce         parsed                                    ok                                    MEDIUM · bms.alto.local: enforcement audit
│ bms back to L4 (no protocol)  network: protocol mcp    parsed                                    ok                                    MEDIUM · bms.alto.local: L4 only (host/port
│ otel access: full             network: explicit rules  parsed                                    ok                                    MEDIUM · otel.alto.local: access: full — re
│ otel host *.alto.local        network: no wildcards    parsed                                    ok                                    HIGH · wildcard host '*.alto.local' — lis
│ add api.openai.com            never inference hosts    parsed                                    ok                                    HIGH · api.openai.com is an inference pro
│ landlock best_effort          fs: hard_requirement     parsed                                    ok                                    MEDIUM · best_effort — a skipped path only
│ curl on bms_mcp too           one binary per endpoint  parsed                                    ok                                    LOW · 2 binaries on one entry — prefer o
│ read_write += /usr            fs: minimal read_write   parsed                                    ok                                    MEDIUM · broad writable paths ['/usr'] — ke
│ run_as_user: root             process: non-root        parsed                                    ✕ process.run_as_user / run_as_group  —
│ otel tls: skip                network: inspection on   parsed                                    ok                                    MEDIUM · otel.alto.local: tls: skip — no in
│ Version: 1 (--full header)    (the playbook trap)      ✕ unknown field `Version`, expected one   ✕ unknown top-level field 'Version'   —
│ mcp: max_bytes (typo)         (schema)                 ✕ network_policies.bms_mcp.endpoints.\[0  ok                                    —

│ weakened copy                 probe (was ✕ deny under prod.yaml)                    now
│ ────────────────────────────  ────────────────────────────────────────────────────  ───────────────
│ bms enforcement: audit        python3.12 → bms.alto.local:8443 MCP tools/call wri…  ⚠ allow (audit)
│ bms back to L4 (no protocol)  python3.12 → bms.alto.local:8443 MCP tools/call wri…  ✓ allow
│ otel access: full             python3.12 → otel.alto.local:4318 POST /v1/logs       ✓ allow
│ otel host *.alto.local        python3.12 → nas.alto.local:4318 POST /v1/traces      ✓ allow
│ add api.openai.com            python3.12 → api.openai.com:443 POST /v1/chat/compl…  ✓ allow
│ curl on bms_mcp too           curl → bms.alto.local:8443 MCP tools/call read_point  ✓ allow
│ read_write += /usr            write /usr/lib/python3/x.py                           ✓ allow
```

The real parser catches YAML and field *names* (`Version` from a `policy get --full` header, a typo inside `mcp:`). `validate` catches bad *values* (root). Only the lint sees the quiet weakenings: the files that parse fine and still open a hole.

Next, the lab starts the **real** mock BMS on this laptop and asks it for its tools. A small course gate forwards the allowed calls to the server and refuses `write_setpoint` before it leaves:

**Expected output** (captured on this Mac)

```
◆ mock BMS on port 8443 (documented 8443). The policy talks about bms.alto.local:8443; here the same server runs on localhost, and the course gate evaluates each call as if it went to bms.alto.local:8443.
$ python week26/common/bms_mcp_server.py  # port 8443 &   [this laptop, background → bms_mcp.log]
✓ ready in 3.0s → http://localhost:8443/mcp
✓ tools/list from the REAL mock server: read_point, list_alarms, get_trend, write_setpoint
■ stopped python (pid 93280)
│ tool (from the real server)  course gate          real answer / reason
│ ───────────────────────────  ───────────────────  ────────────────────────────────────────────────────
│ read_point                   ✓ forwarded          PLANT.KW_PER_RT=0.897 at 2026-09-27T23:45 (mock BMS…
│ list_alarms                  ✓ forwarded          ALARM plant efficiency 0.901 kW/RT > 0.85 since 202…
│ get_trend                    ✓ forwarded          PLANT.KW_PER_RT hourly: 2026-09-27T22:00 0.904; 202…
│ write_setpoint               ✕ 403 (course gate)  bms_mcp: deny_rule tools/call write_setpoint → 403 …
```

The gate is course code with a course-made message. On the Spark, the OpenShell proxy does this job and answers with a 403 JSON body.

**Apply it on the Spark.** This is a change, so the lab sends it through `change()`. It shows a read-only `policy get` first, and runs the `set` only in LIVE mode with 🔓 on:

```bash
# on: spark
openshell policy get alto-ops
openshell policy set alto-ops --policy prod.yaml --wait
openshell policy list alto-ops
```

**Expected output** (captured on this Mac, DRY mode)

```
$ scp prod.yaml <spark>:~/week26/07_hardening/prod.yaml   [DRY]
$ openshell policy get alto-ops   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
version: 1
filesystem_policy:
  include_workdir: true
  ...
network_policies:
  <the entries your sandbox has today>
$ cd ~/week26/07_hardening && openshell policy set alto-ops --policy prod.yaml --wait   [DRY]
◈ (dry run — the change above would be applied here)
```

> ⚠ **Static sections.** `policy set` changes the network sections live. `filesystem_policy`, `landlock` and `process` are fixed when the sandbox is created. To get `hard_requirement` and this `read_write` list in force, create the sandbox with `--policy prod.yaml` (Module 08). The course sources do not say what `policy set` does with a changed static section, so read `openshell policy get alto-ops --full` afterwards.

**Prove the deny path.** Ask the claw to "set chiller 2 setpoint to 6.5°C". Then read the sandbox log and look for the 403 on `write_setpoint`. Look for a failed tool span in Phoenix as well (Module 05). Use `-n`, not `--tail`: `--tail` streams, and the runner never runs a streaming command in the foreground.

```bash
# on: spark
openshell logs alto-ops -n 50 --source sandbox
```

The design idea behind all this: **writes stay out of the sandbox.** The write path is a separate, human-approved service (the Alto Copilot approvals flow). The claw can only *request* a write, through a ticket. You write that entry in the exercise.

✓ Checkpoint: lab 07-2 shows prod.yaml parsed by the real CLI, 0 validate errors, one LOW, and `write_setpoint` refused before it reached the mock BMS.

## 4 · L6.2 — Custom images and blueprints

A policy can only trust binaries that exist. OpenShell pins each one to its first-seen SHA256 (trust on first use), so the tools belong **in the image**, installed at build time, not fetched by the agent at runtime. NemoClaw builds a sandbox from your own image with `nemoclaw onboard --from ./Dockerfile` (or `--from <image>`).

Lab 07-3 generates the Alto Ops Claw image from a small spec: the research tutorial's L3.8 Dockerfile, with the `USER` taken from the policy's `run_as_user`. It lints the result, then lints a "just make it work" copy:

**Expected output** (captured on this Mac, DRY mode)

```
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y python3.12 python3-pip curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages uv && uv pip install --system 'nvidia-nat[langchain,mcp,profiler,opentelemetry]'
COPY workflows/alto_ops /app/alto_ops
RUN uv pip install --system -e /app/alto_ops
COPY workflow.sandbox.yml /app/workflow.yml
USER 1500
WORKDIR /sandbox
✓ final USER 1500 (non-root)
✓ no runtime installs, no secrets, every policy binary (python3.12) comes from apt
◆ The tutorial calls this Dockerfile its own composition: verify the NAT install line on your base image. Course assumption: apt's python3.12 on ubuntu:24.04 is /usr/bin/python3.12, the binary prod.yaml trusts.

→ the same image, 'just make it work' edition:
✕ final USER is root: OpenShell requires a non-root identity
⚠ USER root ≠ policy process.run_as_user 1500
⚠ pipes a download into a shell — pin and verify what you fetch
✕ a secret in ENV/ARG ends up in the image — use an OpenShell provider
✕ installs at runtime (CMD) — bake tools in at build time, so no egress is needed
```

The tutorial calls that Dockerfile its own composition, so verify the NAT install line on your base image. OpenShell rejects root, which is why the lint treats `USER root` as an error.

```bash
# on: spark
nemoclaw onboard --from ./Dockerfile
```

The onboard wizard is interactive. Type it in the ⌨ terminal yourself; the lab only copies the Dockerfile over. The build context also needs `workflows/alto_ops` and `workflow.sandbox.yml`, which Module 08 assembles.

Other blueprint choices the tutorial lists, all behind the same OpenShell boundary:

| Choice | How | Note |
|---|---|---|
| harness | Hermes (`nemohermes`, API on 8642, Langfuse plugin) or Deep Agents (`nemo-deepagents`, Nemotron 3 Ultra default) | first-class alternatives to OpenClaw |
| model routing | `NEMOCLAW_PROVIDER=routed` with `NVIDIA_INFERENCE_API_KEY`; `custom` / `anthropicCompatible` for compatible endpoints | cloud — not for the sovereign tier |
| stay on the Spark | `NEMOCLAW_PROVIDER=ollama`, `vllm` or `install-vllm` | the sovereign default |
| Podman hosts | `NEMOCLAW_GATEWAY_RUNTIME=podman` | selects Podman for the gateway container |

✓ Checkpoint: the generated Dockerfile lints clean, and you can name three things the "just make it work" copy got wrong.

## 5 · L6.3 — Remote and external gateways

There are two ways to reach a gateway that is not on your laptop. They have very different trust boundaries.

**A remote gateway you own**, provisioned and driven from your CLI over SSH. The playbook says:

**Expected output** (REFERENCE — quoted from playbook-openshell/README.md)

```
To manage a gateway on remote hardware from a separate workstation, ensure passwordless SSH works first, then use `openshell gateway start --remote <username>@<hostname>`
| TLS / certificate errors when adding a remote gateway by LAN IP | Gateway certificate is valid for `openshell`, `localhost`, and `127.0.0.1` — not the LAN IP | Map `openshell` to the hardware IP in `/etc/hosts`, then register with `openshell gateway add https://openshell:8080 --remote <user>@<hardware-ip>` |
```

```bash
# on: laptop
# first add "<spark-ip> openshell" to /etc/hosts on this workstation (the lab never edits it)
openshell gateway start --remote <user>@<spark-host>
openshell gateway add https://openshell:8080 --remote <user>@<spark-ip>
```

> ✏ **Corrected from the research tutorial.** Its Lab 6.3 prints `openshell gateway add https://openshell:8080 --remote` with no value after `--remote`. The real CLI rejects that. The playbook writes `--remote <user>@<hardware-ip>`, and so does this module. Also, the laptop CLI 0.0.111 has **no** `gateway start` subcommand at all. The playbook documents it for the OpenShell it installs, so run `openshell gateway --help` on your machine before you script it.

**Expected output** (captured on this Mac)

```
$ openshell gateway add https://openshell:8080 --remote   [this laptop]
  → the tutorial's line, as printed: exit 2 · error: a value is required for '--remote <REMOTE>' but none was supplied
$ openshell gateway add https://openshell:8080 --remote me@spark-01.alto.local   [this laptop]
  → with the SSH target: exit 1 · Error:   × mTLS certificates for gateway 'openshell' were not found.
$ openshell gateway start --remote user@spark-01.alto.local   [this laptop]
  → provision a remote gateway: exit 2 · error: unrecognized subcommand 'start'
```

**An external gateway the platform team owns** (experimental). Where infrastructure already runs OpenShell on Kubernetes / Helm, NemoClaw's `nemoclaw-blueprint-runner` targets it through `blueprint.yaml` (`blueprint/blueprint.yaml`, verbatim from the tutorial):

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

The rules the tutorial lists: a **bare HTTPS origin**; an **exact** OpenShell release (min = max); a **PEM-only CA bundle ≤ 1 MiB** that is a **regular file, not a symlink**; an **absolute** authentication file path. The SDK checks the certificate and hostname against that bundle and uses platform DNS, so you must control how the target hostname resolves. Lab 07-3 writes a throwaway CA and credential file under `.runs/fakeroot/`, validates the tutorial's blueprint with a course validator, and then breaks it eleven ways:

**Expected output** (captured on this Mac)

```
✓ bare HTTPS origin: https://openshell.alto.local:8443
✓ exact OpenShell release: min 0.0.116 · max 0.0.116 · expected 0.0.116
✓ CA bundle: 1 PEM certificate(s), 607 B, regular file · sha256 3e905490faceb9c5…
✓ authentication path: /var/run/openshell-target/authentication (mode 600)
```

```
│ broken copy                  caught by                  message
│ ───────────────────────────  ─────────────────────────  ────────────────────────────────────────────────────
│ endpoint has a path          ✕ bare HTTPS origin        https://openshell.alto.local:8443/api/v1 — want htt…
│ endpoint is http://          ✕ bare HTTPS origin        http://openshell.alto.local:8080 — want https://hos…
│ endpoint has a user          ✕ bare HTTPS origin        https://admin@openshell.alto.local:8443 — want http…
│ a version range              ✕ exact OpenShell release  min 0.0.116 · max 0.1.2 · expected 0.0.116 — must b…
│ expected_release not exact   ✕ exact OpenShell release  min 0.0.116 · max 0.0.116 · expected >=0.0.116 — mu…
│ CA is a symlink              ✕ CA bundle                /var/run/openshell-target/ca-link.pem is a symlink …
│ CA over 1 MiB                ✕ CA bundle                /var/run/openshell-target/ca-big.pem is 1.10 MiB — …
│ CA bundle has a private key  ✕ CA bundle                /var/run/openshell-target/ca-with-key.pem holds PRI…
│ CA is DER, not PEM           ✕ CA bundle                /var/run/openshell-target/ca.der is not PEM-only (b…
│ CA path is a directory       ✕ CA bundle                /var/run/openshell-target/ca-dir.pem is not a regul…
│ relative credential path     ✕ authentication path      'authentication' is not absolute
✓ 11/11 broken copies caught by the rule they break
```

On the Spark, the real runner does these checks. Both commands are read-only: `plan` "validates, fingerprints CA, no network"; `status --external-target` makes "one credential-free health request".

```bash
# on: spark
NEMOCLAW_BLUEPRINT_PATH=$HOME/week26/07_hardening/blueprint nemoclaw-blueprint-runner plan
NEMOCLAW_BLUEPRINT_PATH=$HOME/week26/07_hardening/blueprint nemoclaw-blueprint-runner status --external-target
```

The tutorial writes `NEMOCLAW_BLUEPRINT_PATH=/abs/path/blueprint` and does not say whether it wants the folder or the file. Check `nemoclaw-blueprint-runner --help`.

**The trust boundary** (Part 6, exercise 3):

| | `openshell gateway start --remote` | `nemoclaw-blueprint-runner` + external target |
|---|---|---|
| who runs the gateway | you, from your CLI over SSH | the platform team |
| lifecycle | yours: start, stop, destroy, upgrade | theirs: you get plan and status only |
| authentication | SSH + the mTLS material the gateway provisions | a credential file + a pinned CA bundle |
| OpenShell version | whatever you installed | pinned: min = max = `expected_release` |
| trust boundary | your workstation ↔ your Spark | your claw ↔ someone else's control plane |

Two notes before you promise anything to a customer. NemoClaw's multi-node tab is scoped to **DGX Station**; for two Sparks, NVIDIA's blog covers clustering at the vLLM level, not through NemoClaw, so verify it first. For training or burst capacity, Brev hosts NemoClaw launchables and OpenShell agent sandboxes. The DLI course "Securing Agents with OpenShell and NemoClaw" is the closest official curriculum to this week.

✓ Checkpoint: lab 07-3 catches 11/11 broken blueprints, and you can explain who owns the lifecycle in each of the two gateway paths.

## 6 · Evidence for the owner, and the Alto Copilot tiers

The tier decides whether "no data left the building" can even be true (§6.6):

| Alto Copilot tier | Claw pattern | Inference | Posture | Telemetry |
|---|---|---|---|---|
| Cloud | NAT workflows behind Copilot's gateway; optional OpenShell sandbox per tenant on Brev / K8s | NVIDIA endpoints or Model Router | Development / Integration Testing during build; Locked-Down in prod | Phoenix / OTel per tenant project |
| Sovereign (on-prem) | NemoClaw on DGX Spark / Station; NAT inside OpenShell | local vLLM / Ollama via `inference.local` | Restricted + custom presets (`alto-bms`, `otel_collector`); `hard_requirement` | OTel collector on-prem; audit trail from `openshell` logs |
| Edge | Restricted OpenClaw / Hermes claw next to the BMS, read-only tools only | local NVFP4 model | Locked-Down; writes only through the approvals service | local file exporter + periodic sync |

The owner hears three sentences: the only inference route is on-prem; every network decision is logged; write authority is structurally absent from the agent. **Evidence** backs each one (Part 6, exercise 5). Lab 07-4 checks that every command parses on the laptop CLI:

**Expected output** (captured on this Mac, DRY mode)

```
│ evidence                           command                                               laptop CLI 0.0.111         what good looks like
│ ─────────────────────────────────  ────────────────────────────────────────────────────  ─────────────────────────  ────────────────────────────────────────────────────
│ policy in force: no outside hosts  openshell policy get alto-ops --full                  ✓ parses                   network_policies lists only inside hosts
│ every revision last month          openshell policy list alto-ops                        ✓ parses                   each revision reviewed; none adds an outside host
│ the only inference route           openshell inference get                               ✓ parses                   provider = your on-prem vLLM / Ollama
│ a month of network decisions       openshell logs alto-ops -n 5000 --since 720h --sour…  ✓ parses                   0 allow decisions to outside hosts
│ Landlock applied, no findings      docker logs <openshell-alto-ops container> --tail 50  (docker, not parsed)       'Applying Landlock filesystem sandbox', no Detectio…
│ traces stayed on-prem              docker ps --filter name=otel                          (docker, not parsed)       the OTel collector runs on the Spark
│ the tier                           nemoclaw alto-ops policy list                         (nemoclaw, not on laptop)  Restricted + custom presets only
```

```bash
# on: spark
openshell policy get alto-ops --full
openshell policy list alto-ops
openshell inference get
openshell logs alto-ops -n 5000 --since 720h --source sandbox
```

Whether your gateway still holds 30 days of logs is not in the course sources. Export them daily and hand over the exports. The lab's log checker counts `allow` decisions to hosts outside the building. It runs on two **course-made EXAMPLE logs**, because no course source prints a real `openshell logs` line:

**Expected output** (captured on this Mac)

```
◆ example clean month: 3 allow · 3 deny · 1 inference · 0 line(s) not parsed
✓ example clean month: 0 allow decisions to outside hosts
◆ example leaky month: 5 allow · 3 deny · 1 inference · 1 line(s) not parsed
✕ allow to an OUTSIDE host: 2026-09-21T19:02:11Z alto-ops allow api.openai.com:443 /usr/bin/python3.12 POST /v1/chat/completions audit-violation
✕ allow to an OUTSIDE host: 2026-09-21T19:02:12Z alto-ops allow 203.0.113.7:443 /usr/bin/python3.12 CONNECT
```

The leaky month fails on an `allow` marked as an audit violation. That is `enforcement: audit` doing what the docs say: log, then forward. On a real log the checker also prints how many lines it could not parse. **Zero parsed lines proves nothing**, so adapt the pattern to your format before you sign anything.

✓ Checkpoint: you can list the evidence items and the command behind each, and explain why one audit-mode allow breaks the claim.

## Labs — run them here

**labs/lab07_1_threat_model.py** — Twelve prompt-injected attack steps, decided against a first-draft policy and the production policy, plus the limits no policy fixes.

**labs/lab07_2_prod_policy.py** — The production policy through the real CLI parser, validate and the hardening lint; twelve weakened copies; the real mock BMS tool list; `policy set` via change().

**labs/lab07_3_blueprints_and_gateways.py** — A custom image Dockerfile and its lint, the external-gateway blueprint validator with eleven broken copies, and the remote-gateway commands.

**labs/lab07_4_sovereign_evidence.py** — The "no data left the building" evidence pack: tiers, commands, the policy host check and the log checker.

## Try it yourself

`exercises/ex07_approvals_endpoint.py` is Part 6 exercise 4. The Alto Copilot approvals service (`approvals.alto.local:443`) must be reachable so the claw can request a write, but the claw must never approve one. Four TODOs:

1. Two allow rules: create a ticket (`POST /api/v1/tickets`) and read one (`GET /api/v1/tickets/<id>`).
2. One deny rule that really matches every approve URL.
3. The enforcement mode.
4. Exactly one binary.

```bash
# on: laptop
.venv/bin/python week26/07_hardening/exercises/ex07_approvals_endpoint.py
```

> ✏ **Corrected from the research tutorial.** Its solution prints the deny rule as `{ method: "", path: "/api/v1/tickets//approve" }`: an empty method and a double slash. Most likely two asterisks were eaten as Markdown emphasis. As printed, the rule never matches a real approve URL, so it is dead code, and only the allow-list stops an approval. The course solution uses `{ method: "*", path: "/api/v1/tickets/*/approve" }`. The checker tests that your deny rule really matches `POST /api/v1/tickets/42/approve`.

**Expected output** (captured on this Mac, the starter as shipped)

```
✕ validate: network_policies.approvals.endpoints[0]: protocol rest with no access or rules is rejected — an L7 endpoint without rules does not mean 'allow all' (the `host:443::rest` trap)
✕ TODO 3: enforcement is 'audit' — under audit, a denied approve is logged and then FORWARDED to the service
✕ TODO 4: binaries are [] — list exactly one: the Python that runs the claw
✕ TODO 2: no deny_rule matches POST /api/v1/tickets/42/approve — a deny rule that never matches a real URL is dead code (see the tutorial's `/api/v1/tickets//approve`)
✕ TODO 1/2: fix validate first
✕ harden: MEDIUM approvals.alto.local: enforcement audit — violations are logged but forwarded; flip to enforce once rules are validated

⚠ fix the ✕ lines above, save, and run again.
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ policykit.validate: the production policy + your entry has 0 errors
✓ TODO 3: enforcement: enforce — a denied approve gets a 403, not a log line
✓ TODO 4: one binary, /usr/bin/python3.12
✓ TODO 2: your deny_rule really matches POST /api/v1/tickets/42/approve
✓ TODO 1 + 2: all 9 decisions right — create ✓ read ✓ approve ✕ (every method) delete ✕ curl ✕
✓ harden: no HIGH/MEDIUM finding on the approvals entry

═ Done. The claw can ask for a write; only a human, outside the sandbox, can grant one.
```

<details><summary>Hint — why "*" and not POST in the deny rule?</summary>

Run the checker with `method: POST`. The "approve with a GET link" row fails. In the course model a glob `*` also matches `/`, so the GET rule `/api/v1/tickets/*` matches `/api/v1/tickets/42/approve`. OpenShell's own globs look segment-based (the docs use `**` for "many segments"), so verify on your unit. A deny on every method costs nothing and holds either way.

</details>

<details><summary>Hint — enforcement</summary>

`audit` logs a violation and then **forwards** the request. For an approve call, "we logged it" is not the same as "it did not happen".

</details>

✓ Checkpoint: all six checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `error: a value is required for '--remote <REMOTE>'` | the research tutorial's `gateway add … --remote` has no SSH target | add `<user>@<spark-ip>`, as the playbook does |
| `error: unrecognized subcommand 'start'` on the laptop | the laptop CLI 0.0.111 has no `gateway start` | run it with the OpenShell the playbook installs; check `openshell gateway --help` |
| `mTLS certificates for gateway 'openshell' were not found` | `gateway add --remote` needs the client TLS material the gateway provisions | provision the gateway first (`gateway start --remote`), then add it |
| `unknown field 'Version'` from `policy set` | you fed it `policy get --full` output, which has a metadata header | export with `openshell policy get <sandbox>` (no `--full`) |
| lab 07-2 skips the mock BMS step | port or venv problem | read `week26/07_hardening/.runs/bms_mcp.log` |
| lab 07-3 says the CA file is missing | `openssl` is not on the PATH | install it, or read the other rows: they do not need it |
| the evidence checker parses 0 lines of a real log | the course pattern matches the EXAMPLE shape, not your format | adapt `LINE` in lab 07-4 to your log lines |

## Next

[Lab 08 — Capstone: Alto Ops Claw v1](../08_capstone_alto_ops_claw/TUTORIAL.md): build the custom image, create the sandbox with this production policy, prove the write-deny path, trace one request across three planes, and write the runbook.
