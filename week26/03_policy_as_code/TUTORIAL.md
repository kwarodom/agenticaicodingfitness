# ▶ Reef Lab 03 — OpenShell sandboxes and policy as code

> Part of Week 26 · NemoClaw on DGX Spark, beginner to expert. You type the commands, you see the real output. Laptop labs (the NAT and OpenShell CLIs, the policy model) run for real everywhere. Spark labs also run in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's docs or playbooks, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn how OpenShell enforces a policy: what is locked at creation, what you can change live, and why every rule is bound to a binary.
- Read the policy NemoClaw wrote for your claw, three ways, and learn which export you may push back.
- Take a policy file apart: the static half, the dynamic half, and every rule form (REST, WebSocket, GraphQL, MCP, TCP).
- Break a policy fourteen ways and see which mistakes the real OpenShell CLI catches on your laptop, which the course's policykit model catches, and which only the gateway catches.
- Run the iterate loop (deny → observe → allow → verify) with a `--dry-run` preview before every change.
- Pick a posture profile, add a preset, snapshot, approve an endpoint in the TUI, and wire a sandbox to your own vLLM by hand.
- Write `alto-bms.yaml`: one binary, one private BMS API, admin writes denied.

**Time** ~90 min · **Difficulty** intermediate · **Hardware** none (DRY + laptop) · 1 DGX Spark with the Module 02 claw optional

**Sources:** the course's research tutorial *NemoClaw on DGX Spark — Beginner to Expert* (Part 2, labs L2.1–L2.7), which cites [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema) · [NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies) · [NemoClaw integration policy examples](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/integration-policy-examples) · [NemoClaw customize network policy](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/customize-network-policy) · [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [DGX Spark vLLM instructions](https://build.nvidia.com/spark/vllm/instructions) · [OpenShell on GitHub](https://github.com/NVIDIA/openshell) — plus the local DGX Spark playbooks `playbook-openshell` and `playbook-nemoclaw-applications`.

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs the labs, policykit and the checker |
| The laptop OpenShell CLI | `week26/.venv-openshell/bin/openshell --version` | the real parser every lab in this module asks |
| The Module 02 claw (optional) | `ssh <spark> nemoclaw my-assistant status` | labs 03-1, 03-3 and 03-4 read and change its policy; without it they run DRY |
| Your sandbox name | 🖥 Spark setup → `CLAW_SANDBOX` (default `my-assistant`) | every Spark command in this module uses it |

```bash
# on: laptop
week26/.venv-openshell/bin/openshell --version
command -v nemoclaw || echo "nemoclaw: not on this laptop"
```

**Expected output** (captured on this Mac)

```
openshell 0.0.111
nemoclaw: not on this laptop
```

That is correct. NemoClaw lives on the Spark. On the laptop, OpenShell **0.0.111** is only a parser: the labs run it against a gateway address where nothing listens (`127.0.0.1:9`). Anything the CLI checks by itself fails at once; anything that got past those checks fails with `Connection refused`, which proves it parsed. NemoClaw pins **0.0.116** on the Spark, and the latest release is **0.1.2**. When a flag differs on your unit, `--help` wins.

> 📌 **policykit is a teaching model.** `week26/common/policykit.py` follows the rules the docs describe, so you can reason about a policy before you push it. It is not OpenShell. The enforcement is Landlock, seccomp and the proxy on the Spark. Where policykit and the real CLI disagree, this module shows both.

✓ Checkpoint: `openshell --version` prints 0.0.111 on the laptop, and you know which sandbox name the labs will use.

## 1 · How OpenShell enforces a policy

OpenShell has **two enforcement points**. The static controls are locked when the sandbox is created: the filesystem (Landlock LSM) and the process (seccomp BPF and a privilege drop). The dynamic controls change on a running sandbox with `openshell policy update` or `openshell policy set`: the network (a CONNECT proxy plus an OPA policy engine) and provider credentials (swapped in by the proxy).

Five rules decide what a request can do:

| Rule | What it means for you |
|---|---|
| **Deny by default, bound to binaries** | A connection must match a `network_policies` entry on host, port **and** calling binary. Every entry needs a `binaries` list. OpenShell pins each binary to the SHA256 it first saw (trust on first use) and fails closed on a mismatch. |
| **Network namespace, not env vars** | The sandbox has its own netns. All traffic routes to the proxy at `10.200.0.1`, so a process that ignores `HTTP_PROXY` still only reaches the proxy. |
| **L4 vs L7** | An endpoint with no `protocol` is checked on host, port and binary only. `protocol: rest\|websocket\|graphql\|mcp\|json-rpc\|tcp` turns on request inspection, with `rules` / `deny_rules` or an `access` preset (`full`, `read-only`, `read-write`). |
| **audit vs enforce** | `enforcement` defaults to `audit`: violations are logged and the traffic is forwarded. `enforce` returns 403 with a JSON body. |
| **SSRF protection** | `127.0.0.0/8`, `169.254.0.0/16` and `0.0.0.0` are always blocked, even with `allowed_ips`. Private RFC 1918 ranges need an exact host or a narrow `allowed_ips` CIDR. |

The NemoClaw OpenClaw baseline gives read-write to `/sandbox`, `/tmp`, `/dev/null`, `/dev/pts`, read-only to `/usr`, `/lib`, `/proc`, `/dev/urandom`, `/app`, `/etc`, `/var/log`, `/var/lib/dpkg`, and runs the agent as a dedicated `sandbox` user. Landlock needs ABI 3 (Linux 6.2+). The default `compatibility: best_effort` emits a High-severity `DetectionFinding` when a rule cannot apply; `hard_requirement` refuses to start instead.

✓ Checkpoint: say why `curl` inside the sandbox cannot use an endpoint that lists only `/usr/bin/gh`, and why unsetting `HTTP_PROXY` does not help.

## 2 · L2.1 — Read the policy NemoClaw created for you

Start by reading, not writing. There are three views, and only one of them is safe to edit and push back.

```bash
# on: spark
nemoclaw my-assistant policy list
nemoclaw my-assistant policy get > current-policy.yaml
openshell policy get my-assistant --base > base.yaml
openshell policy get my-assistant --full
openshell policy list my-assistant
```

| Command | What you get |
|---|---|
| `nemoclaw <s> policy get` | the policy with metadata stripped; literal credentials replaced by `[STRIPPED_BY_MIGRATION]` (needs OpenShell 0.0.72+). **This is the file you edit.** |
| `openshell policy get <s> --base` | OpenShell's base policy, without provider-composed entries |
| `openshell policy get <s> --full` | the effective policy, including provider-composed entries, behind a metadata header |
| `openshell policy list <s>` | the revision history |

You should recognise six baseline entries: `nvidia` (`integrate.api.nvidia.com:443`, binary `/usr/local/bin/openclaw`, POST to inference and embedding paths, GET model listings), `clawhub`, `openclaw_api`, `openclaw_docs`, `npm_registry` (GET only, `openclaw` binary only) and the required `managed_inference` route. All are TLS-terminated on 443.

> ⚠ **Two spellings.** The research tutorial (citing the NemoClaw docs) writes `nemoclaw <s> policy list` / `policy add`. The local DGX Spark playbooks (`playbook-nemoclaw`, `playbook-nemoclaw-applications`) write `policy-list` / `policy-add` / `policy-remove`. Lab 03-1 tries one, then the other. Run `nemoclaw <s> --help` on your unit and use what it lists.

The trap the OpenShell playbook warns about: `--full` prepends a metadata header with a `Version` field, and `policy set` refuses it. Lab 03-1 reproduces that with the real parser:

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_1_read_policy.py
```

**Expected output** (captured on this Mac, DRY mode — step 5)

```
▣ STEP 5 · the export trap, for real on this laptop — `--full` is for reading, not for `policy set`
$ openshell policy set parse-probe --policy 03_policy_as_code/.runs/full-dump-like.yaml   [this laptop]
$ openshell policy set parse-probe --policy 03_policy_as_code/.runs/full-dump-fixed.yaml   [this laptop]
$ openshell policy get my-assistant --base --full   [this laptop]
│ what the laptop CLI 0.0.111 was given  what it said
│ ─────────────────────────────────────  ────────────────────────────────────────────────────
│ policy with a `Version:` header line   ✕ YAML: unknown field `Version`, expected one of `v…
│ same file, header removed              ✓ parsed — only the gateway connection failed
│ policy get --base --full               ✕ error: the argument '--base' cannot be used with …
✕ `Version:` header → YAML: unknown field `Version`, expected one of `version`, `filesystem_policy`, `landlock`, `process`, `network_policies`, `network_middlewares`
✕ --base --full → error: the argument '--base' cannot be used with '--full'
◆ The playbook's fix: export with `openshell policy get <s>` (no --full), or strip every line before the first `---`. `--base` and `--full` are two different views, so the CLI refuses both at once.
✓ captured on this Mac: the real parser, no gateway
```

Without a Spark, steps 1–4 print EXAMPLE shapes, and the baseline table says `◈ DRY — not checked` in every row. On a live Spark, the lab also copies your export to `03_policy_as_code/.runs/current-policy.from-spark.yaml` and checks it with policykit and the parser.

✓ Checkpoint: you know which export you will push back (`nemoclaw <s> policy get`, or `openshell policy get` without `--full`), and you have seen the `Version` error once.

## 3 · L2.2 — Anatomy of a policy file

`week26/03_policy_as_code/policies/anatomy.yaml` is the research tutorial's annotated schema policy:

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

The rule forms you will use, one per protocol. `policies/rule_forms.yaml` puts all five into one policy that parses:

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

# MCP (Module 07 returns to this)
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

You can ask the real parser yourself. Point it at a dead gateway and read the last line:

```bash
# on: laptop
cd week26 && OPENSHELL_GATEWAY_ENDPOINT=http://127.0.0.1:9 HOME=.runs/openshell-home \
  .venv-openshell/bin/openshell policy set parse-probe --policy 03_policy_as_code/policies/anatomy.yaml
```

**Expected output** (captured on this Mac)

```
Error:   × transport error
  ├─▶ tcp connect error
  ├─▶ tcp connect error
  ╰─▶ Connection refused (os error 61)
```

`Connection refused` is the good answer here: the YAML parsed, and only the gateway was missing. Lab 03-2 does this for the schema policy, the rule forms and fourteen broken variants, and asks policykit what the rules would allow:

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_2_policy_anatomy.py
```

**Expected output** (captured on this Mac — steps 3 and 4, command echo lines trimmed)

```
▣ STEP 3 · what would these rules allow? (policykit.decide — REST, MCP and TCP only)
│ form  action                                                decision  why (policykit)
│ ────  ────────────────────────────────────────────────────  ────────  ────────────────────────────────────────────────────
│ REST  gh → api.github.com:443 GET /repos/nvidia/nemoclaw    ✓ allow   github_rest: rule allow GET /repos/**
│ REST  gh → api.github.com:443 GET /repos/nvidia/nemoclaw/…  ✕ deny    github_rest: deny_rule * /repos/*/*/rulesets → 403 …
│ REST  gh → api.github.com:443 POST /repos/nvidia/nemoclaw…  ✕ deny    github_rest: no rule allows POST /repos/nvidia/nemo…
│ REST  curl → api.github.com:443 GET /repos/nvidia/nemoclaw  ✕ deny    api.github.com:443 is listed, but not for binary /u…
│ MCP   python3 → mcp.example.com:443 MCP initialize          ✓ allow   tools_mcp: rule allow initialize
│ MCP   python3 → mcp.example.com:443 MCP tools/call search…  ✓ allow   tools_mcp: rule allow tools/call search_web
│ MCP   python3 → mcp.example.com:443 MCP tools/call send_e…  ✕ deny    tools_mcp: deny_rule tools/call send_email → 403 JS…
│ MCP   python3 → mcp.example.com:443 MCP tools/call delete…  ✕ deny    tools_mcp: no MCP rule allows tools/call delete_rep…
│ TCP   psql → db.internal.example:5432                       ✓ allow   network_policies.postgres: db.internal.example:5432…
│ TCP   python3 → db.internal.example:5432                    ✕ deny    db.internal.example:5432 is listed, but not for bin…
◆ policykit does not model WebSocket frames or GraphQL operations — for those groups trust only the parser and, on the Spark, `openshell logs <s> --source sandbox`. It also matches paths with Python's fnmatch, where `*` can cross a `/`; OpenShell writes multi-segment globs as `**`. Keep `*` for one segment.

▣ STEP 4 · fourteen broken variants — who catches what?
│ variant                      policykit  CLI parser  on a real gateway (source)
│ ───────────────────────────  ─────────  ──────────  ───────────────────────────────────────────────
│ read_write: [/]              ✕ caught   · parsed    refused: INVALID_ARGUMENT (research tutorial)
│ rest, no access or rules     ✕ caught   · parsed    the `::rest` trap in YAML — verify on your unit
│ run_as_user: root            ✕ caught   · parsed    push fails validation (OpenShell playbook)
│ endpoint without port        ✕ caught   · parsed    push fails validation (OpenShell playbook)
│ Version: (capital V)         ✕ caught   ✕ caught    unknown field 'Version' (OpenShell playbook)
│ network_policies as a list   ✕ caught   ✕ caught    expected a map (NemoClaw applications playbook)
│ endpoint description:        ✕ caught   ✕ caught    same as the parser: unknown field
│ group comment:               ✕ caught   ✕ caught    same as the parser: unknown field
│ deny_rules wrapped in deny:  · missed   ✕ caught    same as the parser: unknown field `deny`
│ allow matcher `verb:`        · missed   ✕ caught    same as the parser: unknown field
│ port: "443" (a string)       ✕ caught   ✕ caught    same as the parser: expected u16
│ access: readonly (typo)      ✕ caught   · parsed    parser passes it; expect the gateway to refuse
│ protocol: sql                ✕ caught   · parsed    the CLI grammar lists sql — policykit is behind
│ landlock: strict             ✕ caught   · parsed    parser passes it; expect the gateway to refuse
◆ policykit caught 12/14 · the real parser caught 7/14 · variant files in 03_policy_as_code/.runs/variants/
```

Read the two columns together:

- **The parser catches structure**: unknown fields, a list where a map belongs, a string where a port number belongs, and a `deny:` wrapper inside `deny_rules`.
- **policykit catches semantics**: `read_write: [/]`, root, a missing port, a typo in `access`, a REST endpoint with no access and no rules. These pass the parser, so on a Spark it is the gateway that refuses them: the OpenShell playbook's troubleshooting table names root as `run_as_user` and a missing `host`/`port` for a failed push, and the research tutorial names `INVALID_ARGUMENT` for `read_write: [/]`. For the `access` typo and `landlock: strict` no source shows the gateway's message, so verify those on your unit.
- **policykit is not complete**: it misses rule-shape mistakes the parser catches, and it rejects `protocol: sql`, which the 0.0.111 CLI knows.

✓ Checkpoint: for each of `read_write: [/]`, `Version:` and `deny_rules: [{deny: …}]`, you can say who catches it: the laptop parser, policykit, or the gateway.

## 4 · L2.3 — The iterate loop: deny → observe → allow → verify

The documented workflow: create with an initial policy, watch denials, pull, edit, push, verify. On a running claw that means small additive changes, each one previewed first.

```bash
# on: spark
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

Two grammars do the work. An endpoint spec is `host:port[:access[:protocol[:enforcement[:options]]]]`. A rule spec is `host:port:METHOD:path_glob`. Remember the trap: `api.github.com:443::rest` is rejected. An L7 endpoint with a protocol but no access or rules does not mean "allow all".

Lab 03-3 feeds both grammars to the real CLI and to policykit, then runs the loop above on your Spark through `change()`: every change first runs its `--dry-run` twin as a preview, and runs for real only in LIVE mode with 🔓 Allow changes on.

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_3_iterate_loop.py
```

**Expected output** (captured on this Mac, DRY mode — steps 1 and 3, command echo lines trimmed)

```
▣ STEP 1 · --add-endpoint host:port[:access[:protocol[:enforcement[:options]]]] — who rejects what?
│ spec                                           CLI 0.0.111  policykit  why it is in the table
│ ─────────────────────────────────────────────  ───────────  ─────────  ──────────────────────────────────────────
│ api.github.com:443:read-only:rest:enforce      · parses     · parses   the docs' example: L7, read-only, enforced
│ api.github.com:443::rest                       · parses     ✕ rejects  THE TRAP: docs say the gateway rejects it
│ pypi.org:443                                   · parses     · parses   L4 only: host, port, binary
│ timescale.alto.local:5432::tcp                 · parses     · parses   Part 2 ex. 4: empty access before tcp
│ bms.alto.local:8443:read-only:rest:enforce:a…  · parses     · parses   an option: pin a /32
│ realtime.example.com:443:read-write:websocke…  · parses     · parses   the CLI's own --help example
│ api.github.com:443:readonly                    ✕ rejects    ✕ rejects  typo in the access segment
│ api.github.com:443:read-only:http              ✕ rejects    ✕ rejects  http is not a protocol
│ mcp.example.com:443:read-only:mcp              ✕ rejects    · parses   mcp is YAML-only in 0.0.111
│ db.internal.example:5432::sql                  · parses     ✕ rejects  sql: the CLI knows it, policykit not
│ api.github.com:443:read-only:rest:block        ✕ rejects    ✕ rejects  enforcement is enforce or audit
│ api.github.com:443:read-only::enforce          ✕ rejects    · parses   enforcement needs a protocol
│ api.github.com:443:read-write:tcp              ✕ rejects    · parses   tcp takes no access mode
│ api.github.com:443:read-only:rest:enforce:bo…  ✕ rejects    · parses   unknown option
│ api.github.com                                 ✕ rejects    ✕ rejects  no port
│ api.github.com:99999                           ✕ rejects    ✕ rejects  port out of range
▣ STEP 3 · flag rules the CLI enforces before it needs a gateway
│ openshell policy update …         what the laptop CLI said
│ ────────────────────────────────  ────────────────────────────────────────────────────
│ --binary alone                    ✕ --binary can only be used with --add-endpoint
│ no operation at all               ✕ policy update requires at least one operation flag
│ --rule-name with two endpoints    ✕ --rule-name is only supported when exactly one --…
│ --remove-endpoint without a port  ✕ --remove-endpoint expects host:port, got 'pypi.or…
│ --dry-run together with --wait    ✕ --wait cannot be combined with --dry-run
│ a clean --dry-run                 ✓ parsed — only the gateway connection failed
◆ The last row matters: `--dry-run` is NOT offline. It fetches the live policy from the gateway, merges your change into it and shows the result without sending it. That is why the loop below can use it as a safe preview on the Spark. Nothing is sent, so there is nothing to --wait for.
✓ steps 1–3 captured on this Mac: the real OpenShell 0.0.111 parser, no gateway
```

What the laptop taught you:

- `::rest` **parses** in the CLI. The research tutorial, citing the OpenShell sandbox-policies guide, says it is rejected, so the rejection comes from the gateway when it merges. policykit refuses it early, on purpose.
- In 0.0.111, `--add-endpoint` accepts only `tcp`, `rest`, `websocket` and `sql` as protocols. MCP, GraphQL and JSON-RPC endpoints are written in YAML (`policy set`), not with a spec string.
- `tcp` takes no access mode, so Part 2 exercise 4's `timescale.alto.local:5432::tcp` keeps the access segment **empty**.
- `--dry-run` needs the gateway: it pulls the live policy and shows the merge without sending it. The CLI refuses `--dry-run` together with `--wait`.

For presets, prefer the NemoClaw wrapper. It knows the blueprint's baseline; for example, it refuses an `npm` change when the live baseline drifted from the reviewed GET-only entry.

```bash
# on: spark
nemoclaw my-assistant policy add github --dry-run
nemoclaw my-assistant policy add github --yes
nemoclaw my-assistant policy remove github --yes
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml     # custom preset
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml --trusted-private-host 10.20.0.15 --dry-run
```

✓ Checkpoint: you can write the `--add-endpoint` spec for a read-only, enforced REST API and for a Postgres on 5432, and you know which of the two the gateway (not the laptop) would still have to accept.

## 5 · L2.4 — Operator approval in the TUI

When the agent hits an unlisted endpoint, OpenShell blocks it and shows the request in `openshell term`. If you approve it, the approval becomes a new, durable policy revision. It survives restarts of the same sandbox instance and is gone when the sandbox is destroyed and recreated. Before approval, OpenShell's prover flags risky new access (a new host with credentials, a new API method) and waits for a human.

Try it on purpose. `openshell term` is an interactive TUI, so run it in the ⌨ terminal, never from a lab:

```bash
# on: spark
openshell term
```

Then ask the assistant to *"fetch https://httpbin.org/get and show me the headers"*, approve once, and check the new revision:

```bash
# on: spark
openshell policy list my-assistant
```

The 0.0.111 CLI shows where the approval flow is configured. Lab 03-4 prints it, then simulates the lifecycle with policykit:

**Expected output** (captured on this Mac — lab 03-4, step 6)

```
▣ STEP 6 · operator approval in the TUI (L2.4) — what the real CLI says, then a simulation
$ openshell sandbox create --help   [this laptop]
      --approval-mode <APPROVAL_MODE>
          Approval mode for agent-authored policy proposals.
          `manual` (default): every proposal lands in the draft inbox for human review, regardless of the prover verdict.
          `auto`: proposals whose prover delta is empty are approved automatically; proposals with findings still require human approval. Auto mode is an explicit opt-in — `OpenShell`'s default-deny posture is preserved unless you choose otherwise.
          [default: manual]
          [possible values: manual, auto]
✓ captured on this Mac: `openshell sandbox create --help` (0.0.111)
→ in the ⌨ terminal on the Spark: `openshell term`, then ask the assistant to "fetch https://httpbin.org/get and show me the headers", approve once, and read the new revision with `openshell policy list my-assistant`
│ policy state                                    openclaw → httpbin.org:443  why (policykit)
│ ──────────────────────────────────────────────  ──────────────────────────  ────────────────────────────────────────────────────
│ rev N · before approval                         ✕ deny                      httpbin.org:443 is not in network_policies (default…
│ rev N+1 · after one approval (simulated entry)  ✓ allow                     network_policies.httpbin: httpbin.org:443 for binar…
│ same instance, after a restart                  ✓ allow                     network_policies.httpbin: httpbin.org:443 for binar…
│ destroyed + recreated                           ✕ deny                      httpbin.org:443 is not in network_policies (default…
◆ SIMULATION: the entry the TUI really writes may differ — read it with `openshell policy get`. The rule it models is the docs': an approval becomes a durable revision for this sandbox instance and is gone when the sandbox is destroyed and recreated. Before approval, OpenShell's prover flags risky new access (a new host with credentials, a new API method) and waits for a human.
```

The approved entry in the table is a **simulation**. The TUI writes its own entry, so read the real one with `openshell policy get`.

✓ Checkpoint: you can say what happens to a TUI approval after a restart (it stays) and after a destroy-and-recreate (it is gone).

## 6 · L2.5 — Presets and posture profiles

Maintained presets live in `nemoclaw-blueprint/policies/presets/`: `brave`, `brew`, `claude-code`, `discord`, `github`, `gmail`, `googlechat`, `huggingface`, `jira`, `local-inference`, `npm`, `nous-*` (Hermes), `openclaw-pricing`, `outlook`, `public-reference`, `pypi`, `slack`, `tavily`, `teams`, `telegram`, `weather`, `wechat`, `whatsapp`.

Read the risk before you apply one:

| Preset | Risk (NemoClaw security best practices, via the research tutorial) |
|---|---|
| `pypi` | GET/HEAD only, but it lets the agent install arbitrary packages |
| `github` | read/write to repos, via `git` only (binary-scoped to `/usr/bin/git`) |
| `slack`, `discord` | the WebSocket legs use `access: full`, with no inspection |
| `personal-open-internet` | removes hostname, method, path and body restrictions on ports 80/443 |
| `whatsapp`, `brew` | `access: full` + `tls: skip` pass-through (per the NemoClaw applications playbook) |

Four posture profiles from the same guide. Module 07 maps them to AltoTech deployments.

| Profile | Tier | Presets | Inference | Notes |
|---|---|---|---|---|
| Locked-Down | Restricted | none (no web search) | NVIDIA Endpoints or local Ollama | operator approval for everything else; watch the TUI |
| Development | Balanced | `pypi`, `npm` | any | keep binary restrictions; review with `openshell term` |
| Personal | Personal | `personal-open-internet` | any | trusted single-user only; recreate as Balanced when done |
| Integration Testing | custom | tight method/path entries, `protocol: rest` | any | clean up the baseline after tests |

Lab 03-4 reads `nemoclaw <s> policy list` (read-only), guesses the closest profile from real output only, and adds `pypi` through `change()` with a `--dry-run` preview.

✓ Checkpoint: pick a profile for (a) a hotel concierge claw on a shared Spark, (b) your own dev claw that needs `pip install`, and say why Personal is never right on shared hardware.

## 7 · L2.6 — Snapshots, rebuild, recovery

```bash
# on: spark
nemoclaw my-assistant snapshot create --name before-change
# ...make changes...
nemoclaw my-assistant rebuild
```

The Deep Agents quickstart shows these verbs for `nemo-deepagents`; they behave the same for the other CLIs. Lab 03-4 creates the snapshot through `change()` and prints the rebuild command for you to run when you mean it.

The rule for suspected compromise is simple: **recreate the sandbox from trusted inputs, do not try to clean it**. The agent can rewrite its own config tree, so a cleaned sandbox is not a trusted one. This also answers a common question: the fastest way to guarantee that three debugging approvals are gone is to destroy and recreate the sandbox.

✓ Checkpoint: you took (or can take) a snapshot named `before-change`, and you can explain why "recreate" beats "clean up".

## 8 · L2.7 — Raw OpenShell without NemoClaw: bring your own vLLM

This is the OpenShell playbook, condensed. It shows what `nemoclaw onboard` does under the hood and gives you full control of the model server.

```bash
# on: spark
# 1. install OpenShell CLI + gateway service
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
source ~/.bashrc && openshell --help
systemctl --user status --no-pager openshell-gateway
openshell status                      # expect: Connected
sudo loginctl enable-linger $USER     # keep gateway alive after logout

# 2. serve a model with vLLM (host 0.0.0.0, port 8000)
export HF_TOKEN=...; export MODEL_HANDLE="nvidia/Qwen3.6-35B-A3B-NVFP4"
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
openshell inference get

# 5. create a sandbox from the community OpenClaw image (interactive wizard)
export SANDBOX_NAME=openshell-demo
openshell sandbox create --keep --tty --forward 18789 --name "$SANDBOX_NAME" --from openclaw -- openclaw-start
openshell forward start --background 18789 "$SANDBOX_NAME"

# 6. verify isolation, then clean up
openshell term
openshell sandbox delete "$SANDBOX_NAME"; openshell provider delete local-vllm
```

`MODEL_HANDLE` is the DGX Spark model the OpenShell playbook recommends; any handle from the vLLM recipes for DGX Spark works. In the OpenClaw wizard choose **Custom Provider**, base URL `https://inference.local/v1`, any non-empty key (`not-needed`), **OpenAI-compatible**, and the same model id.

After step 4, the playbook tells you what to look for:

**Expected output** (REFERENCE — quoted from the DGX Spark OpenShell playbook)

```
Expected output should show `provider: local-vllm` and your chosen `model`.
```

Two gotchas from the playbook:

1. **The LAN IP, not localhost.** The gateway runs inside Docker. Inside its container, `127.0.0.1` is the container, so bind vLLM to `0.0.0.0` and give the provider the machine's IP (or `host.docker.internal` where it resolves).
2. **Never pass `--policy` with `--from openclaw`.** The community sandbox bundles its policy; a local file path can cause "file not found". The laptop parser does **not** catch this, so lab 03-5 has a guard for it.

Lab 03-5 runs the read-only checks, puts every change behind `change()`, and never runs the installer or the interactive `sandbox create` for you:

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_5_byo_vllm.py
```

**Expected output** (captured on this Mac, DRY mode — step 3 and the parser table of step 5)

```
▣ STEP 3 · the provider URL — the LAN IP, never localhost
$ hostname -I | awk '{print $1}'   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
192.168.1.42
│ OPENAI_BASE_URL                      verdict (course rule, from the playbook)              note
│ ───────────────────────────────────  ────────────────────────────────────────────────────  ────────────
│ http://localhost:8000/v1             ✕ localhost inside the gateway's container is the c…
│ http://127.0.0.1:8000/v1             ✕ loopback: the gateway cannot reach host services …
│ http://0.0.0.0:8000/v1               ✕ 0.0.0.0 is a bind address for vLLM, not a destina…
│ http://host.docker.internal:8000/v1  ⚠ works only where it resolves — check on your unit…
│ http://<spark-ip>:8000/v1            · placeholder — DRY                                   ← your Spark
⚠ DRY: `hostname -I` did not run — the last row is a placeholder, not your Spark's address
◆ Why: the gateway runs inside Docker. Inside its container, 127.0.0.1 is the container, not the Spark, so bind vLLM to 0.0.0.0 and give the provider the machine's IP.
│ the AGENT (inside the sandbox) calls  decision                 why (policykit, teaching model)
│ ────────────────────────────────────  ───────────────────────  ────────────────────────────────────────────────────
│ inference.local:443                   ◆ inspect_for_inference  inference.local is handled by the proxy's inference…
│ 192.168.1.42:8000                     ✕ deny                   private RFC 1918 IP: blocked unless declared as an …
│ 127.0.0.1:8000                        ✕ deny                   127.0.0.1 is loopback / link-local / 0.0.0.0 — alwa…
◆ Two different callers. The GATEWAY needs the LAN IP to reach vLLM. The AGENT never calls vLLM directly: it calls https://inference.local/v1 and the gateway forwards it. So you add no network_policies entry for vLLM at all.
│ sandbox create …        CLI 0.0.111 (laptop)              course guard
│ ──────────────────────  ────────────────────────────────  ────────────────────────────────────────────────────
│ the playbook's command  ✓ parses                          ✓ ok
│ with --policy added     ✓ parses                          ✕ --policy with --from openclaw: the policy is bund…
│ a typo: --kep           ✕ error: unexpected argument '-…  —
```

The second table in step 3 is the point of the whole lab. The **gateway** needs the LAN IP to reach vLLM. The **agent** never calls vLLM at all: it calls `inference.local`, and the gateway forwards the call with the credential. So there is no `network_policies` entry for your model server.

Other sandbox flavours worth knowing: `--from base` (minimal Ubuntu, no agent), `--from sdg`, `--from ./dir` or a Dockerfile; `--gpu`, `--cpu 2 --memory 4Gi`, `--upload`, `--env`, `--label`; `openshell sandbox exec -n <name> -- <cmd>` for one-shot commands. The trailing command in `create` is the health-defining main process: if it exits, the sandbox goes to `Error`. `OPENSHELL_SANDBOX_POLICY=./my-policy.yaml` saves typing `--policy`.

✓ Checkpoint: you can explain why `OPENAI_BASE_URL=http://localhost:8000/v1` fails for the provider, and why the agent's own policy needs no entry for vLLM.

## Labs — run them here

**labs/lab03_1_read_policy.py** — Read the policy NemoClaw created (list, export, base, full, revisions) and reproduce the `--full` → `policy set` trap with the real parser.

**labs/lab03_2_policy_anatomy.py** — The annotated schema, every rule form, and fourteen broken variants checked by policykit and the real OpenShell parser side by side.

**labs/lab03_3_iterate_loop.py** — The endpoint and rule spec grammars on the real CLI, then the deny → allow → verify loop on your Spark with a --dry-run preview before every change.

**labs/lab03_4_presets_posture.py** — Presets and their risks, posture profiles, a preset added and a snapshot taken through change(), and the TUI approval lifecycle simulated.

**labs/lab03_5_byo_vllm.py** — Raw OpenShell with your own vLLM: read-only checks, every change gated, the LAN-IP rule computed, and the --policy/--from openclaw guard.

## Try it yourself

`exercises/ex03_alto_bms_preset.py` is Part 2 exercise 1. Alto Ops Claw must read points from and write setpoints to the hotel BMS at `bms.alto.local:8443`, which resolves to `10.20.0.15`. It has four TODOs:

1. The endpoint: port, `protocol: rest`, `enforcement: enforce`, and `allowed_ips` pinned to `10.20.0.15/32`.
2. The rules: allow `GET /api/v1/points/**` and `POST /api/v1/setpoints/*`; deny `POST /api/v1/admin/**`.
3. The binaries: `/usr/bin/python3` only.
4. A preset name NemoClaw accepts: a lowercase, hyphenated RFC 1123 label.

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/exercises/ex03_alto_bms_preset.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ policykit.validate: no errors, no warnings
✓ endpoint: bms.alto.local:8443 · rest · enforce · allowed_ips [10.20.0.15/32]
✓ binaries: /usr/bin/python3 only
✓ decide() matrix: 9/9 as expected (policykit)
✓ preset name 'alto-bms' is a lowercase, hyphenated RFC 1123 label
✓ the real OpenShell 0.0.111 parser accepts alto-bms.yaml (policy set, no gateway)
◆ the preset form (`preset:` header, for `nemoclaw … policy add --from-file`) is NOT an OpenShell policy: YAML: unknown field `preset`

═ Done. Files: 03_policy_as_code/.runs/alto-bms.yaml and 03_policy_as_code/.runs/alto-bms.preset.yaml. Preview on the Spark with --dry-run first.
```

The checker writes two files to `03_policy_as_code/.runs/`. `alto-bms.yaml` is an OpenShell policy the real parser accepts. `alto-bms.preset.yaml` adds the `preset:` header that the NemoClaw applications playbook shows for `policy-add --from-file` — and the last line proves that `openshell policy set` would refuse it. Two tools, two file shapes.

Then preview it on the Spark before you apply it:

```bash
# on: spark
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml --trusted-private-host 10.20.0.15 --dry-run
```

> ⚠ The research tutorial passes `--trusted-private-host 10.20.0.15` in Section 2.4 but `--trusted-private-host bms.alto.local` in its exercise solution. Check `nemoclaw <s> policy add --help` on your unit for which one it expects, review the address pins the dry run prints, then rerun with `--yes`.

<details><summary>Hint — allow wraps, deny lists</summary>

An allow rule is `- allow: { method: GET, path: "/api/v1/points/**" }`. A deny rule has no wrapper: `- { method: POST, path: "/api/v1/admin/**" }`. Quote paths that contain `*`. Lab 03-2 showed what the parser says about a `deny:` wrapper.

</details>

<details><summary>Hint — why a /32</summary>

A private RFC 1918 address is blocked unless you declare the exact host or open a narrow `allowed_ips` CIDR. The checker also tries `10.20.0.16`, which must stay denied. A `/24` would let the agent reach every device on the BMS VLAN.

</details>

<details><summary>More Part 2 questions (answers inside)</summary>

- **Why is `audit` the default?** It logs violations but forwards traffic, so you learn the real access pattern. Flip to `enforce` once the rules are validated; then non-matching requests get `403 Forbidden` with a JSON body.
- **A colleague wants `read_write: [/]`.** It is refused with `INVALID_ARGUMENT`. Add a specific writable directory such as `/sandbox/tools`, and install tools at image build time (`nemoclaw onboard --from <Dockerfile>`).
- **`psql` to `timescale.alto.local:5432`?** YAML: `endpoints: [{host: timescale.alto.local, port: 5432, protocol: tcp}]` with `binaries: [{path: /usr/bin/psql}]`. CLI: `--add-endpoint timescale.alto.local:5432::tcp --binary /usr/bin/psql` — the empty access segment is required, and lab 03-3 showed that 0.0.111 refuses an access mode on `tcp`.
- **`tls: skip` on a credentialed endpoint?** It turns off placeholder credential rewriting, token injection and L7 inspection; the proxy relays ciphertext blind. A provider-credentialed endpoint also needs `allow_uninspected_credentials: true` as an explicit acknowledgement.

</details>

✓ Checkpoint: every checker line is ✓, and you can say which file you would give to `nemoclaw … policy add --from-file` and which to `openshell policy set`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `policy set` fails with `unknown field 'Version'` | you pushed a `--full` dump; its metadata header has a `Version` field | export with `openshell policy get <s>` (no `--full`) or `nemoclaw <s> policy get`; or strip every line before the first `---` |
| `invalid type: sequence, expected a map` | `network_policies` is a list of `{host, port}` | make it a map keyed by group name, each with `endpoints` and `binaries` |
| the policy "applies cleanly" but every call gets 403 | no `binaries`, or no access mode / rules on the endpoint | add the calling binary and `access:` or `rules:` (lab 03-2's two "missed by the parser" classes) |
| `--add-endpoint protocol segment must be 'tcp', 'rest', 'websocket', or 'sql'` | MCP / GraphQL / JSON-RPC endpoint written as a spec | write it in YAML and push with `openshell policy set` |
| `--wait cannot be combined with --dry-run` | a preview never sends anything | preview with `--dry-run`, then rerun with `--wait` |
| `nemoclaw … policy list` says unknown command | your NemoClaw uses the hyphenated verbs | `nemoclaw <s> policy-list` / `policy-add` / `policy-remove` (the DGX Spark playbooks' spelling) |
| `Preset must declare preset.name (lowercase, hyphenated RFC 1123 label)` | `preset.name` has an underscore or capital | use letters, digits and hyphens: `alto-bms`, not `alto_bms` |
| `failed to verify inference endpoint` after `inference set` | vLLM still loading, or the provider URL uses localhost | warm up with one chat completion; use the LAN IP from `hostname -I`; `--no-verify` only after the host API works |
| "Permission denied" / Landlock errors in the sandbox | the path is not in `read_only` or `read_write` | filesystem policy is static: add the path and **recreate** the sandbox |

## Next

[Lab 04 — Build a claw with NeMo Agent Toolkit](../04_nat_claws/TUTORIAL.md): write Alto Ops Claw as a NAT workflow with a chiller-plant tool, serve it over REST and MCP, and run it inside the kind of sandbox you just learned to write policy for.
