# ▶ Reef Lab 03 — OpenShell sandbox และ policy as code

> ส่วนหนึ่งของ Week 26 · NemoClaw บน DGX Spark ตั้งแต่ระดับเริ่มต้นจนถึงผู้เชี่ยวชาญ คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง แล็บฝั่งแล็ปท็อป (NAT CLI, OpenShell CLI และแบบจำลอง policy) รันจริงได้ทุกที่ แล็บฝั่ง Spark รันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจากเอกสารหรือ playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เข้าใจว่า OpenShell บังคับใช้ policy อย่างไร: อะไรล็อกตั้งแต่ตอนสร้าง อะไรเปลี่ยนได้ขณะรัน และทำไมทุกกฎต้องผูกกับ binary
- อ่าน policy ที่ NemoClaw เขียนให้ claw ของคุณได้สามมุม และรู้ว่า export แบบไหนที่ push กลับได้
- แยกส่วนไฟล์ policy: ครึ่ง static ครึ่ง dynamic และกฎทุกรูปแบบ (REST, WebSocket, GraphQL, MCP, TCP)
- ทำ policy ให้พังสิบสี่แบบ แล้วดูว่าความผิดพลาดไหน OpenShell CLI ตัวจริงจับได้บนแล็ปท็อป ไหน policykit ของคอร์สจับได้ และไหนมีแต่ gateway ที่จับได้
- เดินวงจร iterate loop (deny → observe → allow → verify) โดยดูตัวอย่างด้วย `--dry-run` ก่อนการเปลี่ยนทุกครั้ง
- เลือก posture profile เพิ่ม preset ทำ snapshot อนุมัติ endpoint ใน TUI และต่อ sandbox เข้ากับ vLLM ของคุณเองด้วยมือ
- เขียน `alto-bms.yaml`: binary เดียว BMS API ส่วนตัวหนึ่งตัว และห้ามเขียนส่วน admin

**Time** ~90 นาที · **Difficulty** ระดับกลาง · **Hardware** ไม่ต้องมี (DRY + แล็ปท็อป) · Spark 1 เครื่องที่มี claw จาก Module 02 (ไม่บังคับ)

**Sources:** research tutorial ของคอร์ส *NemoClaw on DGX Spark — Beginner to Expert* (Part 2 แล็บ L2.1–L2.7) ซึ่งอ้างอิง [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema) · [NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies) · [NemoClaw integration policy examples](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/integration-policy-examples) · [NemoClaw customize network policy](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/network-policy/customize-network-policy) · [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [DGX Spark vLLM instructions](https://build.nvidia.com/spark/vllm/instructions) · [OpenShell on GitHub](https://github.com/NVIDIA/openshell) — และ DGX Spark playbook ในเครื่อง `playbook-openshell` กับ `playbook-nemoclaw-applications`

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจด้วย | เพื่ออะไร |
|---|---|---|
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | รันแล็บ, policykit และตัวตรวจแบบฝึกหัด |
| OpenShell CLI บนแล็ปท็อป | `week26/.venv-openshell/bin/openshell --version` | parser ตัวจริงที่ทุกแล็บในโมดูลนี้ใช้ถาม |
| claw จาก Module 02 (ไม่บังคับ) | `ssh <spark> nemoclaw my-assistant status` | แล็บ 03-1, 03-3 และ 03-4 อ่านและเปลี่ยน policy ของมัน ถ้าไม่มีก็รันแบบ DRY |
| ชื่อ sandbox ของคุณ | 🖥 Spark setup → `CLAW_SANDBOX` (ค่าเริ่มต้น `my-assistant`) | ทุกคำสั่งฝั่ง Spark ในโมดูลนี้ใช้ชื่อนี้ |

```bash
# on: laptop
week26/.venv-openshell/bin/openshell --version
command -v nemoclaw || echo "nemoclaw: not on this laptop"
```

**Expected output** (captured on this Mac) [บันทึกจาก Mac เครื่องนี้]

```
openshell 0.0.111
nemoclaw: not on this laptop
```

ถูกต้องแล้ว NemoClaw อยู่บน Spark ส่วนบนแล็ปท็อป OpenShell **0.0.111** เป็นแค่ parser: แล็บจะรันมันโดยชี้ไปที่ gateway ที่ไม่มีอะไรฟังอยู่ (`127.0.0.1:9`) อะไรที่ CLI ตรวจเองได้จะล้มทันที ส่วนอะไรที่ผ่านการตรวจเหล่านั้นไปแล้วจะล้มด้วย `Connection refused` ซึ่งพิสูจน์ว่ามัน parse ผ่าน NemoClaw ตรึงเวอร์ชัน **0.0.116** ไว้บน Spark และรุ่นล่าสุดคือ **0.1.2** ถ้า flag บนเครื่องคุณต่างไป ให้เชื่อ `--help`

> 📌 **policykit เป็นแบบจำลองเพื่อการสอน** `week26/common/policykit.py` ทำตามกฎที่เอกสารอธิบาย เพื่อให้คุณคิดเรื่อง policy ได้ก่อน push มันไม่ใช่ OpenShell ตัวที่บังคับใช้จริงคือ Landlock, seccomp และ proxy บน Spark จุดไหนที่ policykit กับ CLI ตัวจริงเห็นไม่ตรงกัน โมดูลนี้จะแสดงให้ดูทั้งสองฝั่ง

✓ Checkpoint: `openshell --version` บนแล็ปท็อปพิมพ์ 0.0.111 และคุณรู้ว่าแล็บจะใช้ชื่อ sandbox อะไร

## 1 · OpenShell บังคับใช้ policy อย่างไร

OpenShell มี **จุดบังคับใช้สองจุด** ส่วน static ถูกล็อกตั้งแต่ตอนสร้าง sandbox ได้แก่ filesystem (Landlock LSM) และ process (seccomp BPF และการลดสิทธิ์) ส่วน dynamic เปลี่ยนได้บน sandbox ที่กำลังรันด้วย `openshell policy update` หรือ `openshell policy set` ได้แก่ network (CONNECT proxy กับ OPA policy engine) และ credential ของ provider (proxy เป็นคนสลับใส่ให้)

กฎห้าข้อตัดสินว่าคำขอหนึ่งทำอะไรได้:

| กฎ | ความหมายต่อคุณ |
|---|---|
| **Deny by default และผูกกับ binary** | การเชื่อมต่อต้องตรงกับรายการใน `network_policies` ทั้ง host, port **และ** binary ที่เรียก ทุกรายการต้องมีรายการ `binaries` OpenShell จะตรึงแต่ละ binary ไว้กับ SHA256 ที่เห็นครั้งแรก (trust on first use) และปิดทันทีถ้าไม่ตรง |
| **Network namespace ไม่ใช่ env var** | sandbox มี netns ของตัวเอง ทราฟฟิกทั้งหมดวิ่งไปที่ proxy ที่ `10.200.0.1` โปรเซสที่ไม่สนใจ `HTTP_PROXY` ก็ยังไปถึงได้แค่ proxy |
| **L4 กับ L7** | endpoint ที่ไม่มี `protocol` ถูกตรวจแค่ host, port และ binary ส่วน `protocol: rest\|websocket\|graphql\|mcp\|json-rpc\|tcp` จะเปิดการตรวจระดับคำขอ ใช้คู่กับ `rules` / `deny_rules` หรือ `access` preset (`full`, `read-only`, `read-write`) |
| **audit กับ enforce** | `enforcement` ค่าเริ่มต้นคือ `audit`: บันทึกการละเมิดแต่ยังส่งทราฟฟิกต่อ ส่วน `enforce` จะตอบ 403 พร้อม body แบบ JSON |
| **ป้องกัน SSRF** | `127.0.0.0/8`, `169.254.0.0/16` และ `0.0.0.0` ถูกบล็อกเสมอ แม้ใส่ `allowed_ips` ก็ตาม ส่วนช่วง private RFC 1918 ต้องระบุ host ตรงตัว หรือเปิดด้วย `allowed_ips` แบบ CIDR แคบ ๆ |

baseline ของ NemoClaw OpenClaw ให้สิทธิ์อ่านเขียน `/sandbox`, `/tmp`, `/dev/null`, `/dev/pts` ให้อ่านอย่างเดียว `/usr`, `/lib`, `/proc`, `/dev/urandom`, `/app`, `/etc`, `/var/log`, `/var/lib/dpkg` และรันเอเจนต์ด้วยผู้ใช้ `sandbox` โดยเฉพาะ Landlock ต้องการ ABI 3 (Linux 6.2 ขึ้นไป) ค่าเริ่มต้น `compatibility: best_effort` จะออก `DetectionFinding` ระดับ High เมื่อกฎข้อไหนใช้ไม่ได้ ส่วน `hard_requirement` จะไม่ยอมเริ่มเลย

✓ Checkpoint: อธิบายได้ว่าทำไม `curl` ใน sandbox ใช้ endpoint ที่ระบุแค่ `/usr/bin/gh` ไม่ได้ และทำไมการ unset `HTTP_PROXY` ไม่ช่วย

## 2 · L2.1 — อ่าน policy ที่ NemoClaw สร้างให้คุณ

เริ่มจากการอ่าน ยังไม่ต้องเขียน มีสามมุมมอง และมีแค่แบบเดียวที่แก้แล้ว push กลับได้อย่างปลอดภัย

```bash
# on: spark
nemoclaw my-assistant policy list
nemoclaw my-assistant policy get > current-policy.yaml
openshell policy get my-assistant --base > base.yaml
openshell policy get my-assistant --full
openshell policy list my-assistant
```

| คำสั่ง | สิ่งที่ได้ |
|---|---|
| `nemoclaw <s> policy get` | policy ที่ตัด metadata ออกแล้ว credential ตัวจริงถูกแทนด้วย `[STRIPPED_BY_MIGRATION]` (ต้องใช้ OpenShell 0.0.72 ขึ้นไป) **นี่คือไฟล์ที่คุณแก้** |
| `openshell policy get <s> --base` | base policy ของ OpenShell ไม่รวมรายการที่ provider ประกอบเข้ามา |
| `openshell policy get <s> --full` | policy ที่มีผลจริง รวมรายการจาก provider โดยมี metadata header นำหน้า |
| `openshell policy list <s>` | ประวัติ revision |

คุณควรจำรายการ baseline หกตัวได้: `nvidia` (`integrate.api.nvidia.com:443`, binary `/usr/local/bin/openclaw`, POST ไปที่ path ของ inference และ embedding, GET รายชื่อโมเดล), `clawhub`, `openclaw_api`, `openclaw_docs`, `npm_registry` (GET อย่างเดียว เฉพาะ binary `openclaw`) และ route `managed_inference` ที่ขาดไม่ได้ ทั้งหมด terminate TLS บนพอร์ต 443

> ⚠ **สะกดสองแบบ** research tutorial (อ้างอิงเอกสาร NemoClaw) เขียน `nemoclaw <s> policy list` / `policy add` ส่วน DGX Spark playbook ในเครื่อง (`playbook-nemoclaw`, `playbook-nemoclaw-applications`) เขียน `policy-list` / `policy-add` / `policy-remove` แล็บ 03-1 จะลองแบบหนึ่ง แล้วค่อยลองอีกแบบ ให้รัน `nemoclaw <s> --help` บนเครื่องคุณแล้วใช้ตามที่มันแสดง

กับดักที่ OpenShell playbook เตือนไว้: `--full` จะใส่ metadata header ที่มีฟิลด์ `Version` นำหน้า และ `policy set` ไม่รับไฟล์แบบนั้น แล็บ 03-1 ทำให้เห็นกับ parser ตัวจริง:

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_1_read_policy.py
```

**Expected output** (captured on this Mac, DRY mode — step 5) [บันทึกจาก Mac เครื่องนี้ในโหมด DRY — ขั้นที่ 5]

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

ถ้าไม่มี Spark ขั้นที่ 1–4 จะพิมพ์รูปแบบ EXAMPLE และตาราง baseline จะขึ้น `◈ DRY — not checked` ทุกแถว ถ้าต่อ Spark จริง แล็บจะคัดลอก export ของคุณมาไว้ที่ `03_policy_as_code/.runs/current-policy.from-spark.yaml` แล้วตรวจด้วย policykit และ parser ด้วย

✓ Checkpoint: คุณรู้ว่าจะ push export ตัวไหนกลับ (`nemoclaw <s> policy get` หรือ `openshell policy get` แบบไม่มี `--full`) และเคยเห็น error เรื่อง `Version` มาแล้วหนึ่งครั้ง

## 3 · L2.2 — กายวิภาคของไฟล์ policy

`week26/03_policy_as_code/policies/anatomy.yaml` คือ schema policy พร้อมคำอธิบายจาก research tutorial:

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

รูปแบบกฎที่คุณจะใช้ หนึ่งแบบต่อหนึ่ง protocol ไฟล์ `policies/rule_forms.yaml` รวมทั้งห้าแบบไว้ใน policy เดียวที่ parse ผ่าน:

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

คุณถาม parser ตัวจริงเองก็ได้ ชี้มันไปที่ gateway ที่ไม่มีอยู่ แล้วอ่านบรรทัดสุดท้าย:

```bash
# on: laptop
cd week26 && OPENSHELL_GATEWAY_ENDPOINT=http://127.0.0.1:9 HOME=.runs/openshell-home \
  .venv-openshell/bin/openshell policy set parse-probe --policy 03_policy_as_code/policies/anatomy.yaml
```

**Expected output** (captured on this Mac) [บันทึกจาก Mac เครื่องนี้]

```
Error:   × transport error
  ├─▶ tcp connect error
  ├─▶ tcp connect error
  ╰─▶ Connection refused (os error 61)
```

ในกรณีนี้ `Connection refused` คือคำตอบที่ดี: YAML parse ผ่าน ขาดแค่ gateway แล็บ 03-2 ทำแบบนี้กับ schema policy, rule forms และรุ่นที่พังสิบสี่แบบ แล้วถาม policykit ว่ากฎเหล่านี้จะอนุญาตอะไร:

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_2_policy_anatomy.py
```

**Expected output** (captured on this Mac — steps 3 and 4, command echo lines trimmed) [บันทึกจาก Mac เครื่องนี้ — ขั้นที่ 3 และ 4 ตัดบรรทัดคำสั่งออก]

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

อ่านสองคอลัมน์ไปด้วยกัน:

- **parser จับโครงสร้าง**: ฟิลด์ที่ไม่รู้จัก, list ในที่ที่ควรเป็น map, string ในที่ที่ควรเป็นเลขพอร์ต และการห่อ `deny:` ไว้ใน `deny_rules`
- **policykit จับความหมาย**: `read_write: [/]`, root, ไม่มีพอร์ต, `access` สะกดผิด, REST endpoint ที่ไม่มีทั้ง access และ rules พวกนี้ผ่าน parser ได้ ดังนั้นบน Spark ตัวที่ปฏิเสธคือ gateway: ตาราง troubleshooting ของ OpenShell playbook ระบุ root เป็น `run_as_user` และการขาด `host`/`port` ว่าเป็นสาเหตุที่ push ไม่ผ่าน ส่วน research tutorial ระบุ `INVALID_ARGUMENT` สำหรับ `read_write: [/]` สำหรับ `access` ที่สะกดผิดและ `landlock: strict` ยังไม่มีแหล่งไหนแสดงข้อความของ gateway ให้ตรวจบนเครื่องของคุณเอง
- **policykit ไม่ครบ**: มันพลาดความผิดพลาดด้านรูปทรงของกฎที่ parser จับได้ และมันปฏิเสธ `protocol: sql` ที่ CLI 0.0.111 รู้จัก

✓ Checkpoint: สำหรับ `read_write: [/]`, `Version:` และ `deny_rules: [{deny: …}]` คุณบอกได้ว่าใครจับได้: parser บนแล็ปท็อป, policykit หรือ gateway

## 4 · L2.3 — Iterate loop: deny → observe → allow → verify

ขั้นตอนตามเอกสาร: สร้างด้วย policy ตั้งต้น เฝ้าดูการ deny ดึง แก้ push แล้วตรวจยืนยัน บน claw ที่กำลังรัน นั่นหมายถึงการเปลี่ยนแบบเพิ่มทีละน้อย และดูตัวอย่างทุกครั้งก่อน

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

มีไวยากรณ์สองแบบที่ทำงานนี้ endpoint spec คือ `host:port[:access[:protocol[:enforcement[:options]]]]` ส่วน rule spec คือ `host:port:METHOD:path_glob` จำกับดักไว้: `api.github.com:443::rest` ถูกปฏิเสธ endpoint L7 ที่มี protocol แต่ไม่มี access หรือ rules ไม่ได้แปลว่า "อนุญาตทั้งหมด"

แล็บ 03-3 ป้อนไวยากรณ์ทั้งสองให้ CLI ตัวจริงและ policykit แล้วรันลูปข้างบนบน Spark ของคุณผ่าน `change()`: การเปลี่ยนทุกครั้งจะรันคู่แฝด `--dry-run` เป็นตัวอย่างก่อน และจะรันจริงเฉพาะในโหมด LIVE ที่เปิด 🔓 Allow changes ไว้

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_3_iterate_loop.py
```

**Expected output** (captured on this Mac, DRY mode — steps 1 and 3, command echo lines trimmed) [บันทึกจาก Mac เครื่องนี้ในโหมด DRY — ขั้นที่ 1 และ 3 ตัดบรรทัดคำสั่งออก]

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

สิ่งที่แล็ปท็อปสอนคุณ:

- `::rest` **parse ผ่าน** ใน CLI research tutorial ซึ่งอ้างอิงคู่มือ OpenShell sandbox-policies บอกว่ามันถูกปฏิเสธ ดังนั้นการปฏิเสธมาจาก gateway ตอน merge ส่วน policykit ปฏิเสธมันตั้งแต่แรกโดยตั้งใจ
- ใน 0.0.111 `--add-endpoint` รับ protocol แค่ `tcp`, `rest`, `websocket` และ `sql` endpoint แบบ MCP, GraphQL และ JSON-RPC ต้องเขียนเป็น YAML (`policy set`) ไม่ใช่ spec string
- `tcp` ไม่รับ access mode ดังนั้น `timescale.alto.local:5432::tcp` ของแบบฝึกหัดข้อ 4 ใน Part 2 ต้องเว้นช่อง access **ว่างไว้**
- `--dry-run` ต้องใช้ gateway: มันดึง policy ที่ใช้อยู่มาแสดงผลการ merge โดยไม่ส่งอะไรออกไป และ CLI ไม่ยอมให้ใช้ `--dry-run` คู่กับ `--wait`

สำหรับ preset ให้ใช้ตัวห่อของ NemoClaw ดีกว่า มันรู้จัก baseline ของ blueprint เช่น มันจะไม่ยอมเปลี่ยน `npm` ถ้า baseline ที่ใช้อยู่เบี่ยงไปจากรายการ GET-only ที่ผ่านการรีวิวแล้ว

```bash
# on: spark
nemoclaw my-assistant policy add github --dry-run
nemoclaw my-assistant policy add github --yes
nemoclaw my-assistant policy remove github --yes
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml     # custom preset
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml --trusted-private-host 10.20.0.15 --dry-run
```

✓ Checkpoint: คุณเขียน `--add-endpoint` spec สำหรับ REST API แบบอ่านอย่างเดียวที่ enforce แล้ว และสำหรับ Postgres บนพอร์ต 5432 ได้ และรู้ว่าแบบไหนที่ gateway (ไม่ใช่แล็ปท็อป) ยังต้องยอมรับอีกชั้น

## 5 · L2.4 — การอนุมัติโดยผู้ดูแลใน TUI

เมื่อเอเจนต์เรียก endpoint ที่ไม่อยู่ในรายการ OpenShell จะบล็อกไว้และแสดงคำขอนั้นใน `openshell term` ถ้าคุณอนุมัติ การอนุมัตินั้นจะกลายเป็น policy revision ใหม่ที่คงอยู่ถาวร มันอยู่รอดข้ามการรีสตาร์ทของ sandbox instance เดิม และหายไปเมื่อ sandbox ถูกทำลายแล้วสร้างใหม่ ก่อนอนุมัติ prover ของ OpenShell จะติดธงการเข้าถึงใหม่ที่เสี่ยง (host ใหม่ที่มี credential, API method ใหม่) แล้วรอให้คนตัดสิน

ลองทำโดยตั้งใจ `openshell term` เป็น TUI แบบโต้ตอบ จึงต้องรันใน ⌨ terminal ห้ามรันจากแล็บ:

```bash
# on: spark
openshell term
```

จากนั้นขอให้ผู้ช่วย *"fetch https://httpbin.org/get and show me the headers"* อนุมัติหนึ่งครั้ง แล้วดู revision ใหม่:

```bash
# on: spark
openshell policy list my-assistant
```

CLI 0.0.111 บอกว่าขั้นตอนการอนุมัติตั้งค่าได้ที่ไหน แล็บ 03-4 พิมพ์ส่วนนั้นออกมา แล้วจำลองวงจรชีวิตด้วย policykit:

**Expected output** (captured on this Mac — lab 03-4, step 6) [บันทึกจาก Mac เครื่องนี้ — แล็บ 03-4 ขั้นที่ 6]

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

รายการที่ถูกอนุมัติในตารางเป็น **การจำลอง** TUI จะเขียนรายการของมันเอง ให้อ่านของจริงด้วย `openshell policy get`

✓ Checkpoint: คุณบอกได้ว่าการอนุมัติใน TUI เป็นอย่างไรหลังรีสตาร์ท (ยังอยู่) และหลังทำลายแล้วสร้างใหม่ (หายไป)

## 6 · L2.5 — Preset และ posture profile

preset ที่ดูแลอยู่เก็บใน `nemoclaw-blueprint/policies/presets/`: `brave`, `brew`, `claude-code`, `discord`, `github`, `gmail`, `googlechat`, `huggingface`, `jira`, `local-inference`, `npm`, `nous-*` (Hermes), `openclaw-pricing`, `outlook`, `public-reference`, `pypi`, `slack`, `tavily`, `teams`, `telegram`, `weather`, `wechat`, `whatsapp`

อ่านความเสี่ยงก่อนใช้:

| Preset | ความเสี่ยง (NemoClaw security best practices ผ่าน research tutorial) |
|---|---|
| `pypi` | GET/HEAD อย่างเดียว แต่เปิดให้เอเจนต์ติดตั้งแพ็กเกจอะไรก็ได้ |
| `github` | อ่านเขียน repo ได้ ผ่าน `git` เท่านั้น (ผูกกับ binary `/usr/bin/git`) |
| `slack`, `discord` | ช่วงที่เป็น WebSocket ใช้ `access: full` โดยไม่มีการตรวจ |
| `personal-open-internet` | ยกเลิกข้อจำกัดด้าน hostname, method, path และ body บนพอร์ต 80/443 |
| `whatsapp`, `brew` | ส่งผ่านตรงด้วย `access: full` + `tls: skip` (ตาม NemoClaw applications playbook) |

posture profile สี่แบบจากคู่มือเดียวกัน Module 07 จะจับคู่กับงาน deploy ของ AltoTech

| Profile | Tier | Presets | Inference | หมายเหตุ |
|---|---|---|---|---|
| Locked-Down | Restricted | ไม่มี (ไม่มี web search) | NVIDIA Endpoints หรือ Ollama ในเครื่อง | อย่างอื่นต้องให้ผู้ดูแลอนุมัติ; เฝ้าดู TUI |
| Development | Balanced | `pypi`, `npm` | อะไรก็ได้ | คงข้อจำกัดด้าน binary ไว้; รีวิวด้วย `openshell term` |
| Personal | Personal | `personal-open-internet` | อะไรก็ได้ | ผู้ใช้คนเดียวที่ไว้ใจได้เท่านั้น; ใช้เสร็จแล้วสร้างใหม่เป็น Balanced |
| Integration Testing | custom | รายการ method/path ที่แคบ, `protocol: rest` | อะไรก็ได้ | เก็บกวาด baseline หลังทดสอบ |

แล็บ 03-4 อ่าน `nemoclaw <s> policy list` (อ่านอย่างเดียว) เดา profile ที่ใกล้ที่สุดจากผลลัพธ์จริงเท่านั้น และเพิ่ม `pypi` ผ่าน `change()` โดยดูตัวอย่างด้วย `--dry-run` ก่อน

✓ Checkpoint: เลือก profile ให้ (a) claw ผู้ช่วยต้อนรับของโรงแรมบน Spark ที่ใช้ร่วมกัน (b) claw สำหรับพัฒนาของคุณเองที่ต้อง `pip install` และบอกได้ว่าทำไม Personal ไม่เคยถูกต้องบนฮาร์ดแวร์ที่ใช้ร่วมกัน

## 7 · L2.6 — Snapshot, rebuild และการกู้คืน

```bash
# on: spark
nemoclaw my-assistant snapshot create --name before-change
# ...make changes...
nemoclaw my-assistant rebuild
```

Deep Agents quickstart แสดงคำสั่งเหล่านี้สำหรับ `nemo-deepagents` และใช้ได้แบบเดียวกันกับ CLI ตัวอื่น แล็บ 03-4 สร้าง snapshot ผ่าน `change()` และพิมพ์คำสั่ง rebuild ไว้ให้คุณรันเองเมื่อพร้อมจริง ๆ

กฎเมื่อสงสัยว่าถูกเจาะนั้นง่าย: **สร้าง sandbox ใหม่จาก input ที่เชื่อถือได้ อย่าพยายามทำความสะอาด** เอเจนต์เขียนทับ config ของตัวเองได้ sandbox ที่ทำความสะอาดแล้วจึงไม่ใช่ sandbox ที่เชื่อถือได้ นี่ตอบคำถามที่พบบ่อยด้วย: วิธีที่เร็วที่สุดที่จะรับประกันว่าการอนุมัติสามรายการตอนดีบักหายไปหมด คือทำลาย sandbox แล้วสร้างใหม่

✓ Checkpoint: คุณทำ (หรือทำได้) snapshot ชื่อ `before-change` และอธิบายได้ว่าทำไม "สร้างใหม่" ดีกว่า "ทำความสะอาด"

## 8 · L2.7 — OpenShell แบบไม่มี NemoClaw: ใช้ vLLM ของคุณเอง

นี่คือ OpenShell playbook ฉบับย่อ มันแสดงว่า `nemoclaw onboard` ทำอะไรอยู่เบื้องหลัง และให้คุณคุม model server ได้เต็มที่

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

`MODEL_HANDLE` คือโมเดลสำหรับ DGX Spark ที่ OpenShell playbook แนะนำ จะใช้ handle ตัวไหนจาก vLLM recipes สำหรับ DGX Spark ก็ได้ ใน OpenClaw wizard ให้เลือก **Custom Provider**, base URL `https://inference.local/v1`, key อะไรก็ได้ที่ไม่ว่าง (`not-needed`), **OpenAI-compatible** และ model id ตัวเดียวกัน

หลังขั้นที่ 4 playbook บอกว่าต้องดูอะไร:

**Expected output** (REFERENCE — quoted from the DGX Spark OpenShell playbook) [อ้างอิงคำต่อคำจาก DGX Spark OpenShell playbook]

```
Expected output should show `provider: local-vllm` and your chosen `model`.
```

กับดักสองข้อจาก playbook:

1. **ใช้ LAN IP ไม่ใช่ localhost** gateway รันอยู่ใน Docker ภายใน container ของมัน `127.0.0.1` คือตัว container เอง จึงต้อง bind vLLM ที่ `0.0.0.0` และให้ IP ของเครื่องกับ provider (หรือ `host.docker.internal` ถ้า resolve ได้)
2. **ห้ามใส่ `--policy` คู่กับ `--from openclaw`** community sandbox มี policy มาในตัว path ไฟล์ในเครื่องอาจทำให้เกิด "file not found" parser บนแล็ปท็อป **ไม่** จับกรณีนี้ แล็บ 03-5 จึงมีตัวกันไว้

แล็บ 03-5 รันการตรวจแบบอ่านอย่างเดียว ให้ทุกการเปลี่ยนผ่าน `change()` และไม่รันตัวติดตั้งหรือ `sandbox create` แบบโต้ตอบให้คุณ:

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/labs/lab03_5_byo_vllm.py
```

**Expected output** (captured on this Mac, DRY mode — step 3 and the parser table of step 5) [บันทึกจาก Mac เครื่องนี้ในโหมด DRY — ขั้นที่ 3 และตาราง parser ของขั้นที่ 5]

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

ตารางที่สองในขั้นที่ 3 คือหัวใจของแล็บนี้ **gateway** ต้องใช้ LAN IP เพื่อไปถึง vLLM ส่วน **เอเจนต์** ไม่เคยเรียก vLLM ตรง ๆ เลย: มันเรียก `inference.local` แล้ว gateway ส่งต่อพร้อม credential ดังนั้นไม่ต้องมีรายการ `network_policies` สำหรับ model server ของคุณ

sandbox แบบอื่นที่ควรรู้: `--from base` (Ubuntu ขั้นต่ำ ไม่มีเอเจนต์), `--from sdg`, `--from ./dir` หรือ Dockerfile; `--gpu`, `--cpu 2 --memory 4Gi`, `--upload`, `--env`, `--label`; `openshell sandbox exec -n <name> -- <cmd>` สำหรับคำสั่งครั้งเดียว คำสั่งท้ายสุดใน `create` คือ main process ที่กำหนดสุขภาพของ sandbox ถ้ามันจบ sandbox จะเข้าสถานะ `Error` ส่วน `OPENSHELL_SANDBOX_POLICY=./my-policy.yaml` ช่วยให้ไม่ต้องพิมพ์ `--policy` ทุกครั้ง

✓ Checkpoint: คุณอธิบายได้ว่าทำไม `OPENAI_BASE_URL=http://localhost:8000/v1` ใช้กับ provider ไม่ได้ และทำไม policy ของเอเจนต์ไม่ต้องมีรายการสำหรับ vLLM

## Labs — รันแล็บได้ที่นี่

**labs/lab03_1_read_policy.py** — อ่าน policy ที่ NemoClaw สร้าง (list, export, base, full, revision) และทำให้เห็นกับดัก `--full` → `policy set` ด้วย parser ตัวจริง

**labs/lab03_2_policy_anatomy.py** — schema พร้อมคำอธิบาย กฎทุกรูปแบบ และรุ่นที่พังสิบสี่แบบ ตรวจด้วย policykit และ OpenShell parser ตัวจริงเทียบกัน

**labs/lab03_3_iterate_loop.py** — ไวยากรณ์ endpoint spec และ rule spec บน CLI ตัวจริง แล้วเดินลูป deny → allow → verify บน Spark ของคุณ โดยดูตัวอย่างด้วย --dry-run ก่อนการเปลี่ยนทุกครั้ง

**labs/lab03_4_presets_posture.py** — preset และความเสี่ยง, posture profile, การเพิ่ม preset และทำ snapshot ผ่าน change() และจำลองวงจรชีวิตของการอนุมัติใน TUI

**labs/lab03_5_byo_vllm.py** — OpenShell ล้วนกับ vLLM ของคุณเอง: ตรวจแบบอ่านอย่างเดียว ทุกการเปลี่ยนผ่านประตู คำนวณกฎ LAN IP และตัวกัน --policy คู่กับ --from openclaw

## Try it yourself — ลองทำเอง

`exercises/ex03_alto_bms_preset.py` คือแบบฝึกหัดข้อ 1 ของ Part 2 Alto Ops Claw ต้องอ่านค่า point และเขียน setpoint ไปที่ BMS ของโรงแรมที่ `bms.alto.local:8443` ซึ่ง resolve เป็น `10.20.0.15` มี TODO สี่ข้อ:

1. endpoint: พอร์ต, `protocol: rest`, `enforcement: enforce` และ `allowed_ips` ตรึงไว้ที่ `10.20.0.15/32`
2. กฎ: อนุญาต `GET /api/v1/points/**` และ `POST /api/v1/setpoints/*` ห้าม `POST /api/v1/admin/**`
3. binary: `/usr/bin/python3` เท่านั้น
4. ชื่อ preset ที่ NemoClaw ยอมรับ: label แบบ RFC 1123 ตัวพิมพ์เล็กคั่นด้วยขีด

```bash
# on: laptop
.venv/bin/python week26/03_policy_as_code/exercises/ex03_alto_bms_preset.py
```

**Expected output** (captured on this Mac, all TODOs filled) [บันทึกจาก Mac เครื่องนี้ เมื่อเติมทุก TODO แล้ว]

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

ตัวตรวจเขียนไฟล์สองไฟล์ไว้ที่ `03_policy_as_code/.runs/` ไฟล์ `alto-bms.yaml` เป็น OpenShell policy ที่ parser ตัวจริงยอมรับ ส่วน `alto-bms.preset.yaml` เพิ่ม header `preset:` ตามที่ NemoClaw applications playbook แสดงไว้สำหรับ `policy-add --from-file` — และบรรทัดสุดท้ายพิสูจน์ว่า `openshell policy set` จะไม่รับไฟล์นี้ เครื่องมือสองตัว รูปทรงไฟล์สองแบบ

จากนั้นดูตัวอย่างบน Spark ก่อนใช้จริง:

```bash
# on: spark
nemoclaw my-assistant policy add --from-file ./alto-bms.yaml --trusted-private-host 10.20.0.15 --dry-run
```

> ⚠ research tutorial ใส่ `--trusted-private-host 10.20.0.15` ในหัวข้อ 2.4 แต่ใส่ `--trusted-private-host bms.alto.local` ในเฉลยแบบฝึกหัด ให้ดู `nemoclaw <s> policy add --help` บนเครื่องคุณว่ามันต้องการแบบไหน รีวิว address pin ที่ dry run พิมพ์ออกมา แล้วค่อยรันซ้ำด้วย `--yes`

<details><summary>คำใบ้ — allow ห่อไว้ ส่วน deny เขียนเรียงเลย</summary>

กฎ allow เขียนเป็น `- allow: { method: GET, path: "/api/v1/points/**" }` ส่วนกฎ deny ไม่มีตัวห่อ: `- { method: POST, path: "/api/v1/admin/**" }` ใส่เครื่องหมายคำพูดครอบ path ที่มี `*` แล็บ 03-2 แสดงแล้วว่า parser ว่าอย่างไรกับตัวห่อ `deny:`

</details>

<details><summary>คำใบ้ — ทำไมต้อง /32</summary>

ที่อยู่ private แบบ RFC 1918 ถูกบล็อก เว้นแต่คุณระบุ host ตรงตัวหรือเปิด `allowed_ips` แบบ CIDR แคบ ๆ ตัวตรวจจะลอง `10.20.0.16` ด้วย ซึ่งต้องถูกปฏิเสธต่อไป ถ้าใช้ `/24` เอเจนต์จะเข้าถึงทุกอุปกรณ์บน VLAN ของ BMS ได้

</details>

<details><summary>คำถามอื่นของ Part 2 (มีคำตอบข้างใน)</summary>

- **ทำไม `audit` เป็นค่าเริ่มต้น?** มันบันทึกการละเมิดแต่ยังส่งทราฟฟิกต่อ คุณจึงได้เรียนรู้รูปแบบการเข้าถึงจริง สลับเป็น `enforce` เมื่อกฎผ่านการตรวจแล้ว หลังจากนั้นคำขอที่ไม่ตรงกฎจะได้ `403 Forbidden` พร้อม body แบบ JSON
- **เพื่อนร่วมงานอยากได้ `read_write: [/]`** จะถูกปฏิเสธด้วย `INVALID_ARGUMENT` ให้เพิ่มไดเรกทอรีที่เขียนได้แบบเจาะจง เช่น `/sandbox/tools` และติดตั้งเครื่องมือตอน build image (`nemoclaw onboard --from <Dockerfile>`)
- **`psql` ไปที่ `timescale.alto.local:5432`?** YAML: `endpoints: [{host: timescale.alto.local, port: 5432, protocol: tcp}]` พร้อม `binaries: [{path: /usr/bin/psql}]` CLI: `--add-endpoint timescale.alto.local:5432::tcp --binary /usr/bin/psql` — ช่อง access ต้องว่าง และแล็บ 03-3 แสดงแล้วว่า 0.0.111 ไม่รับ access mode บน `tcp`
- **`tls: skip` บน endpoint ที่มี credential?** มันปิดการเขียน credential placeholder ใหม่, การฉีด token และการตรวจ L7 proxy จะส่งต่อข้อมูลที่เข้ารหัสแบบมองไม่เห็น endpoint ที่มี credential ของ provider ต้องมี `allow_uninspected_credentials: true` เพิ่มด้วย เพื่อยอมรับอย่างชัดแจ้ง

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ทุกบรรทัด และคุณบอกได้ว่าไฟล์ไหนจะให้ `nemoclaw … policy add --from-file` และไฟล์ไหนจะให้ `openshell policy set`

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `policy set` ล้มด้วย `unknown field 'Version'` | คุณ push ไฟล์ที่ dump จาก `--full` ซึ่ง metadata header มีฟิลด์ `Version` | export ด้วย `openshell policy get <s>` (ไม่มี `--full`) หรือ `nemoclaw <s> policy get` หรือตัดทุกบรรทัดก่อน `---` ตัวแรกออก |
| `invalid type: sequence, expected a map` | `network_policies` เป็น list ของ `{host, port}` | ทำให้เป็น map ที่ใช้ชื่อกลุ่มเป็น key แต่ละกลุ่มมี `endpoints` และ `binaries` |
| policy "ใช้ได้ไม่มี error" แต่ทุกการเรียกได้ 403 | ไม่มี `binaries` หรือ endpoint ไม่มี access mode / rules | เพิ่ม binary ที่เรียก และ `access:` หรือ `rules:` (สองกลุ่มที่ "parser ไม่จับ" ในแล็บ 03-2) |
| `--add-endpoint protocol segment must be 'tcp', 'rest', 'websocket', or 'sql'` | เขียน endpoint แบบ MCP / GraphQL / JSON-RPC เป็น spec | เขียนเป็น YAML แล้ว push ด้วย `openshell policy set` |
| `--wait cannot be combined with --dry-run` | การดูตัวอย่างไม่ส่งอะไรออกไปเลย | ดูตัวอย่างด้วย `--dry-run` แล้วรันใหม่ด้วย `--wait` |
| `nemoclaw … policy list` บอกว่าไม่รู้จักคำสั่ง | NemoClaw ของคุณใช้คำสั่งแบบมีขีด | `nemoclaw <s> policy-list` / `policy-add` / `policy-remove` (การสะกดแบบ DGX Spark playbook) |
| `Preset must declare preset.name (lowercase, hyphenated RFC 1123 label)` | `preset.name` มีขีดล่างหรือตัวพิมพ์ใหญ่ | ใช้ตัวอักษร ตัวเลข และขีด: `alto-bms` ไม่ใช่ `alto_bms` |
| `failed to verify inference endpoint` หลัง `inference set` | vLLM ยังโหลดไม่เสร็จ หรือ provider URL ใช้ localhost | อุ่นเครื่องด้วย chat completion หนึ่งครั้ง ใช้ LAN IP จาก `hostname -I` ใช้ `--no-verify` เฉพาะหลังยืนยันว่า API บน host ใช้ได้แล้ว |
| "Permission denied" / error ของ Landlock ใน sandbox | path ไม่อยู่ใน `read_only` หรือ `read_write` | filesystem policy เป็นแบบ static: เพิ่ม path แล้ว **สร้าง** sandbox ใหม่ |

## Next — บทถัดไป

[Lab 04 — สร้าง claw ด้วย NeMo Agent Toolkit](../04_nat_claws/TUTORIAL.md): เขียน Alto Ops Claw เป็น NAT workflow ที่มีเครื่องมือวิเคราะห์ chiller plant ให้บริการผ่าน REST และ MCP และรันมันใน sandbox แบบที่คุณเพิ่งเรียนเขียน policy ให้
