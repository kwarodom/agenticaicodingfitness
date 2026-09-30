# ▶ Reef Lab 07 — ระดับ Expert: threat model, hardening, custom blueprint และ remote gateway

> ส่วนหนึ่งของ Week 26 · NemoClaw บน DGX Spark ตั้งแต่ระดับเริ่มต้นจนถึง expert คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง แล็บฝั่ง laptop (NAT CLI, OpenShell CLI และ policy model) รันจริงได้ทุกที่ แล็บฝั่ง Spark รันในโหมด **DRY** ได้ด้วย (ไม่มี Spark, $0) โดยแสดงคำสั่ง ส่วนผลลัพธ์จะเป็นอย่างใดอย่างหนึ่ง: RECORDED จาก Spark จริง, REFERENCE ที่ยกมาจากเอกสารหรือ playbook ของ NVIDIA หรือ EXAMPLE ที่ติดป้ายไว้ชัดเจน

**What you'll actually do** (สิ่งที่คุณจะได้ลงมือทำ)
- ลองโจมตี claw ของตัวเองบนกระดาษ: agent ที่โดน prompt injection ลอง 12 ขั้นตอน แล้วดูผลตัดสินเทียบกันระหว่าง policy ฉบับร่างกับ policy สำหรับ production
- ไล่ hardening checklist แบบ Alto sovereign ทีละบรรทัด และแยกให้ออกว่าบรรทัดไหน lint ตรวจได้ บรรทัดไหนตรวจไม่ได้เลย
- ส่ง production policy ของ research tutorial ผ่านตัวตรวจ 3 ตัว (OpenShell CLI parser ตัวจริง, `policykit.validate`, `policykit.harden`) แล้วจงใจทำให้พัง 12 แบบ เพื่อดูว่าตัวตรวจไหนจับอะไรได้
- เทียบ policy กับรายชื่อ tool **จริง** ของ mock BMS แล้ว apply บน Spark โดยดู preview ก่อน
- สร้าง custom sandbox image, ตรวจ `blueprint.yaml` สำหรับ external gateway (พร้อมสำเนาที่พังอีก 11 ชุด) และดูว่าคำสั่ง remote gateway ใน tutorial ต้องแก้ตรงไหน
- ประกอบชุดหลักฐานว่า "ไม่มีข้อมูลออกจากตึกเลยในเดือนที่แล้ว" และแก้ rule ของ approvals ที่พังใน exercise

**Time** ~90 นาที · **Difficulty** expert · **Hardware** ไม่ต้องมี (DRY + laptop) · DGX Spark 1 เครื่อง (ไม่บังคับ) · Spark เครื่องที่สอง (ไม่บังคับ, สำหรับ remote gateway)

**Sources:** research tutorial ของคอร์ส *NemoClaw on DGX Spark — Beginner to Expert* (Part 6: §6.1–6.6, Lab 6.1–6.3 และ exercise ของ Part 6) ซึ่งอ้างอิง [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema) · [NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [OpenShell on GitHub](https://github.com/NVIDIA/openshell) · [CSA — OpenClaw indirect prompt injection](https://labs.cloudsecurityalliance.org/research/csa-research-note-openclaw-indirect-prompt-injection-2026061/)

## 0 · ก่อนเริ่ม

| ต้องมี | วิธีเช็ก | ใช้ทำอะไร |
|---|---|---|
| Module 03 และ 04 | อ่าน policy เป็น และรู้จัก Alto Ops Claw แล้ว | module นี้ harden ทั้งสองอย่าง |
| OpenShell CLI บน laptop | `week26/.venv-openshell/bin/openshell --version` → 0.0.111 | parser ตัวจริงของ policy (ไม่มี gateway อยู่ด้านหลัง) |
| NAT venv | `week26/.venv-nat/bin/python -c "import mcp"` | ใช้รัน mock BMS ใน lab 07-2 |
| `openssl` | `openssl version` | lab 07-3 ใช้สร้าง CA ทิ้งขว้างสำหรับ blueprint |
| DGX Spark (ไม่บังคับ) | `ssh -o BatchMode=yes <spark> true` | apply policy, รัน blueprint runner, เก็บหลักฐานจริง |

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

> 📌 **สองสิ่งที่ module นี้ไม่ใช่** `policykit` (`decide`, `validate`, `harden`) คือ **teaching model และ lint** ของคอร์ส ไม่ใช่ OpenShell และไม่ใช่ **prover** ของ OpenShell การบังคับใช้จริงคือ Landlock, seccomp และ egress proxy บน Spark ส่วนการตรวจจริงว่า "change นี้เปิดอะไรใหม่บ้าง" คือ prover ซึ่งจะรอให้คนอนุมัติใน `openshell term` ใช้ policykit เพื่อคิด *ก่อน* push แล้วเชื่อคำตอบของ Spark *หลัง* push

✓ Checkpoint: ทั้งสามคำสั่งพิมพ์เวอร์ชันหรือ `mcp ok` ออกมา และคุณอธิบายได้ในประโยคเดียวว่าทำไม policykit ไม่ใช่ prover

## 1 · Threat model: claw ที่โดน prompt injection เอื้อมถึงอะไรได้บ้าง

เริ่มจากสิ่งที่ผู้โจมตีอยากได้ ไม่ใช่เริ่มจาก YAML

| Asset | ตัวอย่าง | ทำไมถึงมีค่า |
|---|---|---|
| filesystem ของ sandbox | `/sandbox` และ config tree ของ agent เอง | secret และ state อยู่ที่นี่ |
| provider credential | API key ของโมเดล | เอาไปใช้ที่ไหนก็ได้ |
| channel token | bot token ของ Telegram / Slack | แต่ละตัวคือช่องทางออกที่ใช้งานได้จริง |
| ข้อมูล CSV / BMS | ไฟล์ export ของ chiller, ค่า point สด | ข้อมูลของลูกค้า |
| **อำนาจเรียก tool** | `write_setpoint` | มันเปลี่ยนตึกจริงได้ |

ผู้โจมตีมีทั้ง indirect prompt injection (อะไรก็ตามที่ agent อ่าน: หน้าเว็บ, อีเมล, ผลลัพธ์ของ MCP tool, ไฟล์), skill หรือ plugin อันตรายจาก hub, dependency ที่ถูกเจาะแล้วดึงเข้ามาผ่าน preset `npm` / `pypi` และคนในที่เข้าถึง host ได้ ทุกแบบยกเว้นคนใน เริ่มต้นเหมือนกันหมด คือ **โมเดลอ่านข้อความที่ผู้โจมตีควบคุม** กลุ่ม CVE ของ OpenClaw จาก Module 01 (TOCTOU file escape, allow-list env disclosure, MCP loopback privilege escalation; ตามที่ research tutorial อ้าง Cyera) และบทวิเคราะห์ indirect prompt injection ของ CSA ก็เริ่มจากจุดนี้เช่นกัน ดังนั้นสิ่งที่คุณควบคุมได้ไม่ใช่ prompt ที่ดีขึ้น แต่คือสิ่งที่ process ทำได้จริงหลังจากนั้น

Lab 07-1 แปลงเรื่องนี้เป็นขั้นตอนโจมตีที่จับต้องได้ 12 ขั้น แล้วตัดสินแต่ละขั้นเทียบกับ **ฉบับร่างที่คอร์สทำขึ้น** (`policies/balanced_draft.yaml`: รูปแบบของ preset pypi, BMS ยังเป็น L4, OTel ยังเป็น audit, `best_effort`) และเทียบกับ **production policy** ของ tutorial (`policies/prod.yaml`)

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

ดูแถว ✓ สองแถวในคอลัมน์ **prod** `include_workdir: true` ทำให้ `/sandbox` เขียนได้ และ agent เป็นเจ้าของ config tree ของตัวเอง จึงอ่าน `/sandbox/.hermes/.env` และเขียนทับ `/sandbox/.openclaw` ได้ เอกสารของ NemoClaw ก็พูดตรงกันว่า tree นี้ไม่ใช่ขอบเขต isolation ไม่มี filesystem rule ไหนแก้เรื่องนี้ได้ ทางแก้จริงมีสองข้อ ข้อแรก อย่าเก็บ secret ไว้ในไฟล์ (ใช้ credential handle ของ OpenShell, Module 05) ข้อสอง ทำให้สิ่งที่ agent อ่านได้ออกไปไหนไม่ได้ ภายใต้ prod ขั้น A3 และ A4 ไม่มี host ให้ส่งข้อมูลออกไป

ข้อจำกัดบางอย่างอยู่นอกไฟล์ policy ทั้งหมด NemoClaw ระบุไว้สี่ข้อ:

| ข้อจำกัด | ทำไมจึงสำคัญ | วิธีลดความเสี่ยง |
|---|---|---|
| ข้ามเส้นทาง gateway ที่ถูกจัดการ | policy และ inference auth ไม่ถูกบังคับใช้กับ runtime ที่เปิดนอกเส้นทางที่ NemoClaw จัดการ | เปิด agent ผ่าน entrypoint ที่ถูกจัดการเท่านั้น ห้าม `docker exec` agent ตัวที่สองเข้าไปใน sandbox |
| lifecycle แบบ native ที่ใช้ UID เดียวกัน | supervisor, gateway และ agent ใช้ UID ของ sandbox ร่วมกัน agent ที่เป็น user เดียวกันส่ง signal หรือปลอมตัวเป็นตัวอื่นได้ | อย่าใส่อะไรใน sandbox ที่คุณไม่ยอมให้ agent |
| การเขียนไฟล์ตรงๆ เลี่ยง scanner | scanner เห็นแค่ tool call ไม่เห็น `echo secret > file` | จำกัดการเขียนด้วย Landlock และอย่าเก็บ secret ในไฟล์ |
| secret ที่ถูก encode ตรวจไม่เจอ | regex redaction พลาด Base64 / hex | ใช้ credential handle ของ OpenShell แทน secret ในไฟล์ |

✓ Checkpoint: คุณบอกได้ว่าขั้นโจมตีสองขั้นไหนยังสำเร็จภายใต้ prod และอธิบายได้ว่าทำไมคำตอบคือ "ไม่มีที่ให้ส่งออกไป" ไม่ใช่ filesystem rule เพิ่มอีกข้อ

## 2 · Hardening checklist (Alto sovereign profile)

§6.2 ของ research tutorial ให้ค่า setting ที่จับต้องได้บรรทัดละหนึ่งค่า พร้อมแหล่งอ้างอิง คอลัมน์สุดท้ายบอกว่า module นี้ตรวจแต่ละบรรทัดด้วยอะไร

| ด้าน | Setting | ตรวจใน module นี้ด้วย |
|---|---|---|
| Network | ทุก endpoint เป็น `protocol: rest` (หรือ `mcp`) พร้อม `enforcement: enforce`; ใช้ `rules` แบบระบุชัด ไม่ใช้ `access: full`; ไม่มี wildcard host; ใช้ `allowed_ips` เป็น CIDR แคบๆ สำหรับ private host | `harden()` (lab 07-2) |
| Inference host | ห้ามใส่ `api.openai.com` หรือ `integrate.api.nvidia.com` ใน policy; ส่ง inference ผ่าน OpenShell เท่านั้น | `harden()` → HIGH |
| Binary | หนึ่ง binary ต่อหนึ่ง endpoint; SHA256 TOFU pinning; ติดตั้ง tool ตอน build image ไม่ใช่ตอน runtime | `harden()` (นับจำนวน) · Dockerfile lint (lab 07-3) · TOFU: OpenShell เท่านั้น |
| Metadata SSRF | NemoClaw ฉีด `AWS_EC2_METADATA_DISABLED=true`; OpenShell block `169.254.0.0/16` เสมอ | `decide()` (block เสมอ) |
| Filesystem | `landlock.compatibility: hard_requirement`; `read_write` ให้น้อยที่สุด; kernel ≥ 6.2 (Landlock ABI 3) | `harden()` · `validate()` (ปฏิเสธ `/`) |
| Kernel / process | seccomp block mount, pivot_root, bpf, perf_event_open, userfaultfd, kexec, memfd_create และ AF_PACKET / AF_BLUETOOTH / AF_VSOCK; `no_new_privs`; `RLIMIT_CORE=0`; user ที่ไม่ใช่ root | `validate()` (root) · ที่เหลือ: OpenShell เท่านั้น |
| Gateway | คง `policy_validation_failure_mode = "fail_closed"` (ค่า default) ไว้ ไม่ใช้ `retain_last_valid` | ไม่อยู่ในไฟล์ policy |
| Tier | Restricted สำหรับ claw แบบ kiosk / เปิดตลอดเวลา; Balanced เฉพาะที่ต้องใช้ `pypi` / `npm` จริง; ห้ามใช้ Personal บนฮาร์ดแวร์ที่ใช้ร่วมกัน | บันทึกตอน onboarding |
| Channel | ถือว่า token ทุกตัวคือช่องทางออกที่ใช้งานได้; pair อุปกรณ์อย่างชัดเจน (`NEMOCLAW_DISABLE_DEVICE_AUTH` ถูกยกเลิกแล้ว และ OpenClaw 2026.9.1 ไม่สนใจค่านี้) | ไม่อยู่ในไฟล์ policy |
| Recovery | snapshot ก่อนเปลี่ยน; ถ้าสงสัย ให้ทำลายแล้วสร้างใหม่จาก input ที่เชื่อถือได้ | runbook ของคุณ (Module 08) |
| Formal check | ให้ prover ของ OpenShell ประเมินว่า change เปิดอะไรใหม่ แล้วรอคนอนุมัติ | prover ไม่ใช่คอร์สนี้ |

**Vendor ขอให้เพิ่ม `api.openai.com:443` "เพื่อให้ agent ใช้ GPT สรุปงานได้"** (Part 6, exercise 1) ให้ปฏิเสธการเพิ่ม entry นี้ใน policy การใส่ provider host ไว้ใน `network_policies` ทำให้ key ต้องอยู่ใน sandbox และข้ามการติดตาม usage ใน lab 07-2 คุณจะเห็น `harden()` ติดธง HIGH และ `decide()` ยอมให้ call ผ่าน ถ้าอนุญาตให้ใช้ cloud model จริงๆ ให้ลงทะเบียนเป็น OpenShell provider แล้วยังเรียก `inference.local` เหมือนเดิม เพื่อให้ gateway เป็นผู้ถือ key:

```bash
# on: spark
# --credential KEY with no =VALUE reads the key from your environment, so it never sits on the command line
openshell provider create --name openai-summaries --type openai --credential OPENAI_API_KEY
openshell inference set --provider openai-summaries --model <model>
```

รูปแบบ `KEY` อย่างเดียวมีอยู่ใน help ของ laptop CLI 0.0.111 ให้เช็ก `openshell provider create --help` บน Spark อีกครั้ง หรือใช้ `NEMOCLAW_PROVIDER=routed` / `custom` ก็ได้ แต่ใน tier **sovereign** คำตอบคือไม่ ไม่ว่าทางไหน เพราะโมเดลต้องรันใน on-prem

**ทำไม `hard_requirement` สำคัญกว่าเมื่อมีหลายเครื่อง** (Part 6, exercise 2) บน edge box หลายรุ่นที่ kernel ไม่เหมือนกัน `best_effort` จะลดระดับ Landlock ลงแบบเงียบๆ และแค่ปล่อย `DetectionFinding` ระดับ High ออกมา ส่วน `hard_requirement` จะไม่ยอม start เลย จุดอ่อนที่ซ่อนอยู่จึงกลายเป็นความล้มเหลวที่ดังพอให้คุณเห็นตอน rollout ไม่ใช่ช่องโหว่ที่ไปเจอตอน audit

✓ Checkpoint: สำหรับทุกบรรทัดใน checklist คุณบอกได้ว่า lint ตรวจได้หรือไม่ และรู้ว่าจะตอบ vendor ว่าอะไร

## 3 · L6.1 — Production policy สำหรับ Alto Ops Claw

นี่คือ policy ของ Lab 6.1 ใน research tutorial แบบคำต่อคำ อยู่ใน `policies/prod.yaml`:

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

แต่ละส่วนให้อะไรคุณ:

- **MCP rules เป็น allow-list** อนุญาต tool อ่านสามตัวโดยระบุชื่อ `deny_rules` ระบุ tool เขียนสองตัวเป็นด่านที่สอง tool ที่ไม่มีใครระบุชื่อ (เช่น `reboot_controller`) ก็ยังถูก deny
- **`allowed_ips: ["10.20.0.15/32"]`** ตรึง BMS host ที่เป็น private ไว้กับ address เดียว
- **`otel_collector` อนุญาตแค่ call เดียว:** `POST /v1/traces` ไม่ใช่ `/v1/logs` และไม่ใช่ GET
- **`on_error: fail_closed`** แปลว่าถ้าตัว redact พัง traffic จะถูก block แทนที่จะหลุดออกไปโดยไม่ถูก redact
- **`hard_requirement`** กับ uid ที่ไม่ใช่ root ล็อกชั้น static ซึ่งถูกกำหนดตายตัวตอนสร้าง sandbox

Lab 07-2 ส่ง policy นี้ผ่านตัวตรวจสามตัว CLI ตัวจริง parse ได้แบบ offline และหยุดเพียงเพราะไม่มี gateway ตอบ:

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

LOW หนึ่งตัวนั้นเกิดจาก lint ของคอร์สเข้มกว่าเอกสาร (private host แบบระบุชื่อตรงๆ ได้รับอนุญาต ที่ถูก block คือ entry แบบ wildcard หรือไม่มี host) ถ้ารู้ IP ของ `otel.alto.local` ก็ตรึงไว้ด้วย `/32`

จากนั้นคือสำเนาที่จงใจทำให้อ่อนลง 12 ชุด ชุดละหนึ่งบรรทัดของ checklist ให้อ่านทีละคอลัมน์ ตัวตรวจแต่ละตัวเห็นคนละอย่าง

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

Parser ตัวจริงจับ YAML และ *ชื่อ* field (`Version` จาก header ของ `policy get --full`, พิมพ์ผิดใน `mcp:`) ส่วน `validate` จับ *ค่า* ที่ผิด (root) มีแต่ lint ที่เห็นจุดอ่อนแบบเงียบ คือไฟล์ที่ parse ผ่านแต่ยังเปิดช่องโหว่อยู่

ต่อมา lab จะเปิด mock BMS **ตัวจริง** บน laptop แล้วขอรายชื่อ tool จากมัน gate เล็กๆ ของคอร์สจะส่ง call ที่อนุญาตต่อไปยัง server และปฏิเสธ `write_setpoint` ก่อนที่มันจะออกไป:

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

gate นี้เป็นโค้ดของคอร์สและข้อความก็เป็นของคอร์สเอง บน Spark หน้าที่นี้เป็นของ OpenShell proxy ซึ่งตอบกลับเป็น 403 พร้อม JSON body

**Apply บน Spark** นี่คือ change ดังนั้น lab จึงส่งผ่าน `change()` จะแสดง `policy get` แบบ read-only ก่อน และรัน `set` เฉพาะในโหมด LIVE ที่เปิด 🔓 ไว้เท่านั้น:

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

> ⚠ **ส่วน static** `policy set` เปลี่ยนส่วน network ได้ทันที แต่ `filesystem_policy`, `landlock` และ `process` ถูกกำหนดตายตัวตอนสร้าง sandbox ถ้าต้องการให้ `hard_requirement` และ `read_write` ชุดนี้มีผลจริง ให้สร้าง sandbox ด้วย `--policy prod.yaml` (Module 08) แหล่งอ้างอิงของคอร์สไม่ได้บอกว่า `policy set` ทำอะไรกับส่วน static ที่เปลี่ยนไป ดังนั้นให้อ่าน `openshell policy get alto-ops --full` หลัง apply ทุกครั้ง

**พิสูจน์เส้นทาง deny** สั่ง claw ว่า "set chiller 2 setpoint to 6.5°C" จากนั้นอ่าน sandbox log แล้วหา 403 ของ `write_setpoint` และดู tool span ที่ fail ใน Phoenix ด้วย (Module 05) ใช้ `-n` ไม่ใช่ `--tail` เพราะ `--tail` เป็น stream และ runner ไม่รันคำสั่งแบบ stream ไว้ด้านหน้า

```bash
# on: spark
openshell logs alto-ops -n 50 --source sandbox
```

แนวคิดการออกแบบเบื้องหลังทั้งหมดนี้: **การเขียนอยู่นอก sandbox** เส้นทางการเขียนเป็นบริการแยกที่คนต้องอนุมัติ (approvals flow ของ Alto Copilot) claw ทำได้แค่ *ขอ* ให้เขียน ผ่าน ticket คุณจะเขียน entry นั้นเองใน exercise

✓ Checkpoint: lab 07-2 แสดงว่า CLI ตัวจริง parse prod.yaml ได้, validate ได้ 0 error, มี LOW หนึ่งตัว และ `write_setpoint` ถูกปฏิเสธก่อนถึง mock BMS

## 4 · L6.2 — Custom image และ blueprint

policy จะเชื่อได้เฉพาะ binary ที่มีอยู่จริง OpenShell ตรึงแต่ละตัวไว้กับ SHA256 ที่เห็นครั้งแรก (trust on first use) ดังนั้น tool ต้องอยู่ **ใน image** ติดตั้งตอน build ไม่ใช่ให้ agent ไปดึงมาตอน runtime NemoClaw สร้าง sandbox จาก image ของคุณเองได้ด้วย `nemoclaw onboard --from ./Dockerfile` (หรือ `--from <image>`)

Lab 07-3 สร้าง image ของ Alto Ops Claw จาก spec เล็กๆ: เป็น Dockerfile ของ L3.8 ใน research tutorial โดยเอา `USER` มาจาก `run_as_user` ของ policy แล้ว lint ผลลัพธ์ จากนั้น lint สำเนาแบบ "ขอให้มันใช้ได้ก่อน":

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

tutorial บอกเองว่า Dockerfile นี้เป็นการประกอบของผู้เขียน ดังนั้นให้ตรวจบรรทัดติดตั้ง NAT กับ base image ของคุณ OpenShell ปฏิเสธ root ซึ่งเป็นเหตุผลที่ lint ถือว่า `USER root` เป็น error

```bash
# on: spark
nemoclaw onboard --from ./Dockerfile
```

wizard ของ onboard เป็นแบบ interactive ให้พิมพ์เองใน ⌨ terminal ตัว lab ทำแค่คัดลอก Dockerfile ไปให้ build context ยังต้องมี `workflows/alto_ops` และ `workflow.sandbox.yml` ด้วย ซึ่ง Module 08 จะประกอบให้ครบ

ตัวเลือกอื่นของ blueprint ที่ tutorial ระบุไว้ ทั้งหมดอยู่หลังขอบเขต OpenShell เดียวกัน:

| ตัวเลือก | ทำอย่างไร | หมายเหตุ |
|---|---|---|
| harness | Hermes (`nemohermes`, API บนพอร์ต 8642, plugin Langfuse) หรือ Deep Agents (`nemo-deepagents`, ค่า default คือ Nemotron 3 Ultra) | ทางเลือกระดับเดียวกับ OpenClaw |
| model routing | `NEMOCLAW_PROVIDER=routed` พร้อม `NVIDIA_INFERENCE_API_KEY`; `custom` / `anthropicCompatible` สำหรับ endpoint ที่เข้ากันได้ | เป็น cloud ไม่เหมาะกับ tier sovereign |
| อยู่บน Spark | `NEMOCLAW_PROVIDER=ollama`, `vllm` หรือ `install-vllm` | ค่า default ของ sovereign |
| host ที่ใช้ Podman | `NEMOCLAW_GATEWAY_RUNTIME=podman` | เลือก Podman สำหรับ container ของ gateway |

✓ Checkpoint: Dockerfile ที่สร้างขึ้นผ่าน lint และคุณบอกได้สามข้อว่าสำเนาแบบ "ขอให้มันใช้ได้ก่อน" ผิดตรงไหน

## 5 · L6.3 — Remote gateway และ external gateway

การเข้าถึง gateway ที่ไม่ได้อยู่บน laptop มีสองแบบ และ trust boundary ต่างกันมาก

**Remote gateway ที่คุณเป็นเจ้าของ** สร้างและควบคุมจาก CLI ของคุณผ่าน SSH playbook บอกว่า:

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

> ✏ **แก้จาก research tutorial** Lab 6.3 ของ tutorial พิมพ์ `openshell gateway add https://openshell:8080 --remote` โดยไม่มีค่าต่อท้าย `--remote` CLI ตัวจริงไม่รับแบบนั้น playbook เขียนเป็น `--remote <user>@<hardware-ip>` และ module นี้ก็ใช้แบบเดียวกัน อีกเรื่องคือ laptop CLI 0.0.111 **ไม่มี** subcommand `gateway start` เลย playbook ระบุคำสั่งนี้สำหรับ OpenShell ที่มันติดตั้ง ดังนั้นให้รัน `openshell gateway --help` บนเครื่องของคุณก่อนเขียนเป็น script

**Expected output** (captured on this Mac)

```
$ openshell gateway add https://openshell:8080 --remote   [this laptop]
  → the tutorial's line, as printed: exit 2 · error: a value is required for '--remote <REMOTE>' but none was supplied
$ openshell gateway add https://openshell:8080 --remote me@spark-01.alto.local   [this laptop]
  → with the SSH target: exit 1 · Error:   × mTLS certificates for gateway 'openshell' were not found.
$ openshell gateway start --remote user@spark-01.alto.local   [this laptop]
  → provision a remote gateway: exit 2 · error: unrecognized subcommand 'start'
```

**External gateway ที่ทีม platform เป็นเจ้าของ** (experimental) ถ้าทีม infrastructure รัน OpenShell บน Kubernetes / Helm อยู่แล้ว `nemoclaw-blueprint-runner` ของ NemoClaw จะชี้ไปที่ gateway นั้นผ่าน `blueprint.yaml` (`blueprint/blueprint.yaml` คำต่อคำจาก tutorial):

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

กฎที่ tutorial ระบุ: **bare HTTPS origin**, OpenShell release แบบ **ตรงตัว** (min = max), **CA bundle ที่เป็น PEM ล้วน ≤ 1 MiB** เป็น **ไฟล์ปกติ ไม่ใช่ symlink** และ path ของไฟล์ authentication แบบ **absolute** SDK ตรวจ certificate และ hostname เทียบกับ bundle นี้ และใช้ DNS ของ platform ดังนั้นคุณต้องควบคุมได้ว่า hostname ของ target resolve ไปที่ไหน Lab 07-3 เขียน CA และไฟล์ credential แบบทิ้งขว้างไว้ใต้ `.runs/fakeroot/` ตรวจ blueprint ของ tutorial ด้วย validator ของคอร์ส แล้วทำให้พัง 11 แบบ:

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

บน Spark ตัว runner จริงเป็นผู้ตรวจเรื่องเหล่านี้ ทั้งสองคำสั่งเป็น read-only: `plan` "validates, fingerprints CA, no network" ส่วน `status --external-target` ส่ง "one credential-free health request"

```bash
# on: spark
NEMOCLAW_BLUEPRINT_PATH=$HOME/week26/07_hardening/blueprint nemoclaw-blueprint-runner plan
NEMOCLAW_BLUEPRINT_PATH=$HOME/week26/07_hardening/blueprint nemoclaw-blueprint-runner status --external-target
```

tutorial เขียน `NEMOCLAW_BLUEPRINT_PATH=/abs/path/blueprint` โดยไม่ได้บอกว่าต้องการโฟลเดอร์หรือไฟล์ ให้เช็กด้วย `nemoclaw-blueprint-runner --help`

**Trust boundary** (Part 6, exercise 3):

| | `openshell gateway start --remote` | `nemoclaw-blueprint-runner` + external target |
|---|---|---|
| ใครรัน gateway | คุณ จาก CLI ของคุณผ่าน SSH | ทีม platform |
| lifecycle | ของคุณ: start, stop, destroy, upgrade | ของเขา: คุณได้แค่ plan และ status |
| authentication | SSH + mTLS ที่ gateway สร้างให้ | ไฟล์ credential + CA bundle ที่ตรึงไว้ |
| เวอร์ชัน OpenShell | อะไรก็ได้ที่คุณติดตั้ง | ตรึงไว้: min = max = `expected_release` |
| trust boundary | workstation ของคุณ ↔ Spark ของคุณ | claw ของคุณ ↔ control plane ของคนอื่น |

สองข้อที่ต้องรู้ก่อนสัญญาอะไรกับลูกค้า แท็บ multi-node ของ NemoClaw จำกัดไว้ที่ **DGX Station** ถ้าเป็น Spark สองเครื่อง blog ของ NVIDIA พูดถึงการทำ cluster ในระดับ vLLM ไม่ใช่ผ่าน NemoClaw ดังนั้นต้องทดสอบก่อน สำหรับการฝึกอบรมหรือเพิ่มกำลังชั่วคราว Brev มี NemoClaw launchable และ OpenShell agent sandbox ให้ใช้ ส่วนคอร์ส DLI "Securing Agents with OpenShell and NemoClaw" คือหลักสูตรทางการที่ใกล้กับสัปดาห์นี้มากที่สุด

✓ Checkpoint: lab 07-3 จับ blueprint ที่พังได้ 11/11 และคุณอธิบายได้ว่าใครเป็นเจ้าของ lifecycle ในแต่ละเส้นทางของ gateway

## 6 · หลักฐานสำหรับเจ้าของโรงแรม และ tier ของ Alto Copilot

tier เป็นตัวตัดสินว่าประโยค "ไม่มีข้อมูลออกจากตึก" เป็นจริงได้หรือไม่ (§6.6):

| Alto Copilot tier | รูปแบบ claw | Inference | Posture | Telemetry |
|---|---|---|---|---|
| Cloud | NAT workflow หลัง gateway ของ Copilot; มี OpenShell sandbox ต่อ tenant บน Brev / K8s ได้ | NVIDIA endpoints หรือ Model Router | Development / Integration Testing ระหว่าง build; Locked-Down ใน prod | Phoenix / OTel แยก project ต่อ tenant |
| Sovereign (on-prem) | NemoClaw บน DGX Spark / Station; NAT อยู่ใน OpenShell | vLLM / Ollama ในเครื่องผ่าน `inference.local` | Restricted + custom preset (`alto-bms`, `otel_collector`); `hard_requirement` | OTel collector ใน on-prem; audit trail จาก log ของ `openshell` |
| Edge | claw OpenClaw / Hermes แบบ Restricted อยู่ข้าง BMS ใช้ tool อ่านอย่างเดียว | โมเดล NVFP4 ในเครื่อง | Locked-Down; เขียนได้ผ่าน approvals service เท่านั้น | file exporter ในเครื่อง + sync เป็นระยะ |

เจ้าของโรงแรมจะได้ยินสามประโยค: เส้นทาง inference มีแค่ on-prem; ทุกการตัดสินใจด้าน network ถูกบันทึก; agent ไม่มีอำนาจเขียนตั้งแต่โครงสร้าง **หลักฐาน** รองรับแต่ละประโยค (Part 6, exercise 5) Lab 07-4 ตรวจว่าทุกคำสั่ง parse ผ่านบน laptop CLI:

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

แหล่งอ้างอิงของคอร์สไม่ได้บอกว่า gateway ของคุณเก็บ log ได้ครบ 30 วันหรือไม่ ให้ export ทุกวันแล้วส่งมอบไฟล์ที่ export ไว้ ตัวตรวจ log ของ lab นับการตัดสิน `allow` ไปยัง host ที่อยู่นอกตึก มันรันกับ **EXAMPLE log ที่คอร์สทำขึ้น** สองชุด เพราะไม่มีแหล่งอ้างอิงไหนของคอร์สพิมพ์บรรทัด `openshell logs` จริงไว้:

**Expected output** (captured on this Mac)

```
◆ example clean month: 3 allow · 3 deny · 1 inference · 0 line(s) not parsed
✓ example clean month: 0 allow decisions to outside hosts
◆ example leaky month: 5 allow · 3 deny · 1 inference · 1 line(s) not parsed
✕ allow to an OUTSIDE host: 2026-09-21T19:02:11Z alto-ops allow api.openai.com:443 /usr/bin/python3.12 POST /v1/chat/completions audit-violation
✕ allow to an OUTSIDE host: 2026-09-21T19:02:12Z alto-ops allow 203.0.113.7:443 /usr/bin/python3.12 CONNECT
```

เดือนที่รั่วล้มเพราะมี `allow` ที่ถูกทำเครื่องหมายว่าเป็น audit violation นั่นคือ `enforcement: audit` ทำงานตามที่เอกสารบอก: บันทึก แล้วส่งต่อ กับ log จริง ตัวตรวจจะบอกด้วยว่ามีกี่บรรทัดที่ parse ไม่ได้ **parse ได้ 0 บรรทัด ไม่ได้พิสูจน์อะไรเลย** ดังนั้นให้ปรับ pattern ให้ตรงกับรูปแบบ log ของคุณก่อนเซ็นรับรองอะไร

✓ Checkpoint: คุณไล่รายการหลักฐานพร้อมคำสั่งของแต่ละข้อได้ และอธิบายได้ว่าทำไม allow แค่ครั้งเดียวในโหมด audit ก็ทำให้คำยืนยันพัง

## Labs — รันได้ที่นี่

**labs/lab07_1_threat_model.py** — ขั้นโจมตี 12 ขั้นของ agent ที่โดน prompt injection ตัดสินเทียบกับ policy ฉบับร่างและ production policy พร้อมข้อจำกัดที่ policy แก้ไม่ได้

**labs/lab07_2_prod_policy.py** — production policy ผ่าน CLI parser ตัวจริง, validate และ hardening lint; สำเนาที่อ่อนลง 12 ชุด; รายชื่อ tool จริงของ mock BMS; `policy set` ผ่าน change()

**labs/lab07_3_blueprints_and_gateways.py** — Dockerfile ของ custom image และ lint, validator ของ blueprint สำหรับ external gateway พร้อมสำเนาที่พัง 11 ชุด และคำสั่ง remote gateway

**labs/lab07_4_sovereign_evidence.py** — ชุดหลักฐาน "ไม่มีข้อมูลออกจากตึก": tier, คำสั่ง, การตรวจ host ใน policy และตัวตรวจ log

## Try it yourself

`exercises/ex07_approvals_endpoint.py` คือ exercise 4 ของ Part 6 approvals service ของ Alto Copilot (`approvals.alto.local:443`) ต้องเข้าถึงได้เพื่อให้ claw ขอเขียนได้ แต่ claw ต้องอนุมัติเองไม่ได้เด็ดขาด มี TODO สี่ข้อ:

1. allow rule สองข้อ: สร้าง ticket (`POST /api/v1/tickets`) และอ่าน ticket หนึ่งใบ (`GET /api/v1/tickets/<id>`)
2. deny rule หนึ่งข้อที่ match URL approve ได้จริงทุกแบบ
3. โหมด enforcement
4. binary เพียงตัวเดียว

```bash
# on: laptop
.venv/bin/python week26/07_hardening/exercises/ex07_approvals_endpoint.py
```

> ✏ **แก้จาก research tutorial** เฉลยของ tutorial พิมพ์ deny rule เป็น `{ method: "", path: "/api/v1/tickets//approve" }` คือ method ว่างและมี slash ซ้อนกัน เป็นไปได้มากว่าเครื่องหมายดอกจันสองตัวถูก Markdown กินไปเป็นตัวเอียง ตามที่พิมพ์ไว้ rule นี้ไม่มีวัน match URL approve จริง จึงเป็นโค้ดที่ไม่มีผล และมีแค่ allow-list ที่กันการ approve ไว้ เฉลยของคอร์สใช้ `{ method: "*", path: "/api/v1/tickets/*/approve" }` และตัวตรวจจะทดสอบว่า deny rule ของคุณ match `POST /api/v1/tickets/42/approve` ได้จริง

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

<details><summary>Hint — ทำไมใช้ "*" ไม่ใช่ POST ใน deny rule?</summary>

ลองรันตัวตรวจด้วย `method: POST` แถว "approve with a GET link" จะล้ม ใน model ของคอร์ส glob `*` match `/` ได้ด้วย rule GET `/api/v1/tickets/*` จึง match `/api/v1/tickets/42/approve` glob ของ OpenShell เองดูเหมือนจะแยกตาม segment (เอกสารใช้ `**` สำหรับ "หลาย segment") ให้ตรวจสอบกับเครื่องของคุณ การ deny ทุก method ไม่มีต้นทุนอะไร และได้ผลไม่ว่าแบบไหน

</details>

<details><summary>Hint — enforcement</summary>

`audit` บันทึก violation แล้ว **ส่งต่อ** request ไป สำหรับ call ที่เป็นการ approve คำว่า "เราบันทึกไว้แล้ว" ไม่เท่ากับ "มันไม่ได้เกิดขึ้น"

</details>

✓ Checkpoint: ตัวตรวจทั้งหกบรรทัดเป็น ✓

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `error: a value is required for '--remote <REMOTE>'` | `gateway add … --remote` ใน research tutorial ไม่มี SSH target | เติม `<user>@<spark-ip>` แบบที่ playbook เขียน |
| `error: unrecognized subcommand 'start'` บน laptop | laptop CLI 0.0.111 ไม่มี `gateway start` | รันด้วย OpenShell ที่ playbook ติดตั้ง และเช็ก `openshell gateway --help` |
| `mTLS certificates for gateway 'openshell' were not found` | `gateway add --remote` ต้องใช้ client TLS ที่ gateway สร้างให้ | สร้าง gateway ก่อน (`gateway start --remote`) แล้วค่อย add |
| `unknown field 'Version'` จาก `policy set` | คุณป้อน output ของ `policy get --full` ซึ่งมี metadata header | export ด้วย `openshell policy get <sandbox>` (ไม่ใส่ `--full`) |
| lab 07-2 ข้ามขั้น mock BMS | ปัญหาเรื่องพอร์ตหรือ venv | อ่าน `week26/07_hardening/.runs/bms_mcp.log` |
| lab 07-3 บอกว่าไม่พบไฟล์ CA | ไม่มี `openssl` ใน PATH | ติดตั้ง หรืออ่านแถวอื่นไปก่อน เพราะแถวอื่นไม่ต้องใช้ |
| ตัวตรวจหลักฐาน parse log จริงได้ 0 บรรทัด | pattern ของคอร์สเขียนให้ตรงกับรูปแบบ EXAMPLE ไม่ใช่รูปแบบของคุณ | ปรับ `LINE` ใน lab 07-4 ให้ตรงกับบรรทัด log ของคุณ |

## Next

[Lab 08 — Capstone: Alto Ops Claw v1](../08_capstone_alto_ops_claw/TUTORIAL.md): build custom image, สร้าง sandbox ด้วย production policy นี้, พิสูจน์เส้นทาง write-deny, trace request เดียวข้ามสาม plane และเขียน runbook
