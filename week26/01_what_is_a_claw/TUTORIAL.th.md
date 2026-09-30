# ▶ Reef Lab 01 — claw คืออะไร? สแตก, sandbox และ harness ทั้งสามตัว

> ส่วนหนึ่งของ Week 26 · NemoClaw บน DGX Spark ตั้งแต่ระดับเริ่มต้นจนถึงระดับผู้เชี่ยวชาญ คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง แล็บบนแล็ปท็อป (CLI ของ NAT และ OpenShell รวมถึงโมเดล policy) รันได้จริงทุกที่ ส่วนแล็บบน Spark รันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (ยกมาจากเอกสารหรือ playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เรียนรู้นิยามของ claw ในประโยคเดียว และส่วนประกอบเจ็ดส่วนของสแตกที่คุณจะตั้งค่ากันตลอดทั้งสัปดาห์
- เข้าใจว่าทำไม "เอเจนต์รันอยู่ใน sandbox" ไม่ได้แปลว่า "เอเจนต์ปลอดภัย"
- ไล่ดูห้าชั้นป้องกันแบบ deny-by-default แล้วแยกออกเป็นสองชั้นที่เปลี่ยนได้ขณะรัน กับสองชั้นที่ถูกล็อกตั้งแต่ตอนสร้าง
- เปรียบเทียบ harness ทั้งสามตัว (OpenClaw, Hermes, Deep Agents) แล้วเลือกให้เหมาะกับแต่ละงาน
- รัน lab 01: หาว่าส่วนไหนของสแตกมีอยู่แล้วบนแล็ปท็อปเครื่องนี้และบน Spark ของคุณ

**Time** ~30 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** ไม่ต้องใช้ (DRY + แล็ปท็อป) · DGX Spark 1 เครื่อง (ไม่บังคับ)

**แหล่งอ้างอิง:** research tutorial ของคอร์ส *NemoClaw on DGX Spark — Beginner to Expert* (Part 0) ซึ่งอ้างถึง [NVIDIA Build a Claw](https://www.nvidia.com/en-us/ai/build-a-claw/) · [NemoClaw how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works) · [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันแล็บและ Reef Lab Runner |
| เครื่องมือบนแล็ปท็อปของ Week 26 | `week26/.venv-nat/bin/nat --version` และ `week26/.venv-openshell/bin/openshell --version` | เอเจนต์ NAT และตัว parse policy ของ OpenShell รันบนแล็ปท็อปของคุณใน Module 03–08 |
| DGX Spark (ไม่บังคับ) | `ssh -o BatchMode=yes <spark> true` | Module 02–03 ติดตั้งและควบคุม claw ตัวจริง ถ้าไม่มี Spark ก็รันแบบ DRY |

ถ้ายังไม่มีเครื่องมือสองตัวนี้บนแล็ปท็อป ใน `week26/README.md` มีคำสั่ง `uv` สองบรรทัดสำหรับสร้างไว้ให้แล้ว

```bash
# on: laptop
week26/.venv-nat/bin/nat --version
week26/.venv-openshell/bin/openshell --version
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
nat, version 1.9.0
openshell 0.0.111
```

> 📌 **เวอร์ชันเปลี่ยนเร็วมาก** research tutorial เขียนโดยอิงเอกสาร NAT 1.8 แต่คอร์สนี้ติดตั้ง NAT **1.9.0** NemoClaw ตรึง (pin) OpenShell ไว้ที่ **0.0.116** บน Spark ขณะที่ OpenShell รุ่นล่าสุดคือ **0.1.2** ส่วน CLI บนแล็ปท็อปคือ **0.0.111** ซึ่งเป็นรุ่นสุดท้ายบน PyPI ที่มี binary สำหรับ macOS คอร์สจึงใช้มันแค่ *parse* policy แบบออฟไลน์เท่านั้น ถ้า flag บนเครื่องคุณไม่ตรงกับที่เขียนไว้ ให้เชื่อ `--help` เป็นหลัก

✓ Checkpoint: ทั้งสองคำสั่งพิมพ์เวอร์ชันออกมา หรือคุณรู้แล้วว่ายังต้องติดตั้งตัวไหนเพิ่ม

## 1 · claw ในประโยคเดียว

หน้า Build-a-Claw ของ NVIDIA อธิบาย NemoClaw ว่าเป็น reference stack แบบโอเพนซอร์สที่ deploy ได้ด้วยคำสั่งเดียว โดยรวม agent harness, secure runtime อย่าง NVIDIA OpenShell และโมเดล Nemotron ไว้ด้วยกัน จากตรงนั้น research tutorial จึงนิยาม claw ไว้ว่า:

> **เอเจนต์ที่ทำงานตลอดเวลาและใช้ tool ได้ (ตัว harness) + โมเดลที่รันบนเครื่องหรือถูก route ไปที่อื่น (ค่าเริ่มต้นคือ Nemotron) + sandbox และขอบเขต policy ที่บังคับใช้ระดับ kernel (OpenShell) ประกอบเข้าด้วยกันด้วยตัวติดตั้งและ CLI (NemoClaw)**

สแตกนี้มีเจ็ดส่วน และคุณจะได้แตะครบทุกส่วนในสัปดาห์นี้

| ส่วนประกอบ | หน้าที่ | รันที่ไหน | Module |
|---|---|---|---|
| **NemoClaw CLI** (`nemoclaw`, `nemohermes`, `nemo-deepagents`) | ตัวติดตั้งและจัดการวงจรชีวิต: onboard, status, logs, policy add/remove, snapshot, rebuild, inference set | Spark host | 02 |
| **OpenShell gateway** | control plane: วงจรชีวิตของ sandbox, credential, revision ของ policy, เส้นทาง inference (พอร์ต 8080) | Spark host (Docker) | 02–03 |
| **OpenShell sandbox + supervisor** | data plane: Landlock, seccomp, network namespace ที่มี policy proxy, การดักจับ `inference.local` | container | 03 |
| **Harness** | OpenClaw (ค่าเริ่มต้น), Hermes หรือ LangChain Deep Agents — agent loop, tool และช่องทางสื่อสาร (channel) | ภายใน sandbox | 02 |
| **Blueprint** | YAML ที่มีเวอร์ชัน: image, agent manifest, network policy, inference profile | repo (`nemoclaw-blueprint/`) | 07 |
| **Inference provider** | vLLM หรือ Ollama บน Spark หรือ endpoint ของ NVIDIA / OpenAI / Anthropic | Spark host หรือคลาวด์ | 02, 04 |
| **NeMo Agent Toolkit (NAT)** | สร้าง, profile, ประเมินผล และ serve agent workflow (YAML + Python) เป็นได้ทั้ง MCP client และ server | Spark host, ภายใน sandbox หรือแล็ปท็อปของคุณ | 04–06 |

```text
 Spark host                                          OpenShell sandbox (container)
┌─────────────────────────────────────┐  manages   ┌──────────────────────────────────────────┐
│ nemoclaw CLI ──► OpenShell gateway ─┼───────────►│ supervisor: Landlock · seccomp · netns    │
│                  :8080  credentials │            │   policy proxy (deny by default)          │
│                  policy revisions   │            │ harness: OpenClaw / Hermes / Deep Agents  │
│ vLLM :8000  /  Ollama :11434  ◄─────┼─ inference │   or a NAT workflow (Module 04)           │
│   (the model — local, on the GB10)  │   .local   │   calls https://inference.local/v1        │
└─────────────────────────────────────┘            └──────────────────────────────────────────┘
```

ลำดับการทำงานตามหน้า how-it-works ของ NemoClaw: CLI บน host คุยกับ gateway และ gateway เป็นผู้จัดการ sandbox เอเจนต์เรียก `https://inference.local` จากนั้น gateway จะใส่ credential ตัวจริงแล้วส่งต่อการเรียกไปยัง provider ที่ตั้งค่าไว้ เอเจนต์จึงไม่เคยถือ key ของ provider เลย

✓ Checkpoint: คุณบอกได้ว่าส่วนไหนถือ credential ของ provider (gateway) และถ้าจะเปลี่ยนจาก OpenClaw ไปเป็น NAT workflow ต้องเปลี่ยนส่วนไหน (harness)

## 2 · ทำไม sandbox จึงเป็นหัวใจของทั้งหมด

OpenClaw ซึ่งเป็น harness ค่าเริ่มต้น เจอปัญหาด้านความปลอดภัยหนักหน่วงตลอดปี research tutorial อ้างถึงการเปิดเผยช่องโหว่สี่รายการของ Cyera เมื่อเดือนกันยายน 2026 ซึ่งสามในสี่รายการถูกโจมตีได้จากจุดตั้งต้นแค่ prompt injection ครั้งเดียว และยังอ้างถึงบันทึก "Claw Chain" ของ Cloud Security Alliance ที่อธิบายว่าช่องโหว่เหล่านี้ต่อกันเป็นลูกโซ่ได้อย่างไร

| CVE (ตาม research tutorial ซึ่งอ้างถึง Cyera) | ประเภท | CVSS |
|---|---|---|
| CVE-2026-44112 | TOCTOU หลุดออกจาก filesystem ด้วยการ **เขียน** | 9.6 |
| CVE-2026-44115 | environment variable รั่วผ่าน execution allowlist | 8.8 |
| CVE-2026-44118 | ยกระดับสิทธิ์ผ่าน MCP loopback | 7.8 |
| CVE-2026-44113 | TOCTOU หลุดออกจาก filesystem ด้วยการ **อ่าน** | 7.7 |

ทุกช่องโหว่เริ่มต้นแบบเดียวกัน คือโมเดลอ่านข้อความที่ผู้โจมตีควบคุมได้ เช่น หน้าเว็บ อีเมล ผลลัพธ์จาก tool หรือไฟล์ การเขียน prompt ให้ดีขึ้นแก้ปัญหานี้ไม่ได้ สิ่งที่ปกป้องคุณได้จริงคือ **สิ่งที่โปรเซสทำได้จริงในทางกายภาพหลังจากนั้น** และนั่นคือสิ่งที่ OpenShell คอยจำกัดไว้

คู่มือความปลอดภัยของ NemoClaw ก็พูดประเด็นเดียวกัน: policy ของ OpenShell และ credential provider คือขอบเขตที่บังคับใช้จริง ส่วน config ที่เอเจนต์แก้ไขเองได้ (`/sandbox/.openclaw`, `/sandbox/.hermes`, `/sandbox/.deepagents`) ถูกระบุไว้ชัดเจนว่า **ไม่** นับเป็นการแยกส่วน (isolation) ที่เชื่อถือได้ เพราะเอเจนต์เขียนทับมันได้

ให้จำภาพนี้ไว้ตลอดทั้งสัปดาห์:

> **harness คือสิ่งที่ไม่น่าไว้ใจซึ่งคุณกำลังกักเอาไว้ OpenShell คือสิ่งที่คุณกำลังตั้งค่าจริง ๆ ส่วน NAT คือวิธีที่คุณใช้สร้างตรรกะของเอเจนต์ที่อยากให้รันอยู่ข้างใน**

✓ Checkpoint: อธิบายให้ GM ของโรงแรมฟังในประโยคเดียว ว่าทำไม "เอเจนต์รันอยู่ใน sandbox" ไม่ได้แปลว่า "เอเจนต์ปลอดภัย" (แบบฝึกหัด 01 ถามข้อนี้)

## 3 · ห้าชั้น สองความเร็ว

การป้องกันเชิงลึก (defence in depth) ของ NemoClaw มีห้าชั้นแบบ deny-by-default สี่ชั้นในนั้นเป็น **policy layer** ที่คุณเขียนด้วย YAML และเปลี่ยนแปลงได้ด้วยความเร็วต่างกันสองแบบ:

| ชั้น | บังคับใช้โดย | ส่วนใน policy | เปลี่ยนได้ขณะ sandbox รันอยู่ไหม? |
|---|---|---|---|
| Filesystem | Landlock LSM (Linux 6.2+, ABI 3) | `filesystem_policy`, `landlock` | **ไม่ได้** — ล็อกตั้งแต่ตอนสร้าง |
| Process | seccomp BPF, การลดสิทธิ์ (privilege drop), ผู้ใช้ที่ไม่ใช่ root | `process` | **ไม่ได้** — ล็อกตั้งแต่ตอนสร้าง |
| Network | CONNECT proxy + OPA policy engine ใน network namespace ของตัวเอง | `network_policies` | **ได้** — `openshell policy update / set` |
| Inference | supervisor ดักจับ `inference.local` แล้ว gateway เป็นผู้ route | ตั้งด้วย `openshell inference set` | **ได้** — hot-reload ได้ |
| Gateway authentication | ตัว gateway เอง (token, การจับคู่อุปกรณ์) | ไม่อยู่ใน sandbox policy | — |

playbook NemoClaw ของ DGX Spark ระบุการแบ่งนี้ไว้ตรง ๆ: network และ inference เป็นแบบ hot-reload ได้ ส่วน filesystem และ process ถูกล็อกตั้งแต่ตอนสร้าง sandbox

การแบ่งแบบนี้มีความหมายในทางปฏิบัติ:

- ถ้าจะ **เปิด host ใหม่** ให้เอเจนต์ ให้อัปเดต sandbox ที่รันอยู่ได้เลย (Module 03, lab 03-3)
- ถ้าจะ **ให้เอเจนต์มีไดเรกทอรีใหม่ที่เขียนได้** ต้องสร้าง sandbox ใหม่ ไม่มีทางแก้แบบสด ๆ
- ถ้าจะ **เปลี่ยนโมเดล** ให้เปลี่ยนเส้นทาง inference โดยที่ sandbox ยังรันต่อไปได้

Lab 01-2 ด้านล่างจะพาไล่ดูการตัดสินใจหนึ่งครั้งต่อหนึ่งชั้น โดยใช้โมเดลสอน `policykit` ของคอร์สกับ policy ที่มีคำอธิบายประกอบจาก Lab 2.2 ของ research tutorial

✓ Checkpoint: บอกได้โดยไม่ต้องเปิดดู ว่าสองชั้นไหน hot-reload ได้ (network, inference) และสองชั้นไหนถูกล็อกตั้งแต่ตอนสร้าง (filesystem, process)

## 4 · harness ทั้งสามตัว

harness คือ agent loop ที่อยู่ภายใน sandbox NemoClaw มีมาให้สามตัว และทั้งสามตัวอยู่หลังขอบเขต OpenShell เดียวกัน:

| Harness | โมเดลค่าเริ่มต้น (NVIDIA Endpoints) | ไดเรกทอรี state | CLI | เก่งเรื่อง |
|---|---|---|---|---|
| **OpenClaw** | `nvidia/nemotron-3-super-120b-a12b` | `/sandbox/.openclaw` | `nemoclaw` | web dashboard ที่พอร์ต 18789, `openclaw tui`, ช่องทาง Telegram / Discord / Slack, ค้นหาด้วย Brave หรือ Tavily |
| **Hermes** | Nemotron 3 Super | `/sandbox/.hermes` | `nemohermes` | API ที่เข้ากันได้กับ OpenAI ที่พอร์ต 8642, Tavily, plugin ของ Langfuse ที่เก็บ key ไว้นอก sandbox |
| **LangChain Deep Agents Code** | Nemotron 3 Ultra | `/sandbox/.deepagents` | `nemo-deepagents` | planner ที่แตก sub-agent ออกมาทำงาน, งานเขียนโค้ด |

บน DGX Spark ปกติคุณจะไม่ได้ใช้ค่าเริ่มต้นของ NVIDIA Endpoints เพราะ Module 02 จะ route inference ไปยังโมเดล vLLM หรือ Ollama ที่รันอยู่ **บนเครื่อง** ทำให้ prompt และข้อมูลไม่ออกไปนอกอุปกรณ์

ทางเลือกที่สี่ของสัปดาห์นี้คือ **claw ของคุณเอง** NAT workflow ที่คุณเขียนใน Module 04 จะรันอยู่ใน sandbox แบบเดียวกัน และเรียก `inference.local` ตัวเดียวกัน

✓ Checkpoint: คุณเลือก harness ได้สำหรับ (a) บอท concierge บน Telegram (b) เอเจนต์ refactor โค้ดใน repo ส่วนตัว (c) เอเจนต์ที่ส่ง trace ไปยัง Langfuse โดยไม่มี key ดิบอยู่ใน sandbox

## 5 · แผนที่คอร์ส และตัวอย่างที่ใช้ตลอดทั้งสัปดาห์

| Module | ส่วนของ research tutorial | สิ่งที่คุณสร้าง |
|---|---|---|
| 01 · บทนี้ | Part 0 | ภาพรวมความเข้าใจ (mental model) |
| 02 · claw ตัวแรก | Part 1 | Spark ที่ตรวจสอบแล้ว, NemoClaw ที่ติดตั้งแล้ว และหลักฐานว่า inference รันบนเครื่อง |
| 03 · policy as code | Part 2 | policy ที่คุณอ่าน เขียน ปรับแก้ซ้ำ และ roll back ได้; นำ vLLM ของคุณเองมาใช้ |
| 04 · NAT claw | Part 3 | **Alto Ops Claw**: เอเจนต์ NAT ที่มี tool สำหรับ chiller plant, serve ผ่าน REST และ MCP ภายใน sandbox |
| 05 · tracing | Part 4 | agent trace, policy log และ harness log ของ request เดียวกัน ที่เชื่อมโยงเข้าหากัน |
| 06 · benchmarking | Part 5 | ตัวเลขของ engine, workflow, คุณภาพ และ overhead ของ sandbox ที่คุณวัดเอง |
| 07 · hardening | Part 6 | threat model, policy สำหรับ production, blueprint แบบกำหนดเอง, remote gateway |
| 08 · capstone | Capstone | Alto Ops Claw v1 ที่มีเอกสารครบและผ่านการให้คะแนน |

ตัวอย่างที่ใช้ตลอดทั้งสัปดาห์คือ **Alto Ops Claw**: ผู้ช่วยด้านงานปฏิบัติการของโรงแรมที่อ่านไฟล์ CSV ที่ export จาก chiller plant ตอบคำถามเรื่องพลังงาน และเรียก BMS (แบบจำลอง) ผ่าน MCP ข้อมูลของมันอยู่ที่ `week26/common/data/chiller_plant.csv` ข้อมูลนี้เป็น **ข้อมูลสังเคราะห์ (synthetic)**: เจ็ดวัน แถวละ 15 นาที โดยหกชั่วโมงสุดท้ายถูกทำให้ประสิทธิภาพตก เพื่อให้เส้นทาง alarm มีอะไรให้ตรวจเจอ

> 🪸 **The Reef** ปุ่ม **🪸 Reef** ของ runner แสดงสแตกของ claw ให้เห็นในภาพเดียว: gateway เป็นประภาคาร แต่ละ sandbox เป็นคอกหนึ่งคอก และ service ต่าง ๆ เป็นจุดสังเกตบนเกาะ ทุกค่ามาจากคำสั่งจริง (`openshell status`, `openshell sandbox list`, `openshell inference get`) หรือจากการ probe ผ่าน HTTP พร้อมแสดง exit code และเวลา ถ้าไม่ได้เชื่อมต่อ Spark เกาะจะว่างเปล่าและบอกเหตุผล มันไม่เคยแสดงสถานะที่แต่งขึ้นเอง แอปภาพเต็มรูปแบบที่เป็นต้นกำเนิดของมันอยู่ใน `week26/alto-reef/`

✓ Checkpoint: คุณเปิด **🪸 Reef** มาแล้วหนึ่งครั้ง และบอกได้ว่ามันแสดงอะไรเมื่อไม่ได้เชื่อมต่อ Spark

## Labs — รันแล็บได้ที่นี่

**labs/lab01_1_claw_map.py** — ตรวจว่าส่วนไหนของสแตก claw มีอยู่บนแล็ปท็อปเครื่องนี้และบน Spark ของคุณ โดยแต่ละส่วนตรวจด้วยคำสั่งจริงแบบอ่านอย่างเดียว

**labs/lab01_2_five_layers.py** — การตัดสินใจ allow/deny หนึ่งครั้งต่อหนึ่ง policy layer บน policy ที่มีคำอธิบายประกอบจาก research tutorial แยกเป็นกลุ่ม hot-reload ได้ กับกลุ่มที่ถูกล็อก

## Try it yourself — ลองทำเอง

`exercises/ex01_claw_basics.py` มี TODO สามจุด:

1. แยก policy layer ทั้งสี่ชั้นออกเป็น `HOT` (เปลี่ยนได้ขณะ sandbox รันอยู่) และ `LOCKED` (ล็อกตั้งแต่ตอนสร้าง)
2. เลือก harness ให้กับงานทั้งสามงานใน Section 4
3. เขียนคำตอบหนึ่งย่อหน้าเรื่อง "อยู่ใน sandbox ไม่ได้แปลว่าปลอดภัย" สำหรับ GM ของโรงแรม ตัวตรวจจะมองหาสองแนวคิดที่คำตอบต้องมี

```bash
# on: laptop
.venv/bin/python week26/01_what_is_a_claw/exercises/ex01_claw_basics.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ layers: network + inference are hot-reloadable · filesystem + process are locked at creation
✓ harnesses: concierge → openclaw · refactor → deepagents · langfuse → hermes
✓ your GM paragraph says what the sandbox limits AND what it does not fix
```

<details><summary>คำใบ้ — ย่อหน้าสำหรับ GM</summary>

sandbox จำกัด **รัศมีความเสียหาย (blast radius)**: คือไฟล์ host, system call และ credential ที่เอเจนต์ซึ่งสับสนหรือถูกเจาะแล้วจะเข้าถึงได้ แต่มัน **ไม่ได้** ทำให้การให้เหตุผลของเอเจนต์ถูกต้อง และไม่ได้ทำให้เอเจนต์รอดพ้นจาก prompt injection ต้องพูดให้ครบทั้งสองด้าน

</details>

<details><summary>คำใบ้ — งาน Langfuse</summary>

มองหา harness ที่ plugin ด้าน observability ได้ key มาในรูป credential placeholder ของ OpenShell (`langfuse-hermes-v1`) แทนที่จะเป็นไฟล์ภายใน sandbox

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสามบรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `week26/.venv-nat/bin/nat: No such file or directory` | ยังไม่ได้สร้าง venv ของ NAT | รันคำสั่ง `uv` สองบรรทัดใน `week26/README.md` |
| `openshell --version` ไม่พิมพ์อะไรออกมาบนแล็ปท็อป | wheel ของ `openshell` รุ่นใหม่กว่า (0.0.116 / 0.1.2) มีแค่ Python SDK ไม่มี CLI | ติดตั้ง `openshell==0.0.111` ลงใน `week26/.venv-openshell` |
| 🪸 Reef ขึ้นว่า "no Spark configured" | `SPARK_HOST` ว่างอยู่ | เปิด 🖥 Spark setup หรือเรียนต่อในโหมด DRY |
| Lab 01-1 แสดง `◈ EXAMPLE` ในทุกแถวของ Spark | โหมด DRY: ไม่มีอะไรรันบน Spark | เชื่อมต่อ Spark แล้วสลับเป็น ⚡ Live |

## Next — บทถัดไป

[Lab 02 — Your first claw on DGX Spark](../02_first_claw/TUTORIAL.md): ตรวจสอบ Spark, ติดตั้ง NemoClaw ด้วยคำสั่งเดียว, onboard ผู้ช่วยที่รันใน sandbox บนโมเดลในเครื่อง และพิสูจน์ว่า inference ไม่เคยออกไปนอกเครื่อง
