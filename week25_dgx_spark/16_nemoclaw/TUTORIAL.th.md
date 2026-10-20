# ▶ Spark Lab 16 — NemoClaw: เอเจนต์ที่ทำงานตลอดเวลาภายใน sandbox

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เข้าใจว่าใครทำหน้าที่อะไร: **OpenClaw** คือเอเจนต์ **OpenShell** คือ sandbox และ **NemoClaw** คือตัวติดตั้งและ CLI ที่นำเอเจนต์ไปใส่ใน sandbox แล้วต่อเข้ากับ vLLM บนเครื่อง
- รัน **preflight** สิบข้อบน Spark ของคุณ จากนั้นติดตั้ง NemoClaw และ onboard เอเจนต์ใน sandbox ตัวแรก (`my-assistant`)
- เข้า Web UI ของเอเจนต์ผ่าน SSH tunnel และตรวจดูสถานะ policy และ log โดยไม่เปลี่ยนแปลงอะไรเลย
- สร้าง **network preset แบบให้สิทธิ์น้อยที่สุด (least-privilege)** บนแล็ปท็อป ตรวจสอบกับทุกกฎที่ playbook ระบุไว้ และนำไปใช้เฉพาะเมื่อคุณเลือกเอง
- ตรวจว่า model endpoint ตอบในแบบที่เอเจนต์ต้องการ: มีรายการโมเดล ตอบข้อความธรรมดาได้ และส่ง **tool call** ที่รูปแบบถูกต้อง
- ตั้งค่าเอเจนต์ตัวอย่างของ NVIDIA สามตัว (developer, deck reviewer, news digest) พร้อมขั้นตอน policy ตามจริงทุกขั้น

**Time** ~60 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บบนแล็ปท็อป)

**Playbook ทางการที่ครอบคลุม:** [Run NemoClaw with a Local LLM](https://build.nvidia.com/spark/nemoclaw) · [Set Up Example NemoClaw Agents](https://build.nvidia.com/spark/nemoclaw-applications)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จแล้ว: `ssh spark-a` ใช้ได้ด้วยคีย์ | `ssh -o BatchMode=yes spark-a true` | lab 16-1 และ 16-2 รันผ่าน SSH |
| Spark ที่ **สะอาด**: ไม่มีข้อมูลส่วนตัว ไม่มีบัญชีจริง ไม่มี credential ของ production | คุณตัดสินใจเอง | กฎความปลอดภัยข้อแรกของ playbook: "Use only a clean environment" |
| `sudo` บน Spark (แบบ interactive ก็ได้) | `sudo -v` บน Spark | ตัวติดตั้งต้องใช้สิทธิ์ root ในบางขั้นตอน |
| Docker 28.x ขึ้นไป โดยไม่ต้อง sudo | `docker ps` บน Spark | NemoClaw รัน gateway และ sandbox เป็น container |
| พื้นที่ดิสก์สำหรับโมเดล | `df -h ~` บน Spark | playbook เตือนว่าโมเดล Express ขนาดใหญ่อาจต้องใช้ **หลายร้อย GB** |
| Python ของ repo นี้ + Ollama บนแล็ปท็อป (ไม่บังคับ) | `.venv/bin/python --version`, `curl -s localhost:11434/api/tags` | lab 16-3 และ 16-4 รันบนแล็ปท็อป |

พื้นฐานที่มีประโยชน์: [Module 05](../05_vllm/TUTORIAL.md) (vLLM และ tool calling) และ [Module 15](../15_openshell_sandbox/TUTORIAL.md) (OpenShell แบบเดี่ยว ๆ) Week 23 Lab 06 อธิบายแนวคิดของ NemoClaw ด้วย policy gate แบบจำลอง ส่วนโมดูลนี้คือการติดตั้งจริงบน Spark

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
curl -s localhost:11434/api/tags | head -c 120; echo
```

> 🔐 **ความปลอดภัยมาก่อน ตลอดทั้งโมดูล** เอเจนต์ที่ทำงานตลอดเวลาและมี tool สามารถอ่านไฟล์ รันคำสั่ง และส่งข้อความได้ในขณะที่คุณไม่ได้เฝ้าดู playbook ระบุความเสี่ยงไว้สี่ข้อ: **data leakage (ข้อมูลรั่วไหล), malicious code execution (การรันโค้ดประสงค์ร้าย), unintended actions (การกระทำที่ไม่ได้ตั้งใจ), prompt injection** sandbox ช่วยลดความเสี่ยงเหล่านี้ แต่ไม่ได้ขจัดมันออกไป ในโมดูลนี้ไม่มีแล็บใดติดตั้ง messaging integration ให้สิทธิ์ หรือเปลี่ยน policy เว้นแต่คุณจะใส่ `--apply` ห้ามวางโทเค็นจริงลงในแชต prompt หรือบรรทัดคำสั่งเด็ดขาด

✓ Checkpoint: คุณมี Spark ที่ยอมใช้เป็นเครื่องทดลองได้ และ `ssh -o BatchMode=yes spark-a true` กลับมาโดยไม่ถามอะไร (หรือคุณตัดสินใจแล้วว่าจะเรียนตามแบบ DRY)

## 1 · ใครทำอะไร: OpenClaw, OpenShell, NemoClaw

สามชื่อ สามหน้าที่ playbook อธิบาย NemoClaw ว่าเป็น "an open-source reference stack that simplifies running OpenClaw always-on assistants more safely"

| ส่วนประกอบ | คืออะไร | คุณจะเจอมันในรูปแบบ |
|---|---|---|
| **OpenClaw** | ตัวเอเจนต์: แชต หน่วยความจำ (memory) tool, skill และตัวตั้งเวลาในตัว (cron) | Web UI ที่ `127.0.0.1:18789`, `openclaw tui`, `openclaw cron …` ภายใน sandbox |
| **OpenShell** | runtime ที่ครอบเอเจนต์ไว้ใน sandbox: แยกส่วน (isolation) ทั้ง filesystem เครือข่าย โปรเซส และ inference | `openshell policy get`, `openshell forward …`, `openshell term` บนเครื่อง host |
| **NemoClaw** | ตัวติดตั้ง (`nemoclaw.sh`) ตัวช่วย onboard และ CLI `nemoclaw` ที่สร้าง sandbox รัน OpenClaw ข้างใน และส่งการเรียกโมเดลไปยัง vLLM บนเครื่อง | `nemoclaw onboard`, `nemoclaw <name> status`, `policy-add`, `logs`, `uninstall` |

NemoClaw ยังรันเอเจนต์อีกสองตัวใน sandbox แบบเดียวกันได้ด้วย: **Hermes** (`NEMOCLAW_AGENT=hermes` หรือ `nemohermes onboard`) และ **LangChain Deep Agents Code** โมดูลนี้ใช้ค่าเริ่มต้นคือ OpenClaw ส่วน Module 17 รัน OpenClaw และ Hermes *โดยไม่มี* sandbox ให้คุณเปรียบเทียบได้

OpenShell บังคับใช้การแยกส่วนสี่ชั้น สองชั้นถูกล็อกตอนสร้าง sandbox อีกสองชั้นเปลี่ยนได้ระหว่างที่รันอยู่:

| ชั้น | ป้องกันอะไร | มีผลเมื่อไร |
|---|---|---|
| Filesystem | การอ่าน/เขียนนอกเส้นทาง (path) ที่อนุญาต | ล็อกตอนสร้าง sandbox (จะเปลี่ยนต้อง `nemoclaw <name> rebuild`) |
| Network | การเชื่อมต่อขาออกที่ไม่ได้รับอนุญาต | hot-reload ได้ระหว่างรัน (`policy-add` / `policy-remove`) |
| Process | การยกระดับสิทธิ์ (privilege escalation) และ syscall อันตราย | ล็อกตอนสร้าง sandbox |
| Inference | เปลี่ยนเส้นทางการเรียก model API ไปยัง backend ที่ควบคุมได้ | hot-reload ได้ระหว่างรัน |

(REFERENCE — ตารางนี้ยกมาจาก playbook) ชั้น inference คือเหตุผลที่เอเจนต์ไม่เคยเห็นพอร์ต vLLM ของคุณโดยตรง: ภายใน sandbox โมเดลอยู่ที่ **`inference.local`**

✓ Checkpoint: คุณบอกได้ว่าจะโทษส่วนประกอบไหนในสามตัวนี้ ถ้า (a) เอเจนต์ตอบได้แย่ (b) เอเจนต์เข้าถึงเว็บไซต์ที่ไม่ควรเข้า (c) `nemoclaw` ขึ้นว่า "command not found"

## 2 · Preflight: lab 16-1

playbook ขอให้ตรวจสามอย่างก่อนเริ่ม:

```bash
# on: spark
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
```

"Expected: Ubuntu 24.04 (or your platform's supported OS), a detected NVIDIA GPU, Docker 28.x+." (REFERENCE — ยกมาจาก playbook) starter prompt ของ playbook ไปไกลกว่านั้น: ต้องการให้ตรวจ Node.js ดิสก์ NemoClaw ที่มีอยู่แล้ว vLLM พอร์ตที่เกี่ยวข้อง และสิทธิ์ผู้ดูแลระบบด้วย Lab 16-1 รันการตรวจทั้งสิบข้อแบบอ่านอย่างเดียว และบอกว่าเส้นทาง onboarding แบบไหนเหมาะกับเครื่อง:

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_1_preflight.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ ถ้ามี Spark คุณจะได้ค่าของเครื่องคุณเองพร้อม ✓ / ✕)

```
│ check                              result     last line of output
│ ─────────────────────────────────  ─────────  ──────────────────────────────────────────────────
│ Operating system                   ◈ example  NAME="Ubuntu"
│ GPU visible                        ◈ example  NVIDIA GB10, 580.95.05
│ Docker 28.x+                       ◈ example  28.3.3
│ Node.js                            ◈ example  node not installed
│ Unified memory                     ◈ example  Mem:             119           9         102
│ Free disk                          ◈ example  /dev/nvme0n1p2  3.7T  412G  3.1T  12% /
│ Non-interactive sudo               ◈ example  passwordless sudo: no — the installer will ask for
│ Existing NemoClaw / OpenShell      ◈ example  (end of list)
│ Ports 8000 · 8080 · 18789 · 18790  ◈ example  none of 8000 8080 18789 18790 is listening
│ vLLM already on :8000              ◈ example  no server on :8000

▣ STEP 11 · which onboarding path fits this Spark?
◆ DRY: with a Spark you would see whether :8000 is free (Express Install) or already serving (Existing vLLM).
```

ทำไมต้องเป็นพอร์ตเหล่านี้: **8000** คือ vLLM (ถ้ามีอะไร serve อยู่แล้ว onboarding นำกลับมาใช้เป็น "Existing vLLM" ได้) **8080** คือ OpenShell gateway (container เก่าที่ค้างจองพอร์ตนี้อยู่เป็นสาเหตุล้มเหลวที่รู้กันดี) และ **18789/18790** คือ dashboard ของ OpenClaw

> 💡 ไม่มี Node.js ก็ไม่เป็นไร: ตัวติดตั้งจะติดตั้ง Node.js **22.16+** ให้ถ้าจำเป็น แต่ถ้ามี Node ที่เก่ากว่า 22.16 และตัวติดตั้งอัปเกรดไม่ได้ จะถือว่าล้มเหลว (ดู Troubleshooting)

✓ Checkpoint: ในโหมด LIVE ทุกแถวเป็น ✓ (หรือคุณรู้วิธีแก้ของทุก ✕) และคุณรู้แล้วว่าจะเลือก Express Install หรือนำ vLLM ที่มีอยู่มาใช้

## 3 · ติดตั้ง NemoClaw

การติดตั้งตาม playbook มีแค่คำสั่งเดียว รัน **บน Spark** มันติดตั้ง Node.js ถ้าจำเป็น ติดตั้ง OpenShell โคลน NemoClaw รุ่น last known good (LKG) คอมไพล์ CLI และสร้าง sandbox:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

การ pipe สคริปต์เข้า `bash` คือการรันโค้ดที่คุณยังไม่ได้อ่าน starter prompt ของ playbook เองก็บอกให้ "download and inspect the official installer" เมื่อไม่แน่ใจ คอร์สนี้แนะนำแบบนั้น: เป็นสคริปต์เดียวกัน แค่บันทึกลงไฟล์ก่อน:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh -o ~/nemoclaw.sh
less ~/nemoclaw.sh            # read what it will do; q to quit
bash ~/nemoclaw.sh
```

สิ่งที่จะเกิดต่อไป:

1. คุณจะเห็น **third-party software notice** (ประกาศซอฟต์แวร์ของบุคคลที่สาม) อ่านให้ดี การยอมรับหมายความว่าคุณรับผิดชอบส่วนประกอบเหล่านั้นเอง
2. บน DGX Spark ตัวติดตั้งอาจเสนอ **Express Install**: vLLM บนเครื่องที่ถูกจัดการให้ โมเดล Express ที่มีการดูแล sandbox ชื่อ `my-assistant` และ policy แบบ **Balanced** กด **Enter** (หรือ `Y`) เพื่อยอมรับ หรือ `n` เพื่อ onboard แบบกำหนดเอง (ส่วนที่ 4)
3. การดาวน์โหลดโมเดลเริ่มขึ้น อาจมีขนาดใหญ่ ส่วนนี้คือช่วงที่ใช้เวลานานที่สุด

ถ้าหลังจากนั้น `nemoclaw` ขึ้นว่า "command not found" ให้โหลดเชลล์ใหม่:

```bash
# on: spark
source ~/.bashrc
nemoclaw list
```

ถ้าจะข้ามคำถาม Express ไปเลย ให้ตั้ง `NEMOCLAW_NO_EXPRESS=1` ก่อนรันตัวติดตั้ง การตั้ง `NEMOCLAW_PROVIDER` ก็ข้าม Express เช่นกัน และจะใช้ provider นั้นแทน

✓ Checkpoint: `nemoclaw list` รันได้บน Spark และแสดง `my-assistant` (Express) หรือตัวติดตั้งกำลังรออยู่ที่ตัวช่วย onboard (แบบกำหนดเอง)

## 4 · Onboard: Express หรือกำหนดเอง

ถ้าคุณยอมรับ Express Install ไปแล้ว onboarding ได้รันเสร็จแล้ว ข้ามไปส่วนที่ 5 ได้เลย สำหรับการตั้งค่าแบบกำหนดเอง (หรือภายหลัง ใช้ `nemoclaw onboard` เพื่อเปลี่ยนการตั้งค่า) ตัวช่วยจะถามเก้าคำถาม:

| # | คำถาม | ตัวเลือกของคอร์สนี้ | ทำไม |
|---|---|---|---|
| 1 | Select your agent | `1` OpenClaw | ค่าเริ่มต้น Module 17 ครอบคลุม Hermes |
| 2 | Configuring inference | ตัวเลือกแบบ local | เก็บข้อมูลไว้บน Spark |
| 3 | Inference models | ตัวที่ตัวติดตั้งเลือกให้ หรือโมเดลใน vLLM ที่คุณรันอยู่ | ต้องรองรับ tool calling (ส่วนที่ 7) |
| 4 | Sandbox name | `my-assistant` | ทุกคำสั่งด้านล่างใช้ชื่อนี้ |
| 5 | Apply this configuration | `Y` | |
| 6 | Enable Brave Web Search | **No** ไปก่อน | เพิ่ม API key และทางออกเครือข่าย (egress) เพิ่มทีหลังได้ด้วยการ rebuild |
| 7 | Messaging channels | **No** | playbook บอกว่า: "Prefer No for a first local install" |
| 8 | Resource profiles | Enter (`No profile`) | |
| 9 | Policy presets | **Balanced** และยอมรับ preset ที่แนะนำ | เป็นระดับที่แนะนำ |

starter prompt ของ playbook ระบุ environment variable สำหรับการรันแบบสคริปต์ (non-interactive) ไว้ ตัวที่มีประโยชน์สำหรับ Spark ได้แก่:

| ตัวแปร | ความหมาย |
|---|---|
| `NEMOCLAW_PROVIDER=vllm` | ใช้ vLLM server ที่ **มีอยู่แล้ว** บน `localhost:8000` |
| `NEMOCLAW_PROVIDER=install-vllm` | ให้ NemoClaw ติดตั้งและจัดการ vLLM เอง |
| `NEMOCLAW_PROVIDER=ollama` (+ `NEMOCLAW_MODEL` ถ้าต้องการ) | Ollama บนเครื่อง |
| `NEMOCLAW_AGENT=hermes` | ใช้ Hermes แทน OpenClaw |
| `NEMOCLAW_NO_EXPRESS=1` | ข้ามคำถาม Express |
| `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1`, `NEMOCLAW_YES=1` | ยอมรับล่วงหน้า **เฉพาะหลังจาก** คุณอ่านสิ่งที่ยอมรับแล้ว |

เมื่อ onboarding เสร็จ คุณจะเห็นสรุปนี้:

**Expected output** (REFERENCE — ยกมาจาก playbook)

```text
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

คุณรัน onboarding ซ้ำเพื่อเพิ่ม sandbox ได้อีก มีสาม flag ที่หน้าตาคล้ายกันแต่ไม่เหมือนกัน:

| Flag | ผล | ใช้เมื่อ |
|---|---|---|
| `--name <new-name>` | sandbox **ใหม่** อยู่ข้าง ๆ ตัวที่มีอยู่ | ต้องการเอเจนต์ตัวที่สองสำหรับงานอื่น |
| `--recreate-sandbox` | rebuild sandbox ที่มีอยู่เพื่อฝังฟีเจอร์เพิ่ม (เช่น Brave Search) | เพิ่มฟีเจอร์ |
| `--fresh` | **ทำลายแล้วสร้างใหม่** sandbox ที่ชื่อเดียวกัน และทิ้งสถานะของตัวช่วย onboard | ใช้เฉพาะเพื่อกู้คืนจาก onboarding ที่ล้มเหลว |

```bash
# on: spark
nemoclaw onboard --gpu --name <new-name>
```

✓ Checkpoint: คุณเห็น "OpenClaw is ready" แล้ว และ `nemoclaw my-assistant status` รายงานสถานะ sandbox

## 5 · คุยกับเอเจนต์: Web UI และเทอร์มินัล

**Web UI** URL ของ dashboard มีโทเค็นสำหรับล็อกอินอยู่ในตัว พิมพ์มันออกมาบน Spark:

```bash
# on: spark
nemoclaw my-assistant dashboard-url --quiet
```

มันจะพิมพ์ URL หน้าตาแบบ `http://127.0.0.1:18790/#token=<token>` (REFERENCE — ตัวอย่างจาก playbook) พอร์ตถูกกำหนดอัตโนมัติ ส่วนใหญ่เป็น 18789 หรือ 18790 **ให้ถือว่า URL ทั้งเส้นเป็นรหัสผ่าน**: อย่าวางลงในแชตหรือ ticket

จากแล็ปท็อป ให้ forward พอร์ตนั้น (ใช้พอร์ตจาก URL ของคุณ) แล้วเปิด URL ในเบราว์เซอร์:

```bash
# on: laptop
ssh -N -L 18789:127.0.0.1:18789 spark-a
```

> ⚠ ใช้ `127.0.0.1` ไม่ใช่ `localhost` ในเบราว์เซอร์ การตรวจ origin ของ gateway ต้องตรงกันเป๊ะ ไม่อย่างนั้นคุณจะเจอ `origin not allowed`

**Terminal UI** เชื่อมต่อเข้าไปใน sandbox แล้วเปิด TUI; **Ctrl+C** ออกจาก TUI และ `exit` ออกจาก sandbox:

```bash
# on: spark
nemoclaw my-assistant connect
openclaw tui
```

ลอง prompt แรกที่ไม่มีพิษภัย เช่น "What model are you running on, and which tools can you use?"

> 💡 playbook ยังระบุ `nemoclaw tunnel start` ไว้ด้วย ซึ่งเป็น tunnel ของ cloudflared ที่ให้ Web UI มี URL **สาธารณะ** คอร์สนี้ไม่ใช้มัน: SSH tunnel ทำให้ dashboard เป็นของคุณคนเดียว

✓ Checkpoint: เอเจนต์ตอบคุณใน Web UI (ผ่าน tunnel) หรือใน `openclaw tui`

## 6 · sandbox และ policy ของมัน

ชั้นเครือข่ายเป็น **allow-list** ระดับ (tier) หนึ่ง (Balanced, Restricted, Open) ถูกเลือกตอน onboarding แล้ว **preset** จะเพิ่มกลุ่มของ host ซ้อนเข้าไป ห้าคำสั่งนี้ครอบคลุมทั้งหมด:

```bash
# on: spark
nemoclaw my-assistant policy-list                                # which presets are applied
openshell policy get my-assistant --full | grep -E "host:|port:" # every host the sandbox may reach
nemoclaw my-assistant policy-add                                 # interactive: add a maintained preset
nemoclaw my-assistant policy-add --from-file ./my-preset.yaml --yes   # add your own preset (hot-reload)
nemoclaw my-assistant policy-remove <preset> --yes                # take one away (hot-reload)
```

Lab 16-2 รันครึ่งที่อ่านอย่างเดียวของชุดนี้ พร้อม **boundary test (การทดสอบขอบเขต)** สองข้อที่ applications playbook ใช้: อินเทอร์เน็ตสาธารณะต้องถูกปฏิเสธ และ `inference.local` ต้องตอบ มันปิดบัง (mask) โทเค็นของ dashboard และทุกอย่างที่หน้าตาเหมือนโทเค็นใน log ก่อนพิมพ์

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_2_status_and_logs.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ บรรทัด 403 มาจาก playbook ที่เหลือเป็นรูปแบบ EXAMPLE)

```
▣ STEP 7 · boundary test 1 — the public internet is refused
$ nemoclaw my-assistant exec -- bash -lc 'curl -sS --max-time 5 https://example.com' 2>&1 | head -3   [DRY]
◈ REFERENCE — expected output from the NVIDIA playbook (not your machine):
curl: (56) CONNECT tunnel failed, response 403

▣ STEP 8 · boundary test 2 — the model route is allowed
$ nemoclaw my-assistant exec -- bash -lc 'curl -sf https://inference.local/v1/models' | head -c 300   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
{"object":"list","data":[{"id":"<your-selected-model>","object":"model"}]}

│ check                 source     what it shows
│ ────────────────────  ─────────  ────────────────────────────────────
│ nemoclaw list         example    — (dry: shape only)
│ status                example    — (dry: shape only)
│ policy-list           example    — (dry: shape only)
│ policy get --full     example    — (dry: shape only)
│ dashboard-url         reference  token masked: never paste it in chat
│ logs (tail 40)        example    — (dry: shape only)
│ curl example.com      reference  — (dry: shape only)
│ curl inference.local  example    — (dry: shape only)
```

ถ้า `example.com` **ไม่** ถูกปฏิเสธ แปลว่า sandbox มีทางออกเครือข่าย (egress) มากกว่าที่คุณคิด: รัน `policy-list` แล้วเอาสิ่งที่ไม่จำเป็นออก

**เขียน preset ของคุณเอง** เอเจนต์ News Digest (ส่วนที่ 8) ต้องอ่านเว็บข่าวสามแห่ง ไฟล์ preset ตาม playbook มีรูปแบบที่แน่นอน และ playbook ระบุความผิดพลาดที่ทำให้มันล้มเหลวไว้:

| ความผิดพลาด | สิ่งที่คุณจะเห็น |
|---|---|
| เขียน `network_policies` เป็น list ของ `{host, port}` | `invalid type: sequence, expected a map` |
| `preset.name` มีขีดล่าง (`news_sources`) | `Preset must declare preset.name (lowercase, hyphenated RFC 1123 label)` |
| endpoint มีแค่ `host` และ `port` (ไม่มี `access: full` + `tls: skip`) | `curl: (56) CONNECT tunnel failed, response 403` |
| ไม่มี list `binaries` ระบุว่าโปรแกรมไหนใช้ egress ได้ | ทุกการ fetch ได้ 403 |

(REFERENCE — ข้อความ error ยกมาจาก applications playbook มีรายละเอียดหนึ่งที่ playbook เขียนไม่ตรงกัน: หมายเหตุหนึ่งบอกว่า group key ต้องใช้ขีดกลางด้วย แต่ตาราง Troubleshooting บอกว่า group key ใช้ขีดล่างได้ ใช้ขีดกลางทุกที่แล้วจะผ่านทั้งสองแบบ)

Lab 16-3 สร้างไฟล์บนแล็ปท็อปของคุณ ตรวจสอบความถูกต้อง และพิสูจน์ว่าตัวตรวจ (validator) จับความผิดพลาดได้ครบทุกข้อ:

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_3_policy_preset.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ยังไม่ได้ตั้งค่า Spark)

```
▣ STEP 1 · generate news-sources.yaml (the playbook's shape)
preset:
  name: news-sources
  description: "Daily news digest source allowlist"

network_policies:
  news-sources:
    name: news-sources
    endpoints:
      - host: developer.nvidia.com
        port: 443
        access: full
        tls: skip
      - host: blogs.nvidia.com
        port: 443
        access: full
        tls: skip
      - host: news.ycombinator.com
        port: 443
        access: full
        tls: skip
    binaries:
      - { path: /usr/local/bin/openclaw }
      - { path: /usr/local/bin/node }
      - { path: /usr/bin/node }
      - { path: /usr/bin/curl }
◆ written to week25/16_nemoclaw/.runs/news-sources.yaml

▣ STEP 2 · parse it back and validate
✓ no findings: valid shape, least-privilege checks pass

▣ STEP 3 · the validator must catch every documented mistake (rows 1-4: playbook · 5-6: this course)
│ broken preset                       validator  first finding
│ ──────────────────────────────────  ─────────  ────────────────────────────────────────────────────
│ underscore in preset.name           ✓ caught   preset.name 'news_sources': must be a lowercase, hy…
│ list instead of a map               ✓ caught   network_policies is a list: 'invalid type: sequence…
│ bare {host, port} (no access mode)  ✓ caught   blogs.nvidia.com: no access mode → the proxy answer…
│ no binaries allow-list              ✓ caught   group 'news-sources' has no binaries allow-list → n…
│ a token pasted into the file        ✓ caught   a credential-shaped string is in the file — never p…
│ wildcard host + a shell binary      ✓ caught   *.example.com: wildcard host — name each site inste…

▣ STEP 4 · put it on the Spark and (only with --apply) add it to the sandbox
$ scp news-sources.yaml <spark>:~/w25/nemoclaw/news-sources.yaml   [DRY]
→ not applied (DRY — no Spark). On the Spark, run:
$ nemoclaw my-assistant policy-add --from-file ~/w25/nemoclaw/news-sources.yaml --yes
$ openshell policy get my-assistant --full | grep -E "host:|port:"
◆ Undo later with: nemoclaw <sandbox> policy-remove news-sources --yes  (network changes hot-reload, no rebuild)

✓ news-sources.yaml is valid and least-privilege
✓ every documented mistake was caught
```

list `binaries` เป็นส่วนหนึ่งของหลัก least privilege: มันระบุว่า **โปรแกรมไหน** ใช้ tunnel ได้ playbook ใส่ `/usr/bin/curl` ไว้เพื่อให้การ fetch จากเชลล์ใช้ได้ แล็บจะเตือนถ้าคุณเพิ่มเชลล์อย่าง `/bin/bash` เพราะนั่นทำให้สคริปต์ใดก็ได้ใน sandbox ใช้ egress นั้นได้

**การเปลี่ยน filesystem ต่างออกไป** network preset hot-reload ได้ แต่ filesystem policy ถูกล็อกตั้งแต่ตอนสร้าง: ถ้าจะทำให้ไดเรกทอรีหนึ่งเป็นแบบอ่านอย่างเดียวในระดับ kernel คุณต้องเพิ่มมันลงใน `read_only` ของ `filesystem_policy` ของ sandbox แล้วรัน `nemoclaw my-assistant rebuild` (สถานะของ workspace ยังอยู่ครบ) ถ้าจะอนุมัติหรือปฏิเสธคำขอเครือข่ายที่ถูกบล็อกแบบสด ๆ ให้ใช้ `openshell term` บนเครื่อง host

✓ Checkpoint: lab 16-3 จบด้วยบรรทัด ✓ สองบรรทัด และคุณอธิบายได้ว่าทำไม entry แบบ `{host, port}` เปล่า ๆ "apply ผ่านเรียบร้อย" แต่ยังได้ 403

## 7 · ตรวจเส้นทางไปยังโมเดล: lab 16-4

เอเจนต์ดีได้แค่เท่ากับ **tool call** ของโมเดล ถ้าโมเดลไม่เคยส่ง `tool_calls` กลับมา tool และ skill ของ OpenClaw จะไม่ทำงานเลย และคุณจะได้แค่แชตบอต Lab 16-4 ตรวจ endpoint ที่ NemoClaw ส่งคำขอไป: vLLM ที่ `:8000` บน Spark ถ้ามันตอบ ไม่อย่างนั้นใช้ Ollama บนแล็ปท็อปเป็นตัวแทน (stand-in) ที่ติดป้ายไว้ แล็บใช้ prompt ทดสอบเบื้องต้น (smoke test) ของ playbook เอง "Reply with exactly: READY" แล้วตามด้วย tool หนึ่งตัว

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/labs/lab16_4_inference_route.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: LAPTOP STAND-IN ความเร็วของแล็ปท็อป ไม่ใช่ตัวเลขของ Spark)

```
◆ endpoint: http://localhost:11434/v1 · Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark

▣ STEP 1 · GET /v1/models
→ GET http://localhost:11434/v1/models
· nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b, kimi-k2.7-code:cloud, gemma4:12b, gemma4:latest …

▣ STEP 2 · a plain answer (the playbook's smoke test prompt)
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  READY
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 9169 ms · 2 tok in 9.2s · 77.0 tok/s

▣ STEP 3 · one tool, one question — does the model emit a tool call?
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER
→ tool_call get_sandbox_status({"sandbox":"my-assistant"})
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 17379 ms · 19 tok in 17.4s · 1.1 tok/s

│ check          result  detail
│ ─────────────  ──────  ──────────────────────────────────────────────
│ models listed  ✓       7 model(s)
│ plain answer   ✓       READY
│ tool call      ✓       get_sandbox_status({"sandbox":"my-assistant"})

✓ this endpoint can drive an agent (LAPTOP STAND-IN)
```

บน Spark ที่มี vLLM server ทำงานอยู่ แล็บเดียวกันจะรายงาน `vLLM on your Spark (:8000)` ถ้าคุณรัน vLLM ของตัวเองให้ NemoClaw ("Existing vLLM", `NEMOCLAW_PROVIDER=vllm`) ให้เปิดมันพร้อมเปิดใช้ tool calling: สำหรับ Qwen3.6 นั้น vLLM playbook ใช้ `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` (Module 05 และ Module 17 ส่วนที่ 2 แสดงคำสั่งฉบับเต็ม) ส่วน vLLM ที่ Express Install จัดการให้ทำเรื่องนี้ให้แล้ว ภายใน sandbox การตรวจตาม playbook คือ `curl -sf https://inference.local/v1/models`

✓ Checkpoint: ทั้งสามแถวเป็น ✓ และคุณบอกได้ว่าทำไม TTFT และ tok/s ของแล็ปท็อปไม่ได้บอกอะไรเกี่ยวกับ Spark เลย

## 8 · เอเจนต์ตัวอย่างสามตัวจาก applications playbook

playbook คู่กันของ NVIDIA มีเอเจนต์พร้อมรันสี่ตัว แต่ละตัวมี **การตั้งค่า policy** มี **agent prompt** มาตรฐาน (canonical) ยาว ๆ ที่คุณวางลงใน Web UI และมีตาราง **knob (ปุ่มปรับ)** สำหรับปรับให้เข้ากับตัวคุณ บทเรียนอยู่ที่ว่าแต่ละตัวได้รับอนุญาตให้ทำอะไรบ้าง:

| เอเจนต์ | เครือข่ายนอกเหนือจาก inference | ไฟล์ | Telegram |
|---|---|---|---|
| Software Development Agent | ไม่มี | สำเนาโปรเจกต์ที่ `/sandbox/project` (อ่าน-เขียนได้) | ไม่บังคับ (แจ้ง "ready") |
| Deck Reviewer (Doc & Deck Red-Team) | ค่าเริ่มต้นไม่มี | `queue/`, `corpus/` อ่านอย่างเดียว; `reports/`, `memory/` เขียนได้ | ไม่บังคับ |
| Calendar Negotiator | ไม่มีในโหมด propose-only | `calendar.ics`, `profile.yaml` อ่านอย่างเดียว; `bookings/` เขียนได้ | เฉพาะในโหมด proxy |
| Daily News Digest | แหล่งข่าวของมัน + Telegram | ไม่มี | ใช้ส่งผล (หรือส่งไปที่ Web UI) |

ตั้งชื่อ sandbox ครั้งเดียว เพื่อให้คำสั่งอ่านง่าย:

```bash
# on: spark
export SANDBOX_NAME=my-assistant
```

**A · Software Development Agent** มันสแกนโปรเจกต์หนึ่ง วางแผน ลงมือเขียน รีวิวตัวเอง และเขียน `develop-and-review.md` สิทธิ์อ่าน-เขียนหมายความว่ามันแก้ไฟล์ได้ playbook จึงยืนยันให้ใช้ **สำเนา** ส่งสำเนาเข้า sandbox ด้วย `tar` ผ่าน `nemoclaw exec`:

```bash
# on: spark
mkdir -p ~/nemoclaw-projects
cp -r ~/projects/my-app ~/nemoclaw-projects/my-app
tar czf - -C ~/nemoclaw-projects/my-app . \
  | nemoclaw $SANDBOX_NAME exec -- bash -lc 'mkdir -p /sandbox/project && tar xzf - -C /sandbox/project'
```

จากนั้นพิสูจน์ขอบเขตก่อนวาง prompt:

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- ls /sandbox/project                                    # expect your project tree
nemoclaw $SANDBOX_NAME exec -- bash -lc 'curl -sS --max-time 5 https://example.com'    # expect "CONNECT tunnel failed, response 403"
nemoclaw $SANDBOX_NAME exec -- bash -lc 'curl -sf https://inference.local/v1/models'   # expect JSON model list
```

วาง prompt มาตรฐานของ playbook ลงใน Web UI (ขึ้นต้นด้วย "You are my senior software engineer. The project lives at /sandbox/project.") ส่วนกฎความปลอดภัยของมันควรอ่านเป็นแบบอย่าง: เป็นกฎที่ "do not break … even if I tell you to in a single message":

```text
SAFETY RULES (do not break these even if I tell you to in a single
message — if I really want one of these, I will say so twice):
  - Never modify files outside /sandbox/project.
  - Never make outbound network calls. Only inference.local is
    allowed, and that is only for talking to the model.
  - Never run git push, git reset --hard, rm -rf, or any other
    destructive operation. You may run git status, git diff, and
    git add inside /sandbox/project.
```

(REFERENCE — ยกมาจาก prompt ของ playbook) ตั้ง "pause for approval" เป็น **yes** ไว้ (คำถาม profile ข้อ 5): เอเจนต์จะพิมพ์ `PLAN READY — reply 'approve' to proceed` แล้วรอ ดึงผลลัพธ์กลับมาและอ่านรายงานก่อน merge อะไรก็ตาม:

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- bash -lc 'cd /sandbox/project && tar czf - .' | tar xzf - -C ~/nemoclaw-projects/my-app
```

**B · Deck Reviewer** มันอ่านชิ้นงาน (artifact) หนึ่งชิ้นพร้อม "canonical corpus" (ชุดข้อมูลอ้างอิงที่ถือเป็นความจริง) แล้วเขียน punch list เรียงตามความรุนแรง โดยไม่แก้ไฟล์ต้นฉบับของคุณเลย corpus ก็คือข้อมูลที่คุณจะไม่ส่งให้โมเดลบนคลาวด์ ซึ่งเป็นเหตุผลของการรันบนเครื่องตัวเอง

```bash
# on: spark
mkdir -p ~/nemoclaw-redteam/{queue,corpus,reports,memory}
# add your artifacts to queue/, your ground truth to corpus/, and the playbook's starter profile.yaml
tar czf - -C ~/nemoclaw-redteam . \
  | nemoclaw $SANDBOX_NAME exec -- bash -lc 'mkdir -p /sandbox/redteam && tar xzf - -C /sandbox/redteam'
nemoclaw $SANDBOX_NAME exec -- bash -lc 'chmod -R a-w /sandbox/redteam/queue /sandbox/redteam/corpus /sandbox/redteam/profile.yaml && chmod -R u+w /sandbox/redteam/reports /sandbox/redteam/memory'
```

ตรวจขอบเขตทั้งสองด้าน: เขียนลง `reports/` ได้ เขียนลง `queue/` ไม่ได้

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/redteam/reports/.write-check && rm /sandbox/redteam/reports/.write-check && echo OK reports'
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/redteam/queue/.write-check 2>&1 | head -1'   # expect "Permission denied"
nemoclaw $SANDBOX_NAME exec -- bash -c 'curl -sS --max-time 5 https://example.com'                        # expect "CONNECT tunnel failed, response 403"
```

> ⚠ playbook บอกชัดว่า `chmod` นี้เป็นขอบเขตแบบ **อ่อน (soft)**: เอเจนต์รันเป็นผู้ใช้ `sandbox` ซึ่งเป็นเจ้าของไฟล์ และสามารถ `chmod` กลับได้ ถ้าต้องการขอบเขตที่ kernel บังคับใช้ ให้เพิ่ม path ลงใน `read_only` ของ `filesystem_policy` แล้วรัน `nemoclaw $SANDBOX_NAME rebuild`

ดึง punch list กลับมาด้วย `nemoclaw $SANDBOX_NAME exec -- bash -lc 'cd /sandbox/redteam && tar czf - reports memory' | tar xzf - -C ~/nemoclaw-redteam`

**C · Daily News Digest** สรุปข่าวตามเวลาที่ตั้งไว้ เป็นตัวเดียวในสามตัวที่ต้องมีทางออกอินเทอร์เน็ต ซึ่งเป็นเหตุผลที่คุณสร้างและตรวจ preset ของมันใน lab 16-3 คัดลอกไฟล์จากแล็บ (หรือเขียนเอง) apply แล้วตรวจ:

```bash
# on: spark
nemoclaw $SANDBOX_NAME policy-add --from-file ~/w25/nemoclaw/news-sources.yaml --yes
openshell policy get $SANDBOX_NAME --full | grep -E "host:|port:"
```

วาง prompt ของ playbook (ขึ้นต้นด้วย "You are my personal news intelligence analyst.") ถ้าไม่ใช้ Telegram ให้แทนบรรทัดเรื่องการส่งผลด้วย `Deliver each briefing to the web UI (this session). Do not use any messaging channel.`

playbook เตือนว่าการเรียกตัวตั้งเวลาจากตัวเอเจนต์เองอาจถูกปฏิเสธเพราะขาด scope และแนะนำให้ลงทะเบียนงาน **จากฝั่งผู้ดูแล (operator)**:

```bash
# on: spark
nemoclaw $SANDBOX_NAME exec -- openclaw cron add \
  --name news-digest --cron "0 8 * * 1-5" --tz America/Los_Angeles \
  --agent default --session-key agent:default:news-digest \
  --message "Run my daily news briefing now and write it to this session." \
  --no-deliver --token ""
nemoclaw $SANDBOX_NAME exec -- openclaw cron list
```

> ⚠ ข้อจำกัดที่รู้กันอยู่ (จาก playbook): การรัน **ตามเวลาที่ตั้งไว้** บนโมเดล local อาจถูก `skipped` เพราะการค้นหา DNS ล่วงหน้าของ `inference.local` ล้มเหลว (`getaddrinfo EAI_AGAIN inference.local`) การคุยสด ๆ ไม่ได้รับผลกระทบ วิธีเลี่ยงของ playbook คือส่ง endpoint ที่ resolve DNS ได้ (`host.openshell.internal:8000` ซึ่ง preset `local-inference` อนุญาตไว้) ให้ `cron add` ด้วย `--model` ทดสอบแบบครั้งเดียวก่อน: "Run the digest task now as a one-off, then keep the schedule for tomorrow."

เอเจนต์ตัวที่สี่ **Calendar Negotiator** ใช้รูปแบบเดียวกับ Deck Reviewer (`~/nemoclaw-calendar/` โดย `calendar.ics` อ่านอย่างเดียว และ `bookings/` เขียนได้) ให้รันในโหมด **propose-only** เพื่อไม่ให้มันส่งข้อความเองเลย

✓ Checkpoint: สำหรับเอเจนต์ทุกตัวที่คุณตั้งค่า การทดสอบ `example.com` ถูกปฏิเสธ และการทดสอบการเขียนตรงกับตารางด้านบน

## 9 · ไม่บังคับ: ช่องทางส่งข้อความ (Telegram)

ข้ามส่วนนี้ไปได้ ถ้าไม่ต้องการใช้เอเจนต์จากมือถือ ช่องทาง (channel) คือทางเข้าที่สอง: ใครก็ตามที่ส่งข้อความหาบอตได้ก็สั่งเอเจนต์ของคุณได้ จึงควรถือว่าเป็นสิทธิ์ที่คุณมอบให้

1. สร้างบอตกับ [@BotFather](https://t.me/BotFather) (`/newbot`) **bot token** ที่ได้กลับมาเป็น credential
2. ลงทะเบียน channel NemoClaw จะถามโทเค็น (อย่าใส่ไว้ในบรรทัดคำสั่ง) แล้ว rebuild sandbox:

```bash
# on: spark
nemoclaw my-assistant channels add telegram
```

3. ตัวช่วยจะถาม **Telegram User ID** (ไม่บังคับ) ให้ใส่ของคุณ (ส่ง `/start` ไปที่ `@userinfobot` เพื่อหา) ถ้าข้ามไป บอตจะต้องจับคู่อุปกรณ์ (device pairing) ก่อนจึงจะตอบ
4. ถ้าข้อความส่งไม่ผ่านด้วย error เรื่องเครือข่ายหรือ policy ให้เพิ่ม egress preset:

```bash
# on: spark
nemoclaw my-assistant policy-list
nemoclaw my-assistant policy-add telegram
```

Telegram ใช้ **long-polling**: sandbox ดึงข้อความจากเซิร์ฟเวอร์ของ Telegram เอง จึง **ไม่ต้องมี URL สาธารณะและไม่ต้องมี cloudflared tunnel** `policy-add telegram` อย่างเดียวไม่ได้ลงทะเบียน channel ถ้าไม่มี `channels add` บอตจะตอบว่า `Error: Channel is unavailable: telegram`

✓ Checkpoint: คุณอธิบายได้ว่าทำไม User ID ที่อนุญาตจึงสำคัญ และทำไมบอต Telegram จึงไม่ต้องเปิดพอร์ตขาเข้าบน Spark

## 10 · อัปเดต หยุด และถอนการติดตั้ง

```bash
# on: spark
nemoclaw update --check              # is a newer LKG release available?
nemoclaw update --yes                # update the host CLI (does not rebuild sandboxes)
nemoclaw upgrade-sandboxes --check   # which sandboxes are stale after an update
```

หยุดการเข้าถึงสาธารณะและการ forward:

```bash
# on: spark
nemoclaw tunnel stop
openshell forward list
openshell forward stop <port>
```

ตัวถอนการติดตั้งในตัวจะลบ sandbox, OpenShell gateway, container/image/volume ของ Docker ที่เป็นของ NemoClaw, CLI และไดเรกทอรีสถานะ แต่จะเก็บ Docker, Node.js, npm และ image ของ vLLM ไว้ และเก็บข้อมูลผู้ใช้ใน `~/.nemoclaw/` ไว้ เว้นแต่คุณจะสั่ง:

```bash
# on: spark
nemoclaw uninstall --yes
```

| Flag | ผล |
|---|---|
| `--yes` | ข้ามคำถามยืนยัน |
| `--keep-openshell` | เก็บ binary `openshell` ไว้ |
| `--delete-models` | ลบ model weights ที่ NemoClaw ดึงมาด้วย |
| `--destroy-user-data` | ลบ `~/.nemoclaw/` ด้วย (`rebuild-backups/`, `backups/`, `sandboxes.json`) |

(REFERENCE — flag ยกมาจาก playbook) การถอนการติดตั้งย้อนกลับยาก ไม่มีแล็บใดในคอร์สนี้รันคำสั่งนี้

✓ Checkpoint: คุณรู้ว่าคำสั่งไหนอัปเดต CLI และคำสั่งไหนบอกว่า sandbox ต้อง rebuild

## Labs — รันแล็บได้ที่นี่

**labs/lab16_1_preflight.py** — การตรวจความพร้อมแบบอ่านอย่างเดียวสิบข้อบน Spark ของคุณ และบอกว่าเส้นทาง onboarding แบบไหน (Express หรือ Existing vLLM) เหมาะสม

**labs/lab16_2_status_and_logs.py** — ดูสถานะ preset host ใน policy ปัจจุบัน การ forward และ log ช่วงสั้น ๆ แบบอ่านอย่างเดียว พร้อม boundary test สองข้อ โดยปิดบังโทเค็นไว้

**labs/lab16_3_policy_preset.py** — สร้าง network preset ของ News Digest บนแล็ปท็อป ตรวจกับกฎของ playbook และ apply เฉพาะเมื่อใส่ `--apply`

**labs/lab16_4_inference_route.py** — ตรวจว่า model endpoint แสดงรายการโมเดล ตอบได้ และส่ง tool call ที่รูปแบบถูกต้อง (vLLM บน Spark หรือตัวแทนบนแล็ปท็อป)

Lab 16-1 และ 16-2 รันแบบ LIVE บน Spark ของคุณหรือแบบ DRY ส่วน Lab 16-3 และ 16-4 รันจริงบนแล็ปท็อป โดย 16-3 จะคัดลอกไฟล์ไปที่ Spark เมื่อตั้งค่าไว้

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 16 — preset แบบ least-privilege และตัวตรวจของมัน** เปิด `week25/16_nemoclaw/exercises/ex16_policy_preset.py` ในไฟล์มี `TODO` สี่จุด:

1. `is_rfc1123(name)`: ชื่อ preset เป็น label แบบ RFC 1123 ที่เป็นตัวพิมพ์เล็กคั่นด้วยขีดกลางหรือไม่?
2. `endpoint(host)`: HTTPS endpoint หนึ่งตัวพร้อม access mode ที่ proxy ต้องการ
3. `build_preset(name, description, hosts, binaries)`: preset ทั้งก้อนในรูป dict
4. `errors(doc)`: จับความล้มเหลวสี่แบบที่ playbook ระบุไว้

```bash
# on: laptop
.venv/bin/python week25/16_nemoclaw/exercises/ex16_policy_preset.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสี่จุด บันทึกจาก Mac เครื่องนี้ด้วยเฉลยอ้างอิง)

```
✓ is_rfc1123: news-sources ✓ · news_sources ✕ · News ✕ · -news ✕ · 64 chars ✕
✓ endpoint: host + port 443 + access full + tls skip
✓ build_preset: a map keyed by group, 3 endpoints, 4 binaries
✓ errors: valid → [] · catches name · map · access · binaries

▣ your preset, as the file you would pass to `nemoclaw <sandbox> policy-add --from-file`
│ preset:
│   name: news-sources
│   description: Daily news digest source allowlist
│ network_policies:
│   news-sources:
│     name: news-sources
│     endpoints:
│     - host: developer.nvidia.com
│       port: 443
│       access: full
│       tls: skip
…
│     binaries:
│     - path: /usr/local/bin/openclaw
│     - path: /usr/local/bin/node
│     - path: /usr/bin/node
│     - path: /usr/bin/curl
```

<details><summary>คำใบ้ — ทำไม entry แบบ {host, port} เปล่า ๆ จึงได้ 403?</summary>

egress proxy ต้องรู้ว่าจะส่งผ่าน traffic *อย่างไร* `access: full` คู่กับ `tls: skip` คือ tunnel แบบส่งผ่านตรง ๆ (raw pass-through) ถ้าไม่มี access mode (หรือทางเลือกระดับ L7 คือ `protocol: rest` + `enforcement: enforce` + `rules`) proxy จะไม่มีกฎให้ใช้ จึงปฏิเสธ CONNECT แม้ว่า host จะอยู่ในรายการ

</details>

<details><summary>ท้าทายเพิ่ม — ตัวตรวจ URL สำหรับ Deck Reviewer</summary>

knob ของ Deck Reviewer ใน playbook มี preset "URL verification" (ไม่บังคับ) ที่ให้เอเจนต์ตรวจลิงก์ด้วย HEAD บน host ไม่กี่แห่ง สร้างมันด้วย `build_preset("url-check", …, ["build.nvidia.com"], ["/usr/local/bin/openclaw", "/usr/local/bin/node"])` ของคุณ ทำไมครั้งนี้คุณอาจไม่ใส่ `/usr/bin/curl`?

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสี่เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `nemoclaw: command not found` หลังติดตั้ง | `source ~/.bashrc` หรือเปิดเทอร์มินัลใหม่ |
| ตัวติดตั้งล้มเหลวด้วย error เรื่องเวอร์ชัน Node.js | Node.js ต้องเป็น 22.16+ วิธีแก้ของ playbook: `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt-get install -y nodejs` แล้วรันตัวติดตั้งใหม่ |
| npm ล้มเหลวด้วย `EACCES` | `mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global && export PATH=~/.npm-global/bin:$PATH` แล้วรันใหม่ |
| Gateway: "port 8080 is held by container…" หรือสร้าง sandbox ไม่สำเร็จ | รัน `nemoclaw onboard` (หรือ `nemoclaw onboard --resume`) อีกครั้ง มันจะนำ gateway ที่ยังดีอยู่กลับมาใช้ และสร้างสถานะที่ค้างเก่าใหม่เมื่อปลอดภัย |
| Gateway ล้มเหลวด้วย cgroup / "Failed to start ContainerManager" | รันตัวติดตั้งใหม่เพื่อให้ได้ OpenShell รุ่นใหม่กว่าก่อน ทางสำรองของ playbook คือแก้ `daemon.json` ด้วย `default-cgroupns-mode: host` (ดู playbook) |
| "No GPU detected" ระหว่าง onboard | เป็นเรื่องปกติบนบางแพลตฟอร์มที่ใช้ unified memory ตัวช่วยยังใช้ vLLM ได้ |
| Inference ค้างหรือหมดเวลา | `curl http://127.0.0.1:8000/v1/models` บน Spark ควรแสดงโมเดลของคุณ รอจนเห็น `Application startup complete` แล้วตรวจ `nemoclaw my-assistant status` |
| Web UI ขึ้นว่า `origin not allowed` | เปิด `http://127.0.0.1:<port>/#token=…` ไม่ใช่ `localhost` |
| การ forward ของ Web UI หยุดทำงาน | `openshell forward stop 18789 my-assistant` แล้ว `openshell forward start 18789 my-assistant --background` |
| `policy-add --from-file` ล้มเหลวด้วย `Preset must declare preset.name …` | ใช้ขีดกลาง ไม่ใช่ขีดล่าง ใน `preset.name` (lab 16-3 จับกรณีนี้ได้) |
| host อยู่ใน policy แล้วแต่ fetch ยังได้ 403 | endpoint ไม่มี access mode หรือกลุ่มไม่มี `binaries` (lab 16-3 จับได้ทั้งสองกรณี) |
| `openshell policy set` ปฏิเสธด้วย `unknown field 'Version'` | ใช้วิธีเพิ่มทีละส่วนด้วย `policy-add --from-file` แทน หรือ `sed -i 's/^Version:/version:/' policy.yaml` แล้วลองใหม่ |
| digest ตามเวลาไม่ทำงานเลย / ขึ้นว่า `skipped` | ลงทะเบียนงานด้วย `openclaw cron add` จากฝั่ง operator (ส่วนที่ 8) สำหรับโมเดล local ดูหมายเหตุเรื่อง DNS ของ `inference.local` |
| บอต Telegram ตอบว่า `Error: Channel is unavailable: telegram` | `policy-add telegram` อย่างเดียวไม่พอ: รัน `nemoclaw <name> channels add telegram` |
| หน่วยความจำตึงทั้งที่ยังไม่เกินความจุ | playbook ให้ล้าง page cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |

## Next — บทถัดไป

ไปต่อที่ [Lab 17 — OpenClaw และ Hermes Agent กับ LLM บนเครื่อง](../17_openclaw_hermes/TUTORIAL.md): รันเอเจนต์ทั้งสองตัวตรง ๆ บน Spark กับ vLLM บนเครื่อง เปรียบเทียบวิธีตั้งค่าและวิธีเรียก tool ของแต่ละตัว และทดสอบความสามารถด้าน tool calling ของ endpoint ของคุณ
