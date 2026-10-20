# ▶ Reef Lab 02 — Claw ตัวแรกบน DGX Spark: ตรวจเครื่อง ติดตั้ง onboard และพิสูจน์ว่า inference อยู่ในเครื่อง

> ส่วนหนึ่งของ Week 26 · NemoClaw บน DGX Spark ตั้งแต่มือใหม่จนถึงระดับผู้เชี่ยวชาญ คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง แล็บฝั่งแล็ปท็อป (NAT CLI, OpenShell CLI, policy model) รันได้จริงทุกที่ แล็บฝั่ง Spark รันในโหมด **DRY** ได้ด้วย (ไม่มี Spark ก็ได้ ไม่เสียเงิน): ระบบจะแสดงคำสั่ง ส่วนผลลัพธ์จะเป็น RECORDED ที่บันทึกจาก Spark จริง, REFERENCE ที่ยกมาจากเอกสารหรือ playbook ของ NVIDIA หรือ EXAMPLE ที่ติดป้ายไว้ชัดเจน

**สิ่งที่คุณจะได้ลงมือทำ**
- ตรวจว่า Spark พร้อม: DGX OS, มี GB10, ใช้ Docker ได้โดยไม่ต้อง sudo, kernel ใหม่พอสำหรับ Landlock และมีดิสก์เหลือพอ
- ติดตั้ง NemoClaw ด้วยคำสั่งเดียว และรู้ว่า Express Install เลือกอะไรให้คุณ
- เขียนบรรทัดติดตั้งแบบ non-interactive ที่ส่งต่อให้ Spark เครื่องที่สองได้ และตรวจมันก่อนที่ใครจะเอาไปวาง
- รู้จักคำสั่ง lifecycle ว่าตัวไหนแล็บรันให้ได้ ตัวไหนต้องรอ 🔓 และตัวไหนเป็นความลับ
- พิสูจน์ด้วยหลักฐานสามชิ้นว่า inference ของ claw ไม่ออกนอกเครื่อง
- เพิ่มช่องทาง Telegram และติดตั้งแบบ Hermes กับ Deep Agents

**Time** ~75 นาที · **Difficulty** beginner · **Hardware** DGX Spark 1 เครื่อง (ถ้าไม่มี: DRY + แล็ปท็อป)

**แหล่งอ้างอิง:** research tutorial ของคอร์ส *NemoClaw on DGX Spark — Beginner to Expert* (Part 1, แล็บ L1.1–L1.7) ซึ่งอ้างถึง [DGX Spark NemoClaw playbook](https://build.nvidia.com/spark/nemoclaw/overview) · [NemoClaw OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart) · [NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart) · [NemoClaw Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/user-guide/deepagents/get-started/quickstart) · [NemoClaw network policies](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies) · [NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [NVIDIA Technical Blog — NemoClaw + OpenClaw on Spark](https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/)

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจอย่างไร | ทำไม |
|---|---|---|
| DGX Spark ที่ล้างเครื่องได้ | DGX OS ใหม่ ไม่มีข้อมูลส่วนตัว | playbook บอกให้รันเดโมบนเครื่องหรือ VM ใหม่ที่ไม่มีข้อมูลส่วนตัวหรือข้อมูลลับ |
| SSH เข้าเครื่องได้ | `ssh -o BatchMode=yes <spark> true` | แล็บรันคำสั่งแบบอ่านอย่างเดียวที่นั่น ตั้งค่าได้ใน 🖥 Spark setup |
| `sudo` บน Spark | คุณรู้รหัสผ่าน | ตัวติดตั้งและการแก้ Docker ต้องใช้ root — **คุณ** พิมพ์เอง แล็บไม่ทำแทน |
| OpenShell parser บนแล็ปท็อป | `week26/.venv-openshell/bin/openshell --version` | แล็บ 02-3 และ 02-4 ตรวจทุกคำสั่ง OpenShell แบบออฟไลน์ |

```bash
# on: laptop
week26/.venv-openshell/bin/openshell --version
bash --version | head -n 1
```

**Expected output** (captured on this Mac) [บันทึกจาก Mac เครื่องนี้]

```
openshell 0.0.111
GNU bash, version 3.2.57(1)-release (arm64-apple-darwin26)
```

> ⚠ **NemoClaw ยังเป็น alpha** research tutorial พูดไว้ชัด: เป็น reference stack แบบ Apache-2.0 สถานะ alpha และให้รายงานปัญหาความปลอดภัยที่ psirt@nvidia.com agent ที่ทำงานตลอดเวลาอาจเข้าถึงอะไรได้กว้าง sandbox ช่วยลดความเสี่ยงนั้น แต่ไม่ได้ทำให้หายไป (Module 01)

ในโมดูลนี้มีเครื่องอยู่สามที่ แยกให้ออกตั้งแต่แรก:

| ที่ไหน | อะไรรันที่นั่น | เข้าถึงอย่างไร |
|---|---|---|
| **แล็ปท็อปเครื่องนี้** | Reef runner, แล็บ, OpenShell *parser*, `bash -n` | คุณอยู่ที่นี่ |
| **Spark host** | `nemoclaw`, `openshell`, Docker, gateway, vLLM หรือ Ollama | ⌨ terminal ที่มี `# on: spark` หรือแล็บผ่าน SSH |
| **ใน sandbox** | harness (OpenClaw / Hermes / Deep Agents), `inference.local` | `nemoclaw <s> connect` หรือ `openshell sandbox exec -n <s> -- …` |

✓ Checkpoint: parser บนแล็ปท็อปพิมพ์เวอร์ชันออกมา และคุณบอกได้ว่าแต่ละคำสั่งในโมดูลนี้รันที่ไหนในสามที่นี้

## 1 · L1.1 — ตรวจ Spark

ก่อนติดตั้งอะไร ให้ยืนยันก่อนว่าเครื่องเป็นอย่างที่คิด playbook ใช้สามคำสั่ง:

```bash
# on: spark
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
```

**Expected output** (REFERENCE — quoted from the DGX Spark NemoClaw playbook) [ยกมาจาก playbook]

```
Expected: Ubuntu 24.04 (or your platform's supported OS), a detected NVIDIA GPU, Docker 28.x+.
```

ส่วนที่เหลือของคอร์สต้องรู้อีกสี่เรื่อง แล็บ 02-1 จึงตรวจให้ด้วย:

| ตรวจ | คำสั่ง | ผ่านเมื่อ | ทำไม |
|---|---|---|---|
| Docker ไม่ต้อง sudo | `docker ps` | ไม่มี `permission denied` | NemoClaw สั่ง Docker ในนามผู้ใช้ของคุณ |
| NVIDIA runtime | runtimes ใน `docker info` | มี `nvidia` | container ของ vLLM รันด้วย `--runtime=nvidia` |
| Kernel | `uname -sr` | Linux **≥ 6.2** | Landlock ABI 3 — ชั้น filesystem ที่จะเจอใน Module 03 |
| ดิสก์ว่าง | `df -BG` ที่ `$HOME` | ≥ 200 GB (กติกาคร่าว ๆ ของคอร์ส) | โมเดล Express ขนาดใหญ่อาจใช้หลายร้อย GB บวก image ของ vLLM |

เกณฑ์ผ่าน L1.1 ของ Reef runner คือฉบับสั้น: **เจอ GB10, มี Docker, kernel ≥ 6.2**

ถ้า `docker ps` ขึ้น `permission denied` ให้แก้เองใน ⌨ terminal แล็บจะพิมพ์บรรทัดเหล่านี้ให้ แต่ไม่รัน `sudo` เอง:

```bash
# on: spark
sudo usermod -aG docker $USER && newgrp docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

บรรทัดสุดท้ายคือหลักฐานว่า container มองเห็น GPU

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_1_verify_spark.py`) [บันทึกจาก Mac เครื่องนี้ในโหมด DRY]

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

ขั้นที่ 4 เป็นของจริง: Mac เครื่องนี้รัน Darwin ซึ่งไม่มี Landlock, ไม่มี seccomp และไม่มี network namespace นี่คือเหตุผลที่ sandbox อยู่บน Spark

✓ Checkpoint: ในโหมด LIVE ตารางขึ้น ✓ ที่ GPU, Docker server และ kernel ≥ 6.2 — หรือคุณรู้ว่าต้องพิมพ์คำสั่งแก้อะไร

## 2 · L1.2 — ติดตั้งด้วยคำสั่งเดียว (แบบ interactive)

บรรทัดเดียวติดตั้งทุกอย่าง รันเองบน Spark ใน ⌨ terminal:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

สิ่งที่เกิดขึ้นตามลำดับ:

1. **ประกาศเรื่องซอฟต์แวร์ของบุคคลที่สาม** คุณกดยอมรับ (การรันแบบสคริปต์ยอมรับด้วย `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` หรือ `--yes-i-accept-third-party-software` — ดูหัวข้อ 3 ว่าต้องวาง *ตรงไหน*)
2. **ติดตั้ง Node.js, OpenShell และ NemoClaw CLI** ตัวติดตั้งต้องใช้ Node.js 22.16+ และจะติดตั้งให้ถ้ายังไม่มี
3. **`nemoclaw onboard` เริ่มเอง** เมื่อ preflight ผ่าน ถ้าตัวติดตั้งพิมพ์ `To finish setup, run:` ให้รัน `nemoclaw onboard` ตามที่แสดงก่อนจะ connect
4. **Express Install** บน Spark คุณจะเจอคำถาม `Run express install with these settings? [Y/n]:` Express คือ managed local vLLM, โมเดล Express ที่ NVIDIA ดูแล, ชื่อ sandbox `my-assistant` และ policy แบบ Balanced ครั้งแรกให้เลือกอันนี้ ตอบ `n` ถ้าอยากเลือก agent, provider, โมเดล และชื่อเอง

> 📌 **sandbox มีอยู่จริงหลังจาก `nemoclaw onboard` เสร็จเท่านั้น** อย่ารัน `launch`, `connect` หรือ `openclaw tui` ก่อนหน้านั้น ถ้าติดตั้งเสร็จแล้วเจอ `nemoclaw` "not found" ให้รัน `source ~/.bashrc`

เมื่อ onboarding เสร็จ playbook แสดงสรุปแบบนี้:

**Expected output** (REFERENCE — quoted from the DGX Spark NemoClaw playbook) [ยกมาจาก playbook]

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

สังเกตสามเรื่อง บรรทัดโมเดลเขียนว่า **(Local vLLM)** URL ของ dashboard ต้องมี token ซึ่งได้จากอีกคำสั่งหนึ่ง (หัวข้อ 4 อธิบายว่าทำไมแล็บไม่พิมพ์มันออกมา) และสรุปนี้สะกดคำสั่ง policy ว่า `policy-add` ขณะที่ research tutorial เขียน `policy add` — รุ่นที่คุณติดตั้งเป็นตัวตัดสิน และ `--help` คือคำตอบสุดท้าย

L1.2 ผ่านเมื่อ `nemoclaw --version` ตอบบน Spark แล็บ 02-2 ปิดท้ายด้วยการตรวจข้อนี้

✓ Checkpoint: คุณรันคำสั่งบรรทัดเดียวใน ⌨ terminal แล้ว (หรืออธิบายได้ทีละขั้นว่ามันจะทำอะไร) และ `nemoclaw --version` ตอบกลับ

## 3 · L1.3 — ติดตั้งด้วยสคริปต์: บรรทัดแบบ non-interactive

มี Spark เครื่องเดียว ใช้ wizard ก็พอ ถ้ามีห้าเครื่อง คุณอยากได้การติดตั้งที่เหมือนกันทุกครั้ง quickstart มีตัวอย่างการรันครั้งแรกแบบ non-interactive เต็มรูปแบบ อันนี้ใช้ NVIDIA Endpoints (คลาวด์):

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

กติกาทั้งหมดในตารางเดียว:

| กติกา | ทำไม |
|---|---|
| `VAR=value` ทุกตัววาง **หลัง `\|`** หน้า `bash` | การกำหนดค่าหน้าคำสั่งส่งถึงเฉพาะคำสั่งนั้น ถ้าวางหน้า `curl` จะมีแค่ตัวดาวน์โหลดที่เห็น ตัวติดตั้งไม่เห็นเลย |
| non-interactive ต้องมี `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` | ไม่มีใครอยู่กดยอมรับประกาศ |
| `NEMOCLAW_AGENT` = `openclaw` \| `hermes` \| `langchain-deepagents-code` | เลือก harness (และ CLI: `nemoclaw`, `nemohermes`, `nemo-deepagents`) |
| `NEMOCLAW_POLICY_TIER` = `restricted` \| `balanced` \| `open` \| `personal` | ระดับ policy (ดูด้านล่าง) |
| `NEMOCLAW_SANDBOX_NAME` — ตัวพิมพ์เล็ก ตัวเลข และ `-` | runner ของคอร์สรับ `^[a-z0-9-]{1,40}$` อยู่ในกรอบนี้ไว้ |
| ตั้ง `NEMOCLAW_PROVIDER` (หรือ `NEMOCLAW_NO_EXPRESS=1`) จะข้าม Express | playbook บอกไว้ เพราะคุณเลือกเองแล้ว |
| key ต้องเป็น `<placeholder>` ในทุกอย่างที่แชร์ | และแทนค่าก่อนรัน: ถ้าไม่ใส่เครื่องหมายคำพูด bash จะอ่าน `<` และ `>` เป็น redirection |

ค่าของ `NEMOCLAW_PROVIDER` จาก quickstart และตาราง provider ใน playbook:

| ค่า | คืออะไร | inference รันที่ | ตัวแปร key |
|---|---|---|---|
| `vllm` | vLLM ที่รันอยู่แล้วบน `localhost:${NEMOCLAW_VLLM_PORT:-8000}` | **บน Spark** | — |
| `install-vllm` | managed vLLM บน Docker (ดาวน์โหลดใหญ่) | **บน Spark** | — |
| `ollama` | Ollama ในเครื่อง ใส่ `NEMOCLAW_MODEL` ได้ | **บน Spark** | — |
| `build` | NVIDIA Endpoints | คลาวด์ | `NVIDIA_INFERENCE_API_KEY` |
| `routed` | Model Router | คลาวด์ | `NVIDIA_INFERENCE_API_KEY` |
| `openrouter` · `openai` · `anthropic` · `gemini` | API แบบ hosted | คลาวด์ | `OPENROUTER_API_KEY` · `OPENAI_API_KEY` · `ANTHROPIC_API_KEY` · `GEMINI_API_KEY` |
| `custom` · `anthropicCompatible` | endpoint ใดก็ได้ที่เข้ากันได้กับ OpenAI / Anthropic | แล้วแต่ว่าชี้ไปที่ไหน | `COMPATIBLE_API_KEY` · `COMPATIBLE_ANTHROPIC_API_KEY` |
| `hermes-provider` | Hermes Provider | — | Hermes เท่านั้น |

สังเกตว่า `local-vllm` **ไม่อยู่** ในรายการนี้ มันคือ *ชื่อ provider ของ OpenShell* ที่ OpenShell playbook สร้าง (คุณจะเห็นใน `openshell inference get`) ไม่ใช่ค่าของ `NEMOCLAW_PROVIDER`

policy tier ทั้งสี่ จาก network-policies reference:

| Tier | อนุญาตอะไร |
|---|---|
| `restricted` | baseline อย่างเดียว |
| `balanced` (ค่าเริ่มต้น) | `npm`, `pypi`, `huggingface`, `brew`, `brave` และ Tavily ถ้าเลือก |
| `open` | เพิ่ม messaging, `jira`, `outlook`, `weather`, `public-reference` |
| `personal` | บังคับ `personal-open-internet`: โปรแกรมใดก็ออกพอร์ต 80/443 ได้ที่ L4 ห้ามใช้บนเครื่องที่ใช้ร่วมกัน |

สวิตช์อื่นจาก quickstart: `NEMOCLAW_WEB_SEARCH_PROVIDER=tavily|none` (Tavily ต้องมี `TAVILY_API_KEY` ด้วย), `NEMOCLAW_GATEWAY_RUNTIME=podman`, `--defer-onboarding` (ติดตั้ง CLI โดยยังไม่สร้าง provider หรือ sandbox) และการตรึงรุ่นเมื่อต้องการให้ทำซ้ำได้เหมือนเดิม:

```bash
# on: spark
curl -fsSL https://www.nvidia.com/nemoclaw.sh | NEMOCLAW_INSTALL_REF= NEMOCLAW_INSTALL_TAG=vX.Y.Z bash
```

แล็บ 02-2 สร้างบรรทัดเหล่านี้จากตัวเลือก ตรวจมัน และให้ bash ตัวจริงบนแล็ปท็อปนี้ parse (`bash -n` อ่านไวยากรณ์ ไม่รันอะไรเลย) ขั้นที่ 2 พิสูจน์กติกาเรื่อง pipe โดยใช้ `echo` แทน `curl`:

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_2_install_line.py`, steps 2–5) [บันทึกจาก Mac เครื่องนี้ในโหมด DRY]

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

ขั้นที่ 5 คือกับดักในบรรทัดตัวอย่างของเอกสาร: `<key>` ที่ยังไม่แก้ทำให้ bash ไปหาไฟล์ชื่อ `key` ในที่นี้มันล้มแบบเห็นชัด แต่ในบรรทัดของแบบฝึกหัด `>` จะเปลี่ยนคำถัดไปให้กลายเป็นชื่อไฟล์ด้วย tier จึงไม่ถูกตั้งค่าโดยไม่มีใครรู้

✓ Checkpoint: คุณอธิบายได้ว่าทำไม `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 curl … | bash` ไม่ได้ยอมรับอะไรเลย — และแล็บ 02-2 แสดงว่าบรรทัดเสียถูกปฏิเสธครบ 8/8

## 4 · L1.4 — คำสั่ง lifecycle ที่ใช้ทุกวัน

นี่คือคำสั่งจาก quickstart, NVIDIA blog และ Deep Agents quickstart คอร์สจัดกลุ่มตามว่าแล็บทำอะไรกับมันได้:

| คำสั่ง | ประเภท | คอร์สรันอย่างไร |
|---|---|---|
| `nemoclaw my-assistant status` | อ่านอย่างเดียว | แล็บรันให้ (LIVE) |
| `nemoclaw my-assistant policy list` | อ่านอย่างเดียว | แล็บรันให้ (LIVE) |
| `nemoclaw list` · `openshell sandbox list` · `openshell forward list` | อ่านอย่างเดียว | แล็บรันให้ (LIVE) |
| `nemoclaw my-assistant policy add <preset> --dry-run` | preview | แสดงผลการ merge ไม่เปลี่ยนอะไร |
| `nemoclaw my-assistant snapshot create --name before-change` | เปลี่ยนแปลง | `change()` — ต้องเปิด 🔓 |
| `nemoclaw my-assistant restart` · `stop` · `start` | เปลี่ยนแปลง | `change()` — ต้องเปิด 🔓 |
| `nemoclaw inference set --model <model> --provider <provider> --sandbox my-assistant` | เปลี่ยนแปลงแบบ hot | sandbox ยังรันต่อ |
| `nemoclaw my-assistant rebuild` · `nemoclaw onboard --recreate-sandbox` | เปลี่ยนแปลง สร้างใหม่ | `change()` — ต้องเปิด 🔓 |
| `nemoclaw onboard --fresh --gpu` | **ลบทิ้ง** แล้วสร้างใหม่ | ⌨ terminal เท่านั้น |
| `nemoclaw launch my-assistant` · `nemoclaw my-assistant connect` | interactive | ⌨ terminal เท่านั้น |
| `nemoclaw my-assistant logs --follow` · `openshell term` | streaming | ⌨ terminal เท่านั้น |
| `nemoclaw my-assistant dashboard-url --quiet` · `gateway-token --quiet` | **ความลับ** | ⌨ terminal เท่านั้น |
| `nemoclaw upgrade-sandboxes --auto` · `nemoclaw credentials reset <PROVIDER> && nemoclaw onboard` | เปลี่ยนแปลง | ⌨ terminal เท่านั้น |

ชุดอ่านอย่างเดียวที่ใช้ประจำ:

```bash
# on: spark
nemoclaw list
nemoclaw my-assistant status
nemoclaw my-assistant policy list
openshell sandbox list
openshell forward list
```

**ทำไมคำสั่ง token ต้องอยู่ใน terminal ของคุณ** `dashboard-url --quiet` พิมพ์ `http://127.0.0.1:18789/#token=<token>` token นี้คือ bearer credential ของ Control UI ใครถือไว้ก็คุยกับและสั่ง agent ที่ทำงานตลอดเวลาของคุณได้ output ของแล็บถูกสตรีมเข้า runner เก็บไว้ในประวัติการรัน และอาจถูก RECORDED ได้ สองคำสั่งนี้จึงให้คุณพิมพ์เองบน Spark:

```bash
# on: spark
nemoclaw my-assistant dashboard-url --quiet
```

ถ้าจะเปิด dashboard จากแล็ปท็อป ให้ forward พอร์ต ใช้ `127.0.0.1` ไม่ใช่ `localhost` — origin check ของ gateway ต้องการให้ตรงเป๊ะ:

```bash
# on: laptop
ssh -L 18789:127.0.0.1:18789 <you>@<spark>
```

บน Spark เอง `openshell forward start 18789 my-assistant --background` เริ่มการ forward และ `openshell forward list` แสดงรายการ playbook บอกว่าพอร์ตถูกกำหนดอัตโนมัติ (มักเป็น 18789 หรือ 18790) ให้ดูจาก URL

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_3_lifecycle.py`, steps 2–5) [บันทึกจาก Mac เครื่องนี้ในโหมด DRY]

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

`forward list` ตอบได้แม้บนแล็ปท็อป เพราะการ forward เป็น process ในเครื่องที่คุณพิมพ์ ส่วนคำสั่งอื่น parse ผ่านแล้วไปหยุดที่ gateway ที่ไม่มีอยู่ ซึ่งคือสิ่งที่ CLI บนแล็ปท็อปพิสูจน์ได้พอดี

✓ Checkpoint: สำหรับทุกคำสั่งในตาราง คุณบอกได้ว่าแล็บรันให้ได้หรือไม่ และทำไม `dashboard-url` ไม่มีวันเป็นหนึ่งในนั้น

## 5 · L1.5 — บทสนทนาแรก และการพิสูจน์ว่า inference อยู่ในเครื่อง

ทักทายก่อน connect เข้า sandbox แล้วส่งคำขอหนึ่งครั้งไปที่ `inference.local` — แบบเดียวกับที่ OpenShell playbook ใช้ทดสอบ:

```bash
# on: spark
# inside the sandbox (nemoclaw my-assistant connect)
curl https://inference.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<MODEL_HANDLE>", "messages": [{"role":"user","content":"Say hello from the Spark."}]}'
```

ในคำขอนั้นไม่มี API key เลย supervisor ดักจับ `inference.local` แล้ว gateway ใส่ credential จริงและส่งต่อไปยัง provider ที่คุณตั้งไว้ NVIDIA blog ยังให้ smoke test แบบ non-interactive ไว้ด้วย:

```bash
# on: spark
# inside the sandbox (nemoclaw my-assistant connect)
openclaw agent --agent main --local -m "hello" --session-id test
```

คำตอบที่ดูดีไม่ใช่หลักฐานว่ามันมาจาก GPU *ของคุณ* ต้องมีหลักฐานสามชิ้น:

| # | หลักฐาน | คำสั่ง | ผ่านเมื่อ |
|---|---|---|---|
| 1 | เส้นทาง (route) | `openshell inference get` (บน host) | provider เป็นแบบ local: `ollama`, `local-vllm`, `vllm` |
| 2 | รายชื่อโมเดล มองจากข้างใน | `openshell sandbox exec -n my-assistant -- curl -s https://inference.local/v1/models` | ได้รายชื่อโมเดลกลับมา |
| 3 | คลาวด์ถูกบล็อก | ข้างใน: `curl https://api.openai.com/v1/models` | ต้อง **ไม่** ผ่าน |

ข้อ 1 + 2 คือเกณฑ์ผ่าน L1.5 ของ Reef runner ข้อ 3 คือแบบฝึกหัดข้อ 3 ของ Part 1: ไม่มี `network_policies` ตัวไหนตรง proxy จึงปฏิเสธการเชื่อมต่อ คุณเห็นได้ใน `openshell term` (บน host; `f` ตามดู, `s` กรองตามแหล่ง, `q` ออก) และใน `openshell logs my-assistant --source sandbox` เอกสารพูดชัดว่าวิธีแก้ **ไม่ใช่** การเพิ่ม `api.openai.com` หรือ `api.anthropic.com` เข้าไปใน policy

```bash
# on: spark
openshell inference get
openshell sandbox exec -n my-assistant -- curl -s https://inference.local/v1/models
openshell sandbox exec -n my-assistant -- curl -sS -o /dev/null -w '%{http_code}\n' --max-time 15 https://api.openai.com/v1/models
openshell logs my-assistant --source sandbox -n 50 --since 5m
```

**Expected output** (REFERENCE — quoted from the OpenShell playbook, for `openshell inference get`) [ยกมาจาก OpenShell playbook]

```
Expected output should show `provider: local-vllm` and your chosen `model`.
```

*ชื่อ* provider เป็นแค่ป้าย `openshell provider list` แสดงว่ามันชี้ไปที่ไหน local หมายถึง IP ของ Spark เอง ไม่ใช่ `localhost` เพราะ gateway รันอยู่ใน Docker

แล็บ 02-4 รันฝั่ง Spark แบบอ่านอย่างเดียว และรันฝั่งแล็ปท็อปจริง: OpenShell parser รับทุกคำสั่ง, policykit model ของคอร์สอธิบายการตัดสินแต่ละครั้ง และ LAPTOP STAND-IN แสดงหน้าตาของคำตอบจาก `/v1/models`

**Expected output** (captured on this Mac, DRY mode — `labs/lab02_4_inference_is_local.py`, steps 4–6) [บันทึกจาก Mac เครื่องนี้ในโหมด DRY]

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

ดูการตัดสินข้อที่สี่ แม้แต่ vLLM ของคุณเองบน LAN IP ของ Spark ก็ถูกปฏิเสธจากใน sandbox ทางเดียวที่ไปถึงโมเดลได้คือ `inference.local`

✓ Checkpoint: ในโหมด LIVE ผลสรุปแสดง provider แบบ local, รายชื่อโมเดล และการเรียก OpenAI ที่ถูกบล็อก — หรือคุณบอกได้ว่าข้อไหนในสามข้อที่ยังขาด

## 6 · L1.6 — เพิ่มช่องทาง Telegram

Telegram ไม่บังคับ ติดตั้งครั้งแรกข้ามไปก่อนได้ Web UI กับ `openclaw tui` ก็เพียงพอ

1. ใน Telegram เปิด `@BotFather` ส่ง `/newbot` แล้วคัดลอก bot token
2. ลงทะเบียนช่องทาง วาง token เมื่อ wizard ถาม — ห้ามใส่ในบรรทัดคำสั่ง:

```bash
# on: spark
nemoclaw my-assistant channels add telegram
```

NemoClaw เก็บ credential ไว้และ **rebuild** sandbox เพื่อให้ OpenClaw ใช้ช่องทางนี้ได้ นี่คือการเปลี่ยนแปลง คุณจึงต้องพิมพ์เอง

3. wizard จะถาม Telegram user ID (ไม่บังคับ) เพื่อจำกัดว่าใครส่ง DM หาบอทได้ ถ้าข้ามไป บอทจะขอ pairing ก่อน ให้อนุมัติรหัสที่มันส่งมา จากใน sandbox:

```bash
# on: spark
# inside the sandbox (nemoclaw my-assistant connect)
openclaw pairing approve telegram <CODE>
```

4. ถ้าข้อความล้มเพราะ policy ให้ตรวจว่ามี preset `telegram` แล้ว (สะกดแบบใน playbook):

```bash
# on: spark
nemoclaw my-assistant policy-list
nemoclaw my-assistant policy-add telegram
```

Telegram ใช้ long-polling จึงไม่ต้องมี public URL หรือ cloudflared tunnel ให้รู้ความเสี่ยงที่เอกสารระบุไว้: preset `telegram` เปิดแค่ Telegram Bot API แต่หลังจากนั้น agent ส่งข้อความไปที่ **แชตใดก็ได้ที่ bot token เข้าถึง**

| อาการ | ดูอะไรก่อน |
|---|---|
| บอทไม่ตอบเลย | `nemoclaw my-assistant status` แล้วค่อย `nemoclaw my-assistant logs` (ไม่ใส่ `--follow`) |
| `409 Conflict` หลัง rebuild | มี process อื่นใช้ bot token เดียวกัน |
| รับข้อความได้แต่ไม่ตอบ | inference ล้ม, policy ปฏิเสธ หรือติดด่าน allowlist / mention — log จะบอกว่าอันไหน |

✓ Checkpoint: ข้อความที่ส่งหาบอทได้คำตอบกลับมาจาก agent (runner จะขอให้คุณยืนยัน เพราะมันมองไม่เห็นโทรศัพท์ของคุณ)

## 7 · L1.7 — แบบ Hermes และ Deep Agents

ตัวติดตั้งเดียวกันสร้าง harness อีกสองแบบได้ เปลี่ยนแค่ `NEMOCLAW_AGENT` และชื่อ CLI:

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

starter prompt ใน playbook ระบุคำสั่ง onboarding ที่ทำแบบเดียวกันไว้ด้วย: `nemohermes onboard` สำหรับ Hermes และ `nemo-deepagents onboard` สำหรับ Deep Agents

| | Hermes (`nemohermes`) | Deep Agents (`nemo-deepagents`) |
|---|---|---|
| state dir | `/sandbox/.hermes` | `/sandbox/.deepagents` |
| เก่งเรื่อง | API แบบ OpenAI-compatible ที่พอร์ต 8642, Tavily, Langfuse (Module 05) | วางแผนด้วย sub-agent, งานเขียนโค้ด |
| messaging ตอน onboarding | มี | ข้าม |
| Ollama ในเครื่อง | มีให้เลือก | ไม่มีให้เลือก จนกว่าเอกสารจะรองรับ (starter prompt ใน playbook) |

เกณฑ์ผ่าน L1.7 คือ sandbox ทั้งสองขึ้นใน `openshell sandbox list` แต่ละ sandbox คือ claw แยกกันที่มี policy ของตัวเอง สาม sandbox จึงหมายถึงสาม policy ที่ต้องอ่านใน Module 03

✓ Checkpoint: `openshell sandbox list` แสดง `my-hermes` และ `my-deepagents` ข้าง `my-assistant` — หรือคุณอธิบายได้ว่าทำไมข้ามไป

## Labs — รันได้ที่นี่

**labs/lab02_1_verify_spark.py** — ตรวจ Spark: OS, GB10, Docker, kernel สำหรับ Landlock และดิสก์ว่าง

**labs/lab02_2_install_line.py** — สร้างและตรวจบรรทัดติดตั้ง NemoClaw แบบคำสั่งเดียว โดยไม่รันมัน

**labs/lab02_3_lifecycle.py** — คำสั่ง lifecycle แบบอ่านอย่างเดียว, ด่านตรวจการเปลี่ยนแปลง และทำไม token ไม่เคยถูกพิมพ์ออกมา

**labs/lab02_4_inference_is_local.py** — พิสูจน์ว่า inference อยู่ในเครื่อง: route, รายชื่อโมเดล, คำทักทายหนึ่งครั้ง และการเรียกที่ต้องล้ม

## ลองทำเอง

`exercises/ex02_install_line.py` มี TODO สามข้อ:

1. **(Part 1 แบบฝึกหัดข้อ 2)** เขียนบรรทัดติดตั้งแบบ non-interactive สำหรับ Hermes claw ชื่อ `alto-hermes` ที่ใช้ vLLM ที่รันอยู่แล้วบนพอร์ต 8000, ค้นเว็บด้วย Tavily และ tier `restricted`
2. **(Part 1 แบบฝึกหัดข้อ 5)** process ไหนต้องเห็น `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1`: `curl` หรือ `bash`?
3. **(Part 1 แบบฝึกหัดข้อ 4)** เปลี่ยนโมเดลของ `my-assistant` หนึ่งครั้งโดยไม่ทำลาย sandbox และอีกครั้งด้วยวิธีที่ทำลาย

ตัวตรวจจะ parse บรรทัดของคุณ (ไม่รันมัน), รัน `bash -n` กับมัน และตรวจคำตอบอีกสองข้อ

```bash
# on: laptop
.venv/bin/python week26/02_first_claw/exercises/ex02_install_line.py
```

**Expected output** (captured on this Mac, all TODOs filled) [บันทึกจาก Mac เครื่องนี้ เมื่อเติม TODO ครบ]

```
✓ install line: hermes · alto-hermes · vllm on 8000 · tavily (key as a placeholder) · restricted · all on the bash side · bash -n OK
✓ the bash side: the variable must be in the environment of the bash process that RUNS the script
✓ model change: `inference set` is hot (route only) · `onboard --fresh` destroys and recreates

⚠ Verify the combination on your unit: each variable is documented on its own (quickstart, Hermes quickstart, network-policies reference), but the docs do not show this exact combination.
═ Done. Paste the line into the ⌨ terminal on a Spark (# on: spark) when you are ready — the course never will.
```

เก็บคำเตือนบรรทัดสุดท้ายไว้ research tutorial พูดตรง ๆ ว่าตัวแปรแต่ละตัวมีในเอกสารแยกกัน แต่เอกสารไม่ได้แสดงชุดนี้รวมกันแบบนี้ ให้ตรวจยืนยันบนเครื่องของคุณเอง

<details><summary>คำใบ้ — TODO 1</summary>

เริ่มจากบรรทัดตัวอย่างของ quickstart ในหัวข้อ 3 เปลี่ยน agent, ชื่อ และ provider แล้วเพิ่มอีกสี่ตัวแปร: พอร์ต vLLM, ผู้ให้บริการค้นเว็บ, Tavily key (เป็น placeholder) และ tier ทุกตัวต้องอยู่หลัง `|`

</details>

<details><summary>คำใบ้ — TODO 3</summary>

ตารางคำสั่งในหัวข้อ 4 มีหนึ่งแถวที่เขียนว่า "เปลี่ยนแปลงแบบ hot" และอีกแถวที่เขียนว่า "ลบทิ้งแล้วสร้างใหม่" แบบ hot เปลี่ยน inference route ซึ่งเป็นส่วนที่ปรับได้ขณะรัน อีกแบบสร้าง sandbox ใหม่ เพราะ image และ blueprint ปรับขณะรันไม่ได้

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสามบรรทัด

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `nemoclaw: command not found` หลังติดตั้ง | shell ยังไม่โหลด PATH ใหม่ | `source ~/.bashrc` หรือเปิด terminal ใหม่ |
| `docker ps` → `permission denied` | ผู้ใช้ของคุณไม่อยู่ในกลุ่ม `docker` | สี่บรรทัดในหัวข้อ 1 พิมพ์ใน ⌨ terminal |
| ตัวติดตั้งล้มเพราะ Node.js | Node.js เก่ากว่า 22.16 | ติดตั้ง Node.js 22.16+ แล้วรันตัวติดตั้งใหม่ |
| `connect` หรือ `openclaw tui` ล้มทันทีหลังติดตั้ง | onboarding ยังไม่เสร็จ | รัน `nemoclaw onboard` ที่ตัวติดตั้งพิมพ์ไว้ และรอจนขึ้น "OpenClaw is ready" |
| gateway: "port 8080 is held by container…" | มี OpenShell gateway อีกตัวรันอยู่ | `nemoclaw onboard` (หรือ `nemoclaw onboard --resume`) จะใช้ตัวเดิมหรือสร้างใหม่ให้ |
| inference ค้าง | vLLM ยังโหลดอยู่ | บน host รัน `curl http://127.0.0.1:8000/v1/models` แล้วรอ `Application startup complete` |
| Web UI ขึ้น `origin not allowed` | คุณใช้ `localhost` | ใช้ `http://127.0.0.1:18789/#token=…` |
| `policy list` เป็น "unknown command" | รุ่นของคุณสะกดว่า `policy-list` | แล็บ 02-3 ลองทั้งสองแบบ; `--help` คือคำตอบสุดท้าย |
| แล็บ 02-4 ขึ้น `⚠ inconclusive` ที่การทดสอบ OpenAI | `sandbox exec` เองล้ม (ชื่อ sandbox ผิด หรือไม่มี gateway) | ตรวจ `openshell sandbox list` และ `CLAW_SANDBOX` ใน 🖥 Spark setup |

## ถัดไป

[Lab 03 — OpenShell sandboxes and policy as code](../03_policy_as_code/TUTORIAL.md): อ่าน policy ที่ NemoClaw สร้างให้ เขียนของคุณเอง วนลูป deny → observe → allow → verify และนำ vLLM ของคุณเองมาใช้
