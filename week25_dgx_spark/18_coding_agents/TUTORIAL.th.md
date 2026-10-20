# ▶ Spark Lab 18 — Coding agent บน inference ในเครื่อง

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- serve โมเดลเขียนโค้ดบน Spark ของคุณด้วย Ollama แบบเดียวกับที่ coding playbook ทั้งสามของ NVIDIA ทำ
- เปิด Claude Code, OpenCode หรือ Codex ให้ต่อกับโมเดลนั้นด้วยคำสั่งเดียว: `ollama launch <agent> --model …`
- เรียนรู้ว่าอะไรเปลี่ยนไปเมื่อโมเดลอยู่ในเครื่อง: ความยาว context ความน่าเชื่อถือของ tool call และความเร็ว
- ตรวจความพร้อม (preflight) ของ endpoint: API สี่แบบที่ coding agent ใช้คุยด้วย การรองรับ tool และ context window
- วัดโมเดลแบบตรงไปตรงมา: tool call จริง การแก้ไฟล์จริง เทสต์จริง ในไดเรกทอรีชั่วคราว (temp dir)
- สร้างไฟล์ตั้งค่าให้เอเจนต์สี่ตัว (Claude Code, OpenCode, Codex, Continue) โดยไม่แตะ global config ของคุณเอง

**Time** ~45 นาที · **Difficulty** ระดับกลาง · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: แล็บจะรันกับ Ollama บนแล็ปท็อปของคุณเป็นตัวแทนที่ติดป้ายไว้)

**Playbook ทางการที่ครอบคลุม:** [CLI coding agent กับ inference ในเครื่อง](https://build.nvidia.com/spark/cli-coding-agent) · [Vibe coding ใน VS Code](https://build.nvidia.com/spark/vibe-coding) · และ Claude Code กับ inference ในเครื่อง (`nvidia/playbook-local-coding-agent` ใน GitHub repo ไม่อยู่ใน index ของ build.nvidia.com/spark ดูส่วนที่ 2)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จแล้ว | `ssh -o BatchMode=yes spark-a true` | แล็บรันคำสั่งบน Spark ผ่าน SSH |
| Module 03 (Ollama) จะช่วยได้ | `ollama --version` บน Spark | ทุก playbook ในที่นี้ serve ผ่าน Ollama |
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันแล็บ โดยมี PyYAML สำหรับ lab 03 |
| Ollama บนแล็ปท็อป (ไม่บังคับ) | `ollama list` บนแล็ปท็อป | ตัวแทนที่ติดป้ายไว้ เมื่อเข้าถึง Spark ไม่ได้ |
| coding agent ไว้ลองใช้ (ไม่บังคับ) | `claude --version`, `opencode --version` หรือ `codex --version` | ส่วนที่ 4 จะเปิดใช้ตัวใดตัวหนึ่ง |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
ollama --version
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
Python 3.13.13
ollama version is 0.34.4
```

> 🔐 แล็บไม่เคยเขียนลง `~/.claude`, `~/.codex`, `~/.config/opencode` หรือ `~/.continue` Lab 03 เขียนไฟล์ที่สร้างทั้งหมดไว้ใต้ `week25/18_coding_agents/.runs/` (อยู่ใน gitignore) แล้วตรวจว่าไฟล์ global config ของคุณไม่ถูกแก้ไข

✓ Checkpoint: คุณเข้าถึง Spark ผ่าน SSH ได้ หรือมี Ollama รันอยู่บนแล็ปท็อปเพื่อใช้เป็นตัวแทน

## 1 · อะไรเปลี่ยนไปเมื่อโมเดลเขียนโค้ดอยู่ในเครื่อง

coding agent (Claude Code, OpenCode, Codex, Continue) คือลูป มันส่งคำขอของคุณ ไฟล์ที่มันอ่านมา และรายการ **tool** (อ่านไฟล์ เขียนไฟล์ รันคำสั่ง) ไปให้โมเดล โมเดลตอบกลับด้วย **tool call** เอเจนต์รัน tool นั้น ส่งผลลัพธ์กลับไป แล้วถามต่อ โมเดลไม่เคยแตะดิสก์ของคุณ เอเจนต์เป็นผู้แตะ แต่จะทำก็ต่อเมื่อโมเดลส่ง tool call ที่มีรูปแบบถูกต้องกลับมาเท่านั้น

เมื่อคุณเปลี่ยนจากโมเดลบนคลาวด์มาใช้โมเดลบน Spark จะมีสี่อย่างที่เปลี่ยนไป:

| | โมเดลเขียนโค้ดบนคลาวด์ | โมเดลในเครื่องบน Spark | สิ่งที่ต้องตรวจ |
|---|---|---|---|
| โค้ดของคุณไปอยู่ที่ไหน | ไปที่ผู้ให้บริการ | อยู่ในเครือข่ายของคุณ | ไม่มี นั่นคือจุดประสงค์ |
| ค่าใช้จ่าย | คิดตาม token | ค่าไฟ | ไม่มี |
| **ความยาว context** | ใหญ่และคงที่ | เท่าที่ *เซิร์ฟเวอร์* จัดสรรให้ ซึ่งอาจน้อยกว่าที่โมเดลรองรับมาก | lab 01: context ที่จัดสรรจริงเทียบกับค่าสูงสุดของโมเดล |
| **ความน่าเชื่อถือของ tool call** | ปรับแต่งมาเพื่อเอเจนต์ | ขึ้นกับโมเดล การ quantize และ chat template ของเซิร์ฟเวอร์ | lab 02: มันเรียก tool ไหม ด้วย JSON ที่ถูกต้องไหม และเทสต์ผ่านไหม? |
| **ความเร็ว** | เร็ว | ถูกจำกัดด้วย memory bandwidth 273 GB/s (Module 01) | TTFT และ tok/s บน Spark ของคุณ |

ความน่าเชื่อถือของ tool call คือเรื่องที่ทำให้คนแปลกใจ โมเดลอาจเขียนโค้ดได้สมบูรณ์แบบแต่ยังล้มเหลวในฐานะเอเจนต์: มันตอบเป็นข้อความบรรยายแทนที่จะเรียก `write_file` มันพิมพ์ tool call ออกมาเป็นข้อความที่เซิร์ฟเวอร์ parse ไม่ได้ หรือมันส่ง JSON ที่เสีย ทุกกรณีคือการแก้ไฟล์ที่ล้มเหลว ตาราง Troubleshooting ของ CLI playbook เองมีแถวนี้: การตั้ง Claude Code แบบต่อตรง "produces prose but does not edit files" (ตอบเป็นข้อความแต่ไม่แก้ไฟล์) เพราะ "some model/server combinations do not emit tool calls reliably"

**ทำไม playbook จึงเลือกโมเดลแบบ MoE** ความเร็วในการ decode ถูกจำกัดด้วยจำนวนไบต์ของ weights ที่ *active* ที่ต้องอ่านต่อ token (Module 01 ส่วนที่ 7) ค่าตั้งต้นสำหรับ Spark ของ CLI playbook คือ `qwen3.6:35b-a3b-mtp-q4_K_M` ซึ่งเป็นโมเดล Mixture-of-Experts ขนาด 35B "a3b" ใน tag หมายถึงมีพารามิเตอร์ active ประมาณ 3B ต่อ token ส่วน Claude Code playbook ใช้ `qwen3.6:27b` แบบ dense นี่คือการคำนวณ โดยใช้ `sparkkit.decode_ceiling_tok_s` ที่ Q4_K_M (4.85 bit ต่อ weight):

| โมเดล | Active ต่อ token | อ่านต่อ token | เพดานความเร็วสายเดียวที่ 273 GB/s |
|---|---|---|---|
| qwen3.6 35B-A3B | ~3B | 1.8 GB | ~150 tok/s |
| qwen3.6 27B (dense) | 27B | 16.4 GB | ~17 tok/s |
| gpt-oss-120b (vibe-coding) | 5.1B | 2.7 GB (MXFP4) | ~101 tok/s |

ตัวเลขเหล่านี้เป็นขอบบนจากการคำนวณ ไม่ใช่ค่าที่วัดได้ เอเจนต์เรียกโมเดลหลายครั้งต่องานหนึ่งชิ้น ช่องว่าง 9× ของเพดานจึงเป็นตัวตัดสินว่าการแก้ไฟล์หนึ่งครั้งจะใช้เวลาเป็นวินาทีหรือเป็นนาที

✓ Checkpoint: คุณบอกได้สามแบบที่โมเดลล้มเหลวในฐานะเอเจนต์ แม้โค้ดของมันจะถูกต้อง และบอกได้ว่าทำไมค่าตั้งต้นของ Spark playbook จึงเป็นโมเดลแบบ MoE

## 2 · สาม playbook แนวคิดเดียว

playbook ทั้งสาม serve โมเดลด้วย Ollama ที่พอร์ต `11434` และต่างกันที่ client:

| Playbook | Client | โมเดล (ตามที่พิมพ์ไว้) | แถวฮาร์ดแวร์ในตาราง matrix ของมัน |
|---|---|---|---|
| [CLI coding agents](https://build.nvidia.com/spark/cli-coding-agent) | Claude Code, OpenCode, Codex CLI แต่ละตัวผ่าน `ollama launch` | `qwen3.6:35b-a3b-mtp-q4_K_M` (~23 GB) · ตัวเลือกเสริม `q8_0` (~39 GB), `bf16` (~71 GB) | **DGX Spark** |
| Claude Code กับ inference ในเครื่อง ([GitHub README](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/playbook-local-coding-agent)) | Claude Code ผ่าน `ollama launch` | `qwen3.6:27b` | **DGX Station** (ดูหมายเหตุ) |
| [Vibe coding ใน VS Code](https://build.nvidia.com/spark/vibe-coding) | ส่วนขยาย Continue ใน VS Code | `gpt-oss:120b` | **DGX Spark** |

> ⚠ `playbook-local-coding-agent/README.md` เวอร์ชันปัจจุบันระบุแค่ **DGX Station** ในตารางฮาร์ดแวร์ และไม่อยู่ใน index ของ build.nvidia.com/spark (ตรวจเมื่อ 2026-09-29) จึงไม่มีหน้า Spark ให้ลิงก์ไป สำหรับ Spark ให้ทำตาม playbook [CLI coding agents](https://build.nvidia.com/spark/cli-coding-agent) ขั้นตอนของ Claude Code ในนั้นเป็นคำสั่งเดียวกับ CLI playbook โมดูลนี้หยิบขั้นตอนเพิ่มมาจากมันหนึ่งขั้น คือการตั้งค่า context 64K (ส่วนที่ 3) สำหรับ Spark ให้ใช้โมเดลของ CLI playbook

ขนาดหน่วยความจำมาจาก prerequisites ของ CLI playbook ทั้งสามแบบใส่ใน 128 GB ของ Spark ได้ ค่าตั้งต้น `q4_K_M` เหลือพื้นที่ราว 100 GB สำหรับ KV cache โมเดลตัวที่สอง หรือ stack ของ Module 19

✓ Checkpoint: คุณรู้ว่าควรทำตาม playbook ไหนสำหรับเอเจนต์บนเทอร์มินัล (CLI coding agents) และตัวไหนสำหรับผู้ช่วยในเอดิเตอร์ (vibe coding)

## 3 · Serve โมเดลเขียนโค้ดบน Spark

นี่คือ Step 1–4 ของ CLI playbook รัน **บน Spark** playbook ติดตั้ง Ollama ด้วยสคริปต์ทางการ ถ้าคุณทำ Module 03 แล้ว Ollama จะมีอยู่แล้ว ให้ตรวจเวอร์ชันแล้วข้ามการติดตั้ง

```bash
# on: spark
cat /etc/os-release | head -n 2
nvidia-smi
ollama --version                       # already installed (Module 03)? skip the next line
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.6:35b-a3b-mtp-q4_K_M
ollama list
```

playbook บอกแค่ว่า `ollama --version` "should show a current Ollama release" (ควรแสดงเวอร์ชันล่าสุดของ Ollama) `ollama launch` และ tag แบบ MTP Q4_K_M ต้องใช้ build ที่ใหม่พอ Ollama รุ่นเก่าจะตอบ `unknown command` หรือ pull ล้มเหลวด้วย HTTP 412 และวิธีแก้ทั้งสองกรณีคือติดตั้ง Ollama ใหม่ ตัวเลือกเสริมที่ใหญ่กว่าจากขั้นเดียวกัน:

```bash
# on: spark
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

**ความยาว context** Claude Code playbook (Step 6) เตือนว่า "Ollama defaults to a 4096 token context length" (ค่าตั้งต้นของ Ollama คือ context 4096 token) และเพิ่มเป็น 64K สำหรับ coding agent คุณตั้งค่าได้ทั้งสำหรับเซสชันเดียว หรือสำหรับทั้งเซิร์ฟเวอร์:

```bash
# on: spark
ollama run qwen3.6:35b-a3b-mtp-q4_K_M
# at the >>> prompt:   /set parameter num_ctx 64000     then /bye
sudo systemctl stop ollama
OLLAMA_CONTEXT_LENGTH=64000 ollama serve        # keep this terminal open
```

> 💡 ค่าตั้งต้นเปลี่ยนไปตามเวอร์ชันของ Ollama CLI playbook บอกว่า Qwen3.6 "ships with a 256K context window by default" และ Ollama 0.34.4 บน Mac เครื่องนี้จัดสรร 262,144 token ให้ `nemotron-3-nano` (lab 01 ด้านล่าง) อย่าเดาตัวเลขเอาเอง Lab 01 อ่านค่าที่เซิร์ฟเวอร์จัดสรรจริงจาก `/api/ps` context ที่ใหญ่ขึ้นใช้หน่วยความจำ KV cache มากขึ้น (สูตรจาก Module 01) จึงควรตั้งให้พอกับที่ repo ของคุณต้องใช้

ทดสอบด้วย prompt ของ playbook เอง หรือแบบที่เอเจนต์ทำ: ด้วย tool บล็อก ⚡ ด้านล่างส่งไปที่ Ollama ของ Spark หรือตัวแทนบนแล็ปท็อป โดยติดป้ายไว้ชัดเจน:

```spark
{"target": "ollama", "which": "a", "model": "qwen3.6:35b-a3b-mtp-q4_K_M",
 "messages": [{"role": "user", "content": "Create math_utils.py with a function add(a, b) that returns a + b. Use the write_file tool."}],
 "max_tokens": 256,
 "tools": [{"type": "function", "function": {"name": "write_file", "description": "Write a text file in the workspace.",
   "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]}
```

คำตอบที่ดีคือบรรทัด `→ tool_call write_file({...})` ไม่ใช่ย่อหน้าของโค้ด

✓ Checkpoint: `ollama list` บน Spark แสดง `qwen3.6:35b-a3b-mtp-q4_K_M` และบล็อก ⚡ คืน tool call `write_file`

## 4 · เปิดเอเจนต์ด้วยคำสั่งเดียว

`ollama launch` เปิดเอเจนต์ที่รองรับโดยต่อเข้ากับโมเดลในเครื่องให้เรียบร้อย จึงไม่ต้องตั้ง environment variable และไม่ต้องเขียนไฟล์ provider ใด ๆ นี่คือคำสั่งใน Step 5 ของ CLI playbook สำหรับแต่ละเอเจนต์ (ติดตั้งเอเจนต์ครั้งเดียว แล้วค่อย launch):

```bash
# on: spark
# Claude Code
curl -fsSL https://claude.ai/install.sh | bash
claude --version
ollama launch claude --model qwen3.6:35b-a3b-mtp-q4_K_M

# OpenCode
curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"
opencode --version
ollama launch opencode --model qwen3.6:35b-a3b-mtp-q4_K_M

# Codex CLI (needs Node.js / npm)
npm install -g @openai/codex
codex --version
ollama launch codex --model qwen3.6:35b-a3b-mtp-q4_K_M
```

`ollama launch --help` บน Ollama 0.34.4 แสดง integration มากกว่าที่ playbook ครอบคลุม (Copilot CLI, Cline, Qwen Code, Hermes, OpenClaw และอื่น ๆ โดย Module 17 ใช้สองตัวหลัง) และยังมี `--config` (ตั้งค่าโดยไม่ launch) และ `--restore` (คืน integration กลับเป็น profile ตั้งต้น) ดังนั้น `launch` **แก้ไข profile ของเอเจนต์ตัวนั้นเอง** บนเครื่องที่คุณรันมัน บน Spark ไม่มีปัญหา แต่ควรรู้ไว้ก่อนจะรันบนแล็ปท็อปที่คุณใช้งานประจำ

จากนั้นคืองานแบบ end-to-end ของ playbook (Step 6): ฟังก์ชันโครง (stub) หนึ่งตัว เทสต์หนึ่งตัว และคำสั่งหนึ่งประโยค

```bash
# on: spark
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo
printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pytest
```

ในเอเจนต์ ให้พิมพ์ `Please implement add() in math_utils.py and make sure the test passes.` ออกจากเอเจนต์ (`/exit` หรือ Ctrl+C) แล้ว:

```bash
# on: spark
cd ~/cli-agent-demo && source .venv/bin/activate
python3 -m pytest -q
deactivate
```

ผลที่ playbook คาดไว้: "Expected output should show the test passing." (ผลลัพธ์ควรแสดงว่าเทสต์ผ่าน) ถ้าเอเจนต์พิมพ์โค้ดออกมาแต่ `math_utils.py` ยังจบด้วย `pass` แปลว่าคุณเจอความล้มเหลวของ tool call จากส่วนที่ 1 แล้ว Lab 02 วัดว่ามันเกิดบ่อยแค่ไหน

✓ Checkpoint: `pytest -q` ผ่านใน `~/cli-agent-demo` หลังจากที่เอเจนต์เป็นผู้แก้ไฟล์ ไม่ใช่คุณ

## 5 · เอเจนต์รันที่ไหน: บน Spark หรือบนแล็ปท็อปของคุณ?

playbook รันเอเจนต์ **บน Spark** (ในเซสชัน SSH) ข้าง ๆ โมเดล นี่คือการตั้งค่าที่ง่ายที่สุด: `localhost:11434` ไม่เปิดสู่เครือข่าย ถ้าคุณอยากให้เอเจนต์หรือเอดิเตอร์อยู่บนแล็ปท็อปแทน แล็ปท็อปต้องเข้าถึง Ollama ของ Spark ได้ ซึ่งมีสองวิธี:

| | SSH tunnel (ค่าตั้งต้นของคอร์ส) | เปิด Ollama สู่เครือข่าย (vibe-coding Step 2) |
|---|---|---|
| บน Spark | ไม่ต้องเปลี่ยนอะไร | systemd drop-in `OLLAMA_HOST=0.0.0.0:11434`, `OLLAMA_ORIGINS=*`, `ufw allow 11434/tcp` |
| URL บนแล็ปท็อป | `http://localhost:21434` | `http://<spark-ip>:11434` |
| ใครอื่นเรียกได้บ้าง | ไม่มีใคร | ทุกคนใน LAN หรือ tailnet **โดยไม่มีการยืนยันตัวตน** |
| เหมาะกับ | นักพัฒนาคนเดียว | ทีม โดยอยู่หลัง gateway ที่มีคีย์ (Module 08) |

tunnel ใช้พอร์ตในเครื่อง 21434 เพราะแล็ปท็อปของคุณอาจรัน Ollama ที่ 11434 อยู่แล้ว (Module 01, lab 03):

```bash
# on: laptop
ssh -N -L 21434:localhost:11434 spark-a
curl -s http://localhost:21434/api/version
```

สำหรับเส้นทาง vibe-coding ขั้นตอนเปิดการเข้าถึงระยะไกลบน Spark ของ playbook คือ:

```bash
# on: spark
sudo systemctl edit ollama          # add the [Service] lines lab 03 generates (ollama-override.conf)
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo ufw allow 11434/tcp
```

จากนั้นรัน `curl -v http://YOUR_HARDWARE_IP:11434/api/version` จากแล็ปท็อป ใน VS Code ให้ติดตั้ง **Continue** เลือก **Ollama** เป็น provider และ **Autodetect** เป็นโมเดล สำหรับ Spark ระยะไกล ให้แทนที่ `config.yaml` ของ Continue ด้วยบล็อกที่ lab 03 สร้างให้ ซึ่งคือ YAML ของ playbook เองที่เติมที่อยู่ของคุณไว้แล้ว

✓ Checkpoint: คุณเลือกแล้วว่าจะใช้ tunnel หรือเปิดพอร์ต และบอกได้ว่า `OLLAMA_ORIGINS=*` บน `0.0.0.0` เปิดอะไรออกไปบ้าง

## 6 · ตรวจความพร้อมของ endpoint: lab 01

ก่อนจะโทษเอเจนต์ ให้ตรวจเซิร์ฟเวอร์ก่อน coding agent ต้องใช้ API หนึ่งในสี่แบบจาก Ollama ขึ้นกับ client:

| Endpoint | ใช้โดย |
|---|---|
| `/api/*` (Ollama แบบ native) | `ollama launch`, provider `ollama` ของ Continue |
| `/v1/chat/completions` (OpenAI chat) | OpenCode และ client ส่วนใหญ่ที่เข้ากันได้กับ OpenAI |
| `/v1/messages` (Anthropic Messages) | Claude Code |
| `/v1/responses` (OpenAI Responses) | Codex |

Lab 01 ตรวจทั้งหมดนี้ รวมถึงความสามารถ `tools` ของโมเดลจาก `/api/show`, tool call `write_file` จริง และ context ที่เซิร์ฟเวอร์จัดสรร บน Spark มันยังรัน `ollama --version` ตรวจว่ามี `ollama launch` และแสดงรายการโมเดลที่ pull แล้วผ่าน SSH

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab01_endpoint_preflight.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้โดยไม่ได้ตั้งค่า Spark: เป็น LAPTOP STAND-IN แถวของ Spark จึงเป็นรูปแบบ EXAMPLE และโมเดลเป็นของแล็ปท็อป)

```
▣ STEP 2 · which endpoint are we testing?
→ http://localhost:11434 · model=nemotron-3-nano:latest · Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark

▣ STEP 3 · native API: version, model details, capabilities, context
│ nemotron-3-nano:latest: 31.6B params · Q4_K_M · capabilities=['completion', 'tools', 'thinking'] · model max context=1,048,576

▣ STEP 4 · the three wire formats: OpenAI chat (OpenCode), Anthropic messages (Claude Code), Responses (Codex)
· /v1/chat/completions → 'READY' in 10.1s
· /v1/messages         → blocks=['thinking'] text='' in 12.6s (stop_reason=max_tokens)
  ~ a thinking model spent the whole 24-token budget on a `thinking` block. Retry with thinking off:
· /v1/messages + thinking:disabled → 'READY.' (stop_reason=end_turn)
· /v1/responses        → 'READY' (status=completed)

▣ STEP 5 · a real tool call, and the context the server actually allocated
→ tool_call write_file({"path":"hello.py","content":"print(\"hi\")"})

│ check                                ok  detail
│ ───────────────────────────────────  ──  ────────────────────────────────────
│ Ollama native API  /api/version      ✓   0.34.4
│ model reports `tools` capability     ✓   completion, tools, thinking
│ OpenAI API  /v1/chat/completions     ✓   READY
│ Anthropic API  /v1/messages          ✓   READY. (only with thinking disabled)
│ OpenAI Responses API  /v1/responses  ✓   READY
│ tool call with valid JSON args       ✓   write_file · 1.8s
│ allocated context ≥ 64,000           ✓   262,144 tokens (model max 1,048,576)
◆ LAPTOP STAND-IN (these are not Spark numbers). Speeds above are this endpoint's, for a ≤ 300-token reply.
✓ every check passed: this endpoint can drive a CLI coding agent.
═ ready for `ollama launch claude --model …`
```

การเรียกครั้งแรกใช้เวลา 10 วินาทีเพราะต้องโหลดโมเดล ซึ่งเป็นเวลา cold start ไม่ใช่ตัววัดความเร็ว บรรทัดของ Anthropic แสดงกับดักจริง: โมเดลแบบ **thinking** บน `/v1/messages` ตอบกลับมาแค่บล็อก `thinking` และใช้งบ 24 token หมดก่อนจะเขียนข้อความใด ๆ เมื่อใส่ `"thinking": {"type": "disabled"}` มันตอบทันที บน Spark ที่ใช้ Qwen3.6 (ตระกูลที่ทำ thinking ได้เช่นกัน) เอเจนต์ที่รู้สึกว่าช้าอาจกำลังใช้ token ไปกับการคิดที่มองไม่เห็น

✓ Checkpoint: ทุกแถวเป็น ✓ เมื่อทดสอบกับ Spark ของคุณ (ตั้ง `SPARK_HOST` แล้ว) หรือกับตัวแทนบนแล็ปท็อป และคุณรู้ว่าเอเจนต์ที่คุณเลือกใช้ endpoint ไหน

## 7 · วัดความน่าเชื่อถือ ไม่ใช่วัดจากความรู้สึก: lab 02

"มันเขียนโค้ดสวยในแชต" ไม่ได้บอกอะไรเลยเกี่ยวกับงานของเอเจนต์ Lab 02 ทำแบบที่เอเจนต์ทำ สามครั้ง: ให้โมเดลได้ stub หนึ่งตัว เทสต์ที่ยังไม่ผ่านหนึ่งตัว และ tool `write_file` หนึ่งตัว มันเขียนสิ่งที่โมเดลส่งมาลงใน temp dir ใหม่ รันเทสต์ด้วย runner จาก stdlib (ไม่ต้องใช้ pytest) และถ้าไม่ผ่าน จะส่งผลเทสต์จริงกลับไปให้โมเดลแก้หนึ่งรอบ งานที่ 1 คืองาน `add()` ของ playbook เอง ส่วนงานที่ 2 และ 3 (`slugify`, `parse_duration`) ต้องใช้ความละเอียดมากขึ้นอีกนิด

> ⚠ Lab 02 รันโค้ดที่โมเดลเขียน มันรันใน temp dir ใน subprocess ที่มี timeout 20 วินาทีและ environment เปล่า หลังจากปฏิเสธโค้ดที่อ้างถึง `os`, `subprocess`, `socket`, `open(` หรือ `eval(` แล้ว นั่นคือราวกันตก ไม่ใช่ sandbox Module 15 (OpenShell) คือ sandbox

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab02_tool_call_probe.py            # --trials 5 for a real verdict
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เป็น LAPTOP STAND-IN: โมเดลบนแล็ปท็อปสามตัว ตัวละหนึ่งรอบ ผลของคุณจะต่างไปเพราะการ sampling เป็นแบบสุ่ม)

```
▣ STEP 1 · does each model advertise tool support? (GET /api/show → capabilities)
│ nemotron-3-nano:latest       completion, tools, thinking
│ gemma4:12b                   completion, vision, audio, tools, thinking
│ gemma3:4b                    completion, vision   ← no tools: an agent cannot edit files

▣ STEP 2 · run 3 tasks × 3 models × 1 trial(s)
✓ nemotron-3-nano:latest   add              44.5s  passed first try
✕ nemotron-3-nano:latest   slugify          15.1s  tests failed again after repair: AssertionError
✕ nemotron-3-nano:latest   parse_duration   11.4s  tool call printed as TEXT — the server's parser missed it, so no edit happened
✓ gemma4:12b               add              11.5s  passed first try
✓ gemma4:12b               slugify          23.6s  passed first try
✓ gemma4:12b               parse_duration   13.2s  passed first try
✕ gemma3:4b                add               0.0s  server refused: registry.ollama.ai/library/gemma3:4b does not support tools

▣ STEP 3 · scorecard
│ model                   tool call  valid args  pass 1st try  pass ≤1 repair  median time
│ ──────────────────────  ─────────  ──────────  ────────────  ──────────────  ───────────
│ nemotron-3-nano:latest  2/3        2/3         1/3           1/3             15.1s
│ gemma4:12b              3/3        3/3         3/3           3/3             13.2s
│ gemma3:4b               0/3        0/3         0/3           0/3             —
→ wrote week25/18_coding_agents/.runs/lab02_scorecard.json
```

รันสามครั้งบน Mac เครื่องนี้ได้รูปแบบเดียวกัน `gemma4:12b` ผ่าน 9 จาก 9 `nemotron-3-nano` ผ่าน `add` ทุกครั้ง แก้ `slugify` ได้หลังรอบซ่อมหนึ่งในสามครั้ง และพิมพ์ tool call ของ `parse_duration` ออกมาเป็นข้อความ (`<tool_call><function=write_file>…`) ทั้งสามครั้ง `gemma3:4b` ไม่มีความสามารถ `tools` เซิร์ฟเวอร์จึงปฏิเสธคำขอทันที นี่ให้บทเรียนสามข้อ:

1. **ความสามารถ (capability) คือประตูที่ข้ามไม่ได้** ตรวจ `/api/show → capabilities` ก่อนสิ่งอื่นใด ไม่มี `tools` ก็ไม่มีเอเจนต์
2. **"พิมพ์เป็นข้อความ" เป็นปัญหาของเซิร์ฟเวอร์/template ไม่ใช่ปัญหาการเขียนโค้ด** โมเดลพยายามเรียก tool แล้ว แต่ในรูปแบบที่เซิร์ฟเวอร์ parse ไม่ได้ นี่คือ "produces prose but does not edit files" ของ playbook และเป็นเหตุผลที่มันแนะนำ `ollama launch` กับโมเดล Qwen3.6 ที่ทดสอบแล้ว
3. **โมเดลที่ใหญ่กว่าไม่ได้เป็นเอเจนต์ที่ดีกว่าเสมอไป** `nemotron-3-nano` ขนาด 31.6B แพ้ `gemma4` ขนาด 12B ในการทดสอบนี้ ให้วัดโมเดลที่คุณจะใช้จริง บน Spark ด้วย `--trials 5`

บน Spark, lab 02 ทดสอบ `qwen3.6:35b-a3b-mtp-q4_K_M` เป็นค่าตั้งต้น (ใช้ `--models a,b` เพื่อเปรียบเทียบ) ยังไม่มีใครบันทึกการรันนั้นไว้สำหรับคอร์สนี้ จึงยังไม่มีผลลัพธ์จาก Spark ให้ดูในที่นี้ ของคุณจะเป็นชุดแรก

✓ Checkpoint: คุณมี scorecard ของโมเดลที่รองรับ tool อย่างน้อยหนึ่งตัว และอธิบายสาเหตุของแต่ละบรรทัด ✕ ได้

## 8 · ไฟล์ตั้งค่าสำหรับทุกเอเจนต์ อย่างปลอดภัย: lab 03

`ollama launch` ไม่ต้องใช้ไฟล์ใด ๆ แต่ใช้ได้เฉพาะบนเครื่องที่รัน Ollama และเฉพาะเอเจนต์ที่มันรองรับ เมื่อเอเจนต์รันบนแล็ปท็อป หรือคุณอยากเก็บการตั้งค่าไว้ใน version control คุณต้องมีไฟล์ config Lab 03 เขียนไฟล์เหล่านั้นสำหรับ endpoint หนึ่งตัวและโมเดลหนึ่งตัว ตรวจความถูกต้องของแต่ละไฟล์ (ด้วย parser ของ YAML, JSON, TOML) แล้วพิมพ์วิธีนำไปใช้ มัน **ไม่เคย** นำอะไรไปใช้เอง

| ไฟล์ | ที่มา | ใช้สำหรับ |
|---|---|---|
| `on_spark_launch.sh` | CLI playbook, Step 3 และ 5 | pull + `ollama launch claude\|opencode\|codex` บน Spark |
| `ollama-override.conf` | vibe-coding playbook, Step 2 | เปิด Ollama ของ Spark สู่เครือข่าย (และ `OLLAMA_CONTEXT_LENGTH` ที่ comment ไว้ เปิดใช้ได้ถ้าต้องการ) |
| `continue-config.yaml` | vibe-coding playbook, Step 6 | Continue ใน VS Code บนแล็ปท็อปของคุณ |
| `claude-code-direct.env` | **คอร์ส** ไม่ใช่ playbook | Claude Code บนแล็ปท็อป → `/v1/messages` ของ Ollama ในเชลล์เดียวเท่านั้น |
| `opencode.json` | **คอร์ส** ไม่ใช่ playbook | config ระดับ *โปรเจกต์* ของ OpenCode → `/v1` |
| `codex-home/config.toml` | **คอร์ส** ไม่ใช่ playbook | Codex กับ `CODEX_HOME` เพื่อไม่ให้แตะ `~/.codex` เลย |

ไฟล์ COURSE ทั้งสามคือการต่อสายสำหรับกรณีที่ใช้ `ollama launch` ไม่ได้ Lab 01 แสดงแล้วว่า Ollama ตอบ `/v1/messages` และ `/v1/responses` ได้ แต่คำเตือนของ playbook ยังใช้อยู่: การตั้งค่าแบบต่อตรงอาจได้ข้อความบรรยายแทนการแก้ไฟล์ ให้รัน lab 02 กับ endpoint เดียวกันก่อน ชื่อฟิลด์ในไฟล์ของ Codex และ OpenCode มาจากเอกสารของเครื่องมือเหล่านั้น ไม่ใช่ของ NVIDIA จึงควรตรวจกับเวอร์ชันที่คุณติดตั้งอยู่

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab03_agent_config_generator.py            # placeholder host
.venv/bin/python week25/18_coding_agents/labs/lab03_agent_config_generator.py --tunnel   # http://localhost:21434
```

**Expected output** (บันทึกจาก Mac เครื่องนี้โดยไม่ได้ตั้งค่า Spark placeholder `YOUR_HARDWARE_IP` ของ playbook จึงยังคงอยู่)

```
▣ STEP 2 · write and validate each file
│ file                    source    size     check
│ ──────────────────────  ────────  ───────  ───────────────────────────────
│ on_spark_launch.sh      PLAYBOOK    539 B  ✓ bash script
│ ollama-override.conf    PLAYBOOK    537 B  ✓ systemd [Service] block
│ continue-config.yaml    PLAYBOOK    324 B  ✓ valid YAML · provider ollama
│ claude-code-direct.env  COURSE      387 B  ✓ 3 variables
│ opencode.json           COURSE      386 B  ✓ valid JSON
│ codex-home/config.toml  COURSE      407 B  ✓ valid TOML · provider defined
→ all files in week25/18_coding_agents/.runs/agent-config/
…
▣ STEP 4 · prove your global agent config was not touched
✓ ~/.claude/settings.json            unchanged
✓ ~/.codex/config.toml               unchanged
✓ ~/.config/opencode/opencode.json   absent (not created)
✓ ~/.continue/config.yaml            absent (not created)
═ Generated and validated; nothing outside .runs/ was written.
```

การนำไปใช้เป็นสิ่งที่คุณเลือกเอง ทีละเอเจนต์ ตัวอย่างเช่น Claude Code บนแล็ปท็อปผ่าน tunnel ใน subshell เพื่อไม่ให้มีอะไรค้างอยู่:

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/labs/lab03_agent_config_generator.py --tunnel
( set -a; . week25/18_coding_agents/.runs/agent-config/claude-code-direct.env; set +a; \
  cd ~/cli-agent-demo && claude --model qwen3.6:35b-a3b-mtp-q4_K_M )
```

✓ Checkpoint: lab 03 จบด้วย `nothing outside .runs/ was written` และคุณบอกได้ว่าไฟล์ไหนมาจาก playbook และไฟล์ไหนเป็นของคอร์ส

## Labs — รันแล็บได้ที่นี่

**labs/lab01_endpoint_preflight.py** — ตรวจความพร้อมของ Ollama endpoint สำหรับ coding agent: เวอร์ชัน ความสามารถด้าน tool ทั้งสี่ API, tool call จริง และ context ที่จัดสรร

**labs/lab02_tool_call_probe.py** — ทดสอบความน่าเชื่อถือ: งานเขียนโค้ดสามงานผ่าน tool write_file ให้คะแนนโดยรันเทสต์จริง พร้อมรอบซ่อมหนึ่งรอบ

**labs/lab03_agent_config_generator.py** — สร้างและตรวจความถูกต้องของไฟล์ตั้งค่า Claude Code, OpenCode, Codex และ Continue ลงใน .runs/ โดยไม่แตะ global config

Lab 01 และ 02 ใช้ Ollama ของ Spark เมื่อมันตอบ ไม่เช่นนั้นใช้ของแล็ปท็อป (ติดป้าย LAPTOP STAND-IN) Lab 03 ทำงานแบบออฟไลน์

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 18 — ให้คะแนนคำตอบของเอเจนต์** eval harness ต้องแปลงคำตอบดิบของโมเดลให้เป็นคำตัดสิน เปิด `week25/18_coding_agents/exercises/ex18_grade_agent_reply.py` ในไฟล์มี `TODO` สามจุด:

1. `tool_args(tool_call)`: คืน argument เป็น dict ไม่ว่าเซิร์ฟเวอร์จะส่งมาเป็น JSON string หรือ dict
2. `safe_path(path)`: ปฏิเสธ path แบบ absolute, `~`, `..` และ backslash เพื่อไม่ให้เอเจนต์เขียนไฟล์นอก workspace ได้
3. `grade(reply, tests_passed)`: คืนค่าหนึ่งใน `no-tool-support`, `text-tool-call`, `prose-only`, `bad-args`, `unsafe-path`, `edit+pass`, `edit+fail`

fixture ทำตามรูปแบบคำตอบที่ lab 02 เจอบน Mac เครื่องนี้ บวกกรณีขอบ (edge case) ที่ทำขึ้นเองอีกสามกรณี

```bash
# on: laptop
.venv/bin/python week25/18_coding_agents/exercises/ex18_grade_agent_reply.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด)

```
✓ tool_args: JSON string → dict · dict passes through · bad JSON and lists → None
✓ safe_path: 2 allowed · 6 refused (empty, absolute, ~, .., nested .., backslash)
✓ grade: all 7 fixtures get the right verdict

▣ your grader, applied to the fixtures
│ lab 02 shape · clean tool call       → edit+pass
│ lab 02 shape · edit, tests fail      → edit+fail
│ lab 02 · tool call printed as text   → text-tool-call
│ lab 02 · gemma3:4b refuses tools     → no-tool-support
│ hand-made · explains instead         → prose-only
│ hand-made · truncated JSON           → bad-args
│ hand-made · escapes workspace        → unsafe-path
```

<details><summary>คำใบ้ — ทำไม "a/../../b.py" ถึงไม่ปลอดภัย ทั้งที่ขึ้นต้นด้วยโฟลเดอร์ปกติ?</summary>

แยกด้วย `/` แล้วหาส่วนที่เป็น `..` ที่ตำแหน่งใดก็ได้ `a/../../b.py` resolve ได้เป็น `../b.py` ซึ่งอยู่เหนือ workspace หนึ่งระดับ การตรวจแค่ `startswith("..")` จะพลาดกรณีนี้

</details>

<details><summary>ท้าทายเพิ่ม — ให้คะแนนผลลัพธ์จริงของ lab 02</summary>

Lab 02 เขียน `.runs/lab02_scorecard.json` ไว้ ให้จับคู่ `note` ของแต่ละระเบียนกับคำตัดสินของคุณ แล้วพิมพ์อัตราการผ่านต่อโมเดล จากนั้นรัน lab 02 ด้วย `--trials 5` บน Spark ของคุณ แล้วเปรียบเทียบ Qwen3.6 กับโมเดลบนแล็ปท็อป

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `ollama launch` แจ้ง unknown command หรือ pull ล้มเหลวด้วย HTTP 412 | Ollama เก่าเกินไปสำหรับ `launch` หรือ tag แบบ MTP Q4_K_M ติดตั้งใหม่จาก ollama.com/download (CLI playbook) |
| เอเจนต์พิมพ์โค้ดออกมาแต่ไม่เคยแก้ไฟล์ | tool call หายไปหรือ parse ไม่ได้ (lab 02: "printed as TEXT") ใช้ `ollama launch <agent>` กับโมเดล Qwen3.6 ของ playbook ตามที่ส่วน Troubleshooting ของมันแนะนำ |
| `does not support tools` (HTTP 400) | โมเดลไม่มีความสามารถ `tools` (ในที่นี้คือ `gemma3:4b`) เลือกโมเดลที่ capabilities ใน `/api/show` มี `tools` |
| คำตอบของ Claude Code ว่างเปล่าหรือถูกตัด | โมเดลแบบ thinking ใช้งบไปกับการคิด (lab 01, Step 4) ให้ token สำหรับ output มากขึ้น หรือปิด thinking ถ้า client ของคุณอนุญาต |
| เอเจนต์ลืมไฟล์ที่เพิ่งอ่านไปเมื่อนาทีก่อน | context เล็กเกินไป ตรวจ context ที่จัดสรรด้วย lab 01 แล้วตั้ง `num_ctx` / `OLLAMA_CONTEXT_LENGTH=64000` (ส่วนที่ 3) |
| `connection refused` ไปยัง `localhost:11434` | Ollama ไม่ได้รันอยู่: `ollama serve` หรือ `sudo systemctl start ollama` |
| Continue บนแล็ปท็อปเชื่อมต่อไม่ได้ | พอร์ตปิดอยู่หรือ bind กับ localhost: รัน `ss -tuln \| grep 11434` บน Spark แล้วทำ vibe-coding Step 2 หรือใช้ tunnel |
| `externally-managed-environment` ตอนติดตั้ง pytest | สร้าง venv ก่อน: `python3 -m venv .venv && source .venv/bin/activate` |
| ตอบช้าหรือ OOM | ปิดงาน GPU อื่น ใช้ค่าตั้งต้น `q4_K_M` ต่อไป และตั้ง `OLLAMA_MAX_LOADED_MODELS=1` บน unified memory playbook จะล้าง cache ด้วย: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |

## Next — บทถัดไป

ไปต่อที่ [Lab 19 — multi-agent chatbot, RAG และ knowledge graph](../19_multi_agent_rag_kg/TUTORIAL.md): รัน supervisor agent ที่มอบงานให้ผู้เชี่ยวชาญด้านเขียนโค้ด การค้นคืน และภาพ แปลงข้อความเป็น knowledge graph ด้วย txt2kg และสร้าง retriever ขนาดเล็กที่อ้างอิงแหล่งที่มาได้
