# ▶ Spark Lab 01 — รู้จัก DGX Spark ของคุณ: เชื่อมต่อ ตรวจเช็ก และวางงบหน่วยความจำ

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เรียนรู้ตัวเลขฮาร์ดแวร์ไม่กี่ตัวที่เป็นตัวตัดสินทุกอย่างในสัปดาห์นี้: unified memory 128 GB, bandwidth 273 GB/s และ FP4
- เชื่อมต่อ Spark ด้วยวิธีทางการ (NVIDIA Sync, SSH แบบตั้งเอง, Tailscale) แล้วชี้ Lab Runner ไปที่เครื่อง
- รัน **Spark doctor**: การตรวจแบบอ่านอย่างเดียว (read-only) แปดข้อ ว่าเครื่องพร้อมใช้งานหรือยัง
- เปิด **DGX Dashboard** ผ่าน SSH tunnel และสร้าง tunnel เส้นเดียวสำหรับทุกพอร์ตที่ใช้ในสัปดาห์นี้
- คำนวณด้วยเลขคณิตว่าโมเดลไหนใส่ใน Spark เครื่องเดียวได้ และโมเดลไหนต้องใช้สองเครื่อง

**Time** ~40 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บที่เป็นการคำนวณ)

**Playbook ทางการที่ครอบคลุม:** [Connect to your Spark](https://build.nvidia.com/spark/connect-to-your-spark) · [Tailscale](https://build.nvidia.com/spark/tailscale) · [DGX Dashboard](https://build.nvidia.com/spark/dgx-dashboard) · [VS Code](https://build.nvidia.com/spark/vscode)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันแล็บและ Lab Runner |
| SSH client | `ssh -V` | ทุกแล็บของ Spark สั่งงาน Spark ผ่าน SSH |
| DGX Spark ในเครือข่ายหรือใน tailnet ของคุณ | `ping <spark>.local` หรือ `tailscale status` | ไม่บังคับ: ถ้าไม่มี ทุกอย่างรันแบบ DRY |
| username และรหัสผ่านของ Spark | จากการบูตครั้งแรก | ใช้ล็อกอิน SSH ครั้งแรก |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
ssh -V
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
Python 3.13.13
OpenSSH_10.2p1, LibreSSL 3.3.6
```

> 🔐 ไม่มีส่วนไหนในคอร์สนี้เก็บรหัสผ่าน SSH ใช้ **คีย์ (key)** ส่วนโทเค็นที่คุณเพิ่มภายหลัง (Hugging Face, NGC) จะถูกเก็บฝั่งเซิร์ฟเวอร์ใน `week25/.env.local` ซึ่งอยู่ใน gitignore และไม่เคยถูกส่งไปที่เบราว์เซอร์

✓ Checkpoint: `ssh -V` พิมพ์เวอร์ชันออกมา และคุณรู้ hostname ของ Spark (พิมพ์อยู่บนการ์ด quick-start เช่น `spark-abcd`)

## 1 · DGX Spark คืออะไร ในภาพเดียว

DGX Spark คือคอมพิวเตอร์ตั้งโต๊ะขนาดเล็กที่สร้างรอบชิป **NVIDIA GB10 Grace Blackwell Superchip** ตัวเดียว: มี CPU แบบ Arm และ GPU แบบ Blackwell อยู่ในแพ็กเกจเดียวกัน และใช้หน่วยความจำก้อนเดียวร่วมกัน

| สเปก | ค่า | ความหมายสำหรับคุณในสัปดาห์นี้ |
|---|---|---|
| หน่วยความจำ | **128 GB LPDDR5x, unified** | CPU และ GPU ใช้ร่วมกัน โมเดลขนาดถึง ~200B พารามิเตอร์ใส่ใน Spark เครื่องเดียวได้ที่ 4-bit |
| Memory bandwidth | **273 GB/s** | เป็นเพดานว่าบทสนทนา *หนึ่งสาย* สร้าง token ได้เร็วแค่ไหน (ส่วนที่ 7) |
| พลังประมวลผล AI | **1 PFLOP ที่ FP4** (sparse) | เหตุผลที่ฟอร์แมต NVFP4 ใน Module 07 สำคัญ |
| CPU | 20 คอร์ Arm (10× Cortex-X925 + 10× Cortex-A725) | ทุกอย่างเป็น **aarch64**: เลือก container และ wheel แบบ ARM64 |
| เครือข่าย | ConnectX-7, 2× QSFP, 200 Gb/s | ต่อสาย Spark สองเครื่องเข้าด้วยกันใน Module 02 |
| พื้นที่เก็บข้อมูล | NVMe สูงสุด 4 TB | โมเดลมีขนาดใหญ่: เผื่อไว้ ~200 GB สำหรับสัปดาห์นี้ |
| OS | DGX OS (Ubuntu 24.04) พร้อม CUDA, Docker และ NVIDIA Container Toolkit | playbook ทั้งหมดถือว่ามีสิ่งนี้อยู่แล้ว |

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

**Unified memory เปลี่ยนนิสัยหนึ่งอย่าง** บนพีซีทั่วไปคุณจะถามว่า "โมเดลใส่ใน VRAM ของ GPU ได้ไหม?" แต่บน Spark ไม่มี VRAM แยก: ตัวโมเดล KV cache โปรเซส Python ของคุณ และ OS ใช้ 128 GB ร่วมกันทั้งหมด ดังนั้น `free -g` จึงเป็นมาตรวัดหน่วยความจำที่ตรงไปตรงมาที่สุด และการปิด Jupyter kernel ที่ลืมทิ้งไว้ก็ช่วยคืนหน่วยความจำให้โมเดลได้

✓ Checkpoint: คุณอธิบายได้ว่าทำไม Spark จึงโหลดโมเดล 70B ที่ใส่ไม่ได้ใน GPU เกมมิ่ง 24 GB ได้ และตัวเลขไหนที่จำกัดความเร็วในการสร้าง token

## 2 · เชื่อมต่อ: NVIDIA Sync หรือ SSH ธรรมดา

playbook [Connect to your Spark](https://build.nvidia.com/spark/connect-to-your-spark) ของ NVIDIA มีสองทางให้เลือก:

- **NVIDIA Sync** (แอปเดสก์ท็อปสำหรับ macOS, Windows, Linux) เพิ่ม Spark ครั้งเดียว แล้วเปิดเทอร์มินัล VS Code, Cursor หรือ DGX Dashboard ได้ในคลิกเดียว มันสร้าง SSH key และ tunnel ให้คุณเอง
- **SSH แบบตั้งเอง (manual)** คือสิ่งที่คอร์สนี้ทำให้เป็นอัตโนมัติ จึงควรลองทำด้วยมือสักครั้ง

ขั้นแรก ตรวจว่าชื่อ mDNS ของ Spark resolve ได้ในเครือข่ายของคุณ:

```bash
# on: laptop
ping -c 3 spark-abcd.local
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
PING spark-abcd.local (10.9.1.9): 56 data bytes
64 bytes from 10.9.1.9: icmp_seq=0 ttl=64 time=6.902 ms
64 bytes from 10.9.1.9: icmp_seq=1 ttl=64 time=116.335 ms
64 bytes from 10.9.1.9: icmp_seq=2 ttl=64 time=33.301 ms
```

(บล็อกนั้นยกมาจาก playbook ชื่อและ IP จึงเป็นของ playbook) ถ้าคุณเห็น `cannot resolve … Unknown host` แปลว่า mDNS ถูกบล็อก ซึ่งพบบ่อยใน Wi-Fi ขององค์กร ให้ใช้ IP address จากเราเตอร์ หรือใช้ Tailscale (ส่วนถัดไป)

ล็อกอินด้วยรหัสผ่านหนึ่งครั้ง แล้วยืนยันว่าคุณอยู่บน Spark:

```bash
# on: laptop
ssh <you>@spark-abcd.local
hostname && uname -m
exit
```

ต่อไปเปลี่ยนเป็นล็อกอินด้วยคีย์ Lab Runner รันคำสั่งด้วย `ssh -o BatchMode=yes` ซึ่ง **ไม่ถามรหัสผ่านเลย** จึงจำเป็นต้องมีคีย์:

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

alias `Host spark-a` ทำให้คำสั่งต่อ ๆ ไปพิมพ์แค่ `ssh spark-a` ได้ ถ้าคุณมี Spark เครื่องที่สอง ให้เพิ่ม `Host spark-b` แบบเดียวกันตอนนี้เลย Module 02 จะใช้มัน

✓ Checkpoint: `ssh -o BatchMode=yes spark-a true` กลับมาทันทีโดยไม่ถามรหัสผ่าน

## 3 · เข้าถึงได้จากทุกที่: Tailscale

mDNS ใช้ได้เฉพาะในเครือข่ายท้องถิ่นเดียวกัน [Tailscale playbook](https://build.nvidia.com/spark/tailscale) นำ Spark และแล็ปท็อปของคุณมาอยู่บนเครือข่ายส่วนตัวที่เข้ารหัส (เรียกว่า *tailnet*) ทำให้ `ssh spark-a` ใช้ได้ทั้งจากบ้าน ออฟฟิศ หรือร้านกาแฟ

บน **Spark** ให้เพิ่ม apt repository ของ Tailscale แล้วติดตั้ง (นี่คือคำสั่งจาก playbook สำหรับ Ubuntu 24.04 "noble"):

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

บน **แล็ปท็อป** ให้ติดตั้งแอป Tailscale ล็อกอินด้วยบัญชีเดียวกัน แล้วรัน:

```bash
# on: laptop
tailscale status
tailscale ping spark-abcd
```

ชี้ SSH alias ของคุณไปที่ชื่อใน tailnet (เช่น `spark-abcd` หรือ `spark-abcd.<tailnet>.ts.net`) แทน `.local` แล้วคุณจะเข้าถึง Spark ได้จากทุกที่ Lab Runner แสดงสถานะ tailnet ไว้บนแถบด้านบน: **Spark A: ● spark-a** เมื่อ ssh ใช้งานได้

> ⚠ Tailscale ทำให้ *ทุก* พอร์ตที่ listen บนทุก interface เข้าถึงได้จาก tailnet ของคุณ เซิร์ฟเวอร์ที่คุณเปิดด้วย `docker run -p 8000:8000` จะถูกเรียกได้จากทุกอุปกรณ์ใน tailnet สะดวกสำหรับแล็บก็จริง แต่ต้องมีการยืนยันตัวตน (authentication) ด้านหน้าก่อนแชร์ tailnet ให้คนอื่น (Module 08 เพิ่มคีย์ของ LiteLLM)

✓ Checkpoint: `tailscale status` แสดงทั้งแล็ปท็อปและ Spark และ `ssh -o BatchMode=yes spark-a true` ใช้ได้ผ่าน tailnet

## 4 · ชี้ Lab Runner ไปที่ Spark ของคุณ

Lab Runner ไม่เคยเดา host เอง เปิด **🖥 Spark setup** บนแถบด้านบนแล้วตั้งค่า:

| การตั้งค่า | ตัวอย่าง | ใช้โดย |
|---|---|---|
| `SPARK_HOST` | `spark-a` (SSH alias ของคุณ) | ทุกแล็บ: `sh()` รันคำสั่งที่นี่ |
| `SPARK_HOST2` | `spark-b` | โมดูลที่ใช้ Spark สองเครื่อง (02, 05, 11) |
| `SPARK_API_HOST` | เว้นว่าง หรือ `localhost` เมื่อคุณทำ tunnel พอร์ต (ส่วนที่ 6) | แล็บที่ใช้ HTTP (serving, gateway, เอเจนต์) |

การตั้งค่าถูกบันทึกลง `week25/.env.local` (อยู่ใน gitignore, สิทธิ์ 0600) หรือจะ export ในเชลล์ก็ได้ จากนั้นถาม `sparkkit` ว่าคำสั่งจะไปรันที่ไหน:

```bash
# on: laptop
.venv/bin/python week25/common/sparkkit.py
```

**Expected output** (แล็ปท็อปที่ยังไม่ได้ตั้งค่า Spark บันทึกจาก Mac เครื่องนี้)

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

เมื่อตั้ง `SPARK_HOST` และเชื่อมต่อได้ บรรทัดที่สองจะเปลี่ยนเป็น `▣ LIVE · Spark A · spark-a` และทุกแล็บจะรันจริง

แล็บรันได้สามที่ และจะพิมพ์บอกเสมอ:

- **`[ssh spark-a]`**: แล็บรันบนแล็ปท็อปของคุณ แล้วส่งแต่ละคำสั่งไปที่ Spark ผ่าน SSH
- **`[Spark A (this machine)]`**: คุณเปิด Lab Runner *บน* Spark เอง (แล้วเปิดใช้ผ่าน tunnel)
- **`[DRY]`**: ไม่มีอะไรรัน คุณจะเห็นคำสั่งและผลลัพธ์ที่ติดป้าย RECORDED, REFERENCE หรือ EXAMPLE

สวิตช์ **💻 laptop stand-in** ใช้กับแล็บ HTTP เท่านั้น เมื่อ endpoint บน Spark ล่ม บล็อก ⚡ และแล็บ serving อาจใช้ Ollama บนแล็ปท็อปแทน คำตอบเป็นของจริง แต่จะติดป้าย `LAPTOP STAND-IN` เพราะความเร็วเป็นของแล็ปท็อป ไม่ใช่ของ Spark

✓ Checkpoint: แถบด้านบนแสดง **Spark A: ●** พร้อม host ของคุณ หรือคุณตัดสินใจแล้วว่าจะเรียนตามในโหมด DRY

## 5 · Spark doctor: lab 01

ก่อนดาวน์โหลดโมเดล 200 GB ให้ตรวจพื้นฐานก่อน Lab 01 รันคำสั่ง **อ่านอย่างเดียว (read-only)** แปดคำสั่งบน Spark แต่ละคำสั่งพิมพ์คำสั่งจริงออกมาก่อน คุณจึงคัดลอกไปวางใน ⌨ terminal ได้ (ตั้งเป็น 🟩 Spark A)

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/labs/lab01_spark_doctor.py
```

**Expected output** (โหมด DRY บันทึกจากแล็ปท็อป ถ้ามี Spark คุณจะได้ค่าของเครื่องคุณเองพร้อม ✓ / ✕)

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

playbook ระบุค่าขั้นต่ำไว้ (driver 580.95.05 ขึ้นไป, CUDA 13.0) แต่ไม่ได้พิมพ์บรรทัดเหล่านี้ตรง ๆ แล็บจึงแสดงเป็น EXAMPLE เฉพาะข้อความที่ยกมาคำต่อคำจาก playbook เท่านั้นที่ติดป้าย REFERENCE ซึ่ง `week25/common/audit_references.py` ตรวจเรื่องนี้ให้ทุกแล็บ

แต่ละการตรวจป้องกันปัญหาอะไรในภายหลัง:

| การตรวจ | ถ้าไม่ผ่าน จะพังภายหลังแบบ… |
|---|---|
| `aarch64` | "exec format error" เมื่อคุณ pull container ที่มีแต่ x86 |
| GB10 + driver ≥ 580.95.05 | container เริ่มทำงานได้ แต่มองไม่เห็น GPU |
| CUDA 13 | wheel สำหรับ fine-tuning (`cu130`) import ไม่ผ่าน |
| ว่าง ~119 GiB | หน่วยความจำไม่พอ (out-of-memory) ตอน vLLM จองพื้นที่ KV cache |
| ดิสก์ ≥ 200 GB | ดาวน์โหลดโมเดลแล้วตายที่ 97% |
| Docker โดยไม่ต้อง sudo | ทุก `docker run` ใน Module 03–07 ต้องใช้ `sudo` |

> 💡 128 GB แสดงเป็นราว **119 GiB** ใน `free -g`: 128 × 10⁹ ไบต์ ÷ 2³⁰ ≈ 119 ไม่มีอะไรหายไป

✓ Checkpoint: ในโหมด LIVE ทุกแถวเป็น ✓ หรือทุกแถวที่เป็น ✕ มีวิธีแก้พิมพ์อยู่ข้างใต้ ในโหมด DRY คุณอธิบายได้ว่าแถว REFERENCE คืออะไร และต่างจากแถว EXAMPLE อย่างไร

## 6 · DGX Dashboard และ tunnel เส้นเดียวสำหรับทุกพอร์ต

**DGX Dashboard** คือเว็บแอปในตัวของ Spark: แสดง telemetry ของ GPU และหน่วยความจำ มี JupyterLab แบบคลิกเดียว และใช้อัปเดตระบบ มัน listen ที่ `localhost:11000` **บน Spark เท่านั้น** จึงต้องเข้าผ่าน SSH tunnel (หรือ NVIDIA Sync ซึ่งสร้าง tunnel ให้):

```bash
# on: laptop
ssh -N -L 11000:localhost:11000 spark-a
# now open http://localhost:11000 and log in with your Spark username + password
```

> ℹ `spark-a` คือ alias `Host spark-a` จาก §3 ใน `~/.ssh/config` บนแล็ปท็อปของคุณ ถ้าไม่มีจะเจอ `Could not resolve hostname spark-a` ให้เพิ่ม alias หรือใช้ชื่อเต็มแทน (เช่น `<you>@spark-abcd.<tailnet>.ts.net` หรือ Tailscale IP `100.x`) รันคำสั่งนี้ใน terminal ของแล็ปท็อปเอง ไม่ใช่ ⌨ terminal ของ lab app เพราะถ้า lab app รันอยู่บน Spark เป้าหมาย "laptop" ของมันคือ Spark เครื่องนั้น และ tunnel จะไปไม่ถึงเบราว์เซอร์ของคุณ

จากนั้น [DGX Dashboard playbook](https://build.nvidia.com/spark/dgx-dashboard) จะให้คุณเปิด JupyterLab จาก dashboard รันเซลล์ Stable Diffusion XL และดู GPU utilisation ไต่ขึ้นในแผง telemetry ผู้ใช้แต่ละคนได้พอร์ต JupyterLab ของตัวเอง หาพอร์ตของคุณด้วย `cat /opt/nvidia/dgx-dashboard-service/jupyterlab_ports.yaml` บน Spark แล้วเพิ่ม `-L` อีกตัวสำหรับพอร์ตนั้น

> ⚠ ติดตั้งอัปเดตระบบจาก **Dashboard → Settings → Updates** ไม่ใช่ด้วย `apt upgrade` เปล่า ๆ การอัปเดตผ่าน dashboard อัปเดต firmware ด้วยและจะรีบูต Spark ให้ทำ *ก่อน* เริ่มรัน fine-tuning ที่ใช้เวลานาน

สัปดาห์นี้เปิดบริการ (service) เก้าตัว Lab 03 ตรวจ (probe) ทุกพอร์ต แล้วเขียนคำสั่ง `ssh -L` บรรทัดเดียวสำหรับทั้งหมด:

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/labs/lab03_ports_and_tunnels.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ที่ยังไม่ได้ตั้งค่า Spark: มีแค่ Ollama ของแล็ปท็อปเองที่ตอบ)

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

สองวิธีในการเข้าถึงบริการบน Spark และควรใช้แบบไหนเมื่อไร:

| | ตรงผ่าน tailnet | SSH tunnel |
|---|---|---|
| URL ในแล็บ | `http://spark-a:8000/v1` | `http://localhost:8000/v1` (ตั้ง `SPARK_API_HOST=localhost`) |
| ใช้ได้กับบริการที่ bind กับ `127.0.0.1` บน Spark | ✕ | ✓ |
| ทุกคนใน tailnet เรียกได้ | ได้ | ไม่ได้ เฉพาะแล็ปท็อปของคุณ |
| เหมาะกับ | งานแล็บแบบเร็ว ๆ | Dashboard, Ollama และทุกอย่างที่ไม่มี auth |

> 💡 **VS Code** ([playbook](https://build.nvidia.com/spark/vscode)): ติดตั้งส่วนขยาย *Remote - SSH* แล้วเชื่อมต่อไปที่ `spark-a` เอดิเตอร์ของคุณจะทำงานกับไฟล์บน Spark ด้วย Python ของ Spark NVIDIA Sync เปิดสิ่งนี้ให้ได้ในคลิกเดียว

✓ Checkpoint: DGX Dashboard เปิดได้ที่ `http://localhost:11000` ผ่าน tunnel ของคุณ และคุณบันทึกคำสั่ง tunnel บรรทัดเดียวจาก lab 03 ไว้แล้ว

## 7 · ใส่ได้ไหม? เลขคณิตหน่วยความจำสำหรับทั้งสัปดาห์

สองสูตรนี้ตัดสินว่าคุณรันโมเดลไหนได้ บน Spark กี่เครื่อง และเร็วประมาณเท่าไร:

```text
memory needed ≈ weights + KV cache + overhead
  weights   = parameters × bits per weight ÷ 8
  KV cache  = 2 (K and V) × layers × KV heads × head dim × context tokens × users × 2 bytes

single-stream speed ≤ memory bandwidth ÷ bytes of ACTIVE weights read per token
```

Lab 02 นำสูตรไปใช้กับโมเดลเปิด (open model) หกตัว เป็นการคำนวณล้วน ๆ ผลลัพธ์จึงเหมือนกันทุกเครื่อง:

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/labs/lab02_memory_budget.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

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

บทเรียนสามข้อที่จะกลับมาตลอดสัปดาห์:

1. **ความละเอียดของตัวเลข (precision) คือคันโยกที่ใหญ่ที่สุด** การเปลี่ยนจาก bf16 เป็น NVFP4 ลดขนาด weights ได้ ~3.5× Llama 3.3 70B เปลี่ยนจาก "สอง Spark" เป็น "Spark เครื่องเดียวแถมเหลือที่ว่าง" (Module 07 ทำสิ่งนี้จริง)
2. **Mixture-of-Experts (MoE) เหมาะกับ Spark** gpt-oss-120b เก็บพารามิเตอร์ 117B แต่อ่านแค่ ~5B ต่อ token หน่วยความจำ 128 GB ซื้อคุณภาพ ขณะที่ 273 GB/s ยังคงความเร็วไว้
3. **เพดานนี้คิดต่อหนึ่งสาย (per stream)** serving engine (vLLM, SGLang ใน Module 05–06) รวมผู้ใช้หลายคนเป็น batch เดียวกัน throughput *รวม* จึงสูงกว่าความเร็วของสายเดียวได้หลายเท่า

ตัวเลขเหล่านี้เป็นขอบบนจากการคำนวณ ไม่ใช่ benchmark ตั้งแต่ Module 03 เป็นต้นไปจะวัด tok/s จริงบน Spark ของคุณ แล้วเทียบกับเพดานนี้

✓ Checkpoint: คุณอธิบายได้ว่าทำไม Llama 3.3 70B ต้องใช้สอง Spark ที่ bf16 แต่ใช้เครื่องเดียวที่ NVFP4 และทำไม gpt-oss-120b จึงเร็วกว่า Qwen3 32B ได้ทั้งที่ใหญ่กว่า

## Labs — รันแล็บได้ที่นี่

**labs/lab01_spark_doctor.py** — การตรวจความพร้อมแบบอ่านอย่างเดียวแปดข้อบน Spark ของคุณ แต่ละข้อพิมพ์เป็นคำสั่งจริง แล้วสรุปเป็นตาราง ผ่าน/ไม่ผ่าน

**labs/lab02_memory_budget.py** — weights + KV cache ของโมเดลหกตัวที่สามระดับ precision จัดวางบน Spark หนึ่งหรือสองเครื่อง พร้อมเพดานความเร็วจาก bandwidth

**labs/lab03_ports_and_tunnels.py** — ตรวจทุกพอร์ตที่ใช้ในสัปดาห์นี้ ทั้งบน Spark และบน localhost แล้วพิมพ์คำสั่ง `ssh -L` บรรทัดเดียวสำหรับทั้งหมด

Lab 01 รันแบบ LIVE บน Spark ของคุณหรือแบบ DRY ส่วน Lab 02 และ 03 รันบนแล็ปท็อป Lab 03 จะตรวจ Spark ด้วยเมื่อตั้ง host ไว้

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 01 — เครื่องคิดเลข "ใส่ได้ไหม?"** เปิด `week25/01_meet_your_spark/exercises/ex01_will_it_fit.py` ในไฟล์มี `TODO` สามจุด:

1. `weights_gb(params_b, bits)`: ขนาดของ weights เป็น GB
2. `kv_cache_gb(layers, kv_heads, head_dim, ctx, batch, bytes_per)`: ขนาด KV cache เป็น GB
3. `placement(need_gb)`: คืนค่า `"1 Spark"`, `"2 Sparks"` หรือ `"too big"`

ตัวตรวจทำงานแบบออฟไลน์และฟรี มันเทียบฟังก์ชันของคุณกับคำตอบที่รู้อยู่แล้ว จากนั้นจัดวางโมเดลสามตัว

```bash
# on: laptop
.venv/bin/python week25/01_meet_your_spark/exercises/ex01_will_it_fit.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด)

```
✓ weights_gb: 8B bf16 = 16 GB · 70B nvfp4 ≈ 39.4 GB
✓ kv_cache_gb: Llama 8B @32K ≈ 4.3 GB · Llama 70B @8K × 4 users ≈ 10.7 GB
✓ placement: 100→1 · 128→1 · 129→2 · 256→2 · 300→too big

▣ your calculator, applied (32K context, one user, 10 GB overhead)
│ Qwen3 32B · fp8          needs   51.4 GB → 1 Spark
│ Llama 3.3 70B · bf16     needs  161.9 GB → 2 Sparks
│ Llama 3.1 405B · nvfp4   needs  254.7 GB → 2 Sparks
```

<details><summary>คำใบ้ — "× 2" ใน KV cache มาจากไหน?</summary>

ทุกชั้น attention เก็บ tensor สองตัวต่อหนึ่ง token: **K**eys และ **V**alues ดังนั้น cache จึงเป็นตัวเลข `2 × layers × kv_heads × head_dim` ตัวต่อ token แล้วคูณด้วยจำนวน token จำนวนผู้ใช้ และจำนวนไบต์ต่อตัวเลข

</details>

<details><summary>ท้าทายเพิ่ม — ผู้ใช้ 20 คนที่ context 32K</summary>

เปลี่ยนตัวอย่างที่นำไปใช้เป็น `batch=20` ตอนนี้ Llama 3.3 70B ต้องใช้ KV cache เท่าไร และยังใส่ใน Spark เครื่องเดียวที่ NVFP4 ได้อยู่ไหม? นี่คือการคำนวณเบื้องหลัง `--max-num-seqs` ของ vLLM

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `ssh: Could not resolve hostname spark-xxxx.local` | mDNS ถูกบล็อกในเครือข่ายนี้ ใช้ IP ของ Spark หรือ Tailscale (ส่วนที่ 3) |
| `Permission denied (publickey)` ใน Lab Runner แต่ล็อกอินด้วยรหัสผ่านได้ | BatchMode ไม่ถามรหัสผ่านเลย รัน `ssh-copy-id` (ส่วนที่ 2) และตรวจ `IdentityFile` ใน `~/.ssh/config` |
| แถบด้านบนขึ้นว่า **Spark A: ○ spark-a** | Spark ปิดอยู่ หลับอยู่ หรือหลุดจาก tailnet ลอง `tailscale ping spark-abcd` แล้วกด ↻ |
| `docker: permission denied … docker.sock` | `sudo usermod -aG docker $USER` แล้วล็อกเอาต์และล็อกอินใหม่ |
| `free -g` แสดงพื้นที่ว่างน้อยกว่า 119 GB มาก | มีโปรเซสอื่นถือหน่วยความจำอยู่ (notebook kernel, container ที่หยุดแล้วแต่ยังค้างใน cache) ถ้าหน่วยความจำยังต่ำหลังงานจบแล้ว playbook ให้ล้าง page cache ด้วย: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| tunnel ของ DGX Dashboard เปิดได้แต่หน้าเว็บว่างเปล่า | dashboard bind กับ localhost บน Spark: ทำ tunnel ด้วย `-L 11000:localhost:11000` ไม่ใช่ `-L 11000:spark-a:11000` |

## Next — บทถัดไป

ไปต่อที่ [Lab 02 — Spark สองเครื่อง คลัสเตอร์เดียว](../02_two_sparks_nccl/TUTORIAL.md): ต่อสาย Spark สองเครื่องด้วย QSFP ตั้งค่าลิงก์ 200 Gb/s และวัดผลด้วย NCCL
