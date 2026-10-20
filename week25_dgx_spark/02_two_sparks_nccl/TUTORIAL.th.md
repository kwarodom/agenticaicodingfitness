# ▶ Spark Lab 02 — Spark สองเครื่อง คลัสเตอร์เดียว: QSFP, 200 Gb/s, NCCL

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เรียนรู้ว่า Spark เครื่องที่สองให้อะไรคุณ (256 GB สำหรับโมเดลใหญ่และการเทรนแบบแบ่งส่วน (sharded)) และไม่ได้ให้อะไร (บทสนทนาเดียวที่เร็วขึ้น 2 เท่า)
- ต่อสาย Spark สองเครื่องด้วยสาย QSFP เส้นเดียว และบอกชื่อ interface ของ ConnectX-7 ทั้งสี่ตัวที่แต่ละ Spark แสดง
- กำหนด IP address ให้ลิงก์ด้วย netplan: แล็บเขียนและตรวจไฟล์ให้ ส่วนคุณเป็นคนตัดสินใจว่าจะ apply เมื่อไร
- ตั้งค่า SSH แบบไม่ต้องใช้รหัสผ่านระหว่าง Spark ทั้งสอง และตรวจลิงก์จากทั้งสองฝั่ง
- build และรันการทดสอบ NCCL ของ NVIDIA แล้วเรียนรู้วิธีอ่าน **algbw** เทียบกับ **busbw** เทียบกับเกณฑ์ผ่านของ playbook
- คำนวณด้วยเลขคณิตว่าลิงก์ 200 Gb/s ทำให้โมเดลแบบ tensor-parallel เสียเวลาเท่าไรในทุก token

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 2 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บที่เป็นการคำนวณ)

**Playbook ทางการที่ครอบคลุม:** [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) · [NCCL](https://build.nvidia.com/spark/nccl) · [Connect Multiple Sparks](https://build.nvidia.com/spark/connect-multiple-sparks) · [Connect Three Sparks](https://build.nvidia.com/spark/connect-three-sparks) · [Multiple Sparks Through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จบน **ทั้งสอง** Spark | lab 01-1 เป็น ✓ ทั้งหมดบนแต่ละเครื่อง | driver, CUDA และ Docker ต้องตรงกัน |
| DGX Spark สองเครื่องที่ใช้ DGX OS รุ่นเมษายน 2026 ขึ้นไป | DGX Dashboard → Settings → Updates | Cluster Assistant ของ NVIDIA Sync ต้องใช้ |
| สาย QSFP หนึ่งเส้น | สาย QSFP112 DAC ที่รองรับ ในโหมด Ethernet | สายเส้นเดียวให้ 200 Gb/s เต็ม |
| **username เดียวกัน** บน Spark ทั้งสอง | `whoami` บนแต่ละเครื่อง | `mpirun` และสคริปต์ของ playbook ถือว่าเป็นแบบนั้น |
| SSH alias `spark-a` และ `spark-b` บนแล็ปท็อป | `ssh -o BatchMode=yes spark-b true` | แล็บสั่งงาน Spark ทั้งสองเครื่อง |

ในโมดูลนี้ **Spark A** คือ `SPARK_HOST` (ตัวสั่งเริ่มงาน หรือ "node 1" ใน playbook) และ **Spark B** คือ `SPARK_HOST2` ("node 2") เพิ่ม Spark B ใน **🖥 Spark setup** แล้วถาม `sparkkit` ว่าคำสั่งจะไปรันที่ไหน:

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python week25/common/sparkkit.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ที่ยังไม่ได้ตั้งค่า Spark)

```
━━ sparkkit self-check
   where would commands run, and which endpoints answer?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◈ DRY · SPARK_HOST not set · commands are shown, not run; outputs are RECORDED, REFERENCE or EXAMPLE (labelled)
◆ on a Spark: False · SPARK_HOST=— · SPARK_HOST2=—
```

เมื่อตั้งค่าทั้งสองและเชื่อมต่อได้ บรรทัดที่สองจะเปลี่ยนเป็น `▣ LIVE · Spark A · spark-a · Spark B · spark-b` และจะมีบรรทัด `◆ Spark A: ssh` และ `◆ Spark B: ssh` เพิ่มขึ้นมา

✓ Checkpoint: `ssh -o BatchMode=yes spark-a whoami` และ `ssh -o BatchMode=yes spark-b whoami` ตอบกลับด้วย username เดียวกันทั้งคู่ หรือคุณตัดสินใจแล้วว่าจะเรียนตามในโหมด DRY

## 1 · Spark สองเครื่องให้อะไร และไม่ได้ให้อะไร

Spark หนึ่งเครื่องมี unified memory 128 GB และอ่านได้ที่ 273 GB/s Spark เครื่องที่สองที่เชื่อมด้วยสาย 200 Gb/s ให้คุณสองอย่าง:

| สิ่งที่ได้ | เพราะ | ใช้ใน |
|---|---|---|
| **256 GB สำหรับโมเดลเดียว** | tensor parallelism (TP=2) วางครึ่งหนึ่งของทุกชั้นไว้บนแต่ละ Spark | Module 05: vLLM serve โมเดลที่ใส่ใน Spark เครื่องเดียวไม่ได้ |
| **การเทรนแบบแบ่งส่วน (sharded)** | FSDP แบ่ง weights, gradients และ optimizer states ไปไว้ทั้งสองเครื่อง | Module 11: Llama 3.1 70B LoRA ใน bf16 |
| **สองเครื่องที่แยกกันทำงาน** | รันโมเดลหรืองานคนละอย่างบนแต่ละเครื่อง | Module 08: gateway ตัวเดียวอยู่หน้าทั้งสองเครื่อง |

สิ่งที่คุณ **ไม่ได้** คือบทสนทนาที่เร็วขึ้นสองเท่า แต่ละ token ยังต้องอ่าน weights จากหน่วยความจำที่ 273 GB/s ต่อ Spark TP=2 ลดจำนวนไบต์ที่แต่ละ Spark ต้องอ่านลงครึ่งหนึ่ง แต่ทุกชั้นต้องรอการแลกเปลี่ยนข้อมูลเล็ก ๆ สองครั้งผ่านสาย และสายช้ากว่าหน่วยความจำราว 12 เท่า:

```text
memory (per Spark)   ████████████████████████████  273 GB/s
QSFP link            ██░░░░░░░░░░░░░░░░░░░░░░░░░░   25 GB/s  (200 Gb/s ÷ 8)
```

ดังนั้น weights ไม่เคยวิ่งผ่านสาย มีแค่ activations เท่านั้น Lab 04 (ส่วนที่ 7) แปลงเรื่องนี้เป็นตัวเลขต่อ token

✓ Checkpoint: คุณบอกชื่อโมเดลหนึ่งตัวที่ต้องใช้ Spark สองเครื่องได้ (Module 01, lab 02 แสดง Llama 3.1 405B ที่ NVFP4) และบอกได้ว่าทำไม Spark เครื่องที่สองจึงไม่ทำให้แชตเดียวเร็วขึ้นสองเท่า

## 2 · สาย: QSFP, ConnectX-7 และชื่อ interface สี่ชื่อ

Spark แต่ละเครื่องมีชิปเครือข่าย **ConnectX-7** พร้อม **พอร์ต QSFP สองพอร์ต** ที่ด้านหลัง เสียบสายหนึ่งเส้นระหว่าง Spark ทั้งสอง กฎของ playbook: **ใช้พอร์ตกายภาพเดียวกันบน Spark ทั้งสองเครื่อง** เพื่อเลี่ยงปัญหาในการทดสอบ NCCL ภายหลัง "Port 0" คือพอร์ต QSFP ที่อยู่ข้างพอร์ต Ethernet ส่วน "port 1" คือพอร์ตที่อยู่ไกลออกไป (Connect Three Sparks, Step 2)

Linux แสดงพอร์ตกายภาพแต่ละพอร์ตเป็น **interface เชิงตรรกะ (logical) สองตัว** Spark หนึ่งเครื่องจึงมีสี่ตัว `ibdev2netdev` จับคู่อุปกรณ์ RDMA แต่ละตัวกับ network interface ของมัน และบอกว่าตัวไหน Up:

```bash
# on: spark
ibdev2netdev
```

**Expected output** (REFERENCE — ยกมาจาก Connect Two Sparks playbook: เสียบสายไว้ที่ port 1)

```
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

วิธีอ่านชื่อ:

| ชื่อ | คืออะไร |
|---|---|
| `enp1s0f1np1`, `enP2p1s0f1np1` | **interface เชิงตรรกะสองตัวของพอร์ตกายภาพ 1** (`f1`) ส่วน port 0 คือ `enp1s0f0np0` + `enP2p1s0f0np0` |
| `rocep1s0f1`, `roceP2p1s0f1` | อุปกรณ์ **RoCE** (RDMA over Converged Ethernet) ที่คู่กัน NCCL และ `ib_write_bw` ใช้ตัวเหล่านี้ |
| `enP7s7` | พอร์ต Ethernet ปกติ: เครือข่าย **management** ที่คุณใช้ SSH (Wi-Fi: `wlP9s9`) |

ข้อเท็จจริงสองข้อจาก playbook ที่คนมักพลาด:

- **สายเส้นเดียวก็ได้ bandwidth เต็ม** ถ้าใช้สองเส้น interface ทั้งสี่ตัวต้องมี IP address จึงจะได้ bandwidth เต็ม
- **interface เชิงตรรกะทั้งสองตัวของพอร์ตคุณต้องมี address คนละ subnet** RDMA benchmark ของ NVIDIA รันการทดสอบหนึ่งครั้งต่อ interface เชิงตรรกะหนึ่งตัวแล้วรวมกัน: 92.57 + 97.28 = 189.85 Gb/s (lab 03 คำนวณค่านี้ใหม่จากตัวอย่างของ playbook)

Lab 01 รันการตรวจเหล่านี้บน Spark ทั้งสองพร้อมกัน รันตอนนี้เพื่อดูว่าอะไร Up แล้วรันอีกครั้งหลังส่วนที่ 5

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab01_link_check.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ แล็บป้อนรายการ interface, address และความเร็วจาก playbook ให้ตัวเอง บวกข้อความตัวอย่างสำหรับส่วนที่เหลือ ผลการรันด้านบนพิมพ์บอกว่าอะไรเป็นอะไร)

```
▣ STEP 3 · compare the two ends
│ node     RDMA device   netdev         QSFP    IPv4               MTU   speed     user    iface list
│ ───────  ────────────  ─────────────  ──────  ─────────────────  ────  ────────  ──────  ──────────
│ Spark A  rocep1s0f1    enp1s0f1np1    port 1  192.168.100.10/24  1500  200 Gb/s  nvidia  reference
│ Spark A  roceP2p1s0f1  enP2p1s0f1np1  port 1  192.168.101.10/24  1500  200 Gb/s  nvidia  reference
│ Spark B  rocep1s0f1    enp1s0f1np1    port 1  192.168.100.11/24  1500  200 Gb/s  nvidia  reference
│ Spark B  roceP2p1s0f1  enP2p1s0f1np1  port 1  192.168.101.11/24  1500  200 Gb/s  nvidia  reference
◈ on this text: same physical port Up on both ends (port 1: enp1s0f1np1 + enP2p1s0f1np1)
◈ on this text: enp1s0f1np1: both ends on one subnet (192.168.100.10/24 ↔ 192.168.100.11/24)
◈ on this text: enP2p1s0f1np1: both ends on one subnet (192.168.101.10/24 ↔ 192.168.101.11/24)
◈ on this text: each logical interface has its own subnet (192.168.100.x, 192.168.101.x)
◈ on this text: MTU is the same on every link interface (1500)
◈ on this text: every link interface negotiated 200000 Mb/s
◈ on this text: same username on both Sparks (nvidia) — mpirun and the helper scripts need it
```

ในโหมด DRY การตรวจจะพิมพ์ `◈ on this text:` แทน ✓ เพราะมันทดสอบข้อความจาก playbook ไม่ใช่สายของคุณ ในโหมด LIVE คุณจะได้ ✓ หรือ ✕ พร้อมวิธีแก้ในทุกบรรทัดที่เป็น ✕

✓ Checkpoint: บน Spark ทั้งสอง `ibdev2netdev` แสดง interface สองตัว **เดียวกัน** เป็น `(Up)` ถ้าไม่มีตัวไหน Up ให้เสียบสายใหม่ รีบูต แล้วตรวจอีกครั้ง

## 3 · ทางง่าย: NVIDIA Sync Cluster Assistant

ตั้งแต่สิงหาคม 2026 ทุก playbook เรื่องการเชื่อมต่อเริ่มต้นด้วยคำแนะนำเดียวกัน: **ใช้ Cluster Assistant ของ NVIDIA Sync** และถ้ามันรายงานว่าสำเร็จ ให้ **ข้ามขั้นตอนตั้งค่าเครือข่ายและ SSH แบบทำเอง** ไปได้เลย บนแล็ปท็อปของคุณ ใน NVIDIA Sync (Connect Multiple Sparks playbook, "Configure with NVIDIA Sync"):

1. เพิ่ม Spark แต่ละเครื่อง (**Add New** → ชื่อหรือ IP, user name, รหัสผ่าน)
2. **Settings → Cluster Assistant → Add New Cluster** ตั้งชื่อ แล้วเลือก Spark ทั้งสอง
3. Sync ตรวจ SSH ฮาร์ดแวร์ ซอฟต์แวร์ระบบ และ `sudo` จากนั้นตรวจหาสายและแสดง **แผนเครือข่าย (network plan)** ตรวจดูแล้วเลือก **Confirm Network Configuration**
4. มันรันการทดสอบความเร็วบนแต่ละลิงก์ ลิงก์จะเป็นสีเขียวเมื่อถึง **ขอบล่าง 184 Gbit/s**
5. มันตั้งค่า SSH แบบใช้คีย์ระหว่าง Spark ทั้งสอง และเพิ่ม SSH alias ให้แต่ละเครื่อง
6. ในหน้าสำเร็จ กด **Copy** รายละเอียดเครือข่ายแล้วบันทึกลงไฟล์

Cluster Assistant ตั้งค่าเฉพาะเครือข่ายเท่านั้น มันไม่ได้ติดตั้งหรือรัน NCCL, vLLM หรือการเทรน สิ่งเหล่านั้นคือส่วนที่เหลือของโมดูลนี้และ Module 05 กับ 11

> 💡 แม้ Cluster Assistant จะทำให้ทุกอย่างแล้ว ก็ให้รัน lab 01 และ lab 03 อยู่ดี มันแสดงให้คุณเห็นว่า Cluster Assistant ตั้งค่า *อะไร* ไปบ้าง (ชื่อ interface, subnet) และวัดลิงก์ด้วย NCCL

✓ Checkpoint: Cluster Assistant แสดงลิงก์ทั้งสองเป็นสีเขียว (ข้ามไปส่วนที่ 6) หรือคุณไปต่อทางทำเองในส่วนที่ 4 และ 5

## 4 · ทางทำเอง: IP address ด้วย netplan

ถ้าไม่ใช้ Cluster Assistant คุณต้องกำหนด static IP ให้ interface ของลิงก์เอง Connect Two Sparks playbook (Step 3, Option 1) เขียนไฟล์ netplan หนึ่งไฟล์ต่อ Spark: interface เชิงตรรกะสองตัวของ port 1 อยู่บน /24 subnet **คนละวง** โดย node 1 ลงท้ายด้วย `.10` และ node 2 ลงท้ายด้วย `.11` (switch playbook เตือนว่า interface สองตัวบน subnet เดียวกันทำให้ routing กำกวมและ NCCL ล้มเหลว)

Lab 02 อ่านว่า interface ไหน Up บนแต่ละ Spark เขียนไฟล์ทั้งสองลงใน `week25/02_two_sparks_nccl/.runs/` บนแล็ปท็อป (อยู่ใน gitignore) ตรวจความถูกต้อง และในโหมด LIVE จะคัดลอกแต่ละไฟล์ไปไว้ที่ `~/w25/40-cx7.yaml` บน Spark ของมัน มัน **ไม่เปลี่ยนแปลงอะไรใน `/etc`** เว้นแต่คุณจะรันจากเชลล์พร้อม `SPARK_APPLY=1` และถึงอย่างนั้นก็ต่อเมื่อมี `sudo -n` แบบไม่ต้องใช้รหัสผ่าน และลิงก์ยังไม่มี address เท่านั้น ปุ่ม ▶ ของ Lab Runner ไม่เคยตั้ง `SPARK_APPLY`

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab02_netplan_plan.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้: สร้างจากรายการ interface ของ playbook)

```
▣ STEP 3 · the two files
── week25/02_two_sparks_nccl/.runs/40-cx7.spark-a.yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.10/24
      dhcp4: no
── week25/02_two_sparks_nccl/.runs/40-cx7.spark-b.yaml
…
✓ both files parse as YAML with network.version 2 and one entry per Up interface
✓ generated from the playbook's interface list, both files match the playbook's Step 3 files byte for byte
```

apply เองทีละ Spark ใน ⌨ terminal นี่คือคำสั่งของ playbook สำหรับ **node 1** (Spark A):

```bash
# on: spark
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.10/24
      dhcp4: no
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

บน **node 2** (Spark B) รันแบบเดียวกันโดยใช้ `.11` แทน `.10`:

```bash
# on: spark-b
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.11/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.11/24
      dhcp4: no
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

ถ้าสายของคุณเสียบอยู่ที่ port 0 ให้ใช้ชื่อ interface ที่ lab 02 พิมพ์ออกมาแทน ตัวเลือกอื่นจาก playbook:

| ตัวเลือก | คำสั่ง | อยู่รอดหลังรีบูตไหม? |
|---|---|---|
| ไฟล์ netplan (ด้านบน) | `sudo netplan apply` | ใช่ |
| IP ชั่วคราว | `sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1 && sudo ip link set enp1s0f1np1 up` (และทำแบบเดียวกันกับ `enP2p1s0f1np1` ด้วย `192.168.101.10/24`) | ไม่ |
| **ย้อนกลับ (rollback)** | `sudo rm /etc/netplan/40-cx7.yaml && sudo netplan apply` | — |

> ⚠ ไฟล์นี้ระบุเฉพาะ interface ของ ConnectX-7 ไม่ได้ระบุ `enP7s7` แต่ `netplan apply` ยังอาจทำให้เครือข่ายสะดุดชั่วครู่ได้ ดังนั้นครั้งแรกควรมีช่องทางเข้าเครื่องสำรองไว้ (console บนเครื่อง หรือเซสชัน SSH ที่สอง)

✓ Checkpoint: `ip addr show enp1s0f1np1` แสดง `inet 192.168.100.10/24` บน Spark A และ `inet 192.168.100.11/24` บน Spark B

## 5 · SSH แบบไม่ต้องใช้รหัสผ่านระหว่าง Spark และลิงก์ที่สะอาด

`mpirun` (ส่วนที่ 6) และการเทรนหลายโหนด (Module 11) เริ่มโปรเซสบน Spark B **จาก Spark A** ดังนั้น Spark A ต้องเข้า Spark B ผ่าน SSH ได้โดยไม่ต้องใช้รหัสผ่าน ตัวเลือกแบบอัตโนมัติของ playbook ค้นหา Spark อีกเครื่องด้วย mDNS (`avahi-browse` จาก `avahi-utils`) แล้วแชร์คีย์ร่วมกันหนึ่งคีย์:

```bash
# on: spark
mkdir -p ~/.ssh && chmod 700 ~/.ssh
curl -fsSL -o discover-sparks https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
bash ./discover-sparks
```

**Expected output** (REFERENCE — ยกมาจาก playbook มันจะถามรหัสผ่านของคุณหนึ่งครั้งต่อโหนด)

```
Found: 192.168.100.10 (node-1.local)
Found: 192.168.100.11 (node-2.local)

Setting up shared SSH access across all nodes...
You may be prompted for your password on each node.

Shared SSH setup complete!
All nodes can now SSH to each other using the shared key (id_ed25519_shared).
```

รัน `mkdir -p ~/.ssh && chmod 700 ~/.ssh` บน Spark B ด้วย สคริปต์จะล้มเหลวถ้าไม่มี `~/.ssh` ตัวเลือกแบบทำเองบน Spark ทั้งสอง: `ssh-copy-id -i ~/.ssh/id_ed25519.pub <username>@192.168.100.10` และ `…@192.168.100.11`

**MTU: ปล่อยไว้ตามเดิม แต่ตรวจว่าทั้งสองฝั่งตรงกัน** playbook ของ Spark ใช้ MTU ค่าเริ่มต้น (ตัวอย่าง `ip addr` แสดง `mtu 1500` และ multi-Spark playbook บอกให้ใช้ค่าเริ่มต้นสำหรับลิงก์ผ่าน switch) MTU 9000 ที่คุณอาจเคยเห็นเป็นของ playbook DGX **Station** (ConnectX-8) ไม่ใช่ Spark ทราฟฟิก RDMA ต่อรอง path MTU ของตัวเอง: ตัวอย่าง RDMA พิมพ์ `Mtu : 1024[B]` ส่วนเสริมของคอร์สที่พิสูจน์ว่า 1500 ใช้ได้ตลอดเส้นทาง (ข้อมูล 1472 ไบต์ + header 28 ไบต์ แบบ "do not fragment"):

```bash
# on: spark
ping -c 3 -M do -s 1472 -I enp1s0f1np1 192.168.100.11
```

ตอนนี้รัน lab 01 อีกครั้ง step 4 ของมันจะ ping Spark B ข้ามสาย และรัน `ssh -o BatchMode=yes 192.168.100.11 hostname` จาก Spark A

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab01_link_check.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ คำตอบของ ping และ ssh เป็นรูปแบบ EXAMPLE)

```
▣ STEP 4 · across the cable: ping, then passwordless SSH from Spark A to Spark B
$ ping -c 3 -W 2 -I enp1s0f1np1 192.168.100.11   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
…
3 packets transmitted, 3 received, 0% packet loss
◈ on this text: Spark A reaches 192.168.100.11 over the QSFP link
$ ssh -o BatchMode=yes -o ConnectTimeout=5 192.168.100.11 hostname   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
spark-b
◈ on this text: passwordless SSH A → B over the link works (answered: spark-b)
```

✓ Checkpoint: ในโหมด LIVE ทุกบรรทัดของ lab 01 เป็น ✓: พอร์ตเดียวกัน หนึ่ง subnet ต่อ interface MTU เท่ากัน 200000 Mb/s user เดียวกัน และ ping กับ SSH ข้ามสายได้

## 6 · NCCL: build รัน และอ่าน algbw เทียบกับ busbw

**NCCL** (NVIDIA Collective Communication Library) คือสิ่งที่ PyTorch, vLLM และ TensorRT-LLM เรียกใช้เพื่อย้าย tensor ระหว่าง GPU รวมถึงข้ามสายของคุณด้วย NCCL playbook build NCCL **v2.30.7-1** สำหรับ Blackwell (`sm_121`) และ benchmark `nccl-tests` ของ NVIDIA บน **ทั้งสอง** Spark มันต้องใช้ `sudo` สำหรับแพ็กเกจหนึ่งตัว จึงให้รันเองใน ⌨ terminal บน Spark A ก่อน แล้วจึงบน Spark B:

```bash
# on: spark
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.30.7-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

(quick start ของ playbook ทำทั้งสอง Spark จาก Spark A: `bash setup.sh <NODE_2_IP>` แล้ว `bash launch.sh --topology direct <NODE_1_IP> <NODE_2_IP>`)

การทดสอบเริ่ม **ครั้งเดียว บน Spark A** `mpirun` จะ SSH เข้า Spark B แล้วเริ่ม rank ที่สองที่นั่น สังเกตว่าแต่ละ flag ชี้ไปที่อะไร:

| Flag | ค่า | ทำไม |
|---|---|---|
| `-H <A>:1,<B>:1` | IP ของเครือข่าย **management** (`ip addr show enP7s7`) | ที่ที่ mpirun SSH เข้าไป หนึ่งโปรเซสต่อ Spark |
| `NCCL_SOCKET_IFNAME`, `UCX_NET_DEVICES`, `OMPI_MCA_btl_tcp_if_include` | `enP7s7` | interface สำหรับทราฟฟิกช่วงตั้งค่า ถ้าใช้ Wi-Fi ให้ใช้ `wlP9s9` บน **ทุก** โหนด |
| `-b 16G -e 16G -f 2` | ข้อความขนาด 16 GB หนึ่งก้อน | ใหญ่พอที่จะใช้ลิงก์เต็ม |

ข้อมูลก้อนใหญ่วิ่งผ่านอุปกรณ์ RoCE ซึ่ง NCCL หาเองได้ ถ้าจะดูว่ามันเลือกตัวไหน ให้เพิ่ม `-x NCCL_DEBUG=INFO` แล้วมองหาบรรทัดที่มี `NET/IB` (เคล็ดลับของคอร์ส)

Lab 03 ตรวจการ build บน Spark ทั้งสอง หา IP ของ management ทั้งสองเครื่อง รันคำสั่ง `all_gather_perf` 16 GB ของ playbook และอ่านผลลัพธ์:

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab03_nccl_bench.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ playbook ไม่ได้พิมพ์ผลลัพธ์ของ NCCL ไว้ ตารางจึงเป็น EXAMPLE ที่เลือกตัวเลขให้เห็นการคำนวณชัด ไม่ใช่ค่าที่วัดได้)

```
▣ STEP 4 · read the table
│ size    time      algbw       busbw       busbw × 8
│ ──────  ────────  ──────────  ──────────  ─────────
│ 16 GiB  390.5 ms  44.00 GB/s  22.00 GB/s  176 Gb/s
◆ algbw = size ÷ time = 17.18 GB ÷ 0.390 s = 44.00 GB/s
◆ busbw = algbw × (n−1)/n = 44.00 × 0.5 = 22.00 GB/s  (nccl-tests printed 22.00)
│ busbw     █████████████████████████░░░  22.00 GB/s
│ pass mark ████████████████████████░░░░  21.875 GB/s  (playbook script, direct link)
│ line rate ████████████████████████████  25.00 GB/s  (200 Gb/s ÷ 8)
◈ on this EXAMPLE text the verdict would be pass — not your link

▣ STEP 5 · parser self-test on the playbook's published RDMA sample (REFERENCE, runs everywhere)
│ RDMA write test          BW average
│ ───────────────────────  ────────────────────────
│ client 1 (rocep1s0f0)    92.57 Gb/s
│ client 2 (roceP2p1s0f0)  97.28 Gb/s
│ total                    189.85 Gb/s = 23.73 GB/s
✓ parser total 189.85 Gb/s matches the playbook's stated 189.85 Gbps
```

**algbw เทียบกับ busbw** `algbw` คือ *ขนาดข้อความ ÷ เวลา* ง่าย ๆ แต่ collective แต่ละแบบย้ายข้อมูลต่อ GPU ไม่เท่ากัน: ใน all-gather ที่มี *n* rank แต่ละ GPU รับข้อมูลจากตัวอื่นแค่ (n−1)/n ของผลลัพธ์ ส่วนใน all-reduce มันส่งและรับ 2(n−1)/n ของผลลัพธ์ nccl-tests คูณ algbw ด้วยตัวคูณนั้นเพื่อให้ได้ **busbw** ซึ่งคืออัตราที่ *สาย* ขนส่งจริง ทำให้ busbw เปรียบเทียบข้ามประเภท operation และจำนวน GPU ได้ และเทียบกับ 25 GB/s ของสายได้ สำหรับ Spark สองเครื่อง:

| Operation | ตัวคูณ busbw (n = 2) | ใช้โดย |
|---|---|---|
| `all_gather` (การทดสอบของ playbook) | (n−1)/n = 0.5 | FSDP ที่รวบรวม weight shard (Module 11) |
| `all_reduce` (`--op all_reduce`) | 2(n−1)/n = 1 | tensor parallel (Module 05), การ sync gradient |

**ผ่านไหม?** สคริปต์คลัสเตอร์ของ NVIDIA เอง (`spark_cluster_setup.py` ใน assets ของ Connect Multiple Sparks) อ่านบรรทัด `# Avg bus bandwidth` และเตือนเมื่อต่ำกว่า **21.875 GB/s (175 Gbps)** สำหรับลิงก์ตรงหรือผ่าน switch และต่ำกว่า **10 GB/s (80 Gbps)** สำหรับวงแหวน (ring) สาม Spark Lab 03 ใช้บรรทัดเดียวกันและตัวเลขเดียวกัน

สำหรับวัด fabric ดิบโดยไม่ผ่าน NCCL คู่มือ benchmark รัน `ib_write_bw` ของ perftest โดยมี server หนึ่งตัวต่อ interface เชิงตรรกะบน Spark A และ client ตัวละหนึ่งบน Spark B (ติดตั้ง `sudo apt install perftest` บนทั้งสองเครื่องก่อน) Lab 03 รันแบบมีจุดจบด้วย `--rdma` ส่วนคำสั่งของ playbook พร้อม IP ตัวอย่างของมันคือ:

```bash
# on: spark
ib_write_bw -d rocep1s0f0 -i 1 -p 12000 -F --report_gbits --run_infinitely
```

```bash
# on: spark-b
ib_write_bw -d rocep1s0f0 -i 1 -p 12000 -F --report_gbits 192.168.200.12 --run_infinitely
```

ทำซ้ำด้วย `roceP2p1s0f0` พอร์ต `12001` และ address ของ subnet ที่สองในเทอร์มินัลอีกสองหน้าต่าง แล้วรวมคอลัมน์ `BW average` ทั้งสอง ใช้อุปกรณ์ที่ Up ของคุณเองและ address ของ Spark A

✓ Checkpoint: ในโหมด LIVE lab 03 พิมพ์ `Avg bus bandwidth … ≥ 21.875 GB/s` จดค่า busbw ของคุณไว้ lab 04 และ Module 11 จะใช้มัน

## 7 · ลิงก์ทำให้โมเดล tensor-parallel เสียอะไรบ้าง

เมื่อ TP=2 แต่ละชั้นจบด้วย **all-reduce สองครั้ง** ของเวกเตอร์ hidden state หนึ่งตัวต่อ token (bf16: hidden size × 2 ไบต์) ซึ่งเล็กมาก: 16 KiB ต่อ all-reduce สำหรับโมเดล 70B การอ่าน weights ลดเหลือครึ่งหนึ่งต่อ Spark สิ่งที่สะสมขึ้นคือ **latency** คงที่ α ของแต่ละ all-reduce ซึ่งต้องจ่าย 2 × จำนวนชั้น ครั้งต่อ token playbook ไม่ได้เผยแพร่ตัวเลข latency ไว้ lab 04 จึงถือ α เป็นสมมติฐานแล้วไล่ค่า (sweep):

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab04_tp_link_cost.py
```

**Expected output** (การคำนวณ บันทึกจาก Mac เครื่องนี้; busbw = เกณฑ์ผ่าน 21.875 GB/s ของ playbook, ขนาด weights คิดแบบ NVFP4)

```
▣ STEP 2 · bytes that cross the link per decoded token (batch 1, bf16 activations)
│ model                  layers  hidden  message   all-reduces  per token  wire time
│ ─────────────────────  ──────  ──────  ────────  ───────────  ─────────  ─────────
│ Llama 3.1 8B           32      4096    8.0 KiB   64           0.52 MB    24 µs
│ Llama 3.3 70B          80      8192    16.0 KiB  160          2.62 MB    120 µs
│ gpt-oss-120b (MoE)     36      2880    5.6 KiB   72           0.41 MB    19 µs
│ Qwen3 235B-A22B (MoE)  94      4096    8.0 KiB   188          1.54 MB    70 µs
│ Llama 3.1 405B         126     16384   32.0 KiB  252          8.26 MB    377 µs

◆ α = 100 µs per all-reduce (assumed)
│ model                  1 Spark         TP=2       of which link  speed-up        TP=2 tok/s ceiling
│ ─────────────────────  ──────────────  ─────────  ─────────────  ──────────────  ──────────────────
│ Llama 3.1 8B             16.5 ms         14.7 ms    6.4 ms       1.12×             68.2
│ Llama 3.3 70B           145.5 ms         88.9 ms   16.1 ms       1.64×             11.3
│ gpt-oss-120b (MoE)       10.5 ms         12.5 ms    7.2 ms       0.84×             80.2
│ Qwen3 235B-A22B (MoE)     — (too big)    41.5 ms   18.9 ms       fits only on 2    24.1
│ Llama 3.1 405B            — (too big)   442.8 ms   25.6 ms       fits only on 2     2.3

▣ STEP 4 · prefill is different: a 4,096-token prompt sends 4,096× the bytes
│ Llama 3.3 70B           10.74 GB          0.49 s
═ Two Sparks buy CAPACITY first: 256 GB holds Llama 405B or Qwen3 235B at 4-bit. …
```

(ผลลัพธ์ฉบับเต็มยังแสดง α = 10 µs และ 30 µs ด้วย ที่ 10 µs โมเดล 70B ได้ถึง 1.95× ส่วน 8B ได้แค่ 1.85×)

บทเรียนสามข้อ:

1. **ไบต์ตอน decode มีน้อย แต่ latency ไม่น้อย** token หนึ่งของ Llama 3.3 70B ส่ง 2.6 MB ข้ามสาย คิดเป็นเวลาบนสายราว 120 µs แต่ latency ของการไป-กลับ 160 รอบอาจแพงกว่านั้น
2. **โมเดล dense ขนาดใหญ่ได้ประโยชน์ โมเดลเล็กหรือแบบ sparse อาจเสียเปรียบ** gpt-oss-120b อ่าน weights แค่ ~5B ต่อ token อยู่แล้ว ถ้า α ช้า TP=2 จะ *ช้ากว่า* Spark เครื่องเดียว (0.84×) ให้ serve มันบน Spark เครื่องเดียว แล้วใช้เครื่องที่สองทำอย่างอื่น
3. **prompt ยาวทำให้สายมีภาระหนัก** การ prefill 4,096 token ของโมเดล 70B ส่งข้อมูล ~10.7 GB คิดเป็นราวครึ่งวินาทีที่ busbw

ตัวเลขเหล่านี้เป็นขอบบนจากการคำนวณ ไม่ใช่ benchmark: ไม่นับการประมวลผล การ launch kernel และการทำงานซ้อนกัน (overlap) วัด α ด้วยการทดสอบ latency ของ playbook (`ib_write_lat -d rocep1s0f0 -i 1 -p 13000 -F` บน Spark A และคำสั่งเดียวกันบวก IP ของ Spark A บน Spark B) แล้วรันใหม่ด้วย `--alpha-us <yours> --busbw <lab 03's value>` Module 05 วัด tok/s จริงของ vLLM บน Spark หนึ่งและสองเครื่อง

✓ Checkpoint: คุณอธิบายได้ว่าทำไม TP=2 ช่วย Llama 3.3 70B มากกว่า gpt-oss-120b และบอกชื่อตัวเลขหนึ่งตัวที่ lab 04 ต้องสมมติขึ้น

## 8 · สาม Spark, สี่ Spark: วงแหวนและ switch

แนวคิดเดียวกันขยายไปยัง Spark จำนวนมากขึ้นได้ Connect Multiple Sparks playbook อนุญาตการจัดวางสามแบบ และบอกว่าอย่าผสมลิงก์ตรงกับลิงก์ผ่าน switch:

| จำนวน Spark | การจัดวาง | สาย | หมายเหตุจาก playbook |
|---|---|---|---|
| 2 | **Direct** (ตรง) | 1 | โมดูลนี้ |
| 3 | **Direct ring** (วงแหวนตรง) | 3: A→B, B→C, C→A | Spark แต่ละเครื่องใช้พอร์ต QSFP **ทั้งสอง** (Port0 → Port1 ของโหนดถัดไป) interface ทั้งสี่ตัวจึงได้ IP บน subnet ของตัวเอง |
| 2–4 (มากกว่านั้นต้องทำเอง) | **Switch** | 1 เส้นต่อ Spark | ทุกพอร์ตที่ต่อกับ Spark ต้องเป็น 200 Gbit/s อยู่บน Layer 2 bridge เดียวกัน ใช้ MTU ค่าเริ่มต้น; สี่ Spark ต้องใช้ switch |

สิ่งที่เปลี่ยนไปในทางปฏิบัติ:

- **การทดสอบ NCCL แบบวงแหวน:** playbook เพิ่ม `-x NCCL_IB_SUBNET_AWARE_ROUTING=1 -x NCCL_NET_PLUGIN=none` ให้ `mpirun` และสคริปต์ของ NVIDIA คาดหวัง busbw แค่ **10 GB/s** (แต่ละช่วง (hop) เป็นลิงก์ตรงแยกกัน)
- **Switch:** ตรวจว่า `ethtool enp1s0f1np1 | grep Speed` แสดง `Speed: 200000Mb/s` บนทุกโหนด auto-negotiation อาจไปหยุดที่ `100000Mb/s` ถ้าเป็นเช่นนั้นให้ตั้ง 200G ที่พอร์ตของ switch และปิด auto-negotiation switch แจก IP ด้วย DHCP ได้ (`dhcp4: true` ใน netplan) หรือคุณกำหนด `192.168.100.1–4` / `192.168.101.1–4` เองก็ได้
- **Cluster Assistant** จัดการ Spark สองถึงสี่เครื่องได้ในทุกการจัดวางเหล่านี้ ถ้ามากกว่าสี่เครื่อง ให้ใช้ขั้นตอนทำเองของ switch playbook

✓ Checkpoint: คุณบอกได้ว่าวงแหวนสาม Spark ต้องใช้สายกี่เส้น และใช้ flag ของ NCCL อะไรเพิ่ม

## Labs — รันแล็บได้ที่นี่

**labs/lab01_link_check.py** — ตรวจลิงก์ QSFP บน Spark ทั้งสอง: interface ที่ Up, address, MTU, ความเร็ว, user แล้ว ping และ SSH แบบไม่ใช้รหัสผ่านข้ามสาย

**labs/lab02_netplan_plan.py** — เขียนและตรวจไฟล์ netplan ของแต่ละ Spark จาก interface ที่ Up; apply เฉพาะเมื่อรันจากเชลล์ด้วย SPARK_APPLY=1

**labs/lab03_nccl_bench.py** — รันการทดสอบ NCCL สอง Spark ของ playbook แยกค่า algbw และ busbw แล้วเทียบกับเกณฑ์ผ่านของ NVIDIA

**labs/lab04_tp_link_cost.py** — การคำนวณ: ไบต์ เวลาบนสาย และ latency ที่ลิงก์เพิ่มให้ทุก token แบบ tensor-parallel สำหรับโมเดลห้าตัว

Lab 01–03 รันแบบ LIVE บน Spark สองเครื่องหรือแบบ DRY ส่วน Lab 04 เป็นการคำนวณและรันได้ทุกที่ Lab 02 ไม่เคยแตะ `/etc` จากปุ่ม ▶ และ parser self-test ของ lab 03 รันบนตัวอย่างที่ NVIDIA เผยแพร่ในทุกโหมด

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 02 — อ่านลิงก์ให้ออก** เปิด `week25/02_two_sparks_nccl/exercises/ex02_read_the_link.py` ในไฟล์มี `TODO` สามจุด:

1. `up_interfaces(text)`: ชื่อ netdev ที่ `ibdev2netdev` รายงานว่า `(Up)`
2. `bus_factor(op, n)`: ตัวคูณที่แปลง algbw เป็น busbw
3. `verdict(busbw, topology)`: `"pass"` หรือ `"low"` เทียบกับเกณฑ์ผ่านของ NVIDIA

ตัวตรวจทำงานแบบออฟไลน์และฟรี มันใช้ตัวอย่าง `ibdev2netdev` จริงจาก playbook สามฉบับ

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/exercises/ex02_read_the_link.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด บันทึกจาก Mac เครื่องนี้)

```
✓ up_interfaces: two-Spark sample → port 1 (2 netdevs) · NCCL sample → port 0 · ring sample → all 4
✓ bus_factor: all_reduce n=2 → 1 · all_gather n=2 → 0.5 · all_reduce n=4 → 1.5 · broadcast → 1
✓ verdict: 22 direct → pass · 21 direct → low · 12 ring → pass · 9.5 ring → low · 21.875 switch → pass

▣ your functions, applied to three practice runs (EXAMPLE numbers, not measurements)
│ all_gather n=2 direct algbw  44.6 GB/s → busbw 22.30 GB/s (178.4 Gb/s) → pass
│ all_reduce n=2 direct algbw  19.0 GB/s → busbw 19.00 GB/s (152.0 Gb/s) → low
│ all_gather n=3 ring   algbw  16.2 GB/s → busbw 10.80 GB/s ( 86.4 Gb/s) → pass
```

<details><summary>คำใบ้ — ทำไมต้องเทียบ busbw ไม่ใช่ algbw กับสาย?</summary>

ใน all-gather แบบสอง rank แต่ละ Spark มีครึ่งหนึ่งของผลลัพธ์อยู่แล้ว จึงมีแค่อีกครึ่งที่ต้องข้ามสาย algbw นับทั้งข้อความและดูเร็วเป็นสองเท่าของสาย busbw = algbw × (n−1)/n นับเฉพาะส่วนที่เดินทางจริง ซึ่งคือสิ่งที่สาย 25 GB/s เป็นตัวจำกัด

</details>

<details><summary>ท้าทายเพิ่ม — วางแผนวงแหวนสาม Spark</summary>

ดู netplan ของ Connect Three Sparks playbook: node 1 มี `192.168.0.1`, `192.168.1.1`, `192.168.2.1`, `192.168.3.1` จดว่าสายแต่ละเส้นขนส่ง subnet ไหน และ interface ใดบน node 2 และ node 3 ที่อยู่ subnet เดียวกับแต่ละ interface ของ node 1 ทำไมวงแหวนจึงต้องใช้หก subnet สำหรับสายสามเส้น?

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| ไม่มี interface ใดแสดง `(Up)` ใน `ibdev2netdev` | เสียบสาย QSFP ใหม่ในพอร์ต **เดียวกัน** บน Spark ทั้งสอง รีบูตทั้งสองเครื่อง แล้วตรวจอีกครั้ง |
| "Network unreachable" | ไฟล์ netplan หายไปหรือยังไม่ได้ apply ตรวจ `ip addr show enp1s0f1np1` แล้วรัน `sudo netplan apply` |
| ทั้งสองฝั่ง Up แต่ ping ข้ามลิงก์ไม่ได้ | ปลายทั้งสองอยู่คนละ subnet หรือมี interface เชิงตรรกะแค่ตัวเดียวที่มี address รัน lab 01 ใหม่ |
| `discover-sparks` เขียนคีย์ไม่สำเร็จ | `mkdir -p ~/.ssh && chmod 700 ~/.ssh` บน Spark ทั้งสอง แล้วลองใหม่ |
| `discover-sparks`: `avahi-browse not found` | ติดตั้ง `avahi-utils` บน Spark ทั้งสอง |
| `mpirun` ค้างหรือหมดเวลา | SSH ระหว่าง Spark ยังถามรหัสผ่าน ทดสอบ `ssh <node-2 management IP>` จาก Spark A แล้ว `mpirun -np 2 -H <A>:1,<B>:1 hostname` |
| การทดสอบ NCCL: `libnccl.so.2: cannot open shared object file` | NCCL ยังไม่ได้ build บนทุกโหนด หรือไม่ได้ export `LD_LIBRARY_PATH` ก่อน `mpirun` |
| busbw ต่ำกว่า 21.875 GB/s มาก | หยุดงาน GPU อื่นแล้วรันใหม่ ตรวจว่า interface เชิงตรรกะทั้งสองมี address และความเร็วเป็น `200000Mb/s` |
| ลิงก์ผ่าน switch อยู่ที่ `100000Mb/s` | ตั้งพอร์ตของ switch เป็น 200G และปิด auto-negotiation (ดูคู่มือของผู้ผลิต switch) |
| Cluster Assistant: การตรวจซอฟต์แวร์ไม่ผ่าน | อัปเดต Spark ทั้งสองเป็นรุ่นเมษายน 2026 ขึ้นไป |

## Next — บทถัดไป

ไปต่อที่ [Lab 03 — Ollama + Open WebUI](../03_ollama_open_webui/TUTORIAL.md): model server ตัวแรกของคุณบน Spark พร้อม chat UI ในเบราว์เซอร์
