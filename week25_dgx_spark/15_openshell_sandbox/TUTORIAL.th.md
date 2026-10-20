# ▶ Spark Lab 15 — OpenShell: sandbox และกำกับดูแล AI agent

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เรียนรู้ส่วนประกอบของ OpenShell: gateway บน Spark, sandbox, provider, เส้นทาง `inference.local` และ policy แบบ YAML
- ติดตั้ง OpenShell บน Spark ตามที่ playbook ทำทุกประการ และติดตั้ง CLI บนแล็ปท็อปจาก venv ภายใน week25
- route การเรียกโมเดลของ sandbox ไปที่ vLLM บน Spark ของคุณ โดยไม่มี API key ออกจากเครื่อง
- เขียน policy ให้เอเจนต์โรงแรม NAT จาก Module 14 แล้วตรวจสองวิธี: ด้วย parser จริงของ OpenShell และ validator ที่คอร์สสร้างขึ้น
- ถามว่า "policy นี้จะอนุญาตอะไรบ้าง?" กับรายการการกระทำของเอเจนต์ และดูล่วงหน้าว่าการเปลี่ยน policy จะเปิดอะไรเพิ่ม
- นำเอเจนต์ NAT ไปไว้ใน sandbox บน Spark ดูการตัดสินใจของมัน และพิสูจน์ว่ามันออกอินเทอร์เน็ตไม่ได้

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: lab 15-2, 15-3 และแบบฝึกหัดทำงานแบบออฟไลน์ทั้งหมด)

**Playbook ทางการที่ครอบคลุม:** [OpenShell](https://build.nvidia.com/spark/openshell) (`nvidia/playbook-openshell` และ `nvidia/openshell` ตัวเก่ากว่า) โครงสร้าง policy เป็นแบบที่ NVIDIA ใส่มาใน healthcare-agent playbook (`assets/sandbox-policy.yaml`) ส่วนกฎของ policy มาจากตาราง troubleshooting ของ OpenShell และ NemoClaw-applications playbook

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 14 เสร็จแล้ว | `week25/.venv-nat/bin/nat --version` | เอเจนต์ที่คุณจะใส่ใน sandbox |
| ทำ Module 05 บน Spark เสร็จแล้ว | vLLM ตอบที่ `:8000` โดยเริ่มด้วย `--host 0.0.0.0` | โมเดลที่ sandbox จะ route ไปหา |
| Docker โดยไม่ต้อง sudo บน Spark พร้อม NVIDIA runtime | `docker ps` บน Spark | OpenShell รันคลัสเตอร์ k3s ภายใน Docker |
| Python ≥ 3.12 บน Spark | `python3 --version` | เป็น prerequisite ของ playbook |
| clone repo นี้ไว้บน Spark | `ls ~/agenticaicodingfitness/week25` | lab 15-4 อัปโหลดโมดูล NAT จากที่นั่นเข้าไปใน sandbox |

> ⚠ คำสั่งแรกของ playbook: **ใช้ environment ที่สะอาด** รันเอเจนต์บน Spark (หรือ VM) ที่ไม่มีข้อมูลส่วนตัว บัญชีจริง หรือ credential ของ production OpenShell ลดความเสี่ยงในการรันเอเจนต์ แต่ไม่ได้ขจัดความเสี่ยงออกไปทั้งหมด

✓ Checkpoint: vLLM บน Spark ตอบ `curl -s http://localhost:8000/v1/models` และ `docker ps` ใช้ได้โดยไม่ต้อง sudo

## 1 · OpenShell คืออะไร ในภาพเดียว

เอเจนต์ที่อ่านไฟล์ รันโปรแกรม และเรียก API ได้ มีประโยชน์และอันตรายด้วยเหตุผลเดียวกัน **OpenShell** คือ sandbox runtime แบบ open-source ของ NVIDIA สำหรับเอเจนต์: มันรันเอเจนต์ใน sandbox ที่แยกขาด และบังคับใช้ policy แบบ YAML เชิงประกาศ (declarative) ว่าเอเจนต์อ่าน เขียน รัน และเข้าถึงอะไรได้บ้าง

| ส่วนประกอบ | คืออะไร | คำสั่ง |
|---|---|---|
| **Gateway** | control plane บน Spark เป็น systemd user service ที่รัน k3s ภายใน Docker | `openshell status` |
| **Sandbox** | environment ของเอเจนต์หนึ่งตัวที่แยกขาด สร้างจาก image (`--from base`, `--from openclaw`, Dockerfile) | `openshell sandbox create / exec / connect / delete` |
| **Policy** | YAML: `filesystem_policy`, `landlock`, `process`, `network_policies` | `openshell policy get / set / update` |
| **Provider** | credential + config ที่ตั้งชื่อไว้ ซึ่ง sandbox ใช้ได้โดยไม่เห็นความลับ | `openshell provider create` |
| **inference.local** | hostname ภายในทุก sandbox ที่ proxy จะ route ไปยังโมเดลของคุณ | `openshell inference set` |

```text
  DGX Spark
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │  openshell-gateway (systemd user service · k3s in Docker)                      │
 │    ┌─ sandbox "hotel-agent" ───────────────────────────────┐                   │
 │    │  NAT agent (python) ── https://inference.local/v1 ──┐  │   provider        │
 │    │   Landlock: write only /sandbox /tmp               │  │   local-vllm ──►  vLLM :8000
 │    │   runs as user "sandbox"                    egress proxy ──────────────►  (host IP, not localhost)
 │    │   curl https://api.openai.com  ──────────►  ✕ deny (not in policy)         │
 │    └────────────────────────────────────────────────────────┘                   │
 └───────────────────────────────────────────────────────────────────────────────┘
```

การควบคุมสามแบบ กลไกสามแบบ กฎด้าน **filesystem** บังคับใช้โดย Linux kernel (Landlock) และตายตัวตั้งแต่ตอนสร้าง sandbox กฎด้าน **process** เลือกผู้ใช้ที่ไม่มีสิทธิ์พิเศษ กฎด้าน **network** บังคับใช้โดย egress proxy: ทุกอย่างถูกปฏิเสธ เว้นแต่ policy group จะระบุ host, port *และ* โปรแกรมที่อนุญาตให้ใช้ไว้ กฎ network โหลดใหม่ได้ทันที (hot-reload) ส่วนกฎ filesystem ต้องสร้าง sandbox ใหม่

✓ Checkpoint: คุณบอกได้ว่ากฎแบบไหนในสามแบบที่เปลี่ยนได้ขณะ sandbox กำลังรันอยู่

## 2 · ติดตั้ง OpenShell บน Spark (playbook Step 1–4)

ตรวจ environment (Step 1) และ Docker (Step 2) ถ้า `docker ps` แจ้ง permission denied playbook จะเพิ่มคุณเข้ากลุ่มด้วย `sudo usermod -aG docker $USER` (แล้วล็อกเอาต์และล็อกอินใหม่) การตั้งค่า NVIDIA runtime ก็ต้องใช้ `sudo` เช่นกัน จึงให้พิมพ์คำสั่งเหล่านี้เอง:

```bash
# on: spark
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

ติดตั้ง CLI ด้วยตัวติดตั้งทางการ (Step 3) มันติดตั้ง `openshell` และลงทะเบียน systemd user service ชื่อ `openshell-gateway` จากนั้นตรวจ gateway (Step 4):

```bash
# on: spark
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
source ~/.bashrc
openshell --help
systemctl --user status --no-pager openshell-gateway
openshell status
sudo loginctl enable-linger $USER
```

`openshell status` ควรรายงานว่า gateway เป็น **Connected** การเริ่มครั้งแรกอาจใช้เวลาหลายนาทีระหว่างที่ Docker pull image และ k3s บูต ติดตามได้ด้วย `journalctl --user -u openshell-gateway -f`

> ⚠ **"วิธีติดตั้งทางเลือก" ของ playbook (`uv pip install openshell`) ไม่ได้ให้ CLI แล้ว** บน PyPI `openshell` 0.1.x (0.1.2 เมื่อ 2026-09-28) เป็นแค่ Python SDK: เป็น wheel `py3-none-any` ตัวเดียวที่ไม่มีคำสั่ง `openshell` release สุดท้ายที่มีไบนารี CLI มาด้วยคือ wheel รุ่น 0.0.x (0.0.111 มี build `macosx_13_0_arm64` และ `manylinux_2_39_aarch64`) บน Spark ให้ใช้ตัวติดตั้งทางการตามด้านบน

**บนแล็ปท็อปของคุณ** คอร์สติดตั้ง CLI 0.0.111 ตัวนั้นลงใน venv เฉพาะของโมดูล — ไม่มีสคริปต์ติดตั้ง ไม่ต้อง sudo — เพื่ออ่าน help ของมัน ตรวจไฟล์ policy ของคุณด้วย parser ของมัน (ส่วนที่ 4) และ (ถ้าต้องการ) จัดการ gateway ของ Spark จากระยะไกล (`openshell gateway add https://openshell:8080 --remote <user>@<spark-ip>`, playbook Step 9) CLI รุ่น 0.0.x ที่คุยกับ gateway รุ่น 0.1.x ไม่รับประกันว่าจะใช้ได้ สำหรับการจัดการระยะไกล ควรใช้เวอร์ชันที่ตรงกัน

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
uv venv -p 3.12 week25/15_openshell_sandbox/.venv-openshell
uv pip install --python week25/15_openshell_sandbox/.venv-openshell/bin/python "openshell==0.0.111"
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_1_install_and_gateway.py
```

**Expected output** (ครึ่งฝั่งแล็ปท็อป บันทึกจาก Mac เครื่องนี้ ส่วนครึ่งฝั่ง Spark เป็น DRY ในที่นี้)

```
▣ STEP B1 · the OpenShell CLI on this laptop (openshell==0.0.111 in week25/15_openshell_sandbox/.venv-openshell)
$ openshell --version   [this laptop]
openshell 0.0.111
…
✓ 16 top-level commands, including sandbox, logs, policy, provider, gateway, inference, term

▣ STEP B2 · status and prerequisites on the laptop (no gateway here — the output says so)
$ openshell status   [this laptop]
Gateway Status
  Status: No gateway configured.
Register a gateway with: openshell gateway add <endpoint>
$ openshell doctor check   [this laptop]
Checking system prerequisites...
  Docker ............. FAILED
Error:   × docker info failed: Cannot connect to the Docker daemon at unix:///var/
  │ run/docker.sock. Is the docker daemon running?
```

"No gateway configured" เป็นข้อความเดียวกับแถวใน troubleshooting ของ playbook สำหรับ gateway ที่ไม่เคยเริ่มทำงาน บน Spark คุณต้องการ **Connected**

✓ Checkpoint: บน Spark `openshell status` ขึ้นว่า Connected บนแล็ปท็อป `openshell --version` พิมพ์ 0.0.111 (ไม่บังคับ)

## 3 · Route การเรียกโมเดลของ sandbox ไปที่ Spark ของคุณ (playbook Step 5–7)

ภายใน sandbox เอเจนต์เรียก `https://inference.local/v1` proxy ของ OpenShell ส่งต่อไปยัง **provider** ที่คุณกำหนด เอเจนต์จึงไม่เคยถือคีย์จริง และไม่ต้องมีเส้นทางอินเทอร์เน็ตไปหาโมเดลเลย

เริ่ม vLLM ด้วยสูตรที่พร้อมสำหรับเอเจนต์ของ playbook จาก Module 05 โดยคง `--host 0.0.0.0` และพอร์ต 8000 ไว้ gateway รันภายใน Docker จึงเข้าถึง vLLM ผ่าน IP address ของ Spark ไม่ใช่ `localhost`:

```bash
# on: spark
curl -sf http://localhost:8000/v1/models
export HARDWARE_IP="$(hostname -I | awk '{print $1}')"
test -n "$HARDWARE_IP"
curl -sf "http://${HARDWARE_IP}:8000/v1/models"
```

สร้าง provider (vLLM ไม่ต้องใช้คีย์ placeholder ที่ไม่ว่างค่าใดก็ใช้ได้) แล้วชี้ `inference.local` ไปที่โมเดล:

```bash
# on: spark
openshell provider create \
    --name local-vllm \
    --type openai \
    --credential OPENAI_API_KEY=not-needed \
    --config OPENAI_BASE_URL="http://${HARDWARE_IP}:8000/v1"
openshell provider list
openshell inference set \
    --provider local-vllm \
    --model nvidia/Qwen3.6-35B-A3B-NVFP4
openshell inference get
```

`openshell inference get` ควรแสดง `provider: local-vllm` และโมเดลของคุณ ถ้า `inference set` แจ้ง `failed to verify inference endpoint` ให้อุ่นเครื่อง vLLM ด้วยคำขอแชตหนึ่งครั้งก่อน `--no-verify` ข้ามการตรวจนี้ได้เมื่อคุณรู้แล้วว่า API บน host ใช้ได้

✓ Checkpoint: `openshell inference get` ระบุ `local-vllm` และ `nvidia/Qwen3.6-35B-A3B-NVFP4`

## 4 · ไฟล์ policy และสองวิธีในการตรวจ

policy ของคอร์สสำหรับเอเจนต์โรงแรม NAT คือ `policies/hotel_agent_policy.yaml` มันลอกโครงสร้างของ policy ที่ NVIDIA ใส่มากับ healthcare-agent playbook:

```yaml
version: 1
filesystem_policy:
  include_workdir: true
  read_only:  [/usr, /lib, /proc, /dev/urandom, /etc]      # the OS: readable, never writable
  read_write: [/sandbox, /tmp, /dev/null]                  # the agent's own files
landlock:
  compatibility: best_effort                               # the shipped policy's choice (see its comment)
process:
  run_as_user: sandbox
  run_as_group: sandbox
network_policies:                                          # a MAP of groups, never a list
  inference:
    name: inference
    endpoints:
      - host: inference.local
        port: 443
    binaries:                                              # which programs may use this egress
      - { path: "/sandbox/.venv-nat/bin/python*" }
      - { path: "/usr/bin/python3*" }
      - { path: /usr/bin/curl }
  pypi:                                                    # setup-only: removed after NAT is installed
    name: pypi
    endpoints:
      - { host: pypi.org, port: 443, access: full, tls: skip }
      - { host: files.pythonhosted.org, port: 443, access: full, tls: skip }
    binaries:
      - { path: "/sandbox/.venv-nat/bin/python*" }
      - { path: "/sandbox/.venv-nat/bin/pip*" }
      - { path: "/usr/bin/python3*" }
```

กฎที่ policy ต้องปฏิบัติตามกระจายอยู่ในสาม playbook รวบรวมไว้ที่เดียวดังนี้:

| กฎ | แหล่งที่มา | ตรวจโดย |
|---|---|---|
| คีย์ระดับบนสุดคือ `version`, `filesystem_policy`, `landlock`, `process`, `network_policies` (`version` เป็นตัวพิมพ์เล็ก) | troubleshooting ของ OpenShell + NemoClaw-applications (`unknown field 'Version'`) | parser ของ OpenShell CLI |
| `network_policies` เป็น map ที่ใช้ชื่อ group เป็นคีย์ | NemoClaw-applications (`invalid type: sequence, expected a map`) | parser ของ OpenShell CLI |
| path ขึ้นต้นด้วย `/` และไม่มี `..` | troubleshooting ของ OpenShell ("Policy push returns exit code 1") | gateway ตอน push |
| `run_as_user` ไม่ใช่ `root` | แถวเดียวกัน | gateway ตอน push |
| ทุก endpoint มี `host` และ `port` | แถวเดียวกัน | gateway ตอน push |
| แต่ละ group ต้องมี `binaries` และแต่ละ endpoint ต้องมี access mode ไม่เช่นนั้นการเรียกจะล้มเหลวด้วย `CONNECT tunnel failed, response 403` | NemoClaw-applications | ไม่มีใครตรวจ — apply ผ่านเรียบร้อยแล้วค่อยปฏิเสธ |

Lab 15-2 สร้าง policy จาก spec ภาษา Python สั้น ๆ แล้วตรวจมันและรุ่นที่เสียแปดแบบด้วย **policykit** (`week25/15_openshell_sandbox/policykit.py` ซึ่งเป็น validator ที่คอร์สสร้างขึ้นสำหรับกฎข้างต้น) และด้วย **parser จริงของ OpenShell** เคล็ดลับคือ: `openshell policy set` parse ไฟล์บนแล็ปท็อปของคุณก่อนจะติดต่อ gateway ดังนั้นการชี้ไปที่พอร์ตที่ปิดอยู่ (`OPENSHELL_GATEWAY_ENDPOINT=http://127.0.0.1:9`) จะแยก "parse error" ออกจาก "parse ผ่าน แล้วเชื่อมต่อไม่ได้" ได้ — ไม่ต้องมี gateway

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_2_policy_builder.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ตัดให้สั้นลง)

```
✓ generated policy == policies/hotel_agent_policy.yaml (same keys, same values)

▣ STEP 2 · eight broken variants — each one is a row in the playbooks' troubleshooting tables
│ policy                            policykit.validate  openshell CLI parser
│ ────────────────────────────────  ──────────────────  ────────────────────────────────────────────────────
│ policies/hotel_agent_policy.yaml  ✓ ok                ✓ parsed
│ healthcare playbook (shipped)     ⚠ 3 warning(s)      ✓ parsed
│ Version (capital V)               ✕ 2 error(s)        ✕ unknown field `Version`, expected one of `version…
│ network_policies as a list        ✕ 1 error(s)        ✕ network_policies: invalid type: sequence, expecte…
│ relative / .. path                ✕ 2 error(s)        ✓ parsed
│ run_as_user: root                 ✕ 1 error(s)        ✓ parsed
│ endpoint without port             ✕ 1 error(s)        ✓ parsed
│ group without binaries            ⚠ 1 warning(s)      ✓ parsed
│ endpoint without access mode      ⚠ 1 warning(s)      ✓ parsed
│ endpoint with description:        ✕ 1 error(s)        ✕ network_policies.pypi.endpoints.\[0\]: unknown fi…
```

อ่านคอลัมน์ไปพร้อมกัน parser จับเรื่อง **โครงสร้าง** (ฟิลด์ที่ไม่รู้จัก, list กับ map, ชนิดข้อมูลผิด) บนแล็ปท็อปของคุณ ส่วนกฎเชิง **ความหมาย (semantic)** จะถูกตีกลับจาก gateway ก็ต่อเมื่อคุณ push ซึ่งเป็นเหตุผลที่คอร์สตรวจสิ่งเหล่านี้ก่อน หมายเหตุตรงไปตรงมาสามข้อจากการสร้างตารางนี้:

- **policy สองตัวของ NVIDIA ขัดกันเอง** healthcare policy ที่ใส่มามีรายการ `{host, port}` เปล่า ๆ สำหรับ npm และ PyPI ขณะที่ NemoClaw-applications playbook บอกว่ารายการเปล่าคือ "the single most common reason" (สาเหตุที่พบบ่อยที่สุด) ของ 403 policykit จึงเตือน และ policy ของคอร์สทำตามกฎที่เข้มกว่า (`access: full, tls: skip`)
- **ตัวอย่างใน runbook ตัวหนึ่ง parse ไม่ผ่านกับ CLI 0.0.111** ตัวอย่าง "Modifying the Policy" ใน RUNBOOK ของ healthcare ใส่ `description:` ไว้ใน endpoint ซึ่ง parser 0.0.111 ปฏิเสธ (`unknown field`) gateway รุ่นใหม่กว่าอาจต่างไป — ตรวจด้วย `openshell policy set --wait` บน Spark ของคุณ
- **IP ส่วนตัว (private IP)** คอมเมนต์ใน policy ที่ใส่มาบอกว่า IP ส่วนตัว/loopback แบบดิบถูกบล็อก "regardless of policy entries" (ไม่ว่ารายการใน policy จะเป็นอย่างไร) แต่ไฟล์เดียวกันกลับอนุญาต IP ของ Docker bridge ให้บริการหนึ่ง และ CLI ก็มี `allowed_ips` ให้ route การเรียกโมเดลผ่าน `inference.local` แล้วคำถามนี้จะไม่เกิดขึ้นเลย

✓ Checkpoint: policy ของคุณเป็น `✓ ok` ในทั้งสองคอลัมน์ และคุณบอกได้ว่าทำไม "relative / .. path" parse ผ่าน แต่ก็ยังจะถูกปฏิเสธ

## 5 · policy นี้จะอนุญาตอะไร? การเฝ้าดูและการปฏิเสธ

ก่อนจะ push policy ให้ถามว่ามันยอมให้เอเจนต์ทำอะไรได้บ้าง Lab 15-3 เล่นซ้ำการกระทำสิบสามรายการ — เอเจนต์โรงแรมที่ทำงานตามหน้าที่ และเอเจนต์ที่ถูก prompt injection พยายามทำข้อมูลรั่ว — ผ่าน **`policykit.decide`** นี่คือ **แบบจำลองเพื่อการสอนที่คอร์สสร้างขึ้น** ของความหมายที่ playbook อธิบายไว้ (path ของ Landlock, egress แบบปฏิเสธโดยปริยาย, binaries ต่อ group, การ route inference, กฎ method ระดับ L7) **ไม่ใช่ตัว OpenShell เอง** เหตุผลของมันจะระบุ "(course assumption)" ทุกจุดที่ playbook ไม่ได้พูดถึง

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_3_what_would_it_allow.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เป็นแบบจำลองของ policykit ไม่ใช่ OpenShell)

```
│ who     action                                                decision     why (the rule that decided)
│ ──────  ────────────────────────────────────────────────────  ───────────  ────────────────────────────────────────────────────
│ agent   write /sandbox/.runs/tickets.jsonl                    ✓ allow      under read_write /sandbox
│ agent   python3.12 → inference.local:443                      → inference  inference.local is handled by the proxy's inference…
│ setup   pip → pypi.org:443                                    ✓ allow      network_policies.pypi: pypi.org:443 for binary /san…
│ hijack  python3.12 → api.openai.com:443                       ✕ deny       api.openai.com:443 is not in network_policies (defa…
│ hijack  curl → pypi.org:443                                   ✕ deny       pypi.org:443 is listed, but not for binary /usr/bin…
│ hijack  read  /home/sandbox/.ssh/id_ed25519                   ✕ deny       not in read_only or read_write (Landlock: Permissio…
│ hijack  write /etc/cron.d/backdoor                            ✕ deny       /etc is read_only (Landlock: Permission denied)
│ hijack  run as root                                           ✕ deny       the sandbox runs as 'sandbox'; it cannot become 'ro…
◆ 7 of 13 actions denied · hijack attempts that got through: none · legitimate actions blocked: none

▣ STEP 2 · preview a change: add GitHub read-only for curl (like `openshell policy update --add-endpoint … --dry-run`)
│ action                                                before  after
│ ────────────────────────────────────────────────────  ──────  ───────  ─────────
│ curl → api.github.com:443 GET /repos/NVIDIA/OpenShe…  ✕ deny  ✓ allow  ◆ CHANGED
│ curl → api.github.com:443 POST /repos/NVIDIA/OpenSh…  ✕ deny  ✕ deny
│ python3.12 → api.github.com:443 GET /                 ✕ deny  ✕ deny
```

สังเกต `curl → pypi.org`: host นั้นได้รับอนุญาต แต่เฉพาะสำหรับ pip และ Python **รายการ binaries คือสิ่งที่หยุดเอเจนต์ที่ถูกยึดไม่ให้ใช้ host ที่อนุญาตด้วยเครื่องมืออื่น**

บน Spark คุณจะเห็นการตัดสินใจจริงแบบสด Step 11 ของ playbook ใช้ terminal UI ส่วน troubleshooting ใช้ log stream:

```bash
# on: spark
openshell term
openshell logs hotel-agent --tail --source sandbox
```

`openshell term` แสดงสถานะของแต่ละ sandbox และ log แบบสดของการเชื่อมต่อขาออกพร้อมการตัดสินใจ: `allow`, `deny` หรือ `inspect_for_inference` (กด `f` เพื่อติดตาม, `s` เพื่อกรองตามแหล่งที่มา, `q` เพื่อออก) วิธีเปลี่ยนสิ่งที่อนุญาต:

| การเปลี่ยนแปลง | คำสั่ง | มีผลเมื่อ |
|---|---|---|
| เพิ่ม endpoint หนึ่งตัว | `openshell policy update hotel-agent --add-endpoint api.github.com:443:read-only:rest:enforce --binary /usr/bin/curl --wait` | hot-reload |
| ดูตัวอย่างก่อน | คำสั่งเดียวกันพร้อม `--dry-run` | ไม่ส่งอะไรไป |
| ลบ group | `openshell policy update hotel-agent --remove-rule pypi --wait` | hot-reload |
| แทนที่ policy ทั้งหมด | `openshell policy get hotel-agent > p.yaml` (ไม่ใช้ `--full`) แก้ไข แล้ว `openshell policy set hotel-agent --policy p.yaml --wait` | hot-reload |
| เปลี่ยนกฎ filesystem | ลบแล้วสร้าง sandbox ใหม่ | เฉพาะ sandbox ใหม่ |

(flag ของ `policy update` มาจาก help ของ CLI 0.0.111 และการใช้ `policy get` โดยไม่มี `--full` หลีกเลี่ยง error `unknown field 'Version'` ในตาราง troubleshooting ของ playbook)

✓ Checkpoint: คุณอธิบายได้ว่าทำไม `curl → pypi.org` ถูกปฏิเสธ ทั้งที่ `pypi.org:443` อยู่ใน policy

## 6 · นำเอเจนต์โรงแรม NAT ไปไว้ใน sandbox

ตอนนี้รวม Module 14 และ 15 เข้าด้วยกัน Lab 15-4 พิมพ์ — และเมื่อใส่ `--yes` จะรัน — ลำดับทั้งหมดบน Spark มัน **ออกแบบโดยคอร์ส**: ขั้น provider และ inference เป็นของ playbook ส่วน sandbox สำหรับเอเจนต์ NAT (แทน image OpenClaw ของ playbook) เป็นของเรา และ **ยังไม่เคยรันบน Spark จริง**

```bash
# on: spark
cd ~/agenticaicodingfitness/week25
mkdir -p ~/w25 && cp 15_openshell_sandbox/policies/hotel_agent_policy.yaml ~/w25/
openshell sandbox create --name hotel-agent --from base --policy ~/w25/hotel_agent_policy.yaml \
  --upload ~/agenticaicodingfitness/week25/14_nat_agents:/sandbox/14_nat_agents --keep --no-tty -- true
openshell sandbox exec -n hotel-agent -- python3 -m venv /sandbox/.venv-nat
openshell sandbox exec -n hotel-agent -- /sandbox/.venv-nat/bin/pip install -q 'nvidia-nat[langchain]~=1.9' greenlet
openshell sandbox exec -n hotel-agent -- /sandbox/.venv-nat/bin/pip install -q --no-deps -e /sandbox/14_nat_agents/hotel_ops_nat
openshell policy update hotel-agent --remove-rule pypi --wait
openshell sandbox exec -n hotel-agent --workdir /sandbox/14_nat_agents \
  --env HOTEL_LLM_BASE_URL=https://inference.local/v1 --env HOTEL_LLM_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --env HOTEL_TICKET_LOG=/sandbox/tickets.jsonl \
  -- /sandbox/.venv-nat/bin/nat run --config_file configs/hotel_agent_spark.yml --input "Room 808 feels hot. Check it and open a ticket if something is wrong."
openshell sandbox exec -n hotel-agent -- curl -s --max-time 5 https://api.openai.com || echo 'blocked ✓'
```

จากแล็ปท็อป ทำแบบเดียวกันด้วยแล็บ (DRY พิมพ์แต่ละคำสั่งพร้อม output แบบ EXAMPLE เมื่อเชื่อม Spark แล้วจะรันการตรวจแบบอ่านอย่างเดียว และจะเปลี่ยนแปลงอะไรก็ต่อเมื่อใส่ `--yes`):

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/labs/lab15_4_sandbox_the_nat_agent.py
```

**Expected output** (EXAMPLE — รูปแบบเพื่อประกอบคำอธิบาย ไม่ใช่ค่าที่วัดได้)

```
▣ STEP 6 · run the agent inside the sandbox — its only way out is inference.local
$ openshell sandbox exec -n hotel-agent --workdir /sandbox/14_nat_agents --env HOTEL_LLM_BASE_URL=https://inference.local/v1 … nat run --config_file configs/hotel_agent_spark.yml --input "Room 808 feels hot. …"   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
Workflow Result:
Room 808 is too warm … Created maintenance ticket MT-A83AD2 (high priority).

▣ STEP 7 · prove the fence: the same sandbox cannot reach the internet (runbook pattern: these should fail)
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
curl: (56) CONNECT tunnel failed, response 403
blocked ✓
```

**ครั้งแรกที่คุณรันสิ่งนี้บน Spark ให้ตรวจหกข้อนี้** — แต่ละข้อเป็นสมมติฐานที่คอร์สทดสอบไม่ได้หากไม่มี Spark:

1. `--from base` มี `python3` พร้อม `venv` และ `pip` ถ้าไม่มี ให้ build image เล็ก ๆ (`--from <dir with a Dockerfile>`) ที่มีสิ่งเหล่านี้
2. `--upload <path>:<path>` คัดลอกโฟลเดอร์ไปไว้ที่ที่แล็บคาดไว้ (`/sandbox/14_nat_agents`)
3. Python เข้าถึง `inference.local` ผ่านการตั้งค่า proxy ของ sandbox และเชื่อถือ certificate ของมัน การทดสอบของ playbook เองคือ `curl https://inference.local/v1/chat/completions` ภายใน sandbox (Step 10) ถ้า TLS ล้มเหลวสำหรับ Python ให้ตั้ง `--env HOTEL_LLM_VERIFY_SSL=false` สำหรับการเรียกครั้งนั้นครั้งเดียว แล้วรายงานกลับมา
4. `--remove-rule pypi` ลบ group ตามชื่อคีย์ของมัน
5. `openshell logs hotel-agent --source sandbox` แสดง `deny` สำหรับ api.openai.com และ `inspect_for_inference` สำหรับการเรียกโมเดล
6. `--keep --no-tty -- true` ทำให้ sandbox ยังรันอยู่หลังจาก `true` จบ (playbook ใช้ `--keep` กับคำสั่งแบบโต้ตอบ help ของ 0.0.111 แสดงแค่ `--no-keep` แต่ก็ยอมรับ `--keep`)

✓ Checkpoint: คำตอบของเอเจนต์อ้างถึง id ของตั๋ว และ `curl` ไปที่ api.openai.com ล้มเหลวจากภายใน sandbox เดียวกัน

## 7 · sandbox ของ playbook เอง: OpenClaw (Step 8–13)

playbook เองนำ **OpenClaw** ซึ่งเป็นเอเจนต์แบบ local-first ใส่ sandbox จาก community image ที่มี policy รวมมาด้วย Module 16 และ 17 ต่อยอดจากสิ่งนี้ นี่คือแก่นของมัน อย่าใส่ `--policy` คู่กับ `--from openclaw` เพราะ policy มากับ image แล้ว:

```bash
# on: spark
export SANDBOX_NAME=openshell-demo
openshell sandbox create \
  --keep \
  --tty \
  --forward 18789 \
  --name "$SANDBOX_NAME" \
  --from openclaw \
  -- openclaw-start
```

onboarding wizard เป็นแบบโต้ตอบ (ใช้ปุ่มลูกศร): เลือก **Custom Provider**, API base URL `https://inference.local/v1`, คีย์ placeholder ใดก็ได้, **OpenAI-compatible** และ model handle เดียวกับ `openshell inference set` หลังจากหนึ่งถึงสองนาที มันจะพิมพ์ URL ของแดชบอร์ด:

**Expected output** (REFERENCE — อ้างอิงจาก playbook)

```
OpenClaw gateway starting in background.
  Logs: /tmp/gateway.log
  UI:   http://127.0.0.1:18789/?token=<unique-token>
```

จากแล็ปท็อป ให้ forward พอร์ตด้วย `openshell forward start --background 18789 "$SANDBOX_NAME"` แล้วเปิด `http://127.0.0.1:18789/#token=<your-token>` เมื่อเสร็จแล้ว ให้เก็บกวาดขณะที่ gateway ยังทำงานอยู่ (Step 13):

```bash
# on: spark
openshell sandbox delete "$SANDBOX_NAME"
openshell sandbox delete hotel-agent
openshell provider delete local-vllm
```

✓ Checkpoint: `openshell sandbox list` ไม่แสดง sandbox ที่คุณไม่ได้ตั้งใจจะเก็บไว้

## Labs — รันแล็บได้ที่นี่

**labs/lab15_1_install_and_gateway.py** — การตรวจ environment และ gateway แบบอ่านอย่างเดียวของ playbook บน Spark พร้อม OpenShell CLI ตัวจริงบนแล็ปท็อปของคุณ

**labs/lab15_2_policy_builder.py** — สร้าง policy ของ hotel-agent แล้วตรวจความถูกต้องของมันและรุ่นที่เสียแปดแบบด้วย policykit และ parser จริงของ OpenShell

**labs/lab15_3_what_would_it_allow.py** — เล่นซ้ำการกระทำของเอเจนต์และของเอเจนต์ที่ถูกยึดผ่านแบบจำลองเพื่อการสอนของ policy และดูตัวอย่างการเปลี่ยน policy

**labs/lab15_4_sandbox_the_nat_agent.py** — ลำดับเต็มบน Spark เพื่อรันเอเจนต์ NAT จาก Module 14 ใน OpenShell sandbox (เปลี่ยนแปลงอะไรก็ต่อเมื่อใส่ `--yes`)

Lab 15-2 และ 15-3 ทำงานแบบออฟไลน์และได้ผลเหมือนกันทุกเครื่อง ครึ่งฝั่ง Spark ของ lab 15-1 และ lab 15-4 เป็น DRY จนกว่าจะเชื่อม Spark

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 15 — ล็อกเอเจนต์ให้แน่นหนา** หลังตั้งค่าเสร็จ เอเจนต์ NAT ไม่ต้องใช้ package index อีก: ต้องการแค่ไฟล์ของมันและโมเดลของมัน เปิด `week25/15_openshell_sandbox/exercises/ex15_lock_down_the_agent.py` แล้วเขียน policy สำหรับ **ช่วงรันจริง (runtime)** ใน `TODO` สามจุด:

1. `filesystem()`: OS อ่านได้ มีแค่ `/sandbox`, `/tmp` และ `/dev/null` ที่เขียนได้
2. `network_policies()`: มี group เดียวเท่านั้นคือ `inference` สำหรับ `inference.local:443` และ Python ของ venv ของ NAT
3. `process()`: รันเป็น `sandbox:sandbox`

ตัวตรวจจะตรวจ policy ของคุณด้วย policykit แล้วเล่นซ้ำการกระทำสิบสองรายการ และเทียบการตัดสินใจแต่ละครั้งกับของเอเจนต์ที่ถูกล็อกไว้แน่นหนา

```bash
# on: laptop
.venv/bin/python week25/15_openshell_sandbox/exercises/ex15_lock_down_the_agent.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด บันทึกจาก Mac เครื่องนี้ ตัดให้สั้นลง)

```
✓ filesystem: /sandbox writable · /etc read-only
✓ network_policies: exactly one group, 'inference'
✓ process: runs as sandbox:sandbox
✓ policykit.validate: 0 errors, 0 warnings

▣ twelve actions, replayed through policykit.decide
│ action                                                should be              your policy              why
│ write /sandbox/tickets.jsonl                          allow                  ✓ allow                  under read_write /sandbox
│ python3.12 → inference.local:443                      inspect_for_inference  ✓ inspect_for_inference  inference.local is handled by the proxy's infere
│ pip → pypi.org:443                                    deny                   ✓ deny                   pypi.org:443 is not in network_policies (default
│ read  /home/sandbox/.ssh/id_ed25519                   deny                   ✓ deny                   not in read_only or read_write (Landlock: Permis
│ run as root                                           deny                   ✓ deny                   the sandbox runs as 'sandbox'; it cannot become
…
✓ every decision matches a locked-down agent
```

<details><summary>คำใบ้ — ทำไม /home/sandbox/.ssh ถึงถูกปฏิเสธ ทั้งที่ไม่มีอะไรเขียนว่า "deny"?</summary>

filesystem policy เป็นรายการอนุญาต (allow-list) Landlock ปฏิเสธทุก path ที่ไม่ได้อยู่ใต้รายการ `read_only` หรือ `read_write` คุณจึงไม่ต้องระบุว่าจะบล็อกอะไร — ระบุแค่สิ่งที่อนุญาต `/home` ไม่อยู่ในรายการใดเลย

</details>

<details><summary>ท้าทายเพิ่ม — ให้เอเจนต์อ่าน API ของระบบอาคาร</summary>

เพิ่ม group `bms` สำหรับ `bms.example.hotel:443` โดยมี `protocol: rest`, `enforcement: enforce` และรายการ `rules` ที่อนุญาตเฉพาะ `GET /rooms/**` สำหรับ Python ของ venv ของ NAT เพิ่มการกระทำสองรายการในตัวตรวจ: `GET /rooms/808` ควรได้รับอนุญาต ส่วน `POST /rooms/808/setpoint` ควรถูกปฏิเสธ นี่คือรูปแบบเดียวกับ group `openfold3` ใน healthcare policy ของ NVIDIA

</details>

✓ Checkpoint: บรรทัดตรวจทั้งหมดเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `openshell status` แสดง "Connection refused" หรือ "No gateway configured" | `systemctl --user start openshell-gateway` แล้ว `journalctl --user -u openshell-gateway --no-pager -n 50` ถ้า service เข้าถึง Docker ไม่ได้: `sudo setfacl -m u:$USER:rw /var/run/docker.sock` แล้วรีสตาร์ต service |
| `uv pip install openshell` ไม่ได้คำสั่ง `openshell` มา | PyPI 0.1.x เป็นแค่ SDK ใช้ตัวติดตั้งทางการบน Spark ส่วนบนแล็ปท็อปให้ตรึงเวอร์ชัน `openshell==0.0.111` (ส่วนที่ 2) |
| `failed to verify inference endpoint` ตอน `inference set` | vLLM ยังโหลดไม่เสร็จ หรือ URL ของ provider ใช้ `localhost` ใช้ IP ของ host จาก `hostname -I` อุ่นเครื่องด้วยคำขอแชตหนึ่งครั้ง แล้วลองใหม่ |
| ทุกอย่างขาออกถูกปฏิเสธ แม้แต่ host ที่อยู่ใน policy | group ไม่มี `binaries` หรือ endpoint ไม่มี access mode (NemoClaw-applications: `CONNECT tunnel failed, response 403`) เฝ้าดูด้วย `openshell logs <name> --tail --source sandbox` |
| `policy set` ล้มเหลวด้วย `unknown field 'Version'` | คุณ export ด้วย `--full` ใช้ `openshell policy get <name>` โดยไม่มี `--full` หรือเปลี่ยนคีย์เป็นตัวพิมพ์เล็ก |
| `failed to parse sandbox policy YAML … invalid type: sequence, expected a map` | `network_policies` ต้องเป็น map ของ group ที่ตั้งชื่อไว้ ไม่ใช่ list ของ endpoint |
| Policy push จบด้วย exit 1, "validation failed" | มี path ที่ไม่ขึ้นต้นด้วย `/`, มี `..` ใน path, `run_as_user: root` หรือ endpoint ที่ไม่มี host/port รันการตรวจของ lab 15-2 กับไฟล์ของคุณ |
| "Permission denied" / error ของ Landlock ภายใน sandbox | path ไม่อยู่ใน `read_only`/`read_write` filesystem policy เป็นแบบคงที่: แก้ YAML แล้วสร้าง sandbox ใหม่ |
| sandbox ค้างอยู่ในสถานะ `Error` | `openshell logs <name>`: YAML ของ policy ไม่ถูกต้อง, credential ของ provider หายไป หรือพอร์ตชนกัน |

## Next — บทถัดไป

ไปต่อที่ [Lab 16 — NemoClaw: เอเจนต์ใน sandbox ที่ทำงานตลอดเวลา](../16_nemoclaw/TUTORIAL.md): reference stack ของ NVIDIA ที่รัน OpenClaw ภายใน OpenShell ตลอด 24 ชั่วโมง พร้อม policy preset ที่คุณเพิ่มและลบได้โดยไม่ต้อง rebuild
