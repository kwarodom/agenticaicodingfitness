# ▶ Spark Lab 19 — Multi-agent chatbot, RAG และ knowledge graph

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- รัน multi-agent chatbot ของ NVIDIA: มี supervisor เป็น gpt-oss-120B ที่ส่งต่องานให้ผู้เชี่ยวชาญ (specialist) ด้านเขียนโค้ด RAG และภาพ ผ่าน MCP
- แปลงข้อความเป็น knowledge graph ด้วย txt2kg (ArangoDB หรือ Neo4j, Ollama หรือ vLLM) และดูว่ามันพังตรงไหน
- เริ่มตัวอย่าง agentic RAG ใน NVIDIA AI Workbench และรู้ว่าส่วนไหนของมันส่งข้อมูลออกจาก Spark
- วาดแผนที่พอร์ต GPU และหน่วยความจำของแต่ละ stack ก่อนเริ่มรัน เพื่อไม่ให้ชนกับ Module 03–18
- สร้างแก่นของแต่ละแนวคิดด้วยตัวเองบนแล็ปท็อป: ตัวสกัด triple แบบ txt2kg และ retriever ที่อ้างอิงแหล่งที่มา (cite) ได้

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: แล็บจะ parse ไฟล์ของ playbook และใช้ Ollama บนแล็ปท็อปเป็นตัวแทนที่ติดป้ายไว้)

**Playbook ทางการที่ครอบคลุม:** [Multi-agent chatbot](https://build.nvidia.com/spark/multi-agent-chatbot) · [txt2kg](https://build.nvidia.com/spark/txt2kg) · [แอป RAG ใน AI Workbench](https://build.nvidia.com/spark/rag-ai-workbench)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 18 เสร็จแล้ว | คุณรู้ว่า tool call คืออะไร และทำไมมันถึงล้มเหลวได้ | supervisor agent ทำงานด้วย tool call ล้วน ๆ |
| Docker โดยไม่ต้อง sudo บน Spark | `docker ps` | ทั้งสอง stack เป็น Docker Compose |
| ดิสก์ว่าง ~75 GB บน Spark | `df -h /` | สำหรับโมเดลตั้งต้นของ chatbot |
| clone playbook ไว้ที่ root ของ repo | `ls dgx-spark-playbooks/nvidia/playbook-txt2kg` | แล็บ 01–03 อ่านไฟล์ของ playbook โดยตรง |
| Ollama บนแล็ปท็อป (ไม่บังคับ) | `ollama list` | ใช้เป็นตัวแทนในแล็บ 02 และ 03 |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
ls dgx-spark-playbooks >/dev/null 2>&1 || git clone https://github.com/NVIDIA/dgx-spark-playbooks
ls dgx-spark-playbooks/nvidia | grep -cE '^playbook-'
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ตัวเลขจะเพิ่มขึ้นเมื่อ NVIDIA เพิ่ม playbook ใหม่)

```
65
```

> ⚠ ตัว chatbot อย่างเดียวใช้หน่วยความจำราว 120 GB ก่อนถึงส่วนที่ 2 ให้หยุดสิ่งที่โมดูลก่อน ๆ เปิดค้างไว้: `ollama stop <model>` สำหรับโมเดลเขียนโค้ดของ Module 18 และใช้ `docker ps` หา container ของ vLLM, SGLang หรือ LiteLLM

✓ Checkpoint: มี `dgx-spark-playbooks/` อยู่ที่ root ของ repo และ `docker ps` รันบน Spark ได้โดยไม่ต้องใช้ `sudo`

## 1 · สามวิธีในการทำให้เอกสารของคุณมีประโยชน์

playbook ทั้งสามตอบคำถามเดียวกัน ("ให้โมเดลใช้เอกสารของฉัน") ด้วยสถาปัตยกรรมที่ต่างกันสามแบบ:

| | Multi-agent chatbot | txt2kg | Agentic RAG (AI Workbench) |
|---|---|---|---|
| แนวคิดหลัก | LLM ที่เป็น **supervisor** ส่งแต่ละคำขอไปยัง tool ผู้เชี่ยวชาญ | LLM แปลงข้อความเป็น triple **(subject, predicate, object)** ใน graph DB | **router** เลือกเส้นทางการค้นคืน (retrieval) แล้ว **grader** ตรวจความเกี่ยวข้องและ hallucination แล้ววนซ้ำ |
| การค้นคืน (retrieval) | เวกเตอร์: Qwen3-Embedding-4B → Milvus | ท่องกราฟ (graph traversal) (เปิดเวกเตอร์เพิ่มได้ด้วย `--vector-search`: Qdrant + Sentence Transformers, `all-MiniLM-L6-v2`) | vector store + ค้นเว็บ (Tavily) |
| โมเดล | gpt-oss-120B (supervisor), Deepseek-Coder 6.7B, Qwen2-VL / Qwen2.5-VL 7B, Qwen3-Embedding-4B | llama3.1:8b (Ollama) หรือ Nemotron Super 49B FP8 (vLLM) | ค่าเริ่มต้นใช้ endpoint ที่ NVIDIA โฮสต์ หรือใช้โมเดลที่โฮสต์เอง |
| รันแบบ | Docker Compose, UI ที่ :3000 | Docker Compose ผ่าน `./start.sh`, UI ที่ :3001 | โปรเจกต์ AI Workbench, UI แบบ Gradio |
| ข้อมูลออกจาก Spark ไหม? | ไม่ (ยกเว้นการดาวน์โหลดโมเดล) | ไม่ (ยกเว้นการดาวน์โหลด) | **ออก ตามค่าเริ่มต้น**: ต้องใช้ `NVIDIA_API_KEY` และ `TAVILY_API_KEY` |
| เก่งเรื่อง | งานผสม: โค้ด เอกสาร และรูปภาพในแชตเดียว | คำถามหลายทอด (multi-hop) ("X เชื่อมกับ Y อย่างไร?") | ตรวจคุณภาพคำตอบของคำถามปลายเปิด |

**ใช้แบบไหนเมื่อไร:** ใช้ vector RAG เมื่ออยากถามว่าเอกสารเขียนว่าอะไร ใช้ knowledge graph เมื่ออยากถามว่าสิ่งต่าง ๆ เชื่อมโยงกันอย่างไรข้ามเอกสาร ใช้ supervisor เมื่อแชตเดียวต้องทำงานหลายประเภท แล็บ 03 ด้านล่างสร้างครึ่งฝั่งการค้นคืนของ RAG และแล็บ 02 สร้างครึ่งฝั่งการสกัดข้อมูลของ txt2kg คุณจึงเห็นรูปแบบความล้มเหลว (failure mode) ของแต่ละแบบได้ด้วยตาตัวเอง

✓ Checkpoint: สำหรับคำถาม "ถ้า chiller 2 ทริป ห้องไหนจะเสียความเย็น?" คุณบอกได้ว่าทำไมกราฟชนะการค้นด้วยเวกเตอร์ธรรมดา และสำหรับ "สรุป PDF นี้" ทำไมกราฟไม่ชนะ

## 2 · Multi-agent chatbot: supervisor หนึ่งตัวกับผู้เชี่ยวชาญสามตัว

นี่คือสิ่งที่โค้ดของ playbook ต่อเข้าด้วยกัน (`assets/backend/` และไฟล์ Compose สองไฟล์):

```text
 browser :3000 ─► frontend ─► backend :8000  (FastAPI · LangGraph · Postgres for chats)
                                   │  supervisor agent = gpt-oss-120b  (llama.cpp, http://gpt-oss-120b:8000/v1)
                                   │  tools come from 4 MCP servers over stdio (backend/client.py):
                                   ├─ write_code        → deepseek-coder  (llama.cpp)
                                   ├─ search_documents  → Milvus :19530 + qwen3-embedding (llama.cpp --embeddings)
                                   ├─ explain_image     → qwen2.5-vl container (trtllm-serve, TensorRT-LLM)
                                   └─ weather (test tool)
```

มีการตัดสินใจเชิงออกแบบสี่ข้อที่ควรลอกไปใช้:

1. **supervisor ห้ามทำงานแทนผู้เชี่ยวชาญ** system prompt ของมัน (`backend/prompts.py`) เขียนว่า "NEVER EVER generate code yourself" และ "DO NOT try to answer questions from documents yourself" การ routing จะเชื่อถือได้ก็ต่อเมื่อ router ไม่ได้รับอนุญาตให้ลัดขั้นตอนเอง
2. **รวมการเรียกที่เป็นอิสระต่อกันเป็น batch และเรียงลำดับการเรียกที่ขึ้นต่อกัน** ตัวอย่าง few-shot ใน prompt เรียก tool สภาพอากาศสองตัวพร้อมกัน แต่รัน `search_documents` ก่อนแล้วค่อย `write_code` เมื่อโค้ดต้องใช้ผลจากการค้น
3. **จำกัดจำนวนรอบแบบตายตัว** `agent.py` ตั้ง `max_iterations = 3` ลูปของ tool ที่สับสนจึงจบลง แทนที่จะเผา GPU ไปเรื่อย ๆ
4. **ผู้เชี่ยวชาญเป็นแค่ endpoint ที่เข้ากันได้กับ OpenAI** MCP tool แต่ละตัวเรียก `http://<container>:8000/v1` เปลี่ยนผู้เชี่ยวชาญได้โดย supervisor ไม่ต้องเปลี่ยน

ผู้เชี่ยวชาญด้าน RAG ตัดไฟล์ที่อัปโหลดเป็น chunk ละ 1000 ตัวอักษร ซ้อนกัน (overlap) 200 และค้นคืน `k=8` chunk (`vector_store.py`) มีสิ่งหนึ่งที่เราสังเกตเห็นตอนอ่านโค้ด: `image_understanding.py` ขอโมเดลชื่อ `Qwen2.5-VL-7B-Instruct` ขณะที่ `docker-compose-models.yml` serve `Qwen/Qwen2-VL-7B-Instruct` ใน container ชื่อ `qwen2.5-vl` ถ้าคำตอบเกี่ยวกับรูปภาพล้มเหลวบน Spark ของคุณ ให้ตรวจจุดนี้ก่อน

การตัดสินใจ routing ของ supervisor คือ tool call หนึ่งครั้ง ลองดูด้วยโมเดลจาก Module 18 (ส่งไปที่ Ollama ของ Spark หรือตัวแทนบนแล็ปท็อป) คำตอบที่ดีจะเรียก `search_documents` ก่อน เพราะโค้ดต้องอาศัยผลของมัน:

```spark
{"target": "ollama", "which": "a", "model": "qwen3.6:35b-a3b-mtp-q4_K_M",
 "messages": [{"role": "system", "content": "You are a supervisor agent. ALWAYS use a tool when the request matches it. NEVER write code yourself. If a call depends on another call's output, make only the first call now."},
              {"role": "user", "content": "Search my documents for the design requirements, then build a website based on them."}],
 "max_tokens": 256,
 "tools": [{"type": "function", "function": {"name": "search_documents", "description": "Search the user's uploaded documents.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
           {"type": "function", "function": {"name": "write_code", "description": "Write complete code.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "programming_language": {"type": "string"}}, "required": ["query", "programming_language"]}}}]}
```

**รันบน Spark** (playbook Step 2–5) การดาวน์โหลดโมเดลใช้เวลา 30 นาทีถึง 2 ชั่วโมง ส่วน `up` ครั้งแรกจะ build image llama.cpp แบบ CUDA ซึ่งใช้เวลา 10–20 นาที:

```bash
# on: spark
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-multi-agent-chatbot/assets
chmod +x model_download.sh
./model_download.sh
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

playbook ระบุว่า "the Qwen2.5-VL model container may report as unhealthy while starting up" (container ของ Qwen2.5-VL อาจรายงานว่า unhealthy ระหว่างเริ่มทำงาน) จากนั้น forward พอร์ตของ UI และ API มาที่แล็ปท็อป แล้วเปิด `http://localhost:3000`:

```bash
# on: laptop
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 spark-a
```

หากต้องการลอง RAG agent ให้อัปโหลด PDF whitepaper ของ NVIDIA Blackwell (มีลิงก์ใน Step 6 ของ playbook) ด้วย **Upload Documents** ติ๊กเลือกใน **Select Sources** แล้วคลิกไทล์ใดไทล์หนึ่ง ต้องการหน่วยความจำเหลือมากขึ้น? Step 8 ของ playbook เปลี่ยน supervisor เป็น gpt-oss-20B: แก้ `model_download.sh`, `docker-compose-models.yml` และตัวแปร `MODELS=` ใน `docker-compose.yml` แล้วรัน `up` อีกครั้ง เก็บกวาดด้วย:

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-multi-agent-chatbot/assets
docker compose -f docker-compose.yml -f docker-compose-models.yml down
docker volume rm "$(basename "$PWD")_postgres_data"      # deletes the chat history volume
```

✓ Checkpoint: คุณไล่ตามคำขอหนึ่งรายการจากเบราว์เซอร์ไปถึง `deepseek-coder` แล้วกลับมาได้ และบอกชื่อไฟล์ที่ห้าม supervisor เขียนโค้ดเองได้

## 3 · วาดแผนที่ stack ก่อนเริ่มรัน: lab 01

สอง stack, host port สิบสามพอร์ต และบริการที่ใช้ GPU หกตัว บนเครื่องที่คุณรัน Ollama และ vLLM ไปแล้ว Lab 01 อ่านไฟล์ Compose ของ playbook เองแล้วหาจุดที่ชนกันให้คุณ:

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab01_stack_map.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ parse จาก `dgx-spark-playbooks@3410c65` คำสั่ง Spark ใน Step 4 จะพิมพ์รูปแบบ EXAMPLE ในโหมด DRY)

```
▣ STEP 1 · every service, per stack (from the Compose files)
│ stack                service                image                                         host ports  gpu  note
│ ───────────────────  ─────────────────────  ────────────────────────────────────────────  ──────────  ───  ──────────────────────
│ multi-agent chatbot  backend                (built locally)                               8000
│ multi-agent chatbot  frontend               (built locally)                               3000
│ multi-agent chatbot  postgres               postgres:15-alpine                            5432
│ multi-agent chatbot  milvus                 milvusdb/milvus:v2.5.15-20250718-3a3b374f-gp  19530,9091
│ multi-agent chatbot  qwen2.5-vl             nvcr.io/nvidia/tensorrt-llm/release:spark-si  —           GPU
│ multi-agent chatbot  gpt-oss-120b           local/llama.cpp:server-cuda                   —           GPU
│ txt2kg (./start.sh)  app                    (built locally)                               3001
│ txt2kg (./start.sh)  arangodb               arangodb:latest                               8529
│ txt2kg (./start.sh)  ollama                 ollama-custom:latest                          11434       GPU
│ txt2kg --neo4j       neo4j                  neo4j:5-community                             7474,7687
│ txt2kg --vllm        vllm                   (built locally)                               8001        GPU
│ …

▣ STEP 2 · port collisions — between stacks, and with servers earlier modules started
│ port   published by                                         both stacks?  course server on that port
│ ─────  ───────────────────────────────────────────────────  ────────────  ──────────────────────────────────
│ 8000   chatbot:backend, txt2kg:sentence-transformers (opt)  yes           Module 05 (vLLM) / Module 06 (NIM)
│ 11434  txt2kg:ollama                                        —             Modules 03/18 (host Ollama)
◆ (opt) = only with --vector-search. Docker refuses to start a container whose host port is taken
  ('port is already allocated'). Stop the earlier server first, or run one stack at a time.

▣ STEP 3 · memory: what the playbooks say each stack needs
│ multi-agent chatbot     ~120 GB (playbook: 'uses ~120 GB of memory by defau…  REFERENCE
│   downloads             gpt-oss-120B ~63 GB · Deepseek-Coder 6.7B ~7 GB · Q…  REFERENCE
│ txt2kg (Ollama stacks)  llama3.1:8b default → ~4.8 GB of Q4_K_M weights + K…  arithmetic
│ txt2kg --vllm           nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 → ~49 …  arithmetic
│ Module 18 coding model  qwen3.6:35b-a3b-mtp-q4_K_M ~23 GB                     REFERENCE
```

(Step 1 ถูกตัดให้สั้นลงในที่นี้ แล็บพิมพ์แถวบริการทั้ง 29 แถว) สิ่งที่มันบอกคุณ:

- **:11434** stack ตั้งต้นของ txt2kg เปิด container Ollama *ของมันเอง* (`ollama-compose`) บนพอร์ต Ollama ของ host ถ้า Ollama ของระบบบน Spark จาก Module 03 และ 18 ยังรันอยู่ ให้หยุดมันก่อน (`sudo systemctl stop ollama`) ไม่เช่นนั้น container `ollama` ของ txt2kg จะ bind พอร์ตไม่ได้
- **:8000** backend ของ chatbot ใช้พอร์ตเดียวกับ vLLM และ NIM ให้หยุดเซิร์ฟเวอร์ของ Module 05/06 ก่อน
- **หน่วยความจำ** chatbot ตัวเดียวก็กินหน่วยความจำ Spark ไปเกือบหมด ให้รัน chatbot หรือ txt2kg อย่างใดอย่างหนึ่ง ไม่ใช่ทั้งคู่

✓ Checkpoint: ก่อนสั่ง `docker compose up` คุณรู้ว่าต้องหยุดเซิร์ฟเวอร์ตัวไหนจากโมดูลก่อน ๆ สำหรับแต่ละ stack

## 4 · txt2kg: ข้อความ → triple → กราฟ

txt2kg เป็นแอป Next.js (UI ที่ :3001) ที่ตัดเอกสารของคุณเป็น chunk ขอ triple จาก LLM เก็บลง ArangoDB หรือ Neo4j และแสดงกราฟด้วย Three.js WebGPU หน้า query ของมันจะท่องกราฟเพื่อเพิ่ม entity ที่เกี่ยวข้องลงใน prompt ก่อนที่ LLM จะตอบ stack ต่าง ๆ ของ playbook:

| คำสั่งเริ่ม | Graph DB | LLM | โมเดลตั้งต้น |
|---|---|---|---|
| `./start.sh` | ArangoDB (:8529) | Ollama (:11434) | `llama3.1:8b` |
| `./start.sh --neo4j` | Neo4j (:7474, bolt :7687) | Ollama (:11434) | `llama3.1:8b` |
| `./start.sh --vllm` | Neo4j | vLLM (:8001) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8` |

เพิ่ม `--vector-search` ให้ตัวไหนก็ได้เพื่อใช้ Qdrant + Sentence Transformers บน Spark (playbook Step 1–3):

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-txt2kg/assets
./start.sh --help
./start.sh --neo4j
docker exec ollama-compose ollama pull llama3.1:8b
```

คอร์สนี้เริ่มด้วย `--neo4j` เพราะ Neo4j คือ graph database ที่ Week 15 สอนไปแล้ว ค่าเริ่มต้นของ playbook คือ `./start.sh` เปล่า ๆ (ArangoDB) และทั้งสองแบบใช้ได้บน Spark playbook ระบุว่าปัญหา page size 64 KB ของ ArangoDB เกิดกับ DGX Station ไม่ใช่ DGX Spark เปิด UI ผ่าน tunnel แล้วอัปโหลดไฟล์ markdown, text หรือ CSV (ฟอร์แมตที่ playbook ระบุไว้):

```bash
# on: laptop
ssh -N -L 3001:localhost:3001 -L 7474:localhost:7474 -L 7687:localhost:7687 spark-a
# UI: http://localhost:3001 · Neo4j Browser: http://localhost:7474
```

prompt สำหรับสกัดข้อมูลอยู่ใน `frontend/app/api/ollama/route.ts`: "Extract subject-predicate-object triples … Normalize entity names to their canonical form … Return results in JSON format as an array of objects with "subject", "predicate", "object" fields" เมื่อ parse JSON ไม่ได้ จะมีทางสำรอง (fallback) ที่อ่านบรรทัดแบบ `a - b - c` ส่วน `utils/text-processing.ts` แปลงทุกส่วนเป็นตัวพิมพ์เล็ก แล้วตัด triple ซ้ำโดยใช้ `subject|predicate|object` เป็นคีย์ ขนาด chunk ของมันคือ 20,000 ตัวอักษร ซ้อนกัน 1,000 พร้อมคอมเมนต์ว่า "Optimized for Gemma3:27b on DGX Spark"

เพื่อความเร็ว ส่วน Troubleshooting ของ playbook แนะนำ `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_LOADED_MODELS=1` และ `OLLAMA_KV_CACHE_TYPE=q8_0` หยุดด้วย flag เดียวกับตอนเริ่ม: `./stop.sh --neo4j`

✓ Checkpoint: UI ของ txt2kg เปิดได้ที่ `localhost:3001` และคุณบอกได้ว่าไฟล์ไหนเก็บ prompt สำหรับสกัดข้อมูล

## 5 · Triple บนแล็ปท็อปของคุณ: lab 02

Lab 02 คือ txt2kg ใน Python หน้าจอเดียว input คือส่วน "Basic idea" ของ playbook ทั้งสามในโมดูลนี้ system prompt เป็นของ txt2kg คำต่อคำ ตัว parse การแปลงเป็นตัวพิมพ์เล็ก และการตัดตัวซ้ำทำตามโค้ดของ txt2kg จากนั้นแล็บจะสร้างกราฟแล้วถามคำถามกับมัน ใช้ LLM สามครั้ง ครั้งละ 300 token

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab02_txt2kg_mini.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ใช้ LAPTOP STAND-IN `gemma4:12b` ไม่ใช่ `llama3.1:8b` ของ txt2kg บน Spark การสกัดมีการสุ่ม (sampling) triple ของคุณจึงจะต่างไป)

```
▣ STEP 2 · extract triples — one LLM call per chunk, txt2kg's system prompt
◆ multi-agent-chatbot   7 triples · 300 tok in 23.3s (gemma4:12b) · reply cut at 300 tokens, kept the complete objects
◆ txt2kg                8 triples · 300 tok in 35.3s (gemma4:12b) · reply cut at 300 tokens, kept the complete objects
◆ rag-ai-workbench      7 triples · 300 tok in 48.1s (gemma4:12b) · reply cut at 300 tokens, kept the complete objects

▣ STEP 3 · normalise + de-duplicate (txt2kg: lowercase, then mergeTriples on subject|predicate|object)
◆ 22 raw triples → 22 unique · 33 entities
│ supervisor agent                is powered by               gpt-oss-120b                        multi-agent-chatbot
│ supervisor agent                orchestrates                specialized downstream agents       multi-agent-chatbot
│ ollama                          is a                        local llm inference engine          txt2kg
│ neo4j                           is a                        graph database                      txt2kg
│ nvidia ai workbench             is used to clone and run    pre-built agentic rag application   rag-ai-workbench
│ …

▣ STEP 4 · the graph: hubs, entities that link documents, and a path
│ entity                             links  mentioned in
│ pre-built agentic rag application  4      rag-ai-workbench
│ supervisor agent                   3      multi-agent-chatbot
│ playbook                           2      multi-agent-chatbot, txt2kg
◆ entities that appear in more than one playbook: playbook
◆ connected components: 11 (1 would mean every fact is reachable from every other)
→ no multi-hop path between the top hubs: the extracted facts do not share entity names
→ wrote week25/19_multi_agent_rag_kg/.runs/triples.json and triples.cypher
```

triple อ่านแล้วดูดี แต่ *กราฟ* แย่: ข้อเท็จจริง 22 ข้อกระจายเป็น 11 เกาะ และ entity เดียวที่เชื่อมสองเอกสารคือคำว่า "playbook" ที่ไร้ประโยชน์ รันสองครั้งบน Mac เครื่องนี้ได้รูปแบบเดียวกัน นี่คือบทเรียนจริงของ txt2kg และคุณเห็นมันได้ที่นี่ภายในหนึ่งนาที แทนที่จะต้องอัปโหลดไปเป็นชั่วโมงก่อน:

1. **การระบุว่า entity ไหนคือตัวเดียวกัน (entity resolution) เป็นตัวตัดสินคุณภาพกราฟ** "agentic retrieval-augmented generation" กับ "pre-built agentic rag application" สำหรับคุณคือสิ่งเดียวกัน แต่สำหรับกราฟคือสองโหนด การแปลงเป็นตัวพิมพ์เล็กไม่พอ pipeline จริงจะเพิ่มขั้นตอน alias หรือ merge, schema ของชนิด entity ที่อนุญาต หรือใช้โมเดลที่ใหญ่ขึ้น (stack `--vllm` ของ txt2kg ใช้ Nemotron Super 49B)
2. **งบ token ตัดคำตอบทิ้ง** ทุกคำตอบชนเพดาน 300 token ตัว parse เก็บเฉพาะ JSON object ที่สมบูรณ์ ซึ่งเป็นเหตุผลที่ route ของ txt2kg เองอนุญาต `maxTokens = 4096`
3. **ผลลัพธ์ที่โหลดต่อได้** `triples.cypher` มีคำสั่ง `MERGE` วางลงใน Neo4j Browser บน stack `--neo4j` (`localhost:7474`) เพื่อดูเกาะเหล่านั้นด้วยตาตัวเอง

✓ Checkpoint: คุณรัน lab 02 แล้ว และอธิบายได้ว่าทำไมกราฟของมันมีมากกว่าหนึ่ง component

## 6 · Agentic RAG ใน AI Workbench

playbook ที่สามไม่ใช่ stack แบบ Compose แต่เป็นโปรเจกต์ **NVIDIA AI Workbench** ([workbench-example-agentic-rag](https://github.com/NVIDIA/workbench-example-agentic-rag)) พร้อมแชตแบบ Gradio ส่วนที่เป็น "agentic" คือลูปที่ครอบการค้นคืนไว้: **router** เลือกวิธีตอบแต่ละ query จากนั้นระบบจะ "evaluates responses for relevancy and hallucination, and iterates through evaluation and generation cycles" (ประเมินคำตอบด้านความเกี่ยวข้องและ hallucination แล้ววนรอบประเมินและสร้างคำตอบซ้ำ) แท็บ **Monitor** แสดงการตัดสินใจเหล่านั้น

ขั้นตอนเป็นการคลิกใน GUI บนเดสก์ท็อปของ Spark (playbook Step 1–7):

1. เปิด **NVIDIA AI Workbench** → **Begin Installation** → **Let's Get Started** ถ้าเห็น "container tool failed to reach ready state … docker is not running" ให้รีบูตแล้วเปิดใหม่
2. เตรียมคีย์สองตัว: NVIDIA API key จาก NGC ที่มีสิทธิ์ **Public API Endpoints** และคีย์ของ Tavily
3. **Local** → **Clone Project** → `https://github.com/NVIDIA/workbench-example-agentic-rag` → **Clone**
4. ในแถบสีเหลือง คลิก **Configure** แล้วใส่ `NVIDIA_API_KEY` และ `TAVILY_API_KEY`
5. **Environment → Project Container → Apps → Chat** ถาม `How do I add an integration in the CLI?` แล้วทำตาม quickstart ในแอปเพื่ออัปโหลดชุดข้อมูลตัวอย่าง

> ⚠ **ตรวจเรื่องอธิปไตยข้อมูล (sovereignty)** ด้วยค่าเริ่มต้น โปรเจกต์นี้เรียก endpoint ที่ NVIDIA โฮสต์และการค้นเว็บของ Tavily ดังนั้น query และข้อความที่ค้นคืนมาจะออกจาก Spark playbook ระบุว่ามัน "can work with NVIDIA-hosted API endpoints or self-hosted models" หากต้องการเวอร์ชันที่อยู่ในเครื่องทั้งหมด ให้ชี้ไปที่โมเดลบน Spark ของคุณ (Module 05–08) และตัดการค้นเว็บออก ใส่คีย์เฉพาะใน **Project Secrets** ของ Workbench เท่านั้น ห้ามใส่ในโค้ดหรือโน้ตบุ๊กเด็ดขาด

✓ Checkpoint: แชต Gradio ตอบ query ตัวอย่างได้ และแท็บ Monitor แสดงการตัดสินใจของ router หรือคุณบอกชื่อบริการภายนอกสองตัวที่ค่าเริ่มต้นเรียกใช้ได้

## 7 · การค้นคืนที่คุณตรวจดูได้: lab 03

ระบบ RAG ทุกตัวคือ "ค้นคืน แล้วสร้างคำตอบ" (retrieve, then generate) และการค้นคืนเป็นตัวกำหนดเพดานว่าการสร้างคำตอบจะถูกได้แค่ไหน Lab 03 สร้าง retriever ด้วย standard library ล้วน ๆ: TF-IDF บนทุกส่วนของ README ของ DGX Spark playbook ทุกตัว มันให้คะแนนตัวเองด้วยคำถามแปดข้อที่รู้คำตอบอยู่แล้ว จากนั้นส่ง 3 ส่วนแรกให้ LLM ที่ต้องอ้างอิงแหล่งที่มา แล้วตรวจการอ้างอิงนั้น

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab03_playbook_finder.py
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab03_playbook_finder.py --ask "serve a VLM for live video" --no-llm
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ การค้นคืนเป็นการคำนวณล้วนและได้ผลเหมือนกันทุกเครื่อง ส่วนคำตอบมาจาก LAPTOP STAND-IN)

```
▣ STEP 1 · index: split every playbook README on '## ' headings
◆ 65 playbooks · 1510 sections · 9,030 distinct terms

▣ STEP 2 · eval: eight questions with known answers
│    question                                        top 3 playbooks                                       score
│ ─  ──────────────────────────────────────────────  ────────────────────────────────────────────────────  ─────
│ ≈  fine-tune a vision language model on my own im  live-vlm-webui, vlm-finetuning, unsloth               0.25
│ ✓  high-throughput LLM serving with tensor parall  vllm, connect-two-sparks, trt-llm                     0.24
│ ✓  turn my documents into a knowledge graph I can  txt2kg, llms, vss                                     0.35
│ ✓  run Claude Code against a local model           local-coding-agent, cli-coding-agent, vscode          0.52
│ ✓  cable two Sparks together for distributed work  connect-two-sparks, connect-multiple-sparks, connec…  0.38
│ ✓  sandbox and govern an AI agent's network and f  openshell, tailscale, nemoclaw-applications           0.31
│ ✓  a supervisor agent that delegates to coding an  multi-agent-chatbot, cli-coding-agent, rag-ai-workb…  0.24
│ ≈  code completion in VS Code from a model on my   vscode, vibe-coding, local-coding-agent               0.34
◆ hit@1 = 6/8 · hit@3 = 8/8   (✓ right at #1 · ≈ right in top 3 · ✕ missed)

▣ STEP 3 · retrieve for: 'Which playbook should I use to build a knowledge graph from my PDFs and ask it questions?'
[1] 0.258  Build Knowledge Graphs with txt2kg  —  nvidia/playbook-txt2kg/README.md § Step 5. Upload documents and build knowledge graphs
[2] 0.136  Set Up Example NemoClaw Agents  —  nvidia/playbook-nemoclaw-applications/README.md § Step 2. Agent prompt
[3] 0.129  Deploy a Video Search and Summarization Agent  —  nvidia/playbook-vss/README.md § Step 10. Next steps

▣ STEP 4 · grounded answer: the LLM may use ONLY these three sources, and must cite them
· ANSWER  To build a knowledge graph and ask questions about your documents, you should use the **Build Knowledge Graphs with txt2kg** playbook [1]. This process involves uploading documents, extracting triples, and using a query interface where the LLM generates responses based on enriched graph context [1].
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 6268 ms · 58 tok in 9.1s · 20.2 tok/s
✓ every citation [1] points at a source it was given: nvidia/playbook-txt2kg/README.md § Step 5. Upload documents and build knowledge graphs
```

อ่านข้อที่พลาด ไม่ใช่แค่คะแนน สองแถวที่เป็น ≈ แพ้ให้ playbook ที่ใช้คำเดียวกัน ("VLM", "VS Code") แต่เจตนาต่างกัน bag-of-words แยก "fine-tune a VLM" ออกจาก "stream video to a VLM" ไม่ได้ นี่คือช่องว่างที่เวกเตอร์ Qwen3-Embedding ของ chatbot มีไว้เพื่อปิด

จากนั้นอ่านคำตอบ มัน **มีหลักฐานรองรับ (grounded)** (มีการอ้างอิงจริงหนึ่งรายการ ซึ่งตรวจแล้ว) แต่ไม่ **ครบถ้วน (complete)**: คำถามพูดถึง PDF แต่ playbook ของ txt2kg ระบุ markdown, text และ CSV playbook ที่รับ PDF คือ RAG agent ของ chatbot การอ้างอิงบอกคุณว่าข้อความมาจากไหน แต่ไม่ได้บอกว่า retriever พลาดอะไรไป

✓ Checkpoint: คุณอธิบายแถว ≈ ทั้งสองแถวได้ และอธิบายได้ว่าทำไมคำตอบที่อ้างอิงถูกต้องก็ยังเป็นคำแนะนำที่ผิดได้

## Labs — รันแล็บได้ที่นี่

**labs/lab01_stack_map.py** — parse ไฟล์ Compose ของ chatbot และ txt2kg: ทุกบริการ host port และ GPU พอร์ตที่ชนกับโมดูลก่อน ๆ และหน่วยความจำของแต่ละ stack

**labs/lab02_txt2kg_mini.py** — txt2kg ฉบับย่อ: สกัด triple จาก playbook สามตัวด้วย prompt ของ txt2kg เอง รวมเข้าด้วยกัน แล้ว query กราฟ

**labs/lab03_playbook_finder.py** — การค้นคืนแบบ TF-IDF ด้วย stdlib บน README ของ playbook ทุกตัว ให้คะแนนด้วย hit@k พร้อมคำตอบจาก LLM ที่ตรวจการอ้างอิงแล้ว

Lab 01 ทำงานแบบออฟไลน์ ยกเว้นขั้นตอนบน Spark ที่อ่านอย่างเดียว Lab 02 และ 03 ใช้ Ollama ของ Spark เมื่อมันตอบ ไม่เช่นนั้นใช้ของแล็ปท็อป (ติดป้าย LAPTOP STAND-IN)

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 19 — กราฟเบื้องหลัง GraphRAG** เปิด `week25/19_multi_agent_rag_kg/exercises/ex19_graph_context.py` triple 13 ตัวในไฟล์อธิบายการต่อสายของ chatbot (ส่วนที่ 2) พร้อมตัวพิมพ์เล็กใหญ่ที่ปนกันและข้อมูลซ้ำแบบที่ตัวสกัดสร้างขึ้นจริง ในไฟล์มี `TODO` สามจุด:

1. `normalise(triple)`: แปลงเป็นตัวพิมพ์เล็ก ตัดช่องว่างหัวท้าย และยุบช่องว่างภายในให้เหลือช่องเดียว
2. `build_graph(triples)`: ตัดตัวซ้ำโดยคงลำดับที่พบครั้งแรก และสร้างเซตเพื่อนบ้าน (neighbour set) ทั้งสองทิศทาง
3. `path(graph, a, b)`: เส้นทางสั้นที่สุดแบบ breadth-first หรือ `None`

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/exercises/ex19_graph_context.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด)

```
✓ normalise: lowercase · strip · collapse inner spaces
✓ build_graph: 13 raw → 11 unique · backend links frontend, supervisor agent, postgres
✓ path: frontend → milvus in 4 hops · deepseek-coder → qwen3-embedding in 5 · unknown → None

▣ your graph, applied: the context GraphRAG adds for 'How does a chat reach the vector database?'
│ (frontend) -[sends chats to]-> (backend)
│ (backend) -[runs]-> (supervisor agent)
│ (supervisor agent) -[calls tool]-> (search_documents)
│ (search_documents) -[retrieves from]-> (milvus)
```

<details><summary>คำใบ้ — ทำไม "13 raw → 11 unique" ต้องเรียก normalise() ก่อน?</summary>

`("Backend", "stores chats in", "Postgres")` กับ `(" backend ", "Stores Chats In", "postgres")` คือข้อเท็จจริงเดียวกัน ให้ตัดตัวซ้ำจาก tuple ที่ normalise แล้ว `list(dict.fromkeys(...))` คงลำดับที่พบครั้งแรกไว้ให้

</details>

<details><summary>ท้าทายเพิ่ม — แก้ปัญหาเกาะของ lab 02</summary>

เพิ่มตาราง alias ให้ lab 02 (`{"pre-built agentic rag application": "agentic rag", "retrieval-augmented generation": "rag", …}`) แล้วนำไปใช้ใน `norm()` เหลือ connected component กี่ตัว? จากนั้นโหลด `.runs/triples.cypher` เข้า Neo4j แล้วนับด้วย Cypher

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `address already in use` หรือ `port is already allocated` บน :11434 (txt2kg) | Ollama ของ Spark เองถือพอร์ตอยู่: `sudo systemctl stop ollama` แล้ว `./start.sh` ใหม่ (lab 01, Step 2) |
| `port is already allocated` บน :8000 (chatbot) | มี container vLLM/NIM จาก Module 05–06 รันอยู่: `docker ps` แล้ว `docker stop <name>` |
| container ของ chatbot unhealthy หรือ OOM | หยุดงาน GPU อื่น (`nvidia-smi`) เปลี่ยน supervisor เป็น gpt-oss-20B (playbook Step 8) และล้าง cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `qwen2.5-vl` แสดง unhealthy ทันทีหลัง `up` | เป็นเรื่องปกติระหว่างเริ่มทำงาน ตามที่ playbook ระบุ ถ้าคำตอบเรื่องรูปภาพยังล้มเหลว ให้ตรวจชื่อโมเดลที่ไม่ตรงกันตามที่กล่าวไว้ในส่วนที่ 2 |
| ดาวน์โหลดโมเดลหยุดกลางทาง | ลบไฟล์ที่ดาวน์โหลดไม่ครบใน `models/` แล้วรัน `./model_download.sh` อีกครั้ง |
| "Cannot access gated repo" ระหว่างดาวน์โหลด | รัน `hf auth login` บน Spark และขอสิทธิ์เข้าถึงโมเดลบน Hugging Face |
| UI ของ txt2kg แสดงแบนเนอร์ว่ากำลังเริ่มต้น (initializing) | LLM ยังโหลดไม่เสร็จ สำหรับ `--vllm` ให้ติดตาม `docker logs vllm-service -f` (ครั้งแรกใช้เวลา 30 นาทีขึ้นไป) |
| AI Workbench: 401 / 403 | คีย์ผิด หรือคีย์ NGC ไม่มีสิทธิ์ **Public API Endpoints** ให้เปลี่ยนคีย์ใน Project Secrets แล้วรีสตาร์ต |
| Lab 02 parse ได้ 0 triple | โมเดลไม่ได้ตอบเป็น JSON ลอง `--laptop-model gemma4:12b` หรือใช้โมเดลที่ใหญ่กว่าบน Spark |

## Next — บทถัดไป

ไปต่อที่ [Lab 20 — capstone: เอเจนต์โรงแรมแบบ sovereign](../20_capstone_sovereign_agent/TUTORIAL.md): fine-tune, serve, route ผ่าน gateway และรัน NAT agent ใน sandbox โดยใช้แนวคิดเรื่องการค้นคืนและกราฟจากโมดูลนี้เป็น tool ของมัน
