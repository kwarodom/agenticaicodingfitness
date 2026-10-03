# ▶ Spark Lab 06 — SGLang, TensorRT-LLM, NIM และ Nemotron: ประชัน serving engine

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เปิด serving engine เพิ่มอีกสามตัวตาม playbook ของแต่ละตัว: **SGLang** ที่ :30000, **TensorRT-LLM** ที่ :8355 และ **NIM** ที่ :8000
- ใช้จุดแข็งสองข้อของ SGLang: JSON ที่บังคับตาม schema และ prefix caching ที่มองเห็นได้ใน `cached_tokens`
- serve โมเดลของ NVIDIA เอง คือ **Nemotron 3 Nano** (vLLM) และ **Nemotron 3 Super** (vLLM หรือ TensorRT-LLM) บน Spark เครื่องเดียว
- จัด **การประชันที่ยุติธรรม (fair bake-off)**: โมเดลเดียวกัน prompt เดียวกัน การตั้งค่าเดียวกัน รันทีละ engine และรายงานค่ามัธยฐาน (median)
- แปลงความต้องการของงาน (workload) ให้เป็นการเลือก engine โดยมีข้อเท็จจริงจาก playbook รองรับทุกการเลือก

**Time** ~60 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + laptop stand-in)

**Playbook ทางการที่ครอบคลุม:** [SGLang](https://build.nvidia.com/spark/sglang) · [TensorRT-LLM](https://build.nvidia.com/spark/trt-llm) · [NIM สำหรับ LLM](https://build.nvidia.com/spark/nim-llm) · [Nemotron](https://build.nvidia.com/spark/nemotron)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 05 เสร็จแล้ว | คุณเคยเปิดและปิด `vllm-server` มาแล้วหนึ่งครั้ง | โมดูลนี้เปรียบเทียบกับ vLLM |
| ไม่มี engine ค้างรันอยู่ | `docker ps` บน Spark ไม่แสดง container ที่กำลัง serve | ทุก engine จองหน่วยความจำส่วนใหญ่ของ 128 GB (ส่วนที่ 1) |
| ล็อกอิน Hugging Face บน Spark แล้ว | `hf auth whoami` | SGLang, TensorRT-LLM และ Nemotron ดาวน์โหลดจาก Hugging Face |
| NGC API key (สำหรับ NIM) | จาก <https://ngc.nvidia.com/setup/api-key> | container ของ NIM และโมเดลมาจาก `nvcr.io` |
| ดิสก์ว่าง ~100 GB | `df -h /` | container image สามตัวบวกโมเดล |

```bash
# on: spark
docker ps --format '{{.Names}}  {{.Image}}  {{.Ports}}'
hf auth whoami
df -h / | tail -1
```

> 🔐 ล็อกอิน NGC **บน Spark** ครั้งเดียว โดยพิมพ์คีย์ตอนที่ถูกถาม: `docker login nvcr.io --username '$oauthtoken'` ส่วน playbook ของ NIM ส่ง `$NGC_API_KEY` เข้าคำสั่งเดียวกันผ่าน pipe ไม่ว่าจะทางไหน ห้ามวางคีย์ลงในแล็บหรือใน Lab Runner

✓ Checkpoint: `docker ps` บน Spark ไม่มี container ที่กำลัง serve และคุณมี NGC API key เตรียมไว้แล้ว

## 1 · สี่ engine, API เดียว

ทุก engine ในโมดูลนี้พูด API แบบเดียวกันที่เข้ากันได้กับ OpenAI (OpenAI-compatible) แล็บ LiteLLM (Module 08) และ NAT (Module 14) จึงคุยกับทุกตัวได้ด้วย client ตัวเดียว สิ่งที่ต่างกันคือวิธีที่แต่ละตัวทำความเร็ว และปริมาณการตั้งค่าที่ต้องใช้:

| Engine | Image (tag ตรงตาม playbook) | พอร์ต | คำสั่ง serve | สโลแกนใน playbook / จุดแข็ง |
|---|---|---|---|---|
| vLLM (Module 05) | `vllm/vllm-openai:latest` | 8000 | `vllm serve` | continuous batching, PagedAttention; สูตร (recipe) ที่พร้อมใช้กับเอเจนต์ |
| **SGLang** | `lmsysorg/sglang:latest-cu130` | 30000 | `sglang serve --model-path` | prefix cache แบบ RadixAttention, structured output ด้วย xGrammar |
| **TensorRT-LLM** | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` (เครื่องเดียว) · `1.3.0rc5` (หลายเครื่อง) | 8355 | `trtllm-serve` | "Lower-latency responses and higher throughput for the largest models" |
| **NIM** | `nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest` | 8000 | (มีในตัว) | "Prebuilt, GPU-optimized model containers with a ready-to-use HTTP endpoint" |

**รันทีละ engine** ด้วยเหตุผลสองข้อ:

1. **พอร์ต** NIM และ vLLM ใช้ 8000 ทั้งคู่ วิธีแก้ของ playbook เมื่อพอร์ตไม่ว่างคือ map ไปยังพอร์ตอื่นบน host เช่น `-p 8001:8000`
2. **หน่วยความจำ** แต่ละ engine จองสัดส่วนคงที่ของ unified memory ตอนเริ่มทำงาน: vLLM `--gpu-memory-utilization 0.8`, SGLang `--mem-fraction-static 0.85`, TensorRT-LLM `free_gpu_memory_fraction: 0.9` สองตัวที่ค่าเหล่านี้ใส่ใน 128 GB ไม่ได้

```bash
# on: spark
docker rm -f vllm-server sglang-server trtllm-server nim-llm-demo 2>/dev/null; docker ps
```

✓ Checkpoint: คุณบอกพอร์ตของแต่ละ engine ได้ และอธิบายได้ว่าทำไม NIM และ vLLM จึงรันพร้อมกันด้วยค่าตั้งต้นไม่ได้

## 2 · SGLang: prefix caching และ JSON แบบมีโครงสร้าง

**RadixAttention** ของ SGLang เก็บ KV cache ของทุก prefix ที่เคยเห็นไว้ในโครงสร้างต้นไม้ (tree) เมื่อคำขอถัดไปขึ้นต้นด้วย token ชุดเดิม (system prompt เดิม ประวัติแชตเดิม หรือ context ของ RAG ชุดเดิม) SGLang จะข้ามงาน prefill ส่วนนั้นไป เอเจนต์และแชตหลายรอบ (multi-turn) ส่ง prefix ยาว ๆ ซ้ำทุกรอบ จึงประหยัดได้มาก ส่วน **structured output** (xGrammar) บังคับให้คำตอบตรงกับ JSON schema ตั้งแต่ระหว่างที่สร้าง

environment และค่าตั้งต้นพื้นฐานของ playbook ค่าเริ่มต้นคือ `Qwen/Qwen3-8B`: โมเดล dense ขนาด 8B ที่ warm-up เร็ว และไม่ต้องขอสิทธิ์ (ungated):

```bash
# on: spark
export HF_TOKEN=""                        # empty is fine for public models such as Qwen/Qwen3-8B
export MODEL_HANDLE="Qwen/Qwen3-8B"
export MAX_MODEL_LEN=8192
export SGLANG_IMAGE="lmsysorg/sglang:latest-cu130"
docker pull "$SGLANG_IMAGE"
docker run -d \
  --name sglang-server \
  --gpus all \
  --ipc host \
  --cap-add SYS_NICE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  "$SGLANG_IMAGE" \
  sglang serve --model-path "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length $MAX_MODEL_LEN \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --enable-cache-report \
    --trust-remote-code
timeout 900 bash -c 'until curl -sf http://localhost:30000/health > /dev/null 2>&1; do sleep 10; done' \
  || { echo "Server failed to start within 900s"; docker logs sglang-server | tail -50; exit 1; }
```

| Flag | หมายเหตุจาก playbook |
|---|---|
| `--context-length` | ชื่อใน SGLang ของ `--max-model-len` ใน vLLM |
| `--mem-fraction-static 0.85` | เทียบเท่า `--gpu-memory-utilization` ของ SGLang บน Spark ให้ลดลงไปทาง `0.75` เมื่อหน่วยความจำตึง |
| `--attention-backend flashinfer` | attention backend ที่ผ่านการ validate สำหรับ Blackwell |
| `--enable-cache-report` | เติมค่า `usage.prompt_tokens_details.cached_tokens` ให้คุณเห็นว่า prefix cache ทำงาน |
| `--quantization modelopt_fp4` | เพิ่มเมื่อใช้ checkpoint แบบ NVFP4 ที่อยู่ในรายการที่รองรับ (เช่น `nvidia/Qwen3-32B-FP4`) |

> 💡 การเปิดครั้งแรกจะดาวน์โหลด weights และ capture CUDA graph playbook เผื่อเวลาไว้ ~10–15 นาทีสำหรับ `Qwen/Qwen3-8B` และนานกว่านั้นสำหรับโมเดล MoE ขนาดใหญ่

ลองถามอะไรสักอย่าง Qwen3 คิดก่อนตอบ จึงควรเผื่อ token ไว้ให้:

```spark
{"target": "sglang", "which": "a", "model": "Qwen/Qwen3-8B",
 "messages": [{"role": "user", "content": "What is the difference between speed and velocity? Two sentences."}],
 "max_tokens": 150}
```

Lab 06-3 รัน Step 6 และ 7 ของ playbook จาก Python: คำขอแบบ `json_schema` ที่แล็บตรวจคำตอบเทียบกับ schema และบทสนทนาสองรอบกับติวเตอร์ฟิสิกส์ ซึ่งรอบที่สองควรรายงาน `cached_tokens` > 0:

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/labs/lab03_sglang_features.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: ไม่มี Spark จึงใช้ Ollama แทน ไม่ใช่ SGLang)

```
◆ endpoint: LAPTOP STAND-IN · Ollama on this Mac (not SGLang, not the Spark) · model gemma3:4b

▣ STEP 1 · structured output — response_format json_schema (playbook Step 7; max_tokens 150, so answers are kept short)
· ANSWER  { "languages": [     {         "name": "Python",         "primary_use": "Data science, web development, scripting",         "year_created": 1991     },     …
✓ valid JSON matching the schema · 3 languages · Python (1991), JavaScript (1995), Java (1995)

▣ STEP 2 · prefix caching — the playbook's two-turn conversation (Step 6)
│ turn 1: prompt_tokens   74 · cached_tokens 0
│ turn 2: prompt_tokens  194 · cached_tokens 69
◆ LAPTOP STAND-IN: Ollama keeps its own prompt cache, so it may report cached tokens too. That shows the API field, not SGLang's RadixAttention.
```

ต่างจาก playbook สองจุด: แล็บจำกัด `max_tokens` ไว้ที่ 150 (playbook ใช้ 512 สำหรับคำขอ JSON) และขอคำตอบสั้น ๆ เพื่อไม่ให้ JSON ถูกตัดกลางทาง การรันครั้งแรกที่ให้คำตอบยาวถูกตัดจริง และ `json.loads` ล้มเหลว นี่คือรูปแบบความผิดพลาดที่ต้องระวัง สัญญาณอีกอย่างของ playbook อยู่ใน log ของเซิร์ฟเวอร์:

```bash
# on: spark
docker logs sglang-server 2>&1 | grep "cached-token" | tail -10
```

playbook บอกให้มองหาค่า `#cached-token` ที่มากกว่า 0 ในรอบหลัง ๆ

> ⚠ โมเดลแบบ hybrid Mamba/SSM เช่น `Qwen/Qwen3.6-35B-A3B` จะรายงาน cached tokens เป็น 0 เสมอ: SGLang ข้ามการใช้ prefix ซ้ำข้ามคำขอสำหรับโมเดลเหล่านี้ ให้ทดสอบ prefix caching กับโมเดลที่ใช้ attention แบบมาตรฐาน เช่น `Qwen/Qwen3-8B` (playbook)

```bash
# on: spark
docker stop sglang-server && docker rm sglang-server
```

✓ Checkpoint: lab 06-3 พิมพ์ `✓ valid JSON matching the schema` และบน SGLang รอบที่สองรายงาน `cached_tokens` มากกว่า 0

## 3 · TensorRT-LLM: runtime ที่ NVIDIA ปรับแต่งมาแล้ว บน :8355

TensorRT-LLM คือไลบรารีของ NVIDIA ที่รวม kernel ที่ปรับแต่งแล้ว การจัดการหน่วยความจำ quantization และ parallelism สำหรับ inference ส่วน `trtllm-serve` ห่อมันเป็นเซิร์ฟเวอร์ที่เข้ากันได้กับ OpenAI การปรับจูนส่วนใหญ่อยู่ในไฟล์ YAML เล็ก ๆ ที่ส่งผ่าน `--extra_llm_api_options` ไม่ใช่ใน flag บนบรรทัดคำสั่ง

ตรวจว่า container มองเห็น GPU แล้วรันเส้นทางตั้งต้นของ playbook คือ Llama 3.1 8B Instruct ที่ NVFP4 คำสั่งนี้ใช้ `--network host` จึงไม่ต้องใช้ `-p`: เซิร์ฟเวอร์ listen ที่พอร์ต 8355 ของ Spark โดยตรง:

```bash
# on: spark
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"     # or rely on the token from `hf auth login`
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
mkdir -p "$HOME/.cache/huggingface"
docker run --rm --gpus all "$TRTLLM_IMAGE" python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

docker run --name trtllm-server --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download "$MODEL_HANDLE" &&
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --host 0.0.0.0 \
      --port 8355 \
      --max_batch_size 64 \
      --trust_remote_code \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```

| การตั้งค่า | หน้าที่ |
|---|---|
| `kv_cache_config.free_gpu_memory_fraction: 0.9` | สัดส่วน KV cache ของ TensorRT-LLM คล้าย flag utilization ของ vLLM |
| `--max_batch_size 64` | จำนวน sequence สูงสุดใน batch เดียว คล้าย `--max-num-seqs` ของ vLLM |
| `cuda_graph_config.enable_padding` | เติม (pad) batch ให้เท่าขนาด CUDA graph ที่ capture ไว้ เพื่อให้ replay ได้เร็ว |

คำสั่งนี้รันแบบ foreground (`-it`) เปิดเทอร์มินัลนั้นทิ้งไว้ แล้วทดสอบจากเทอร์มินัลที่สองด้วยคำขอของ playbook:

```bash
# on: spark
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Llama-3.1-8B-Instruct-FP4", "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}], "max_tokens": 64}'
```

```spark
{"target": "trtllm", "which": "a", "model": "nvidia/Llama-3.1-8B-Instruct-FP4",
 "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}], "max_tokens": 64}
```

playbook ครอบคลุมอะไรอีกบ้าง และอะไรที่ยังไม่ครอบคลุม:

- **Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16** มีสูตรของตัวเองที่ใช้ `--reasoning_parser nano-v3 --tool_parser qwen3_coder` และไฟล์ `nano_v3.yaml` (ดู Step 5 ของ playbook)
- **gpt-oss-20b / 120b** ต้องตั้งค่า `TIKTOKEN_ENCODINGS_BASE` ไว้ต้นสคริปต์ `bash -c` (ดู playbook)
- **หลายเครื่อง (multi-node)** ใช้ OpenMPI แทน Ray: มี hostfile ที่ระบุ IP ของ QSFP หนึ่งตัวต่อ Spark, `trtllm-mn-entrypoint.sh` ของ playbook, image `1.3.0rc5` และ `trtllm-llmapi-launch trtllm-serve … --tp_size 2` โมเดลที่ผ่านการ validate ในกรณีนี้คือ `nvidia/Qwen3-235B-A22B-FP4`
- **Qwen3.6-35B-A3B แบบพร้อมใช้กับเอเจนต์**: playbook บอกว่าการตั้งค่าตอนเปิดยัง "pending validation" (รอการ validate) และเตือนไม่ให้ยืมค่า parser จากโมเดลตระกูลอื่น ถ้าจะใช้กับเอเจนต์ในตอนนี้ ให้ใช้ vLLM (Module 05)

หยุดด้วย `Ctrl+C` ในเทอร์มินัลที่ serve อยู่ `--rm` จะลบ container ให้

✓ Checkpoint: curl ที่ :8355 คืนค่า array `choices` และคุณบอกได้ว่า TensorRT-LLM เก็บการตั้งค่า KV cache ไว้ที่ไหน

## 4 · NIM: container สำเร็จรูปหนึ่งตัวต่อหนึ่งโมเดล

NIM คือ container ที่มีโมเดล engine และการตั้งค่าที่จูนแล้วอยู่ข้างในครบ คุณเลือก container แทนที่จะเลือกโมเดลและชุด flag เอง ค่าตั้งต้นของ playbook สำหรับ DGX Spark คือ NIM ของ Llama 3.1 8B Instruct ส่วนตัวอื่น ๆ มีอยู่ใน [NGC catalog](https://catalog.ngc.nvidia.com) เช่น Qwen3-32B สำหรับ DGX Spark

```bash
# on: spark
docker login nvcr.io --username '$oauthtoken'        # paste the NGC API key at the prompt
export NGC_API_KEY="<YOUR_NGC_API_KEY>"               # the container needs it to download model assets
export CONTAINER_NAME="nim-llm-demo"
export IMG_NAME="nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest"
export LOCAL_NIM_CACHE=~/.cache/nim
export LOCAL_NIM_WORKSPACE=~/.local/share/nim/workspace
mkdir -p "$LOCAL_NIM_WORKSPACE" && chmod -R a+w "$LOCAL_NIM_WORKSPACE"
mkdir -p "$LOCAL_NIM_CACHE" && chmod -R a+w "$LOCAL_NIM_CACHE"
docker run -it --rm --name=$CONTAINER_NAME \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -v "$LOCAL_NIM_WORKSPACE:/opt/nim/workspace" \
  -p 8000:8000 \
  $IMG_NAME
```

> ⚠ `export NGC_API_KEY=…` ทำให้คีย์ไปอยู่ใน history ของเชลล์นั้น บน Spark ควรใช้ `read -rs NGC_API_KEY && export NGC_API_KEY` แทน (อ่านคีย์โดยไม่แสดงบนหน้าจอ)

model id ที่ถูก serve คือ `meta/llama-3.1-8b-instruct` ไม่ใช่ handle ของ Hugging Face คำขอทดสอบของ playbook:

```bash
# on: spark
curl -X 'POST' 'http://0.0.0.0:8000/v1/chat/completions' \
  -H 'accept: application/json' -H 'Content-Type: application/json' \
  -d '{"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "system", "content": "detailed thinking on"}, {"role": "user", "content": "Can you write me a song?"}], "top_p": 1, "n": 1, "max_tokens": 15, "frequency_penalty": 1.0, "stop": ["hello"]}'
```

NIM ใช้พอร์ต 8000 เหมือน vLLM ตัว runner จึงเข้าถึงด้วย target `nim`:

```spark
{"target": "nim", "which": "a", "model": "meta/llama-3.1-8b-instruct",
 "messages": [{"role": "user", "content": "Can you write me a song? Four lines."}], "max_tokens": 80}
```

| | NIM | vLLM / SGLang / TensorRT-LLM |
|---|---|---|
| สิ่งที่คุณเลือก | container | โมเดล image และ flag |
| การจูน | ทำให้แล้ว แยกตามโมเดล | คุณทำเอง: หน่วยความจำ batch และ parser |
| โมเดล | ตาม NIM catalog | checkpoint ใดก็ได้บน Hugging Face ที่รองรับ รวมถึงโมเดลที่คุณ fine-tune เอง |
| การล็อกอิน | NGC API key | Hugging Face token สำหรับโมเดลที่ gated |

หยุดด้วย `Ctrl+C` (container ใช้ `--rm`) โมเดลยังอยู่ใน `~/.cache/nim` playbook จะลบด้วย `rm -rf "$LOCAL_NIM_CACHE"` เฉพาะเมื่อคุณต้องการพื้นที่ดิสก์คืนเท่านั้น

✓ Checkpoint: NIM ตอบที่ :8000 ด้วยโมเดล `meta/llama-3.1-8b-instruct` และ `/v1/models` แสดง id นั้น

## 5 · Nemotron Nano และ Super บน Spark เครื่องเดียว

Nemotron 3 คือตระกูลโมเดลเปิดของ NVIDIA สำหรับการให้เหตุผล (reasoning) การใช้เครื่องมือ (tool use) และ context ยาว playbook ของ Nemotron serve สองขนาดบน Spark เครื่องเดียว:

| | Nemotron 3 Nano Omni | Nemotron 3 Super |
|---|---|---|
| Checkpoint | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (weights ในเครื่อง) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| Engine และ image | vLLM `vllm/vllm-openai:v0.20.0` | vLLM `vllm/vllm-openai:cu130-nightly` **หรือ** TensorRT-LLM `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9` |
| พอร์ต · ชื่อที่ serve | 8000 · `nemotron_3_nano_omni` | vLLM: 8000 · `nemotron-3-super` · TensorRT-LLM: **8123** · อ่านจาก log |
| ตัวเลือกด้านหน่วยความจำ | `--max-num-seqs 8`, `--max-model-len 131072`, `--gpu-memory-utilization 0.8` | `--max-num-seqs 4`, `--max-model-len 1000000`, `--gpu-memory-utilization 0.90`, `--kv-cache-dtype fp8` |
| Parser | `--reasoning-parser nemotron_v3` · `--tool-call-parser qwen3_coder` | vLLM: plugin `super_v3` · TensorRT-LLM: `nano-v3` · ทั้งคู่ใช้ `qwen3_coder` |

**Nano (vLLM)** ดาวน์โหลด weights ของ Omni ไปไว้ในโฟลเดอร์บน Spark แล้วชี้ `WEIGHTS` ไปที่โฟลเดอร์นั้น image พื้นฐานไม่มีแพ็กเกจด้านเสียง คำสั่งจึงติดตั้ง `vllm[audio]` ก่อน:

```bash
# on: spark
WEIGHTS=/path/to/nemotron-3-nano-omni-weights
docker pull vllm/vllm-openai:v0.20.0
docker run --rm -it \
  --gpus all \
  --ipc=host -p 8000:8000 \
  --shm-size=16g \
  --name vllm-nemotron-omni \
  -v "${WEIGHTS}:/model:ro" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.20.0 -c  \
  "pip install vllm[audio] && vllm serve /model \
  --served-model-name=nemotron_3_nano_omni \
  --max-num-seqs 8 \
  --max-model-len 131072 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --limit-mm-per-prompt '{\"video\": 1, \"image\": 1, \"audio\": 1}' \
  --media-io-kwargs '{\"video\": {\"fps\": 2,  \"num_frames\": 256}}' \
  --allowed-local-media-path=/ \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder"
```

ถ้าหน่วยความจำไม่พอ playbook จะลด `--gpu-memory-utilization` เป็น `0.70` ก่อน แล้วจึงลด `--max-model-len` เป็น `32768` เนื่องจาก serve ด้วย vLLM บน :8000 target `vllm` จึงเข้าถึงได้:

```spark
{"target": "vllm", "which": "a", "model": "nemotron_3_nano_omni",
 "messages": [{"role": "user", "content": "New York is a great city because..."}], "max_tokens": 150}
```

**Super (เส้นทาง vLLM)** ดาวน์โหลด plugin ของ reasoning parser ก่อน environment variable สี่ตัวคือวิธีแก้สำหรับ GPU เดียวของ playbook (kernel NVFP4 แบบ Marlin, context ยาว และ all-reduce):

```bash
# on: spark
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
docker pull vllm/vllm-openai:cu130-nightly
docker run --rm -it --gpus all \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py \
  -p 8000:8000 \
  vllm/vllm-openai:cu130-nightly \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --max-model-len 1000000 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

```spark
{"target": "vllm", "which": "a", "model": "nemotron-3-super",
 "messages": [{"role": "user", "content": "Summarize what Latent MoE changes for routing traffic."}], "max_tokens": 150}
```

**Super (เส้นทาง TensorRT-LLM)** ดาวน์โหลด checkpoint ลงโฟลเดอร์ในเครื่อง เขียนไฟล์ `extra-llm-api-config.yml` ของ playbook ไว้ข้าง ๆ (Step 9 ของแท็บ Nemotron Super; คัดลอกให้ตรงทุกตัวอักษร) แล้ว:

```bash
# on: spark
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --local-dir ./nemotron-super-nvfp4
docker run --rm -it --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e TLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -p 8123:8123 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9 \
  trtllm-serve nemotron-super-nvfp4 \
    --host 0.0.0.0 \
    --port 8123 \
    --max_batch_size 8 \
    --tp_size 1 --ep_size 1 \
    --max_num_tokens 8192 \
    --trust_remote_code \
    --reasoning_parser nano-v3 \
    --tool_parser qwen3_coder \
    --extra_llm_api_options extra-llm-api-config.yml \
    --max_seq_len 1048576
```

> ⚠ สูตรนี้ใช้พอร์ต **8123** ไม่ใช่ 8355 ที่ TensorRT-LLM ใช้ตามปกติ Lab 06-1 ตรวจทั้งสองพอร์ต ถ้าจะให้บล็อก ⚡ และแล็บอื่น ๆ ชี้ไปที่มัน ให้ตั้ง `SPARK_URL_TRTLLM=http://spark-a:8123/v1` ใน 🖥 Spark setup และอ่านชื่อโมเดลที่ serve จาก log ของ `trtllm-serve` (playbook ใช้ค่า placeholder)

ทำไม Super ถึงใส่ใน Spark เครื่องเดียวได้ (จากหมายเหตุด้านสถาปัตยกรรมใน playbook): **LatentMoE** รัน expert ในมิติที่ถูกบีบอัด และเปิดใช้พารามิเตอร์ราว 12B จาก 120B ต่อ token **MTP** (ชั้น multi-token-prediction หนึ่งชั้นที่ฝังมาใน checkpoint) ร่าง (draft) 3 token สำหรับ speculative decoding (Module 07) ชั้น **Mamba-2 hybrid** เก็บ SSM state แทน KV cache ที่โตขึ้นเรื่อย ๆ นี่คือเหตุผลที่ config ของ TensorRT-LLM ตั้ง `enable_block_reuse: false`: state ของ Mamba ทำ prefix cache ไม่ได้

> 💡 มี Spark สองเครื่อง? checkpoint แบบ **FP8** (128.4 GB ความแม่นยำสูงกว่า NVFP4) รันข้ามทั้งสองเครื่องด้วย vLLM tensor parallel ได้: [Module 05 ส่วนที่ 6](../05_vllm/TUTORIAL.md)

✓ Checkpoint: โมเดล Nemotron หนึ่งตัวตอบผ่าน OpenAI API ได้ และคุณบอกได้ว่าสูตร Nemotron แต่ละสูตรใช้พอร์ตและชื่อที่ serve อะไร

## 6 · การประชันที่ยุติธรรม

"Engine X เร็วกว่า" ไม่มีความหมายเลย ถ้าทั้งสอง engine ไม่ได้ทำงานชิ้นเดียวกัน กฎที่แบบฝึกหัดตรวจ:

| สิ่งที่ต้องเหมือนกัน | ทำไม |
|---|---|
| โมเดล **และ** precision | Llama 3.1 8B Instruct อยู่ในรายการของทุก engine: `nvidia/Llama-3.1-8B-Instruct-NVFP4` (vLLM), `nvidia/Llama-3.1-8B-Instruct-FP4` (SGLang, TensorRT-LLM) และ NIM ของ Llama 3.1 8B สำหรับ DGX Spark |
| prompt, `max_tokens`, concurrency | ความยาวต่างกันคืองานต่างกัน |
| `temperature: 0` | greedy decoding ทำงานเท่าเดิมทุกครั้งที่รัน |
| ส่งคำขอ warm-up ก่อน | คำขอแรกต้องจ่ายค่าโหลดและค่า capture CUDA graph |
| **มี engine รันอยู่ตัวเดียว** | engine ใช้ 128 GB และ 273 GB/s ร่วมกัน |
| รายงาน **median** จากการรันหลายครั้ง | ค่าที่ช้าผิดปกติครั้งเดียว หรือการรันที่ฟลุคครั้งเดียว ไม่ควรเป็นตัวตัดสินผล |

Lab 06-1 ใช้กฎเหล่านี้กับทุก engine ที่ตอบสนอง:

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/labs/lab01_engine_bakeoff.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: ไม่มี Spark ทุก endpoint บน Spark จึงล่ม และมีแค่แถวของแล็ปท็อปที่ได้รัน)

```
▣ STEP 1 · probe every OpenAI endpoint on the Spark
│ engine                                port   base URL         status  models
│ ────────────────────────────────────  ─────  ───────────────  ──────  ──────
│ vLLM / NIM                            8000   (no Spark host)  ○ down  —
│ SGLang                                30000  (no Spark host)  ○ down  —
│ TensorRT-LLM                          8355   (no Spark host)  ○ down  —
│ TensorRT-LLM (Nemotron Super recipe)  8123   (no Spark host)  ○ down  —
│ llama.cpp                             30080  (no Spark host)  ○ down  —
│ LM Studio                             1234   (no Spark host)  ○ down  —
│ Ollama                                11434  (no Spark host)  ○ down  —

▣ STEP 2 · run 3 prompts on each engine that is up (warm-up first · temperature 0 · max_tokens 100)
→ no Spark engine is up · running the method on Ollama on THIS laptop (gemma3:4b) as a LAPTOP STAND-IN

▣ STEP 3 · comparison
│ engine                              model      median TTFT  median tok/s  tokens  first answer starts
│ ──────────────────────────────────  ─────────  ───────────  ────────────  ──────  ────────────────────────────────────────────────
│ LAPTOP STAND-IN (Ollama, this Mac)  gemma3:4b  2286 ms      77.6          256     Tensor parallelism distributes a single tensor a
◆ LAPTOP STAND-IN: this row shows the method, not an engine. Its numbers are this Mac's and are never compared with a Spark engine.
```

บน Spark ของคุณ วงจรคือ: เปิด engine หนึ่งตัว (ส่วนที่ 2–5 หรือ Module 05) → รัน lab 06-1 → หยุด engine → เปิดตัวถัดไป → รัน lab 06-1 อีกครั้ง การรันแต่ละครั้งจะเพิ่มแถวของ engine นั้น เก็บตารางไว้ ถ้าต้องการทดสอบผู้ใช้หลายคนพร้อมกัน ให้ชี้ lab 05-3 ไปที่ URL ของแต่ละ engine ด้วย: engine หนึ่งอาจชนะที่ผู้ใช้หนึ่งคน แต่แพ้ที่แปดคน

✓ Checkpoint: คุณมีตารางจาก lab 06-1 ที่มีแถวของ engine บน Spark อย่างน้อยหนึ่งแถว (หรือแถวของแล็ปท็อปในโหมด DRY) และบอกได้สามข้อว่าอะไรทำให้การเปรียบเทียบไม่ยุติธรรม

## 7 · เลือก engine ไหนดี?

การวัดบอกได้ว่า engine ไหนเร็วที่สุด แต่ **ความต้องการที่ขาดไม่ได้ (hard needs)** ของงานคุณเป็นตัวตัดสินว่า engine ไหนเข้ารอบบ้าง Lab 06-2 ให้คะแนนแต่ละ engine จากสิ่งที่ playbook ของ Spark แสดงไว้เท่านั้น คัด engine ที่ไม่มีเส้นทางที่มีเอกสารรองรับสำหรับความต้องการนั้นออก และพิมพ์ข้อเท็จจริงจาก playbook ที่อยู่เบื้องหลังแต่ละการเลือก (เป็นการถอดความ; คำในเครื่องหมายคำพูดยกมาคำต่อคำ):

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/labs/lab02_which_engine.py --need tools,two-sparks
```

**Expected output** (ออฟไลน์ ผลลัพธ์เหมือนกันทุกเครื่อง)

```
▣ STEP 2 · six workloads
│ workload                                              needs           pick                ruled out
│ ────────────────────────────────────────────────────  ──────────────  ──────────────────  ───────────
│ Hotel concierge agent (Module 14): tool calls, long…  tools+prefix    vLLM :8000          NIM
│ RAG answers that must be valid JSON for a dashboard   json+prefix     SGLang :30000       —
│ Llama 3.3 70B at bf16 — bigger than one Spark         two-sparks      vLLM :8000          SGLang, NIM
│ Nemotron 3 Super for reasoning, one Spark             nemotron-super  vLLM :8000          SGLang, NIM
│ A demo tomorrow, supported container, no tuning       least-setup     NIM :8000           —
│ Lowest latency for one validated model                latency         TensorRT-LLM :8355  —

▣ STEP 4 · your workload: tools + two-sparks
│ engine        score
│ ────────────  ─────────
│ vLLM          4
│ SGLang        ruled out
│ TensorRT-LLM  3
│ NIM           ruled out
═ pick: vLLM on :8000
```

หลักคิดง่าย ๆ ที่ได้ออกมา:

- **เอเจนต์บน Spark เครื่องเดียว → vLLM** มีสูตรพร้อมใช้กับเอเจนต์ที่ผ่านการ validate เพียงตัวเดียว และ tool calling ได้รับการพิสูจน์ตั้งแต่ต้นจนจบ (lab 05-4)
- **RAG ที่ใช้ JSON หนัก ๆ และ prompt ที่ใช้ร่วมกัน → SGLang** structured output และ prefix caching คือจุดแข็งที่มีเอกสารรองรับ
- **Spark สองเครื่อง → vLLM (Ray) หรือ TensorRT-LLM (MPI)** playbook ของ SGLang และ NIM ไม่มีเส้นทางแบบหลายเครื่องบน Spark
- **ทางที่เร็วที่สุดไปสู่เดโมที่มีการรองรับ → NIM** ถ้าโมเดลที่คุณต้องการมี NIM สำหรับ DGX Spark
- **latency สำหรับโมเดลเดียวที่จะรันเป็นเดือน ๆ → TensorRT-LLM** แล้ววัดเทียบกับ vLLM ด้วย lab 06-1

scorecard สะท้อนสิ่งที่ playbook *แสดงไว้* คะแนน 1 หมายถึง "ลองดูแล้ววัดผล" ไม่ได้หมายถึง "ไม่รองรับ"

✓ Checkpoint: คุณเลือก engine สำหรับ capstone ของ Module 20 ได้ (เอเจนต์ที่เรียกใช้เครื่องมือและ serve โมเดลที่ fine-tune แล้ว) และบอกข้อเท็จจริงจาก playbook ที่อยู่เบื้องหลังการเลือกนั้นได้

## Labs — รันแล็บได้ที่นี่

**labs/lab01_engine_bakeoff.py** — ตรวจทุก endpoint แบบ OpenAI บน Spark แล้วรัน prompt ชุดเดียวกันกับทุก engine ที่ทำงานอยู่

**labs/lab02_which_engine.py** — แปลงความต้องการของงานให้เป็นการเลือก engine โดยมีข้อเท็จจริงจาก playbook รองรับทุกการเลือก

**labs/lab03_sglang_features.py** — JSON ที่บังคับตาม schema และ prefix cache ของ SGLang ตรวจสอบด้วย Python

Lab 01 และ 03 เรียก engine บน Spark ของคุณ หรือ laptop stand-in ที่ติดป้ายไว้ ส่วน Lab 02 เป็นแบบออฟไลน์และรันได้ทุกที่

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 06 — วางแผนการประชันที่ยุติธรรม** เปิด `week25/06_sglang_trtllm_nim/exercises/ex06_fair_bakeoff.py` ในไฟล์มี `TODO` สี่จุด:

1. `PORTS`: พอร์ต OpenAI ของแต่ละ engine ตาม playbook (รวมถึงสูตร TensorRT-LLM ของ Nemotron Super)
2. `port_conflicts(ports)`: ทุกคู่ของ engine ที่ใช้พอร์ตเดียวกัน
3. `fairness_problems(a, b)`: ทุกเหตุผลที่ทำให้การเปรียบเทียบการรันสองครั้งไม่ยุติธรรม
4. `summarise(runs)`: ค่า median ของ TTFT และ tok/s จากการรันซ้ำหลายครั้ง

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/exercises/ex06_fair_bakeoff.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสี่จุด บันทึกจาก Mac เครื่องนี้)

```
✓ PORTS: vLLM 8000 · SGLang 30000 · TensorRT-LLM 8355 · NIM 8000 · Nemotron Super TRT-LLM 8123
✓ port_conflicts: finds every shared port · on a Spark: NIM and vLLM both want :8000
✓ fairness_problems: identical → fair · precision, temperature, warm-up, a second engine, max_tokens + concurrency, model + prompts all caught
✓ summarise: median of 3 runs (TTFT 140 ms, 38.0 tok/s) — one slow outlier does not decide the result

▣ your plan, applied to four engines serving Llama 3.1 8B Instruct
│ vllm    :8000   start → warm-up → 3 prompts → stop
│ nim     :8000   start → warm-up → 3 prompts → stop
│ trtllm  :8355   start → warm-up → 3 prompts → stop
│ sglang  :30000  start → warm-up → 3 prompts → stop
◆ nim and vllm share :8000 — run them one after the other, or map one to -p 8001:8000
```

<details><summary>คำใบ้ — ทำไมต้องใช้ median ไม่ใช่ค่าเฉลี่ย (mean) หรือค่าที่ดีที่สุด?</summary>

ในข้อมูลทดสอบมีการรันหนึ่งครั้งที่ใช้เวลา 900 ms กว่าจะได้ token แรก (ค่าผิดปกติ (outlier): อาจมีโปรเซสอื่นตื่นขึ้นมาพอดี) ค่าเฉลี่ยจะเป็น 387 ms ซึ่งไม่ตรงกับการรันจริงครั้งไหนเลย ส่วนการรันที่ดีที่สุด (120 ms) เป็นแค่โชค median (140 ms) คือสิ่งที่คำขอทั่วไปเจอจริง Python มี `statistics.median` ให้ใช้

</details>

<details><summary>คำใบ้ — <code>{"max_tokens": 256, "concurrency": 8}</code> มีปัญหากี่ข้อ?</summary>

สองข้อ: หนึ่งข้อต่อกฎที่ถูกละเมิด สร้าง list ที่มีข้อความหนึ่งรายการต่อกฎที่คุณตรวจ แล้ว return list นั้น แม้ว่าจะว่างก็ตาม

</details>

<details><summary>ท้าทายเพิ่ม — ทำให้การประชันเป็นแบบ concurrent</summary>

เพิ่มคอลัมน์ `concurrency` ใน lab 06-1 โดยรันแต่ละ engine ที่ 1 และ 8 คำขอพร้อมกัน (นำ `run()` จาก lab 05-3 มาใช้ซ้ำ) engine ที่ชนะที่ 1 ยังชนะที่ 8 ด้วยหรือไม่?

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสี่เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | vLLM หรือ NIM ยังรันอยู่ รัน `docker ps` แล้วหยุดมัน หรือ map เป็น `-p 8001:8000` |
| หน่วยความจำไม่พอตอนเปิด engine ตัวที่สอง | แต่ละ engine จองหน่วยความจำส่วนใหญ่ไว้ (ส่วนที่ 1) หยุดตัวแรกก่อน บน Spark ให้ล้าง page cache ด้วย: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| SGLang หน่วยความจำไม่พอ | ลด `--mem-fraction-static` (เช่น `0.7`) และ/หรือ `--context-length` (playbook) |
| SGLang ตอบคำขอแรกช้า | เป็นช่วง JIT ของ kernel และการ capture CUDA graph รอข้อความพร้อมใช้งานใน `docker logs sglang-server` |
| `cached_tokens` ของ SGLang เป็น 0 หรือ `n/a` | เพิ่ม `--enable-cache-report` สำหรับโมเดล hybrid Mamba/SSM (Qwen3.6-35B-A3B) ค่า 0 เป็นเรื่องปกติ (playbook) |
| response_format แบบ `json_schema` คืน error บน SGLang | ใช้ `lmsysorg/sglang:latest-cu130` (playbook) |
| JSON จาก lab 06-3 parse ไม่ผ่าน | `max_tokens` ตัดคำตอบกลางทาง เพิ่มค่า (playbook ใช้ 512) หรือขอฟิลด์ที่สั้นลง |
| TensorRT-LLM หน่วยความจำไม่พอขณะโหลด weights | ตั้ง `TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1` ก่อนเปิด container (playbook) |
| TensorRT-LLM ไม่ตอบที่ 8355 | ยังโหลดอยู่ หรือปิดตัวไปแล้ว: ดูเทอร์มินัลที่ serve และ `lsof -i :8355` สำหรับ Nemotron Super พอร์ตคือ 8123 |
| NIM ขึ้น `Invalid credentials` ตอน `docker login` | username คือ `$oauthtoken` ตรงตัวอักษร (อยู่ในเครื่องหมายคำพูดเดี่ยว) วางคีย์โดยไม่มีช่องว่างเกิน |
| NIM คืน 404 สำหรับโมเดล | ใช้ `"model": "meta/llama-3.1-8b-instruct"` ซึ่งเป็น id ของ NIM เอง ไม่ใช่ handle ของ Hugging Face |
| Nemotron Super: `Error loading reasoning parser` | รัน `wget` สำหรับ `super_v3_reasoning_parser.py` แล้วสั่ง `docker run` จากไดเรกทอรีนั้น (playbook) |
| Nemotron: `curl` คืน error เรื่องโมเดล / 404 | ใช้ชื่อที่ serve: `nemotron_3_nano_omni` (Nano) หรือ `nemotron-3-super` (Super บน vLLM) |

## Next — บทถัดไป

ไปต่อที่ [Lab 07 — NVFP4 quantization และ speculative decoding](../07_nvfp4_speculative/TUTORIAL.md): quantize โมเดลเป็น NVFP4 ด้วยตัวเอง และเร่ง decoding ด้วย draft model หรือ MTP
