# ▶ Reef Lab 06 — วัดประสิทธิภาพบน DGX Spark: engine, workflow, คุณภาพ และภาษีของ sandbox

> ส่วนหนึ่งของ Week 26 · NemoClaw บน DGX Spark ตั้งแต่ระดับเริ่มต้นจนถึงผู้เชี่ยวชาญ คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง แล็บฝั่งแล็ปท็อป (NAT CLI, OpenShell CLI และ policy model) รันจริงได้ทุกที่ ส่วนแล็บฝั่ง Spark รันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark ไม่เสียเงิน) โดยจะแสดงคำสั่งให้ดู และผลลัพธ์จะเป็นอย่างใดอย่างหนึ่ง: RECORDED ที่บันทึกจาก Spark จริง, REFERENCE ที่ยกมาจากเอกสารหรือ playbook ของ NVIDIA หรือ EXAMPLE ที่ติดป้ายไว้ชัดเจน

**สิ่งที่คุณจะได้ลงมือทำ**
- แยกคำถาม "claw ของฉันเร็วไหม?" ออกเป็นสี่คำถาม คือ engine, workflow, คุณภาพ และ overhead ของ sandbox แล้ววัดแต่ละข้อด้วยเครื่องมือของมันเอง
- คำนวณเพดานความเร็ว decode แบบ single-stream จากแบนด์วิดท์ 273 GB/s ของ Spark และเข้าใจว่าทำไม engine จริงถึงทำได้ต่ำกว่านั้น
- รัน `nat eval` จริงพร้อม NAT profiler กับ Alto Ops Claw แล้วอ่านค่า p50/p95, จำนวน LLM call ต่อเทิร์น, token และคะแนนคุณภาพ จากไฟล์ที่มันเขียนออกมา
- รัน `nat sizing calc` จริง (ขนาดเล็ก) คำนวณ GPU ที่ต้องใช้ซ้ำด้วยมือ และเห็นว่าทำไมข้อมูลสองจุดพิสูจน์อะไรไม่ได้
- วัด "ภาษี" ให้ถูกวิธี: eval ชุดเดียวกันผ่านสองเส้นทาง แล้วตัดสินว่าส่วนต่างที่เห็นมากกว่า noise หรือไม่
- จับเวลา claw ผ่าน API ของมันเอง แบบที่คุณจะทำกับ Hermes บนพอร์ต 8642 แล้วรวมทั้งสี่ชั้นไว้ในรายงานเดียว

**Time** ~90 นาที · **Difficulty** advanced · **Hardware** แล็ปท็อป (NAT 1.9.0 + Ollama) · มี DGX Spark 1 เครื่องก็ได้ ไม่บังคับ

**Sources:** research tutorial ของคอร์ส *NemoClaw on DGX Spark — Beginner to Expert* (Part 5, แล็บ L5.1–L5.5) ซึ่งอ้างอิง [NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html) · [NAT evaluate](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html) · [NAT sizing calculator](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/sizing-calc.html) · [ai-muninn — Nemotron 3 Nano on DGX Spark](https://ai-muninn.com/en/blog/dgx-spark-nemotron-3-nano-w4a16-74-toks) · [Exxact — local agents on DGX Spark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark) · [Exxact — inference engines on DGX Spark](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark) · [Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) · [NVIDIA developer forum — Nemotron 3 NVFP4](https://forums.developer.nvidia.com/t/dgx-spark-nemotron3-and-nvfp4-getting-to-65-tps/355261) · [Exxact local-agent-benchmark](https://github.com/Exxact-Software/local-agent-benchmark)

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจด้วย | ใช้ทำอะไร |
|---|---|---|
| NAT 1.9.0 พร้อม extra ของ profiler และ eval | `week26/.venv-nat/bin/nat --version` | `nat eval`, profiler และ `nat sizing calc` รันจริงบนแล็ปท็อปเครื่องนี้ |
| Ollama บนแล็ปท็อป ที่มี `nemotron-3-nano` และ `gemma3:4b` | `curl -s http://localhost:11434/api/tags` | ตัวแทน vLLM บน Spark: nemotron-3-nano ใช้กับ agent ส่วน gemma3:4b ใช้กับการ sweep engine ที่ต้องการความเร็ว |
| แพ็กเกจ Alto Ops (Module 04) | ติดตั้งแบบ editable ไว้ใน `week26/.venv-nat` | tool `chiller_kpi` ที่คำถามใน eval เรียกใช้ |
| DGX Spark (ไม่บังคับ) | container `vllm-nat` (research tutorial Lab 3.2), sandbox `alto-ops` (Lab 3.8) และ Hermes claw | ตัวเลข engine, sandbox และ harness ของจริง |

```bash
# on: laptop
week26/.venv-nat/bin/nat --version
curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | grep -E "nemotron-3-nano|gemma3:4b"
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
nat, version 1.9.0
"name":"nemotron-3-nano:latest"
"name":"gemma3:4b"
```

> 📌 **กติกาสองข้อสำหรับทุกตัวเลขในโมดูลนี้**
> 1. **ตัวเลขจากแล็ปท็อปคือ LAPTOP STAND-IN** มาจาก Ollama บน Mac เครื่องนี้ ซึ่งแล็บอื่นอาจใช้อยู่พร้อมกัน ตัวเลขจึงแกว่ง มันมีไว้สอน *วิธีวัด* และห้ามนำไปเทียบกับตัวเลขของ Spark เด็ดขาด
> 2. **ตัวเลขของ Spark ยกมาเท่านั้น ไม่แต่งขึ้นเอง** ยังไม่มีโมดูลไหนได้รันบน Spark จริง ตัวเลขของ Spark ทุกตัวด้านล่างเป็นผลวัดของบุคคลที่สาม และระบุว่า "ตาม <แหล่งที่มา> ซึ่งอ้างใน research tutorial" ส่วนบล็อกผลลัพธ์ของ Spark ที่ไม่มีแหล่งที่มาคือรูปแบบ EXAMPLE ที่เว้น `…` ไว้ให้ตัวเลขของคุณ
>
> research tutorial เขียนโดยอิงเอกสาร NAT 1.8 แต่คอร์สนี้ใช้ NAT **1.9.0** จุดไหนที่ 1.9.0 ต่างออกไป จะบอกไว้ด้านล่างทุกจุด

✓ Checkpoint: `nat --version` แสดง 1.9.0 และมีโมเดลครบทั้งสองตัว หรือคุณรู้แล้วว่าจะรันได้แค่ส่วน DRY กับส่วนคำนวณ

## 1 · สี่ชั้น วัดแยกกัน

"claw ช้า" แปลได้สี่แบบ และแต่ละแบบมีเครื่องมือของมันเอง:

| ชั้น | คำถาม | เครื่องมือ | ตัวเลขหลัก | แล็บ |
|---|---|---|---|---|
| 1 · Engine / โมเดล | โมเดล decode เร็วแค่ไหน ทั้งตอนรันเดี่ยวและตอนมีโหลด | `vllm bench serve`, `ollama run --verbose` | TTFT, tok/s ต่อ request, tok/s รวม | 06-1 |
| 2 · Workflow | agent หนึ่งเทิร์นใช้เวลาเท่าไร และทำไม | `nat eval` + NAT profiler | runtime p50/p95, จำนวน LLM call และ token ต่อเทิร์น | 06-2, 06-3 |
| 3 · คุณภาพ | claw ทำงานได้ถูกต้องไหม | NAT evaluator (ragas, trajectory) หรือการตรวจแบบตรงตัว | คะแนนต่อคำถาม | 06-2 |
| 4 · Overhead ของ sandbox | proxy, การดัก TLS และ Landlock ของ OpenShell เพิ่มเวลาเท่าไร | `nat eval` ชุดเดียวกัน รันบน host และใน sandbox | Δ runtime p95 | 06-4 |

ความเร็วอย่างเดียวไม่ใช่คำตอบ research tutorial อ้าง agent benchmark ของ Exxact ว่า engine ที่เร็วแต่ใช้โมเดลที่เรียก tool พลาด จะช้ากว่าในทางปฏิบัติเมื่อเทียบกับตัวที่ช้าแต่เชื่อถือได้ ความน่าเชื่อถือและวินัยในการตอบแบบมีโครงสร้าง (structured output) สำคัญกว่า tok/s ดิบ

ตารางนี้คือตัวเลขอ้างอิงของ Spark ที่ research tutorial รวบรวมไว้ แต่ละตัวเป็นผลวัดของผู้เขียนคนหนึ่ง บนเครื่องเดียว กับซอฟต์แวร์เวอร์ชันเดียว ให้ลองวัดซ้ำเอง อย่านำไปอ้างเป็นสเปก

| โมเดล / engine (ตาม research tutorial) | ตัวชี้วัด | ค่า | แหล่งที่มา |
|---|---|---|---|
| Nemotron 3 Nano 30B-A3B, vLLM, W4A16 NVFP4 | single-stream decode | 74.75 tok/s | ai-muninn |
| ตัวเดียวกัน, W4A4 NVFP4 | single / รวมที่ c=16 | 58.27 / 786 tok/s | ai-muninn |
| ตัวเดียวกัน, W4A16 | รวม | ~400 tok/s | ai-muninn |
| Nemotron 3 Nano NVFP4, vLLM | single-stream | 65+ tok/s | NVIDIA developer forum |
| nemotron-3-nano:30b, Ollama | tok/s เฉลี่ยในงาน agent | 64.7 | Exxact benchmark |
| nemotron-3-super:120b-a12b, Ollama | tok/s เฉลี่ย; งานที่ผ่าน | 16.4; 17/17 | Exxact benchmark |
| qwen3.5:35b-a3b / qwen3.5:122b-a10b, Ollama | tok/s เฉลี่ย | 48.2 / 20.1 | Exxact benchmark |
| gemma4:26b, Ollama เทียบ vLLM | single-stream | ~64 เทียบ ~30 tok/s | Exxact engines |
| gemma4:26b, vLLM | รวม ที่ 10+ concurrent | >300 tok/s | Exxact engines |
| gemma4:26b, Ollama `OLLAMA_NUM_PARALLEL=4` | รวม | ~122 tok/s | Exxact engines |
| NAT ReAct + 1 tool บน Nemotron 3 Nano FP8 | end-to-end ต่อคำถาม | ~13 s | Classmethod |

วิธีอ่านตัวเลขเหล่านี้ จากแหล่งเดียวกัน:

- **Decode ถูกจำกัดด้วยแบนด์วิดท์** ตาม ai-muninn เพดาน single-stream ของ MoE ที่ active 3B บน Spark อยู่แถว 80 ต้น ๆ tok/s ไม่ว่า engine จะมีลูกเล่นอะไร
- **ความเร็วที่รู้สึกว่าใช้ได้** ตาม Exxact agent ที่เร็วกว่า ~20 tok/s รู้สึกว่าใช้งานได้ และ 40+ รู้สึกว่าตอบไว
- **engine ไหนเหมาะกับใคร** ตามการเปรียบเทียบ engine ของ Exxact Ollama ชนะเรื่อง latency ของผู้ใช้คนเดียว ส่วน vLLM ชนะเรื่อง throughput เมื่อมีผู้ใช้หลายคน และโมเดล Nemotron แบบ Mamba-hybrid ไม่ได้ประโยชน์จาก parallelism ของ Ollama
- **เฝ้าดูหน่วยความจำ** Exxact แนะนำให้มี memory watchdog เพราะแรงกดดันของ unified memory ทำให้ Spark ฮาร์ดรีเซ็ตได้

✓ Checkpoint: สำหรับแต่ละชั้นในสี่ชั้น คุณบอกได้ว่าใช้เครื่องมืออะไร และตัวเลขตัวไหนที่จะใส่ในรายงาน

## 2 · L5.1 — Engine: เพดาน, single stream และการ sweep concurrency

**เริ่มจากการคำนวณ** ตอนที่โมเดล decode หนึ่ง stream ทุก token ใหม่ต้องอ่าน weight ที่ *active* ทุกตัวจากหน่วยความจำหนึ่งรอบ ความเร็ว single-stream สูงสุดที่เป็นไปได้จึงเท่ากับแบนด์วิดท์หารด้วยจำนวนไบต์ของ weight ที่ active ต่อ token Nemotron 3 Nano 30B-A3B ใช้พารามิเตอร์ราว 3B ต่อ token และ Spark มี 273 GB/s:

| รูปแบบ | Weight ที่ active ต่อ token | เพดานแบบคิดง่าย |
|---|---|---|
| BF16 | 6.0 GB | 46 tok/s |
| FP8 | 3.0 GB | 91 tok/s |
| NVFP4 (~4.5 บิต รวม scale) | ~1.7 GB | 162 tok/s |

นี่คือขอบบน และ engine จริงจะได้ต่ำกว่านี้ เพราะตอน decode ยังต้องอ่าน KV cache และชั้นที่ไม่ใช่ expert ด้วย และไม่มีชิปตัวไหนวิ่งได้เต็มแบนด์วิดท์ นั่นคือเหตุผลที่เพดานในทางปฏิบัติของ ai-muninn (80 ต้น ๆ) ต่ำกว่าเส้น NVFP4 อยู่มาก การคำนวณนี้บอกสองอย่าง: รูปแบบไหนมีโอกาสถึงเป้าของคุณ และตัวเลขที่วัดได้เข้าใกล้กำแพงแค่ไหนแล้ว

**จากนั้นวัดบน Spark** ใช้ benchmark ที่มากับ vLLM โดยใช้ prompt สุ่มยาว 512 token และคำตอบยาว 256 token รัน single stream ก่อน แล้วค่อย sweep concurrency:

```bash
# on: spark
docker exec vllm-nat vllm bench serve --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --backend openai-chat --endpoint /v1/chat/completions --host 127.0.0.1 --port 8000 \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --num-prompts 32 --max-concurrency 1
for c in 2 4 8 16; do docker exec vllm-nat vllm bench serve --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --backend openai-chat --endpoint /v1/chat/completions --host 127.0.0.1 --port 8000 \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --num-prompts $((c*8)) --max-concurrency $c; done
```

**Expected output** (EXAMPLE — รูปแบบตัวอย่าง ไม่ใช่ผลวัด)

```
============ Serving Benchmark Result ============
Successful requests:                     32
Maximum request concurrency:             1
Benchmark duration (s):                  …
Output token throughput (tok/s):         …
Mean TTFT (ms):                          …
Mean TPOT (ms):                          …
==================================================
```

บันทึกสามค่าต่อระดับ concurrency: **TTFT**, **output tok/s ต่อ request** และ output tok/s **รวม** จากนั้นรัน prompt ชุดเดียวกันบน Ollama เพื่อให้มีทั้งสอง engine เทียบกัน `ollama run nemotron-3-nano:30b --verbose` จะพิมพ์เวลาของมันเองหลังทุกคำตอบ ถ้าอยากได้ชุดงาน agent สำเร็จรูป harness ของ Exxact เป็น open source (ลิงก์อยู่ใน Sources)

เช็กเร็ว ๆ หนึ่ง request จาก runner (มันรายงาน TTFT และ tok/s และบอกว่ารันที่ไหน):

```spark
{"target": "vllm", "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", "messages": [{"role": "user", "content": "In four short bullet points, explain why a hotel chiller plant's kW/RT gets worse at low cooling load."}], "max_tokens": 256}
```

**บนแล็ปท็อป ใช้วิธีเดียวกัน** Lab 06-1 พิมพ์ตารางเพดานและตารางอ้างอิง แสดงคำสั่งของ Spark แล้วรัน sweep เล็ก ๆ กับ Ollama จริง: gemma3:4b วอร์มอัปหนึ่งครั้ง แล้วรันที่ concurrency 1, 2 และ 4 ด้วย prompt เดียวกัน (รวม 9 call)

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_1_engine_sweep.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ส่วน sweep ของ stand-in)

```
◆ c=1: 2 requests · 192 tokens in 3.0s · TTFT p50 57 ms · 75.8 tok/s per request · 63.9 tok/s aggregate
◆ c=2: 2 requests · 192 tokens in 4.1s · TTFT p50 1120 ms · 48.9 tok/s per request · 47.1 tok/s aggregate
◆ c=4: 4 requests · 384 tokens in 8.4s · TTFT p50 3188 ms · 47.5 tok/s per request · 45.9 tok/s aggregate
│ concurrency  requests  TTFT p50  tok/s per request  tok/s aggregate                      
│ ───────────  ────────  ────────  ─────────────────  ───────────────  ────────────────────
│ 1            2         57 ms     75.8               63.9             ████████████████████
│ 2            2         1120 ms   48.9               47.1             ███████████████░░░░░
│ 4            4         3188 ms   47.5               45.9             ██████████████░░░░░░
◆ Aggregate changed only ×0.7: this Ollama is queueing requests (OLLAMA_NUM_PARALLEL) or other labs are using it. The research tutorial cites Exxact: Ollama wins single-user latency, vLLM wins multi-user throughput.
```

ให้อ่านรูปทรง ไม่ใช่ขนาด บน vLLM คุณควรเห็น tok/s รวมเพิ่มขึ้นตาม concurrency ขณะที่แต่ละ request ช้าลง เพราะการ batch แบ่งแบนด์วิดท์กันใช้ ถ้า tok/s รวมไม่ขยับ อย่างที่อาจเกิดกับ Ollama บนแล็ปท็อป แปลว่าเซิร์ฟเวอร์กำลังเข้าคิว request ทีละตัว (`OLLAMA_NUM_PARALLEL`) หรือมีแล็บอื่นใช้อยู่

✓ Checkpoint: คุณอธิบายได้ว่าทำไมเพดาน NVFP4 (162 tok/s) สูงกว่า 74.75 tok/s ที่ ai-muninn วัดได้มาก และเส้น tok/s รวมที่แบนราบหมายถึงอะไร

## 3 · L5.2 — Workflow และคุณภาพ: `nat eval` พร้อม profiler

benchmark ของ engine ไม่ได้บอกอะไรเกี่ยวกับ *หนึ่งเทิร์นของ agent* คำถามหนึ่งข้อของ Alto Ops ใช้ LLM สองครั้ง (เลือก tool แล้วเขียนคำตอบ) รัน tool หนึ่งครั้ง และจ่ายทุก token ของทั้งสอง prompt `nat eval` จะรัน dataset ผ่าน workflow และเมื่อตั้ง `profiler:` ไว้ มันจะบันทึกทุก event ของ LLM และ tool

**Dataset** แต่ละบรรทัดของ JSONL มี `question` (อินพุต) และ `answer` (คำตอบอ้างอิง) ถ้าชื่อคอลัมน์ต่างไป ใช้บล็อก `structure` ใน config ของ dataset จับคู่ให้ Lab 06-2 คำนวณคำตอบอ้างอิงจาก `chiller_plant.csv` ด้วยสูตรเดียวกับ tool `chiller_kpi` จึงไม่มีใครต้องพิมพ์ค่า KPI เอง:

| id | คำถาม | คำตอบ (จาก CSV) |
|---|---|---|
| 1 | What is the average plant kW/RT over the last 6 hours? | 0.901 |
| 2 | Is the chiller plant efficiency in ALARM or OK over the last 6 hours? | ALARM |
| 3 | What is the average plant kW/RT over the last 24 hours? | 0.703 |
| 4 | What was the average cooling load in RT over the last 24 hours? | 629.2 |

research tutorial ขอให้มี 20 คำถาม แต่แล็ปท็อปที่ใช้ร่วมกันได้แค่ 4

**Config** บน Spark ใช้ `eval_config.yml` ของ research tutorial คือ workflow ของ Alto Ops จาก Part 3 บวกบล็อก `eval:` นี้ สำเนาในคอร์สคือ `configs/eval_config.spark.yml`

```yaml
eval:
  general:
    output_dir: ./.tmp/eval/alto_ops/
    max_concurrency: 4
    dataset:
      _type: jsonl
      file_path: ./data/alto_ops_eval.jsonl
    profiler:
      token_uniqueness_forecast: true
      workflow_runtime_forecast: true
      compute_llm_metrics: true
      csv_exclude_io_text: true
      prompt_caching_prefixes:
        enable: true
        min_frequency: 0.5
      bottleneck_analysis:
        enable_nested_stack: true
      concurrency_spike_analysis:
        enable: true
        spike_threshold: 7
  evaluators:
    accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: local_vllm
    trajectory:
      _type: trajectory
      llm_name: local_vllm
```

สำเนาสำหรับแล็ปท็อป `configs/eval_config.yml` ใช้บล็อก profiler เดียวกัน แต่เปลี่ยนสามอย่าง: ใช้ 4 คำถาม, `max_concurrency: 2` และใช้ evaluator ที่คำนวณจาก trace (`avg_llm_latency`, `avg_num_llm_calls`, `avg_workflow_runtime`, `avg_tokens_per_llm_end`) ซึ่งไม่ต้องใช้ LLM เป็นกรรมการ

> ⚠ **NAT 1.9.0: `nat validate` ไม่จับ key ของ profiler ที่สะกดผิด** model ของ profiler ไม่สนใจ field ที่ไม่รู้จัก ดังนั้น `token_uniqueness_forecst: true` จะผ่าน validate ได้ ✓ แล้วไม่ทำอะไรเลย Lab 06-2 จึงตรวจแต่ละ key กับ `ProfilerConfig.model_fields` แทน ผลคือ key ทั้งเก้าตัวที่ research tutorial ใช้มีอยู่จริงใน 1.9.0

**รัน:**

```bash
# on: laptop
week26/.venv-nat/bin/nat eval --config_file week26/06_benchmarking/configs/eval_config.yml
```

```bash
# on: spark
cd ~/alto_ops && nat eval --config_file eval_config.yml
nat eval --config_file eval_config.yml --override eval.general.max_concurrency 1   # serial baseline
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ lab 06-2 step 3)

```
=== EVALUATION SUMMARY ===
Workflow Status: COMPLETED (workflow_output.json)
Total Runtime: 74.75s
Workflow Runtime (p95): 39.76s
LLM Latency (p95): 27.35s

Per evaluator results:
| Evaluator        |   Avg Score | Output File                  |
|------------------|-------------|------------------------------|
| llm_latency      |       17.61 | llm_latency_output.json      |
| llm_calls        |        2    | llm_calls_output.json        |
| workflow_runtime |       35.41 | workflow_runtime_output.json |
| tokens_per_call  |      615.75 | tokens_per_call_output.json  |
```

**อะไรบ้างที่อยู่ในโฟลเดอร์ output** research tutorial ระบุไฟล์เหล่านี้:

**Expected output** (REFERENCE — ยกมาจาก research tutorial, Lab 5.2)

```
#  workflow_output.json  accuracy_output.json  trajectory_accuracy_output.json  config_effective.yml
#  all_requests_profiler_traces.json  inference_optimization.json  standardized_data_all.csv  workflow_profiling_report.txt
```

NAT 1.9.0 บนแล็ปท็อปเครื่องนี้เขียนไฟล์ของ profiler และ config ครบทั้งหกไฟล์ ส่วนไฟล์ของกรรมการใช้กติกาต่างออกไป: evaluator แต่ละตัวเขียน `<evaluator key>_output.json` ดังนั้น key `trajectory:` ของ tutorial จะเขียน `trajectory_output.json` ไม่ใช่ `trajectory_accuracy_output.json` นอกจากนี้ 1.9.0 ยังเขียน `config_original.yml`, `config_metadata.json`, `workflow_profiling_metrics.json`, `gantt_chart.png` และไฟล์ของ evaluator แบบ trace ตัวละหนึ่งไฟล์

**อ่านตัวเลข** `inference_optimization.json` เก็บ p90/p95/p99 และช่วงความเชื่อมั่น (confidence interval) ของ workflow runtime และ LLM latency ส่วน `standardized_data_all.csv` มีหนึ่งแถวต่อหนึ่ง event จึงให้ p50 และ token ต่อ call ได้ และ `workflow_profiling_report.txt` คือโครงสร้างต้นไม้ของคอขวด

**Expected output** (บันทึกจาก Mac เครื่องนี้ lab 06-2 step 5)

```
│ metric                p50   p90   p95   note           
│ ────────────────────  ────  ────  ────  ───────────────
│ workflow runtime (s)  36.7  39.6  39.8  n=4 · mean 35.4
│ LLM latency (s)       —     26.7  27.4  n=8 · mean 17.6
◆ 95% confidence interval of the MEAN runtime: 30.9–39.9 s. With 4 rows it is wide: that is the noise any later comparison (lab 06-4) has to beat.
│ standardized_data_all.csv · LLM_END rows  count  mean  p50  max
│ ────────────────────────────────────────  ─────  ────  ───  ───
│ prompt_tokens                             8      409   409  450
│ completion_tokens                         8      207   217  305
◆ 8 LLM calls for 4 questions = 2.0 per agent turn (tool call → tool → final answer). A ReAct loop or retries would show up here first.
✓ id 1: expected '0.901' · 'The average plant kW/RT over the last 6 hours is **0.901 kW per RT**. '
✓ id 2: expected 'ALARM' · 'The chiller plant efficiency is currently in **ALARM** status over the last 6 hours. This '
✓ id 3: expected '0.703' · 'The average plant kW/RT over the last 24 hours is **0.703** (status: OK). This indicates t'
✓ id 4: expected '629.2' · 'The average cooling load over the last 24 hours was **629.2 RT** (refrigeration tons). Thi'
◆ no-judge quality check (the reference string appears in the answer): 4/4. It is cheap and exact for numeric KPIs. For free-text answers use the LLM judges (--judge).
```

one-liner ของ pandas ใน research tutorial ใช้ได้บน 1.9.0 แต่ต้องกรองเอาเฉพาะแถว `LLM_END` ก่อน เพราะ CSV มีแถว `LLM_START` ที่ token เป็นศูนย์ปนอยู่ และมันทำให้ค่าเฉลี่ยลดลงครึ่งหนึ่ง:

```python
import pandas as pd
df = pd.read_csv("week26/06_benchmarking/.runs/eval/alto_ops/standardized_data_all.csv")
ends = df[df.event_type == "LLM_END"]
print(ends.groupby("llm_name")[["prompt_tokens", "completion_tokens"]].describe())
```

**คุณภาพ** สำหรับ KPI ที่เป็นตัวเลข การตรวจแบบตรงตัวทั้งถูกและซื่อตรง: สตริงคำตอบอ้างอิงปรากฏอยู่ในคำตอบหรือไม่ ซึ่งได้ 4/4 ข้างบน สำหรับคำตอบที่เป็นข้อความอิสระ คุณต้องมีกรรมการ `lab06_2 --judge` ให้คะแนนหนึ่งแถวด้วย evaluator `ragas` AnswerAccuracy และ `trajectory` ของ research tutorial โดยใช้โมเดลบนแล็ปท็อปเป็นกรรมการตัดสินตัวเอง:

**Expected output** (บันทึกจาก Mac เครื่องนี้ `--judge`)

```
=== EVALUATION SUMMARY ===
Workflow Status: COMPLETED (workflow_output.json)
Total Runtime: 15.99s
Workflow Runtime (p95): 15.99s
LLM Latency (p95): 8.34s

Per evaluator results:
| Evaluator   |   Avg Score | Output File            |
|-------------|-------------|------------------------|
| trajectory  |         1   | trajectory_output.json |
| accuracy    |         0.5 | accuracy_output.json   |
✓ accuracy_output.json: average_score 0.5
✓ trajectory_output.json: average_score 1.0
```

agent ตอบแถวที่ 1 ได้ตรงเป๊ะ (0.901) แต่กรรมการยังให้ 0.5 นี่คือคำเตือนของ research tutorial ที่เกิดขึ้นจริง: การให้โมเดลตัดสินตัวเองใช้ได้แค่เป็น smoke test และมีอคติ ถ้าจะทำรายงานจริง ให้ใช้กรรมการที่เก่งกว่า (Super หรือ Ultra ผ่าน NVIDIA endpoints หรือ Spark เครื่องที่สอง)

✓ Checkpoint: จาก eval ครั้งเดียว คุณบอกได้ทั้ง runtime p50 และ p95, จำนวน LLM call ต่อเทิร์น, prompt token เฉลี่ยต่อ call และคะแนนคุณภาพ และบอกได้ว่าแต่ละค่ามาจากไฟล์ไหน

## 4 · L5.3 — Sizing: ต้องใช้ Spark กี่เครื่องสำหรับเครือโรงแรม?

`nat sizing calc` ตอบคำถามสำหรับใบเสนอราคา: *ต้องใช้ GPU กี่ตัว สำหรับผู้ใช้ N คน ที่ latency เป้าหมาย?* มันรัน eval ของคุณหนึ่งครั้งต่อหนึ่งค่า concurrency และบันทึก p95 ของ LLM latency และ p95 ของ workflow runtime จากนั้นลากเส้นตรง `p95 = slope × concurrency + intercept` แล้วแก้สมการหาค่าตามเป้าของคุณ:

- concurrency ที่ GPU ทดสอบหนึ่งตัวรับได้: **c\* = (target − intercept) / slope**
- จำนวน GPU ที่ต้องใช้: **target users / c\* × test GPU count**

ให้นับ Spark หนึ่งเครื่องเป็น GPU หนึ่งตัว บน Spark ใช้ค่าของ research tutorial (ผู้ใช้ 40 คนที่เป็น GM และวิศวกร, p95 15 วินาที):

```bash
# on: spark
export CONFIG_FILE=eval_config.yml CALC_OUTPUT_DIR=./.tmp/sizing/alto_ops
nat sizing calc --config_file $CONFIG_FILE --calc_output_dir $CALC_OUTPUT_DIR \
  --concurrencies 1,2,3,4,6,8,12,16,24,32 --num_passes 2 \
  --test_gpu_count 1 --target_workflow_runtime 15 --target_users 40
# later, re-fit without re-running:
nat sizing calc --offline_mode --calc_output_dir $CALC_OUTPUT_DIR --test_gpu_count 1 --target_workflow_runtime 10 --target_users 100
```

กติกาสามข้อจากเอกสารของ sizing calculator:

| กติกา | เหตุผล |
|---|---|
| ใช้ค่า concurrency **สิบค่าขึ้นไป** | การลากเส้นตรงต้องมีจุดพอให้ผลน่าเชื่อถือ |
| แยกโฟลเดอร์ output ของ calculator **ออกจาก** โฟลเดอร์ output ของ eval | มันเขียน job แยกตาม concurrency และ `--offline_mode` จะกลับมาอ่านโฟลเดอร์นั้นซ้ำ |
| ตัวเลข GPU เป็นค่า **คร่าว ๆ — ไม่ใช่สำหรับ production** | มันสมมติว่าขยายตัวแบบเชิงเส้น ใช้ได้สำหรับใบเสนอราคาแรก |

บนแล็ปท็อป lab 06-3 รัน calculator จริงแบบเล็ก ๆ: concurrency 1 และ 2 หนึ่ง pass (agent รัน 3 ครั้ง) ด้วย `configs/sizing_config.yml` ของมันเอง เพื่อให้ output ของ eval แยกกัน จากนั้นคำนวณซ้ำด้วยมือ แล้ว re-fit แบบ offline

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_3_sizing.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
Targets: LLM Latency ≤ 0.0s, Workflow Runtime ≤ 60.0s, Users = 40
Test parameters: GPUs = 1
Per concurrency results:
|   Concurrency |   p95 LLM Latency |   p95 WF Runtime |   Total Runtime |   GPUs (WF Runtime, Rough) |
|---------------|-------------------|------------------|-----------------|----------------------------|
|             1 |           6.33829 |          10.8256 |         10.8256 |                    7.21707 |
|             2 |          16.1058  |          26.8455 |         27.1506 |                    8.94851 |

=== GPU ESTIMATES ===
Estimated GPU count (Workflow Runtime): 9.8
✓ wrote week26/06_benchmarking/.runs/sizing/alto_ops/online/job_1790740665/ (calc_runner_output.json + two PNG plots)
│ concurrency  measured p95 runtime  line   
│ ───────────  ────────────────────  ───────
│ 1            10.83 s               10.83 s
│ 2            26.85 s               26.85 s
│                                      by hand  nat sizing calc
│ ───────────────────────────────────  ───────  ───────────────
│ slope (s per extra concurrent user)  16.020   16.020         
│ intercept (s)                        -5.194   -5.194         
│ R²                                   —        1.000          
◆ concurrency one GPU sustains at ≤ 60 s: (60 − -5.19) / 16.020 = 4.07
◆ GPUs for 40 users: 40 / 4.07 × 1 test GPU = 9.83  ·  nat sizing calc: 9.83
✓ the calculator's number is this line, nothing more
⚠ R² = 1.000 from 2 points means nothing: two points always make a perfect line. The sizing docs recommend ten or more concurrency values for a robust fit.
```

เส้นที่คำนวณด้วยมือตรงกับ calculator เป๊ะ จึงไม่มีเวทมนตร์อะไรในนั้น แต่ให้ดูที่ R² = 1.000 ข้อมูลสองจุดลากเป็นเส้นตรงได้สมบูรณ์แบบเสมอ "เส้นที่ fit สมบูรณ์" นี้จึงเป็นบทเรียน มันพิสูจน์อะไรไม่ได้ เมื่อใช้ `--num_passes 1` NAT 1.9.0 จะตัด dataset ให้เหลือ concurrency × passes แถว (1 แถวที่ c=1, 2 แถวที่ c=2) แล็ปท็อปจึงรันได้ในขนาดเล็ก

✓ Checkpoint: ถ้าได้ slope, intercept, เป้าหมาย และจำนวนผู้ใช้ คุณคำนวณจำนวน GPU ด้วยมือได้ และบอกได้ว่าทำไม R² = 1 ของแล็ปท็อปไม่มีความหมาย

## 5 · L5.4 — ภาษีของ sandbox

OpenShell ไม่ได้มาฟรี ทุก model call จากใน sandbox วิ่งไปที่ `https://inference.local` supervisor ดักมันไว้ policy engine ตรวจมัน และมันต้องข้ามอีกหนึ่ง hop ผ่าน veth pair ก่อนที่ gateway จะส่งต่อ วิธีวัดเป็นตัวเลข:

1. รัน `nat eval` ที่ **เหมือนกันทุกอย่าง** (ก) บน host ของ Spark ชี้ไปที่ `http://localhost:8000/v1` และ (ข) ใน OpenShell sandbox จาก Lab 3.8 ชี้ไปที่ `https://inference.local/v1`
2. รันทั้งสองขาที่ `max_concurrency` 1 และ 4
3. เทียบ p95 ของ workflow runtime จาก `inference_optimization.json`
4. เผยแพร่ส่วนต่างพร้อม `openshell --version`

ไม่มีแหล่งไหนให้ตัวเลข overhead อย่างเป็นทางการ ผลวัดของคุณ **คือ** ตัวอ้างอิง จึงต้องแนบ noise และเวอร์ชันไว้ข้าง ๆ เสมอ

```bash
# on: spark
openshell --version
cd ~/alto_ops && for c in 1 4; do nat eval --config_file eval_config.yml \
  --override eval.general.max_concurrency $c --override eval.general.output_dir ./.tmp/tax/host_c$c/; done
openshell sandbox upload alto-ops ./eval_config.sandbox.yml /sandbox/eval_config.yml
openshell sandbox exec -n alto-ops --workdir /sandbox -- nat eval --config_file /sandbox/eval_config.yml \
  --override eval.general.max_concurrency 1 --override eval.general.output_dir /sandbox/.tmp/tax/sandbox_c1/
openshell sandbox download alto-ops /sandbox/.tmp/tax/sandbox_c1 ./.tmp/tax/sandbox_c1
```

`configs/eval_config.sandbox.yml` ต่างจาก config ของ host แค่เส้นทางไปหาโมเดล (`base_url: https://inference.local/v1`) และ path ของไฟล์ใต้ `/sandbox` การอัปโหลดเข้า sandbox ถือเป็นการเปลี่ยนแปลง แล็บจึงรันบรรทัดนั้นผ่าน `change()` ซึ่งจะรันก็ต่อเมื่อเปิด 🔓 Allow changes Lab 06-4 ยังตรวจการ parse ของคำสั่ง `openshell sandbox` ทั้งสามด้วย CLI บนแล็ปท็อป ซึ่งไม่ได้ต่อกับ gateway ใด

**บนแล็ปท็อป ใช้ตัวแทนของเส้นทางที่สอง** Mac ไม่มี sandbox Lab 06-4 จึงรัน eval สองคำถามชุดเดียวกันสองครั้ง: (ก) ตรงไปที่ Ollama และ (ข) ผ่าน reverse proxy เล็ก ๆ ที่เขียนด้วย Python พร้อม allow-list (`benchkit.HopProxy`) proxy นี้ปฏิเสธทุกอย่างยกเว้น `POST /v1/chat/completions` และ `GET /v1/models` และจับเวลาตัวเอง มัน **ไม่ใช่** OpenShell: ไม่มีการดัก TLS ไม่มี Landlock และไม่มี network namespace มันแค่ให้วิธีวัดมีเส้นทางที่สองไว้วัด

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_4_sandbox_tax.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ step 2–3)

```
◆ Workflow Runtime (p95): 13.03s
◆ hop proxy on 127.0.0.1:8090 → http://localhost:11434/v1 (allow: POST /v1/chat/completions, GET /v1/models; deny the rest)
$ nat eval --config_file week26/06_benchmarking/configs/eval_config.yml --override eval.general.max_concurrency 1 --override eval.general.output_dir week26/06_benchmarking/.runs/tax/hop_c1/ --override eval.general.dataset.file_path week26/06_benchmarking/.runs/tax_rows.jsonl --override llms.local_llm.base_url http://127.0.0.1:8090/v1   [this laptop]
◆ Workflow Runtime (p95): 23.77s
✓ GET /api/pull through the hop → 403 (denied, like an unlisted endpoint)
■ stopped hop proxy on :8090
◆ the hop itself: 4 allowed · 1 denied · its own time per LLM call ≈ 4.12 ms (max 12.23 ms), next to ≈ 9.3 s upstream

▣ STEP 3 · the delta calculator over the two result directories
│ path                                  p50 s  p95 s  mean s  95% CI of mean  n
│ ────────────────────────────────────  ─────  ─────  ──────  ──────────────  ─
│ A · direct (host stand-in)            11.69  13.03  11.69   9.62–13.76      2
│ B · via hop proxy (sandbox stand-in)  19.41  23.77  19.41   12.70–26.13     2
◆ Δ p95 = +10.74 s (+82.5 %)
⚠ n = 2 per leg: too few rows for a confidence interval to mean much. Treat the verdict below as a demonstration of the check, not as evidence.
⚠ the two confidence intervals overlap: this delta is WITHIN NOISE. Do not publish it as a tax. Add rows and repetitions (`nat eval --reps`) until the intervals separate, or report 'not measurable at n = …'.
◆ The one trustworthy laptop number is the hop's own time: ≈ 4.1 ms per LLM call, measured inside the proxy. Run-to-run noise on this shared Ollama is seconds. How big the tax is on a Spark, no source says, so measure it with enough rows and repetitions to see it above the noise.
```

ตัวคำนวณส่วนต่างตัดสินแบบนี้ ถ้าขา B ออกมา *เร็วกว่า* นั่นคือ drift (มีโหลดอื่น หรือ cache อุ่นอยู่) ไม่ใช่ภาษี ถ้าช่วงความเชื่อมั่น 95% ของค่าเฉลี่ยทั้งสองขาซ้อนทับกัน ส่วนต่างนั้นอยู่ในระดับ noise ให้รายงานว่า "วัดไม่ได้ที่ n = …" และอย่าเผยแพร่ตัวเลข ส่วนต่างจะนับว่าจริงก็ต่อเมื่อช่วงทั้งสองแยกออกจากกัน ตัวเลขเดียวบนแล็ปท็อปที่เชื่อได้คือเวลาของ proxy เอง ไม่กี่มิลลิวินาทีต่อ call เทียบกับ noise ระหว่างรอบที่เป็นวินาที นี่คือเหตุผลที่การวัดภาษีของ sandbox จริงต้องใช้หลายแถวและทำซ้ำหลายรอบ (`nat eval --reps`)

✓ Checkpoint: คุณบอกได้ครบห้าอย่างที่ต้องเผยแพร่พร้อมตัวเลขภาษีของ sandbox (เวอร์ชัน OpenShell, เวอร์ชัน NAT และโมเดล, จำนวนแถว/รอบ/concurrency, p95 ทั้งสองขาและ Δ, ช่วงความเชื่อมั่นของทั้งสองขา)

## 6 · L5.5 — Benchmark ระดับ harness และรายงานสี่ชั้น

claw ของ OpenClaw และ Hermes ไม่มี NAT profiler จึงต้องจับเวลาผ่าน API ที่มันเปิดไว้ สำหรับ Hermes คือ API ที่เข้ากันได้กับ OpenAI บนพอร์ต 8642 ส่งงานที่เหมือนกัน 20 ครั้งแล้วหา p50/p95:

```bash
# on: spark
for i in $(seq 1 20); do
  /usr/bin/time -f "%e" curl -s -X POST http://localhost:8642/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"hermes","messages":[{"role":"user","content":"Summarise /sandbox/data/chiller_plant.csv in 3 bullets"}]}' >/dev/null
done 2>&1 | sort -n | awk '{a[NR]=$1} END {print "p50",a[int(NR*0.5)],"p95",a[int(NR*0.95)]}'
```

จับคู่แต่ละรอบกับอีกสองตัวนับ: จำนวน `inspect_for_inference` ใน `openshell term` ให้จำนวน LLM call ต่องาน และ span ของ Phoenix หรือ Langfuse (Module 05) ให้จำนวน token ต่องาน

บนแล็ปท็อป lab 06-5 รันลูปเดียวกันจริง กับ `nat serve` ของ Alto Ops Claw (`/v1/chat/completions`, พอร์ต `free_port(8001)`): งานเหมือนกัน 5 ครั้ง ทีละครั้ง คำนวณ p50/p95 สองวิธี แล้วสร้างรายงานสี่ชั้นจากสรุปที่ lab 06-1 ถึง 06-4 บันทึกไว้ใน `.runs/`

> ⚠ **NAT 1.9.0: `nat serve` ต้องใช้ `greenlet` แต่ extra ของ NAT ไม่ได้ติดตั้งมาให้** front end ของ FastAPI import `sqlalchemy.ext.asyncio` ตอนเริ่มทำงาน และ import นั้นไม่ยอมโหลดถ้าไม่มี greenlet ตอนสร้างคอร์สนี้ `week26/.venv-nat` ยังไม่มี greenlet และ `nat serve` ออกทันที ตอนนี้ venv มี greenlet แล้ว ถ้าของคุณยังไม่มี lab 06-5 จะตรวจเจอ แล้ววาง stub ที่ติดป้ายชัดเจนไว้บน `PYTHONPATH` (`.runs/pyshim/greenlet.py`) พร้อมแจ้งให้ทราบ ซึ่งปลอดภัย เพราะ async job store ที่จะใช้ greenlet จะเริ่มทำงานก็ต่อเมื่อติดตั้ง Dask ไว้เท่านั้น บน Spark ให้ติดตั้ง greenlet ลงใน environment ของ NAT

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_5_harness_bench.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
✓ ready in 2.8s → http://127.0.0.1:8001/docs
→ POST http://127.0.0.1:8001/v1/chat/completions × 5 · one at a time · the same task every time
◆ task 1: 17.80 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 2: 34.58 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 3: 34.20 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 4: 19.71 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 5: 20.41 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
■ stopped nat (pid 13682)

▣ STEP 3 · p50 / p95 — interpolated vs the tutorial's nearest-rank awk line
│ method                              p50 s  p95 s
│ ──────────────────────────────────  ─────  ─────
│ interpolated (benchkit.percentile)  20.41  34.50
│ sort | awk a[int(NR*p)]             19.71  34.20
◆ With n = 5, awk's p95 is just the 4th-fastest run: one slow outlier moves it a lot. The tutorial sends 20 tasks for a reason; on the Spark, send 20 or more.
◆ 1 distinct answer text(s) for 5 identical tasks at temperature 0. Check correctness as well as speed.

▣ STEP 4 · the four layers side by side (all LAPTOP STAND-IN, from .runs/summary_*.json)
│ layer          LAPTOP STAND-IN result                                from      measured        
│ ─────────────  ────────────────────────────────────────────────────  ────────  ────────────────
│ 1 engine       gemma3:4b: 66 tok/s single · 14 aggregate at c=4      lab 06-1  2026-09-30 10:55
│ 2 workflow     p50 36.7 · p95 39.8 s · 2.0 LLM calls/turn            lab 06-2  2026-09-30 10:57
│ 3 quality      4/4 correct (reference string in answer)              lab 06-2  2026-09-30 10:57
│ 4 sandbox tax  Δ p95 +10.7 s (within noise) · hop itself 4.1 ms/ca…  lab 06-4  2026-09-30 10:59
│ harness (API)  p50 20.4 s · p95 34.5 s over 5 tasks                  lab 06-5  2026-09-30 11:01
✓ the report has one line per layer, each with where it came from and when
⚠ LAPTOP STAND-IN: every number above is this Mac, with nemotron-3-nano / gemma3 on a shared Ollama. None of them is comparable with a Spark figure. On the Spark, rerun labs 06-1 to 06-5 live and this table fills with yours.
```

ดูสองแถวของ percentile `awk a[int(NR*p)]` ของ research tutorial คือ percentile แบบ nearest-rank เมื่อมี 5 รอบ "p95" ของมันก็คือรอบที่เร็วเป็นอันดับ 4 เท่านั้น เมื่อมี 20 รอบก็คืออันดับ 19 บน Spark ให้ส่งงาน 20 ครั้งขึ้นไป และบอกด้วยว่าใช้วิธีคำนวณ percentile แบบไหน

✓ Checkpoint: รายงานของคุณมีหนึ่งบรรทัดต่อหนึ่งชั้น แต่ละบรรทัดบอกแล็บที่มาและเวลาที่วัด และไม่มีบรรทัดของแล็ปท็อปวางอยู่ข้างตัวเลขของ Spark

## Labs — run them here

**labs/lab06_1_engine_sweep.py** — เพดานแบนด์วิดท์, benchmark ของ vLLM บน Spark และ sweep ตัวแทนบนแล็ปท็อปของจริงที่ concurrency 1, 2 และ 4

**labs/lab06_2_nat_eval_profiler.py** — `nat eval` จริงพร้อม profiler กับ Alto Ops Claw: dataset จาก CSV, การตรวจ key, ไฟล์ที่ 1.9.0 เขียน, percentile, token และคะแนนคุณภาพ (`--judge` เพิ่มกรรมการที่เป็น LLM)

**labs/lab06_3_sizing.py** — `nat sizing calc` จริงแบบเล็ก ๆ, คำนวณจำนวน GPU ซ้ำด้วยมือ และ re-fit แบบ offline

**labs/lab06_4_sandbox_tax.py** — วิธีวัดภาษีของ sandbox บน Spark และตัวแทนบนแล็ปท็อปที่มีสองเส้นทางไปหาโมเดลเดียวกัน พร้อมคำตัดสินว่าอยู่ในระดับ noise หรือไม่

**labs/lab06_5_harness_bench.py** — ส่งงานเหมือนกันผ่าน API ของ claw (Hermes บน Spark, `nat serve` บนแล็ปท็อป), p50/p95 สองวิธี และรายงานสี่ชั้น

## Try it yourself

`exercises/ex06_bench_math.py` มี TODO สี่ข้อ และตัวตรวจทำงานแบบ offline:

1. เขียน `percentile(xs, p)` แบบ linear interpolation
2. แปลงค่ารวม 786 tok/s ที่ c=16 ของ ai-muninn เป็นความเร็วต่อ session (research tutorial Part 5 แบบฝึกหัดข้อ 4)
3. เขียน `linear_fit(xs, ys)` แบบ least-squares ที่ sizing calculator ใช้
4. เขียน `gpus_needed(target, users, slope, intercept)` และให้มันปฏิเสธเป้าหมายที่ต่ำกว่า intercept

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/exercises/ex06_bench_math.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อเติม TODO ครบ)

```
✓ percentile: p50 of 1..10 = 5.5 · p95 = 9.55 · one value is its own p95
✓ per session: 786 tok/s ÷ 16 sessions ≈ 49 tok/s each (bandwidth is shared)
✓ linear fit: p95 ≈ 1.00 s × concurrency + 12.00 s on the practice points
✓ sizing: ≤ 15 s → c* = 3 per Spark → 40 users need ≈ 13.3 Sparks · a target below the intercept raises
```

จากนั้นลองคิดแบบฝึกหัดสองข้อจาก Part 5 ของ research tutorial:

- Spark ของคุณได้ 64 tok/s แบบ single-stream กับ Nemotron 3 Nano บน Ollama แต่ claw ยังรู้สึกช้า จงบอกสาเหตุสามข้อที่ไม่ใช่ engine และเครื่องมือที่เปิดเผยแต่ละข้อ
- profiler รายงาน spike ของ `concurrency_spike_analysis` ค่า 9 ที่ t=42 s โดยตั้ง `spike_threshold: 7` ไว้ หมายความว่าอะไร และคุณจะเปลี่ยนอะไร

<details><summary>Hint — ทำไมความเร็วต่อ session ถึงลดลง</summary>

throughput รวมถูกจำกัดด้วยแบนด์วิดท์หน่วยความจำ การ batch ทำให้ engine เสิร์ฟได้หลาย stream ต่อการอ่าน weight หนึ่งรอบ ค่า **รวม** จึงเพิ่มขึ้น แต่แต่ละ stream ได้ส่วนแบ่งน้อยลง: 786 ÷ 16 ≈ 49 tok/s ต่อ session ซึ่งยังสูงกว่าเกณฑ์ "40+ รู้สึกว่าตอบไว" ของ Exxact

</details>

<details><summary>Hint — สาเหตุสามข้อที่ไม่ใช่ engine</summary>

(ก) เรียก LLM ไปกลับหลายรอบเกินไปต่อเทิร์น (ลูป ReAct หรือการ retry): ดูจำนวน LLM call ต่อเทิร์น และ `workflow_profiling_report.txt` (ข) prompt ยาว: ดู `prompt_tokens` ใน `standardized_data_all.csv` และเปิด `prompt_caching_prefixes` (ค) tool ช้า หรือโดน policy ปฏิเสธแล้ว retry: ดู span ของ tool ใน Phoenix และบรรทัด `deny` ใน `openshell logs`

</details>

<details><summary>Hint — concurrency spike</summary>

ในจังหวะนั้นมี NAT function ทำงานพร้อมกันเก้าตัว เกิน threshold ที่ 7 และ profiler บอกได้ว่าเป็นตัวไหนบ้าง อาจเป็นเพราะ agent แตกงานเรียก tool มากเกินไป หรือ `max_concurrency` ของ eval สูงเกินกว่าที่ Spark เครื่องเดียวรับได้ ให้ลด concurrency หรือจำกัดการรัน tool แบบขนาน

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด และคุณตอบคำถามทั้งสองข้อได้ในหนึ่งประโยคต่อข้อ

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `nat serve` ออกพร้อมข้อความ "The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed" | front end ของ FastAPI ใน NAT 1.9.0 import `sqlalchemy.ext.asyncio` แต่ environment ของ NAT ไม่มี greenlet | lab 06-5 ใส่ stub ที่ติดป้ายไว้บน `PYTHONPATH` ให้เอง บน Spark ให้ติดตั้ง greenlet ลงใน environment ของ NAT |
| `nat eval --dataset x.jsonl` ล้มพร้อมข้อความ "If using all scalar values, you must pass an index" | `--dataset` อ่านไฟล์ JSON ไม่ใช่ JSONL | ชี้ `eval.general.dataset.file_path` ไปที่ JSONL หรือใช้ `--override eval.general.dataset.file_path x.jsonl` (lab 06-4 ทำแบบนี้) |
| option ของ profiler ดูเหมือนไม่ทำงาน | key สะกดผิด: `nat validate` ไม่สนใจ field ของ profiler ที่ไม่รู้จัก | ตรวจ key กับ `ProfilerConfig.model_fields` (lab 06-2 step 2) |
| tok/s รวมไม่เพิ่มแม้ concurrency เพิ่ม | Ollama กำลังเข้าคิว (`OLLAMA_NUM_PARALLEL`) หรือมีแล็บอื่นใช้ร่วม | เป็นเรื่องปกติบนแล็ปท็อป บน Spark ให้เทียบกับ vLLM ซึ่ง batch ได้ |
| `nat sizing calc` หยุดพร้อมข้อความ "must be greater than the intercept" | runtime เป้าหมายต่ำกว่า intercept ของเส้น หรือ noise ทำให้ slope ติดลบ | เพิ่มค่าเป้าหมาย เพิ่ม concurrency และจำนวน pass แล้ว re-fit ด้วย `--offline_mode` |
| กรรมการ ragas ให้คะแนนคำตอบที่ถูก 0.5 | โมเดลเล็กตัวเดียวกันตัดสินตัวเอง | ใช้กรรมการที่เก่งกว่าสำหรับรายงานจริง และใช้การตรวจแบบตรงตัวกับ KPI ที่เป็นตัวเลข |
| ตัวเลขบนแล็ปท็อปเปลี่ยนมากระหว่างรอบ | Ollama ใช้ร่วมกัน: แล็บอื่นรันพร้อมกัน | นี่คือเหตุผลที่ตัวเลขแล็ปท็อปทุกตัวติดป้าย LAPTOP STAND-IN ให้เทียบเฉพาะรอบที่รันในนาทีเดียวกัน หรือใช้ Spark |
| พอร์ต 8001 หรือ 8090 ถูกใช้อยู่ | server ของแล็บอื่นกำลังรัน | แล็บใช้ `free_port()` และพิมพ์พอร์ตที่ได้ออกมาให้ |

## Next

[Lab 07 — Expert: threat model, hardening, custom blueprints, remote gateways](../07_hardening/TUTORIAL.md): threat model ของ claw, policy สำหรับ production, custom blueprint และ remote gateway รายงาน benchmark ห้าบรรทัดที่คุณสร้างไว้ที่นี่จะกลายเป็น baseline ที่คุณวัดซ้ำหลังการ hardening ทุกครั้ง
