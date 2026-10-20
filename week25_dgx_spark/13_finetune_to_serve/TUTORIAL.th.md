# ▶ Spark Lab 13 — ปิดวงจร: ประเมิน merge serve และ route โมเดลที่คุณ fine-tune

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- วัด **baseline** ที่ fine-tune ต้องเอาชนะให้ได้: โมเดลตั้งต้น (base model) กับ prompt ที่ดี โดยไม่ต้องเทรน
- ให้คะแนน LoRA fine-tune จาก Module 09 ด้วย **กฎเดียวกัน**: JSON ถูกต้อง แผนกถูก ไม่พลาดเหตุฉุกเฉิน และไม่แต่งเวลาขึ้นเอง
- serve fine-tune ได้สามแบบ: **adapter บน vLLM** โดยไม่ต้อง merge, **โมเดลที่ merge แล้ว** บน vLLM หรือ **GGUF บน Ollama**
- วางมันไว้หลัง **alias ของ LiteLLM** ตัวเดียวพร้อม fallback และดูว่าทำไม fallback ที่เงียบ ๆ ถึงซ่อน fine-tune ที่ตายไปแล้วได้
- เขียน **ship gate**: โค้ดที่ตัดสินว่า fine-tune จะได้ขึ้นใช้งานจริงหรือไม่

**Time** ~45 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: baseline, gateway และ gate รันบนแล็ปท็อป)

**Playbook ทางการที่ครอบคลุม:** ไม่มีโดยตรง โมดูลนี้เป็น **เนื้อหาเฉพาะของคอร์ส (course-original)** ที่เชื่อม [LLaMA Factory](https://build.nvidia.com/spark/llama-factory) (Module 09), [vLLM](https://build.nvidia.com/spark/vllm) (Module 05) และ LiteLLM gateway (Module 08) เข้าด้วยกัน flag LoRA ของ vLLM เป็น CLI ของ vLLM เอง ไม่ใช่ของ NVIDIA playbook และเนื้อหาจะบอกไว้ทุกจุดที่มันปรากฏ

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ชุดข้อมูลของ Module 09 ใน repo | `ls week25/09_llama_factory/data/` แสดง `hotel_ops_eval.json` | ข้อความจากแขก 60 ข้อความที่กันไว้ (held-out) ซึ่งโมเดลไม่เคยเห็นตอนเทรน |
| การเทรนของ Module 09 บน Spark เสร็จแล้ว (ไม่บังคับ) | `ls ~/w25/m09/saves/qwen3-4b-hotel/lora/sft` บน Spark | adapter ที่โมดูลนี้จะ serve และให้คะแนน |
| ขั้น predict ของ Module 09 รันแล้ว (ไม่บังคับ) | `ls ~/w25/m09/saves/qwen3-4b-hotel/lora/predict/` บน Spark | คำตอบของ fine-tune ต่อ 60 ข้อความ |
| venv ของ LiteLLM จาก Module 08 | `ls week25/.venv-litellm/bin/litellm` | lab 13-4 ใช้เปิด gateway |
| Ollama บนแล็ปท็อป (ไม่บังคับ) | `curl -s localhost:11434/api/tags` | ตัวแทนสำหรับ baseline เมื่อไม่มี Spark ตอบ |

```bash
# on: laptop
ls week25/09_llama_factory/data/ week25/.venv-litellm/bin/litellm
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
dataset_info.json
hotel_ops.json
hotel_ops_eval.json
week25/.venv-litellm/bin/litellm
```

✓ Checkpoint: มีไฟล์ eval และไบนารีของ LiteLLM อยู่ ถ้าคุณมี Spark ก็ต้องมีโฟลเดอร์ adapter ของ Module 09 ด้วย

## 1 · วงจรทั้งหมดในภาพเดียว

การเทรนโมเดลเป็นแค่ช่วงกลางของงาน ไม่ใช่จุดจบ fine-tune จะได้ที่ยืนใน production ก็ต่อเมื่อมันเอาชนะทางเลือกที่ถูกกว่าได้ บนชุดทดสอบที่มันไม่เคยเห็น ด้วยกฎที่โค้ดของคุณตรวจได้

```text
 Module 09                     this module
┌──────────┐  adapter  ┌──────────────────────┐   ┌─────────────────────┐   ┌───────────────────┐
│  train   │ ────────► │ predict + SCORE      │──►│ SERVE               │──►│ ROUTE + GATE      │
│  LoRA    │           │ same 60 messages,    │   │ A adapter on vLLM   │   │ one alias for     │
│  on the  │           │ same rules as the    │   │ B merged on vLLM    │   │ clients, fallback,│
│  Spark   │           │ baseline (lab 13-1/2)│   │ C GGUF on Ollama    │   │ ship / don't ship │
└──────────┘           └──────────────────────┘   └─────────────────────┘   └───────────────────┘
       ▲                                                                              │
       └──────────────────────── retrain when the gate says ✕ ────────────────────────┘
```

งานนี้คือ hotel router จาก Module 09 ข้อความจากแขกเข้าไป JSON หนึ่งบรรทัดออกมา:

```text
{"department": "security", "priority": "urgent", "reply": "Security is on the way …"}
```

`labs/_hotel_eval.py` เก็บกฎการให้คะแนนไว้ และทุกแล็บในโมดูลนี้ import มัน fine-tune และ baseline จึงถูกตัดสินด้วยโค้ดชุดเดียวกันทุกประการ:

| ตัวชี้วัดของ gate | เกณฑ์ที่ต้องผ่านเพื่อ ship | ทำไมตั้งเกณฑ์นี้ |
|---|---|---|
| `json_valid` | ≥ 98% | โค้ด parse คำตอบ JSON ที่ไม่ถูกต้องคือระบบล่ม ไม่ใช่แค่พิมพ์ผิด |
| `department_acc` | ≥ 90% | งานหลักของ router |
| `urgent_recall` | **100%** | การพลาด "มีควันในทางเดิน" ไม่ใช่ความคลาดเคลื่อนจากการปัดเศษ |
| `reply_policy` | ≥ 95% | system prompt ห้ามสัญญาเวลาที่แขกไม่ได้ถาม |

✓ Checkpoint: คุณบอกได้ว่าทำไม `urgent_recall` มีเกณฑ์เข้มกว่า `department_acc`

## 2 · Baseline: lab 13-1

ก่อนจะเชื่อ fine-tune ให้วัดทางเลือกที่ถูกกว่าก่อน: **base model กับ prompt ที่ดี** Lab 13-1 ส่งข้อความ held-out 60 ข้อความ พร้อม system prompt เดียวกับที่ข้อมูลเทรนใช้ ไปยังโมเดลที่ยังไม่ได้ fine-tune มันลอง vLLM บน Spark ที่ serve `Qwen/Qwen3-4B-Instruct-2507` (โมเดลที่ Module 09 fine-tune) ก่อน แล้วจึงลอง Ollama บนแล็ปท็อปของคุณ แล้วจึงเป็น DRY

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab01_baseline.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ไม่มี Spark ให้เข้าถึง baseline จึงเป็น gemma4:12b ในฐานะ LAPTOP STAND-IN ซึ่งเป็นโมเดลคนละตัวและใหญ่กว่า base ของ Spark)

```
▣ STEP 1 · send 60 held-out guest messages with Module 09's system prompt
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
│ 10/60 answered · 17s
…
│ 60/60 answered · 168s
◆ 60 predictions from gemma4:12b (LAPTOP STAND-IN) → week25/13_finetune_to_serve/.runs/predictions_baseline_laptop.jsonl

▣ STEP 2 · score them with the same rules the fine-tune will face
│ gate metric     baseline  needed to ship
│ ──────────────  ────────  ──────────────  ─
│ json_valid      97%       ≥ 98%           ✕
│ department_acc  97%       ≥ 90%           ✓
│ urgent_recall   80%       ≥ 100%          ✕
│ reply_policy    97%       ≥ 95%           ✓
│ priority accuracy 82% · urgent cases in this set: 10

▣ STEP 3 · where does it go wrong?
│ expected security       got None           × 2
│ not valid router JSON: '{"department": "security", "priority": urgent, "reply": "Thank you for alerting us; our se'
│ not valid router JSON: '{"department": "security", "priority": urgent, "reply": "Thank you for alerting us; our se'
═ baseline does not pass the ship gate (LAPTOP STAND-IN). Lab 13-2 scores the fine-tune's predictions with the same rules; lab 13-4 puts them side by side.
```

อ่านผลนี้ให้ละเอียด เพราะมันเป็นแบบที่พบได้ทั่วไป:

- **ความแม่นยำด้านแผนก 97% ดูดีมาก** และถ้าเป็นการเดโมก็คงผ่าน
- **ความผิดพลาดกระจุกตัวอยู่ที่กรณีที่สำคัญ** คำตอบที่ไม่ถูกต้องทั้งสองข้อเป็นเหตุฉุกเฉินด้านความปลอดภัย (security) โมเดลเขียน `"priority": urgent` โดยไม่มีเครื่องหมายคำพูด โค้ดจึง parse ไม่ได้ และรายงานเรื่องควันก็หายไป โมเดลที่ใหญ่กว่ากับ prompt ที่ดียังทำเหตุฉุกเฉินหล่นไป 2 จาก 10
- **ความแม่นยำด้าน priority อยู่ที่ 82% และความผิดพลาดมีภาษาของมัน** โมเดลตั้งคำขอปกติ 9 รายการเป็น urgent และ **ทั้ง 9 รายการเป็นภาษาไทย**: ข้อความภาษาไทยปกติ 9 จาก 10 ข้อความ ("ทีวีในห้อง 527 เปิดไม่ติด") ถูกตั้งเป็น urgent เทียบกับข้อความภาษาอังกฤษ 0 จาก 40 ข้อความ คะแนนเฉลี่ยซ่อนอคติ (bias) ที่ทีมกะดึกซึ่งพูดภาษาไทยจะรู้สึกได้ตั้งแต่คืนแรก ให้แยกผล (slice) ทุกการประเมินตามภาษา (และตามแผนก) ก่อนจะเชื่อถือมัน

นี่คือเหตุผลที่ตรงไปตรงมาสำหรับการ fine-tune โมเดลเล็กบนงานที่แคบ: ความสม่ำเสมอกับฟอร์แมตของคุณ ภาษาของคุณ และกรณีขอบของคุณ มากกว่าความฉลาดทั่วไป ชุดข้อมูลเทรนของ Module 09 ผสมภาษาอังกฤษและภาษาไทยด้วยเหตุผลนี้เอง อีกวิธีแก้คือทำให้ JSON ที่ไม่ถูกต้องเกิดขึ้นไม่ได้เลย structured outputs ของ vLLM (ใส่ JSON schema ในคำขอ) จำกัดการ decode ไว้ โมเดลจึงเขียน `urgent` โดยไม่มีเครื่องหมายคำพูดไม่ได้ ลองทั้งสองวิธี แล้วให้ gate เป็นผู้ตัดสิน

✓ Checkpoint: คุณรัน lab 13-1 แล้ว (บน Spark หรือแล็ปท็อป) และมีไฟล์ `.runs/predictions_baseline_*.jsonl` อยู่ คุณบอกชื่อตัวชี้วัดที่ baseline ไม่ผ่านได้

## 3 · ให้คะแนน fine-tune: lab 13-2

lab 04 ของ Module 09 รันขั้น predict ของ LLaMA Factory บน Spark ไปแล้ว:

```bash
# on: spark
cd ~/w25/m09 && llamafactory-cli train configs/hotel_predict.yaml
```

มันเขียน `generated_predictions.jsonl` โดยมี object `{"prompt", "predict", "label"}` หนึ่งตัวต่อข้อความ held-out หนึ่งข้อความ (เป็นฟอร์แมต output ของ predict ใน LLaMA Factory ส่วน config ใช้ greedy decoding เพื่อไม่ให้คะแนนแกว่งระหว่างการรันแต่ละครั้ง) Lab 13-2 อ่านไฟล์นั้นผ่าน SSH (อ่านอย่างเดียว) ให้คะแนนด้วย gate เดียวกัน และเก็บสำเนาไว้ใน `.runs/` ให้ lab 13-4:

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab02_score_finetune.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY: ไม่มี fine-tune ให้อ่าน แล็บจึงแจ้งไว้ และให้คะแนนไฟล์ baseline แทนเพื่อแสดงรูปแบบรายงาน)

```
▣ STEP 1 · get the fine-tune's predictions
$ test -f ~/w25/m09/saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl && wc -l < … || echo MISSING   [DRY]
◈ DRY: no fine-tune predictions reachable; scoring predictions_baseline_laptop.jsonl instead

▣ STEP 2 · score 60 predictions · BASELINE predictions_baseline_laptop.jsonl (no fine-tune available — shown so you can read the report)
│ gate metric     score  needed to ship
│ ──────────────  ─────  ──────────────  ─
│ json_valid      97%    ≥ 98%           ✕
│ department_acc  97%    ≥ 90%           ✓
│ urgent_recall   80%    ≥ 100%          ✕
│ reply_policy    97%    ≥ 95%           ✓
…
▣ STEP 4 · slice by guest language — an average can hide a bias
│ language  n   department  priority  normal → urgent  urgent recall
│ ────────  ──  ──────────  ────────  ───────────────  ─────────────
│ en        50  96%         96%       0%               80%
│ th        10  100%        10%       90%              100%
═ BASELINE predictions_baseline_laptop.jsonl (no fine-tune available — shown so you can read the report): does not pass the ship gate.
◆ LLaMA Factory's own metrics (BLEU/ROUGE in predict_results.json) measure word overlap with the label. For a router, the gate above (valid JSON, right department, no missed emergency) is what matters.
```

Step 4 คือการแยกผลจากส่วนที่ 2 ที่คำนวณออกมาเป็นตัวเลข: baseline ส่งข้อความภาษาไทยไปถูกแผนกทุกครั้ง แต่ตั้งคำขอภาษาไทยปกติ 9 จาก 10 เป็น urgent สิบข้อความเป็นตัวอย่างที่เล็ก และการรันครั้งเดียวของโมเดลตัวเดียวไม่ใช่คำตัดสินเกี่ยวกับภาษาไทยโดยทั่วไป แต่นี่คือช่องว่างแบบที่ fine-tune บนข้อมูลสองภาษาควรปิดได้ และเป็นสิ่งที่ค่าเฉลี่ยซ่อนไว้พอดี ให้เปรียบเทียบแถวนี้ของ fine-tune

เมื่อมี Spark, step 2 จะแสดงตัวเลขของ fine-tune เอง ไฟล์ที่ให้คะแนนแล้วจากแหล่งใดก็ได้เปรียบเทียบกันได้ด้วย `--file path.jsonl`

> 💡 **ทำไมไม่ใช้ BLEU และ ROUGE ของ LLaMA Factory?** มันนับคำที่ซ้ำกันระหว่างคำตอบกับ label คำตอบที่ส่งเรื่องกลิ่นแก๊สไปให้ *housekeeping* ด้วยภาษาอังกฤษที่สวยงามอาจได้คะแนน ROUGE ดีก็ได้ ให้คะแนนสิ่งที่โค้ดของคุณนำไปใช้จริง

✓ Checkpoint: เมื่อมี Spark คุณมี `.runs/predictions_finetune_spark.jsonl` และตาราง gate ของมัน ถ้าไม่มี คุณอธิบายได้ว่า lab 13-2 จะอ่านอะไรและอ่านจากที่ไหน

## 4 · Serve มัน: adapter, merged หรือ GGUF (lab 13-3)

LoRA fine-tune คือโฟลเดอร์ adapter weights ขนาด ~60 MB ที่วางอยู่บน base model ซึ่งไม่ได้ถูกเปลี่ยนแปลง มีสามวิธีในการ serve มัน และ lab 13-3 พิมพ์ทั้งสามวิธีพร้อม base model และ rank จริงของ adapter คุณ:

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab03_serve_it.py            # plan only
.venv/bin/python week25/13_finetune_to_serve/labs/lab03_serve_it.py --launch   # copy the adapter + start path A
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY config ของ adapter เป็นรูปแบบ EXAMPLE)

```
▣ STEP 1 · read the adapter's own config: which base, which rank?
$ cat ~/w25/m09/saves/qwen3-4b-hotel/lora/sft/adapter_config.json && ls -la … | grep -E 'safetensors|tokenizer'   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
{"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507", "r": 16, "lora_alpha": 32, …}
✓ base = Qwen/Qwen3-4B-Instruct-2507 · rank r = 16 → --max-lora-rank must be ≥ 16

▣ STEP 2 · three ways to serve it

→ A · adapter on vLLM, no merge (one base can carry many adapters)
  docker run -d --name w25-hotel-vllm --gpus all --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/w25/adapters:/adapters" \
    --entrypoint '' vllm/vllm-openai:latest \
    vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --gpu-memory-utilization 0.3 \
      --enable-lora --lora-modules hotel-ft=/adapters/hotel-ft --max-lora-rank 16

→ B · merged model on vLLM (one plain model, no LoRA flags)
  cd ~/w25/m09 && llamafactory-cli export configs/hotel_merge.yaml     # writes ~/w25/m09/saves/qwen3-4b-hotel/merged
  …

→ C · GGUF on Ollama (smallest, simplest, slowest to update)
  python ~/llama.cpp/convert_hf_to_gguf.py ~/w25/m09/saves/qwen3-4b-hotel/merged --outtype q8_0 --outfile ~/w25/hotel-router-q8_0.gguf
  …

▣ STEP 3 · start path A on the Spark (opt-in)
◆ Not launched. Re-run with --launch to copy the adapter to ~/w25/adapters/hotel-ft and start vLLM.
```

> ⚠ **ส่วนที่คอร์สเพิ่มเข้ามา** `--enable-lora`, `--lora-modules` และ `--max-lora-rank` ของเส้นทาง A เป็น flag ของ vLLM เอง ไม่มี NVIDIA playbook ตัวไหนครอบคลุมการ serve LoRA Module 05 ส่วนที่ 7 แสดงวิธียืนยัน flag เหล่านี้สำหรับ vLLM เวอร์ชันของคุณ เส้นทาง B ใช้ merge config ของ LLaMA Factory จาก Module 09 เส้นทาง C ใช้ตัวแปลงของ llama.cpp จาก build ของ Module 04 ใน `~/llama.cpp` ตัวแปลงจะเก็บ chat template ของโมเดลไว้ใน GGUF ถ้า `ollama run hotel-router` ตอบในรูปแบบแปลก ๆ ให้เพิ่มบรรทัด `TEMPLATE` ใน Modelfile (เอกสารของ Ollama แสดงรูปแบบไว้)

| | A · adapter บน vLLM | B · merged บน vLLM | C · GGUF บน Ollama |
|---|---|---|---|
| สิ่งที่คุณส่งมอบ | โฟลเดอร์ adapter ~60 MB | โมเดลเต็ม (~8 GB ที่ bf16 สำหรับ 4B) | ไฟล์ที่ quantize แล้วหนึ่งไฟล์ |
| เทรนใหม่ → ใช้งานจริง | สลับโฟลเดอร์ แล้วรีสตาร์ต | merge ใหม่ แล้วรีสตาร์ต | merge, แปลง, `ollama create` |
| หลาย fine-tune บน base เดียว | ✓ ใส่ `--lora-modules` ได้หลายตัว base อยู่ในหน่วยความจำตัวเดียว | ✕ ตัวละหนึ่งโมเดล | ✕ ตัวละหนึ่งโมเดล |
| ความเร็ว | มี overhead ของ LoRA เล็กน้อยต่อ token | ความเร็วเต็มของ base | ความเร็วของ llama.cpp ใช้หน่วยความจำน้อยกว่า |
| เหมาะกับ | การปรับรอบแล้วรอบเล่า, A/B test, adapter ต่อลูกค้า | เวอร์ชันที่คุณ ship | edge, แล็ปท็อป, ผู้ใช้ Open WebUI |

`--gpu-memory-utilization 0.3` เหลือที่ไว้ให้ engine ตัวที่สองหรือ gateway บน Spark เครื่องเดียวกัน โมเดล 4B ที่ bf16 มี weights ~8 GB และ lab 01 ของ Module 05 คำนวณส่วนที่เหลือให้

✓ Checkpoint: คุณบอกได้ว่าจะใช้เส้นทางไหนระหว่างที่ยังเทรนใหม่อยู่ และจะใช้เส้นทางไหนสำหรับเวอร์ชันที่ ship

## 5 · Route มัน และตั้ง gate (lab 13-4)

client ไม่ควร hard-code `hotel-ft` หรือ hostname ของ Spark เลย Lab 13-4 เขียน config ของ LiteLLM (ด้วยตัวสร้างของ Module 08) ที่มี **alias เดียว** คือ `hotel-router` พร้อมลำดับ fallback:

```text
hotel-router  →  hotel-ft on Spark A's vLLM       (the fine-tune)
      └─ fails → hotel-base on Spark A's vLLM     (same base model, prompt-only)
            └─ fails → gemma4:12b on the laptop   (last resort)
```

มันเปิด gateway บนแล็ปท็อปนี้ ส่งข้อความ held-out ห้าข้อความไปที่ `hotel-router` และอ่าน response header ของ LiteLLM เพื่อแสดงว่าใครเป็นผู้ตอบจริง จากนั้นวางไฟล์ prediction ที่บันทึกไว้เทียบกัน:

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab04_route_and_gate.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: เป็น LiteLLM 1.89 proxy จริง ไม่มี Spark ทุกการเรียกจึง fallback ไปที่แล็ปท็อป)

```
▣ STEP 1 · write the gateway config: hotel-router → hotel-ft → hotel-base → laptop
│ alias         backend model                            api_base
│ ────────────  ───────────────────────────────────────  ─────────────────────────
│ hotel-router  hosted_vllm/hotel-ft                     http://spark-a:8000/v1
│ hotel-base    hosted_vllm/Qwen/Qwen3-4B-Instruct-2507  http://spark-a:8000/v1
│ hotel-laptop  openai/gemma4:12b                        http://localhost:11434/v1
│ fallbacks: hotel-router → hotel-base → hotel-laptop

▣ STEP 2 · start the gateway and send 5 held-out messages to `hotel-router`
$ LITELLM_MASTER_KEY=sk-w25-… week25/.venv-litellm/bin/litellm --config week25/13_finetune_to_serve/.runs/hotel-gateway.yaml --host 127.0.0.1 --port 4000 --telemetry False
◆ gateway up on http://127.0.0.1:4000 after 2.6 s (pid 52797, log 08_litellm_gateway/.runs/litellm-4000.log)
◆ gateway stopped (pid 52797); port 4000 is free again: True
│ guest message                        HTTP  answered by  fallbacks  right dept + JSON
│ ───────────────────────────────────  ────  ───────────  ─────────  ─────────────────
│ Good evening. Could you recommend …  200   gemma4:12b   1          ✓
│ Hello, the bathroom in 1418 wasn't…  200   gemma4:12b   1          ✓
│ Could you recommend a good seafood…  200   gemma4:12b   1          ✓
│ What is the Wi-Fi password? I'm in…  200   gemma4:12b   1          ✓
│ The bathroom in 606 wasn't cleaned…  200   gemma4:12b   1          ✓
◆ Every answer needed a fallback: the fine-tune is not being served right now. Clients noticed nothing — which is exactly why you must LOG `x-litellm-attempted-fallbacks`, or a dead fine-tune goes unnoticed.

▣ STEP 3 · side by side: baseline vs fine-tune, same 60 messages, same gate
│ gate metric     baseline (predictions_baseline_laptop.jsonl)  needed
│ ──────────────  ────────────────────────────────────────────  ──────
│ json_valid      97%                                           ≥ 98%
│ department_acc  97%                                           ≥ 90%
│ urgent_recall   80%                                           ≥ 100%
│ reply_policy    97%                                           ≥ 95%
⚠ no fine-tune predictions: the ship decision needs Module 09's adapter scored on a Spark (lab 13-2).
```

บทเรียนสองข้อจากการรันที่ไม่มีอะไรบน Spark ทำงานอยู่เลย:

1. **Fallback ซ่อนความล้มเหลว** ทุกคำขอได้ HTTP 200 และคำตอบที่ดี แดชบอร์ดที่ดูแค่ status code จะดูสมบูรณ์แบบ ขณะที่ fine-tune ที่คุณจ่ายเงินเทรนไม่ได้ serve อะไรเลย header `x-litellm-attempted-fallbacks` เป็นสัญญาณเดียว จึงต้อง log มันและตั้งการแจ้งเตือน
2. **การตัดสินใจคือตาราง ไม่ใช่ความรู้สึก** เมื่อมี Spark, step 3 จะแสดงคอลัมน์ที่สองสำหรับ fine-tune และพิมพ์ **SHIP** หรือ **DO NOT SHIP** เก็บไฟล์ baseline ไว้: การเทรนใหม่ทุกครั้งในอนาคตจะถูกให้คะแนนเทียบกับมัน

> 💡 การ fallback จาก `hotel-ft` ไป `hotel-base` เป็นการลด **คุณภาพ** ไม่ใช่แค่ลดความพร้อมใช้งาน สำหรับ router ที่เกี่ยวกับความปลอดภัย คุณอาจเลือกให้ล้มเหลวแบบเสียงดัง (HTTP 503 และส่งต่อให้เจ้าหน้าที่ที่เป็นคน) ดีกว่าตอบด้วยโมเดลที่ gate ปฏิเสธไปแล้ว config ของ gateway คือที่ที่คุณทำให้การเลือกนั้นชัดเจน

✓ Checkpoint: คุณรัน lab 13-4 แล้ว เห็น `x-litellm-attempted-fallbacks` ในตาราง และอธิบายได้ว่าทำไมแดชบอร์ดที่เป็น 200 ทั้งหมดไม่ใช่หลักฐานว่า fine-tune ใช้งานอยู่

## Labs — รันแล็บได้ที่นี่

**labs/lab01_baseline.py** — base model แบบใช้แค่ prompt กับข้อความโรงแรม held-out 60 ข้อความ ให้คะแนนด้วย ship gate: ตัวเลขที่ fine-tune ต้องเอาชนะ

**labs/lab02_score_finetune.py** — อ่าน prediction ของ fine-tune จาก LLaMA Factory บน Spark แล้วให้คะแนนด้วยกฎชุดเดียวกันทุกประการ

**labs/lab03_serve_it.py** — อ่าน base และ rank ของ adapter พิมพ์สามวิธีในการ serve และ (เมื่อเลือกเปิดเอง) เริ่ม adapter บน vLLM

**labs/lab04_route_and_gate.py** — alias ของ LiteLLM ตัวเดียวพร้อมลำดับ fallback ใครเป็นผู้ตอบแต่ละคำขอจริง และการตัดสินใจ ship แบบวางเทียบกัน

`labs/_hotel_eval.py` เป็นตัวให้คะแนนที่ใช้ร่วมกัน ไม่ใช่แล็บ capstone (Module 20) ก็ import มันเช่นกัน

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 13 — ship gate** เปิด `week25/13_finetune_to_serve/exercises/ex13_ship_gate.py` มี `TODO` สามจุด:

1. `parse_answer(text)`: ดึง JSON ของ router ออกจาก output จริงของโมเดล (แบบสะอาด, อยู่ใน ```json fences หรืออยู่หลังประโยคพูดคุย) และคืน `None` เมื่อ JSON ไม่ถูกต้อง
2. `promises_time(reply, guest_message)`: ทำเครื่องหมาย "in 10 minutes" และ "within the hour" แต่ไม่ทำกับเวลาจองของแขกที่ถูกพูดทวนกลับ
3. `ship(metrics, baseline)`: เกณฑ์ gate ทั้งสี่ข้อ บวก "ต้องไม่แย่กว่า baseline" โดยคืนรายการกฎที่ไม่ผ่าน

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/exercises/ex13_ship_gate.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด)

```
✓ parse_answer: clean · fenced · with chatter → dict; unquoted value and prose → None
✓ promises_time: 'in 10 minutes' ✓ · 'within the hour' ✓ · echoed 19:00 ✗ · no time ✗
✓ ship: the good run ships · the bad run is blocked by urgent_recall, reply_policy and the baseline

▣ why the bad run is blocked
│ urgent_recall failed (90%)
│ reply_policy failed (90%)
│ beats baseline department_acc failed (95%)

═ A 95% department score still fails: one missed emergency is worse than ten misrouted towel requests.
```

<details><summary>คำใบ้ — ทำไม `{"priority": urgent}` ต้องถูกปฏิเสธ?</summary>

นี่คือสิ่งที่ baseline บนแล็ปท็อปเขียนไว้สำหรับรายงานควันสองรายการใน lab 13-1 พอดี `json.loads` จะ raise error ผู้เรียกใช้ router จึงจะ crash หรือทิ้งข้อความนั้นไป parser แบบผ่อนปรนที่ "แก้" ให้จะซ่อนบั๊กของโมเดลที่คุณต้องเห็นใน gate

</details>

<details><summary>ท้าทายเพิ่ม — structured outputs แทนการ fine-tune</summary>

vLLM รับ JSON schema ในคำขอได้ (`"response_format": {"type": "json_schema", …}`) ซึ่งจำกัดการ decode ให้ได้ JSON ที่ถูกต้อง เพิ่มมันลงในคำขอของ lab 13-1 รันใหม่กับ base model บน Spark แล้วดูว่าตัวชี้วัดไหนของ gate ขยับ มันแก้ความล้มเหลวแบบไหนได้ และแบบไหน (แผนกผิด, พลาด urgent) ที่แก้ได้ด้วยการเทรนหรือ prompt ที่ดีขึ้นเท่านั้น?

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| lab 13-2: `generated_predictions.jsonl does not exist` | รันขั้น predict ของ Module 09 lab 04 ก่อน (`llamafactory-cli train configs/hotel_predict.yaml` จาก `~/w25/m09`) |
| vLLM: `LoRA rank 16 is greater than max_lora_rank 8` | ตั้ง `--max-lora-rank` ให้ไม่น้อยกว่า `r` ของ adapter (lab 13-3 อ่านค่านี้จาก `adapter_config.json`) |
| vLLM แสดง base model แต่ไม่มี `hotel-ft` | path ของ `--lora-modules` อยู่ **ภายใน container** (`/adapters/hotel-ft`) ตรวจการ mount `-v "$HOME/w25/adapters:/adapters"` และ `ls ~/w25/adapters/hotel-ft` |
| `/v1/models` แสดง `hotel-ft` แต่คำตอบดูเหมือน base model | adapter ถูกเทรนด้วย `template: qwen3_nothink` ส่ง system prompt เดียวกับที่ข้อมูลเทรนใช้ และเปรียบเทียบกับ baseline ของ lab 13-1 เพื่อให้แน่ใจ |
| lab 13-4: ทุกแถวแสดง fallback | ยังไม่มีอะไร serve `hotel-ft` ที่ :8000 ของ Spark A รัน lab 13-3 ด้วย `--launch` แล้วตรวจ `curl http://spark-a:8000/v1/models` |
| lab 13-4: `litellm … not found` | สร้าง venv จาก Module 08: `uv venv week25/.venv-litellm && uv pip install -p week25/.venv-litellm/bin/python "litellm[proxy]==1.89.0"` |
| คะแนน baseline แกว่งระหว่างการรัน | ใช้ `temperature: 0` (แล็บทำอยู่แล้ว) และไฟล์ eval ที่คงที่ predict config ของ LLaMA Factory ตั้ง `do_sample: false` ด้วยเหตุผลเดียวกัน |

## Next — บทถัดไป

ไปต่อที่ [Lab 14 — NeMo Agent Toolkit: เอเจนต์บนโมเดลใน Spark ของคุณ](../14_nat_agents/TUTORIAL.md): ให้ tool และลูปการให้เหตุผล (reasoning loop) กับ router โดย serve จาก Spark ของคุณ
