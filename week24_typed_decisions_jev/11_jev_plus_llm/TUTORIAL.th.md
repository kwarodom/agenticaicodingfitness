# ▶ Jev Lab 11 — Jev + LLM ที่คุณเลือก: Jev ตัดสินใจ, LLM เขียน

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิด (typed decisions) ด้วย Jev — Jev ไม่เขียนข้อความยาว ๆ ดังนั้นผลิตภัณฑ์จริงทุกตัวต้องจับคู่ Jev กับ LLM แบบ generative (โมเดลที่เขียนข้อความได้) ในแล็บนี้คุณจะเลือก LLM เอง (Claude, ChatGPT, Gemini, DeepSeek, Kimi, GLM, OpenRouter หรือโมเดลในเครื่องผ่าน Ollama) ต่อไว้ด้านหลัง Jev เปรียบเทียบหลายโมเดลบนงานเดียวกัน และให้ Jev ตรวจสิ่งที่ LLM เขียนก่อนจะถึงมือคน

**พูดง่าย ๆ:** Jev คือเพื่อนร่วมงานที่ติ๊กช่องได้เร็วมาก ("เรื่องแอร์ ด่วน ไม่ได้ขอเงินคืน") ส่วน LLM คือเพื่อนร่วมงานที่เขียนอีเมลตอบ แล็บนี้ให้แต่ละตัวทำเฉพาะสิ่งที่ถนัด แล้วเก็บ "กฎ" ไว้ในโค้ดของคุณเอง

**สิ่งที่คุณจะได้ลงมือทำ**
- ดูว่าเครื่องนี้ใช้ LLM provider ไหนได้บ้าง และเลือกหนึ่งตัวสำหรับทั้งคอร์สจากตัวเลือก **LLM** บนหัวของ Lab Runner
- สร้าง pipeline: Jev จัดเส้นทาง (route) → โค้ดเลือก prompt ที่อนุมัติแล้ว → LLM ร่างคำตอบ
- รันงานเดียวกันกับทุกโมเดลที่มี key ทั้งภาษาอังกฤษและภาษาไทย
- ให้ Jev ตรวจร่างของ LLM (สัญญาว่าจะคืนเงินไหม? แต่งชื่อช่างขึ้นมาเองไหม?) แล้วส่งต่อให้คนตรวจ
- เรียนรู้กับดัก "โมเดลบอกชื่อตัวเองผิด": คำอ้างของโมเดลเกี่ยวกับตัวมันเองไม่ใช่หลักฐาน

**Time** ~45 นาที · **Difficulty** ปานกลาง · **Cost** ≈ $0.0005 ค่า Jev + ค่า token ของ LLM ไม่กี่เซนต์ (live) · $0 (dry)

## 0 · ทำไมต้องใช้สองโมเดล?

Jev เป็นโมเดลแบบ **System One**: ตัดสินใจเร็ว ได้คำตอบแบบมีชนิด (typed) ที่โค้ดนำไปแตกเงื่อนไข (branch) ได้ทันที ส่วน LLM แบบ generative เป็นโมเดลแบบ **System Two**: เขียน อธิบาย และให้เหตุผลเป็นข้อความอิสระ ทั้งสองเก่งคนละเรื่อง และใช้เวลาต่างกันมาก

| งาน | ให้ใครทำ | เหตุผล |
|---|---|---|
| "ทีมไหน? ด่วนแค่ไหน? ขอเงินคืนไหม?" | **Jev** | คำตอบแบบมีชนิด + ความน่าจะเป็น ~0.5 วินาที ราคาเศษเสี้ยวเซนต์ |
| ดึงเลขห้อง ตรวจว่ามีอักษรไทยไหม | **โค้ด** | เป็นข้อเท็จจริงแน่นอน regex ไม่เคยเดา |
| เลือก system prompt | **โค้ด** | ใช้เฉพาะ prompt ที่อนุมัติแล้ว เก็บไว้ในทะเบียน (registry) |
| เขียนคำตอบถึงแขก | **LLM ของคุณ** | เขียนลื่นไหลในภาษาของแขก |
| "ร่างนี้สัญญาคืนเงินไหม? แต่งชื่อขึ้นมาไหม?" | **Jev** | ตรวจผลลัพธ์ของ LLM แบบมีชนิด |
| ส่งจริงไหม? | **คน** | `execute: False` เสมอ |

วัดจริงใน lab 02 บนเครื่องนี้: การตัดสินใจจัดเส้นทางของ Jev เฉลี่ย **660 ms** ส่วนการร่างคำตอบของ Claude เฉลี่ย **2,765 ms** การตัดสินใจแบบมีชนิดคือขั้นที่ถูกและเร็ว การเขียนคือขั้นที่ช้า

```text
guest message ─► Jev: department · urgency · wants refund?     (typed, ~0.5 s)
              ─► code: language · room number · APPROVED prompt  (exact)
              ─► your LLM: reply draft                          (prose, seconds)
              ─► Jev: checks the DRAFT                           (typed, ~0.5 s)
              ─► code: review policy → staff approval | human review   execute: False
```

✓ Checkpoint: คุณบอกได้ว่าขั้นไหนของ pipeline คือ Jev ขั้นไหนคือโค้ด ขั้นไหนคือ LLM และอธิบายได้ว่าทำไมโค้ดเป็นผู้เลือก prompt

## 1 · รู้จักผู้ให้บริการ (providers)

ผู้ให้บริการทุกรายด้านล่างเรียกผ่านคำสั่งเดียวกันคือ `llmkit.generate()` ใน `week24/common/llmkit.py` Claude ใช้ Messages API ของ Anthropic ส่วนรายอื่นใช้ API แบบ `/chat/completions` ที่เข้ากันได้กับ OpenAI ดังนั้นการสลับโมเดลเปลี่ยนแค่คำเดียว

| id | ผู้ให้บริการ | ตัวอย่างโมเดล (ตรวจสด 2026-09-27) | ชื่อตัวแปร key | ขอ key ที่ |
|---|---|---|---|---|
| `claude` | Anthropic | `claude-sonnet-5`, `claude-opus-5-5`, `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `chatgpt` | OpenAI | `gpt-5.4-mini`, `gpt-5.5`, `gpt-5.4-nano` | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| `gemini` | Google | `gemini-3.5-flash`, `gemini-3.1-pro-preview` | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `deepseek` | DeepSeek | `deepseek-flash`, `deepseek-v4-pro` | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| `kimi` | Moonshot AI | `kimi-k3`, `kimi-k2.6` | `KIMI_API_KEY` | [platform.moonshot.ai](https://platform.moonshot.ai/console/api-keys) |
| `glm` | Z.ai / Zhipu | `glm-5.3`, `glm-5.3-flash` | `ZAI_API_KEY` | [z.ai](https://z.ai/manage-apikey/apikey-list) |
| `openrouter` | OpenRouter | `openai/gpt-5.4-mini`, `anthropic/claude-sonnet-5`, … | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai/settings/keys) |
| `ollama` | Ollama (ในเครื่อง) | `gemma4:latest` หรือโมเดลใดก็ได้ที่ pull ไว้ | ไม่ต้องใช้ | [ollama.com](https://ollama.com/download) |

ข้อควรรู้ที่มีผลจริง:

- **Ollama** รันบนเครื่องของคุณเอง: $0 ไม่ต้องมี key ข้อมูลไม่ออกจากเครื่อง แต่ช้าที่สุดในที่นี้ (GPU ของแล็ปท็อป)
- **OpenRouter** ใช้ key เดียวเข้าถึงได้หลายร้อยโมเดล เหมาะกับการลองหลายโมเดลโดยไม่ต้องสมัครหลายบัญชี
- **บัญชีภูมิภาคจีน**: ตั้งค่า `KIMI_BASE_URL=https://api.moonshot.cn/v1` หรือ `GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4`
- **คุณภาพภาษาไทยต่างกันในแต่ละโมเดล** ต้องวัดกับข้อความจริงของคุณเอง (ขั้นที่ 4) อย่าคิดเอาเอง
- **โมเดลที่ "คิด" ก่อนตอบ** (DeepSeek, Kimi, GLM-5.x, GPT-5.x, Gemini) ใช้ token "reasoning" ที่มองไม่เห็นก่อนเขียนคำตอบ ถ้า `max_tokens` น้อยเกินไป คำตอบอาจว่างหรือถูกตัดกลางประโยค
- **ราคา** เปลี่ยนบ่อยและต่างกันในแต่ละราย คอร์สนี้จึงไม่ใส่ราคาตายตัว ให้ดูหน้า pricing ของแต่ละราย และเทียบจำนวน token ที่แล็บพิมพ์ออกมา

Lab 01 พิมพ์ตารางนี้สำหรับเครื่อง *ของคุณ* โดยบอกเพียงว่าพบ key ที่ไหน ไม่เคยแสดงค่า key:

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab01_pick_your_llm.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · which providers can this machine use? (key SOURCE only — values are never shown)
│    id          provider                default model        key
│ ─  ──────────  ──────────────────────  ───────────────────  ──────────
│ ✓  claude      Claude (Anthropic)      claude-sonnet-5      repo .env
│ ✓  chatgpt     ChatGPT (OpenAI)        gpt-5.4-mini         repo .env
│ ✓  gemini      Gemini (Google)         gemini-3.5-flash     repo .env
│ ✓  deepseek    DeepSeek                deepseek-flash       repo .env
│ ✓  kimi        Kimi (Moonshot AI)      kimi-k3              repo .env
│ ✓  glm         GLM (Z.ai / Zhipu)      glm-5.3              repo .env
│ ✓  openrouter  OpenRouter (any model)  openai/gpt-5.4-mini  repo .env
│ ✓  ollama      Ollama (local, $0)      gemma4:latest        not needed
◆ 8 of 8 providers ready: claude, chatgpt, gemini, deepseek, kimi, glm, openrouter, ollama

▣ STEP 2 · your choice (the runner's LLM picker, or JEV_LLM_PROVIDER / JEV_LLM_MODEL)
→ provider claude · model claude-sonnet-5

▣ STEP 3 · one hello — then compare what the model SAYS with what the API REPORTS
   │ hello from Claude
◆ Claude (Anthropic) · claude-sonnet-5 · LIVE · 22 in / 8 out tok · 1840 ms
```

ถ้าเพิ่ง clone repo มา ทุกแถวจะขึ้น `·` และ `missing — set ANTHROPIC_API_KEY` จนกว่าจะเพิ่ม key (ขั้นถัดไป) ทุกอย่างยังรันได้ในโหมด DRY

✓ Checkpoint: lab 01 แสดง provider ที่มี ✓ อย่างน้อยหนึ่งราย (Ollama นับด้วยถ้ากำลังรันอยู่) หรือคุณเข้าใจว่าทำไมทุกแถวขึ้นว่า missing

## 2 · Key: ห้ามอยู่ในโค้ด และวิธีเลือกโมเดล

**ห้ามวาง key ลงในไฟล์ `.py` notebook หรือบทเรียนเด็ดขาด** แล็บจะค้นหา key ตามลำดับนี้:

1. **environment** ของ process (เช่น `export OPENAI_API_KEY=…` ใน shell)
2. **`week24/.env.local`** ซึ่งกล่อง **🔑 Keys** ของ Lab Runner เขียนให้ (อยู่ใน .gitignore และอ่านได้เฉพาะคุณ)
3. **`.env` ที่รากของ repo** (อยู่ใน .gitignore)

**ถ้าคุณ clone repo นี้มา** เลือกวิธีใดวิธีหนึ่ง:

- เปิดกล่อง **🔑 Keys** ใน Lab Runner วาง key แล้วกด Save หรือ
- คัดลอกแม่แบบแล้วกรอกเอง:

```bash
cp -n week24/.env.example .env   # -n = never overwrite an .env you already have
# then edit .env — set only the keys you have, leave the rest empty
# already have a .env? copy just the lines you need from week24/.env.example into it
```

ทั้งสองไฟล์อยู่ใน `.gitignore` จึงไม่มีทาง commit key ขึ้นไปโดยไม่ตั้งใจ คุณต้องการเพียง `TYPESAFE_API_KEY` สำหรับ Jev และ key ของ LLM **หนึ่ง** ราย (Ollama ไม่ต้องใช้ key)

**การเลือก LLM:** หัวของ Lab Runner มีตัวเลือก **LLM** (provider + model) ทุกครั้งที่กด ▶ Run ตัวเลือกของคุณจะถูกส่งให้แล็บเป็น `JEV_LLM_PROVIDER` และ `JEV_LLM_MODEL` ถ้ารันใน terminal ให้ตั้งเอง:

```bash
JEV_LLM_PROVIDER=gemini .venv/bin/python week24/11_jev_plus_llm/labs/lab01_pick_your_llm.py
JEV_LLM_PROVIDER=glm JEV_LLM_MODEL=glm-5.3-flash .venv/bin/python week24/11_jev_plus_llm/labs/lab02_route_then_write.py
```

ลองใช้ LLM ได้ตรงนี้เลย บล็อกด้านล่างจะใช้ provider ที่เลือกไว้บนหัวหน้า:

```llm
{
  "system": "You write short replies to hotel guests for staff to review. 2 sentences. Never promise refunds.",
  "user": "Guest in room 1203: the air conditioning is freezing and won't turn off. It's 1am. Write the reply.",
  "max_tokens": 400
}
```

ลองเล่น: เปลี่ยนตัวเลือก LLM เป็น provider อื่นแล้วถามใหม่ จากนั้นลองบล็อกภาษาไทยด้านล่าง แล้วสังเกตว่าคำตอบเป็นภาษาเดียวกับแขกไหม:

```llm
{
  "system": "คุณเขียนคำตอบสั้น ๆ ถึงแขกของโรงแรม เพื่อให้พนักงานตรวจก่อนส่ง ตอบเป็นภาษาไทย 2 ประโยค ห้ามสัญญาว่าจะคืนเงิน",
  "user": "แขกห้อง 1203: แอร์เย็นมากและปิดไม่ได้ ตอนนี้ตีหนึ่งแล้ว ช่วยเขียนคำตอบให้หน่อย",
  "max_tokens": 1200
}
```

สังเกต: บล็อกภาษาไทยให้งบ `max_tokens` 1200 ส่วนบล็อกภาษาอังกฤษให้ 400 ตอนทดลองด้วยงบ 400 DeepSeek, Kimi และ Ollama ใช้ token หมดไปกับการ "คิด" จนไม่ได้คำตอบภาษาไทยเลย ขณะที่ prompt ภาษาอังกฤษขนาดเท่ากันยังได้คำตอบ ถ้าระบบของคุณตอบแขกเป็นภาษาไทย ให้เผื่องบ token ไว้มากกว่าภาษาอังกฤษ

✓ Checkpoint: คุณบอกได้ว่ามีสามที่ที่ระบบค้นหา key และได้รันแล็บด้วย provider อื่นที่ไม่ใช่ค่าเริ่มต้นอย่างน้อยหนึ่งครั้ง

## 3 · Jev จัดเส้นทาง → โค้ดเลือก prompt ที่อนุมัติแล้ว → LLM เขียน

`week24/11_jev_plus_llm/pipeline.py` เก็บ pipeline ทั้งหมดไว้ในโค้ดราว 150 บรรทัดที่อ่านเข้าใจง่าย ขั้นแรก Jev จัดเส้นทางข้อความของแขก ลองเรียกได้ตรงนี้:

```jev
{
  "state": {"message": "Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which hotel team should handle `message`?",
      "criteria": {
        "hvac": "Air conditioning, heating, ventilation, room too hot or too cold",
        "housekeeping": "Cleaning, towels, linen, amenities, room tidiness",
        "maintenance": "Plumbing, electrical, broken furniture, doors, lights, TV",
        "front_desk": "Bookings, billing, check-in/out, general questions",
        "unknown": "No clear request, or none of the above"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the request in `message`?",
      "criteria": ["Routine: no time pressure stated", "Should be handled today", "Needs attention within the hour", "Possible danger or a total outage right now"]
    },
    "wants_compensation": {
      "type": "noul",
      "instructions": "Does `message` ask for a refund, discount, free night or other compensation?"
    }
  }
}
```

จากนั้น **โค้ด** (ไม่ใช่ LLM) ทำสามอย่าง:

1. **ข้อเท็จจริงแน่นอน:** ภาษา (ดูจากอักษรไทยด้วย regex) และเลขห้อง (regex) ไม่มีการเดา
2. **ด่านความไม่แน่ใจ (uncertainty gate):** ถ้า Jev ไม่แน่ใจว่าเป็นทีมไหน (ความน่าจะเป็นสูงสุด < 0.70 หรือสองทีมสูสีกัน) คำขอจะไปที่ front desk
3. **prompt ที่อนุมัติแล้ว:** system prompt มาจาก `APPROVED_PROMPTS[team]` ซึ่งมีกฎบ้านห้ามสัญญาคืนเงิน และห้ามแต่งชื่อ เวลา หรือราคาขึ้นมาเอง LLM จะเห็นเพียง prompt ที่อนุมัติแล้วกับรายการข้อเท็จจริงสั้น ๆ เท่านั้น

Lab 02 รันข้อความสามแบบ (อังกฤษ ไทย และลูกค้าโกรธขอเงินคืน) ผ่านทั้งเส้นทางด้วย LLM ที่คุณเลือก:

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab02_route_then_write.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · “Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep.”
» department  hvac         (confidence 1.00)  → team: hvac
» urgency     2.07 on 0…3   · wants compensation 0.03
» code facts  language=en · room=1203
   │ Thank you for letting us know, and we're sorry for the trouble this is causing your family so
   │ late at night. We've asked our HVAC team to check the air conditioning in Room 1203 within the
   │ hour. …
   │ Guest Services
◆ Claude (Anthropic) · claude-sonnet-5 · LIVE · 233 in / 95 out tok · 2219 ms

▣ STEP 2 · “ห้อง 815 ยังไม่มีผ้าเช็ดตัวเลยค่ะ รบกวนเอามาให้หน่อยนะคะ”
» department  housekeeping (confidence 1.00)  → team: housekeeping
» code facts  language=th · room=815
   │ … ทางเราได้แจ้งทีมแม่บ้านให้รีบนำผ้าเช็ดตัวไปให้ภายในวันนี้แล้วค่ะ …

▣ STEP 3 · “I was charged twice for my stay. I want a refund today or I'm leaving a review.”
» department  front_desk   (confidence 1.00)  → team: front_desk
» urgency     1.02 on 0…3   · wants compensation 0.98
→ compensation requested: the draft must NOT promise it — a manager is flagged separately
   │ Thank you for letting us know about the duplicate charge — … Our front desk team has been
   │ alerted and will follow up with you today to look into this. …

◆ Jev decision avg 660 ms · claude draft avg 2765 ms → the typed decision is the cheap, fast step; writing is the slow one
═ execute: False — every draft goes to a staff member for approval; nothing is sent.
```

ดูขั้นที่ 3: แขกเรียกร้องเงินคืน (`wants_compensation` 0.98) และร่างคำตอบ **รับทราบโดยไม่สัญญา** กฎบ้านใน prompt ที่อนุมัติแล้วทำงานได้ตามที่ตั้งใจ การตัดสินใจเรื่องคืนเงินเป็นของผู้จัดการ ไม่ใช่ของโมเดล

✓ Checkpoint: คุณชี้ได้ว่าบรรทัดไหนใน `pipeline.py` ที่โค้ด (ไม่ใช่ LLM) เป็นผู้เลือก system prompt

## 4 · งานเดียวกัน กับทุกโมเดลที่คุณมี

Lab 03 ให้ Jev จัดเส้นทางแต่ละข้อความ **ครั้งเดียว** แล้วให้ทุก provider ที่มี key เขียนคำตอบเดียวกันพร้อมกัน (แบบขนาน แล็บจึงเสร็จเร็ว) จากนั้นพิมพ์ latency จำนวน token ที่ตอบ ความยาว แขกไทยได้คำตอบภาษาไทยไหม และร่างจบประโยคครบหรือเปล่า

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab03_compare_models.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · “Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep.”
» Jev → team hvac · language en (code) · 520 ms
│ provider    model (API says)         ms     out tok  chars  reply language  complete?
│ claude      claude-sonnet-5          2304   108      279    ✓ English       ✓
│ chatgpt     gpt-5.4-mini-2026-03-17  1622   123      223    ✓ English       ✓
│ gemini      gemini-3.5-flash         2350   76       341    ✓ English       ✓
│ deepseek    deepseek-flash           3865   583      225    ✓ English       ✓
│ kimi        kimi-k3                  18649  562      408    ✓ English       ✓
│ glm         glm-5.3                  2355   79       346    ✓ English       ✓
│ openrouter  openai/gpt-5.4-mini      1616   44       180    ✓ English       ✓
│ ollama      gemma4:latest            21694  594      256    ✓ English       ✓

▣ STEP 2 · “ห้อง 815 ยังไม่มีผ้าเช็ดตัวเลยค่ะ รบกวนเอามาให้หน่อยนะคะ”
» Jev → team housekeeping · language th (code) · 586 ms
│ claude      claude-sonnet-5          4494   173      198    ✓ Thai          ✓
│ chatgpt     gpt-5.4-mini-2026-03-17  1452   102      201    ✓ Thai          ✓
│ gemini      gemini-3.5-flash         1854   58       192    ✓ Thai          ✓
│ kimi        kimi-k3                  12662  490      219    ✓ Thai          ✓
│ …
```

อ่านผลอย่างซื่อตรง:

- **นี่คือการสังเกตครั้งเดียว ไม่ใช่ benchmark** รันสองครั้งตัวเลขก็ขยับ Kimi ใช้เวลา 12–23 วินาทีในแต่ละรอบที่เราทดลอง
- **"out tok" รวม token ที่ใช้ "คิด" ซึ่งมองไม่เห็น** ของโมเดลที่ใช้เหตุผล DeepSeek ใช้ 583 token เพื่อคำตอบยาว 225 ตัวอักษร ส่วน Gemini ใช้ 76 ต่อให้ราคาต่อ token เท่ากัน ค่าใช้จ่ายจริงก็ต่างกัน
- **"complete?"** คือการตรวจด้วยโค้ดแบบแน่นอน (ร่างจบเหมือนข้อความที่เขียนเสร็จไหม) ระหว่างสร้างแล็บนี้ Kimi เคยใช้งบ 700 token หมดไปกับการคิดจนไม่ได้คำตอบเลย และ Gemini เคยหยุดกลางประโยค ("…check the air") ตอนนี้แล็บให้งบ token เผื่อไว้มาก และคอลัมน์นี้จะจับได้เมื่อเกิดขึ้นอีก
- **"model (API says)"** คือเวอร์ชันที่ตอบจริง OpenAI ส่งกลับ `gpt-5.4-mini-2026-03-17` ที่มีวันที่กำกับ แม้เราเรียกด้วยชื่อย่อ `gpt-5.4-mini` ให้บันทึกค่านี้ไว้
- ทั้งแปดรายตอบแขกไทยเป็นภาษาไทย ส่วนภาษาไทยนั้น *ฟังเป็นธรรมชาติ* สำหรับแขกคนไทยหรือไม่ ต้องให้คนที่พูดภาษาไทยตรวจบนข้อความจริงสักสิบหรือยี่สิบข้อความ

✓ Checkpoint: จากตารางของคุณเอง คุณบอกได้ว่าจะลองโมเดลไหนก่อนสำหรับคำตอบภาษาไทย และจะเลี่ยงโมเดลไหนถ้ามีงบเวลาตอบเพียง 2 วินาที

## 5 · ให้ Jev ตรวจร่างของ LLM

LLM เขียนได้ลื่นไหล ข้อผิดพลาดจึงมองข้ามได้ง่าย ร่างคำตอบอาจสัญญาคืนเงินที่ไม่มีใครอนุมัติ หรือเอ่ยชื่อช่างที่ไม่มีอยู่จริง ดังนั้นหลัง LLM เขียนเสร็จ Jev จะตอบคำถามแบบมีชนิดสี่ข้อเกี่ยวกับ **ร่าง** และโค้ดเพิ่มการตรวจแบบแน่นอนอีกหนึ่งข้อ:

| การตรวจ | ใครตรวจ | ไม่ผ่านเมื่อ |
|---|---|---|
| `promises_compensation` | Jev noul | ≥ 0.5 — เสนอคืนเงิน ส่วนลด คืนฟรี อัปเกรด… |
| `invents_facts` | Jev noul | ≥ 0.5 — มีชื่อ เวลา ราคา หรือสาเหตุ ที่ไม่มีทั้งในข้อความแขกและในรายการข้อเท็จจริง |
| `reply_language` | Jev choice | ไม่ตรงกับภาษาของแขก (ซึ่งโค้ดคำนวณไว้) |
| `addresses_request` | Jev noul | < 0.5 — ร่างไม่ได้ตอบสิ่งที่แขกถาม |
| มีเลขห้องไหม | **โค้ด** | `facts["room"]` มีค่า แต่ไม่ปรากฏในร่าง (ตรวจสตริงตรงตัว) |

ไม่ผ่านข้อใดข้อหนึ่ง → **ให้คนตรวจ (human review)** ผ่านทั้งหมด → **ให้พนักงานอนุมัติ (staff approval)** ซึ่งก็ยังต้องมีคนกดอยู่ดี ลองตรวจร่างที่แย่ดู:

```jev
{
  "state": {
    "guest_message": "Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep.",
    "facts": {"room": "1203", "team_asked": "hvac", "urgency": "within the hour"},
    "draft": "So sorry about the cold room! We will refund tonight's stay in full, and our technician Somchai will be at your door by 1:15am. Guest Services"
  },
  "questions": {
    "promises_compensation": {"type": "noul", "instructions": "Does `draft` promise or offer a refund, discount, free night, upgrade or other compensation?"},
    "invents_facts": {"type": "noul", "instructions": "Does `draft` state a specific fact — a person's name, a time or deadline, a price, or a cause of the problem — that appears in neither `guest_message` nor `facts`?"},
    "addresses_request": {"type": "noul", "instructions": "Does `draft` respond to what the guest actually asked for in `guest_message`?"}
  }
}
```

ลองเล่น: ลบประโยคเรื่องคืนเงินแล้วถามใหม่ noul ตัวไหนลดลง? จากนั้นลบ "Somchai" และ "by 1:15am"

Lab 04 นำร่างจริงจาก LLM ของคุณกับร่างแย่ที่เราจงใจใส่ไว้ ผ่านการตรวจ:

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab04_jev_checks_the_draft.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 3 · Jev checks A · LLM draft
» promises_compensation  noul    0.03  █░░░░░░░░░░░░░░░░░░░░░░░  likely NO
» invents_facts          noul    0.15  ████░░░░░░░░░░░░░░░░░░░░  likely NO
» reply_language         choice  → en   (confidence 1.00)
» addresses_request      noul    0.90  ██████████████████████░░  likely YES
✓ passes every check → goes to a staff member for one-click approval

▣ STEP 4 · Jev checks B · planted draft
» promises_compensation  noul    0.98  ████████████████████████  likely YES
» invents_facts          noul    0.98  ████████████████████████  likely YES
» addresses_request      noul    0.73  ██████████████████░░░░░░  uncertain
→ HUMAN REVIEW — 3 problem(s):
   ✕ promises compensation — only a manager may offer that
   ✕ states facts that are not in the message or the facts list
   ✕ room 1203 is missing from the draft

═ execute: False — Jev never sends; it only decides whether a person must look first.
```

การตรวจเลขห้องเป็นโค้ดธรรมดาโดยตั้งใจ: "มี `1203` ในข้อความไหม" มีคำตอบที่แน่นอน โมเดลจึงไม่ช่วยอะไรเพิ่ม ข้อควรระวังจากการรันจริงของเรา: กับร่าง **ภาษาไทย** ของ Claude ค่า `invents_facts` ออกมา **0.64** เพราะ Jev มองว่า "ภายในวันนี้" เป็นข้อมูลที่แต่งขึ้น ทั้งที่ `urgency: today` อยู่ในรายการข้อเท็จจริงแล้ว นี่คือสัญญาณเตือนผิดในทิศที่ปลอดภัย (มีคนมาตรวจ) แต่ก็บอกว่าการตรวจข้อความภาษาไทยต้องปรับ threshold แยก บนร่างภาษาไทยจริง

✓ Checkpoint: คุณอธิบายได้ว่าทำไมการตรวจเลขห้องเป็นโค้ด ในขณะที่การตรวจ "แต่งข้อมูลขึ้นเอง" เป็นงานของ Jev

## 6 · กับดักเรื่องตัวตนของโมเดล

ระหว่างสร้างแล็บนี้ เราขอให้ทุก provider "reply with exactly: hello from <your model name>" ได้คำตอบที่น่าสนใจสองข้อ:

- **DeepSeek** (`deepseek-flash`) ตอบว่า **"hello from ChatGPT"**
- **GLM** (`glm-5.3`) ตอบว่า **"hello from GLM-4.6"**

ไม่มีโมเดลไหนตั้งใจโกหก โมเดลเรียนรู้จากข้อความที่คนเขียนถึงโมเดลอื่น และมักไม่รู้เวอร์ชันของตัวเอง บทเรียนสำหรับระบบจริง:

- **บันทึกค่า `model` ที่ API ส่งกลับ** อย่าใช้คำบรรยายตัวเองของโมเดล Lab 01 พิมพ์ทั้งสองค่าเทียบกันให้ดู
- หลักเดียวกันใช้กับทุกคำอ้างที่ LLM พูดถึงตัวเอง ("ฉันตรวจฐานข้อมูลแล้ว", "ฉันมั่นใจ") ต้องยืนยันด้วยโค้ดหรือด้วยการตรวจแบบมีชนิดของ Jev ห้ามเชื่อตัวข้อความ

✓ Checkpoint: คุณบอกได้ว่าจะเก็บฟิลด์ไหนใน audit log เพื่อพิสูจน์ว่าโมเดลไหนเป็นผู้เขียนคำตอบ

## แล็บ — รันได้ที่นี่

**labs/lab01_pick_your_llm.py** — provider ไหนมี key (บอกเฉพาะที่มา ไม่แสดงค่า) ตัวเลือกของคุณ และคำทักทายหนึ่งครั้ง: สิ่งที่โมเดลพูด เทียบกับสิ่งที่ API รายงาน

**labs/lab02_route_then_write.py** — Jev จัดเส้นทางข้อความแขกสามข้อความ โค้ดเลือก prompt ที่อนุมัติแล้ว และ LLM ของคุณเขียนคำตอบแต่ละข้อ

**labs/lab03_compare_models.py** — งานเดียวกันกับทุก provider ที่คุณมี ทั้งอังกฤษและไทย: latency, token, ภาษา และการตรวจว่าคำตอบครบ

**labs/lab04_jev_checks_the_draft.py** — Jev ตรวจร่างจริงจาก LLM และร่างแย่ที่จงใจใส่ไว้ นโยบายส่งร่างแย่ไปให้คนตรวจ

เลือกผู้เขียนด้วยตัวเลือก **LLM** บนหัวหน้าก่อนกด ▶ Run ทุกแล็บรันในโหมด DRY จากคำตอบที่บันทึกไว้ได้ด้วย

## ลองทำเอง

**แบบฝึกหัด 11: ป้องกันคำตอบด้วยตัวเอง** เปิด `week24/11_jev_plus_llm/exercises/ex11_guarded_reply.py` มี TODO สองข้อ:

1. `MY_CHECKS`: คำถาม Jev สี่ข้อเกี่ยวกับร่าง (noul สามข้อ choice หนึ่งข้อ) ครอบด้วย `guarded({...})`
2. `my_review()`: นโยบาย เพิ่มเหตุผลหนึ่งข้อต่อกฎที่ไม่ผ่าน กฎเลขห้องเขียนเป็นโค้ดธรรมดา และ `execute` เป็น `False` เสมอ

ตัวตรวจจะรันกรณีทดสอบแบบออฟไลน์ห้ากรณีก่อน (ฟรี) เมื่อผ่านทั้งหมด LLM ที่คุณเลือกจะเขียนร่างจริงสองฉบับ เพิ่มร่างแย่ที่จงใจใส่ไว้ แล้วให้การตรวจของคุณตัดสินแต่ละฉบับ

```bash
.venv/bin/python week24/11_jev_plus_llm/exercises/ex11_guarded_reply.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ MY_CHECKS has exactly the four keys
✓ question types are right (three nouls + one choice)
✓ reply_language offers en / th / mixed / other
✓ every question carries the injection guard
✓ policy: clean English draft → staff_approval (0 reason(s))
✓ policy: promises a refund → human_review (1 reason(s))
✓ policy: invented technician + missing room → human_review (2 reason(s))
✓ policy: English reply to a Thai guest → human_review (1 reason(s))
✓ policy: off-topic, everything else fine → human_review (1 reason(s))

▣ offline checks passed — now live: claude (claude-sonnet-5) writes, your checks decide
━━ planted bad draft: So sorry! We will refund tonight's stay, and Somchai will be there by 1:15am.
» compensation 0.98 · invents 0.98 · language en · on-topic 0.63
→ HUMAN REVIEW: promises compensation; invents facts; room 1203 missing
```

<details><summary>คำใบ้: ทำไมตรวจเลขห้องด้วยโค้ด?</summary>

"มี `1203` ในร่างไหม" ก็คือ `facts["room"] in draft` ซึ่งแน่นอน ฟรี และไม่เคยผิด ถ้าถาม Jev จะเพิ่มทั้งค่าใช้จ่ายและโอกาสผิดพลาดเล็ก ๆ ให้คำถามที่ไม่ต้องใช้วิจารณญาณเลย ใช้โมเดลกับเรื่อง "ความหมาย" และใช้โค้ดกับเรื่อง "ข้อเท็จจริง"

</details>

<details><summary>ท้าทายเพิ่ม: ผู้เขียนสำรองคนที่สอง</summary>

เมื่อการตรวจส่งร่างไปให้คนดู ลองให้ provider *อีกราย* เขียนใหม่อีกครั้ง (เช่น Claude → Gemini) แล้วตรวจซ้ำ นับว่าร่างที่สองผ่านบ่อยแค่ไหน อย่าลืมว่ายังต้องมีคนอนุมัติทุกครั้งที่ส่ง

</details>

✓ Checkpoint: การตรวจออฟไลน์ขึ้น ✓ ทั้งหมด และร่างแย่ที่จงใจใส่ไว้จบที่ human review

## แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `HTTP 401 (key missing or invalid)` | key ของ provider นั้นผิดหรือหมดอายุ วางใหม่ใน 🔑 Keys หรือใน `.env` หรือเลือก provider อื่น |
| `HTTP 404 (unknown model id…)` | ชื่อโมเดลเปลี่ยนไปแล้ว เลือกจากรายการโมเดลใน Lab Runner (ซึ่งถามจาก provider แบบสด) |
| `[the model used its whole token budget thinking …]` | โมเดลที่ใช้เหตุผลใช้ `max_tokens` หมดไปกับการคิด เพิ่ม `max_tokens` หรือเลือกโมเดลรุ่น `-flash`/`-mini` |
| คำตอบจบกลางประโยค / `complete? ⚠` | สาเหตุเดียวกัน ให้ token เพิ่ม |
| Ollama: `network/timeout` | เปิดด้วย `ollama serve` และ pull โมเดล (`ollama pull gemma4`) หรือตั้ง `OLLAMA_BASE_URL` |
| Kimi / GLM ได้ `401` จากจีน | ใช้ `KIMI_BASE_URL=https://api.moonshot.cn/v1` หรือ `GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4` |
| `◈ no recording (DRY)` สำหรับบาง provider | DRY เล่นซ้ำได้เฉพาะสิ่งที่บันทึกไว้ สลับเป็น ⚡ Live พร้อม key ของ provider นั้น |
| แขกเขียนไทยแต่ได้คำตอบภาษาอังกฤษ | ตรวจว่ากฎบ้าน `reply in {language}` เข้าไปใน prompt แล้ว จากนั้นลองโมเดลอื่น เพราะความสามารถภาษาไทยต่างกัน |

## ถัดไป

ไปต่อที่ [Lab 12 — Capstone: ผู้ช่วยรับคำขอของแขกในโรงแรมอัจฉริยะ](../12_capstone_hotel_copilot/TUTORIAL.md) คุณจะนำทุกอย่างมารวมกัน และขั้นร่างคำตอบที่สร้างในแล็บนี้ก็คือจุดที่ LLM ที่คุณเลือกจะเสียบเข้าไป
