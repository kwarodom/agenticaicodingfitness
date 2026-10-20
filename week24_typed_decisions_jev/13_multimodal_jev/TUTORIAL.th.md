# ▶ Jev Lab 13 — มากกว่าข้อความ: ภาพ เสียง และวิดีโอ — โมเดลรับรู้ Jev ตัดสิน โค้ดเป็นผู้ตัดสินใจ

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิด (typed decision) ด้วย Jev — ในบทนี้ Jev อ่านเฉพาะ **ข้อความ** ส่วนรูปถ่าย ข้อความเสียง และคลิป CCTV ไม่ใช่ข้อความ ดังนั้นโมเดลแบบ multimodal จะ *รับรู้* สื่อเหล่านั้นก่อน แล้วเขียนออกมาเป็นข้อความว่าเห็นหรือได้ยินอะไร จากนั้น Jev จึงตัดสินใจแบบมีชนิดอย่างรวดเร็วบนข้อความนั้น และโค้ดของคุณเป็นผู้ตัดสินใจว่าจะทำอะไรต่อ สื่อทั้งหมดในบทนี้เป็นข้อมูลสังเคราะห์ (synthetic)

**สิ่งที่คุณจะได้ลงมือทำ**
- ทำแผนที่ว่าโมเดลไหน **มองเห็น (see)** **ได้ยิน (hear)** **พูด (speak)** และ **ดูวิดีโอ (watch)** ได้บ้างด้วย key ที่คุณมี — และจับเวลาเทียบกับการเรียก Jev หนึ่งครั้ง
- เปลี่ยนรูปเทอร์โมสแตต รูปน้ำรั่ว และใบแจ้งหนี้ที่เป็นการฉ้อโกง ให้เป็นการตัดสินใจแบบมีชนิด — โดยให้ *โค้ด* อ่านตัวเลข ไม่ใช่โมเดล
- เปลี่ยนข้อความเสียงภาษาอังกฤษและภาษาไทยให้เป็น ticket และคำตอบแบบเสียงพูด — และจับหมายเลขห้องที่โมเดลถอดเสียง **ทั้งสองตัว** ถอดผิด
- เปลี่ยนคลิป CCTV ให้เป็นการแจ้งเตือน — และเห็นว่าทำไมคำว่า "ไอน้ำ (steam)" หรือ "ควัน (smoke)" ต้องไม่ทำให้ผลลัพธ์เปลี่ยน
- ใช้ Jev เป็น **ผู้เฝ้าประตู (gatekeeper)** เพื่อจ่ายเงินให้การรับรู้ที่ช้าและแพงเฉพาะเมื่อข้อความบอกว่าจำเป็นเท่านั้น

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Cost** ≈ $0.0005 สำหรับ Jev + ไม่กี่เซนต์สำหรับการเรียกโมเดลรับรู้เมื่อรันจริง · $0 ในโหมด dry

## 0 · Text-only กับ multimodal — และ Jev อยู่ตรงไหน

โมเดล *multimodal* รับอินพุตได้มากกว่าข้อความ เช่น ภาพ เสียง วิดีโอ หรือสตรีมแบบสด ในบทนี้ Jev อ่าน **ข้อความเท่านั้น** — มันคือผู้ตัดสินแบบ System One ที่เร็ว ส่วนโมเดล multimodal ทำหน้าที่เป็นตาและหู ทั้งสองส่วนทำงานร่วมกันแบบนี้:

> **อัปเดต ตุลาคม 2026:** ตอนนี้ Jev 1.14 (และ 1.13 เมื่อใส่ header ยืนยัน) อ่านภาพนิ่งได้โดยตรงแล้ว ในฐานะส่วนหนึ่งของ "danger zone" (โซนทดลอง) ของ TypeSafe แต่ pipeline ของบทนี้ยังเป็นเส้นทางสำหรับ production และเป็นเส้นทางเดียวที่รองรับเสียง วิดีโอ และตัวเลขที่แม่นยำ — [Lab 15 — Jev มองเห็นภาพ](../15_jev_sees_images/TUTORIAL.th.md) วัดผลเส้นทางตรงเทียบกับเส้นทางนี้บนรูปชุดเดียวกัน

```text
photo / voicemail / CCTV clip
   │
   ▼  multimodal model (perceive)      pixels & sound → TEXT            2–4 s  (13 s local)
   │
   ▼  your code (exact facts)          regex numbers · arithmetic · caller-ID
   │
   ▼  Jev (judge the text)             queue · safety · suspicious       ~0.5 s, ≈ $0.00003
   │
   ▼  your code (one policy)           proposal · alert · human review    execute: False
```

นี่คือความสามารถที่ key ใน repo ของคุณเข้าถึงได้วันนี้ (ตรวจสอบแบบ live เมื่อ 2026-09-27):

| ความสามารถ | อะไรเข้า → อะไรออก | โมเดลที่ใช้ในบทนี้ |
|---|---|---|
| **see** | ภาพ → ข้อความ | `gemini-3.5-flash` · `gpt-5.4-mini` · `claude-sonnet-5` · `gemma4:latest` (Ollama บนเครื่อง, $0) |
| **hear** | เสียง → ข้อความถอดเสียง (transcript) | `gpt-4o-mini-transcribe` · `gemini-3.5-flash` |
| **speak** | ข้อความ → เสียงพูด | `gpt-4o-mini-tts` · `gemini-3.8-flash-tts` |
| **watch** | วิดีโอ → ข้อความ | `gemini-3.8-flash` |
| **live** | การโทรด้วยเสียงแบบสตรีมมิง | `gemini-3.8-live` · `gpt-realtime-2.1-mini` → บทที่ 14 |
| มีใน key เหล่านี้ด้วย | any-to-any และการสร้างภาพ/วิดีโอ | `gemini-omni-1.1-flash`, `gpt-image-*`, `veo`, `sora` — กล่าวถึงแต่ไม่ได้สร้างในบทนี้ |

รูปถ่ายสังเคราะห์ใน `media/` สร้างด้วยโมเดลสร้างภาพ ส่วนหน้าจอเทอร์โมสแตต กราฟแนวโน้ม และใบแจ้งหนี้วาดด้วยโค้ด และเสียงพูดสร้างด้วย text-to-speech

**ทำไมต้องวาง Jev ต่อจากโมเดล multimodal?** เพราะโมเดลรับรู้ทำหน้าที่ *บรรยาย* — ยาว และใช้ถ้อยคำที่เปลี่ยนไปทุกครั้งที่รัน Jev เปลี่ยนคำบรรยายใด ๆ ให้เป็น schema เล็ก ๆ ที่มีชนิดและทดสอบได้แบบเดิมเสมอ ทำให้ policy ชุดเดียวใช้ได้กับทุกประเภทสื่อและทุกโมเดลรับรู้ แล็บ 01 แสดงความต่างด้านความเร็วบนเครื่องของคุณ:

```bash
.venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab01_capability_map.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · who can do what on this machine (from your keys — no API calls)
│ provider    see                        hear                      speak                   watch
│ chatgpt     ✓ gpt-5.4-mini             ✓ gpt-4o-mini-transcribe  ✓ gpt-4o-mini-tts       —
│ claude      ✓ claude-sonnet-5          —                         —                       —
│ gemini      ✓ gemini-3.5-flash         ✓ gemini-3.5-flash        ✓ gemini-3.8-flash-tts  ✓ gemini-3.8-flash
│ ollama      ✓ gemma4:latest            —                         —                       —
│ openrouter  ✓ google/gemini-3.5-flash  —                         —                       —
…
▣ STEP 3 · latency side by side
│ step          provider  model                   time    source
│ see           gemini    gemini-3.5-flash        4.03 s  live
│ hear          chatgpt   gpt-4o-mini-transcribe  1.24 s  live
│ speak         chatgpt   gpt-4o-mini-tts         2.11 s  live
│ watch         gemini    gemini-3.8-flash        2.59 s  live
│ Jev (decide)  typesafe  jev-1.13.0              0.44 s  live
```

**ตัวเลือก LLM (LLM picker)** ที่หัวของ runner มีผลตรงนี้ด้วย: แล็บจะใช้ผู้ให้บริการที่คุณเลือกสำหรับความสามารถนั้น *ถ้ามันทำได้* (Claude มองเห็นได้ แต่ฟังและพูดไม่ได้) มิฉะนั้นจะใช้ผู้ให้บริการตัวแรกที่มี key และทำได้

✓ Checkpoint: คุณอธิบายได้ว่าทำไม Jev ไม่จำเป็นต้องเห็นพิกเซลเลย และบอกได้ว่าผู้ให้บริการไหนของคุณที่ฟังเสียงได้

## 1 · ภาพ → การตัดสินใจ

สามภาพ สามบทเรียน เริ่มจากเทอร์โมสแตต:

![หน้าจอเทอร์โมสแตต ห้อง 1203 — 29.1 °C ตั้งไว้ 24.0 °C (สังเคราะห์)](media/thermostat_room1203.png)

เราขอให้โมเดลมองเห็นตอบเป็นบรรทัดเดียวตามรูปแบบที่กำหนด จากนั้น **โค้ด** อ่านตัวเลขด้วย regex และคำนวณส่วนต่าง Jev ตอบเฉพาะคำถามเชิงวิจารณญาณ — *หน้าจออ่านได้ชัดเจนหรือไม่?* — เพราะเลขคณิตเป็นหน้าที่ของโค้ด (แล็บ 03)

![น้ำขังใต้เครื่องส่งลมเย็น AHU (สังเคราะห์)](media/leak_under_ahu.jpg)

สำหรับรูปน้ำรั่ว โมเดลมองเห็นจะเขียนคำบรรยาย — รวมถึงประโยค **"Unsure:"** ที่บอกสิ่งที่ไม่แน่ใจ — แล้ว Jev เปลี่ยนคำบรรยายนั้นเป็น queue · safety · severity · hedged ลองใช้คำถามชุดเดียวกันของ Jev กับคำบรรยายนี้ด้วยตัวเอง:

```jev
{
  "state": {"description": "The image shows a large, grey air handling unit situated in a basement utility room. A significant puddle of dark liquid has accumulated on the concrete floor directly underneath the machine, which appears to be dripping from the frame above. This poses an immediate slip hazard. Unsure: I cannot tell whether the leaking liquid is water condensation, oil, or another type of mechanical fluid."},
  "questions": {
    "queue": {"type": "choice", "instructions": "Which engineering queue should handle the issue in `description`?",
      "criteria": {"water_leak": "Water or another liquid leaking, dripping or pooling", "smoke_or_fire": "Smoke, steam, haze, burning, fire or sparks", "comfort": "Temperature or comfort complaint with no visible damage", "none": "Nothing wrong is described", "unclear": "The description is too vague or contradictory to choose"}},
    "safety": {"type": "noul", "instructions": "Does `description` describe something that could injure a person or damage electrical equipment — smoke, steam, haze, fire, sparks, or water near electrical equipment?"},
    "hedged": {"type": "noul", "instructions": "Does `description` say the image is unclear, or that the model cannot tell something important?"}
  }
}
```

ทดลอง: ลบประโยค `Unsure:` ออกแล้วถามใหม่ — ค่า `hedged` ควรลดลง นี่คือวิธีบอก *คน* ว่าโมเดลรับรู้เองก็ไม่แน่ใจ

![ใบแจ้งหนี้จากซัพพลายเออร์ที่ขอให้โอนเงินไปบัญชีธนาคารใหม่ (สังเคราะห์)](media/invoice_bank_change.png)

ใบแจ้งหนี้คือกับดัก: **เลขคณิตถูกต้อง** (84,000 + 12,500 = 96,500; VAT 7 % = 6,755; รวม 103,255 — ตรวจด้วยโค้ด) ดังนั้นการตรวจตัวเลขอย่างเดียวจะปล่อยผ่าน แต่ *ความหมาย* — "บัญชีธนาคารของเราเปลี่ยนแล้ว โอนวันนี้ ห้ามโทรมายืนยัน" — คือรูปแบบการฉ้อโกงคลาสสิก และนั่นคือการตัดสินของ Jev

```bash
.venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab02_image_to_decision.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · thermostat display (media/thermostat_room1203.png)
   │ room=1203; measured_c=29.1; setpoint_c=24.0; mode=COOL
» code facts  {'room': '1203', 'measured_c': 29.1, 'setpoint_c': 24.0, 'mode': 'COOL', 'deviation_c': 5.1, 'warm_deviation': True}
» Jev readable 0.97
→ proposal: HVAC ticket for room 1203 — +5.1 °C above setpoint (computed in code, not by any model)

▣ STEP 2 · maintenance photo (media/leak_under_ahu.jpg)
   │ … A significant puddle of dark liquid has accumulated on the concrete floor …
   │ Unsure: I cannot tell whether the leaking liquid is water condensation, oil, or another type of mechanical fluid.
» queue                  choice  → water_leak   (confidence 1.00)
» safety                 noul    0.88  █████████████████████░░░  likely YES
» severity               score   3.00 on 0…3   (confidence 1.00)
» hedged                 noul    0.94  ███████████████████████░  likely YES
→ proposal: {'route': 'safety_review', 'alert': True, 'reasons': ['safety signal 0.88 ≥ 0.30'], 'execute': False}

▣ STEP 3 · supplier invoice (media/invoice_bank_change.png)
» code math   {'ok': True, 'subtotal': 96500.0, 'vat': 6755.0, 'total': 103255.0, 'why': 'items + 7% VAT = total'}
» suspicious             noul    0.98  ████████████████████████  likely YES
» category               choice  → supplier_invoice   (confidence 0.98)
→ proposal: {'route': 'finance_review', 'alert': False, 'reasons': ['suspicious payment request → finance review; never pay from this document'], 'execute': False}
```

✓ Checkpoint: คุณชี้ได้ว่าบรรทัดไหนที่ *โค้ด* — ไม่ใช่โมเดล — คำนวณค่า +5.1 °C และอธิบายได้ว่าทำไมใบแจ้งหนี้ผ่านการตรวจตัวเลขแต่ยังต้องส่งให้ฝ่ายการเงินตรวจสอบ

## 2 · เปลี่ยนโมเดลมองเห็น — policy ไม่ต้องเปลี่ยน

ขั้นที่ 4 ของแล็บ 02 ส่งรูปเทอร์โมสแตตรูปเดียวกันให้ผู้ให้บริการด้านภาพ **ทุกราย** ที่คุณมี key แยกคำตอบแต่ละรายด้วย regex ตัวเดียวกัน แล้วเปรียบเทียบกันในโค้ด:

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 4 · same thermostat, every vision provider with a key
│ provider  model                    room  measured  setpoint  mode  time   source
│ gemini    gemini-3.5-flash         1203  29.1      24.0      COOL  2.1 s  live
│ chatgpt   gpt-5.4-mini-2026-03-17  1203  29.1      24.0      COOL  1.7 s  live
│ claude    claude-sonnet-5          1203  29.1      24.0      COOL  1.9 s  live
│ ollama    gemma4:latest            1203  29.1      24.0      FAN   0.7 s  live
═ 4 provider(s) parsed: they DISAGREE on at least one fact → send to a person (compared in code).
  The Jev questions and the policy did not change when the vision model did.
```

โมเดลบนคลาวด์สามตัวอ่านโหมดเป็น `COOL` ส่วน `gemma4` บนเครื่องอ่านเป็น `FAN` (หน้าจอแสดงทั้ง "COOL" และ "FAN AUTO") และในอีกรอบหนึ่ง โมเดลบนเครื่องตอบกลับมา **ว่างเปล่า** ตอนเริ่มเครื่องครั้งแรก (cold start ประมาณ 12 วินาที) — โค้ดถือว่าคำตอบที่แยกข้อมูลไม่ได้คือ *อ่านไม่ได้* แทนที่จะเดา การเปรียบเทียบโมเดลรับรู้สองตัวในโค้ดเป็นการตรวจที่ถูกและแข็งแรง ส่วน Jev และ policy ยังเหมือนเดิมทุกประการไม่ว่าจะใช้ตัวไหน

✓ Checkpoint: คุณรันขั้นที่ 4 แล้ว และบอกได้ว่าโค้ดทำอะไรเมื่อคำตอบของผู้ให้บริการรายหนึ่งแยกข้อมูลไม่ได้

## 3 · เสียง → ticket → คำตอบแบบเสียงพูด

ข้อความเสียงสังเคราะห์สามชิ้น:

![ข้อความเสียงภาษาอังกฤษ — ห้อง 1507 ห้องน้ำน้ำท่วม (สังเคราะห์)](media/voicemail_flood_en.wav)

![ข้อความเสียงภาษาไทย — ห้อง 1203 แอร์ไม่เย็น มีกลิ่นไหม้ (สังเคราะห์)](media/voicemail_burning_th.wav)

![ข้อความเสียงภาษาอังกฤษ — ขอผ้าเช็ดตัวที่ห้อง 812 ไม่รีบ (สังเคราะห์)](media/request_towels_en.wav)

แล็บ 03 ถอดเสียงแต่ละชิ้น ให้ **โค้ด** หาภาษาและหมายเลขห้อง ให้ **Jev** ตัดสิน department · urgency · safety ใช้ policy ตัวเดียวกับภาพ และสุดท้าย **พูด** คำตอบจากแม่แบบที่อนุมัติแล้ว — บันทึกไว้ใต้ `.runs/` ให้คุณเปิดฟังได้

ช่วงสำคัญคือข้อความเสียงภาษาไทย ผู้เข้าพักพูดชัดเจนว่า *ห้อง 1203* และระบบโทรศัพท์ (caller-ID) ก็บอกว่า 1203 แต่โมเดลถอดเสียงเขียนว่า *ห้องหนึ่งสองสองสาม* — **1223** — และโมเดลถอดเสียง *ตัวที่สอง* (Gemini) ก็เขียนว่า **1223** เช่นกัน โมเดลสองตัวเห็นตรงกันไม่ได้แปลว่าถูก

```bash
.venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab03_voice_to_ticket.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 2 · media/voicemail_burning_th.wav
   │ สวัสดีค่ะ ห้องหนึ่งสองสองสามนะคะ
   │ แอร์ไม่เย็นเลยตั้งแต่มาคืนแล้วตอนนี้มีกลิ่นไหม้ออกมาจากช่องแอร์ด้วย ช่วยส่งช่างมาดูด่วนเลยนะคะ
◆ hear · chatgpt · gpt-4o-mini-transcribe · LIVE · 121 in / 58 out tok · 2454 ms
» code facts  language=th · room heard=1223 · caller-ID room=1203  ⚠ MISMATCH
» department             choice  → hvac   (confidence 1.00)
» urgency                score   2.97 on 0…3   (confidence 0.97)
» safety                 noul    0.93  ██████████████████████░░  likely YES
→ ticket {"room": "1203", "team": "hvac", "urgency": 2.97, "route": "safety_review", "alert": true, "reasons": ["safety signal 0.93 ≥ 0.30", "transcript says room 1223 but caller-ID says 1203 → confirm"], "execute": false}
🎧 saved → week24_typed_decisions_jev/13_multimodal_jev/.runs/reply_voicemail_burning_th.wav   “ขอบคุณสำหรับข้อความเกี่ยวกับห้อง 1203 ทีมช่างแอร์ได้รับแจ้งแล้ว เจ้าหน้าที่จะติดต่อกลับโดยเร็วที่สุด”
…
▣ STEP 4 · a second transcriber on the Thai voicemail
   │ สวัสดีค่ะ ห้อง 1223 นะคะ แอร์ไม่เย็นเลยตั้งแต่เมื่อคืน …
» code: room heard by gemini = 1223 · caller-ID = 1203
▣ STEP 5 · summary
│ voicemail                 lang  room heard  caller-ID  team          safety  route
│ voicemail_flood_en.wav    en    1507        1507       plumbing      0.97    safety_review
│ voicemail_burning_th.wav  th    1223        1203       hvac          0.93    safety_review
│ request_towels_en.wav     en    812         812        housekeeping  0.02    housekeeping
```

กฎสี่ข้อที่แล็บนี้เขียนไว้ในโค้ด:

- **ข้อเท็จจริงที่ต้องแม่นยำมาจากระบบที่เชื่อถือได้** หมายเลขห้องใน ticket มาจาก caller-ID ส่วนตัวเลขใน transcript มีไว้เปรียบเทียบเท่านั้น และถ้าไม่ตรงกันจะขอให้คนยืนยัน
- **Transcript มีไว้สำหรับความหมาย** แม้ตัวเลขจะผิดไปหนึ่งหลัก Jev ก็ยังได้ department (hvac) และอันตราย (กลิ่นไหม้, safety 0.93) ถูกต้อง — ความหมายรอดจากความผิดพลาดเล็กน้อยของการถอดเสียง แต่ข้อเท็จจริงที่ต้องแม่นยำไม่รอด
- **ทำความสะอาดก่อนตัดสิน** โมเดลถอดเสียงบางตัวเติม timestamp (`00:00:`) หรือแยกคำภาษาไทยด้วยช่องว่าง — `clean_transcript()` จัดการกรณีแรก ส่วน Jev รับมือกับกรณีที่สองได้
- **คำตอบมาจากแม่แบบที่อนุมัติแล้ว** คำตอบแบบเสียงพูดบอกว่า "เจ้าหน้าที่จะติดต่อกลับ" — ไม่เคยสัญญาเวลาหรือการซ่อม และใช้หมายเลขห้องที่ *เชื่อถือได้* เสมอ

✓ Checkpoint: คุณเปิดฟังคำตอบหนึ่งไฟล์จาก `.runs/` แล้ว และอธิบายได้ว่าทำไม ticket จึงเป็นห้อง 1203 ทั้งที่ transcript ทั้งสองบอกว่า 1223

## 4 · วิดีโอ → การแจ้งเตือน

![คลิป CCTV ห้องเครื่อง B2 — มีหมอกควันปรากฏหลังประมาณ 4 วินาที (สังเคราะห์)](media/cctv_plantroom.mp4)

แล็บ 04 ขอให้โมเดลวิดีโอส่งรายการเหตุการณ์เป็น JSON แยกข้อมูลด้วย **โค้ด** จากนั้นเรียก Jev **ครั้งเดียว** โดยมี hazard Noul หนึ่งข้อและ kind Choice หนึ่งข้อ **ต่อหนึ่งเหตุการณ์** (speculative fan-out จากแล็บ 03) แล้วโค้ดแจ้งเตือนจากเหตุการณ์อันตรายที่เกิดเร็วที่สุด

บทเรียนอยู่ที่ถ้อยคำ คลิปแสดงหมอกสีขาวลอยขึ้นจากมอเตอร์ปั๊ม โมเดลวิดีโอเรียกมันว่า **"steam (ไอน้ำ)"** ในรอบหนึ่ง และ **"vapor or smoke (ไอหรือควัน)"** ในอีกรอบ กฎความปลอดภัยที่ทำงานเฉพาะเมื่อเจอคำว่า *smoke* จะพลาดรอบแรก ลองดู:

```jev
{
  "state": {"events": ["Steam begins leaking from the blue compressor unit on the left", "Vapor or smoke starts venting from the blue compressor unit on the left", "Empty plant room operating under normal conditions"]},
  "questions": {
    "hazard_0": {"type": "noul", "instructions": "Does `events[0]` describe smoke, steam, haze, fire, sparks or leaking fluid that could be dangerous in a plant room?"},
    "hazard_1": {"type": "noul", "instructions": "Does `events[1]` describe smoke, steam, haze, fire, sparks or leaking fluid that could be dangerous in a plant room?"},
    "hazard_2": {"type": "noul", "instructions": "Does `events[2]` describe smoke, steam, haze, fire, sparks or leaking fluid that could be dangerous in a plant room?"},
    "smoke_only_0": {"type": "noul", "instructions": "Does `events[0]` mention smoke or fire?"}
  }
}
```

ทดลอง: เปรียบเทียบ `hazard_0` กับ `smoke_only_0` คำถามแคบ ๆ จะอ่านคำแบบตรงตัว (แล็บ 03) และตอบว่า *ไม่* สำหรับ "steam" ให้เขียนคำถามถึงอันตราย ไม่ใช่ถึงคำคำเดียว

```bash
.venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab04_video_to_alert.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · plain prompt: list the events
│ t (s)  event (from the video model)                      could also be  hazard  Jev kind
│ 0      Empty plant room operating under normal conditi…                 0.03    normal
│ 4      Steam begins leaking from the blue compressor u…                 0.98    steam_or_vapour
│ 6      Steam spreads and begins filling the plant room                  0.97    steam_or_vapour
▣ STEP 2 · careful prompt: list events AND the most likely alternative interpretation
│ t (s)  event (from the video model)                      could also be  hazard  Jev kind
│ 0      plant room operating under normal conditions      none           0.04    normal
│ 4      steam begins leaking from the blue chiller comp…  smoke          0.93    steam_or_vapour
│ 6      steam plume thickens and spreads across the cei…  smoke          0.90    steam_or_vapour
▣ STEP 3 · code decides
→ plain: ALERT at t=4s — steam_or_vapour — “Steam begins leaking from the blue compressor unit on the left” → page the duty engineer to look in person
→ careful: ALERT at t=4s — steam_or_vapour — “steam begins leaking from the blue chiller compressor” → page the duty engineer to look in person
```

**Prompt แบบระมัดระวัง (careful prompt)** ขอให้โมเดลวิดีโอบอกการตีความทางเลือกที่เป็นไปได้มากที่สุด ("could also be: smoke") ซึ่งไม่มีค่าใช้จ่ายเพิ่ม และส่งต่อความไม่แน่นอนของโมเดลให้ผู้ตรวจสอบที่เป็นคน ระบบสัญญาณเตือนไฟไหม้ยังทำงานแยกจาก pipeline นี้ — Jev เพิ่มการแจ้งเตือนได้ แต่ไม่มีวันระงับการแจ้งเตือน

✓ Checkpoint: คุณอธิบายได้ว่าทำไมคำถามเรื่องอันตรายต้องระบุทั้ง smoke, steam *และ* haze และ `could_also_be` ช่วยอะไรเพิ่ม

## 5 · Jev เป็นผู้เฝ้าประตูของการรับรู้ที่มีราคาแพง

การรับรู้คือส่วนที่ช้าและเสียเงิน: หลายวินาทีต่อภาพ และนานกว่านั้นสำหรับวิดีโอ ไฟล์แนบจำนวนมากไม่จำเป็นต้องผ่านขั้นนี้ — เช่นคำขอผ้าเช็ดตัวที่มีข้อความเสียงแนบมา หรือจดหมายข่าวจากซัพพลายเออร์ที่มีรูปแนบ แล็บ 05 ส่ง **เฉพาะข้อความรายงาน** ให้ Jev ใน **การเรียกครั้งเดียว** (Noul หนึ่งข้อต่อหนึ่งรายงาน) แล้วโค้ดส่งไปรับรู้เฉพาะไฟล์แนบที่ Jev ชี้ว่าควรดู

```bash
.venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab05_perception_gatekeeper.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 2 · code perceives ONLY what Jev flagged (threshold 0.5)
│ #  report                                            look?  perception      what it saw
│ 0  Housekeeping: water on the floor next to AHU-2 …  0.93   image → gemini  A large, grey HVAC unit or mechanical enclosure
│ 1  Front desk: guest in 812 asked for two extra to…  0.14   skip
│ 2  CCTV motion alert: CAM-03 plant room B2, 10-sec…  0.83   video → gemini  Yes, around the five-second mark, a sudden and h
│ 3  Accounts: routine supplier newsletter with our …  0.04   skip
│ 4  Guest in 1203 says the room is too warm; they s…  0.84   image → gemini  room=1203; measured_c=29.1; setpoint_c=24.0; mod
▣ STEP 3 · what the gate saved
◆ perception calls made: 3 of 5 · time spent 8.7 s
◆ skipped 2 attachment(s) ≈ 5.8 s of perception avoided (at this run's average of 2.9 s) · Jev gate cost 763 tokens ≈ $0.000032
```

ในระดับโรงแรม — ไฟล์แนบหลายพันชิ้นต่อวัน — ประตูนี้ช่วยประหยัดเงินและเวลาคิวได้จริง ส่วนการเรียก Jev ที่ใช้ตัดสินใจมีค่าใช้จ่ายเพียงเศษเสี้ยวของหนึ่งเซนต์ threshold (เกณฑ์) เป็นของคุณที่จะปรับ: การข้ามเหตุอันตรายหนึ่งครั้งเสียหายมากกว่าการเรียกดูภาพเพิ่มหนึ่งครั้ง จึงควรเอนไปทางดูไว้ก่อน

✓ Checkpoint: คุณบอกได้ว่าไฟล์แนบสองชิ้นไหนถูกข้าม และทำไมการข้ามจึงปลอดภัย

## 6 · ทำไมต้องใช้ Jev ร่วมกับโมเดล multimodal — ประโยชน์ที่ได้

| ประโยชน์ | สิ่งที่เห็นในบทนี้ |
|---|---|
| **ชั้นการตัดสินใจเดียวสำหรับทุกประเภทสื่อ** | `policy()` ตัวเดียวจัดการได้ทั้งรูปถ่าย ใบแจ้งหนี้ ข้อความเสียง และคลิป CCTV |
| **เปลี่ยนโมเดลรับรู้ได้อิสระ** | ผู้ให้บริการด้านภาพสี่ราย คำถาม Jev ชุดเดียว ไม่ต้องแก้ policy |
| **การตัดสินใจที่มีชนิดและทดสอบได้** | โมเดลรับรู้เขียนเป็นย่อหน้า แต่ Jev คืน `water_leak` / 0.88 / 3.0 ที่เขียน unit test ได้ (แบบฝึกหัด) |
| **เร็วและถูก** | ~0.4–0.7 วินาที และ ≈ $0.00003 ต่อการตัดสินใจ เทียบกับ 1–4 วินาที (คลาวด์) หรือ ~12 วินาที (บนเครื่อง) ต่อการรับรู้หนึ่งครั้ง |
| **การเฝ้าประตู (gatekeeping)** | Jev อ่านข้อความที่ถูกก่อน แล้วตัดสินว่าจำเป็นต้องใช้พิกเซลที่แพงหรือไม่ |
| **การตรวจสอบ (verification)** | Jev อ่านความลังเลของโมเดลรับรู้เอง (`hedged`, `could_also_be`) และโค้ดเปรียบเทียบโมเดลรับรู้สองตัว |
| **ความปลอดภัยอยู่ในโค้ด** | threshold การแจ้งเตือนต่ำ ข้อเท็จจริงจากระบบที่เชื่อถือได้ (caller-ID) สำคัญกว่า transcript แม่แบบคำตอบที่อนุมัติแล้ว และ `execute: False` |

สิ่งที่ Jev **ไม่** ทำในบทนี้: มันไม่ดูภาพ (Lab 15 ลองเส้นทางทดลองนั้น) ไม่ฟังเสียง ไม่อ่านตัวเลขจากหน้าจอ และไม่ตรวจเลขคณิต แต่ละอย่างเป็นหน้าที่ของโมเดลรับรู้หรือของโค้ด

✓ Checkpoint: คุณบอกประโยชน์จากตารางได้สามข้อ และชี้ได้ว่าเห็นแต่ละข้อในแล็บไหน

## Labs — รันได้ที่นี่

**labs/lab01_capability_map.py** — ผู้ให้บริการไหนมองเห็น ฟัง พูด หรือดูวิดีโอได้ด้วย key ของคุณ และแต่ละรายเร็วแค่ไหนเมื่อเทียบกับการเรียก Jev

**labs/lab02_image_to_decision.py** — เทอร์โมสแตต รูปน้ำรั่ว และใบแจ้งหนี้ → ข้อความจากโมเดลภาพ → ข้อเท็จจริงจากโค้ด → Jev → policy เดียว แล้วส่งภาพเดียวกันให้ผู้ให้บริการด้านภาพทุกราย

**labs/lab03_voice_to_ticket.py** — ข้อความเสียงภาษาอังกฤษและภาษาไทย → transcript → ตรวจกับ caller-ID → Jev → ticket → คำตอบแบบเสียงพูดบันทึกไว้ที่ `.runs/`

**labs/lab04_video_to_alert.py** — คลิป CCTV → เหตุการณ์แบบ JSON → เรียก Jev แบบ fan-out ครั้งเดียว → การแจ้งเตือน เทียบ prompt แบบธรรมดากับแบบระมัดระวัง (ไอน้ำหรือควัน)

**labs/lab05_perception_gatekeeper.py** — เรียก Jev ครั้งเดียวกับข้อความรายงานห้าชิ้น เพื่อตัดสินว่าไฟล์แนบไหนควรส่งไปรับรู้

ทั้งห้าแล็บรันได้ทั้งแบบ LIVE ด้วย key ของคุณ หรือแบบ DRY จากการบันทึก และจะใช้ผู้ให้บริการที่คุณเลือกใน **LLM picker** ทุกครั้งที่ผู้ให้บริการนั้นมีความสามารถที่ต้องใช้

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 13 — policy เดียวสำหรับทุกประเภทสื่อ** เปิดไฟล์ `week24_typed_decisions_jev/13_multimodal_jev/exercises/ex13_multimodal_policy.py` มี TODO สองข้อ:

1. เขียน `HAZARD_QUESTION` — Noul เกี่ยวกับ `event` ที่นับทั้ง smoke, steam **และ** haze เป็นอันตรายที่เป็นไปได้
2. เขียน `my_policy(signal)` — กฎห้าข้อ: การรับรู้ล้มเหลว → ให้คนตรวจ; safety ≥ 0.30 → แจ้งเตือน; suspicious ≥ 0.50 → ส่งฝ่ายการเงินตรวจ; หมายเลขห้องไม่ตรงกัน → "confirm room"; นอกนั้นใช้ queue และ `execute` เป็น `False` เสมอ

การทดสอบแบบออฟไลน์เจ็ดข้อจะรันก่อนโดยไม่มีค่าใช้จ่าย เมื่อผ่านทั้งหมด ตัวตรวจจะถามคำถาม hazard ของคุณกับ Jev โดยใช้คำบรรยายเหตุการณ์จริงสามข้อที่โมเดลวิดีโอเขียนไว้ในแล็บ 13-4

```bash
.venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/exercises/ex13_multimodal_policy.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ HAZARD_QUESTION is a noul about `event` that names smoke, steam and haze
✓ test 1: human_review
✓ test 2: safety_review + alert
…
✓ test 7: plumbing
▣ asking Jev your hazard question about three real video-model descriptions …
│ video-model event                                 hazard  your policy
│ Steam begins to leak from the blue industrial e…  0.96    safety_review
│ Vapor or smoke starts venting from the blue com…  0.94    safety_review
│ Normal operating conditions in the plant room     0.04    plant_room
```

<details><summary>คำใบ้ — ลำดับของกฎมีความสำคัญ</summary>

ตรวจ `perception_ok` ก่อนและ return ทันที จากนั้นตัดสินเรื่องการแจ้งเตือน (safety) ต่อด้วยเรื่องการเงิน แล้วจึงตรวจหมายเลขห้อง — และตั้ง `route = "finance_review"` เฉพาะเมื่อการแจ้งเตือนยังไม่ได้ทำให้เป็น `safety_review`

</details>

<details><summary>ท้าทายเพิ่ม — ลองใช้สื่อของคุณเอง</summary>

วางรูปสังเคราะห์ของคุณเองไว้ใน `media/` แล้วเรียก `mk.see(mk.pick("see"), "…/your.jpg", P.SEE_SCENE)` จาก Python shell จากนั้นรัน `P.SCENE_QUESTIONS` กับข้อความที่ได้ ห้ามใช้รูป เสียง หรือภาพ CCTV ของผู้เข้าพักจริงโดยไม่ได้รับอนุมัติ — การเรียกโมเดลรับรู้จะส่งสื่อไปยังผู้ให้บริการ

</details>

✓ Checkpoint: การทดสอบทั้งเจ็ดข้อผ่าน และทั้งถ้อยคำ "steam" และ "smoke" ไปถึง `safety_review`

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `no provider with a key can 'hear'` | เพิ่ม key ของ OpenAI หรือ Gemini ผ่าน 🔑 Keys — Claude มองเห็นได้แต่ฟังไม่ได้ |
| การเรียกโมเดลรับรู้ใช้เวลาหลายนาที | ผู้ให้บริการช้าเป็นบางครั้ง (เราเคยเจอการเรียก TTS ที่ใช้ 2 นาที) ให้รันใหม่ หรือเปลี่ยนผู้ให้บริการใน LLM picker |
| `gemma4` บนเครื่องตอบว่างเปล่าหรือแปลก ๆ | การเรียกครั้งแรกหลังโหลดโมเดลช้าและอาจว่างเปล่า โค้ดจะถือว่าอ่านไม่ได้ — รันใหม่หรือใช้โมเดลบนคลาวด์ |
| Transcript มี `00:00:` หรือมีช่องว่างระหว่างคำภาษาไทย | เป็นเรื่องปกติของโมเดลถอดเสียงบางตัว `clean_transcript()` ลบ timestamp ออก และ Jev อ่านภาษาไทยที่มีช่องว่างได้ |
| หมายเลขห้องใน transcript ผิด | เกิดขึ้นได้บ้าง — นี่คือเหตุผลที่ ticket ใช้ caller-ID และแจ้งว่าไม่ตรงกัน |
| `◈ DRY — no recorded … answer` | คุณเปลี่ยน prompt ไฟล์ หรือผู้ให้บริการ ให้สลับเป็น ⚡ Live หรือคืนค่าที่เปลี่ยน |
| `watch` ล้มเหลวกับวิดีโอขนาดใหญ่ | วิดีโอแบบ inline มีขนาดจำกัด ให้ใช้คลิปสั้น ๆ (คลิปนี้ยาว 10 วินาที ขนาด 210 KB) |

## Next — ต่อไป

ไปต่อที่ [แล็บ 14 — เซสชันแบบสดและเรียลไทม์](../14_live_realtime_jev/TUTORIAL.th.md): การโทรด้วยเสียงแบบสตรีมมิงกับ `gemini-3.8-live` หรือ `gpt-realtime-2.1-mini` โดยมี Jev เป็นผู้ตัดสินในระดับเสี้ยววินาทีในทุก ๆ turn
