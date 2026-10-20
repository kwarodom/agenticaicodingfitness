# ▶ Jev Lab 12 — Capstone: copilot รับคำขอของแขกในโรงแรมอัจฉริยะ

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมี type ด้วย Jev ทุกอย่างจากแล็บ 01–11 รวมอยู่ในระบบเล็ก ๆ ที่มีรูปร่างเหมือนของจริง ข้อความจากแขก (EN / TH / ภาษาผสม) กลายเป็น **ข้อเสนอ ticket** ที่ถูกจัดเส้นทาง จัดลำดับความสำคัญ และตรวจความปลอดภัยแล้ว โดยไม่มีอะไรถูกลงมือทำเองเลย

**พูดง่าย ๆ:** แขกเขียนมาว่า *"The toilet in 1507 is overflowing!"* Jev ตอบคำถามสั้น ๆ ห้าข้อเกี่ยวกับข้อความนี้: ทีมไหน รุนแรงแค่ไหน มีอันตรายไหม ต้องให้คนดูไหม มีการบอกหมายเลขห้องหรือไม่ จากนั้นโค้ด Python ของคุณทำส่วนที่ต้องแม่นยำเอง (หา `1507` ด้วย regex และตรวจว่าเป็นภาษาไทยหรืออังกฤษ) ใช้กฎที่เขียนไว้ง่าย ๆ แล้วพิมพ์ ticket ออกมาให้คนอนุมัติ คุณจะได้สร้างเส้นทางทั้งหมดนี้ในแล็บนี้

**สองคำที่คุณจะเจอ:** *uncertainty gate* คือกฎอย่าง "ถ้า Jev มั่นใจเรื่องทีมไม่ถึง 70% ให้ส่งให้คนตัดสิน" (แล็บนี้ใช้ 0.70) *shadow mode* (รันคู่ขนานเงียบ ๆ) คือการรันระบบใหม่ข้างระบบเดิม บันทึกจุดที่ทั้งสองเห็นไม่ตรงกัน โดยไม่เปลี่ยนอะไรสำหรับผู้ใช้จริง

**สิ่งที่คุณจะได้ลงมือทำ**
- ออกแบบ request แบบ batch ของ Jev หนึ่งรายการ ที่มีการตัดสินอิสระห้าข้อ
- ให้งานที่ต้องแม่นยำอยู่ในโค้ด: การตรวจภาษาและการดึงหมายเลขห้อง
- เขียน policy ที่มี safety override, uncertainty gate และลำดับความสำคัญ แล้วทดสอบแบบออฟไลน์
- สร้าง ticket แบบ JSON พร้อมร่างคำตอบจาก template โดยมี `execute: False`
- รัน copilot ตัวใหม่ใน **shadow mode** ข้าง router แบบ keyword แล้วอ่านจุดที่เห็นไม่ตรงกัน
- สร้างทั้งหมดด้วยตัวเองในแบบฝึกหัด capstone

**Time** ~60 นาที · **Difficulty** ระดับกลาง → ขั้นสูง · **Cost** ≈ $0.001 แบบ live · $0 แบบ dry

## 0 · การออกแบบในหน้าเดียว

แขกเขียนมาว่า: *"There's a burning smell coming from the air conditioner in 1507!"* แต่ละส่วนทำหน้าที่ดังนี้:

```text
guest message
  ├─ CODE  detect_language()   Thai Unicode block?  → "en" / "th" / "mixed"      exact, free
  ├─ CODE  extract_room()      regex                → "1507"                     exact, free
  ├─ JEV   ONE batched call    department · severity · safety · needs_human · mentions_room
  ├─ CODE  policy()            safety ≥ 0.20 → duty manager, P0, human approval
  │                            unknown / torn → front-desk review
  │                            severity → P1 / P2 / P3
  └─ CODE  build_ticket()      JSON ticket + reply DRAFT from approved templates · execute: False
```

กฎเบื้องหลังทุกบรรทัดคือกฎเดียวกับที่คุณใช้มาตั้งแต่แล็บ 01: **โค้ดสำหรับข้อเท็จจริง, Jev สำหรับการตัดสิน, โค้ดสำหรับ policy, คนสำหรับการลงมือทำ**

- **ภาษา** ไม่ใช่การตัดสิน: อักษรไทยอยู่ในช่วง Unicode ที่รู้แน่นอน regex บรรทัดเดียวจึงแม่นยำและไม่เสียค่าใช้จ่าย
- **หมายเลขห้อง** เป็น *ค่าที่ต้องคัดลอก* ไม่ใช่สิ่งที่ต้องเดา regex เป็นคนคัดลอก Jev ตอบแค่คำถามเชิง *ความหมาย* ว่า "ข้อความบอกหมายเลขห้องหรือไม่" แล้วโค้ดของคุณใช้คำตอบนั้นตรวจทานซ้ำ
- **คำตอบถึงแขก** มาจาก template ที่อนุมัติแล้วซึ่งโค้ดเป็นคนเติมค่า ตรงนี้คือจุดที่ LLM แบบ generative *อาจ* ช่วยปรับถ้อยคำให้เป็นส่วนตัวในภายหลัง โดยยังเป็นร่างที่คนต้องอนุมัติ

✓ Checkpoint: สำหรับแต่ละขั้นในห้าขั้น คุณบอกได้ว่าทำไมมันเป็นงานของโค้ด หรือทำไมเป็นงานของ Jev

## 1 · คำถามห้าข้อ — ลองแบบ live

คำถามทั้งห้าข้อตัดสิน `state` เดียวกัน จึงอยู่ใน request **เดียว** คุณจ่ายค่า state ครั้งเดียว คำถามถูกตอบแบบขนาน และไม่มีคำถามไหนเห็นคำตอบของข้ออื่น

```jev
{
  "state": {"message": "There's a burning smell coming from the air conditioner in 1507!"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which hotel team should handle the request in `message`?",
      "criteria": {
        "hvac": "Room too hot or too cold, air-conditioning, ventilation, a noisy or leaking AC unit",
        "housekeeping": "Cleaning, making up the room, towels, linen, pillows, toiletries, amenities",
        "maintenance": "Plumbing, toilet, shower, hot water, electrical, lights, TV, door lock, broken furniture",
        "front_desk": "Bookings, billing, check-out, room keys, general information",
        "food_beverage": "Room service, restaurant, breakfast, minibar, food orders",
        "unknown": "No identifiable request, or none of the teams above"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How much does the problem in `message` affect the guest's stay?",
      "criteria": ["No problem: a question or a simple request", "Minor inconvenience", "A service failure that disrupts the stay", "Possible danger to people or property"]
    },
    "safety": {
      "type": "noul",
      "instructions": "Does `message` explicitly mention smoke, fire, a burning smell, sparks, gas, water flooding the floor, electric shock or an injury?"
    },
    "needs_human": {
      "type": "noul",
      "instructions": "Does `message` ask for a manager or a person, express strong anger, or ask for a refund or compensation?"
    },
    "mentions_room": {
      "type": "noul",
      "instructions": "Does `message` state the guest's room number?"
    }
  }
}
```

ทีนี้ลองเปลี่ยนข้อความทีละอย่าง แล้วดูว่าคำตอบไหนขยับ:

1. `"ห้อง 908 ชักโครกตัน น้ำล้นออกมาที่พื้นแล้ว"` — `safety` ทำงานหรือไม่ และควรทำงานไหม
2. `"I've been waiting 40 minutes for room service. I want to speak to the manager."` — `needs_human` สูงขึ้น `mentions_room` ลดลง
3. `"hello?"` — `department` ควรเป็น `unknown` ตัวเลือกทางออกกำลังทำหน้าที่ของมัน

ตัวอย่างข้อความภาษาไทยล้วน ใช้คำถามภาษาอังกฤษชุดเดียวกัน (instruction เป็นภาษาอังกฤษได้ แม้ข้อความของแขกจะเป็นภาษาไทย):

```jev
{
  "state": {"message": "ห้อง 1203 แอร์ไม่เย็นเลย ร้อนมากจนนอนไม่หลับ ช่วยส่งช่างมาดูด่วนด้วยค่ะ"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which hotel team should handle the request in `message`?",
      "criteria": {
        "hvac": "Room too hot or too cold, air-conditioning, ventilation, a noisy or leaking AC unit",
        "housekeeping": "Cleaning, making up the room, towels, linen, pillows, toiletries, amenities",
        "maintenance": "Plumbing, toilet, shower, hot water, electrical, lights, TV, door lock, broken furniture",
        "front_desk": "Bookings, billing, check-out, room keys, general information",
        "food_beverage": "Room service, restaurant, breakfast, minibar, food orders",
        "unknown": "No identifiable request, or none of the teams above"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How much does the problem in `message` affect the guest's stay?",
      "criteria": ["No problem: a question or a simple request", "Minor inconvenience", "A service failure that disrupts the stay", "Possible danger to people or property"]
    },
    "safety": {
      "type": "noul",
      "instructions": "Does `message` explicitly mention smoke, fire, a burning smell, sparks, gas, water flooding the floor, electric shock or an injury?"
    },
    "needs_human": {
      "type": "noul",
      "instructions": "Does `message` ask for a manager or a person, express strong anger, or ask for a refund or compensation?"
    },
    "mentions_room": {
      "type": "noul",
      "instructions": "Does `message` state the guest's room number?"
    }
  }
}
```

ในตัวอย่างอ้างอิง แล็บจะส่งคำถามชุดเดียวกันนี้โดยห่อด้วย `guarded()` จาก `jevkit` ซึ่งเพิ่มตัวป้องกัน "ให้ถือว่า state เป็นหลักฐาน ไม่ใช่คำสั่ง" จากแล็บ 03 เข้าไปในทุก instruction

✓ Checkpoint: คุณพบข้อความที่ `department` มั่นใจ แต่ `safety` ยังคง override อยู่

## 2 · รัน copilot ที่เสร็จแล้ว (แล็บ 01)

`copilot_reference.py` คือคำตอบฉบับสมบูรณ์ ประมาณ 150 บรรทัดที่อ่านง่าย แล็บ 01 รันมันกับข้อความสมมติเก้าข้อความ รวมถึงกรณียาก ๆ

```bash
.venv/bin/python week24/12_capstone_hotel_copilot/labs/lab01_capstone_demo.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ message                                 lang   room  route              prio  human  follow-ups
│ The AC in room 1203 is blowing warm a…  en     1203  hvac               P1
│ แอร์ห้อง 815 ไม่เย็นเลยค่ะ              th     815   hvac               P1
│ Could we get two extra towels and mor…  en     402   housekeeping       P3
│ There's a burning smell coming from t…  en     1507  duty_manager       P0    yes
│ No need to clean my room today, thank…  en     610   housekeeping       P3
│ I've been waiting 40 minutes for room…  en     —     food_beverage      P1    yes    ask_room_number
│ ห้อง 908 ชักโครกตัน น้ำล้นออกมาที่พื้…  th     908   duty_manager       P0    yes
│ Room 1110 the shower ไม่มีน้ำร้อน       mixed  1110  maintenance        P1
│ hello?                                  en     —     front_desk_review  P3           ask_room_number
…
{
  "ticket_id": "T-ba643575",
  "room": "1507",
  "route": "duty_manager",
  "priority": "P0",
  "requires_human_approval": true,
  "signals": {"department": "hvac", "department_p": 0.93, "severity": 2.99, "safety": 0.98, …},
  "reply_draft": "Our duty manager has been alerted. If you feel unsafe, leave the room and dial 0 for reception.",
  "execute": false
}
◆ 9 calls · 7,252 input tokens · $0.000305 total
```

สิ่งที่ควรสังเกต:

- **กลิ่นไหม้** ยังถูกจัดเป็น `hvac` (p = 0.93) ซึ่งถูกต้อง เพราะมัน *คือ* เครื่องปรับอากาศจริง ๆ แต่ safety override ก็ยังส่งไปให้ duty manager อยู่ดี คำตอบเรื่องแผนกถูกเก็บไว้ใน `signals` วิศวกรจึงเห็นด้วย
- **ชักโครกน้ำล้นภาษาไทย** ได้ `safety = 0.94` เพราะน้ำบนพื้นเสี่ยงทั้งลื่นล้มและไฟฟ้า ตั้งเกณฑ์ไว้ต่ำโดยเจตนา (0.20): สัญญาณเตือนผิดเสียแค่โทรศัพท์หนึ่งสาย แต่พลาดไปจะเสียมากกว่านั้นมาก โรงแรมของคุณอาจต้องการเส้นทาง `flooding` แยก นั่นคือการเปลี่ยน policy ใน `if` เดียว ไม่ต้องเปลี่ยนโมเดล
- **"No need to clean my room today — 610"** เป็นประโยคปฏิเสธ มันยังไปที่ housekeeping ในระดับ P3 ซึ่งถูกต้อง: housekeeping ต้องรู้ว่าไม่ต้องเข้าไป
- **"40 minutes"** และ **"29 degrees"** *ไม่* ถูกนับเป็นหมายเลขห้อง regex ตัดระยะเวลา อุณหภูมิ จำนวนเงิน และเวลาออก
- **แขกที่โกรธ** ไม่ได้บอกหมายเลขห้อง ticket จึงมีงานติดตาม `ask_room_number` และ `mentions_room = 0.02` ของ Jev ก็เห็นด้วย

✓ Checkpoint: คุณรันแล็บ 01 แล้ว และอธิบายได้ว่าทำไม ticket เรื่องกลิ่นไหม้ต้องมี `requires_human_approval: true` แม้ทุกสัญญาณจะมั่นใจ

## 3 · โค้ดกับ Jev ตรวจทานกันและกัน

เมื่อสองส่วนที่เป็นอิสระต่อกันเห็นไม่ตรงกัน ให้ติดธงกรณีนั้นไว้ อย่าเลือกฝ่ายชนะเงียบ ๆ `build_ticket()` เปรียบเทียบผลของ regex กับ `mentions_room` ของ Jev:

| regex เจอหมายเลขห้อง? | `mentions_room` ของ Jev | งานติดตาม |
|---|---|---|
| เจอ | ≥ 0.5 | ไม่มี — ทั้งสองเห็นตรงกัน |
| เจอ | < 0.5 | `check_room_number` — ตัวเลขนั้นอาจเป็นอย่างอื่น |
| ไม่เจอ | ≥ 0.5 | `confirm_room_number` — บอกไว้ในรูปแบบที่ regex หาไม่เจอ ("room twelve-oh-three") |
| ไม่เจอ | < 0.5 | `ask_room_number` |

ในการรัน live ของแบบฝึกหัด capstone ข้อความ *"There's smoke coming out of the bathroom fan in 1507!"* ได้ผลว่า regex เจอ `1507` แต่ Jev อ่านว่า **ไม่มี** การบอกหมายเลขห้อง ticket จึงมี `check_room_number` ตัวเลขเดี่ยว ๆ หลังคำว่า "in" น่าจะเป็นห้อง แต่ก็ไม่แน่นอน การตรวจทานซ้ำนำความสงสัยนั้นไปวางไว้ในที่ที่คนจะมองเห็น

✓ Checkpoint: คุณยกตัวอย่างข้อความได้หนึ่งข้อความสำหรับแต่ละแถวทั้งสี่ในตาราง

## 4 · Policy คือโค้ด — ทดสอบได้โดยไม่ต้องใช้โมเดล

`policy()` ใน `copilot_reference.py` ไม่มีโมเดลอยู่ข้างใน มันอ่านคำตอบแบบมี type แล้วใช้กฎของคุณตามลำดับ:

```python
if answers["safety"]["noul"] >= 0.20:                     # a) safety beats everything
    return {"route": "duty_manager", "priority": "P0", "requires_human_approval": True, ...}
if dept["choice"] == "unknown":                           # b) nothing to route → ask the guest
    route = "front_desk_review"
elif top < 0.70 or top - second < 0.20:                   #    torn between teams → a person decides
    route = "front_desk_review"
else:
    route = dept["choice"]
priority = "P1" if sev >= 1.5 else "P2" if sev >= 0.5 else "P3"   # c) severity → priority
escalate_to_human = answers["needs_human"]["noul"] >= 0.5          # d) people who asked for people
```

เพราะมันอ่านแค่คำตอบแบบมี type คุณจึงทดสอบได้ด้วย **dict คำตอบที่เขียนเอง**: ไม่เรียก API ไม่เสียเงิน และทำซ้ำได้เสมอ แบบฝึกหัด capstone ทำแบบนี้ด้วย fixture หกชุด รวมถึง "มั่นใจแต่เป็น unknown" และ "ลังเลระหว่างสองทีม" (0.55 vs 0.45)

threshold (0.20 / 0.70 / 0.20) เป็น **ค่าตัวอย่างที่ยังไม่ได้ calibrate** ในโรงแรมจริง คุณต้องเลือกค่าเหล่านี้บน calibration split (แล็บ 09) แยกตามภาษา แล้วล็อกไว้

✓ Checkpoint: คุณบอกได้ว่ากฎข้อไหนทำงานก่อน เมื่อข้อความทั้ง "โกรธ" และมี "ควัน" — และทำไมลำดับนั้นถูกต้อง

## 5 · Shadow mode — วางข้าง router เดิม โดยไม่เปลี่ยนอะไร (แล็บ 02)

ก่อนที่ router ใหม่จะแตะต้องแขกจริง มันต้องรัน **อยู่ในเงา** ระบบปัจจุบันยังจัดเส้นทางต่อไป และคุณแค่บันทึกว่าระบบใหม่ *จะ* ทำอะไร ในที่นี้ระบบเดิมคือ router แบบ keyword ทั่วไปที่ใช้ keyword แรกที่เจอ

```bash
.venv/bin/python week24/12_capstone_hotel_copilot/labs/lab02_shadow_mode.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ message                                   human              keyword (live)     Jev (shadow)          Jev ok
│ There's a burning smell coming from the…  duty_manager       hvac               duty_manager       ≠  ✓
│ ห้อง 908 ชักโครกตัน น้ำล้นออกมาที่พื้นแ…  duty_manager       maintenance        duty_manager       ≠  ✓
│ It's freezing cold in here and the room…  maintenance        hvac               maintenance        ≠  ✓
│ Is breakfast included in my booking?      front_desk         food_beverage      front_desk         ≠  ✓
│ The shower is fine now, thanks. But my …  front_desk         maintenance        front_desk         ≠  ✓
◆ agreement keyword vs Jev : 8/13
◆ keyword router correct   : 8/13
◆ Jev copilot correct      : 13/13
```

**สงสัยผลลัพธ์ของตัวเองไว้ก่อน** คนเขียนข้อความ 13 ข้อนี้ *และ* รายการ keyword คือคนเดียวกัน ผลนี้จึงเข้าข้างตัวเอง การอ่านอย่างซื่อตรงคือ: แถวที่เป็น ≠ แสดง *ชนิด* ของความล้มเหลวที่ router แบบ keyword มี — "breakfast" ในคำถามเรื่องการจอง, "shower" ใน "the shower is fine now", และไม่มีแนวคิดเรื่องความปลอดภัยเลย มันไม่ใช่หลักฐานว่า Jev แม่นยำ 100% การจะเลื่อนขั้นต้องใช้ชุด held-out ขนาดใหญ่จาก traffic จริง (ที่ลบข้อมูลระบุตัวตนแล้ว) ผลแยกตาม class ทั้ง EN/TH และจำนวนครั้งที่พลาดเรื่องความปลอดภัย

มีแถวหนึ่งที่ถกเถียงได้จริง ๆ: *"It's freezing cold in here and the room is too dark — the lights don't work."* นั่นคือสองปัญหา คือ HVAC และ maintenance `choice` ตัวเดียวต้องเลือกแค่อย่างเดียว เมื่อ label หลายตัวเป็นจริงได้พร้อมกัน ให้ถาม **หนึ่ง `noul` ต่อหนึ่งทีม** (แล็บ 02) แล้วเปิด ticket ให้แต่ละทีม

```jev
{
  "state": {"message": "It's freezing cold in here and the room is too dark — the lights don't work. Room 312."},
  "questions": {
    "for_hvac": {"type": "noul", "instructions": "Does `message` report a room temperature or air-conditioning problem?"},
    "for_maintenance": {"type": "noul", "instructions": "Does `message` report a broken light, electrical, plumbing or furniture problem?"},
    "for_housekeeping": {"type": "noul", "instructions": "Does `message` ask for cleaning, towels, linen or amenities?"}
  }
}
```

ตอนที่เรารัน: `for_hvac` 0.95, `for_maintenance` 0.97, `for_housekeeping` 0.08 ได้สอง ticket และไม่มีใครต้องตัดสินว่าปัญหาไหน "ชนะ" noul ที่เป็นอิสระต่อกันไม่จำเป็นต้องรวมกันได้ 1 และในกรณีนี้ก็ไม่ควรรวมได้ 1

✓ Checkpoint: คุณอธิบายได้ว่าทำไม 13/13 บนข้อความที่ผู้เขียนแต่งเอง ไม่ใช่เหตุผลที่จะเปลี่ยน router ที่ใช้งานจริง

## 6 · จะพาไปต่อที่ AltoTech ได้อย่างไร

**พูดง่าย ๆ:** สูตรที่คุณเพิ่งสร้าง (Jev ตอบคำถามแบบมี type ไม่กี่ข้อ, โค้ดทำงานที่ต้องแม่นยำและใช้กฎ, คนเป็นผู้อนุมัติ) ไม่ได้ใช้ได้แค่กับโรงแรม แต่ละแถวด้านล่างคือสูตรเดียวกันกับ input ต่างกัน อ่านจากซ้ายไปขวา: อะไรเข้ามา, Jev ตัดสินอะไร และอะไรที่ยังอยู่ในโค้ดธรรมดา


รูปแบบเดียวกันนี้ใช้กับฝั่งอาคารของธุรกิจได้ด้วย เปลี่ยนข้อความและแผนก แต่คงสถาปัตยกรรมไว้:

| Input | สิ่งที่ Jev ตัดสิน | สิ่งที่โค้ดเก็บไว้ |
|---|---|---|
| ข้อความจากแขก (แล็บนี้) | ทีม, ความรุนแรง, ความปลอดภัย, ต้องให้คนดู | หมายเลขห้อง, ภาษา, policy, template |
| บันทึกของผู้ปฏิบัติงาน + telemetry (แล็บ 07) | คิวการตรวจสอบ, การกล่าวถึงความปลอดภัย | ค่าเบี่ยงเบน, ความสดของข้อมูล, flag คุณภาพ — คำนวณทั้งหมด |
| อีเมลจากผู้เช่า (แล็บ 05) | หมวดหมู่, ความเร่งด่วน, ข้อความน่าสงสัย | การยืนยันตัวตนผู้ส่ง, การอนุมัติด้านการเงิน |
| คำถามถึง Copilot (แล็บ 04) | หัวข้อ, ต้องใช้ข้อมูล live, ต้องใช้ผู้เชี่ยวชาญหลายคน | สิทธิ์การเข้าถึง, model registry, การ dispatch |

การ rollout ตามคู่มือต้นฉบับ: **lab → shadow → assisted operations → governed expansion → router integration** พูดง่าย ๆ คือ ฝึกบนข้อมูลปลอมก่อน แล้วรันเงียบ ๆ ข้างระบบจริง จากนั้นให้มัน *เสนอแนะ* ต่อพนักงาน ค่อย ๆ ขยายขอบเขตอย่างระมัดระวัง และให้มันจัดเส้นทางคำขอจริงเป็นขั้นสุดท้าย แต่ละขั้นมีเงื่อนไขการผ่านที่วัดได้ เช่น "accuracy บนข้อความภาษาไทยยังสูงกว่าเกณฑ์ที่ตกลงกันไว้ตลอดหนึ่งเดือน"

✓ Checkpoint: คุณร่างแผนภาพห้ากล่องแบบเดียวกันนี้สำหรับเวิร์กโฟลว์อื่นของ AltoTech ได้หนึ่งเวิร์กโฟลว์

## แล็บ — รันที่นี่

**labs/lab01_capstone_demo.py** — copilot ที่เสร็จแล้วกับข้อความ EN/TH/ภาษาผสมเก้าข้อความ: ตาราง เหตุผล และ ticket ฉบับเต็มหนึ่งใบ

**labs/lab02_shadow_mode.py** — copilot ของ Jev รันในเงาข้าง router แบบ keyword: ความเห็นตรงกัน และทุกจุดที่เห็นไม่ตรงกัน

ทั้งสองใช้ `copilot_reference.py` — เปิดดูได้เลย มันคือเฉลยของแบบฝึกหัด

## ลองทำเอง

**แบบฝึกหัด 12 — สร้าง copilot** เปิด `week24/12_capstone_hotel_copilot/exercises/ex12_build_the_copilot.py` มี TODO สี่ข้อ แต่ละข้อถูกตรวจ **แบบออฟไลน์** ก่อนเรียก API ใด ๆ:

1. **TODO 1 — `QUESTIONS`**: คำถามห้าข้อ พร้อม key ที่ตรงเป๊ะ ตัวเลือกของ department ต้องมีห้าทีมบวก `unknown`
2. **TODO 2 — `extract_room()`**: regex ที่หา `room 1203`, `Room 402.`, `ห้อง 815` และ `— 610.` ที่อยู่โดด ๆ ได้ แต่ไม่จับ `40 minutes`, `29 degrees`, `1500 baht` หรือ `11:30`
3. **TODO 3 — `policy()`**: safety override → uncertainty gate → priority → escalation ทดสอบกับ fixture คำตอบที่เขียนเองหกชุด
4. **TODO 4 — `build_ticket()`**: ข้อเสนอ พร้อม `follow_ups` และ `execute: False`

```bash
.venv/bin/python week24/12_capstone_hotel_copilot/exercises/ex12_build_the_copilot.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ TODO 2 · extract_room()
✓ 'No need to clean today — 610.' → '610'
✓ 'waited 40 minutes' → None
▣ TODO 3 · policy()
✓ low safety still triggers (0.25) → duty_manager P0
✓ torn between two teams → front_desk_review P2
✓ confident but unknown → front_desk_review P3
▣ all offline checks passed — running your copilot LIVE on 5 messages
━━ “There's smoke coming out of the bathroom fan in 1507!”
{"room": "1507", "route": "duty_manager", "priority": "P0", "escalate_to_human": true, "requires_human_approval": true, "follow_ups": ["check_room_number"], "execute": false}
━━ “I want to speak to a manager about my bill. This is unacceptable.”
{"room": null, "route": "front_desk", "priority": "P2", "escalate_to_human": true, "requires_human_approval": false, "follow_ups": ["ask_room_number"], "execute": false}
```

<details><summary>คำใบ้ — TODO 2, regex หมายเลขห้อง</summary>

ทำสองรอบ รอบแรกหารูปแบบที่มี keyword: `(?:\broom|\brm\.?|ห้อง)\s*#?\s*(\d{3,4})\b` พร้อม `re.IGNORECASE` ถ้าไม่เจอ ให้หาตัวเลข 3–4 หลักที่อยู่โดด ๆ ซึ่งไม่ได้เป็นส่วนหนึ่งของตัวเลขที่ยาวกว่าหรือเวลา (`(?<![\d.:])(\d{3,4})(?!\d)(?![.:]\d)`) และไม่ได้ตามด้วยหน่วย (`(?!\s*(?:min|minutes|degrees|°|%|baht|฿))`) ทดสอบกับทุก fixture เพราะ regex ทำให้คุณประหลาดใจเสมอ

</details>

<details><summary>คำใบ้ — TODO 3, gate</summary>

`top, second = top2(answers["department"]["probabilities"])` ตรวจ safety **ก่อน** และ `return` ทันที เพื่อไม่ให้อะไรในลำดับหลังลดระดับ ticket ด้านความปลอดภัยลงได้ จากนั้นตัดสินเส้นทาง แล้วลำดับความสำคัญ แล้วจึง `escalate_to_human`

</details>

<details><summary>ท้าทายเพิ่ม — อัปเกรดสามอย่าง</summary>

1. แทนที่ choice `department` ตัวเดียวด้วย `noul` หนึ่งตัวต่อหนึ่งทีม (section 5) แล้วเปิด ticket หนึ่งใบต่อทุกทีมที่ได้มากกว่า 0.5
2. เพิ่มเส้นทาง `flooding` สำหรับกรณีน้ำท่วมพื้น ให้ไปทั้ง maintenance *และ* duty manager
3. บันทึกทุก ticket เป็น JSON หนึ่งบรรทัด พร้อม `model`, `request_id`, hash ของ question schema และการตัดสินใจสุดท้ายของคน นั่นคือจุดเริ่มต้นของชุดประเมินที่มี label ของคุณ (แล็บ 09)

</details>

✓ Checkpoint: การตรวจแบบออฟไลน์ทุกข้อเป็น ✓ และการรัน live ของคุณได้ ticket ห้าใบ ทุกใบมี `"execute": false`

## การแก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `TODO 1: keys must be …` | question id เป็นสตริงที่ต้องตรงเป๊ะ: `department`, `severity`, `safety`, `needs_human`, `mentions_room` |
| fixture หมายเลขห้องล้มเหลวที่ `— 610.` | lookahead ของคุณปฏิเสธจุดที่ตามหลัง — ให้ยอมรับ `.` เมื่อไม่มีตัวเลขตามหลัง |
| fixture ของ policy "low safety still triggers" ล้มเหลว | คุณใช้ `> 0.2` หรือเกณฑ์ที่สูงกว่า — กฎคือ `≥ 0.20` และตรวจ **ก่อน** ทุกอย่าง |
| `HTTP 422` ตอนรัน live | มีคำถามที่รูปแบบผิด — `score` ต้องมี *list* ของระดับ, `choice` ต้องมี *dict* ของตัวเลือก |
| ticket จากการรัน live ต่างจาก expected output เล็กน้อย | คะแนนที่ก้ำกึ่งขยับได้ระหว่างเวอร์ชัน — ตรวจ `model` แล้วอ่าน `signals` |
| `◈ DRY … placeholder` | คุณเปลี่ยนคำถามในโหมด DRY จึงไม่มีผลที่บันทึกไว้ให้เล่นซ้ำ — สลับเป็น ⚡ Live |

## ถัดไป

คุณเรียน Week 24 จบแล้ว สิ่งที่ควรสร้างต่อ:

- **นำไปรันแบบ shadow ที่ทำงาน** เลือกคิวงานจริงของ AltoTech หนึ่งคิว (ที่ลบข้อมูลระบุตัวตนแล้ว เช่น router ของ Copilot หรือ inbox ที่ใช้ร่วมกัน) แล้วรันการประเมินจากแล็บ 09 บนตัวอย่างที่มี label สักสองสามร้อยตัวต่อภาษา
- **อ่าน pattern ที่ตอนนี้คุณมีชิ้นส่วนครบแล้ว:** [confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing.md), [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md), [composite scoring](https://docs.typesafe.ai/patterns/composite-scoring.md) และ [intent routing](https://docs.typesafe.ai/patterns/intent-routing.md)
- **ลอง cookbook ที่ไปไกลกว่าการจัดประเภท:** [re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md), [citation checks](https://docs.typesafe.ai/cookbooks/citation_check.md) และ [pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md)
- **รู้จักขีดจำกัดก่อนขึ้น production:** [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) และ [confidence](https://docs.typesafe.ai/confidence.md)
- **ใช้แบบ local เมื่อ policy กำหนด:** กลับไปดู [Lab 10](../10_jev_vs_laya/TUTORIAL.th.md) แล้วรันฝั่ง Laya (ทางเลือกเสริม) ของ benchmark ที่ใช้ร่วมกันบน corpus เดียวกัน
- กลับไปจุดเริ่มต้น: [Lab 01 — Hello, Jev](../01_hello_jev/TUTORIAL.th.md)
