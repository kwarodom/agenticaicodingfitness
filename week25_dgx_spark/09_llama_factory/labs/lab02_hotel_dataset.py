#!/usr/bin/env python3
"""Lab 09-2 · Build a real hotel-operations dataset in LLaMA Factory's Alpaca format, and validate it.

The task: a guest writes a request ("the AC in 1412 isn't cooling"); the model answers with ONE line
of JSON — which department handles it, how urgent it is, and a short polite reply in the guest's
language (English or Thai). That output is easy to check by code, which Module 13 uses to score the
fine-tune.

This lab writes three files into week25/09_llama_factory/data/ (deterministic: same files every run):
  hotel_ops.json        training records   (Alpaca: instruction · input · output · system)
  hotel_ops_eval.json   held-out records, built from request templates the model never trains on
  dataset_info.json     the registry LLaMA Factory reads to find a dataset by name

Then it validates them the way a trainer would fail on them — before you spend GPU time.
Runs on this laptop. No Spark, no network, no cost.

Run: .venv/bin/python week25/09_llama_factory/labs/lab02_hotel_dataset.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, result, step, table  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
DEPTS = ["housekeeping", "engineering", "front_desk", "food_beverage", "concierge", "security"]
PRIORITIES = ["normal", "urgent"]
CUTOFF_LEN = 1024                     # the cutoff_len lab03 writes into the training YAML
SEED = 25

SYSTEM = ("You are the guest-request router for a hotel. Read the guest's message and answer with one line "
          "of JSON: {\"department\": one of housekeeping|engineering|front_desk|food_beverage|concierge|security, "
          "\"priority\": normal|urgent, \"reply\": a short, polite reply to the guest in the guest's language}. "
          "Do not promise a specific time.")

# ── slot values ───────────────────────────────────────────────────────────────
SLOTS = {
    "n": ["2", "3", "4"],
    "item": ["an extra pillow", "a baby cot", "more coffee capsules", "two bathrobes", "a toothbrush kit",
             "extra hangers", "a hair dryer", "an iron and ironing board", "slippers"],
    "hk_time": ["11:00", "13:00", "14:00", "15:00"],
    "late_time": ["12:00", "13:00", "14:00", "15:00", "16:00"],
    "taxi_time": ["05:30", "06:00", "07:15", "09:00", "14:30", "18:00"],
    "dinner_time": ["18:30", "19:00", "19:30", "20:00"],
    "dish": ["a club sandwich", "a Caesar salad", "pad thai", "green curry with rice", "a margherita pizza",
             "two bowls of tom yum soup", "a fruit platter", "a cheeseburger with fries"],
    "dish_th": ["ผัดไทย", "ข้าวผัดกุ้ง", "ต้มยำกุ้ง", "แกงเขียวหวานไก่กับข้าว", "ส้มตำ", "ข้าวมันไก่"],
    "place": ["the Grand Palace", "Ayutthaya", "the floating market", "Chatuchak Weekend Market",
              "a Muay Thai show", "the Chao Phraya dinner cruise"],
    "place_th": ["พระบรมมหาราชวัง", "อยุธยา", "ตลาดน้ำ", "ตลาดนัดจตุจักร"],
    "thing": ["my laptop", "a black backpack", "my passport wallet", "a child's teddy bear", "my phone charger"],
    "spot": ["the lobby", "the pool", "the gym", "the breakfast room"],
}
EN_OPEN = ["", "", "Hi, ", "Hello, ", "Good evening. ", "Excuse me, "]
EN_CLOSE = ["", "", " Thanks!", " Thank you.", " Thanks a lot."]
TH_OPEN = ["", "สวัสดีค่ะ ", "สวัสดีครับ "]

# ── request templates: (department, priority, lang, split, guest request, reply) ──
# split "eval" marks a template held out of training, so the eval set tests generalisation,
# not memorised wording.
T = [
    # housekeeping
    ("housekeeping", "normal", "en", "train", "Could I get {n} extra towels in room {room}?",
     "Of course. Housekeeping will bring {n} extra towels to room {room} shortly."),
    ("housekeeping", "normal", "en", "train", "Can we have {item} for room {room}?",
     "Certainly. Housekeeping will bring {item} to room {room} shortly."),
    ("housekeeping", "normal", "en", "train", "Room {room}: please skip cleaning today, we don't want to be disturbed.",
     "Noted. Housekeeping will not service room {room} today. Just let us know if you need anything."),
    ("housekeeping", "normal", "en", "train", "We need fresh sheets in {room}, my son spilled juice on the bed.",
     "No problem at all. Housekeeping will come to room {room} to change the sheets."),
    ("housekeeping", "normal", "en", "train", "Can housekeeping come to room {room} after {hk_time} instead of the morning?",
     "Certainly. Housekeeping will service room {room} after {hk_time}."),
    ("housekeeping", "normal", "en", "eval", "The bathroom in {room} wasn't cleaned properly, there's hair in the sink.",
     "I'm sorry about that. Housekeeping will come to room {room} to clean the bathroom again."),
    ("housekeeping", "normal", "th", "train", "ขอผ้าเช็ดตัวเพิ่ม {n} ผืนที่ห้อง {room}",
     "ได้เลยค่ะ แม่บ้านจะนำผ้าเช็ดตัว {n} ผืนไปที่ห้อง {room} ค่ะ"),
    ("housekeeping", "normal", "th", "train", "ช่วยมาทำความสะอาดห้อง {room} หลัง {hk_time} ได้ไหม",
     "ได้เลยค่ะ แม่บ้านจะเข้าไปทำความสะอาดห้อง {room} หลังเวลา {hk_time} ค่ะ"),
    # engineering
    ("engineering", "normal", "en", "train", "The air conditioning in room {room} isn't cooling.",
     "I'm sorry for the discomfort. An engineer will come to room {room} to check the air conditioning."),
    ("engineering", "urgent", "en", "train", "There's water leaking from the ceiling in {room}!",
     "Thank you for telling us. An engineer is on the way to room {room} now. Please keep your belongings away from the leak."),
    ("engineering", "normal", "en", "train", "The TV in room {room} won't turn on.",
     "Sorry about that. An engineer will come to room {room} to look at the TV."),
    ("engineering", "normal", "en", "train", "There's no hot water in the shower in room {room}.",
     "I apologise for the inconvenience. An engineer will come to room {room} to check the hot water."),
    ("engineering", "urgent", "en", "train", "The toilet in room {room} is overflowing onto the floor.",
     "We're sending an engineer to room {room} right away. Please step out of the bathroom until they arrive."),
    ("engineering", "normal", "en", "eval", "The bedside lamp and the desk light in {room} both stopped working.",
     "Sorry for the trouble. An engineer will come to room {room} to fix the lights."),
    ("engineering", "normal", "th", "train", "แอร์ห้อง {room} ไม่เย็นเลย",
     "ขออภัยในความไม่สะดวกค่ะ ช่างจะขึ้นไปตรวจแอร์ที่ห้อง {room} ค่ะ"),
    ("engineering", "urgent", "th", "train", "น้ำรั่วจากเพดานห้อง {room} ด่วนมาก",
     "ขอบคุณที่แจ้งค่ะ ช่างกำลังขึ้นไปที่ห้อง {room} ทันที กรุณาย้ายของมีค่าออกห่างจากจุดที่น้ำรั่วค่ะ"),
    ("engineering", "normal", "th", "eval", "ทีวีในห้อง {room} เปิดไม่ติด",
     "ขออภัยค่ะ ช่างจะขึ้นไปตรวจทีวีที่ห้อง {room} ค่ะ"),
    # front desk
    ("front_desk", "normal", "en", "train", "Can I get a late checkout until {late_time} for room {room}?",
     "I've passed your request to the Front Desk. They will confirm a {late_time} checkout for room {room} shortly."),
    ("front_desk", "normal", "en", "train", "My key card for room {room} stopped working.",
     "Sorry about that. Please stop by the Front Desk, or we can send someone to room {room} with a new key card."),
    ("front_desk", "normal", "en", "train", "Could you print the invoice for room {room}?",
     "Certainly. The Front Desk will prepare the invoice for room {room}."),
    ("front_desk", "normal", "en", "train", "I'd like to extend my stay in room {room} by {n} nights.",
     "Thank you for staying longer with us. The Front Desk will check availability for {n} more nights in room {room}."),
    ("front_desk", "normal", "en", "train", "Room {room} is right next to the elevator, can I move to a quieter room?",
     "I'm sorry about the noise. The Front Desk will look for a quieter room and contact you in room {room}."),
    ("front_desk", "normal", "en", "eval", "What is the Wi-Fi password? I'm in room {room}.",
     "The Front Desk will send the Wi-Fi details to room {room} right away."),
    ("front_desk", "normal", "th", "train", "ขอเช็คเอาท์ช้าถึง {late_time} ได้ไหม ห้อง {room}",
     "รับทราบค่ะ แผนกต้อนรับจะตรวจสอบและยืนยันการเช็คเอาท์ถึง {late_time} สำหรับห้อง {room} ค่ะ"),
    # food & beverage
    ("food_beverage", "normal", "en", "train", "Can I order {dish} to room {room}?",
     "Certainly. Room Service will prepare {dish} and bring it to room {room}."),
    ("food_beverage", "normal", "en", "train", "Please bring a bottle of sparkling water and two glasses to {room}.",
     "Of course. Room Service will bring sparkling water and two glasses to room {room}."),
    ("food_beverage", "normal", "en", "train", "Our breakfast order for room {room} still hasn't arrived.",
     "I'm sorry for the wait. Room Service is checking your order for room {room} now."),
    ("food_beverage", "normal", "en", "train", "I'd like a table for {n} at the restaurant tonight at {dinner_time}.",
     "With pleasure. Our restaurant team will reserve a table for {n} at {dinner_time} and confirm with you."),
    ("food_beverage", "normal", "en", "eval", "Do you have vegan dishes on the room service menu? Room {room}.",
     "Yes, we do. Room Service will send the vegan menu options to room {room}."),
    ("food_beverage", "normal", "th", "train", "ขอสั่ง{dish_th}ไปที่ห้อง {room}",
     "ได้เลยค่ะ รูมเซอร์วิสจะจัดเตรียม{dish_th}และนำไปส่งที่ห้อง {room} ค่ะ"),
    ("food_beverage", "normal", "th", "train", "จองโต๊ะอาหารเย็น {n} ที่ เวลา {dinner_time}",
     "ยินดีค่ะ ห้องอาหารจะจองโต๊ะสำหรับ {n} ท่าน เวลา {dinner_time} และยืนยันให้ทราบค่ะ"),
    ("food_beverage", "normal", "th", "eval", "อาหารเช้าที่สั่งไว้ห้อง {room} ยังไม่มาเลย",
     "ขออภัยที่ให้รอค่ะ รูมเซอร์วิสกำลังตรวจสอบรายการอาหารของห้อง {room} ค่ะ"),
    # concierge
    ("concierge", "normal", "en", "train", "Can you book a taxi to the airport at {taxi_time}?",
     "Certainly. The Concierge will book a taxi to the airport for {taxi_time} and confirm with you."),
    ("concierge", "normal", "en", "train", "Could you arrange a trip to {place} for {n} people?",
     "With pleasure. The Concierge will arrange a trip to {place} for {n} people and send you the details."),
    ("concierge", "normal", "en", "train", "Can you get us tickets for {place} tomorrow?",
     "Of course. The Concierge will check tickets for {place} tomorrow and get back to you."),
    ("concierge", "normal", "en", "train", "Where can I exchange money near the hotel? I'm in room {room}.",
     "The Concierge will send directions to the nearest money exchange to room {room}."),
    ("concierge", "normal", "en", "eval", "Could you recommend a good seafood restaurant within walking distance?",
     "Happily. The Concierge will prepare a few seafood restaurant suggestions within walking distance for you."),
    ("concierge", "normal", "th", "train", "ช่วยเรียกแท็กซี่ไปสนามบินตอน {taxi_time} ได้ไหม",
     "ได้เลยค่ะ คอนเซียร์จจะจองแท็กซี่ไปสนามบินเวลา {taxi_time} และแจ้งยืนยันให้ทราบค่ะ"),
    ("concierge", "normal", "th", "train", "อยากไปเที่ยว{place_th} {n} คน ช่วยจัดให้หน่อย",
     "ยินดีค่ะ คอนเซียร์จจะจัดทริปไป{place_th}สำหรับ {n} ท่าน และส่งรายละเอียดให้ค่ะ"),
    # security
    ("security", "urgent", "en", "train", "Someone keeps knocking on the door of room {room} and won't go away.",
     "Please don't open the door. Security is on the way to room {room} now."),
    ("security", "normal", "en", "train", "I think I left {thing} in {spot}. Has anyone found it?",
     "Security will check lost property for {thing} from {spot} and let you know."),
    ("security", "urgent", "en", "train", "There's a strong smell of smoke in the corridor near room {room}.",
     "Thank you for alerting us. Security is on the way to the corridor near room {room} now. Please stay ready to leave if the alarm sounds."),
    ("security", "urgent", "en", "train", "My friend in room {room} fainted, we need help!",
     "Help is on the way to room {room} now. Our security team is trained in first aid. Please stay with your friend."),
    ("security", "urgent", "en", "eval", "Someone left an unattended bag next to {spot} a long time ago.",
     "Thank you for telling us. Security is going to {spot} now. Please keep away from the bag."),
    ("security", "urgent", "th", "train", "มีคนมาเคาะประตูห้อง {room} ไม่หยุดเลย กลัวมาก",
     "ไม่ต้องกังวลนะคะ เจ้าหน้าที่รักษาความปลอดภัยกำลังไปที่ห้อง {room} ทันที กรุณาอย่าเปิดประตูจนกว่าเจ้าหน้าที่จะมาถึงค่ะ"),
]
TRAIN_PER_DEPT, EVAL_PER_DEPT = 84, 10


def fill(template: str, values: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: values[m.group(1)], template)


def make(rng: random.Random, dept, prio, lang, req, reply) -> dict:
    values = {k: rng.choice(v) for k, v in SLOTS.items()}
    values["room"] = f"{rng.randint(2, 24)}{rng.randint(1, 30):02d}"
    text = fill(req, values)
    if lang == "en":
        opener = rng.choice(EN_OPEN)
        if opener.endswith(", ") and not re.match(r"I\b", text):
            text = text[0].lower() + text[1:]                  # "Hi, could I …", but keep "Hi, I think …"
        text = opener + text + rng.choice(EN_CLOSE)
    else:
        text = rng.choice(TH_OPEN) + text
    out = json.dumps({"department": dept, "priority": prio, "reply": fill(reply, values)}, ensure_ascii=False)
    return {"instruction": text.strip(), "input": "", "output": out, "system": SYSTEM}


def build(split: str, per_dept: int, rng: random.Random, taken: set) -> list[dict]:
    rows = []
    for dept in DEPTS:
        temps = [t for t in T if t[0] == dept and t[3] == split]
        got, tries = 0, 0
        while got < per_dept and tries < per_dept * 200:
            tries += 1
            d, p, lang, _, req, reply = temps[got % len(temps)]
            rec = make(rng, d, p, lang, req, reply)
            if rec["instruction"] in taken:
                continue
            taken.add(rec["instruction"])
            rows.append(rec)
            got += 1
    rng.shuffle(rows)
    return rows


def approx_tokens(s: str) -> int:
    """Rough token estimate: ~4 chars/token for English, ~2 for Thai script (not a real tokenizer)."""
    thai = sum(1 for ch in s if "฀" <= ch <= "๿")
    return round((len(s) - thai) / 4 + thai / 2)


# ── build ─────────────────────────────────────────────────────────────────────
banner("Lab 09-2 · hotel-operations dataset (Alpaca format)",
       "guest request → {department, priority, reply} · built and validated on this laptop", status=False)

step(1, "generate records from request templates (seed 25, so every run writes the same files)")
rng, taken = random.Random(SEED), set()
train = build("train", TRAIN_PER_DEPT, rng, taken)
evalset = build("eval", EVAL_PER_DEPT, rng, taken)
n_tr_t = sum(1 for t in T if t[3] == "train")
note(f"{len(T)} templates: {n_tr_t} for training, {len(T) - n_tr_t} held out for evaluation only")
print(json.dumps(train[0], ensure_ascii=False, indent=2).replace(SYSTEM, "<SYSTEM prompt, "
                                                                      f"{len(SYSTEM)} chars>"))

step(2, "write the dataset files and the dataset_info.json registry")
DATA.mkdir(parents=True, exist_ok=True)
columns = {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
info = {"hotel_ops": {"file_name": "hotel_ops.json", "columns": columns},
        "hotel_ops_eval": {"file_name": "hotel_ops_eval.json", "columns": columns}}
for name, rows in (("hotel_ops.json", train), ("hotel_ops_eval.json", evalset)):
    (DATA / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(DATA / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
for f in ("hotel_ops.json", "hotel_ops_eval.json", "dataset_info.json"):
    p = DATA / f
    print(f"→ wrote {p.relative_to(DATA.parents[1])}  ({p.stat().st_size / 1024:.1f} KB)")

# ── validate: re-read from disk, exactly as a trainer would ───────────────────
step(3, "validate — the checks a failed training run would otherwise teach you")
info = json.loads((DATA / "dataset_info.json").read_text(encoding="utf-8"))
good = True
loaded = {}
for name, entry in info.items():
    path = DATA / entry["file_name"]
    exists = path.is_file()
    good &= check(exists, f"{name}: file_name '{entry['file_name']}' exists next to dataset_info.json",
                  f"{name}: '{entry['file_name']}' not found in {DATA}")
    if not exists:
        continue
    rows = json.loads(path.read_text(encoding="utf-8"))
    loaded[name] = rows
    missing = {col for col in entry["columns"].values() if any(col not in r for r in rows)}
    good &= check(isinstance(rows, list) and not missing,
                  f"{name}: {len(rows)} records, every mapped column present ({', '.join(entry['columns'].values())})",
                  f"{name}: columns missing from some records: {sorted(missing)}")

bad_json, bad_enum, room_mismatch, empty = [], [], [], []
for name, rows in loaded.items():
    for r in rows:
        if not r["instruction"].strip() or not r["output"].strip():
            empty.append(r)
            continue
        try:
            o = json.loads(r["output"])
        except json.JSONDecodeError:
            bad_json.append(r)
            continue
        if o.get("department") not in DEPTS or o.get("priority") not in PRIORITIES or not o.get("reply"):
            bad_enum.append(r)
        m = re.search(r"\b(\d{3,4})\b", r["instruction"])
        if m and m.group(1) not in o.get("reply", ""):
            room_mismatch.append(r)
good &= check(not empty, "no empty instruction or output", f"{len(empty)} records have an empty field")
good &= check(not bad_json, "every output is one line of valid JSON", f"{len(bad_json)} outputs are not valid JSON")
good &= check(not bad_enum, f"department ∈ {len(DEPTS)} allowed values · priority ∈ normal|urgent · reply non-empty",
              f"{len(bad_enum)} outputs use a value outside the allowed set")
good &= check(not room_mismatch, "every reply repeats the guest's room number (no invented rooms)",
              f"{len(room_mismatch)} replies mention a different room than the request")
tr_i = {r["instruction"] for r in loaded.get("hotel_ops", [])}
ev_i = {r["instruction"] for r in loaded.get("hotel_ops_eval", [])}
good &= check(len(tr_i) == len(loaded.get("hotel_ops", [])) and not (tr_i & ev_i),
              "no duplicate requests, and no request appears in both train and eval",
              f"{len(tr_i & ev_i)} requests leak from train into eval")
longest = max(approx_tokens(SYSTEM + r["instruction"] + r["output"]) for rows in loaded.values() for r in rows)
good &= check(longest < CUTOFF_LEN, f"longest record ≈ {longest} tokens, well under cutoff_len {CUTOFF_LEN} "
              "(nothing gets truncated)", f"longest record ≈ {longest} tokens ≥ cutoff_len {CUTOFF_LEN}")

step(4, "balance — a router trained on 90% housekeeping learns to say 'housekeeping'")
rows = []
for dept in DEPTS:
    tr = [r for r in train if json.loads(r["output"])["department"] == dept]
    ev = [r for r in evalset if json.loads(r["output"])["department"] == dept]
    th = sum(1 for r in tr if any("฀" <= c <= "๿" for c in r["instruction"]))
    urg = sum(1 for r in tr if json.loads(r["output"])["priority"] == "urgent")
    rows.append([dept, len(tr), len(ev), th, urg])
table(rows, ["department", "train", "eval", "train · Thai", "train · urgent"])
prio = Counter(json.loads(r["output"])["priority"] for r in train)
note(f"priority in train: normal {prio['normal']} · urgent {prio['urgent']} — urgent is rarer, as in a real hotel. "
     "Watch its recall in Module 13, not just overall accuracy.")

if not good:
    result("✕ fix the dataset before training — every ✕ above would waste a GPU run.")
    sys.exit(1)
result(f"dataset ready: {len(train)} train + {len(evalset)} eval records in week25/09_llama_factory/data/. "
       "Lab 03 writes the training YAML that points at 'hotel_ops' and uploads both to the Spark.")
