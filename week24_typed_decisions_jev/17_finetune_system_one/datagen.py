#!/usr/bin/env python3
"""datagen — the synthetic training pool for Module 17 (hotel-engineering tickets, EN + TH).

Compositional on purpose: ~80 hand-written SITUATIONS (what is wrong, with its three labels), each
written 2–3 ways, wrapped in an opener and a closer that fits its urgency, with a random room and
floor. That gives thousands of distinct tickets from a page of text, and every ticket carries the
labels of the situation it came from, so no one has to label 600 rows by hand.

The weakness is just as deliberate: generated tickets all "sound like" this file. That is why the
TEST set (data/test_handwritten.json) is written separately, in other styles, and why Lab 17-1
proves the two do not overlap before anything is trained.

    import datagen
    pool = datagen.generate(seed=17)          # list of {"id","sid","lang","text","queue","safety","urgency"}

Labels follow Module 16's tickets.json: queue ∈ hvac/housekeeping/plumbing/safety/front_desk,
safety = smoke, burning, fire, sparks or water near electricity, urgency 0 routine · 1 today · 2 immediate.
All synthetic, written for this module on 2026-10-17. Not real guests or sites.
"""
from __future__ import annotations

import random

# (sid, lang, queue, safety, urgency, phrasings)
SITUATIONS: list[tuple[str, str, str, bool, int, list[str]]] = [
    # ── English · hvac ──────────────────────────────────────────────────────
    ("en_hvac_warm", "en", "hvac", False, 1, [
        "The air conditioner in {room} is running but the room stays at 28 degrees.",
        "AC in room {room} only blows warm air, it is not cooling at all.",
        "It is really hot in {room}, the aircon does not seem to work."]),
    ("en_hvac_thermo", "en", "hvac", False, 1, [
        "The thermostat screen in {room} is blank and the AC will not turn on.",
        "Thermostat in {room} is stuck on 30 and ignores the buttons."]),
    ("en_hvac_heat", "en", "hvac", False, 1, [
        "The heating in {room} is far too strong and we cannot turn it down.",
        "Room {room} is boiling, the heater will not switch off."]),
    ("en_hvac_noise", "en", "hvac", False, 1, [
        "The AC unit in {room} is rattling loudly and we cannot sleep.",
        "There is a loud buzzing from the air vent in {room}."]),
    ("en_hvac_fixed", "en", "hvac", False, 0, [
        "The AC in {room} was making noise earlier but it stopped, no need to send anyone.",
        "Ignore my earlier message about the aircon in {room}, it is cooling fine now."]),
    ("en_hvac_musty", "en", "hvac", False, 0, [
        "The air from the vent in {room} smells a bit musty, could someone check the filter sometime?",
        "Slightly stale air when the AC runs in {room}, not urgent."]),
    ("en_hvac_precool", "en", "hvac", False, 0, [
        "Could you pre-cool room {room} to 22 degrees before we come back this evening?",
        "Please set the AC in {room} to 23 degrees for our arrival tonight."]),
    ("en_hvac_ballroom", "en", "hvac", False, 2, [
        "The ballroom on floor {floor} is getting hotter by the minute and the event has 300 guests inside right now.",
        "Chiller seems down, the whole conference hall on floor {floor} is overheating during the meeting."]),
    ("en_hvac_drip", "en", "hvac", False, 1, [
        "The AC unit above the wardrobe in {room} is dripping water onto the carpet.",
        "Water drips from the air conditioner in {room} onto the floor near the window."]),
    # ── English · housekeeping ─────────────────────────────────────────────
    ("en_hk_towels", "en", "housekeeping", False, 0, [
        "Could we have two more bath towels in {room}?",
        "Please bring extra towels and a bath mat to room {room}."]),
    ("en_hk_clean", "en", "housekeeping", False, 1, [
        "Our room {room} has not been cleaned yet today, can someone come this afternoon?",
        "Room {room} still needs service, the beds are not made."]),
    ("en_hk_pillows", "en", "housekeeping", False, 0, [
        "Can I get an extra pillow and a blanket in {room}?",
        "We would like two firm pillows for room {room} please."]),
    ("en_hk_coffee", "en", "housekeeping", False, 0, [
        "We are out of coffee pods and drinking water in {room}.",
        "Please refill the minibar water and tea bags in {room}."]),
    ("en_hk_stain", "en", "housekeeping", False, 1, [
        "The bed sheets in {room} have a stain, please change them.",
        "Found a dirty pillowcase in {room}, could you replace the linen?"]),
    ("en_hk_trash", "en", "housekeeping", False, 0, [
        "The bin in {room} is full, please collect it when you pass by.",
        "Could housekeeping take the rubbish from {room}?"]),
    ("en_hk_cot", "en", "housekeeping", False, 0, [
        "We need a baby cot in {room} for tonight.",
        "Please set up a crib in room {room} before 8pm."]),
    ("en_hk_spill", "en", "housekeeping", False, 1, [
        "We spilled red wine on the carpet in {room}, can someone clean it?",
        "Coffee got spilled all over the sofa in {room}."]),
    # ── English · plumbing ─────────────────────────────────────────────────
    ("en_pl_overflow", "en", "plumbing", False, 2, [
        "The toilet in {room} is overflowing and water is spreading across the floor.",
        "Toilet in {room} will not stop filling, the bathroom floor is flooding."]),
    ("en_pl_drain", "en", "plumbing", False, 1, [
        "The shower drain in {room} is very slow, the water pools around our feet.",
        "Sink in {room} drains really slowly."]),
    ("en_pl_hotwater", "en", "plumbing", False, 1, [
        "There is no hot water in the shower in {room}.",
        "Only cold water comes out of the taps in {room}."]),
    ("en_pl_tap", "en", "plumbing", False, 0, [
        "The bathroom tap in {room} drips a little, whenever someone is around.",
        "Small drip from the basin tap in {room}, nothing urgent."]),
    ("en_pl_clog", "en", "plumbing", False, 1, [
        "The toilet in {room} is blocked and will not flush.",
        "Our toilet in {room} is clogged."]),
    ("en_pl_pressure", "en", "plumbing", False, 1, [
        "Water pressure in the shower in {room} is very weak.",
        "The shower in {room} barely trickles."]),
    ("en_pl_burst", "en", "plumbing", False, 2, [
        "A pipe under the sink in {room} just burst and water is spraying everywhere.",
        "Water is gushing from under the basin in {room}, please come now."]),
    ("en_pl_ceiling", "en", "plumbing", False, 1, [
        "Water is dripping from the bathroom ceiling in {room} into the bathtub.",
        "There is a wet patch on the bathroom ceiling in {room} and it drips slowly."]),
    ("en_pl_showerhead", "en", "plumbing", False, 1, [
        "The shower head in {room} broke off the hose.",
        "Shower handle in {room} is loose and sprays sideways."]),
    # ── English · safety ───────────────────────────────────────────────────
    ("en_sf_panel", "en", "safety", True, 2, [
        "Smoke is coming from the electrical cupboard on floor {floor}.",
        "There is smoke from the power panel next to room {room}."]),
    ("en_sf_burning", "en", "safety", True, 2, [
        "Strong burning smell in the corridor on floor {floor}.",
        "Something smells like burning plastic near room {room}."]),
    ("en_sf_sparks", "en", "safety", True, 2, [
        "The socket by the desk in {room} sparked when I plugged in my laptop.",
        "Sparks flew out of the bedside outlet in {room}."]),
    ("en_sf_wet_tv", "en", "safety", True, 2, [
        "Water is leaking from the ceiling onto the TV and the sockets in {room}.",
        "Rain water is coming in and running down the wall onto the power strip in {room}."]),
    ("en_sf_alarm", "en", "safety", True, 2, [
        "The smoke alarm in {room} is going off and there is a haze in the room.",
        "Fire alarm sounding on floor {floor} and I can see smoke in the hallway."]),
    ("en_sf_kettle", "en", "safety", True, 2, [
        "The kettle cable in {room} melted and started smoking.",
        "The hair dryer in {room} smoked and smells burnt."]),
    ("en_sf_fridge", "en", "safety", True, 2, [
        "There is a puddle of water around the minibar fridge plug in {room}.",
        "Water is pooling under the extension cord behind the desk in {room}."]),
    ("en_sf_shock", "en", "safety", True, 2, [
        "I got an electric shock from the lamp switch in {room}.",
        "The bathroom light switch in {room} gave my son a shock."]),
    # ── English · front desk ───────────────────────────────────────────────
    ("en_fd_extend", "en", "front_desk", False, 0, [
        "We would like to extend our stay in {room} by one more night.",
        "Is it possible to stay two extra nights? We are in {room}."]),
    ("en_fd_double", "en", "front_desk", False, 1, [
        "I was charged twice for breakfast, please refund one charge. Room {room}.",
        "My card shows the deposit taken two times for room {room}."]),
    ("en_fd_late", "en", "front_desk", False, 0, [
        "Can we have a late checkout at 2pm for {room}?",
        "Would a 1pm checkout be possible for room {room}?"]),
    ("en_fd_move", "en", "front_desk", False, 0, [
        "We would prefer a quieter room away from the elevator than {room}.",
        "Could we move from {room} to a room with a sea view?"]),
    ("en_fd_key", "en", "front_desk", False, 1, [
        "My key card stopped working and I am locked out of {room}.",
        "The door of {room} will not open with my card anymore."]),
    ("en_fd_invoice", "en", "front_desk", False, 0, [
        "Please issue the invoice for {room} with our company tax ID.",
        "I need a receipt with the company name for my stay in {room}."]),
    ("en_fd_taxi", "en", "front_desk", False, 0, [
        "Please book a taxi to the airport for 6am tomorrow, room {room}.",
        "Can you arrange an airport transfer for {room} on Friday?"]),
    ("en_fd_lost", "en", "front_desk", False, 1, [
        "I think I left my phone in the lobby café, I am in room {room}.",
        "Lost a black wallet near the pool today, guest of {room}."]),
    # ── Thai · hvac ────────────────────────────────────────────────────────
    ("th_hvac_warm", "th", "hvac", False, 1, [
        "แอร์ห้อง {room} เปิดอยู่แต่ไม่เย็นเลย",
        "ห้อง {room} ร้อนมาก แอร์มีแต่ลมร้อน"]),
    ("th_hvac_remote", "th", "hvac", False, 1, [
        "รีโมทแอร์ห้อง {room} กดไม่ติด หน้าจอไม่ขึ้นเลย",
        "ตั้งอุณหภูมิแอร์ห้อง {room} ไม่ได้ ค้างอยู่ที่ 30 องศา"]),
    ("th_hvac_noise", "th", "hvac", False, 1, [
        "แอร์ห้อง {room} มีเสียงดังมาก นอนไม่หลับ",
        "ช่องแอร์ห้อง {room} มีเสียงหึ่งๆ ตลอดเวลา"]),
    ("th_hvac_fixed", "th", "hvac", False, 0, [
        "แอร์ห้อง {room} กลับมาเย็นแล้ว ไม่ต้องส่งช่างมานะ",
        "เรื่องแอร์ห้อง {room} เมื่อกี้ไม่เป็นไรแล้ว ยกเลิกได้เลย"]),
    ("th_hvac_hall", "th", "hvac", False, 2, [
        "ห้องจัดเลี้ยงชั้น {floor} ร้อนขึ้นเรื่อยๆ ตอนนี้มีแขกอยู่ข้างในหลายร้อยคน",
        "ระบบแอร์ห้องประชุมชั้น {floor} ดับ ร้อนมากระหว่างงาน"]),
    ("th_hvac_drip", "th", "hvac", False, 1, [
        "แอร์ห้อง {room} มีน้ำหยดลงพรมตรงหน้าต่าง",
        "น้ำแอร์ห้อง {room} หยดใส่พื้นข้างตู้เสื้อผ้า"]),
    # ── Thai · housekeeping ────────────────────────────────────────────────
    ("th_hk_towels", "th", "housekeeping", False, 0, [
        "ขอผ้าเช็ดตัวเพิ่มสองผืนที่ห้อง {room}",
        "รบกวนเอาผ้าเช็ดตัวกับพรมเช็ดเท้ามาเพิ่มที่ห้อง {room}"]),
    ("th_hk_clean", "th", "housekeeping", False, 1, [
        "ห้อง {room} ยังไม่ได้ทำความสะอาดเลยวันนี้ ช่วยมาทำช่วงบ่ายได้ไหม",
        "ห้อง {room} ยังไม่มีคนมาเก็บห้อง เตียงยังไม่ได้ปู"]),
    ("th_hk_pillow", "th", "housekeeping", False, 0, [
        "ขอหมอนเพิ่มอีกใบกับผ้าห่มที่ห้อง {room}",
        "อยากได้หมอนแบบแข็งสองใบที่ห้อง {room}"]),
    ("th_hk_water", "th", "housekeeping", False, 0, [
        "น้ำดื่มกับกาแฟในห้อง {room} หมดแล้ว",
        "รบกวนเติมน้ำเปล่ากับชาในห้อง {room}"]),
    ("th_hk_stain", "th", "housekeeping", False, 1, [
        "ผ้าปูที่นอนห้อง {room} มีคราบ ช่วยเปลี่ยนให้หน่อย",
        "ปลอกหมอนห้อง {room} สกปรก ขอเปลี่ยนใหม่"]),
    ("th_hk_cot", "th", "housekeeping", False, 0, [
        "ขอเตียงเด็กที่ห้อง {room} สำหรับคืนนี้",
        "ช่วยเตรียมเตียงเด็กอ่อนไว้ที่ห้อง {room} ก่อนสองทุ่ม"]),
    # ── Thai · plumbing ────────────────────────────────────────────────────
    ("th_pl_overflow", "th", "plumbing", False, 2, [
        "ชักโครกห้อง {room} น้ำล้นออกมาเต็มพื้นห้องน้ำแล้ว",
        "น้ำในชักโครกห้อง {room} ไหลไม่หยุด ท่วมห้องน้ำแล้ว"]),
    ("th_pl_drain", "th", "plumbing", False, 1, [
        "ท่อระบายน้ำฝักบัวห้อง {room} ตัน น้ำขังเต็มเลย",
        "อ่างล้างหน้าห้อง {room} น้ำลงช้ามาก"]),
    ("th_pl_hotwater", "th", "plumbing", False, 1, [
        "ห้อง {room} ไม่มีน้ำอุ่นเลย",
        "เครื่องทำน้ำอุ่นห้อง {room} ไม่ทำงาน มีแต่น้ำเย็น"]),
    ("th_pl_tap", "th", "plumbing", False, 0, [
        "ก๊อกน้ำห้องน้ำห้อง {room} มีน้ำหยดนิดหน่อย ไม่รีบ",
        "ก๊อกอ่างล้างหน้าห้อง {room} ปิดไม่สนิท หยดทีละนิด"]),
    ("th_pl_clog", "th", "plumbing", False, 1, [
        "ชักโครกห้อง {room} ตัน กดไม่ลง",
        "ห้อง {room} ส้วมตันค่ะ"]),
    ("th_pl_burst", "th", "plumbing", False, 2, [
        "ท่อใต้อ่างห้อง {room} แตก น้ำพุ่งออกมาเต็มเลย",
        "น้ำทะลักออกมาจากใต้อ่างล้างหน้าห้อง {room} มาด่วน"]),
    # ── Thai · safety ──────────────────────────────────────────────────────
    ("th_sf_panel", "th", "safety", True, 2, [
        "มีควันออกจากตู้ไฟชั้น {floor}",
        "ตู้ไฟข้างห้อง {room} มีควันขึ้น"]),
    ("th_sf_burning", "th", "safety", True, 2, [
        "ได้กลิ่นไหม้แรงมากที่ทางเดินชั้น {floor}",
        "มีกลิ่นเหมือนพลาสติกไหม้แถวห้อง {room}"]),
    ("th_sf_sparks", "th", "safety", True, 2, [
        "ปลั๊กไฟข้างโต๊ะห้อง {room} มีประกายไฟตอนเสียบสายชาร์จ",
        "เต้ารับหัวเตียงห้อง {room} ไฟแลบออกมา"]),
    ("th_sf_wet", "th", "safety", True, 2, [
        "น้ำรั่วจากเพดานหยดใส่ทีวีกับปลั๊กไฟในห้อง {room}",
        "น้ำฝนไหลลงผนังโดนปลั๊กพ่วงในห้อง {room}"]),
    ("th_sf_alarm", "th", "safety", True, 2, [
        "สัญญาณเตือนไฟไหม้ห้อง {room} ดัง แล้วมีควันในห้อง",
        "กริ่งไฟไหม้ชั้น {floor} ดัง เห็นควันที่ทางเดิน"]),
    ("th_sf_shock", "th", "safety", True, 2, [
        "โดนไฟดูดตอนกดสวิตช์โคมไฟในห้อง {room}",
        "สวิตช์ไฟห้องน้ำห้อง {room} ไฟดูดลูกชาย"]),
    # ── Thai · front desk ──────────────────────────────────────────────────
    ("th_fd_extend", "th", "front_desk", False, 0, [
        "ขอต่อห้อง {room} อีกหนึ่งคืนได้ไหม",
        "อยากอยู่ต่ออีกสองคืน ตอนนี้พักห้อง {room}"]),
    ("th_fd_late", "th", "front_desk", False, 0, [
        "ขอเช็คเอาท์ช้าเป็นบ่ายโมงได้ไหม ห้อง {room}",
        "ห้อง {room} ขอเลื่อนเวลาเช็คเอาท์เป็นบ่ายสอง"]),
    ("th_fd_double", "th", "front_desk", False, 1, [
        "โดนตัดเงินค่าอาหารเช้าซ้ำสองครั้ง ห้อง {room} ช่วยคืนเงินให้ด้วย",
        "บัตรโดนกันวงเงินมัดจำสองรอบ ห้อง {room}"]),
    ("th_fd_key", "th", "front_desk", False, 1, [
        "คีย์การ์ดใช้ไม่ได้ เข้าห้อง {room} ไม่ได้",
        "แตะบัตรแล้วประตูห้อง {room} ไม่เปิด"]),
    ("th_fd_taxi", "th", "front_desk", False, 0, [
        "ช่วยเรียกแท็กซี่ไปสนามบินตอนหกโมงเช้าพรุ่งนี้ ห้อง {room}",
        "ขอจองรถรับส่งสนามบินวันศุกร์ ห้อง {room}"]),
    ("th_fd_invoice", "th", "front_desk", False, 0, [
        "ขอใบกำกับภาษีในนามบริษัทสำหรับห้อง {room}",
        "รบกวนออกใบเสร็จชื่อบริษัทให้ห้อง {room}"]),
]

OPENERS = {
    "en": ["", "", "Hi, ", "Hello, ", "Good evening. ", "Hi front desk, ", "Sorry to bother you, ",
           "Guest message: ", "Note from the night shift: "],
    "th": ["", "", "สวัสดีค่ะ ", "สวัสดีครับ ", "รบกวนหน่อยค่ะ ", "ขอแจ้งเรื่องหน่อยครับ ", "ข้อความจากแขก: "],
}
# Closers are chosen by urgency, so a closer never contradicts the label.
CLOSERS = {
    "en": {0: ["", " Thanks.", " No rush.", " Whenever convenient.", " Thank you!"],
           1: ["", " Thanks.", " Please send someone today.", " Could you look at it?", " Thank you."],
           2: ["", " Please hurry!", " Please come right now.", " Urgent!", " Send someone immediately."]},
    "th": {0: ["", " ค่ะ", " ครับ", " ไม่รีบค่ะ", " ขอบคุณค่ะ"],
           1: ["", " ค่ะ", " ครับ", " ช่วยส่งคนมาดูวันนี้ด้วย", " ขอบคุณครับ"],
           2: ["", " ด่วนค่ะ", " รีบมาด่วนครับ", " ช่วยด้วยค่ะ", " มาเดี๋ยวนี้เลย"]},
}
FLOORS = list(range(2, 26))


def _room(rng: random.Random) -> str:
    return f"{rng.choice(FLOORS)}{rng.randint(1, 30):02d}"


def render(sit: tuple, rng: random.Random) -> str:
    sid, lang, queue, safety, urgency, phrasings = sit
    floor = rng.choice(FLOORS)
    body = rng.choice(phrasings).format(room=f"{floor}{rng.randint(1, 30):02d}", floor=floor)
    opener = rng.choice(OPENERS[lang])
    if opener.endswith(", ") and not body.startswith(("I ", "AC")):
        body = body[0].lower() + body[1:]                  # "Hi, could we…", not "Hi, Could we…"
    text = opener + body + rng.choice(CLOSERS[lang][urgency])
    if lang == "en" and rng.random() < 0.12:              # some people type in lower case, no full stop
        text = text.lower().rstrip(".")
    return text.strip()


def generate(seed: int = 17, per_situation: int = 8) -> list[dict]:
    """`per_situation` distinct tickets for every situation (duplicates are re-drawn)."""
    rng = random.Random(seed)
    out, seen = [], set()
    for sit in SITUATIONS:
        sid, lang, queue, safety, urgency, _ = sit
        made, tries = 0, 0
        while made < per_situation and tries < per_situation * 20:
            tries += 1
            text = render(sit, rng)
            if text in seen:
                continue
            seen.add(text)
            made += 1
            out.append({"id": f"{sid}_{made:02d}", "sid": sid, "lang": lang, "text": text,
                        "queue": queue, "safety": safety, "urgency": urgency})
    return out


def split_by_situation(pool: list[dict], val_frac: float = 0.2, seed: int = 17) -> tuple[list, list]:
    """GROUP split: every ticket of a situation lands on the same side. Val then asks 'does it
    generalise to a situation it never saw?' instead of 'does it recognise a re-worded copy?'."""
    rng = random.Random(seed)
    by_cls: dict[tuple, list[str]] = {}
    for s in SITUATIONS:
        by_cls.setdefault((s[1], s[2]), []).append(s[0])
    val_sids = set()
    for (lang, queue), sids in sorted(by_cls.items()):    # stratified: each (language, queue) gives up
        rng.shuffle(sids)                                  # at least one situation to val
        val_sids.update(sids[:max(1, round(len(sids) * val_frac))])
    return ([t for t in pool if t["sid"] not in val_sids], [t for t in pool if t["sid"] in val_sids])


def split_random(pool: list[dict], val_frac: float = 0.2, seed: int = 17) -> tuple[list, list]:
    """The tempting split: shuffle tickets. Re-worded copies of the same situation end up on both sides."""
    rng = random.Random(seed)
    rows = pool[:]
    rng.shuffle(rows)
    k = round(len(rows) * val_frac)
    return rows[k:], rows[:k]


if __name__ == "__main__":
    p = generate()
    print(len(SITUATIONS), "situations →", len(p), "tickets")
    for t in p[:: max(1, len(p) // 12)]:
        print(f"  {t['queue']:<12} {t['urgency']} {t['text']}")
