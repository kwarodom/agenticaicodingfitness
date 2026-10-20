# ▶ Mod Gym Lesson 01 — mod คืออะไร

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณจะได้เรียนวิธีเปลี่ยนหน้าตาและพฤติกรรมของตัว Claude Code เอง จากในเทอร์มินัลที่คุณเขียนโค้ดอยู่ทุกวัน คอร์สนี้มีสองส่วนที่สอนแปดบทเรียนเดียวกัน คือ tutorial ชุดนี้ (ฉบับเต็ม) และ **🏋 Mod Gym** ซึ่งเป็นแผง (panel) ภายใน Claude Code (ฉบับย่อ พร้อมตัวตรวจงาน) Mod Gym เองก็เป็น mod ตัวหนึ่ง ทุกอย่างที่มันแสดงบนจอจึงเป็นสิ่งที่คุณจะได้เรียนสร้างเอง ตลอดบทเรียนคุณจะค่อย ๆ สร้าง lab mod เล็ก ๆ หนึ่งตัวชื่อ `hello-mod` ฉบับที่เสร็จแล้วอยู่ที่ `week29_claude_code_mods/reference/hello-mod`

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- วาง mod ลงบนแผนที่จุดต่อขยายของ Claude Code: skill, settings hook, MCP server, LSP server, output style, status line และ mod
- เข้าใจไอเดียเดียวที่ mod ทุกตัวสร้างขึ้นมา: hook `($, e, next)` ที่ **observe** (เฝ้าดู), **rewrite** (เขียนทับ) หรือ **answer** (ตอบเอง) event ได้
- ตรวจสอบ (audit) mod ก่อนจะไว้ใจมัน ด้วยบรรทัด `hooks:` และ `calls:` ของ `claude plugin validate`
- ติดตั้ง Mod Gym แล้วเดินชม: แผง แท็บสามแท็บ เทป Events แบบสด แถบ (band) เหนือช่อง prompt และ tool ที่ Claude ใช้โค้ชคุณ

**Time** ~30 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้

**แหล่งอ้างอิง:** [Anthropic blog — Claude Code mods](https://claude.com/blog/claude-code-mods) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Plugin components](https://code.claude.com/docs/en/plugins/components) · [Status line](https://code.claude.com/docs/en/statusline) · [Host a marketplace](https://code.claude.com/docs/en/plugins/host-marketplace) · ไฟล์ประกาศ API ที่ Claude Code 2.1.292 เขียนให้ (`.claude-plugin/types/claude-code/index.d.ts` ซึ่งถือเป็นข้อมูลที่ถูกต้องที่สุดเมื่อเอกสารกับโค้ดไม่ตรงกัน)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Claude Code 2.1.287 ขึ้นไป | `claude --version` | mod เริ่มมีใน 2.1.287 (เทอร์มินัล) |
| repo นี้ | `ls week29_claude_code_mods/mod-gym` | Mod Gym ติดตั้งจากโฟลเดอร์นี้ |
| Week 15 (skill และ plugin) | รู้ว่า `SKILL.md` คืออะไร | บทนี้เทียบ mod กับ skill |

```bash
# on: laptop
claude --version
```

**Expected output** (บันทึกจริงบน Mac เครื่องนี้)

```
2.1.292 (Claude Code)
```

✓ Checkpoint: เวอร์ชันของคุณเป็น 2.1.287 ขึ้นไป

## 1 · เจ็ดวิธีต่อขยาย Claude Code

plugin คือโฟลเดอร์ที่มีไฟล์ manifest `.claude-plugin/plugin.json` และจะบรรจุชิ้นส่วนเหล่านี้แบบผสมกันอย่างไรก็ได้:

| ชิ้นส่วน | ไฟล์ | ทำงานแบบ | วาด UI ได้ไหม | ตัวอย่างในคลาส |
|---|---|---|---|---|
| Skill | `skills/<name>/SKILL.md` | Markdown ที่ Claude อ่านเมื่อเหมาะกับงาน | ไม่ได้ | lab skill 20 ตัวของ `agentic-coding-fitness` |
| Settings hook | `hooks/hooks.json` → `hooks` | shell command, HTTP call หรือ prompt ที่รันตาม lifecycle event | ไม่ได้ | เพิ่มแถว CSV เมื่อ `SessionEnd` |
| MCP server | `.mcp.json` | process แยกที่เสนอ tool ให้ใช้ | ไม่ได้ | API รายชื่อนักเรียน |
| LSP server | `.lsp.json` | language server สำหรับ code intelligence | ไม่ได้ | diagnostics ของ Python ในแล็บ |
| Output style | `output-styles/<name>.md` | น้ำเสียงใน system prompt | ไม่ได้ | โทน "โค้ชฟิตเนส" |
| Status line | setting `statusLine` | script ที่พิมพ์หนึ่งบรรทัด | หนึ่งบรรทัด | % ของ context และค่าใช้จ่าย |
| **Mod** | `hooks/hooks.json` → `modules` | **ฟังก์ชันที่รันอยู่ภายใน Claude Code** | **ได้** | Mod Gym แผงของคอร์สนี้ |

แถวสุดท้ายเป็นของใหม่ Anthropic เปิดตัว mod เมื่อ 1 ตุลาคม 2026 ว่าเป็น "small TypeScript functions that change how Claude Code works. A mod can rewrite a prompt, add new UI, replace a built-in feature, or add entirely new functionality." mod คือ plugin ที่ `hooks/hooks.json` มี key `modules` และ key นี้อย่างเดียวเท่านั้นที่ทำให้ plugin กลายเป็น mod

สิ่งที่มีแต่ mod ทำได้:

- วาด pane และ band ของตัวเอง พร้อมปุ่ม แท็บ และช่องกรอกข้อความ
- วาดแถวของ Claude Code เองใหม่ เช่น spinner
- พักการเรียก tool ไว้ระหว่างถามคำถามคุณ
- รัน `/command` โดยไม่ต้องมี turn ของ Claude เลย
- แชร์ค่าระหว่าง hook ของตัวเอง และเก็บค่าไว้ข้ามการ reload

✓ Checkpoint: คุณบอกได้ว่าจะใช้ชิ้นส่วนไหนสำหรับ "checklist ที่ Claude อ่าน", "บันทึกทุก session ลง CSV" และ "แผงที่มีปุ่ม"

## 2 · ไอเดียเดียว: `($, e, next)`

hook ของ mod ทุกตัวมีรูปเดียวกัน:

```ts
on('tool.call', ($, e, next) => next(e))
```

- `e` คือ input ของ event เป็นค่าธรรมดาที่ถูก freeze ไว้ (ในที่นี้คือ tool ตัวไหน กับ argument อะไร)
- `next(e)` จะรัน plugin ที่อยู่ชั้นล่างกว่าคุณ แล้วตามด้วยพฤติกรรมของ Claude Code เอง และ resolve เป็นผลลัพธ์ของ event
- `$` คือทุกอย่างที่อยู่นอก module ของคุณ: หน้าจอ (`$.ui`), store (`$.store`), ไฟล์ (`$.fs`), โปรแกรม (`$.process`), โมเดล (`$.model`) และอื่น ๆ mod ไม่มี Node และไม่มี DOM มันเข้าถึงโลกภายนอกได้ผ่าน `$` เท่านั้น

สิ่งที่คุณทำรอบ ๆ `next` คือสิ่งที่กำหนดว่า hook นั้นเป็นแบบไหน:

| คุณเขียน | Pattern |
|---|---|
| `return next(e)` | **observe** |
| `return next({ ...e, x })` | **rewrite** สิ่งที่ส่วนที่เหลือของ chain จะเห็น |
| `const r = await next(e); …; return r` | **wrap**: ลงมือทำบางอย่างกับผลลัพธ์ |
| `return { … }` โดยไม่เรียก `next` | **answer** แทน Claude Code |

Lesson 03 ว่าด้วยสี่แบบนี้ทั้งบท ตอนนี้ให้สังเกตว่า mod คือ middleware ไอเดียเดียวกับ agent loop และ tool guard ที่คุณสร้างใน Week 4–5 แต่ตอนนี้ครอบอยู่รอบ coding agent ที่คุณใช้ทุกวัน

## 3 · ความไว้ใจ: อ่าน mod ก่อนโหลด

mod **ไม่ได้อยู่ใน sandbox** มันรันด้วยสิทธิ์ของคุณ อ่านไฟล์และความลับของคุณได้ อนุมัติการเรียก tool ได้ และใช้โควต้า usage ของคุณได้ นิสัยแรกจึงเป็นการ audit คำสั่ง `claude plugin validate <dir>` อ่าน manifest และซอร์สของ module แบบเดียวกับที่ engine จะอ่าน แล้วพิมพ์ว่า module hook อะไรและเรียกอะไรบ้าง

audit Mod Gym ก่อนติดตั้ง:

```bash
# on: laptop (repo root)
claude plugin validate week29_claude_code_mods/mod-gym
```

**Expected output** (บันทึกจริงบน Mac เครื่องนี้ ตัดบางส่วน)

```
Validating hooks: …/week29_claude_code_mods/mod-gym/hooks/hooks.json

  ❯ ./register.tsx hooks: session.start, prompt.submit, tool.call, turn.complete, tool.call{tool=mcp__mod-gym__gym_status}, command.run{command=mod-gym}, ui.render{component=AbovePrompt}, ui.render{component=Pane, requestId=mod-gym}
  ❯ ./register.tsx answers its own command: command.run{command=mod-gym}
  ❯ ./register.tsx gating hook without .catch: prompt.submit
  ❯ ./register.tsx gating hook without .catch: tool.call
  ❯ ./register.tsx gating hook with .catch: tool.call{tool=mcp__mod-gym__gym_status}
  ❯ ./register.tsx calls: $.clock.now, $.command.register, $.env.get (via home), $.fs.exists (via runOne), $.fs.list (via runOne), $.fs.read (via runOne), $.process.run (via runOne), $.prompt.fill (via askClaude), $.session.version (via runOne), $.state.get, $.state.set, $.store.get, $.store.set, $.tool.register, $.ui.ask, $.ui.close, $.ui.copy, $.ui.open, $.ui.resolve, $.ui.status (via refreshStatus), $.ui.toast
  ❯ ./register.tsx env writes: nothing
  ❯ ./register.tsx env reads: HOME
  …
✔ Validation passed
```

วิธีอ่าน:

- **hooks:** ทุก event ที่ mod ฟังอยู่ `tool.call` ที่ไม่มี matcher แปลว่ามันเห็นการเรียก tool *ทุกครั้ง* (Mod Gym แค่จับเวลาไว้แสดงในเทป Events)
- **calls:** ทุกเมธอดของ `$` ที่มันเข้าถึงได้ และเข้าถึงผ่านฟังก์ชันไหนของมัน ให้ถามกับทุกบรรทัดว่า: ฉันคาดไว้แล้วหรือเปล่า Mod Gym อ่านไฟล์และรัน `$.process.run` (เพื่อรัน `claude plugin validate` และ `claude plugin test` กับ lab mod ของคุณเท่านั้น) และอ่าน environment variable ตัวเดียวคือ `HOME` มันไม่เรียก `$.http` หรือ `$.model` เลย
- **gating hook:** hook ที่อยู่ในจุดที่คำตอบของมันปฏิเสธบางอย่างได้ gating hook ที่ throw จะถูกข้ามไป ดังนั้นสำหรับ guard คุณต้องการ `with .catch` (Lesson 04)

✓ Checkpoint: คุณชี้ได้ว่าบรรทัดไหนบอกว่า mod รันโปรแกรมบนเครื่องคุณได้หรือไม่

## 4 · ของที่มีมาให้ซึ่งควรรู้จัก

| Built-in | ให้อะไรคุณ |
|---|---|
| `/plugin-authoring` | skill ในตัวที่สอน Claude ให้เขียน mod Claude จะเขียนลง `~/.claude/dev-mods/<session-id>/` และเสนอ hot reloading |
| `/plugin` | เรียกดู ติดตั้ง เปิดและปิด plugin แท็บ Installed จะนับ mod ที่ทำงานอยู่ |
| `/reload-plugins` | อ่าน plugin ที่ติดตั้งไว้ใหม่ รวมถึงตัวที่ติดตั้งจากโฟลเดอร์ในเครื่อง |
| `claude plugin validate <dir>` | การ audit และการตรวจแบบ static |
| `claude plugin test <dir>` | รัน `*.test.ts` ของ mod โดยไม่ต้องมี session ไม่ต้อง sign-in ไม่ต้องใช้ network (Lesson 08) |

## 5 · ติดตั้ง Mod Gym

โฟลเดอร์ `week29_claude_code_mods/` ของ repo นี้เป็น plugin marketplace (`week29_claude_code_mods/.claude-plugin/marketplace.json` ชื่อ `fitness-week29`) ให้เพิ่ม marketplace นี้ แล้วติดตั้ง Mod Gym ที่ user scope:

```bash
# on: laptop (repo root)
claude plugin marketplace add ./week29_claude_code_mods
claude plugin install mod-gym@fitness-week29 --scope user
claude plugin list
```

**Expected output** ของคำสั่งสุดท้าย (บันทึกจริงบน Mac เครื่องนี้ เฉพาะรายการ Mod Gym)

```
  ❯ mod-gym@fitness-week29
    Version: 0.1.0
    Read from: /Users/altodev/Desktop/agenticaicodingfitness/week29_claude_code_mods/mod-gym
    Scope: user
    Status: ✔ enabled
```

`Read from:` ระบุโฟลเดอร์ใน repo โดยตรง plugin ที่ติดตั้งจาก marketplace แบบโฟลเดอร์จะถูกอ่านจากโฟลเดอร์นั้นเอง หลัง `git pull` คุณจึงแค่รัน `/reload-plugins` ใน session ไม่ต้องติดตั้งใหม่ ไม่ต้องขยับเวอร์ชัน

เมื่อ repo ของคลาสบน GitHub มี Mod Gym แล้ว นักเรียนที่มี marketplace ของคอร์สอยู่แล้วให้อัปเดตแทน ที่ prompt ของ Claude Code:

```text
/plugin marketplace update agentic-coding-fitness
/plugin install mod-gym@agentic-coding-fitness
```

เลือกแหล่งเดียว อย่าใช้ทั้งสองแหล่ง เพราะ plugin ชื่อเดียวกันสองชุดจะพยายามโหลดทั้งคู่

เริ่ม session ใหม่ของ Claude Code (หรือรัน `/reload-plugins` ใน session ที่เปิดอยู่) เพื่อให้ mod โหลด

✓ Checkpoint: `claude plugin list` แสดง `mod-gym` พร้อม `Status: ✔ enabled`

## 6 · เดินชม

พิมพ์ `/mod-gym` ที่ prompt แผงจะเปิดขึ้นพร้อมรับคีย์บอร์ด

```text
🏋 Mod Gym Week 29 · Claude Code mods · 0/8 lessons passed
[ Lesson ] [ Events ] [ Progress ]
Lesson 01 of 08 · What a mod is
…lesson text…
Checks
· Claude Code is 2.1.287 or newer
Long form: week29_claude_code_mods/01_what_is_a_mod/TUTORIAL.md · lab: ~/mod-gym-lab/hello-mod
[ Prev ] [ Next ] [ Check ] [ Ask Claude ] [ Close ]
```

(layout ข้างบนเป็นภาพร่าง สีและความกว้างในเทอร์มินัลของคุณจะต่างไป)

| คีย์ | ทำอะไร |
|---|---|
| `1` `2` `3` | แท็บ **Lesson**, **Events** และ **Progress** |
| `p` / `n` | บทก่อนหน้า / บทถัดไป |
| `c` | **Check**: รันตัวตรวจของบทนี้กับ lab mod ของคุณ แล้วแสดง ✓ / ✗ พร้อมเหตุผล |
| `a` | **Ask Claude**: ใส่คำถามโค้ชของบทนี้ลงใน prompt ของคุณ แก้ได้หรือกด Enter เลย |
| `f` | **Copy starter** (เฉพาะ Lesson 02): คัดลอก shell script ที่สร้าง `hello-mod` |
| `x` | ล้างเทป Events (แท็บ Events) |
| `r` | รีเซ็ตความคืบหน้า (แท็บ Progress จะถามก่อน) |
| `q` หรือ Esc | ปิดแผง |

**แท็บ Events** คือ hook ของ Mod Gym เองแบบสด ส่ง prompt อะไรก็ได้ แล้วแถวจะปรากฏ:

| เครื่องหมาย | Pattern | ตัวอย่างแถว |
|---|---|---|
| `◦` | observe: `return next(e)` | `prompt.submit` พร้อมคำแรก ๆ ของ prompt คุณ |
| `⟳` | wrap: `await next(e)` แล้วลงมือทำ | `tool.call` พร้อมชื่อ tool, argument, ผลลัพธ์ (`ok`, `error`, `denied`) และเวลาที่ใช้ |
| `◆` | answer: ไม่มี `next` | (tool `gym_status` ของ Mod Gym เอง) |

**band** คือบรรทัดเดียวเหนือ prompt ของคุณ: `🏋 Mod Gym L01 What a mod is · 0/8 passed [Open] [Hide]` Mod Gym วาดมันไว้ *ใต้* สิ่งที่ plugin อื่นวาดไว้ใน band แทนที่จะวาดทับ ซ่อนได้ด้วยปุ่มของมันหรือ `/mod-gym band off` เรียกกลับด้วย `/mod-gym band on`

**status line** แสดง `Mod Gym L01 · 0/8`

**tool `gym_status`** Mod Gym ลงทะเบียน tool ที่ Claude เห็นเป็น `mcp__mod-gym__gym_status` มันคืนค่าบทเรียนของคุณ, path ของ tutorial, โฟลเดอร์แล็บ และผลตรวจล่าสุด พร้อมหมายเหตุ "ask one question at a time, let the student type the code" ดังนั้นเมื่อคุณกด **Ask Claude**, Claude จะเปิดดูได้ว่าคุณอยู่ตรงไหน แล้วโค้ชแทนที่จะเขียนแล็บให้คุณ

**โฟลเดอร์แล็บของคุณ** ตัวตรวจจะหา mod ของคุณที่ `<lab>/hello-mod` ค่าเริ่มต้นของ lab คือ `~/mod-gym-lab` เปลี่ยนได้ด้วย:

```text
/mod-gym lab ~/code/my-mods
```

คำสั่งอื่น: `/mod-gym next`, `/mod-gym prev`, `/mod-gym 3` (กระโดดไปบท 3), `/mod-gym check`, `/mod-gym status`, `/mod-gym reset`

✓ Checkpoint: แท็บ Events มีแถว `⟳ tool.call` จาก session ของคุณอย่างน้อยหนึ่งแถว

## ในแผง Mod Gym

รัน `/mod-gym 1` แล้วกด **c** Lesson 01 มีตัวตรวจหนึ่งข้อ

**Expected output** (`/mod-gym check` บันทึกจริงบน Mac เครื่องนี้)

```
Lesson 01 · What a mod is · 1/1 checks
✓ Claude Code is 2.1.287 or newer (running 2.1.292)
```

เมื่อผ่านครบ บทนั้นจะถูกทำเครื่องหมาย ✓ ในแท็บ Progress และมี toast ขึ้นว่า `Mod Gym: lesson 01 passed 💪`

## ลองทำเอง — audit mod ของคนแปลกหน้า

`week29_claude_code_mods/mod-agentic-coding-fitness/` เป็น mod ติดตามความคืบหน้าที่ร่างขึ้นจากบันทึกการค้นคว้าของสัปดาห์นี้ และยังไม่เคยถูกรันเลย **อย่า**ติดตั้งมัน ให้ audit มัน:

```bash
# on: laptop (repo root)
claude plugin validate week29_claude_code_mods/mod-agentic-coding-fitness
```

ตอบสามคำถามโดยดูจากผลลัพธ์อย่างเดียว:

1. matcher ของ hook ตัวไหนที่ validator อ่านไม่ออก และทำไมเรื่องนี้สำคัญต่อการ audit
2. การเรียกตัวไหนที่ใช้ usage ของ Claude ของคุณ
3. hook ตัวไหนบ้างที่ปฏิเสธบางอย่างได้ และตัวไหนจะปล่อยผ่านถ้ามัน throw

<details><summary>เฉลย</summary>

**Expected output** (บันทึกจริงบน Mac เครื่องนี้ ตัดบางส่วน)

```
  ❯ ./register.js hooks: session.start, tool.call{tool=?}, tool.call{tool=Bash}, turn.complete, prompt.submit, command.run{command=fitness}, command.run{command=fitness-log}, command.run{command=fitness-next}, command.run{command=fitness-grade}, ui.render{component=Spinner}, ui.render{component=Pane}
  …
  ❯ ./register.js gating hook without .catch: tool.call{tool=?}
  ❯ ./register.js gating hook without .catch: tool.call{tool=Bash}
  ❯ ./register.js gating hook without .catch: prompt.submit
  ❯ ./register.js calls: $.command.register, $.model.complete, $.session.usage, $.store.get, $.store.set (via save), $.tool.register, $.ui.close, $.ui.invalidate, $.ui.open, $.ui.resolve, $.ui.status

✔ Validation passed
```

1. `tool.call{tool=?}`: matcher เป็นตัวแปร `LOG_TOOL` ไม่ใช่ string literal การ audit จึงบอกไม่ได้ว่ามันตอบ tool ตัวไหน คุณต้องไปอ่านซอร์สเอง ควรเลือก mod ที่ matcher เป็น literal
2. `$.model.complete` (คำสั่ง `/fitness-grade` ขอให้ Haiku ให้คะแนนบทสะท้อนความคิด)
3. สามบรรทัดที่เป็น `gating hook without .catch` ไม่มีตัวไหนเป็น guard ที่ปฏิเสธอะไร การ fail open จึงรับได้ในกรณีนี้ แต่สำหรับ guard จริงจะรับไม่ได้ (Lesson 04)

"Validation passed" แปลว่า engine จะโหลดมัน ไม่ได้แปลว่า mod ปลอดภัยหรือถูกต้อง

</details>

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `/mod-gym` ไม่ใช่คำสั่ง | session เริ่มก่อนการติดตั้ง | รัน `/reload-plugins` หรือเริ่ม session ใหม่ |
| `claude plugin install` บอกว่าไม่พบ plugin | ไม่ได้เพิ่ม marketplace จาก repo root | รัน `claude plugin marketplace add ./week29_claude_code_mods` จาก repo root แล้วติดตั้งอีกครั้ง |
| `mod-gym` สองรายการใน `claude plugin list` | ติดตั้งจากทั้ง `fitness-week29` และ `agentic-coding-fitness` | ถอนการติดตั้งหนึ่งตัว: `claude plugin uninstall mod-gym@fitness-week29` |
| band ไม่ขึ้น | ถูกซ่อนไว้ หรือ plugin อื่นที่อยู่ชั้นบนวาด band โดยไม่ส่งต่อ | `/mod-gym band on` ส่วน status line และ `/mod-gym` ยังใช้ได้ |
| แผงเปิดแบบ inline ใต้ transcript ไม่ใช่ sidebar | เทอร์มินัลไม่ได้อยู่ใน layout แบบ fullscreen | เป็นเรื่องปกติ: เฉพาะ layout แบบ fullscreen ที่วาง pane ไว้ข้าง transcript แผงทำงานเหมือนกันเมื่อเป็น inline |

## ต่อไป

[Lesson 02 — mod ตัวแรกของคุณ: สามไฟล์](../02_first_mod/TUTORIAL.th.md): เขียน `hello-mod` ตรวจด้วย validate แล้วดูข้อความของคุณเองบน spinner ของ Claude Code
