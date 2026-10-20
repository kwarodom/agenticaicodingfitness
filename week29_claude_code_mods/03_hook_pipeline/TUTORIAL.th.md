# ▶ Mod Gym Lesson 03 — The hook pipeline: observe, rewrite, wrap, answer

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณจะเรียนวิธีเปลี่ยนหน้าตาและพฤติกรรมของ Claude Code เอง จากในเทอร์มินัลที่คุณเขียนโค้ดอยู่ทุกวัน บทเรียนชุดนี้คือฉบับยาว ส่วน **🏋 Mod Gym** ซึ่งเป็น panel ใน Claude Code คือฉบับสั้นที่มีการตรวจงานให้ ตลอดบทเรียนคุณจะค่อย ๆ สร้าง lab mod ตัวเดียวชื่อ `hello-mod` ฉบับที่เสร็จสมบูรณ์อยู่ที่ `week29/reference/hello-mod`

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- อ่าน hook ทุกตัวในรูป `($, e, next)` และเรียนรู้สี่อย่างที่ hook ทำกับ `next` ได้: **observe**, **rewrite**, **wrap**, **answer**
- เพิ่ม hook สามตัวให้ `hello-mod`: ตัวนับ tool call (wrap), context หนึ่งบรรทัดในทุก prompt (rewrite) และการตรวจตอนจบแต่ละ turn (observe)
- พิสูจน์ว่ามันทำงานอย่างที่ว่าด้วย `claude plugin test` โดยไม่ต้องมี session และไม่ต้องใช้เน็ต
- หาคำตอบว่า เมื่อมีหลาย hook อยู่บน event เดียวกัน ตัวไหนทำงานก่อน ด้วยการวัดจริง
- เข้าใจว่าทำไมตัวแปรระดับ module อย่าง `calls` ถึงกลับไปเริ่มที่ศูนย์ทุกครั้งที่คุณกด save
- อ่านแท็บ **Events** ของ Mod Gym ในฐานะตัวอย่างสด: มันคือ hook ของ panel เองที่กำลังเฝ้าดู session ของคุณ

**Time** ~45 นาที · **Difficulty** เริ่มต้นถึงระดับกลาง · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้

**แหล่งอ้างอิง:** [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [React to events](https://code.claude.com/docs/en/plugins/mods/events) · [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · ไฟล์ประกาศ API ที่ Claude Code 2.1.292 เขียนให้ (`.claude-plugin/types/claude-code/index.d.ts` ซึ่งเป็นตัวตัดสินเมื่อเอกสารกับโค้ดไม่ตรงกัน)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Lesson 02 | `claude plugin validate ~/mod-gym-lab/hello-mod` ผ่าน | บทนี้เพิ่ม hook ลงใน mod ตัวนั้น |
| Claude Code 2.1.287 ขึ้นไป | `claude --version` | mods มาใน 2.1.287 |
| ติดตั้ง Mod Gym แล้ว | `/mod-gym` เปิด panel ได้ | Section 6 อ่านแท็บ Events ของมัน |

แล็บของบทนี้สร้างไว้ใน repo แล้ว คุณจึงเอา mod ของคุณไปเทียบได้:

```text
week29/03_hook_pipeline/
  labs/hello-mod-l03/              hello-mod as it stands after this lesson, with tests
  labs/order-demo/                 three hooks on one event, to measure the order
  exercises/solutions/heavy-set/   the exercise solution
```

## 1 · hook ทุกตัวมีรูปเดียว

hook ทุกตัวใน mod ทุกตัวมีรูปแบบเดียวกัน:

```ts
on('<event>', { /* optional matcher */ }, async ($, e, next) => { … })
```

| ชื่อ | คืออะไร |
|---|---|
| `$` | mods API: `$.ui`, `$.tool`, `$.store`, `$.process` … ทุกอย่างที่อยู่นอก module ของคุณ |
| `e` | input ของ event เป็นค่าธรรมดาที่ถูก freeze ไว้ (สำหรับ `tool.call`: `e.tool`, `e.command`, …) |
| `next` | ส่วนที่เหลือของ chain: plugin ที่อยู่ใต้คุณ แล้วตามด้วยพฤติกรรมของ Claude Code เอง |

`next(e)` resolve เป็น **result** ของ event สิ่งที่คุณทำรอบ ๆ การเรียกครั้งเดียวนี้คือ pattern ทั้งหมด:

| คุณเขียน | pattern | ผล |
|---|---|---|
| `return next(e)` | **observe** | ดูอย่างเดียว ไม่เปลี่ยนอะไร |
| `return next({ ...e, x })` | **rewrite** | ทุกอย่างที่อยู่ข้างใต้เห็น `e` ฉบับของคุณ |
| `const r = await next(e); …; return r` | **wrap** | ปล่อยให้มันเกิดขึ้น แล้วทำอะไรต่อกับผลลัพธ์ |
| `return { … }` โดยไม่เรียก `next` เลย | **answer** | คุณคือผลลัพธ์ ไม่มีอะไรข้างใต้ได้ทำงาน |

spinner suffix ใน Lesson 02 ก็เป็น **rewrite** อยู่แล้ว: มันส่ง `e` ฉบับสำเนาที่มี `props.suffix` ใหม่ให้ `next` แล้ว Claude Code ก็วาดฉบับสำเนานั้น

✓ Checkpoint: บอกได้ว่า Spinner hook จาก Lesson 02 เป็น pattern ไหนในสี่แบบ และเพราะอะไร

## 2 · เพิ่ม hook สามตัวให้ hello-mod

เปิด `~/mod-gym-lab/hello-mod/hooks/register.ts` แล้วแก้ให้เป็นแบบนี้ (Spinner hook มาจาก Lesson 02):

```ts
import type { Register } from 'claude-code'

export const register: Register = on => {
  // L02 · rewrite what Claude Code draws
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )

  // L03 · wrap: let every tool run, then count it
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    $.ui.status('hello-mod · ' + calls + ' tool calls')
    return r
  })

  // L03 · rewrite: add a line of context Claude reads with the prompt
  on('prompt.submit', ($, e, next) =>
    next({ ...e, context: [...(e.context ?? []), 'Reply like a fitness coach.'] }),
  )

  // L03 · observe: look at the finished turn, change nothing
  on('turn.complete', ($, e, next) => {
    if (e.durationMs > 120_000) $.ui.toast('Long set: ' + Math.round(e.durationMs / 1000) + 's')
    return next(e)
  })
}
```

อ่าน hook แต่ละตัวเทียบกับตารางใน Section 1:

- **`tool.call` (ไม่มี matcher)** ทำงานกับทุก tool ที่ Claude ใช้ มัน await `next(e)` ดังนั้น tool จะรันจริงก่อน แล้วค่อยนับและอัปเดต status line มันคืน `r` ซึ่งเป็นผลลัพธ์ของ tool เองโดยไม่แตะ ถ้าลืม `return r` Claude จะไม่ได้ผลลัพธ์ของ tool เลย
- **`prompt.submit`** ส่งสำเนาของ prompt ต่อไป โดยมีรายการเพิ่มใน `context` อีกหนึ่งรายการ context คือข้อความที่ Claude อ่านพร้อม prompt ของคุณ แต่คุณไม่เห็นใน transcript ส่วน `text` ที่คุณพิมพ์ไม่ถูกเปลี่ยน
- **`turn.complete`** อ่าน `e.durationMs` (มี `e.answer`, `e.reason`, `e.isAborted`, `e.usage` ด้วย) แล้วคืน `next(e)` โดยไม่เปลี่ยนอะไร

ลอง validate บรรทัด `hooks:` คือสิ่งที่ engine จะ hook และ `calls:` คือ method ทุกตัวบน `$` ที่ module เรียกถึงได้:

```bash
# on: laptop
claude plugin validate week29/03_hook_pipeline/labs/hello-mod-l03
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้, Claude Code 2.1.292; ตัดบรรทัดของ manifest ออก)

```
…
  ❯ ./register.ts hooks: ui.render{component=Spinner}, tool.call, prompt.submit, turn.complete
  ❯ ./register.ts gating hook without .catch: tool.call
  ❯ ./register.ts gating hook without .catch: prompt.submit
  ❯ ./register.ts calls: $.ui.status, $.ui.toast
✔ Validation passed with warnings
```

*warnings* มีแค่เรื่องไม่มี field `author` ใน `plugin.json` ส่วน **gating hook** คือ hook ที่อยู่ ณ จุดที่คำตอบของมันปฏิเสธสิ่งที่ถูกขอได้ เช่น tool call หรือ prompt คำสั่ง validate แจ้งเป็นข้อเท็จจริง ไม่ใช่ปัญหา ส่วนคำถามว่า gating hook ต้องมี `.catch` หรือไม่ เป็นเรื่องของ Lesson 04

✓ Checkpoint: `claude plugin validate ~/mod-gym-lab/hello-mod` ของคุณเองแสดง `tool.call` และ `prompt.submit` ในบรรทัด `hooks:`

## 3 · พิสูจน์ด้วย test kit

`claude plugin test` รัน `tests/*.test.ts` กับ engine จริง โดยไม่ต้องมี session ไม่ต้อง sign-in และไม่ต้องใช้เน็ต ใน test นั้น hook ที่คุณเพิ่มด้วย `on` ของ test เองจะอยู่ **ใต้** plugin ของคุณ และทำหน้าที่แทน Claude Code:

```ts
import { expect, test } from 'claude-code/testing'

// The engine fills these fields in a real session; a test spells them out.
const ORIGIN = { kind: 'composer' } as const

test('wrap: the tool result comes back unchanged and the status counts it', async ($, on) => {
  const status: string[] = []
  on('ui.status', ($, e) => { status.push(String(e.text)); return { value: undefined } })
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran: ' + e.command }))
  const r = await $.tool.call({ tool: 'Bash', command: 'ls' })
  await $.tool.call({ tool: 'Bash', command: 'pwd' })
  expect(r.result).toBe('ran: ls')
  expect(status).toEqual(['hello-mod · 1 tool calls', 'hello-mod · 2 tool calls'])
})

test('rewrite: the prompt reaches Claude Code with one more line of context', async ($, on) => {
  on('prompt.submit', ($, e) => ({ text: e.text, context: e.context }))
  const r = await $.prompt.submit({ text: 'plan my leg day', wait: false, origin: ORIGIN })
  expect(r.text).toBe('plan my leg day')
  expect(r.context).toEqual(['Reply like a fitness coach.'])
})
```

สองเรื่องที่คนมักพลาด:

- stub hook รับ `($, e)` เหมือน hook ทั่วไป ถ้าเขียน `e => …` สิ่งที่คุณได้ในชื่อ `e` คือ `$`
- การเรียก `$` ที่ plugin ทำ (ในที่นี้คือ `$.ui.status`) ตอบด้วย `{ value }` ส่วน event (`tool.call`, `prompt.submit`) ตอบด้วย result ของ event นั้นเอง

```bash
# on: laptop
claude plugin test week29/03_hook_pipeline/labs/hello-mod-l03
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้)

```
tests/pipeline.test.ts:
(pass) wrap: the tool result comes back unchanged and the status counts it [16.30ms]
(pass) rewrite: the prompt reaches Claude Code with one more line of context [6.55ms]

 2 pass
 0 fail
```

คัดลอก `tests/pipeline.test.ts` ไปไว้ใน `hello-mod/tests/` ของคุณเอง แล้วรันที่นั่นด้วย

✓ Checkpoint: test ทั้งสองผ่านกับ mod **ของคุณ**

## 4 · ใครทำงานก่อน? วัดเอาเลย

เมื่อหลาย hook ใช้ event เดียวกัน พวกมันจะซ้อนกันเป็น stack ใน `labs/order-demo` มี hook สามตัวบน `tool.call` ของ Bash: A และ B แก้ command ขาเข้า และติดป้ายที่ผลลัพธ์ขาออก ส่วน C ตอบ `whoami` เองโดยไม่เรียก `next` เลย:

```ts
export const register: Register = on => {
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const r = await next({ ...e, command: e.command + ' >A' })
    return { result: String(r.result) + ' <A' }
  })
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const r = await next({ ...e, command: e.command + ' >B' })
    return { result: String(r.result) + ' <B' }
  })
  // answer: C never calls next, so nothing beneath it runs
  on('tool.call', { tool: 'Bash' }, ($, e, next) =>
    e.command.startsWith('whoami') ? { result: 'coach (answered by C)' } : next(e),
  )
}
```

Bash stub ของ test ทำหน้าที่แทน tool จริง และสะท้อน command ที่มันได้รับในที่สุดกลับมา:

```ts
test('the first hook registered is the outermost', async ($, on) => {
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran [' + e.command + ']' }))
  const r = await $.tool.call({ tool: 'Bash', command: 'ls' })
  expect(r.result).toBe('ran [ls >A >B] <B <A')
  const w = await $.tool.call({ tool: 'Bash', command: 'whoami' })
  expect(w.result).toBe('coach (answered by C) <B <A')
})
```

```bash
# on: laptop
claude plugin test week29/03_hook_pipeline/labs/order-demo
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้)

```
tests/order.test.ts:
(pass) the first hook registered is the outermost [15.87ms]

 1 pass
 0 fail
Ran 1 test across 1 file. [0.15s]
```

assertion ที่ผ่านบอกอะไรเรา:

```text
call ──► A ──► B ──► C ──► (Claude Code's Bash)
            >A     >B        ran [ls >A >B]
result ◄── <A ◄── <B ◄───────────┘
```

- ภายใน plugin เดียวกัน **hook ที่ลงทะเบียนก่อนคือตัวนอกสุด** มันเห็น call เป็นตัวแรก และเห็นผลลัพธ์เป็นตัวสุดท้าย
- การ rewrite (`>A`) ถูกเห็นโดยทุกอย่างที่อยู่ข้างใต้ และไม่เคยถูกเห็นโดยอะไรที่อยู่ข้างบน
- การ answer (C กับ `whoami`) หยุดไม่ให้ call ลงไปต่อ แต่ hook ที่อยู่ **ข้างบน** ยังคง wrap ผลลัพธ์ของมัน (`<B <A`)
- **ระหว่าง plugin** engine ซ้อนพวกมันเป็นชั้น (tier): `prepend` (ที่องค์กรวาง policy plugin ของตัวเอง), `user`, `append`, `builtin`, `core` อย่าสร้างอะไรที่พึ่งลำดับระหว่าง user plugin ธรรมดาสองตัว ให้เขียน hook แต่ละตัวให้ถูกต้องไม่ว่ามันจะไปอยู่ตรงไหน

✓ Checkpoint: ทายได้ก่อนรันว่า `cat` จะคืนอะไรเมื่อผ่าน `order-demo`

## 5 · ทำไม `calls` ถึงกลับไปเป็นศูนย์

กด save `register.ts` ขณะที่มี session โหลด mod ของคุณอยู่ (หรือรัน `/reload-plugins`) status line จะเริ่มนับใหม่ที่ `1 tool calls` การ reload คือ **การโหลด module ใหม่ทั้งหมด**: `register` รันใหม่ใน environment ใหม่ ตัวแปรระดับ module หรือใน closure ทุกตัว (`let calls = 0`) จึงเริ่มใหม่ และ `session.start` ก็ fire อีกครั้ง

| ค่าอยู่ที่ไหน | รอดจาก save / reload | รอดข้าม session |
|---|---|---|
| `let` ใน module ของคุณ | ไม่ | ไม่ |
| `$.state` (atom) | ใช่ | ไม่ |
| `$.store` | ใช่ | ใช่ (ต่อเครื่อง) |

Lesson 06 จะย้าย `calls` ไปไว้ใน atom ระหว่างนี้ให้ถือว่าตัวนับคือ "นับตั้งแต่ save ครั้งล่าสุด"

✓ Checkpoint: อธิบายได้ว่าทำไมตัวนับถึงตกเป็นศูนย์หลัง save และบอกได้ว่าที่ไหนจะเก็บค่าไว้ได้

## 6 · ใน Mod Gym panel

Mod Gym สร้างจาก pattern สี่แบบเดียวกันนี้ ผลลัพธ์ `claude plugin validate` ของมันแสดง hook ที่มันใส่ลงใน session ของคุณ:

```bash
# on: laptop
claude plugin validate week29/mod-gym
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้; แสดงสองบรรทัดแรกของ `❯ ./` และตัดบรรทัดหลังจากนั้น)

```
…
  ❯ ./register.tsx hooks: session.start, prompt.submit, tool.call, turn.complete, tool.call{tool=mcp__mod-gym__gym_status}, command.run{command=mod-gym}, ui.render{component=AbovePrompt}, ui.render{component=Pane, requestId=mod-gym}
  ❯ ./register.tsx answers its own command: command.run{command=mod-gym}
…
```

- `prompt.submit` และ `turn.complete` ทำหน้าที่ **observe**: เขียนหนึ่งแถวลง Events tape แล้ว `return next(e)`
- `tool.call` (ไม่มี matcher) ทำหน้าที่ **wrap**: จับเวลา `await next(e)` แล้วบันทึกชื่อ tool, argument สั้น ๆ, `ok` / `error` / `denied` และจำนวนมิลลิวินาที
- `tool.call{tool=mcp__mod-gym__gym_status}` ทำหน้าที่ **answer**: Mod Gym ให้บริการ tool ของตัวเองโดยไม่เรียก `next`

ทีนี้ดูของจริง:

1. พิมพ์ `/mod-gym 3` (กระโดดไปบทนี้) แล้ว `/mod-gym` เพื่อเปิด panel
2. กด **2** เพื่อไปแท็บ *Events* แล้วส่ง prompt อะไรก็ได้ที่ทำให้ Claude ใช้ tool เช่น "list the files in week29"
3. อ่านแต่ละแถว: `◦` คือ observe, `⟳` คือ wrap, `◆` คือ answer แถว `tool.call` แต่ละแถวแสดงเวลาที่ tool ใช้ **ภายใน** `await next(e)` ของ Mod Gym
4. กด **1** แล้ว **c** Mod Gym จะตรวจว่า `register.ts` ของคุณ hook `tool.call`, await `next(`, rewrite `prompt.submit` และ `claude plugin validate` แสดง `tool.call`
5. กด **a** เพื่อใส่คำถามโค้ชลงใน prompt ของคุณ มันจะขอให้ Claude ติดป้าย pattern ให้ hook แต่ละตัวของคุณ

✓ Checkpoint: การตรวจของ Lesson 03 ทั้งสี่ข้อขึ้น ✓ ใน panel

## ลองทำเอง

เพิ่ม **คำเตือน heavy set**: เมื่อ turn หนึ่งใช้ tool call มากกว่า 5 ครั้ง ให้แสดง toast ตอนจบ turn คุณต้องมี hook สามตัว: ตัวหนึ่งเริ่มนับใหม่เมื่อมีการส่ง prompt ตัวหนึ่งนับ tool call และอีกตัวอ่านค่าที่นับได้เมื่อ turn จบ จากนั้นเขียน test ที่ส่ง prompt เรียก tool 6 ครั้ง แล้วจบ turn

<details><summary>เฉลย — <code>exercises/solutions/heavy-set</code></summary>

```ts
import type { Register } from 'claude-code'

export const register: Register = on => {
  let turnCalls = 0

  // observe: a new prompt starts a new set
  on('prompt.submit', ($, e, next) => {
    turnCalls = 0
    return next(e)
  })

  // wrap: count each tool call of the turn
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    turnCalls += 1
    return r
  })

  // observe the end of the turn; toast after a heavy one
  on('turn.complete', ($, e, next) => {
    if (turnCalls > 5) $.ui.toast('Heavy set: ' + turnCalls + ' tool calls in one turn')
    return next(e)
  })
}
```

```ts
import { expect, test } from 'claude-code/testing'

test('a turn with 6 tool calls raises one toast', async ($, on) => {
  const toasts: string[] = []
  on('ui.toast', ($, e) => { toasts.push(String(e.text)); return { value: undefined } })
  on('prompt.submit', ($, e) => ({ text: e.text }))
  on('tool.call', () => ({ result: 'ok' }))
  on('turn.complete', ($, e) => ({ text: e.answer }))

  await $.prompt.submit({ text: 'refactor everything', wait: false, origin: { kind: 'composer' } })
  for (let i = 0; i < 6; i++) await $.tool.call({ tool: 'Read', file_path: '/lab/f' + i })
  await $.turn.complete({ turnId: 't1', answer: 'done', durationMs: 900, isAborted: false, reason: 'answer' })
  expect(toasts).toEqual(['Heavy set: 6 tool calls in one turn'])
})
```

```bash
# on: laptop
claude plugin test week29/03_hook_pipeline/exercises/solutions/heavy-set
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้)

```
tests/heavy.test.ts:
(pass) a turn with 6 tool calls raises one toast [17.14ms]

 1 pass
 0 fail
```

ทำไมการ reset ถึงอยู่ใน `prompt.submit` ไม่ใช่ `turn.complete`: turn อาจจบด้วยการถูก abort และ prompt ถัดไปคือช่วงเวลาเดียวที่คุณรู้แน่ว่า set ใหม่เริ่มแล้ว

</details>

✓ Checkpoint: test heavy-set ของคุณผ่าน

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| Claude บอกว่า tool ไม่คืนอะไรเลยหลังคุณเพิ่ม `tool.call` | hook ไม่ได้ `return` ผลลัพธ์ของ `next(e)` | `const r = await next(e); …; return r` |
| ใน test มี `hook was skipped: … no implementation for …` | plugin เรียกบางอย่างที่ไม่มีอะไรข้างใต้ตอบ | เพิ่ม stub ด้วย `on('<event>', ($, e) => …)` |
| stub อ่าน `e.text` ได้ `undefined` | เขียนเป็น `e => …` ดังนั้น `e` จริง ๆ คือ `$` | เขียนเป็น `($, e) => …` |
| ตัวนับใน status ตกเป็น 1 หลัง save | การ reload เริ่ม module ใหม่ | ปกติ Lesson 06 จะเก็บไว้ใน `$.state` |

## Next

[Lesson 04 — Guards: deny, hold and catch](../04_guards/TUTORIAL.md): hook ที่ **answer** `tool.call` ด้วย `{ deny }` จะหยุดไม่ให้ tool ทำงาน Lesson 04 สร้าง guard กัน force push และแสดงด้วยความล้มเหลวที่วัดได้จริงว่า ลำดับของ hook ที่คุณเพิ่งเรียนไป เป็นตัวตัดสินว่า guard จะกันอยู่หรือไม่
