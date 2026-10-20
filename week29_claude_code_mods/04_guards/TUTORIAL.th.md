# ▶ Mod Gym Lesson 04 — Guards: deny, hold and catch

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณจะเรียนวิธีเปลี่ยนหน้าตาและพฤติกรรมของ Claude Code เอง จากในเทอร์มินัลที่คุณเขียนโค้ดอยู่ทุกวัน บทเรียนชุดนี้คือฉบับยาว ส่วน **🏋 Mod Gym** ซึ่งเป็น panel ใน Claude Code คือฉบับสั้นที่มีการตรวจงานให้ ตลอดบทเรียนคุณจะค่อย ๆ สร้าง lab mod ตัวเดียวชื่อ `hello-mod` ฉบับที่เสร็จสมบูรณ์อยู่ที่ `week29_claude_code_mods/reference/hello-mod`

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- หยุด tool call ก่อนที่มันจะรัน ด้วยการ **answer** `tool.call` เป็น `{ deny }`
- **พัก (hold)** call ที่เสี่ยงไว้แล้วถามคนก่อนด้วย `$.ui.ask` และเรียนรู้ว่าจะเกิดอะไรขึ้นเมื่อไม่มีใครตอบได้ (`claude -p`)
- ทำให้ guard **fail closed** ด้วย `.catch` และ `next.called` แล้วอ่านบรรทัด `gating hook with .catch` ใน `claude plugin validate`
- จำลองความล้มเหลวจริงที่เราวัดได้ระหว่างสร้างคอร์สนี้: observer ที่ดูไม่มีพิษภัยปล่อยให้ `git push --force` ผ่านไปได้ แล้วแก้สามวิธี และพิสูจน์แต่ละวิธีด้วย test
- เชื่อมโยงกับ sample mod `blast-radius` และ mutation gate สำหรับ sandbox ของ AltoTech Reef

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้

**แหล่งอ้างอิง:** [React to events](https://code.claude.com/docs/en/plugins/mods/events) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · [Manage mods](https://code.claude.com/docs/en/plugins/mods/admin) · [claude-code-playground sample mods (`blast-radius`)](https://github.com/anthropics/claude-code-playground/tree/main/claude-code/mods) · ไฟล์ประกาศ API ที่ Claude Code 2.1.292 เขียนให้ (`.claude-plugin/types/claude-code/index.d.ts` ซึ่งเป็นตัวตัดสินเมื่อเอกสารกับโค้ดไม่ตรงกัน)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Lesson 03 | บอกได้ว่าเมื่อสอง hook ใช้ event เดียวกัน ตัวไหนอยู่นอกสุด | บั๊กของบทนี้เป็นบั๊กเรื่องลำดับ |
| `hello-mod` ที่มี hook ของ Lesson 03 | `claude plugin test ~/mod-gym-lab/hello-mod` ผ่าน | คุณจะเพิ่ม guard ลงไปในนั้น |
| git repo สำหรับทดลอง (ไม่บังคับ) | `git init /tmp/push-lab` | ไว้ลอง guard แบบสดโดยไม่แตะ remote จริง |

แล็บสร้างไว้ใน repo แล้ว สามตัวในนั้น **ตั้งใจให้ fail**: เพื่อแสดงบั๊ก

```text
week29_claude_code_mods/04_guards/
  labs/guard-without-catch/   a guard with no .catch            (test fails: the push runs)
  labs/guard-before/          observer with .catch, then guard  (test fails: the push runs)
  labs/guard-after/           guard first, observer without it  (test passes)
  labs/two-guards/            two guards stacked on Bash        (refuses, but with the wrong reason)
  exercises/solutions/one-guard/
```

## 1 · Deny: guard คือ hook ที่ answer

ใน Lesson 03 hook ที่คืนค่าโดยไม่เรียก `next` คือการ **answer** แทน Claude Code บน `tool.call` คำตอบ `{ deny: 'reason' }` หมายความว่า tool จะไม่รันเลย ไม่มี permission prompt และ Claude จะได้รับเหตุผลของคุณเป็นผลลัพธ์ของ tool จึงเปลี่ยนแผนได้

```ts
const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/

on('tool.call', { tool: 'Bash' }, ($, e, next) =>
  FORCE_PUSH.test(e.command)
    ? { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    : next(e),
)
```

เขียนเหตุผลให้ Claude อ่าน ไม่ใช่ให้ตัวคุณเองอ่าน: บอกว่าอะไรถูกบล็อกและควรทำอะไรแทน ถ้าเขียนแค่ "denied" Claude จะลองทำเรื่องเดิมซ้ำด้วยการสะกดแบบอื่น

matcher `{ tool: 'Bash' }` สำคัญ ข้างใน hook `e.command` ถูกระบุชนิดเป็น string ได้ก็เพราะ matcher จำกัด `e` ให้เป็น input ของ tool Bash

✓ Checkpoint: บอกได้ว่า Claude เห็นอะไรเมื่อ guard นี้ deny call (เหตุผลนั้น ในฐานะผลลัพธ์ของ tool)

## 2 · Hold: ถามก่อนปฏิเสธ

การปฏิเสธ force push ทุกครั้งเป็นวิธีที่หยาบไป บางครั้งนักเรียนตั้งใจจริง ๆ `$.ui.ask` จะพัก call ไว้แล้วแสดงกล่องคำถามของ Claude Code เอง:

```ts
on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
  if (FORCE_PUSH.test(e.command)) {
    const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
    if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
  }
  return next(e)
})
```

ข้อเท็จจริงสามข้อเกี่ยวกับ `$.ui.ask` ทั้งหมดมาจากคำประกาศของมันใน types ของ 2.1.292:

- มัน resolve เป็น label ที่คนเลือก หรือเป็นข้อความอิสระที่พิมพ์ใต้ "Other" นั่นคือเหตุผลที่โค้ดเทียบกับ `'Push anyway'` แบบตรงตัว และถือว่าอย่างอื่นทั้งหมดคือ "ไม่"
- ตัวมันเองคือ **tool call**: เป็น `tool.call` ของ `AskUserQuestion` ที่วาดโดย `ui.render` บน `AskUserQuestion` hook อื่นบน `tool.call` จึงเห็นมันผ่านไปด้วย Section 5 พูดถึงเรื่องนี้โดยตรง
- มันจะ **reject** เมื่อคนปิดกล่องคำถามทิ้ง และ reject เสมอใน `claude -p` ซึ่งไม่มีใครให้ถาม

นี่คือ pattern ของ sample mod `blast-radius` ใน `claude-code-playground`: มันพัก `rm -rf` และ force push ไว้ แสดงว่าจะกระทบอะไรบ้าง แล้วให้เลือก Proceed / Cancel

**ลองแบบสด (ไม่บังคับ)** เพิ่ม hook นี้ลงใน `hello-mod` ของคุณ เริ่ม `claude --plugin-dir ~/mod-gym-lab/hello-mod` ใน repo ทดลอง แล้วขอให้ Claude รัน `git push --force` คุณจะได้คำถามก่อนที่อะไรจะรัน เลือก **Cancel**: push จะไม่รัน และ Claude จะได้รับเหตุผลที่คุณ deny เป็นผลลัพธ์ของ tool

✓ Checkpoint: บอกได้สองสถานการณ์ที่ `$.ui.ask` reject แทนที่จะคืน label

## 3 · เมื่อ guard ล้มเอง

hook ที่ throw จะถูก **ข้าม (skipped)** และ chain ก็ทำงานต่อโดยไม่มีมัน สำหรับ observer นั่นถูกต้องพอดี แต่สำหรับ guard มันคือรูรั่ว: ถ้า `$.ui.ask` reject, guard จะ throw, ถูกข้าม แล้ว push ก็รัน

`labs/guard-without-catch` มี guard ข้างบนโดยไม่มี `.catch` test ของมันทำหน้าที่แทน tool Bash ของ Claude Code ด้วย `() => ({ result: 'pushed' })` และไม่ตอบอย่างอื่นเลย ดังนั้น `$.ui.ask` จะ reject แบบเดียวกับที่เกิดใน `claude -p`:

```ts
test('a force push is refused when no one can answer the question', async ($, on) => {
  // stands in for Claude Code's Bash tool. Nothing answers AskUserQuestion,
  // so $.ui.ask rejects, as it does in `claude -p`.
  on('tool.call', { tool: 'Bash' }, () => ({ result: 'pushed' }))
  const r = await $.tool.call({ tool: 'Bash', command: 'git push --force origin main' })
  expect(r).toEqual({ deny: 'hello-mod: the force-push guard failed, so the command was not run.' })
  const ok = await $.tool.call({ tool: 'Bash', command: 'git push origin feature' })
  expect(ok.result).toBe('pushed')
})
```

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/guard-without-catch
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้; แล็บนี้ตั้งใจให้ fail)

```
tests/guard.test.ts:
(fail) a force push is refused when no one can answer the question [16.00ms]
  AssertionError: expect(received).toEqual()
  
  Expected: {
    deny: "hello-mod: the force-push guard failed, so the command was not run."
  }
  Received: { result: "pushed" }
  
  the engine reported:
    hello-mod's tool.call hook was skipped: hello-mod: no implementation for tool.call

 0 pass
 1 fail
```

`Received: { result: "pushed" }`: guard ถูกข้ามไป และ force push ก็รัน

วิธีแก้คือใส่ `.catch` ที่การลงทะเบียน handler ของมันจะทำงาน **แทน** hook เมื่อ hook ล้ม:

```ts
on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
  …
}).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))
```

อ่าน handler เป็นสองกรณี:

| `next.called` | หมายความว่า | handler ทำอะไร |
|---|---|---|
| `false` | guard ล้ม **ก่อน** ที่จะปล่อย call ผ่านไป | ปฏิเสธ: `{ deny }` (fail closed) |
| `true` | guard เรียก `next` ไปแล้ว แปลว่ามันอนุมัติแล้ว | `next(e)` เล่นซ้ำผลที่การเรียกครั้งนั้นได้ไว้ ไม่มีอะไรรันซ้ำสองครั้ง |

นี่คือเหตุผลที่ guard ต้องตัดสิน **ก่อน** เรียก `next`: เมื่อเป็นแบบนั้น คำถาม "เคยเรียก next แล้วหรือยัง" ก็คือคำถามเดียวกับ "อนุมัติแล้วหรือยัง"

validate จะบอกว่า gating hook ตัวไหนมี handler:

```bash
# on: laptop
claude plugin validate week29_claude_code_mods/04_guards/labs/guard-after
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้; ตัดบรรทัดของ manifest ออก)

```
…
  ❯ ./register.ts hooks: tool.call{tool=Bash}, tool.call
  ❯ ./register.ts gating hook with .catch: tool.call{tool=Bash}
  ❯ ./register.ts gating hook without .catch: tool.call
  ❯ ./register.ts calls: $.ui.ask
✔ Validation passed with warnings
```

`gating hook with .catch: tool.call{tool=Bash}` คือ guard ส่วน `gating hook without .catch: tool.call` คือตัวนับ และ Section 5 อธิบายว่าทำไมตัวนั้น **ไม่ควร** มี `.catch`

✓ Checkpoint: บอกได้ว่า `next.called` ใน handler มีค่าเท่าไรเมื่อ `$.ui.ask` reject (`false`) และ handler คืนอะไร

## 4 · มุมมองตอน audit: gating hook

`claude plugin validate` แสดง `gating hook` สำหรับทุก hook ที่อยู่ ณ จุดที่คำตอบของมันปฏิเสธสิ่งที่ถูกขอได้: `tool.call`, `tool.check`, `prompt.submit`, `command.run` และอื่น ๆ มันคือข้อเท็จจริง ไม่ใช่คำเตือน ก่อนติดตั้ง mod ของใครก็ตาม ให้อ่านบรรทัดเหล่านั้นคู่กับ `calls:`:

- gating hook **ที่มี** `.catch` คือ guard ของใครสักคน อ่านเหตุผลของการ deny ดู
- gating hook **ที่ไม่มี** `.catch` ถ้าไม่ใช่ตัวที่ตั้งใจ fail open (observer) ก็คือ guard ที่มีรูรั่ว
- บรรทัด `calls:` ที่มี `$.process` หรือ `$.http` ที่คุณไม่ได้คาดไว้ ควรถามให้ชัดก่อนอย่างอื่น องค์กรบังคับกฎแบบนี้ได้ด้วย policy mod ของตัวเองใน tier `prepend` ซึ่งอยู่เหนือ user plugin ทุกตัว (ดู *Manage mods*)

✓ Checkpoint: จาก output ข้างบน บอกได้ว่า gating hook ตัวไหนคือ guard และตัวไหนคือ observer

## 5 · บั๊กที่เราวัดได้: observer แบบ fail-open ที่อยู่เหนือ guard

ระหว่างสร้างคอร์สนี้ เราใส่สิ่งที่ดูเหมือนตาข่ายนิรภัยที่ไม่มีพิษภัยให้ตัวนับของ Lesson 03 คือ `.catch(($, e, next) => next(e))` ("ถ้าตัวนับพัง ก็ปล่อยให้ tool รันไปเถอะ") และลงทะเบียนมัน **ก่อน** guard นี่คือ `labs/guard-before`:

```ts
export const register: Register = on => {
  // L03 · a catch-all observer, registered FIRST and made "fail-open" with .catch
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    return r
  }).catch(($, e, next) => next(e))

  // L04 · the guard, registered SECOND
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (FORCE_PUSH.test(e.command)) {
      const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
      if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    }
    return next(e)
  }).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))
}
```

guard มี `.catch` แบบ fail closed ของมันแล้ว ใช้ test เดียวกับ Section 3:

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/guard-before
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้; แล็บนี้ตั้งใจให้ fail)

```
tests/guard.test.ts:
(fail) a force push is refused when no one can answer the question [17.82ms]
  AssertionError: expect(received).toEqual()
  
  Expected: {
    deny: "hello-mod: the force-push guard failed, so the command was not run."
  }
  Received: { result: "pushed" }
  
  the engine reported:
    hook failed closed: hello-mod: errorKind=HooksError errorChars=31 (tool.call; its .catch answered)
    hello-mod's tool.call hook was skipped: hello-mod: no implementation for tool.call

 0 pass
 1 fail
```

engine ถึงกับรายงานว่า `hook failed closed … its .catch answered` แต่ push ก็ยังรันอยู่ดี ไล่ดูทีละขั้น:

```text
Bash "git push --force" ──► counter (outer, .catch → next(e))
                                 └─► guard ── $.ui.ask ──► tool.call AskUserQuestion
                                                             └─► passes through the counter too (no matcher)
                                                                  └─► nothing can answer → the call fails
counter's .catch says "next(e)" → the Bash call goes on without the guard's refusal → "pushed"
```

คำถามที่ guard ถาม ตัวมันเองก็คือ `tool.call` (Section 2) และตัวนับแบบรับทุกอย่างก็นั่งอยู่บน `tool.call` ทุกตัว เมื่อ call นั้นล้ม handler แบบ fail-open ของตัวนับก็ตอบ และมันตอบ **ให้กับ Bash call ที่มันกำลัง wrap อยู่** ในเทอร์มินัลจริงกล่องคำถามปกติจะทำงาน คุณจึงจะเห็นปัญหานี้เฉพาะตอนที่ไม่มีใครตอบได้: การรัน `claude -p`, งาน CI, หรือกล่องคำถามที่ถูกปิดทิ้ง

วิธีแก้สามแบบ แต่ละแบบวัดด้วย test เดียวกัน:

| วิธีแก้ | เปลี่ยนอะไร | ผล |
|---|---|---|
| **A · guard มาก่อน** (`labs/guard-after`) | ลงทะเบียน guard ก่อน observer และเอา `.catch` ของ observer ออก | ✓ ปฏิเสธ |
| **B · ไม่มี catch แบบ fail-open** | คงลำดับเดิม แค่ลบ `.catch(($, e, next) => next(e))` ออกจาก observer | ✓ ปฏิเสธ (เรารันแล้ว) |
| **C · อย่า match call ข้างใน** | คงทั้งสองไว้ แต่ให้ observer ใช้ matcher `{ tool: '!AskUserQuestion' }` | ✓ ปฏิเสธ (เรารันแล้ว) |

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/guard-after
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้)

```
tests/guard.test.ts:
(pass) a force push is refused when no one can answer the question [17.10ms]

 1 pass
 0 fail
Ran 1 test across 1 file. [0.15s]
```

ให้ใช้ A วิธี B แสดงให้เห็นว่า `.catch` ของ observer ไม่ได้เพิ่มอะไรเลย: hook ที่ throw ถูกข้ามอยู่แล้ว observer จึง fail open ได้เองโดยไม่ต้องมีมัน วิธี C ใช้ได้ แต่ต้องอาศัยว่าคุณรู้ทุก `$` call ที่ guard ทุกตัวทำ ทั้ง `week29_claude_code_mods/reference/hello-mod` และตัว Mod Gym เองใช้แบบ A: guard ลงทะเบียนก่อน และ observer แบบรับทุกอย่างไม่มี `.catch`

**อีกกรณีที่วัดได้: guard สองตัวบน tool เดียว** `labs/two-guards` เพิ่ม guard ตัวที่สองบน Bash (ปฏิเสธการลบแบบ recursive นอกโฟลเดอร์แล็บ) ลงทะเบียนไว้เหนือ guard กัน force push โดยทั้งคู่มี handler แบบ fail closed:

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/two-guards
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้; ตัดบางส่วน)

```
…
(fail) a force push is refused when no one can answer the question [17.92ms]
  AssertionError: expect(received).toEqual()
  
  Expected: {
    deny: "hello-mod: the force-push guard failed, so the command was not run."
  }
  Received: { deny: "hello-mod: the delete guard failed." }
…
 1 pass
 1 fail
```

ยังปฏิเสธอยู่ เพราะ handler ทั้งสองตัว fail closed แต่ใช้เหตุผลของ guard ตัวนอก Claude จึงได้รับเรื่องราวที่ผิด ให้มี **guard เดียวต่อหนึ่ง tool** โดยใส่กฎทุกข้อของ tool นั้นไว้ข้างใน นั่นคือแบบฝึกหัด

✓ Checkpoint: อธิบายได้ในสองประโยคว่าทำไม `.catch` ของตัวนับปล่อยให้ push รัน และคุณจะเลือกวิธีแก้ไหน

## 6 · เชื่อมกับ AltoTech: mutation gate สำหรับ Reef

ใน Week 26 Reef Lab Runner อนุญาตให้บทเรียนเปลี่ยน NemoClaw sandbox ได้ผ่าน gate เดียวต่อ session เท่านั้น guard ให้ gate แบบเดียวกันในระดับ agent ภายใน Claude Code: `tool.call` hook ตัวเดียวบน `Bash` ที่ลงทะเบียนก่อน คอยพักทุกคำสั่งที่จะเปลี่ยน OpenShell sandbox หรือ policy ของมัน ถามด้วย `$.ui.ask` ว่า sandbox ไหนและเพราะอะไร แล้วปฏิเสธพร้อมเหตุผลที่ Claude นำไปทำต่อได้ ถ้าสร้างตามกฎของบทนี้ มันจะ fail closed เมื่อไม่มีใครตอบ agent ที่รันแบบ headless จึงเปลี่ยน sandbox ไม่ได้เลย ส่วนคำสั่งที่อ่านอย่างเดียวผ่านไปที่ `next(e)` ได้ทันที

## 7 · ใน Mod Gym panel

1. พิมพ์ `/mod-gym 4` แล้ว `/mod-gym` และอยู่ที่แท็บ *Lesson*
2. เพิ่ม guard (Section 2 และ 3) ลงใน `~/mod-gym-lab/hello-mod` โดยลงทะเบียน **ก่อน** ตัวนับของ Lesson 03
3. กด **c** Mod Gym จะตรวจว่า module ของคุณคืน `deny`, มี handler `.catch(` และ `claude plugin validate` รายงาน `gating hook with .catch`
4. กด **2** แล้วขอให้ Claude รัน `git push --force` ใน repo ทดลอง จากนั้นเลือก **Cancel** แถว Bash บน Events tape จะขึ้น `→ denied` หรือไม่ ขึ้นกับว่า plugin ตัวไหนซ้อนอยู่เหนืออีกตัว (Lesson 03 เรื่องระหว่าง plugin): ถ้า guard ของคุณอยู่เหนือ Mod Gym, Mod Gym จะไม่เห็น call นั้นเลย
5. กด **a** เพื่อรับ prompt โค้ช: Claude จะรีวิว guard ของคุณแบบผู้ตรวจด้านความปลอดภัย

✓ Checkpoint: การตรวจของ Lesson 04 ทั้งสามข้อขึ้น ✓ ใน panel

## ลองทำเอง

เพิ่มกฎข้อที่สองให้ guard ของคุณ: ปฏิเสธโดยไม่ต้องถาม สำหรับ `rm` แบบ recursive ใด ๆ ที่เป้าหมายเริ่มที่ `/`, `~`, `$HOME` หรือ `..` ส่วนการลบในโฟลเดอร์แล็บ (`rm -rf build/`) ต้องยังรันได้ ใส่กฎนี้ **ไว้ใน Bash guard ตัวเดิม** (Section 5 เรื่อง guard สองตัว) ให้ guard ยังลงทะเบียนเป็นตัวแรก และเขียน test สำหรับทั้งสองกรณี

<details><summary>เฉลย — <code>exercises/solutions/one-guard</code></summary>

```ts
import type { Register } from 'claude-code'

const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/
const WIDE_DELETE = /\brm\s+-[a-z]*r[a-z]*\s+(\/|~|\$HOME|\.\.)/

export const register: Register = on => {
  // L04 · one guard for Bash, registered first; every Bash rule lives here
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (WIDE_DELETE.test(e.command)) {
      return { deny: 'hello-mod: recursive delete outside the lab folder refused: ' + e.command }
    }
    if (FORCE_PUSH.test(e.command)) {
      const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
      if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    }
    return next(e)
  }).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))

  // L03 · the observer, second, with no .catch
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    return r
  })
}
```

```ts
import { expect, test } from 'claude-code/testing'

test('wide recursive deletes are refused, lab deletes run', async ($, on) => {
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran: ' + e.command }))
  for (const cmd of ['rm -rf /', 'rm -rf ~/projects', 'rm -fr ../other', 'rm -r $HOME']) {
    const r = await $.tool.call({ tool: 'Bash', command: cmd })
    expect(r.deny).toContain('refused')
  }
  const ok = await $.tool.call({ tool: 'Bash', command: 'rm -rf build/' })
  expect(ok.result).toBe('ran: rm -rf build/')
})
```

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/exercises/solutions/one-guard
```

**Expected output** (จับจากเครื่อง Mac เครื่องนี้)

```
(pass) wide recursive deletes are refused, lab deletes run [15.77ms]
…
(pass) a force push is refused when no one can answer the question [16.95ms]

 2 pass
 0 fail
```

regex เป็นแค่เนินชะลอความเร็ว ไม่ใช่ sandbox: `rm -rf "$PWD/../x"`, `find / -delete` หรือสคริปต์ที่ลบแทนคุณ ล้วนเล็ดลอดไปได้ guard ที่ต้องกันได้จริงต้องทำงานกับ path ที่ resolve แล้ว (`$.fs.stat(path, { resolve: true })` ให้ `realPath`) และใช้ allow-list ของตำแหน่งที่อนุญาต ไม่ใช่ deny-list ของการสะกด

</details>

✓ Checkpoint: test ทั้งสองในเฉลยของคุณผ่าน และ test กัน force push ยังผ่านหลังเพิ่มกฎเรื่องการลบ

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| force push รันใน `claude -p` แต่ถูกพักไว้ในเทอร์มินัล | `$.ui.ask` reject เมื่อไม่มีหน้าจอ; guard ไม่มี `.catch` หรือมี observer แบบ fail-open อยู่เหนือมัน | เพิ่ม `.catch` แบบ fail closed; ลงทะเบียน guard ก่อน; เอา `.catch(next(e))` ออกจาก observer |
| `labs/guard-before` และ `labs/guard-without-catch` fail | ตั้งใจให้เป็นแบบนั้น | เทียบกับ `labs/guard-after` |
| เหตุผลของการ deny มาจาก guard อีกตัว | มี guard สองตัวที่มี `.catch` ซ้อนกันบน tool เดียว | รวมเป็น guard เดียวต่อหนึ่ง tool |
| `e.command` ไม่มีชนิดข้อมูล | hook ไม่มี matcher `{ tool: 'Bash' }` | เพิ่ม matcher |

## Next

[Lesson 05 — Draw: panes, bands, status, toast](../05_drawing/TUTORIAL.md): จนถึงตอนนี้ mod ของคุณสื่อสารได้แค่ผ่าน status line และผลลัพธ์ tool ของ Claude Lesson 05 จะวาด band ของคุณเองเหนือ prompt และ pane แบบเดียวกับของ Mod Gym ด้วย element จาก `$.ui.resolve(e)`
