# ▶ Mod Gym Lesson 08 — Test, validate, ship (capstone)

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณเรียนรู้ mod ด้วยการแก้ coding agent ที่คุณใช้อยู่ทุกสัปดาห์ แผง **Mod Gym** (`/mod-gym`) พาคุณผ่าน 8 บทเรียนและตรวจ lab mod ของคุณ `hello-mod` ไปเรื่อย ๆ ขณะที่มันโตขึ้น บทสุดท้ายนี้ทำให้ mod น่าเชื่อถือและแบ่งปันได้: เทสที่รันได้โดยไม่มี session, type-check แบบเข้มงวด, `claude plugin validate` และไฟล์ marketplace เพื่อให้เพื่อนร่วมชั้นติดตั้งได้ด้วยบรรทัดเดียว capstone จะตรวจ starter mod ของจริงที่ถูกเขียนตามเอกสารแต่ไม่เคยถูกรันเลย

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เขียนเทสด้วยชุดเครื่องมือ `claude-code/testing`: stub ที่อยู่ใต้ plugin ของคุณและตอบแทน Claude Code
- เรียนรู้กฎสี่ข้อที่ทำให้เทสแรกทุกตัวในคอร์สนี้ล้ม และข้อความที่บอกว่าคุณละเมิดข้อไหน
- type-check mod ด้วยไฟล์ประกาศที่ Claude Code เขียนให้สำหรับ build ของคุณ
- อ่าน `claude plugin validate` แบบผู้รีวิว เพิ่ม `marketplace.json` แล้วส่งมอบ: บรรทัดติดตั้งจาก GitHub, marketplace แบบโฟลเดอร์, การเพิ่มเวอร์ชัน
- **Capstone:** เอา starter จากงานวิจัย `week29/mod-agentic-coding-fitness` มารันผ่าน test, validate และ `tsc` แล้วแก้สิ่งที่พังจริง ๆ

**Time** ~75 นาที · **Difficulty** ระดับกลาง · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้

**แหล่งอ้างอิง:** [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference) · [Host a marketplace](https://code.claude.com/docs/en/plugins/host-marketplace) · [Plugin evals](https://code.claude.com/docs/en/plugin-evals) · ไฟล์ประกาศ API ที่ Claude Code 2.1.292 เขียนให้ (`claude-code.d.ts`) ซึ่งถือเป็นข้อยุติเมื่อไม่ตรงกับเอกสาร

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| บทเรียน 01–07 | `hello-mod` มี `/reps` และ `mcp__hello-mod__log_reps` | เทสด้านล่างเรียกทั้งสองตัว |
| Claude Code 2.1.287 ขึ้นไป | `claude --version` → ที่นี่ได้ `2.1.292 (Claude Code)` | `claude plugin test` มาพร้อมกับ mod |
| Node (สำหรับ `tsc`) | `npx --version` | หัวข้อ 4 รัน TypeScript ผ่าน `npx` |
| GitHub repo (ไม่บังคับ) | `gh auth status` | ใช้แค่ตอนเผยแพร่ในหัวข้อ 6 |

lab mod ที่ทำเสร็จแล้วพร้อมเทสอยู่ที่ `week29/reference/hello-mod` เปิดเทียบได้ทุกครั้งที่ติด

## 1 · ชุดเทส: คุณรับบทเป็น Claude Code

`claude plugin test <dir>` รัน `tests/*.test.ts` ทุกไฟล์ในสภาพแวดล้อมแบบเดียวกับที่ hook ของคุณรัน แต่ไม่มี session ไม่ต้อง sign-in ไม่มีเครือข่าย และไม่มีระบบไฟล์ engine โหลด plugin ของคุณแบบเดียวกับที่ session โหลด ข้างใต้มัน **ไม่มีอะไรเลย**: hook ที่คุณลงทะเบียนด้วย `on` ของเทสจะทำหน้าที่แทน Claude Code

```ts
// tests/hello-mod.test.ts
import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// The engine fills these fields in a real session; a test spells them out.
const START = { cwd: '/home/student', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

// Every hook `on` adds here sits beneath the plugin and answers in Claude Code's place.
function engine(on: On, saved: Record<string, unknown> = {}) {
  mock.store(on, saved)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
}

test('/reps starts at zero and log_reps adds to it', async ($, on) => {
  engine(on)
  await $.session.start(START)
  expect((await $.command.run(cmd('reps', ''))).text).toContain('0 reps')
  const r = await $.tool.call({ tool: 'mcp__hello-mod__log_reps', n: 10 })
  expect(String(r.result)).toContain('Total: 10')
  expect((await $.command.run(cmd('reps', ''))).text).toContain('10 reps')
})

test('reps saved by an earlier session are loaded from $.store', async ($, on) => {
  engine(on, { reps: 7 })
  await $.session.start(START)
  expect((await $.command.run(cmd('reps', ''))).text).toContain('7 reps')
})
```

ชุดเครื่องมือมีตัวแทนสำเร็จรูปให้สามตัว: `mock.store(on, entries?)` (`$.store` ในหน่วยความจำ), `mock.clock(on)` (นาฬิกาที่คุณเลื่อนเวลาเอง) และ `mock.env(on, { HOME: … })` ที่เหลือคุณ stub เอง

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/hello-mod
```

**Expected output** (บันทึกบน Mac เครื่องนี้กับ `week29/reference/hello-mod` ซึ่งมีเทสเพิ่มอีกสองตัว: guard และ context ของ prompt)

```
tests/hello-mod.test.ts:
(pass) /reps starts at zero and log_reps adds to it [22.22ms]
(pass) reps saved by an earlier session are loaded from $.store [9.41ms]
(pass) a force push is refused when no one can answer the question (the guard fails closed) [11.02ms]
(pass) the prompt gets the coach context [9.24ms]

 4 pass
 0 fail
Ran 4 tests across 1 file. [0.19s]
```

✓ Checkpoint: `claude plugin test` ของคุณเองเขียวทั้งหมด

## 2 · กฎสี่ข้อ และข้อความของแต่ละข้อ

เทสแรกทุกตัวในคอร์สนี้ล้ม นี่คือเหตุผล พร้อมข้อความที่ชุดเทสพิมพ์ออกมาจริง

**กฎข้อ 1 · stub คือ hook: `($, e)` ไม่ใช่ `(e)`** ตัวแทนเหล่านี้รับ argument ชุดเดียวกับ hook ของ plugin stub ที่รับ argument ตัวเดียวจะได้ `$` มา `e.key` จึงเป็น `undefined` และไม่มีอะไรล้มให้เห็นชัด ๆ: starter ใน capstone เก็บทุกค่าไว้ใต้ key `"undefined"` แต่เทสของมันก็ยังผ่าน เราพิสูจน์ด้วย stub ที่จดว่าถูกถามอะไร:

```
  Expected: ["progress"]
  Received: [undefined]
```

**กฎข้อ 2 · `session.start` ต้องมี `cwd` และต้องมีบางอย่างตอบมัน** `$.session.start({})` ล้ม และเทสที่ไม่มี stub ของ `session.start` ก็ล้มเหมือนกัน:

```
  HooksError: agentic-coding-fitness: next() passed an argument with no { cwd }
```

```
  HooksError: no implementation for session.start

  nothing beneath the plugins answers session.start: a test answers it with on('session.start', ...)
```

ข้อความที่สองคือกฎทั่วไป: **"no implementation for X" แปลว่าให้ stub X** มันบอกชื่อ event มาให้ แค่เพิ่ม `on('X', …)` อีกเรื่องหนึ่ง: ลงทะเบียน stub ทุกตัว *ก่อน* การเรียก `$` ครั้งแรก stub ที่มาช้าจะ throw `on("ui.close") after the test first called $`

**กฎข้อ 3 · การปฏิเสธของ plugin กลับมาเป็น `{ deny }`** เมื่อ guard ของคุณปฏิเสธการเรียกที่เทสทำผ่าน `$.tool.call` ผลลัพธ์จะเป็น `{ deny: '…' }` ไม่ใช่ `isError` นี่คือเทส guard ของ reference:

```ts
test('a force push is refused when no one can answer the question (the guard fails closed)', async ($, on) => {
  engine(on)
  // $.ui.ask is a tool.call of AskUserQuestion; with nothing here to answer it, it rejects
  on('tool.call', { tool: 'Bash' }, () => ({ result: 'pushed' }))
  await $.session.start(START)
  const r = await $.tool.call({ tool: 'Bash', command: 'git push --force origin main' })
  expect(r.deny).toContain('force-push guard failed')
  const ok = await $.tool.call({ tool: 'Bash', command: 'git push origin feature' })
  expect(ok.result).toBe('pushed')
})
```

เทสนี้คือที่มาของกฎเรื่องลำดับในบทที่ 04: เมื่อมี observer แบบรับทุก tool ลงทะเบียนก่อนและใส่ `.catch(($, e, next) => next(e))` เทสเดียวกันนี้คืนค่า `{ result: "pushed" }`

**กฎข้อ 4 · stub ที่ขาดไปอาจเงียบ** hook ที่ throw จะถูกข้าม ลองลบ stub ของ `tool.register` ออกจากเทสของ reference แล้วทั้งสี่ตัวก็ยังผ่าน: ไม่มีเทสไหนพึ่งการที่ tool ถูกลงทะเบียน ถ้าการลงทะเบียนสำคัญกับคุณ ให้ assert มันด้วย

✓ Checkpoint: คุณบอกวิธีแก้ `no implementation for turn.complete` ได้โดยไม่ต้องเปิดดู (เพิ่ม `on('turn.complete', ($, e) => ({ text: e.answer }))`)

## 3 · เทส UI: mount pane แล้วกดปุ่มของมัน

`$.ui.mount` วาด component ผ่าน plugin ของคุณบน surface ที่คุณระบุ แล้วให้คุณสั่งงานมันด้วย `key` เทสของ Mod Gym เอง (`week29/mod-gym/tests/mod-gym.test.ts`) วนผ่านสอง surface:

```ts
for (const surface of ['terminal', 'desktop'] as const) {
  await $.command.run(cmd('mod-gym', '1'))
  const ui = await $.ui.mount({ plugin: 'mod-gym', surface, ...PANE })
  expect(await ui.find({ type: 'Text', text: /Lesson 01 of 08/ })).toBeDefined()
  await ui.press({ key: 'next' })
  expect(await ui.find({ type: 'Text', text: /Lesson 02 of 08 · Your first mod/ })).toBeDefined()
  // …
  await ui.unmount()
}
```

`PANE` เก็บ props ของ `Pane` ครบชุด: `title`, `isFocused`, `bodyColumns`, `placement`, `scroll` และ `view` การกดจะรัน `onPress` ตัวจริงของปุ่ม bug ใน handler จึงโผล่ตรงนี้ capstone เจอแบบนี้หนึ่งตัว

## 4 · Type-check: ผู้รีวิวที่ถูกที่สุด

เมื่อ Claude Code โหลด mod ของคุณจากโฟลเดอร์แล้ว (`claude --plugin-dir` หรือแม้แต่ `claude -p --plugin-dir … "/reps"` ครั้งเดียว) มันจะวางไฟล์ประกาศของ build นี้ไว้ใน `.claude-plugin/types/` และวาง `tsconfig.json` ที่ extends ไฟล์เหล่านั้นไว้ข้าง mod ของคุณ จากนั้น:

```bash
# on: laptop
npx -y -p typescript@5.6 tsc -p ~/mod-gym-lab/hello-mod; echo "tsc exit $?"
```

**Expected output** (บันทึกบน Mac เครื่องนี้ กับ reference mod)

```
tsc exit 0
```

`tsconfig.json` ที่ถูกวางไว้เป็น `strict` พร้อม `noUncheckedIndexedAccess` และรวม `hooks`, `types` และ `tests` ไว้ type-check ก่อนรันเทส: กับไฟล์เทสต้นฉบับของ starter `tsc` รายงาน 20 error และ error แรก ๆ ก็คือกฎข้อ 1 และ 2 พอดี:

```
orig/tests/fitness.test.ts(8,44): error TS2339: Property 'key' does not exist on type 'EngineInterface'.
…
orig/tests/fitness.test.ts(15,25): error TS2345: Argument of type '{}' is not assignable to parameter of type 'SessionStartInput'.
```

เขียน JavaScript อยู่? ใส่ JSDoc type ที่ entry point `/** @param {import('claude-code').On} on */` แล้ว `tsc` ที่เปิด `allowJs`/`checkJs` จะใส่ type ให้ทุก `$`, `e` และ `next` ข้างใต้

✓ Checkpoint: `tsc` จบด้วย exit 0 บน mod ของคุณ

## 5 · Validate: อ่านแบบผู้รีวิว

`claude plugin validate` อ่านไฟล์ marketplace, manifest และ hooks module แบบเดียวกับที่ engine จะอ่าน โดยไม่รันอะไรเลย อ่านบรรทัดสามแบบนี้:

- **`hooks:`**: ทุก event และ matcher ถ้าเห็น `?` (เช่น `tool.call{tool=?}`) แปลว่า matcher ไม่ใช่ literal และตัวอ่านมองไม่เห็นมัน
- **`calls:`**: ทุกอย่างที่ mod เข้าถึงได้ผ่าน `$` อะไรที่คุณไม่คาดว่าจะเห็นใต้ `$.process`, `$.http` หรือ `$.fs` คือคำถามที่ต้องถามผู้เขียน
- **`gating hook with / without .catch:`**: hook ที่ความล้มเหลวของมันตัดสินว่าบางอย่างจะผ่านหรือไม่

เพิ่มไฟล์ marketplace ไว้ข้าง `plugin.json` (`.claude-plugin/marketplace.json`):

```json
{
  "name": "hello-mod",
  "owner": { "name": "Agentic Coding Fitness student" },
  "plugins": [{ "name": "hello-mod", "source": "./" }]
}
```

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (บันทึกบน Mac เครื่องนี้ ส่วนบนของรายงาน)

```
Validating marketplace manifest: …/hello-mod/.claude-plugin/marketplace.json

⚠ Found 2 warnings:

  ❯ description: No marketplace description provided. Adding a description helps users understand what this marketplace offers
  ❯ plugins[0] plugin.json → author: No author information provided. Consider adding author details for plugin attribution

Validating plugin: …/hello-mod/.claude-plugin/plugin.json
…
```

warning ไม่ทำให้ validate ล้ม error ทำ error อีกสองแบบที่คุณเจอในบทก่อน ๆ: key ของ `$.state` ที่ไม่มีใน `types/index.d.ts` (`hello-mod.reps is not declared`) และชื่อ plugin ที่ดูเหมือนเป็นของ Anthropic เอง (`claude-…`, `anthropic-…`)

✓ Checkpoint: validate อ่านทั้งสามไฟล์และจบด้วย `✔ Validation passed` (มีหรือไม่มี warning ก็ได้)

## 6 · ส่งมอบ

**จาก GitHub** push โฟลเดอร์ mod เป็น repository ของมันเอง (`"source": "./"` ในไฟล์ marketplace ชี้ไปที่ root ของ repo) เพื่อนร่วมชั้นพิมพ์ที่ prompt ของ session ในเทอร์มินัล:

```text
/plugin install hello-mod --marketplace <you>/<repo>
```

เขาตอบ `y` เพื่อเพิ่ม marketplace แล้วเลือก scope mod ที่มีแค่ hooks module จะทำงานทันที

**จากโฟลเดอร์ (วิธีที่ Mod Gym ติดตั้งบนเครื่องของผู้สอน)** marketplace เป็นโฟลเดอร์ได้ `week29/.claude-plugin/marketplace.json` คือหนึ่งในนั้น มันระบุ `mod-gym` (`./mod-gym`) และ `hello-mod` ตัว reference (`./reference/hello-mod`):

```bash
# on: laptop — do this once, from the repo root
claude plugin marketplace add ./week29
claude plugin install mod-gym@fitness-week29 --scope user
claude plugin list
```

**Expected output** (บันทึกบน Mac เครื่องนี้ รายการ `mod-gym` จาก `claude plugin list`)

```
  ❯ mod-gym@fitness-week29
    Version: 0.1.0
    Read from: /Users/altodev/Desktop/agenticaicodingfitness/week29/mod-gym
    Scope: user
    Status: ✔ enabled
```

`Read from:` คือตัวโฟลเดอร์เอง ไม่ใช่สำเนา แก้ไฟล์ในนั้นแล้วรัน `/reload-plugins` ใน session ของคุณ: ไม่ต้องเพิ่มเวอร์ชัน ไม่ต้องติดตั้งใหม่

**การเพิ่มเวอร์ชัน** การติดตั้งแบบอื่นทั้งหมด (แหล่ง GitHub หรือ git) จะรันสำเนาที่ทำไว้ตอนติดตั้ง ซึ่ง cache ตาม `version` การแก้ไขจะไปถึงเพื่อนร่วมชั้นก็ต่อเมื่อคุณเพิ่ม `version` ใน `plugin.json` และเขารัน `/plugin marketplace update` แล้ว `claude plugin update hello-mod` แล้ว `/reload-plugins`

**marketplace ของคลาส** `.claude-plugin/marketplace.json` ที่ root ของ repo (marketplace `agentic-coding-fitness` ตัวที่นักเรียนเพิ่มไว้แล้ว) ตอนนี้มีสอง plugin:

```
[('agentic-coding-fitness', './plugins/agentic-coding-fitness'), ('mod-gym', './week29/mod-gym')]
```

เมื่อ push แล้ว นักเรียนรัน `/plugin marketplace update agentic-coding-fitness` แล้วตามด้วย `/plugin install mod-gym@agentic-coding-fitness`

**ด่านตรวจอีกหนึ่งด่าน (ไม่บังคับ)** `claude plugin eval` เปรียบเทียบ Claude ที่มีและไม่มี plugin ของคุณ บนเคสใน `evals/` ส่วน `claude plugin eval init` ร่างเคสเหล่านั้นให้ในรูปแบบที่ build ของคุณรับ starter มีอยู่หนึ่งเคส `evals/coach-explains-mods/`

✓ Checkpoint: มี `marketplace.json` แล้ว validate ผ่าน และคุณอธิบายได้ว่าทำไมการติดตั้งจาก GitHub ต้องเพิ่มเวอร์ชัน แต่การติดตั้งจากโฟลเดอร์ไม่ต้อง

## 7 · Capstone: ตรวจ starter จากงานวิจัย

`week29/mod-agentic-coding-fitness` คือ starter v3.0.0 จากงานวิจัยของสัปดาห์นี้: pane ติดตามความคืบหน้า `/fitness`, `/fitness-log`, `/fitness-grade`, tool `log_lab` และการบันทึกอัตโนมัติเมื่อ `pytest` รันผ่าน ผู้เขียนเขียนมันจากเอกสารและบอกไว้ตรง ๆ: *"The starter `register.js` and tests were not executed against a live Claude Code."* งานของคุณ: ทำให้มันผ่าน test, validate **และ** `tsc` โดยแก้เฉพาะสิ่งที่ผิดจริง ทำบนสำเนา ปล่อยต้นฉบับไว้ตามเดิม

```bash
# on: laptop
cp -R week29/mod-agentic-coding-fitness ~/mod-gym-lab/starter && find ~/mod-gym-lab/starter -name .DS_Store -delete
claude plugin validate ~/mod-gym-lab/starter
cd ~/mod-gym-lab/starter && claude plugin test .
```

**Expected output** (บันทึกบน Mac เครื่องนี้ ตัดผล validate เหลือเฉพาะบรรทัดที่น่าสนใจ)

```
  ❯ ./register.js hooks: session.start, tool.call{tool=?}, tool.call{tool=Bash}, turn.complete, prompt.submit, command.run{command=fitness}, …
  ❯ ./register.js gating hook without .catch: tool.call{tool=?}
…
✔ Validation passed

tests/fitness.test.ts:
(fail) /fitness-log records a lab and reports pass/fail [23.28ms]
  HooksError: agentic-coding-fitness: next() passed an argument with no { cwd }
…
 0 pass
 3 fail
```

validate ผ่าน แต่ก็แสดงปัญหาให้เห็นแล้ว (`tool=?`) และเทสล้มทุกตัว ไล่ทีละจุดแบบที่คุณทำกับโค้ดที่ไม่เคยเทส: ทีละความล้มเหลว รันใหม่หลังแก้แต่ละครั้ง นี่คือสิ่งที่เราเจอบน Mac เครื่องนี้ ตามลำดับ:

| # | อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|---|
| 1 | `next() passed an argument with no { cwd }` | `$.session.start({})` | `$.session.start(START)` ที่มี `cwd`, `surface`, `isInteractive` |
| 2 | `no implementation for session.start` | ไม่มี stub ตอบมัน | `on('session.start', ($, e) => ({ cwd: e.cwd }))` |
| 3 | `no implementation for turn.complete` | เทส auto-log ยิง `turn.complete` แต่ข้างใต้ไม่มีอะไร | `on('turn.complete', ($, e) => ({ text: e.answer }))` |
| 4 | ผ่าน 3 ตัว แต่ stub ของ store เห็น `[undefined]` | stub เขียนเป็น `(e) =>` (กฎข้อ 1) | `mock.store(on)`; stub ทุกตัวเป็น `($, e) =>` |
| 5 | `tool.call{tool=?}` ใน validate | matcher สร้างจาก `'mcp__' + PLUGIN + '__log_lab'` | ใช้ literal `'mcp__agentic-coding-fitness__log_lab'` |
| 6 | กด **Close** แล้ว throw `ui.close: id is 1 to 64 of letters, digits, _ or - (host check)` | `$.ui.close(PANE)` ส่ง string | `$.ui.close({ id: PANE })` |
| 7 | `tsc` ที่เปิด `checkJs`: 43 error | JS module ที่ไม่มี type; ในนั้นมี `'key' does not exist in type 'TextProps'`, การลบ `Date - Date` และ `find(...)` ที่ destructure โดยไม่มีค่าสำรอง | ใส่ JSDoc type ให้ `register`, state ของ module และ helper; เอา `key` ออกจาก `Text`; `getTime()`; `?? ['', 'unknown']` |

หลังแก้ข้อ 3 เทสต้นฉบับสามตัวเขียวแล้ว แต่ **เขียวยังไม่ใช่เสร็จ** ข้อ 4 ถึง 7 มาจากไฟล์เทสที่สองที่ถามคำถามที่เทสต้นฉบับไม่เคยถาม: ความคืบหน้าของเมื่อวานโหลดได้ไหม Claude เรียก `log_lab` ได้จริงไหม pane วาดออกมาไหม **Close** ใช้ได้ไหม ไฟล์นั้นเจอ bug ของปุ่ม Close ซึ่งจะเกิดกับนักเรียนทุกคนที่กด **q**

สิ่งที่การตรวจนี้บอกเกี่ยวกับการออกแบบของ starter ด้วย: pane, suffix ของ spinner, การบันทึก `pytest` อัตโนมัติ และข้อความใน `turn.complete` ทำงานได้หมดเมื่อเทสเข้าถึงมันได้ สิ่งที่ผิดส่วนใหญ่อยู่ในเทสและใน type ซึ่งเป็นหน้าตาปกติของ starter ที่ไม่เคยถูกเทส

**Expected output** (บันทึกบน Mac เครื่องนี้ หลังแก้ครบทั้งเจ็ดข้อ)

```
tests/audit.test.ts:
(pass) progress saved yesterday is loaded today [25.84ms]
(pass) Claude can log a lab with the log_lab tool [13.47ms]
(pass) the /fitness pane draws [16.13ms]

tests/fitness.test.ts:
(pass) /fitness-log records a lab and reports pass/fail [25.78ms]
(pass) a passing weekNN pytest run is auto-logged and announced after the turn [13.13ms]
(pass) /fitness-grade uses the model stub [11.72ms]

 6 pass
 0 fail
Ran 6 tests across 2 files. [0.19s]
```

```
  ❯ ./register.js hooks: session.start, tool.call{tool=mcp__agentic-coding-fitness__log_lab}, tool.call{tool=Bash}, …
✔ Validation passed
tsc exit 0
```

✓ Checkpoint: test, validate และ `tsc` ผ่านทั้งหมดบนสำเนาของคุณ และคุณบอกได้ว่าข้อไหนที่นักเรียนจะเจอเป็นอย่างแรก (ปุ่ม Close)

## ในแผง Mod Gym

พิมพ์ `/mod-gym 8` แล้วกด **c** gym จะตรวจ `~/mod-gym-lab/hello-mod`:

| รายการตรวจ | ผ่านเมื่อ |
|---|---|
| has a tests/*.test.ts | `hello-mod/tests/` มีไฟล์ที่ลงท้ายด้วย `.test.ts` |
| claude plugin test passes | `claude plugin test` จบด้วย exit 0 |
| marketplace.json exists | มีไฟล์ `hello-mod/.claude-plugin/marketplace.json` |
| validate passes with the marketplace | `claude plugin validate` จบด้วย exit 0 |

ผ่านครบแปดบท แล้วแท็บ **Progress** (**3**) จะแสดง ✓ แปดตัว

## แบบฝึกหัด — เขียนรายงาน capstone

ทำหัวข้อ 7 ด้วยตัวเองบน `~/mod-gym-lab/starter` สำหรับแต่ละการแก้ เขียนหนึ่งบรรทัด: อาการ สาเหตุ และสิ่งที่เปลี่ยน แล้วเพิ่มไฟล์เทสที่สองที่จะจับ bug ของปุ่ม Close ได้ก่อนนักเรียนคนไหนจะเจอ

<details><summary>เฉลย</summary>

**`tests/fitness.test.ts`**: stub ที่ถูกต้อง และ input ครบ

```ts
import { expect, mock, test } from 'claude-code/testing'

// Runs with `claude plugin test` from the plugin directory: no session, sign-in, or network.
// Stubs answer in Claude Code's place. Register every stub before the first call on `$`.
// A stub is a hook like any other: ($, e, next). A one-argument stub receives `$`, not the event.

const START = { cwd: '/lab', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

test('/fitness-log records a lab and reports pass/fail', async ($, on) => {
  mock.store(on)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.invalidate', () => ({ value: undefined }))

  await $.session.start(START)
  const ok = await $.command.run(cmd('fitness-log', '05 agent-loops 0.86'))
  expect(ok.text).toContain('05/agent-loops → 0.86 (pass)')
  // … the low-score and usage cases as before
})
```

ในเทส auto-log ให้เพิ่ม `on('turn.complete', ($, e) => ({ text: e.answer }))` และยิงมันด้วย field ครบ: `$.turn.complete({ turnId: 't1', answer: 'ok', durationMs: 10, isAborted: false, reason: 'answer' })` ในเทส grader ให้แทน stub ของ store สองตัวด้วย `mock.store(on)`

**`tests/audit.test.ts`**: คำถามที่เทสต้นฉบับไม่เคยถาม

```ts
import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

const START = { cwd: '/lab', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

function engine(on: On, saved: Record<string, unknown> = {}) {
  mock.store(on, saved)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.invalidate', () => ({ value: undefined }))
}

test('progress saved yesterday is loaded today', async ($, on) => {
  engine(on, { progress: { labs: { '02/llm-fundamentals': { score: 1, at: 'x' } }, sessions: [] } })
  await $.session.start(START)
  expect((await $.command.run(cmd('fitness-next'))).text).toContain('week 03')
})

test('Claude can log a lab with the log_lab tool', async ($, on) => {
  engine(on)
  await $.session.start(START)
  const r = await $.tool.call({ tool: 'mcp__agentic-coding-fitness__log_lab', week: '05', skill: 'agent-loops', score: 0.9 })
  expect(String(r.result)).toContain('05/agent-loops')
})

test('the /fitness pane draws', async ($, on) => {
  engine(on)
  on('session.usage', () => ({ value: { context: { percent: 12 } } as never }))
  const closed: unknown[] = []
  on('ui.close', ($, e) => { closed.push(e); return { value: undefined } })
  await $.session.start(START)
  const ui = await $.ui.mount({
    plugin: 'agentic-coding-fitness', surface: 'terminal', component: 'Pane', requestId: 'fitness',
    props: { title: 'Agentic Coding Fitness', isFocused: true, bodyColumns: 80, placement: 'dock', scroll: { offset: 0, bodyRows: 40 }, view: {} },
  })
  expect(await ui.find({ type: 'Text', text: /W02/ })).toBeDefined()
  await ui.press({ key: 'close' })
  expect(closed).toEqual([expect.objectContaining({ id: 'fitness' })])
  await ui.unmount()
})
```

**`hooks/register.js`**: diff ทั้งหมด (เฉพาะบรรทัด `+`/`-`)

```diff
+/** @type {{ labs: Record<string, { score: number, at: string }>, sessions: string[] }} */
+/** @type {string | null} */
+/** @param {string[]} sessions */
-    const diff = Math.round((cursor - new Date(d + 'T00:00:00Z')) / 86400000)
+    const diff = Math.round((cursor.getTime() - new Date(d + 'T00:00:00Z').getTime()) / 86400000)
+/** @typedef {import('claude-code').EngineInterface} Engine */
+/** @param {Engine} $ */
+/** @param {Engine} $ @param {string} week @param {string} skill @param {number} score */
+/** @param {string} args */
+/** @param {import('claude-code').On} on */
-  on('tool.call', { tool: LOG_TOOL }, async ($, e) => {
+  on('tool.call', { tool: 'mcp__agentic-coding-fitness__log_lab' }, async ($, e) => {
-    const [, skill] = CURRICULUM.find(([cw]) => cw === w)
+    const [, skill] = CURRICULUM.find(([cw]) => cw === w) ?? ['', 'unknown']
-      return Text({ key: 'row-' + w, dimColor: !lab, children: [mark + ' W' + w + '  ' + s + score] })
+      return Text({ dimColor: !lab, children: [mark + ' W' + w + '  ' + s + score] })
-            Button({ key: 'close', label: 'Close', hotkey: 'q', onPress: () => $.ui.close(PANE) }),
+            Button({ key: 'close', label: 'Close', hotkey: 'q', onPress: () => $.ui.close({ id: PANE }) }),
```

สำหรับ `tsc` ให้ใช้ tsconfig จากหัวข้อ 4 โดยเพิ่ม `"allowJs": true, "checkJs": true` และรวม `hooks` กับ `tests` ไว้ การรันจบด้วย `tsc exit 0`

</details>

✓ Checkpoint: เทสเขียวหกตัว, `✔ Validation passed` พร้อม `tool=mcp__agentic-coding-fitness__log_lab` และ `tsc exit 0`

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `no implementation for X` | ไม่มีอะไรใต้ plugin ตอบ event X | เพิ่ม `on('X', ($, e) => …)` ก่อนการเรียก `$` ครั้งแรก |
| `on("X") after the test first called $` | ลงทะเบียน stub หลัง `$.session.start` | ย้ายขึ้นไปด้านบน |
| `e.key` / `e.path` ของ stub เป็น `undefined` | stub เขียนเป็น `(e) =>` | เขียนเป็น `($, e) =>` |
| `expect(r.isError)` ได้ `undefined` กับการเรียกที่ถูกปฏิเสธ | การปฏิเสธของ plugin คือ `{ deny }` | assert ที่ `r.deny` |
| `tsc -p <mod>` หา `claude-code` ไม่เจอ | mod ยังไม่เคยถูกโหลด จึงไม่มี `.claude-plugin/types/` | โหลดมันสักครั้ง (`claude -p --plugin-dir <mod> "/reps"`) หรือให้ tsconfig ชี้ไปที่ `claude-code.d.ts` |
| เพื่อนร่วมชั้นไม่ได้รับการแก้ของคุณ | สำเนาที่ติดตั้งถูก cache ตาม `version` | เพิ่ม `version`; เขารัน `/plugin marketplace update`, `claude plugin update <mod>`, `/reload-plugins` |

## ต่อไป

คุณเรียนจบ Week 29 แล้ว ติดตั้ง Mod Gym ไว้ต่อไป: แท็บ **Events** ของมันคือมุมมองสดของ hook pipeline ในทุก session และ `/mod-gym status` แสดง streak ของคุณ ก้าวต่อไป: เอา starter ที่แก้แล้วมาเปลี่ยนชื่อ แล้วทำให้เป็น mod ติดตามความคืบหน้าในคลาสของคุณเอง
