# ▶ Mod Gym Lesson 07 — Command, tool และ model

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณเรียนรู้ mod ด้วยการแก้ coding agent ที่คุณใช้อยู่ทุกสัปดาห์ แผง **Mod Gym** (`/mod-gym`) พาคุณผ่าน 8 บทเรียนและตรวจ lab mod ของคุณ `hello-mod` ไปเรื่อย ๆ ขณะที่มันโตขึ้น บทนี้ทำให้ mod ของคุณมีสิ่งที่ *ถูกเรียกใช้* ได้: slash command ที่คุณพิมพ์เอง tool ที่ Claude เป็นคนเรียก และทางออกจาก mod อีกสามทาง (model เล็กบนคลาวด์ model ที่รันบนเครื่องของคุณเอง และโปรแกรมบนเครื่อง)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- ลงทะเบียน slash command ใน `session.start` แล้วตอบมันใน `command.run` ด้วย `{ text }` ไม่มี turn ของ Claude ไม่เสีย token
- ลงทะเบียน tool ดูว่า Claude เห็นมันในชื่อ `mcp__hello-mod__log_reps` ให้บริการมันด้วย hook `tool.call` แล้วดู Claude เรียกมันจริง
- อ่านตัวอย่างจริงสองตัวใน Mod Gym เอง: tool `gym_status` และปุ่ม **Ask Claude** (`$.prompt.fill`)
- สร้าง `setpoint-coach` mod เล็ก ๆ สำหรับระบบ HVAC ของโรงแรม ที่ถาม model บนคลาวด์ (`$.model.complete`) model ในเครื่อง (`$.http.fetch` ไปที่ Ollama หรือ DGX Spark) และโปรแกรม (`$.process.run`)
- ตัดสินใจว่าฟีเจอร์หนึ่ง ๆ ควรเป็นรูปแบบไหนในห้าแบบ แล้วเทสทั้งหมดโดยไม่ใช้เครือข่าย

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้ (Ollama ไม่บังคับ ใช้แค่ขั้นตอน sovereign)

**แหล่งอ้างอิง:** [Claude Code mods (Anthropic blog)](https://claude.com/blog/claude-code-mods) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Use the mods API](https://code.claude.com/docs/en/plugins/mods/api) · [React to events](https://code.claude.com/docs/en/plugins/mods/events) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · ไฟล์ประกาศ API ที่ Claude Code 2.1.292 เขียนให้ (`claude-code.d.ts`) ซึ่งถือเป็นข้อยุติเมื่อไม่ตรงกับเอกสาร

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| บทเรียน 01–06 | `hello-mod` มี band, atom `reps` และ `types/index.d.ts` | บทนี้อ่านและเขียน atom `reps` |
| Claude Code 2.1.287 ขึ้นไป | `claude --version` → ที่นี่ได้ `2.1.292 (Claude Code)` | mod ต้องใช้ |
| ติดตั้ง Mod Gym แล้ว | `/mod-gym` เปิดแผงขึ้นมา | ใช้ตรวจท้ายบท |
| Ollama (ไม่บังคับ) | `curl -s localhost:11434/api/tags` มี `gemma4:12b` | ใช้แค่ `/spark-ask` ในหัวข้อ 4 |

lab mod ของคุณอยู่ที่ `~/mod-gym-lab/hello-mod` (เปลี่ยนได้ด้วย `/mod-gym lab <path>`) เวอร์ชันที่ทำเสร็จแล้วของทุกอย่างในบทนี้อยู่ที่ `week29/reference/hello-mod`

## 1 · Slash command: ได้คำตอบโดยไม่มี turn ของ Claude

command มีสองส่วน คุณ **ลงทะเบียน** มันครั้งเดียวใน `session.start` เพื่อให้มันขึ้นใน typeahead และ `/help` แล้วคุณ **ตอบ** มันใน hook `command.run` ที่ matcher ระบุชื่อมัน

```ts
// hooks/register.tsx — inside register(on)
on('session.start', async ($, e, next) => {
  const saved = await $.store.get('reps')
  if (typeof saved === 'number') await update($, reps, () => saved)
  await $.command.register({ name: 'reps', description: 'Show the reps hello-mod has logged' })
  // … the tool from section 2 goes here too
  return next(e)
})

on('command.run', { command: 'reps' }, async $ => ({ text: (await read($, reps)) + ' reps logged' }))
```

สามเรื่องที่ควรสังเกต:

- `session.start` ถูก await ก่อน prompt แรก command ที่ลงทะเบียนตรงนั้นจึงมีให้ใช้ตั้งแต่ turn แรก
- hook `command.run` คืน `{ text }` และไม่เรียก `next` เลย มัน **ตอบ** แทน Claude Code ไม่มีการถาม model อะไร command นี้จึงไม่เสีย token
- อยากรับ argument? ใส่ `argumentHint: '<room>'` ตอนลงทะเบียน แล้วอ่าน `e.args` (ข้อความดิบที่ตามหลังชื่อ command) ใน hook

รันแบบ headless โดยโหลด mod จากโฟลเดอร์ของมัน:

```bash
# on: laptop
claude -p --plugin-dir ~/mod-gym-lab/hello-mod "/reps"
```

**Expected output** (บันทึกบน Mac เครื่องนี้, Claude Code 2.1.292)

```
hello-mod: 0 reps logged
```

Claude Code ใส่ชื่อปลั๊กอินไว้หน้าผลลัพธ์ของคำสั่งให้เอง (`hello-mod: …`) จึงไม่ต้องใส่ไว้ใน `text` ไม่อย่างนั้นจะแสดงซ้ำสองครั้ง

`hello-mod:` ตัวแรกคือ Claude Code บอกชื่อ plugin ที่ตอบ ตัวที่สองคือข้อความของ reference mod เอง เมื่อรู้ว่า engine ใส่ชื่อให้แล้ว คุณก็ตัดมันออกจากข้อความของคุณได้

`claude plugin validate` รายงาน command นี้ว่าเป็น command ที่ module ตอบเอง:

```
  ❯ ./register.tsx answers its own command: command.run{command=reps}
```

✓ Checkpoint: `/reps` พิมพ์ตัวเลขออกมา และคุณอธิบายได้ว่าทำไมมันไม่เสีย token

## 2 · Tool ที่ Claude เรียกใช้ได้

command มีไว้ให้ **คุณ** ส่วน tool มีไว้ให้ **Claude**: มันขึ้นในรายการ tool ของ model และ model เป็นคนตัดสินใจว่าจะเรียกเมื่อไร ลงทะเบียนแบบเดียวกัน แล้วให้บริการมันด้วย hook `tool.call`

```ts
// in session.start, after the command
await $.tool.register({
  name: 'log_reps',
  description: 'Record reps the student finished. Call it when the student says they did reps.',
  inputSchema: { type: 'object', properties: { n: { type: 'number' } }, required: ['n'] },
})

// serve the tool Claude sees as mcp__hello-mod__log_reps
on('tool.call', { tool: 'mcp__hello-mod__log_reps' }, async ($, e) => {
  const n = Number(e.n)
  if (!Number.isFinite(n) || n <= 0) return { result: 'n must be a positive number' }
  await update($, reps, r => r + n)
  return { result: 'Logged ' + n + ' reps. Total: ' + (await read($, reps)) }
})
```

- model เห็น tool ในชื่อ `mcp__<plugin>__<name>` matcher ของคุณต้องใช้ชื่อเต็มนี้
- เขียน matcher เป็น **string literal** `claude plugin validate` อ่านซอร์สโดยไม่รันมัน ชื่อที่ต่อด้วย `+` จะขึ้นเป็น `tool=?` (คุณจะเจอของจริงใน capstone ของบทที่ 08)
- `description` คือ prompt บอกว่า *เมื่อไร* ควรเรียก tool ไม่ใช่แค่ว่ามันทำอะไร
- ตรวจ input เสมอ schema เป็นแค่คำขอต่อ model ไม่ใช่การรับประกัน
- การเรียกที่ไม่มี hook ไหนตอบ จะล้มเหลวพร้อม error บอกเหตุผล

บอก Claude เป็นภาษาธรรมดา แล้วอ่านยอดกลับมาด้วย command ในโหมด `-p` ไม่มีใครกดอนุญาต tool ได้ จึงต้องอนุญาต tool ตัวนี้ตัวเดียวด้วยชื่อ:

```bash
# on: laptop
claude -p "I just did 10 squats. Log them with your tool, then tell me the total in one short line." \
  --plugin-dir ~/mod-gym-lab/hello-mod --allowedTools=mcp__hello-mod__log_reps
claude -p --plugin-dir ~/mod-gym-lab/hello-mod "/reps"
```

**Expected output** (บันทึกบน Mac เครื่องนี้ บรรทัดแรกเป็นคำพูดของ model เองและจะต่างไป)

```
10 reps logged — total: 10. Great job! 💪
hello-mod: 10 reps logged
```

session ที่สองอ่านค่า `10` กลับมาได้ เพราะ hook `state.set` จากบทที่ 06 คัดลอกทุกการเปลี่ยนแปลงของ `reps` ลง `$.store`

✓ Checkpoint: Claude เรียก `mcp__hello-mod__log_reps` แล้ว และ `/reps` ใน session ใหม่แสดงยอดใหม่

## 3 · ตัวอย่างจริงสองตัวใน Mod Gym

เปิด `week29/mod-gym/hooks/register.tsx` แผงนี้ใช้ทั้งสองรูปแบบ

**Tool `gym_status`** Mod Gym ลงทะเบียน tool ที่มี input schema ว่าง description ของมันบอก Claude ว่าควรเรียกเมื่อไรและควรโค้ชอย่างไร:

```ts
await $.tool.register({
  name: 'gym_status',
  description:
    "Where the student is in Mod Gym (Week 29: Claude Code mods): current lesson, its tutorial path, the student's lab folder, last check results and passed lessons. Call it before coaching the student on a Mod Gym lesson; then guide with questions rather than writing their lab code.",
  inputSchema: { type: 'object', properties: {} },
})
```

hook ตอบเป็นข้อความ JSON: บทเรียน โฟลเดอร์แล็บ ผลตรวจล่าสุด และกฎการโค้ช มันยังมี `.catch` ที่ตอบ `'Mod Gym status is unavailable right now.'` ถ้าการสร้างสถานะ throw ออกมา tool ควรให้คำตอบ *บางอย่าง* กับ model เสมอ

```ts
const TOOL = 'mcp__mod-gym__gym_status'
on('tool.call', { tool: TOOL }, async $ => {
  // … build `status` from the atoms
  return { result: JSON.stringify(status, null, 2) }
}).catch(() => ({ result: 'Mod Gym status is unavailable right now.' }))
```

`TOOL` เป็น `const` ที่เก็บ string literal ตัวเดียว validate จึงยังอ่านมันออก: `tool.call{tool=mcp__mod-gym__gym_status}`

**ปุ่ม Ask Claude** กด **a** ในแผง Mod Gym *ไม่ได้* ส่ง prompt ให้คุณ มันเติมข้อความลงช่อง prompt แล้วให้คุณแก้ก่อนกด Enter:

```ts
async function askClaude($: $): Promise<void> {
  const lesson = lessonAt(await read($, lessonAtom))
  const lab = await labDir($)
  const text =
    '[Mod Gym · lesson ' + lesson.id + ': ' + lesson.title + ' · lab folder ' + lab + ' · tutorial ' + lesson.tutorial + ']\n' +
    lesson.ask
  await $.prompt.fill({ text, mode: 'replace' })
  $.ui.toast('Question is in your prompt: edit it or press Enter.')
}
```

`$.prompt.submit` จะเริ่ม turn เองทันที ส่วน `$.prompt.fill` ให้คนยังอยู่ในวงจร ซึ่งเป็นค่าเริ่มต้นที่ถูกต้องสำหรับอะไรก็ตามที่ใช้ token

✓ Checkpoint: คุณบอกได้ว่าตัวไหน model เป็นคนสั่ง (tool) และตัวไหนคนเป็นคนสั่ง (ปุ่ม)

## 4 · ออกไปข้างนอก: model บนคลาวด์ model ในเครื่อง และโปรแกรม

hooks module ไม่มี Node ไม่มี `fetch` ของตัวเอง และไม่มี shell ทุกอย่างที่อยู่ข้างนอกต้องผ่าน `$` ซึ่งเป็นสิ่งที่ทำให้ mod ตรวจสอบได้ด้วย: `claude plugin validate` แสดงทุกการเรียก สร้าง lab mod ตัวที่สอง `setpoint-coach` ไว้ข้าง `hello-mod`:

```text
setpoint-coach/.claude-plugin/plugin.json   { "name": "setpoint-coach", "version": "0.1.0", "description": "Lesson 07 demo: a cloud model, a local model and a program, each behind a slash command." }
setpoint-coach/hooks/hooks.json             { "modules": ["./register.ts"] }
setpoint-coach/hooks/register.ts
```

```ts
import type { Register } from 'claude-code'

const SYSTEM = 'You explain hotel HVAC setpoints to a duty manager. Two sentences, no jargon.'
const SPARK = 'http://localhost:11434/v1/chat/completions' // Ollama here; a DGX Spark on the LAN in class

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'hvac-explain', description: 'Ask a small cloud model about a setpoint', argumentHint: '<question>' })
    await $.command.register({ name: 'spark-ask', description: 'Ask the local model (sovereign tier)', argumentHint: '<question>' })
    await $.command.register({ name: 'git-where', description: 'Show the branch this session is on' })
    return next(e)
  })

  // A side question to a small cloud model: no history, no tools, always a result
  on('command.run', { command: 'hvac-explain' }, async ($, e) => {
    if (!e.args.trim()) return { text: 'Usage: /hvac-explain <question>' }
    const r = await $.model.complete({ model: 'haiku', system: SYSTEM, prompt: e.args, maxTokens: 120, timeoutMs: 20_000 })
    return { text: r.isAnswered ? r.text.trim() : 'No answer (' + r.reason + ')' }
  })

  // The same question to a model on your own box: the data never leaves the building
  on('command.run', { command: 'spark-ask' }, async ($, e) => {
    if (!e.args.trim()) return { text: 'Usage: /spark-ask <question>' }
    const r = await $.http.fetch(SPARK, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma4:12b',
        reasoning_effort: 'none',
        max_tokens: 120,
        messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: e.args }],
      }),
    })
    if (!r.ok) return { text: 'Local model failed: HTTP ' + r.status }
    const reply = JSON.parse(r.text) as { choices: { message: { content: string } }[] }
    return { text: (reply.choices[0]?.message.content ?? '').trim() }
  })

  // A program, run by argv: no shell, so no quoting bugs and no injected `; rm`
  on('command.run', { command: 'git-where' }, async $ => {
    const r = await $.process.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], { cwd: await $.session.cwd(), timeoutMs: 10_000 })
    return { text: r.exitCode === 0 ? 'On branch ' + r.stdout.trim() : 'Not a git repo (' + r.stderr.trim() + ')' }
  })
}
```

แต่ละการเรียกสัญญาอะไรไว้:

- **`$.model.complete`** ส่ง prompt หนึ่งชุดไปที่ model ผ่าน client ของ session เอง ไม่มีประวัติสนทนาและไม่มี tool มันไม่ throw เพราะสิ่งที่ผู้ให้บริการทำ: มัน resolve เป็น `isAnswered: true` พร้อม `text` หรือ `isAnswered: false` พร้อม `reason` (`api-error`, `empty-reply`, `aborted` เมื่อหมดเวลา `timeoutMs`) ตรวจ `isAnswered` ทุกครั้ง
- **`$.http.fetch`** resolve เป็น `{ status, ok, headers, text }` ชี้มันไปที่ endpoint แบบ OpenAI-compatible บน Ollama หรือ DGX Spark แล้วคำถามจะไม่ออกจากตึกเลย: นี่คือชั้น sovereign ของ Alto Copilot (`gemma4` เป็น thinking model ใส่ `reasoning_effort: 'none'` เพื่อให้ตอบสั้น)
- **`$.process.run(argv)`** รันโปรแกรมด้วยรายการ argument ไม่ใช่สตริงของ shell `; rm -rf ~` ที่อยู่ใน argument เป็นแค่ข้อความ มัน resolve เป็น `{ exitCode, stdout, stderr, … }`

validate มัน แล้วอ่านบรรทัด `calls:` แบบที่ผู้รีวิวอ่าน:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/setpoint-coach
```

**Expected output** (บันทึกบน Mac เครื่องนี้ เฉพาะส่วน hooks)

```
  ❯ ./register.ts hooks: session.start, command.run{command=hvac-explain}, command.run{command=spark-ask}, command.run{command=git-where}
  ❯ ./register.ts answers its own command: command.run{command=hvac-explain}
  ❯ ./register.ts answers its own command: command.run{command=spark-ask}
  ❯ ./register.ts answers its own command: command.run{command=git-where}
  ❯ ./register.ts calls: $.command.register, $.http.fetch, $.model.complete, $.process.run, $.session.cwd

✔ Validation passed with warnings
```

(warning คือ `author` ที่ไม่มีใน `plugin.json`) ทีนี้รันทั้งสามตัว:

```bash
# on: laptop — from inside this repo, then from a folder that is not a git repo
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/git-where"
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/hvac-explain Why does the BMS hold guest rooms at 24 °C when they are unoccupied?"
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/spark-ask Why does the BMS hold guest rooms at 24 °C when they are unoccupied?"
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/spark-ask"
```

**Expected output** (บันทึกบน Mac เครื่องนี้ คำตอบของ model จะต่างไป เวลาจริง 7.1 วินาทีสำหรับ haiku และ 12.5 วินาทีสำหรับ gemma4:12b ทั้งสองนับรวมเวลาเปิด Claude Code แล้ว)

```
setpoint-coach: On branch master
setpoint-coach: Not a git repo (fatal: not a git repository (or any of the parent directories): .git)
setpoint-coach: The BMS keeps guest rooms at about 24 °C to maintain a comfortable, stable temperature so the room doesn’t feel too hot or too cold when the guest returns. It’s also a way to protect the building and fixtures from large temperature swings while the room is empty.
setpoint-coach: The system maintains a consistent temperature to prevent the rooms from becoming too hot or too cold for guests. This ensures the space remains comfortable and stable when someone checks back into the room.
setpoint-coach: Usage: /spark-ask <question>
```

อ่านสองคำตอบนี้แบบวิศวกรโรงแรม: **ไม่มีคำตอบไหนพูดว่า "เพื่อประหยัดพลังงาน"** ซึ่งเป็นเหตุผลหลักของการตั้งค่าตอนห้องว่าง model ทำตามที่ system prompt ขอ (สองประโยค ไม่มีศัพท์เทคนิค) และไม่มีอะไรบอกเป้าหมายของโรงแรมกับมัน prompt เป็นส่วนหนึ่งของโค้ดใน mod ของคุณ: เทสมันเหมือนเทสโค้ด

✓ Checkpoint: validate แสดง `$.model.complete`, `$.http.fetch` และ `$.process.run` และคุณบอกได้ว่าตัวไหนเก็บข้อมูลแขกไว้ในโรงแรม

## 5 · รูปแบบไหนเหมาะ?

| คุณต้องการ | ใช้ | ใครเป็นคนเริ่ม | ต้นทุน | ตัวอย่างใน copilot HVAC ของโรงแรม |
|---|---|---|---|---|
| ข้อมูลหรือการกระทำที่เรียกได้ทันที | slash command (`$.command.register` + `command.run`) | คุณ | ไม่มี | `/setpoint 1203` อ่านค่าห้องจาก BMS |
| ให้ Claude ใช้ข้อมูลของคุณกลางงาน | tool (`$.tool.register` + `tool.call`) | model | token ของ turn นั้น | `room_setpoint` ระหว่างที่ Claude ร่างคำตอบเรื่องห้องร้อน |
| การตัดสินเล็ก ๆ ราคาถูก | `$.model.complete({ model: 'haiku' })` | hook ของคุณ | การเรียกเล็ก ๆ หนึ่งครั้ง | จัดประเภทคำร้องเรียนของแขกเป็น "ร้อนไป" / "หนาวไป" / "เสียงดัง" |
| ข้อมูลที่ต้องอยู่ในสถานที่ | `$.http.fetch` ไปที่ model ในเครื่อง | hook ของคุณ | ฮาร์ดแวร์ของคุณ | อธิบาย alarm ของ chiller บน DGX Spark ในห้องเครื่อง |
| คำตอบจากโปรแกรมที่มีอยู่แล้ว | `$.process.run([...])` | hook ของคุณ | process หนึ่งตัว | `git`, `pytest`, CLI ส่งออกข้อมูลจาก BMS |

หลักง่าย ๆ: ถ้าคนเป็นคนเลือกเวลา ทำเป็น command ถ้า model เป็นคนเลือกเวลา ทำเป็น tool ถ้า hook ต้องใช้ ให้เรียกตรง ๆ และใส่ timeout ทุกครั้ง

## 6 · เทสโดยไม่ใช้เครือข่าย

`claude plugin test` รัน `tests/*.test.ts` โดยไม่มี session ไม่ต้อง sign-in และไม่มีเครือข่าย การเรียกออกข้างนอกแต่ละอย่างเป็น stub hook ที่อยู่ใต้ plugin ของคุณ บทที่ 08 อธิบายชุดเทสโดยละเอียด ตรงนี้เป็นรูปแบบสำหรับการเรียกทั้งสาม (`tests/setpoint-coach.test.ts`):

```ts
import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

const START = { cwd: '/hotel', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const
const USAGE = { input_tokens: 30, output_tokens: 20, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }

function engine(on: On) {
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
}

test('/hvac-explain shows the answer, or why there is none', async ($, on) => {
  engine(on)
  let calls = 0
  on('model.complete', ($, e) => {
    calls += 1
    expect(e.model).toBe('haiku')
    return calls === 1
      ? { value: { isAnswered: true, text: ' 24 °C saves energy while the room is empty. ', usage: USAGE } }
      : { value: { isAnswered: false, reason: 'aborted', usage: USAGE } }
  })
  await $.session.start(START)
  expect((await $.command.run(cmd('hvac-explain', 'why 24?'))).text).toBe('24 °C saves energy while the room is empty.')
  expect((await $.command.run(cmd('hvac-explain', 'why 24?'))).text).toBe('No answer (aborted)')
})

test('/spark-ask posts to the local model and reads choices[0]', async ($, on) => {
  engine(on)
  on('http.fetch', ($, e) => {
    expect(e.url).toContain('/v1/chat/completions')
    expect(JSON.parse(e.init?.body ?? '{}').model).toBe('gemma4:12b')
    return { value: { ok: true, status: 200, headers: {}, text: JSON.stringify({ choices: [{ message: { content: 'Local answer.' } }] }) } }
  })
  await $.session.start(START)
  expect((await $.command.run(cmd('spark-ask', 'why 24?'))).text).toBe('Local answer.')
})

test('/git-where runs git by argv, in the session folder', async ($, on) => {
  engine(on)
  on('session.cwd', () => ({ value: '/hotel' }))
  on('process.run', ($, e) => {
    expect(e.argv).toEqual(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    expect(e.init?.cwd).toBe('/hotel')
    return { value: { exitCode: 0, stdout: 'main\n', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  await $.session.start(START)
  expect((await $.command.run(cmd('git-where'))).text).toBe('On branch main')
})
```

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/setpoint-coach
```

**Expected output** (บันทึกบน Mac เครื่องนี้)

```
tests/setpoint-coach.test.ts:
(pass) /hvac-explain shows the answer, or why there is none [19.28ms]
(pass) /spark-ask posts to the local model and reads choices[0] [8.82ms]
(pass) /git-where runs git by argv, in the session folder [7.99ms]

 3 pass
 0 fail
Ran 3 tests across 1 file. [0.17s]
```

กรณีที่สองในเทสแรกคือกรณีสำคัญ: มันพิสูจน์ว่า command บอก *เหตุผล* ที่ไม่มีคำตอบ แทนที่จะพิมพ์ `undefined`

✓ Checkpoint: เทสเขียวสามตัว และไม่มีตัวไหนแตะเครือข่ายเลย

## ในแผง Mod Gym

พิมพ์ `/mod-gym 7` (หรือกด **n** จนถึงบทที่ 07) แล้วกด **c** gym จะตรวจ `~/mod-gym-lab/hello-mod`:

| รายการตรวจ | ผ่านเมื่อ |
|---|---|
| registers a slash command | `register.tsx` มี `$.command.register(` |
| registers a tool for Claude | `register.tsx` มี `$.tool.register(` |
| serves mcp__hello-mod__… | `register.tsx` มี `mcp__hello-mod__` |
| validate lists command.run | `claude plugin validate` จบด้วย exit 0 และผลลัพธ์มีคำว่า `command.run` |

กด **a** เพื่อใส่คำถามของบทนี้ลงในช่อง prompt

## แบบฝึกหัด — ฟังก์ชันเดียว สองประตู

เพิ่มการดูค่า setpoint ของห้องให้ `setpoint-coach` ที่ทั้ง **คุณ** และ **Claude** ใช้ได้: command `/setpoint <room>` และ tool `room_setpoint` ที่ตอบจากตารางเดียวกัน (ตัวแทนของ BMS) ผ่านฟังก์ชันเดียว ห้องที่ไม่รู้จักต้องได้คำตอบที่ชัดเจน ไม่ใช่ crash เพิ่มเทสที่เรียกทั้งสองประตู

<details><summary>เฉลย</summary>

ใส่ตารางและฟังก์ชันไว้ด้านบนของ `register.ts`:

```ts
const SETPOINTS: Record<string, number> = { '1101': 24, '1203': 22, 'lobby': 23 } // a stand-in for the BMS

function setpointOf(room: string): string {
  const c = SETPOINTS[room.trim().toLowerCase()]
  return c === undefined ? 'No room "' + room + '" in this hotel.' : 'Room ' + room + ' is set to ' + c + ' °C.'
}
```

ลงทะเบียนทั้งสองตัวใน `session.start`:

```ts
await $.command.register({ name: 'setpoint', description: 'Show a room setpoint', argumentHint: '<room>' })
await $.tool.register({
  name: 'room_setpoint',
  description: 'Read the current temperature setpoint of one hotel room or space, e.g. "1203" or "lobby".',
  inputSchema: { type: 'object', properties: { room: { type: 'string' } }, required: ['room'] },
})
```

ให้บริการทั้งสองตัว:

```ts
// Exercise: one function, two doors: you type /setpoint, Claude calls room_setpoint
on('command.run', { command: 'setpoint' }, async ($, e) =>
  ({ text: e.args.trim() ? setpointOf(e.args) : 'Usage: /setpoint <room>' }))

on('tool.call', { tool: 'mcp__setpoint-coach__room_setpoint' }, async ($, e) => ({ result: setpointOf(String(e.room ?? '')) }))
```

เทส:

```ts
test('/setpoint and room_setpoint answer from the same table', async ($, on) => {
  engine(on)
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  await $.session.start(START)
  expect((await $.command.run(cmd('setpoint', '1203'))).text).toBe('Room 1203 is set to 22 °C.')
  const r = await $.tool.call({ tool: 'mcp__setpoint-coach__room_setpoint', room: 'lobby' })
  expect(r.result).toBe('Room lobby is set to 23 °C.')
  expect((await $.tool.call({ tool: 'mcp__setpoint-coach__room_setpoint', room: '9999' })).result).toContain('No room')
})
```

**Expected output** (บันทึกบน Mac เครื่องนี้: `claude plugin test` แล้วลองทั้งสองประตูจริง)

```
(pass) /setpoint and room_setpoint answer from the same table [9.08ms]

 4 pass
 0 fail
```

```bash
claude -p "What is room 1203 set to? Use your tool. One line." --plugin-dir ~/mod-gym-lab/setpoint-coach --allowedTools=mcp__setpoint-coach__room_setpoint
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/setpoint 1203"
```

```
Room 1203 is set to 22 °C.
setpoint-coach: Room 1203 is set to 22 °C.
```

บรรทัดแรกคือคำตอบของ Claude หลังเรียก tool บรรทัดที่สองคือ command ซึ่งไม่มี turn เลย

</details>

✓ Checkpoint: ทั้งสองประตูตอบจากฟังก์ชันเดียวกัน และเทสครอบคลุมห้องที่ไม่รู้จัก

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `/reps` ไม่ขึ้นใน typeahead | ลงทะเบียน command นอก `session.start` หรือ `session.start` throw ก่อนถึงบรรทัดนั้น | ลงทะเบียนใน `session.start`; รัน `claude --debug` แล้วหาบรรทัด hook ที่ถูกข้าม |
| Claude บอกว่าไม่มี tool `log_reps` | mod ไม่ได้โหลด หรือ `$.tool.register` throw | `claude plugin validate`; ดูบรรทัดสีจางตอน reload ใน transcript |
| การเรียก tool ล้มเหลวว่าไม่มี hook ตอบ | matcher ไม่ใช่ชื่อเต็ม `mcp__<plugin>__<name>` | ใช้ `mcp__hello-mod__log_reps` ให้ตรงทุกตัว เป็น literal |
| ในโหมด `-p` Claude ขออนุญาตแล้วหยุด | โหมด headless ไม่มีใครกดอนุญาต tool | ใส่ `--allowedTools=mcp__hello-mod__log_reps` |
| `/hvac-explain` พิมพ์ `No answer (aborted)` | หมดเวลา `timeoutMs` | เพิ่มเวลา หรือทำ prompt ให้สั้นลง |
| `/spark-ask` พิมพ์ `Local model failed: HTTP 404` | ยังไม่ได้ pull model | `ollama pull gemma4:12b` หรือเปลี่ยน `model` |

## ต่อไป

[บทที่ 08 — Test, validate, ship](../08_test_and_ship/TUTORIAL.th.md): ทำให้เทสเหล่านี้เป็นนิสัย type-check mod เผยแพร่มันเพื่อให้เพื่อนร่วมชั้น `/plugin install` ได้ และตรวจ starter mod ที่ถูกเขียนขึ้นโดยไม่เคยถูกรันเลย
