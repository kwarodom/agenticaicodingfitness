# ▶ Mod Gym Lesson 06 — State ที่อยู่รอดหลัง reload

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณจะได้เรียนวิธีเปลี่ยนหน้าตาและพฤติกรรมของ Claude Code เอง จากในเทอร์มินัลที่คุณใช้เขียนโค้ดทุกวัน คอร์สนี้มีสองครึ่งที่สอนแปดบทเรียนเดียวกัน คือ tutorial ชุดนี้ (ฉบับยาว) และ **🏋 Mod Gym** ซึ่งเป็น panel ภายใน Claude Code (ฉบับสั้น พร้อมตัวตรวจ) Mod Gym เองก็เป็น mod ทุกอย่างที่มันทำบนหน้าจอจึงเป็นสิ่งที่คุณจะได้เรียนสร้าง ตลอดบทเรียนคุณจะค่อย ๆ ขยาย lab mod ตัวเล็กหนึ่งตัวชื่อ `hello-mod` ฉบับที่เสร็จแล้วอยู่ใน `week29/reference/hello-mod`

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เลือกว่าค่าหนึ่งค่าควรอยู่ที่ไหน: module variable, atom ของ `$.state` หรือ `$.store` โดยดูจากว่ามันจะเป็นอย่างไรเมื่อ reload, เมื่อ `/clear` และใน session ของวันพรุ่งนี้
- ย้ายตัวนับของ `hello-mod` ไปเป็น atom ด้วย `atom`, `read` และ `update` แล้วดู band วาดใหม่เองโดยไม่ต้องมี `$.ui.invalidate`
- เขียน contract ของ `PluginState` ใน `types/index.d.ts` ระบุมันใน `plugin.json` และเจอกับกับดักของ validator ที่เราวัดได้: shape ต้องเขียนแบบ inline
- เห็นกฎ "การวาดต้องบริสุทธิ์" ถูกบังคับใช้จริง: render hook ที่เขียน state จะถูกข้าม
- เก็บ `reps` ข้าม session ไว้ใน `$.store` และทดสอบด้วย store ที่คุณมองเข้าไปข้างในได้

**Time** ~45 นาที · **Difficulty** ระดับกลาง · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้

**แหล่งอ้างอิง:** [Draw in the interface](https://code.claude.com/docs/en/plugins/mods/interface) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Use the mods API](https://code.claude.com/docs/en/plugins/mods/api) · [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · ไฟล์ type declaration ที่ Claude Code 2.1.292 เขียนให้ทุก mod (`.claude-plugin/types/claude-code/index.d.ts`) ซึ่งถือเป็นข้อยุติเมื่อหน้าเอกสารกับ type ไม่ตรงกัน

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| บทเรียน 05 | `hello-mod` มี `hooks/register.tsx`, band และ pane | บทนี้เปลี่ยนสิ่งที่ band อ่าน |
| Claude Code 2.1.287+ | `claude --version` → `2.1.292 (Claude Code)` บนเครื่องนี้ | mod ต้องใช้ |
| โฟลเดอร์แล็บของคุณ | `claude plugin validate ~/mod-gym-lab/hello-mod` ผ่าน | Mod Gym ตรวจโฟลเดอร์นี้ |

✓ Checkpoint: band ของคุณแสดงจำนวน tool call ที่มาจาก module variable `let calls`

## 1 · สามที่สำหรับเก็บค่า

| ที่ไหน | อยู่ถึงเมื่อไร | ใช้ร่วมกับใคร | วาดตัวที่อ่านใหม่ให้ไหม | ใช้กับ |
|---|---|---|---|---|
| module `let` | การบันทึกหรือ reload module ครั้งถัดไป | module นี้เท่านั้น | ไม่ (ต้อง `$.ui.invalidate` เอง) | cache ที่หายได้ |
| atom ของ `$.state` | session จบ; ถูก reset ด้วย `/clear`, `/resume`, `/branch` ด้วย | ทั้ง session (plugin อื่นอ่านได้) | **ใช่** | สิ่งที่ภาพวาดแสดง |
| `$.store` | ตลอดไป บนเครื่องนี้ | ทุก session ของ plugin นี้บนเครื่องนี้ (สูงสุด 4 MiB) | ไม่ | ความคืบหน้า, การตั้งค่า, ประวัติ |

ทำไมเรื่องนี้สำคัญกับ mod เป็นพิเศษ: **reload คือการโหลด module ใหม่ทั้งหมด** เมื่อคุณบันทึก `register.tsx` ระหว่างพัฒนา Claude Code จะรัน `register` ใหม่ใน environment ใหม่ ยิง `session.start` อีกครั้ง และทิ้ง timer เก่า `let` ทุกตัวเริ่มนับใหม่ ส่วนค่าใน `$.state` (host ถือไว้ตลอด session) และใน `$.store` (เก็บบนดิสก์) ยังอยู่

ดังนั้น `let calls = 0` จากบทเรียน 03 จะกลับเป็นศูนย์ทุกครั้งที่คุณบันทึก นี่คือบั๊กที่บทนี้จะแก้

✓ Checkpoint: สำหรับ `calls`, `reps` และ "band ถูกซ่อนอยู่ไหม" คุณบอกได้ว่าแต่ละตัวควรอยู่แถวไหนของตาราง และเพราะอะไร

## 2 · Atom: `atom`, `read`, `update`

atom คือชื่อแบบมี type สำหรับค่าหนึ่งค่าใน `$.state` พร้อมค่าเริ่มต้น:

```ts
import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

// L06 · values the session keeps across reloads, declared in types/index.d.ts
const reps = atom({ plugin: 'hello-mod', key: 'reps' } as const, 0)
const calls = atom({ plugin: 'hello-mod', key: 'calls' } as const, 0)
const isBandHidden = atom({ plugin: 'hello-mod', key: 'isBandHidden' } as const, false)
```

- `plugin` และ `key` ต้องเป็น **literal** (`as const`) เพื่อให้ `claude plugin validate` แสดงได้ว่า module อ่านและเขียนอะไร ถ้าใช้ `key: K` (ตัวแปร) validate จะ fail: `the state library's atom takes a source the scan can read: a reference whose plugin and key are string literals …`
- `await read($, reps)` อ่านค่า ภายใน render hook การอ่านนั้นยัง **subscribe** ภาพวาดไว้ด้วย เมื่อค่าเปลี่ยน ภาพวาดที่อ่านค่านั้นจะถูกวาดใหม่ตรงตัว
- `await update($, reps, n => n + 1)` อ่าน ใช้ฟังก์ชันของคุณ แล้วเขียน และลองใหม่ถ้ามีคนเขียนแทรกระหว่างนั้น การกดสองครั้งเร็ว ๆ จึงนับครบทั้งคู่

ตัวนับจะกลายเป็น:

```ts
// L03 · wrap: let every tool run, then count it (now in an atom, not a let)
on('tool.call', async ($, e, next) => {
  const ran = await next(e)
  await update($, calls, n => n + 1)
  $.ui.status('hello-mod · ' + (await read($, calls)) + ' tool calls')
  return ran
})
```

ไม่มี `$.ui.invalidate`: band อ่าน `calls` อยู่ การเขียน `calls` จึงทำให้มันวาดใหม่

และ band แสดง reps พร้อมปุ่มเพิ่มหนึ่งครั้ง และปุ่มซ่อน band:

```tsx
// L05 · a band above the prompt; reading atoms subscribes it to their changes
on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
  if (e.props.hasSurvey || (await read($, isBandHidden))) return next(e)
  const { Box, Text, Button } = $.ui.resolve(e)
  const below = await next(e)
  return (
    <Box flexDirection="column">
      {below}
      <Box flexDirection="row" columnGap={1}>
        <Text color="success">hello-mod</Text>
        <Text dimColor>{await read($, reps)} reps</Text>
        <Button key="rep" onPress={() => update($, reps, n => n + 1)}>
          +1 rep
        </Button>
        <Button key="hide" dimColor onPress={() => update($, isBandHidden, () => true)}>
          Hide
        </Button>
      </Box>
    </Box>
  )
})
```

✓ Checkpoint: `register.tsx` ไม่มี `let calls` เหลืออยู่ และไม่มี `$.ui.invalidate`

## 3 · Contract: `types/index.d.ts`

atom ทุกตัวต้องประกาศไว้ใน **contract**: ไฟล์ `.d.ts` ที่บอก Claude Code (และ TypeScript) ว่าแต่ละค่ามี type อะไร สร้าง `types/index.d.ts`:

```ts
declare module 'claude-code' {
  interface PluginState {
    'hello-mod': { reps: number; calls: number; isBandHidden: boolean }
  }
}
```

และระบุมันใน `.claude-plugin/plugin.json`:

```json
{
  "name": "hello-mod",
  "version": "0.6.0",
  "description": "My first mod",
  "types": "./types/index.d.ts"
}
```

validate:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (บันทึกบน Mac เครื่องนี้; ย่อ path)

```
Validating plugin manifest: …/hello-mod/.claude-plugin/plugin.json

⚠ Found 1 warning:

  ❯ author: No author information provided. Consider adding author details for plugin attribution

  ❯ types ./types/index.d.ts declares on $: nothing (no EngineInterface member)
  ❯ types ./types/index.d.ts declares state: hello-mod.reps, hello-mod.calls, hello-mod.isBandHidden

Validating hooks: …/hello-mod/hooks/hooks.json

  ❯ ./register.tsx hooks: session.start, ui.render{component=Spinner}, tool.call{tool=Bash}, tool.call, ui.render{component=AbovePrompt}, state.set{plugin=hello-mod, key=reps}
  ❯ ./register.tsx gating hook with .catch: tool.call{tool=Bash}
  ❯ ./register.tsx gating hook without .catch: tool.call
  ❯ ./register.tsx gating hook without .catch: state.set{plugin=hello-mod, key=reps}
  ❯ ./register.tsx calls: $.state.get, $.state.set, $.store.get, $.store.set, $.ui.ask, $.ui.resolve, $.ui.status
  ❯ ./register.tsx state writes: hello-mod.calls, hello-mod.isBandHidden, hello-mod.reps
  ❯ ./register.tsx state reads: hello-mod.calls, hello-mod.isBandHidden, hello-mod.reps

✔ Validation passed with warnings
```

(ผลลัพธ์นี้รวม hook `session.start` และ `state.set` จากหัวข้อ 5 ไว้แล้ว) อ่านบรรทัด `types` สองบรรทัดคู่กับบรรทัด `state` สองบรรทัด: ทุก key ที่ module อ่านหรือเขียน ต้องเป็น key ที่ contract ประกาศไว้

**กับดักที่เราวัดได้: เขียน shape แบบ inline** contract ที่ดูเรียบร้อยแต่ตั้งชื่อ shape ผ่าน type alias จะ *ไม่ถูก* validator อ่าน:

```ts
export type HelloModState = { reps: number; calls: number; isBandHidden: boolean }

declare module 'claude-code' {
  interface PluginState {
    'hello-mod': HelloModState
  }
}
```

```
✘ Found 3 errors:

  ❯ modules../register.tsx: hello-mod.calls is not declared: the manifest's types contract must name it in interface PluginState { hello-mod: { calls: ... } }
  ❯ modules../register.tsx: hello-mod.isBandHidden is not declared: the manifest's types contract must name it in interface PluginState { hello-mod: { isBandHidden: ... } }
  ❯ modules../register.tsx: hello-mod.reps is not declared: the manifest's types contract must name it in interface PluginState { hello-mod: { reps: ... } }


✘ Validation failed
```

คุณจะเจอ error สามตัวเดียวกันนี้ถ้าลืมใส่ `"types"` ใน `plugin.json` alias สำหรับ type ของ **ค่า** ใช้ได้ (contract ของ Mod Gym ใช้ `checks: CheckOutcome[]`) แต่ object ของ key ที่อยู่ใต้ชื่อ plugin ของคุณต้องเขียนออกมาเต็ม ๆ

✓ Checkpoint: validate พิมพ์ `declares state: hello-mod.reps, hello-mod.calls, hello-mod.isBandHidden`

## 4 · การวาดต้องบริสุทธิ์

render hook **อ่าน** state แล้วคืน tree มันไม่เขียนเลย การเขียนต้องอยู่ใน handler (`onPress`) หรือใน hook ของ event อื่น (`tool.call`, `session.start`, …)

validator มองไม่เห็นเรื่องนี้ ลองทำดูสักครั้งเพื่อให้รู้จักข้อความ เราใส่บรรทัดผิดหนึ่งบรรทัดใน band:

```tsx
await update($, calls, n => n + 1) // wrong: a render hook writes state
```

`claude plugin validate` ยังผ่าน แต่การ mount band ใน test ไม่ผ่าน:

```
hello-mod's ui.render hook was skipped: hello-mod: HooksError: hello-mod: state.set: denied: it was made while ui.render is being dispatched, and drawing is pure (a render hook reads state and returns a tree; write from a handler such as onPress, or from another event) (host check)
```

band hook ที่ถูกข้ามแปลว่าไม่มี band เลย เอาบรรทัดนั้นออก

อีกกฎหนึ่งจากเอกสาร: ใน handler ให้เขียนจากค่าที่อ่าน **ตอนนี้** ไม่ใช่จากค่าที่ closure จับไว้ตอนวาด `update($, reps, n => n + 1)` ทำให้คุณแล้ว ส่วน `$.state.set(reps, shownValue + 1)` ที่ใช้ค่าซึ่งจับไว้ตอนวาดอาจทำให้การกดหายไปหนึ่งครั้ง

✓ Checkpoint: คุณบอกได้ว่าการเขียนแต่ละครั้งใน mod ของคุณเกิดที่ไหน (handler ตัวไหนหรือ event ไหน)

## 5 · เก็บไว้ข้าม session ด้วย `$.store`

`$.state` จบไปพร้อม session สำหรับ reps ที่ควรยังอยู่ในวันพรุ่งนี้ ให้เขียนลง `$.store` ด้วย และโหลดใน `session.start`:

```ts
// L06 · load what lasts across sessions
on('session.start', async ($, e, next) => {
  const saved = await $.store.get('reps')
  if (typeof saved === 'number') await update($, reps, () => saved)
  return next(e)
})

// L06 · keep reps past the session: whenever the atom changes, write it to the store
on('state.set', { plugin: 'hello-mod', key: 'reps' }, async ($, e, next) => {
  const r = await next(e)
  await $.store.set('reps', e.value)
  return r
})
```

`state.set` คือ event ที่การเขียน atom ทุกครั้งต้องผ่าน การ hook key ของคุณเองตรงนั้นทำให้มีที่เดียวที่สะท้อนค่า atom ลง store ซึ่งเป็นสิ่งที่ `week29/reference/hello-mod` ทำ tool `log_reps` ในบทเรียน 07 เขียน `reps` จาก hook `tool.call` และการเขียนนั้นไปถึง hook นี้

**สิ่งที่เราวัดได้เกี่ยวกับการกดปุ่ม** ใน `claude plugin test` การเขียนที่มาจาก handler `onPress` ของ band เองทำให้ band วาดใหม่ แต่ **ไม่** ไปถึง hook `state.set` ของ plugin: store ยังว่างเปล่า ส่วนการเขียนจาก hook `tool.call` ไปถึง จนกว่าคุณจะตรวจเรื่องนี้ใน session จริงแล้ว ให้เขียน store ใน handler ด้วย:

```tsx
<Button
  key="rep"
  onPress={async () => {
    await update($, reps, n => n + 1)
    await $.store.set('reps', await read($, reps))
  }}
>
  +1 rep
</Button>
```

ข้อเท็จจริงเกี่ยวกับ `$.store` ที่ควรรู้:

- มันอยู่บนเครื่องนี้: ทุก session ของ `hello-mod` บนแล็ปท็อปนี้เห็น `reps` ค่าเดียวกัน และเทอร์มินัลสองหน้าต่างที่เปิดพร้อมกันก็ใช้ร่วมกัน
- ค่าต้องเป็น JSON ธรรมดา เก็บให้เล็ก (เพดานคือ 4 MiB)
- มันอยู่รอดผ่าน reload, `/clear` และการรีสตาร์ต ถ้าจะเริ่มใหม่ต้องลบ key เอง (`$.store.delete('reps')`)

✓ Checkpoint: `register.tsx` มี `$.store.set(` และมี hook `session.start` ที่อ่าน `$.store.get('reps')`

## 6 · ทดสอบด้วย store ที่มองเข้าไปข้างในได้

`mock.store(on)` ให้ store ที่ใช้งานได้กับ plugin ของคุณ แต่คุณมองเข้าไปข้างในไม่ได้ ถ้าจะตรวจว่าเขียนอะไรลงไป ให้ตอบ `store.get` และ `store.set` เอง บันทึกไฟล์นี้เป็น `tests/state.test.ts`:

```ts
import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

const START = { cwd: '/lab', surface: 'terminal', isInteractive: true } as const
const BAND = {
  component: 'AbovePrompt',
  props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns: 80, scroll: { offset: 0, bodyRows: 40 }, view: {} },
} as const

// A store you can look inside: the plugin's $.store calls land here.
function engine(on: On, store: Record<string, unknown>) {
  on('store.get', ($, e) => ({ value: store[e.key] }))
  on('store.set', ($, e) => { store[e.key] = e.value; return { value: undefined } })
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Box({}))
}

test('+1 rep redraws the band and is written to $.store', async ($, on) => {
  const store: Record<string, unknown> = {}
  engine(on, store)
  await $.session.start(START)
  const band = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  expect(await band.find({ type: 'Text', text: '0 reps' })).toBeDefined()
  await band.press({ key: 'rep' })
  await band.press({ key: 'rep' })
  expect(await band.find({ type: 'Text', text: '2 reps' })).toBeDefined()
  expect(store.reps).toBe(2)
  await band.unmount()
})

test('reps saved by an earlier session are loaded at session.start', async ($, on) => {
  engine(on, { reps: 7 })
  await $.session.start(START)
  const band = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  expect(await band.find({ type: 'Text', text: '7 reps' })).toBeDefined()
  await band.unmount()
})

test('Hide removes the band until the session ends', async ($, on) => {
  engine(on, {})
  await $.session.start(START)
  const band = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  await band.press({ key: 'hide' })
  expect(await band.find({ key: 'rep' })).toBeUndefined()
  await band.unmount()
})
```

ใน test `session.start` ต้องการ input ครบ (`cwd`, `surface`, `isInteractive`) ส่วนใน session จริง Claude Code เติมให้เอง

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/hello-mod
```

**Expected output** เมื่อปุ่ม `+1 rep` เรียกแค่ `update` (บันทึกบน Mac เครื่องนี้): band วาดใหม่เป็น `2 reps` แต่ store ยังว่าง

```
tests/state.test.ts:
(fail) +1 rep redraws the band and is written to $.store [24.41ms]
  AssertionError: expect(received).toBe()
  
  Expected: 2
  Received: undefined
(pass) reps saved by an earlier session are loaded at session.start [11.34ms]
(pass) Hide removes the band until the session ends [10.39ms]

 2 pass
 1 fail
Ran 3 tests across 1 file. [0.17s]
```

**Expected output** เมื่อ handler เขียน store ด้วย (หัวข้อ 5):

```
tests/state.test.ts:
(pass) +1 rep redraws the band and is written to $.store [26.36ms]
(pass) reps saved by an earlier session are loaded at session.start [11.77ms]
(pass) Hide removes the band until the session ends [10.04ms]

 3 pass
 0 fail
Ran 3 tests across 1 file. [0.19s]
```

นี่แหละคือสิ่งที่ test มีไว้: หน้าจอดูถูกต้อง แต่ข้อมูลไม่ได้ถูกบันทึก

✓ Checkpoint: test ผ่านทั้งสามตัว

## ใน panel ของ Mod Gym

พิมพ์ `/mod-gym 6` แล้ว `/mod-gym` และกด **c** ตัวตรวจทั้งห้าคือ:

| ตัวตรวจ | ผ่านเมื่อ |
|---|---|
| types/index.d.ts exists | มี `hello-mod/types/index.d.ts` |
| plugin.json names the types | `plugin.json` มี field `"types"` |
| uses an atom | `register.tsx` เรียก `atom(` |
| persists to $.store | `register.tsx` เรียก `$.store.set(` |
| validate passes (contract checked) | `claude plugin validate` exit 0 แปลว่าทุก atom ถูกประกาศแล้ว |

Mod Gym เองก็สร้างแบบนี้: บทเรียน, แท็บ, ผลตรวจ และ Events tape ของมันเป็น atom (อยู่รอดเมื่อ panel ถูก reload) ส่วนบทเรียนที่คุณผ่าน, โฟลเดอร์แล็บ และ streak อยู่ใน `$.store` (อยู่ถึงวันพรุ่งนี้) กด **3** เพื่อดูแท็บ Progress ซึ่งเป็นฝั่ง `$.store`

## ลองทำเอง — ทำให้ Hide จำได้

ตอนนี้ **Hide** อยู่ได้แค่จนจบ session ทำให้มันอยู่จนกว่าคุณจะเปิด band กลับมา: เก็บตัวเลือกไว้ใน `$.store` โหลดมันใน `session.start` แล้วเพิ่ม test ที่กด **Hide** เริ่ม session ใหม่ และไม่พบ band

ตัวตรวจสำหรับตัวเอง: test ใหม่ผ่าน และสามตัวจากหัวข้อ 6 ก็ผ่านด้วย

<details><summary>เฉลย</summary>

ใน `session.start` หลังจากโหลด `reps`:

```ts
if ((await $.store.get('isBandHidden')) === true) await update($, isBandHidden, () => true)
```

ปุ่ม Hide เขียนทั้งสองที่:

```tsx
<Button
  key="hide"
  dimColor
  onPress={async () => {
    await update($, isBandHidden, () => true)
    await $.store.set('isBandHidden', true)
  }}
>
  Hide
</Button>
```

เพิ่มใน `tests/state.test.ts`:

```ts
test('Hide is remembered by the next session', async ($, on) => {
  const store: Record<string, unknown> = {}
  engine(on, store)
  await $.session.start(START)
  const first = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  await first.press({ key: 'hide' })
  await first.unmount()
  expect(store.isBandHidden).toBe(true)

  await $.session.start(START) // what a reload or tomorrow's session runs
  const again = await $.ui.mount({ plugin: 'hello-mod', surface: 'terminal', ...BAND })
  expect(await again.find({ key: 'rep' })).toBeUndefined()
  await again.unmount()
})
```

**Expected output** (บันทึกบน Mac เครื่องนี้)

```
tests/state.test.ts:
(pass) +1 rep redraws the band and is written to $.store [24.44ms]
(pass) reps saved by an earlier session are loaded at session.start [10.60ms]
(pass) Hide removes the band until the session ends [10.54ms]
(pass) Hide is remembered by the next session [12.57ms]

 4 pass
 0 fail
Ran 4 tests across 1 file. [0.19s]
```

วิธีเปิด band กลับมาคือ command ซึ่งเป็นบทเรียน 07: `/mod-gym band on` ของ Mod Gym ทำแบบนี้พอดี

</details>

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `hello-mod.reps is not declared` | ไม่มี contract, ไม่มี `"types"` ใน `plugin.json` หรือ shape เป็น type alias | เขียน `'hello-mod': { reps: number; … }` แบบ inline และระบุไฟล์ใน `plugin.json` |
| ตัวนับกลับเป็น 0 เมื่อบันทึกไฟล์ | มันยังเป็น module `let` | ย้ายไปเป็น atom |
| `state.set: denied: … drawing is pure` | render hook เขียน state | เขียนจาก `onPress` หรือจาก event อื่น |
| band แสดงตัวเลขถูก แต่พรุ่งนี้เป็น 0 | การเขียนไม่เคยไปถึง `$.store` | เขียน store ใน handler (หัวข้อ 5) และทดสอบด้วย store ที่มองเข้าไปได้ |
| `the state library's atom takes a source the scan can read …` | `plugin` หรือ `key` เป็นตัวแปร ไม่ใช่ literal | `atom({ plugin: 'hello-mod', key: 'reps' } as const, 0)` |
| test fail ด้วย `no implementation for store.get` | plugin อ่าน store แต่ไม่มีอะไรตอบ | `mock.store(on)` หรือ stub `store.get`/`store.set` ของคุณเอง |

## ต่อไป

[Lesson 07 — Commands, tools and models](../07_commands_tools_models/TUTORIAL.th.md): เพิ่ม `/reps` command ที่ตอบได้โดยไม่ต้องมี turn ของ Claude และ `log_reps` tool ที่ Claude เรียกเพื่อเพิ่มค่าใน atom เดียวกันได้
