# ▶ Mod Gym Lesson 05 — วาดบนหน้าจอ: pane, band, status, toast

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณจะได้เรียนวิธีเปลี่ยนหน้าตาและพฤติกรรมของ Claude Code เอง จากในเทอร์มินัลที่คุณใช้เขียนโค้ดทุกวัน คอร์สนี้มีสองครึ่งที่สอนแปดบทเรียนเดียวกัน คือ tutorial ชุดนี้ (ฉบับยาว) และ **🏋 Mod Gym** ซึ่งเป็น panel ภายใน Claude Code (ฉบับสั้น พร้อมตัวตรวจ) Mod Gym เองก็เป็น mod ทุกอย่างที่มันทำบนหน้าจอจึงเป็นสิ่งที่คุณจะได้เรียนสร้าง ตลอดบทเรียนคุณจะค่อย ๆ ขยาย lab mod ตัวเล็กหนึ่งตัวชื่อ `hello-mod` ฉบับที่เสร็จแล้วอยู่ใน `week29/reference/hello-mod`

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เลือกที่ที่เหมาะสำหรับแสดงสิ่งต่าง ๆ: status line, toast, band เหนือช่อง prompt หรือ pane
- วาดด้วย element ที่ได้จาก `$.ui.resolve(e)` (ไม่ใช่จากการ import) และดูว่าตาราง element เปลี่ยนไปตาม surface อย่างไร: terminal, desktop, VS Code, mobile
- เพิ่ม band ให้ `hello-mod` ที่วาด **ใต้** สิ่งที่ plugin อื่นวาดไว้แล้ว แทนที่จะลบทับ
- เปิด pane จากปุ่มด้วย `$.ui.open` แล้ววาดมันด้วย hook `ui.render` บน `Pane`
- อ่านบรรทัดจาง ๆ *refused … the engine drew its own* และเขียน UI test ที่ mount band และ pane ของคุณบนสอง surface แล้วกดปุ่มของมัน

**Time** ~45 นาที · **Difficulty** ระดับกลาง · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้

**แหล่งอ้างอิง:** [Draw in the interface](https://code.claude.com/docs/en/plugins/mods/interface) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Use the mods API](https://code.claude.com/docs/en/plugins/mods/api) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · ไฟล์ type declaration ที่ Claude Code 2.1.292 เขียนให้ทุก mod (`.claude-plugin/types/claude-code/index.d.ts`) ซึ่งถือเป็นข้อยุติเมื่อหน้าเอกสารกับ type ไม่ตรงกัน

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| บทเรียน 02–04 | `hello-mod` ของคุณมี suffix ของ spinner, ตัวนับ tool และ guard กัน force-push | บทนี้เพิ่มการวาดต่อจากนั้น |
| Claude Code 2.1.287+ | `claude --version` → `2.1.292 (Claude Code)` บนเครื่องนี้ | mod ต้องใช้ |
| โฟลเดอร์แล็บของคุณ | `ls ~/mod-gym-lab/hello-mod/hooks` แสดง `hooks.json register.ts` | Mod Gym ตรวจโฟลเดอร์นี้ (เปลี่ยนได้ด้วย `/mod-gym lab <path>`) |

✓ Checkpoint: `claude plugin validate ~/mod-gym-lab/hello-mod` จบด้วย `✔ Validation passed`

## 1 · สี่วิธีในการแสดงบางอย่างบนหน้าจอ

| คุณต้องการ | เรียก | อยู่นานแค่ไหน | เริ่ม turn ของ Claude ไหม? |
|---|---|---|---|
| ข้อมูลสั้น ๆ ที่เห็นตลอด | `$.ui.status(text)` (`undefined` คือล้าง) | จนกว่าคุณจะเปลี่ยน | ไม่ |
| การแจ้งเตือนที่หายไปเอง | `$.ui.toast(text)` | ไม่กี่วินาที | ไม่ |
| แถวหนึ่งแถวเหนือช่อง prompt | hook `ui.render` บน `{ component: 'AbovePrompt' }` | ทุกครั้งที่วาดใหม่ | ไม่ |
| panel ด้านข้างแบบ Mod Gym | `$.ui.open({ id, title })` + hook `ui.render` บน `{ component: 'Pane', requestId: id }` | จนกว่าจะปิด | ไม่ |

สองแบบแรกเป็นการเรียกธรรมดาที่ทำได้จาก hook ไหนก็ได้ สองแบบหลังเป็น **render hook**: Claude Code จะถาม hook ของคุณว่า "ตรงนี้วาดอะไร?" ทุกครั้งที่วาดจุดนั้น และ hook ของคุณตอบด้วย tree ของ element หรือด้วย `next(e)` เพื่อปล่อยจุดนั้นให้ส่วนที่เหลือของ chain

คุณใช้ `$.ui.status` ไปแล้วในบทเรียน 03 Mod Gym ใช้ครบทั้งสี่: รายการ status `Mod Gym L01 · 0/8`, toast *lesson passed 💪*, band `🏋 Mod Gym` และตัว panel เอง

✓ Checkpoint: คุณบอกได้ว่าตอนผ่านบทเรียน 04 Mod Gym ใช้แบบไหนในสี่แบบนี้

## 2 · Element มาจาก surface ไม่ได้มาจากการ import

mod ไม่มี React ไม่มี DOM และไม่มี element ที่เป็น global element ที่ hook วาดได้มาจาก surface ที่ขอให้วาด:

```tsx
const { Box, Text, Button, Markdown } = $.ui.resolve(e)
```

`e.surface` บอกว่าภาพจะไปแสดงที่ไหน และแต่ละ surface มีตารางของตัวเอง:

| Surface | Element |
|---|---|
| `terminal` | `Box` `Text` `Button` `Input` `Select` `Link` `Code` `Markdown` `Client` `Raster` `Image` |
| `desktop` | `Box` `Text` `Button` `Input` `Select` `Svg` `Link` `Code` `Markdown` `Client` |
| `vscode` | `Box` `Text` `Button` `Input` `Select` `Svg` `Link` `Code` `Markdown` |
| `mobile` | `Box` `Text` `Button` `Svg` `Link` `Code` `Markdown` (ยังไม่มีช่องกรอกข้อมูล) |

ดังนั้น band ที่วาดด้วย `Box`, `Text` และ `Button` ใช้ได้ทุก surface ส่วน band ที่วาด `Svg` จะใช้ในเทอร์มินัลไม่ได้ ให้เขียนเป็น JSX: มัน compile กับ `h` ที่เป็น global และไฟล์ต้องลงท้ายด้วย `.tsx`

prop บางตัวที่คุณจะใช้บ่อย:

| Element | prop ที่ควรรู้ |
|---|---|
| `Box` | `flexDirection`, `columnGap`, `rowGap`, `width`, `flexGrow`, `overflow` |
| `Text` | `color` (theme key เช่น `success`, `error`, `claude` หรือค่าสี), `bold`, `dimColor`, `wrap="truncate-end"` |
| `Button` | `key`, `onPress`, `hotkey` (ตัวเลขหนึ่งหลักหรือตัวอักษรพิมพ์เล็ก), `variant="primary"`, `role="dismiss"`, `dimColor`; label คือ child ที่เป็น string หนึ่งตัว |
| `Markdown` | `text`: heading, list, ตาราง และโค้ด วาดแบบเดียวกับคำตอบของ Claude |

`e.props.bodyColumns` (บน `AbovePrompt` และ `Pane`) คือความกว้างของกล่องที่คุณวาดลงไป ให้กำหนดขนาด tree ตามค่านี้ ไม่ใช่ตามความกว้างของเทอร์มินัลทั้งหมด เพราะ pane ที่ dock อยู่แคบกว่าหน้าจอ

✓ Checkpoint: คุณอธิบายได้ว่าทำไม band ที่ใช้ `Svg` จึงวาดบน desktop ได้แต่วาดในเทอร์มินัลไม่ได้

## 3 · เปลี่ยนชื่อเป็น `.tsx` ก่อน

JSX ในไฟล์ `.ts` parse ไม่ผ่าน ถ้าคุณเพิ่ม band ด้านล่างแล้วลืมเปลี่ยนชื่อ validate จะบอก:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (บันทึกบน Mac เครื่องนี้; ย่อ path)

```
Validating hooks: …/hello-mod/hooks/hooks.json

✘ Found 1 error:

  ❯ modules../register.ts: jsx: …/hello-mod/hooks/register.ts does not parse: Expected ">" but found "flexDirection" (line 38, column 12); Expected ")" but found "width" (line 38, column 35); Expected "}" but found "." (line 38, column 43); Expected ";" but found "props" (line 38, column 44); Unexpected .… [+21 chars]

✘ Validation failed
```

เปลี่ยนชื่อไฟล์และชี้ `hooks.json` ไปที่ชื่อใหม่:

```bash
# on: laptop
cd ~/mod-gym-lab/hello-mod
mv hooks/register.ts hooks/register.tsx
echo '{ "modules": ["./register.tsx"] }' > hooks/hooks.json
```

ถ้าเปลี่ยนชื่อแต่ไม่แก้ `hooks.json` validate ก็จะบอกเช่นกัน:

```
  ❯ modules../register.ts: stale: …/hello-mod/hooks/register.ts: no such file
```

✓ Checkpoint: `hooks/hooks.json` ระบุ `./register.tsx` และไฟล์นั้นมีอยู่จริง

## 4 · band ที่แบ่งแถวกับคนอื่น

เพิ่ม hook นี้ใน `register.tsx` มันวาดหนึ่งแถว: ชื่อ mod, จำนวน tool call จากบทเรียน 03 และปุ่มสองปุ่ม

```tsx
// L05 · a band above the prompt, drawn beneath what other plugins put there
on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
  if (e.props.hasSurvey) return next(e)
  const { Box, Text, Button } = $.ui.resolve(e)
  const below = await next(e)
  return (
    <Box flexDirection="column" width={e.props.bodyColumns}>
      {below}
      <Box flexDirection="row" columnGap={1}>
        <Text color="success">hello-mod</Text>
        <Text dimColor>{calls} tool calls</Text>
        <Button key="hi" onPress={() => $.ui.toast('Hi from hello-mod!')}>
          Say hi
        </Button>
        <Button key="stats" variant="primary" onPress={() => void $.ui.open({ id: PANE, title: 'hello-mod stats' })}>
          Stats
        </Button>
      </Box>
    </Box>
  )
})
```

มีสามอย่างในนี้ที่ตั้งใจใส่:

1. **`await next(e)` และ `{below}`** band คือจุดเดียวที่ทุก plugin ใช้ร่วมกัน hook ที่คืน tree ของตัวเอง **โดยไม่** เรียก `next` จะตอบแทน chain ทั้งหมด plugin ที่อยู่ด้านล่างจึงไม่ได้วาดเลย Mod Gym วาดแถว `🏋 Mod Gym` ไว้ตรงนั้น Clean View วาดแผนงานของมันไว้ตรงนั้น การเรียก `next(e)` ก่อนแล้วใส่ผลลัพธ์ลงในคอลัมน์ของคุณ จะเก็บแถวของพวกเขาไว้และเพิ่มแถวของคุณ
2. **`if (e.props.hasSurvey) return next(e)`** เมื่อ Claude Code แสดง survey ของตัวเองตรงจุดนั้น ให้หลีกทาง
3. **`key` บน `Button` ทุกตัว** key คือสิ่งที่การกดใช้หา handler ของมัน และเป็นสิ่งที่ test ใช้หาปุ่ม

handler รันอยู่ใน plugin ของคุณ `onPress` เป็น `async` ได้ และเป็นที่ที่ถูกต้องสำหรับการเปลี่ยนแปลงค่าต่าง ๆ (บทเรียน 06 จะแสดงว่าทำไม render hook ห้ามทำ)

**การวาดใหม่** `calls` ยังเป็น module variable จากบทเรียน 03 render hook ที่อ่านตัวแปรธรรมดาไม่รู้ว่ามันเปลี่ยนเมื่อไร hook ตัวนับจึงต้องขอให้วาดใหม่เอง:

```ts
on('tool.call', async ($, e, next) => {
  const ran = await next(e)
  calls += 1
  $.ui.status('hello-mod · ' + calls + ' tool calls')
  $.ui.invalidate('ui.render') // a module variable changed: ask for a redraw
  return ran
})
```

บทเรียน 06 จะแทนที่สิ่งนี้ด้วย `$.state` ซึ่งวาดตัวที่อ่านค่าใหม่ให้เอง

✓ Checkpoint: คุณอธิบายได้ว่าจะเกิดอะไรกับ band ของ Mod Gym ถ้า hook ของคุณคืน tree โดยไม่เรียก `next(e)`

## 5 · pane ที่เปิดจากปุ่ม

ปุ่ม *Stats* ด้านบนเรียก `$.ui.open({ id: PANE, title: 'hello-mod stats' })` การเปิดเป็นแค่การจองพื้นที่ render hook ที่ match กับ `requestId` นั้นคือตัววาด:

```tsx
const PANE = 'hello-stats'

// L05 · a pane: opened by $.ui.open, drawn by this hook
on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
  const { Box, Text, Button, Markdown } = $.ui.resolve(e)
  return (
    <Box flexDirection="column" rowGap={1} width={e.props.bodyColumns}>
      <Text bold>hello-mod stats</Text>
      <Markdown key="about" text={'**' + calls + '** tool calls this session.\n\n- the band above the prompt\n- this pane\n- the status line'} />
      <Button key="close" hotkey="q" role="dismiss" onPress={() => $.ui.close({ id: PANE })}>
        Close
      </Button>
    </Box>
  )
})
```

ข้อสังเกตจาก type:

- pane ที่เปิดจากสิ่งที่ผู้ใช้ทำ (กดปุ่ม, สั่ง command) จะได้ที่นั่งที่ความกว้างเท่าไรก็ได้ ส่วน pane ที่เปิดเองโดยไม่มีใครขอ จาก `session.start` หรือ timer จะได้ที่นั่งเมื่อเทอร์มินัลกว้างตั้งแต่ 144 คอลัมน์เท่านั้น ดังนั้นให้เปิด pane จากการกดปุ่มหรือ command
- `focus`, `closeOnEscape` และ `holdToasts` รับค่าได้แค่ `true` การใส่ `focus: true, closeOnEscape: true` ทำให้ pane รับปุ่มคีย์บอร์ด และ **Esc** จะปิดมัน Mod Gym เปิด panel ของมันแบบนี้
- `hotkey="q"` กดปุ่มนั้นขณะที่ pane ถือคีย์บอร์ดอยู่ `role="dismiss"` ระบุว่าเป็นปุ่มปิด: เทอร์มินัลวาดเป็นปุ่มธรรมดา desktop วาดเป็นเครื่องหมายปิดแบบ native
- hook นี้ไม่เรียก `next`: pane เป็นของคุณคนเดียว จึงไม่มีอะไรด้านล่างให้เก็บไว้

validate ทั้ง mod:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (บันทึกบน Mac เครื่องนี้; ย่อ path)

```
Validating hooks: …/hello-mod/hooks/hooks.json

  ❯ ./register.tsx hooks: ui.render{component=Spinner}, tool.call{tool=Bash}, tool.call, ui.render{component=AbovePrompt}, ui.render{component=Pane, requestId=hello-stats}
  ❯ ./register.tsx gating hook with .catch: tool.call{tool=Bash}
  ❯ ./register.tsx gating hook without .catch: tool.call
  ❯ ./register.tsx calls: $.ui.ask, $.ui.close, $.ui.invalidate, $.ui.open, $.ui.resolve, $.ui.status, $.ui.toast

✔ Validation passed with warnings
```

(warning หนึ่งตัวคือ `author: No author information provided` เพิ่ม `author` ใน `plugin.json` ได้ถ้าต้องการ)

โหลดในเทอร์มินัลที่สองแล้วลองใช้: `claude --plugin-dir ~/mod-gym-lab/hello-mod` band จะโผล่ใต้แถวอื่น ๆ ของ prompt **Say hi** แสดง toast **Stats** เปิด pane **q** หรือ **Esc** ปิดมัน

✓ Checkpoint: validate แสดง `ui.render{component=AbovePrompt}` และ `ui.render{component=Pane, requestId=hello-stats}`

## 6 · เมื่อ tree ถูกปฏิเสธ

Claude Code ตรวจทุก tree กับตาราง element ของ surface ก่อนวาด tree ที่ไม่เข้ากัน (element ที่ surface ไม่มี, prop ที่ element ไม่รับ) จะ **ไม่ถูกวาด**: engine วาดของตัวเองแทนและบอกเหตุผล ระหว่างที่ session hot-reload โฟลเดอร์ของคุณ transcript จะแสดงเป็นบรรทัดจาง ๆ หนึ่งบรรทัด นอกนั้นจะไปอยู่ใน debug log (`claude --debug`)

นี่คือตัวอย่างจริง เราใส่ prop ที่ไม่มีอยู่ `sparkle="yes"` ให้ `Box` ของ band แล้ว mount ใน test:

```
hooks: ui.render (AbovePrompt) refused: Box prop "sparkle" is not allowed; the engine drew its own
```

และ `Button` ที่ child เป็น element แทนที่จะเป็น string:

```
hello-mod's ui.render hook was skipped: hello-mod: HooksError: JSX element <Button> needs a label: the label prop, or one string child
```

นิสัยสองอย่างที่จับปัญหาพวกนี้ได้ก่อนโหลด mod:

- **Type-check** เมื่อ Claude Code โหลด mod ของคุณแล้ว `.claude-plugin/types/` จะมี API และ `tsconfig.json` จากนั้น `npx -p typescript tsc -p ~/mod-gym-lab/hello-mod` จะชี้ prop ที่ผิด: `Property 'sparkle' does not exist on type 'IntrinsicAttributes & BoxProps & ElementChildren'`
- **mount ใน test** (หัวข้อถัดไป) tree ที่ถูกปฏิเสธจะทำให้ test fail ด้วยเหตุผลเดียวกัน

✓ Checkpoint: คุณรู้ว่าบรรทัด *refused* ไปโผล่ที่ไหน และ "the engine drew its own" หมายถึงอะไรสำหรับ band ของคุณ

## 7 · ทดสอบการวาดบนสอง surface

`claude plugin test` mount component ผ่าน plugin ของคุณบน surface ที่คุณระบุได้ กดปุ่มด้วย key และหา element ได้ บันทึกไฟล์นี้เป็น `tests/draw.test.ts`:

```ts
import { expect, test } from 'claude-code/testing'

// Mount props a real session fills in; a test spells them out.
const SITE = { scroll: { offset: 0, bodyRows: 40 }, view: {} } as const
const BAND = { component: 'AbovePrompt', props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns: 80, ...SITE } } as const
const PANE = { component: 'Pane', requestId: 'hello-stats', props: { title: 'hello-mod stats', isFocused: true, bodyColumns: 60, placement: 'dock', ...SITE } } as const

test('the band draws on terminal and desktop, and its buttons act', async ($, on) => {
  const toasts: string[] = []
  const opened: string[] = []
  on('ui.toast', ($, e) => { toasts.push(e.text); return { value: undefined } })
  on('ui.open', ($, e) => { opened.push(e.id); return { value: { isPlaced: true } } })
  // Beneath the band stands Claude Code's own (empty) band: draw it as an empty Box.
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Box({}))

  for (const surface of ['terminal', 'desktop'] as const) {
    const band = await $.ui.mount({ plugin: 'hello-mod', surface, ...BAND })
    expect(await band.find({ type: 'Text', text: 'hello-mod' })).toBeDefined()
    await band.press({ key: 'hi' })
    await band.press({ key: 'stats' })
    await band.unmount()

    const pane = await $.ui.mount({ plugin: 'hello-mod', surface, ...PANE })
    expect(await pane.find({ type: 'Text', text: 'hello-mod stats' })).toBeDefined()
    expect(await pane.find({ key: 'close' })).toBeDefined()
    await pane.unmount()
  }
  expect(toasts).toEqual(['Hi from hello-mod!', 'Hi from hello-mod!'])
  expect(opened).toEqual(['hello-stats', 'hello-stats'])
})
```

สิ่งที่ควรสังเกต:

- hook ที่ test เพิ่มด้วย `on(...)` อยู่ **ใต้** plugin ของคุณและทำหน้าที่แทน Claude Code มันรับ `($, e)` เหมือน hook ทั่วไป stub ของ `ui.toast` และ `ui.open` บันทึกสิ่งที่ปุ่มของคุณขอ
- เพราะ band ของคุณเรียก `next(e)` จึงต้องมีอะไรสักอย่างตอบอยู่ด้านล่าง ถ้าไม่มี stub `ui.render` การ mount จะ fail ด้วย `no implementation for ui.render`
- การวนลูปบน `['terminal', 'desktop']` พิสูจน์ว่า band ไม่ได้ผูกกับ surface เดียว

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/hello-mod
```

**Expected output** (บันทึกบน Mac เครื่องนี้)

```
tests/draw.test.ts:
(pass) the band draws on terminal and desktop, and its buttons act [27.81ms]

 1 pass
 0 fail
Ran 1 test across 1 file. [0.15s]
```

✓ Checkpoint: test หนึ่งตัวผ่าน และมันรันบนทั้งสอง surface

## ใน panel ของ Mod Gym

พิมพ์ `/mod-gym 5` เพื่อข้ามมาบทนี้ แล้ว `/mod-gym` เพื่อเปิด panel และกด **c** ตัวตรวจทั้งสี่คือ:

| ตัวตรวจ | ผ่านเมื่อ |
|---|---|
| hooks.json points at the JSX module | `hooks/hooks.json` ระบุ `register.tsx` |
| draws with ui.render | `register.tsx` มี `on('ui.render', { component: 'AbovePrompt' …` หรือ `'Pane'` |
| gets elements from $.ui.resolve(e) | `register.tsx` เรียก `$.ui.resolve(` |
| validate passes | `claude plugin validate` exit 0 |

กด **a** เพื่อใส่คำถามของบทนี้ลงใน prompt ให้ Claude และกด **2** เพื่อดู Events tape ขณะลองใช้ band ของคุณ

## ลองทำเอง — band ที่พอดีกับเทอร์มินัลแคบ

ที่ 40 คอลัมน์ แถวของ band คุณยาวเกินไป แก้ band ให้เมื่อ `e.props.bodyColumns` น้อยกว่า 50 จะเหลือแค่ชื่อกับปุ่ม **Stats** แล้วเพิ่ม test ที่ mount band ที่ 100 และ 40 คอลัมน์บนทั้งสอง surface

ตัวตรวจสำหรับตัวเอง: ที่ 100 คอลัมน์ `find({ key: 'hi' })` ต้องมีค่า ที่ 40 ต้องเป็น `undefined` และ `find({ key: 'stats' })` ต้องมีค่า

<details><summary>เฉลย</summary>

ใน band hook ตัดสินใจครั้งเดียว แล้วตัดสิ่งที่ไม่พอดีออก:

```tsx
const isNarrow = e.props.bodyColumns < 50
const below = await next(e)
return (
  <Box flexDirection="column" width={e.props.bodyColumns}>
    {below}
    <Box flexDirection="row" columnGap={1}>
      <Text color="success">hello-mod</Text>
      {!isNarrow && <Text dimColor>{calls} tool calls</Text>}
      {!isNarrow && (
        <Button key="hi" onPress={() => $.ui.toast('Hi from hello-mod!')}>
          Say hi
        </Button>
      )}
      <Button key="stats" variant="primary" onPress={() => void $.ui.open({ id: PANE, title: 'hello-mod stats' })}>
        Stats
      </Button>
    </Box>
  </Box>
)
```

`tests/narrow.test.ts`:

```ts
import { expect, test } from 'claude-code/testing'

const SITE = { scroll: { offset: 0, bodyRows: 40 }, view: {} } as const
const band = (bodyColumns: number) =>
  ({ component: 'AbovePrompt', props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns, ...SITE } }) as const

test('a narrow band keeps only the name and Stats', async ($, on) => {
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Box({}))
  for (const surface of ['terminal', 'desktop'] as const) {
    const wide = await $.ui.mount({ plugin: 'hello-mod', surface, ...band(100) })
    expect(await wide.find({ key: 'hi' })).toBeDefined()
    await wide.unmount()

    const narrow = await $.ui.mount({ plugin: 'hello-mod', surface, ...band(40) })
    expect(await narrow.find({ key: 'hi' })).toBeUndefined()
    expect(await narrow.find({ key: 'stats' })).toBeDefined()
    await narrow.unmount()
  }
})
```

**Expected output** (บันทึกบน Mac เครื่องนี้)

```
tests/narrow.test.ts:
(pass) a narrow band keeps only the name and Stats [27.21ms]

 2 pass
 0 fail
Ran 2 tests across 2 files. [0.17s]
```

</details>

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `does not parse: Expected ">" but found …` | มี JSX ในไฟล์ `.ts` | เปลี่ยนชื่อเป็น `.tsx` และแก้ `hooks.json` |
| `stale: …/register.ts: no such file` | เปลี่ยนชื่อไฟล์แล้วแต่ไม่ได้แก้ `hooks.json` | `{ "modules": ["./register.tsx"] }` |
| band ของ Mod Gym (หรือของ Clean View) หายไป | band ของคุณคืน tree โดยไม่เรียก `next(e)` | `const below = await next(e)` แล้วใส่ `{below}` ในคอลัมน์ของคุณ |
| `refused: Box prop "…" is not allowed` | prop ที่ element ไม่รับ | ดู prop ของ element ใน type; type-check ด้วย `tsc` |
| `<Button> needs a label` | Button ที่มี child เป็น element | ให้ child เป็น string หนึ่งตัว หรือใส่ prop `label` |
| ตัวเลขใน band ไม่ขยับ | มันอ่าน module variable และไม่มีใครขอให้วาดใหม่ | `$.ui.invalidate('ui.render')` หรือใช้ `$.state` (บทเรียน 06) |
| test fail ด้วย `no implementation for ui.render` | hook ของคุณเรียก `next(e)` แต่ไม่มีอะไรตอบอยู่ด้านล่าง | stub `ui.render` สำหรับ component นั้น เหมือนในหัวข้อ 7 |
| pane ที่เปิดจาก `session.start` ไม่เคยโผล่ | pane ที่ไม่มีใครขอได้ที่นั่งเมื่อกว้างตั้งแต่ 144 คอลัมน์เท่านั้น | เปิดจากการกดปุ่มหรือ command |

## ต่อไป

[Lesson 06 — State that survives a reload](../06_state_and_store/TUTORIAL.th.md): ย้าย `calls` ออกจาก module variable ไปไว้ใน `$.state` เพื่อให้ band วาดใหม่เองและเก็บตัวเลขไว้ได้เมื่อคุณบันทึกไฟล์
