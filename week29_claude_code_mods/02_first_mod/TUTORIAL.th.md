# ▶ Mod Gym Lesson 02 — mod ตัวแรกของคุณ: สามไฟล์

> ส่วนหนึ่งของ Week 29 · Claude Code mods คุณจะได้เรียนวิธีเปลี่ยนหน้าตาและพฤติกรรมของตัว Claude Code เอง จากในเทอร์มินัลที่คุณเขียนโค้ดอยู่ทุกวัน tutorial คือฉบับเต็ม ส่วน **🏋 Mod Gym** (`/mod-gym`) คือฉบับย่อภายใน Claude Code พร้อมตัวตรวจ lab mod ของคุณ ตลอดบทเรียนคุณจะค่อย ๆ สร้าง mod หนึ่งตัวชื่อ `hello-mod` ในโฟลเดอร์แล็บของคุณ (ค่าเริ่มต้น `~/mod-gym-lab`) ฉบับที่เสร็จแล้วอยู่ที่ `week29/reference/hello-mod`

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เขียน mod จริงที่เล็กที่สุด: สามไฟล์ ไม่มีขั้น build ไม่มี Node ไม่มี bundler
- เขียนทับส่วนหนึ่งของหน้าจอ Claude Code เอง: เติมข้อความของคุณต่อท้าย spinner
- validate mod, โหลดด้วย `claude --plugin-dir` แล้วแก้มันระหว่างที่มันรันอยู่
- หาไฟล์ type ของ API ที่ Claude Code เขียนไว้ข้าง mod ของคุณ และ type-check mod ด้วย `tsc`
- อ่าน error สามแบบจาก validator ที่มือใหม่เจอบ่อยที่สุด

**Time** ~40 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** ใช้แค่แล็ปท็อปเครื่องนี้ (`npx` ของ Node สำหรับ type-check ซึ่งไม่บังคับ)

**แหล่งอ้างอิง:** [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference) · ไฟล์ประกาศ API ที่ Claude Code 2.1.292 เขียนให้ (`.claude-plugin/types/claude-code/index.d.ts`)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Lesson 01 | ติดตั้ง Mod Gym แล้วและ `/mod-gym` เปิดได้ | แผงจะตรวจงานของบทนี้ |
| Claude Code 2.1.287+ | `claude --version` | mod |
| โฟลเดอร์แล็บของคุณ | `/mod-gym status` แสดง `lab …` | ตัวตรวจหา `<lab>/hello-mod` |
| Node (ไม่บังคับ) | `npx --version` | ใช้แค่ type-check ด้วย `tsc` ในหัวข้อ 6 |

ทุกคำสั่งข้างล่างใช้ `LAB=~/mod-gym-lab` ถ้าคุณเปลี่ยน lab ด้วย `/mod-gym lab <path>` ไปแล้ว ให้ใช้ path นั้นแทน

## 1 · สามไฟล์

```text
hello-mod/
  .claude-plugin/plugin.json   the manifest: who the plugin is
  hooks/hooks.json             the "modules" key: this is what makes it a mod
  hooks/register.ts            the hooks module: your code
```

(`plugin.json` คือ manifest บอกว่า plugin นี้คือใคร, `hooks.json` มี key `"modules"` ซึ่งเป็นสิ่งที่ทำให้มันเป็น mod, `register.ts` คือ hooks module หรือโค้ดของคุณ)

**`.claude-plugin/plugin.json`**

```json
{ "name": "hello-mod", "version": "0.1.0", "description": "My first mod" }
```

`name` จะกลายเป็นตัวตนของ plugin ในทุกที่: ใน `/plugin`, ในชื่อ tool (`mcp__hello-mod__…`, Lesson 07) และใน key ของ state (Lesson 06) validator จะปฏิเสธชื่อที่ดูเหมือนเป็นของ Anthropic เอง เช่นชื่อใดก็ตามที่ขึ้นต้นด้วย `claude-`, `anthropic-` หรือ `cc-plugin-`

**`hooks/hooks.json`**

```json
{ "modules": ["./register.ts"] }
```

path เดียว นับจากตำแหน่งของ `hooks.json` ไฟล์เดียวกันนี้ใส่ settings hook แบบคลาสสิกไว้ใต้ key `hooks` ได้ด้วย plugin หนึ่งตัวเป็นได้ทั้งสองแบบ

**`hooks/register.ts`**

```ts
import type { Register } from 'claude-code'

export const register: Register = on => {
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )
}
```

ทีละบรรทัด:

- `import type { Register } from 'claude-code'` มีไว้ให้ type-checker เท่านั้น ตอนรันจริง import นี้ว่างเปล่า
- `register` รับ `on` มา การเรียก `on(event, matcher?, hook)` แต่ละครั้งจะเพิ่ม hook หนึ่งตัว
- `'ui.render'` เกิดทุกครั้งที่ Claude Code วาด component หนึ่งชิ้น matcher `{ component: 'Spinner' }` กรองให้เหลือแค่ spinner ซึ่งคือแถว "Thinking…" ที่คุณเห็นระหว่างที่ Claude ทำงาน
- props ของ spinner คือ `word`, `message`, `suffix` และ `mode` (`'requesting' | 'responding' | 'thinking' | 'tool-input' | 'tool-use'`) เราส่งสำเนาของ event ที่มี `suffix` ใหม่ให้ `next` นี่คือ pattern **rewrite**: Claude Code ยังวาด spinner ของตัวเองเหมือนเดิม แต่จาก props ที่เราเปลี่ยนแล้ว

สร้างไฟล์ด้วยมือ หรือให้ Mod Gym พิมพ์ให้: เปิด `/mod-gym 2` แล้วกด **f** (**Copy starter**) มันจะคัดลอก script นี้สำหรับโฟลเดอร์แล็บของคุณ แล้ววางลงใน shell

```bash
# on: laptop
LAB=~/mod-gym-lab
mkdir -p "$LAB/hello-mod/.claude-plugin" "$LAB/hello-mod/hooks"
cat > "$LAB/hello-mod/.claude-plugin/plugin.json" <<'EOF'
{ "name": "hello-mod", "version": "0.1.0", "description": "My first mod" }
EOF
cat > "$LAB/hello-mod/hooks/hooks.json" <<'EOF'
{ "modules": ["./register.ts"] }
EOF
cat > "$LAB/hello-mod/hooks/register.ts" <<'EOF'
import type { Register } from 'claude-code'

export const register: Register = on => {
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )
}
EOF
claude plugin validate "$LAB/hello-mod"
```

## 2 · Validate

**Expected output** ของบรรทัดสุดท้าย (บันทึกจริงบน Mac เครื่องนี้ path ของคุณจะต่างไป)

```
Validating plugin manifest: …/hello-mod/.claude-plugin/plugin.json

⚠ Found 1 warning:

  ❯ author: No author information provided. Consider adding author details for plugin attribution

Validating hooks: …/hello-mod/hooks/hooks.json

  ❯ ./register.ts hooks: ui.render{component=Spinner}
  ❯ ./register.ts calls: nothing on $

✔ Validation passed with warnings
```

บรรทัด `hooks:` คือสิ่งที่ engine จะลงทะเบียน `calls: nothing on $` คือการ audit จาก Lesson 01: mod นี้ไม่เข้าถึงอะไรนอกตัวเองเลย คำเตือนเรื่อง author ไม่บังคับ ใส่ `"author": { "name": "…" }` ถ้าอยากให้หายไป

✓ Checkpoint: `✔ Validation passed` (จะมีหรือไม่มี warning ก็ได้)

## 3 · สามความผิดพลาดที่ validator จับได้

validator อ่านซอร์สของคุณแบบเดียวกับที่ engine จะอ่าน มันต้องการชื่อ event เป็น **string literal** เพราะมันอ่านโดยไม่รันโค้ดของคุณ นี่คือสิ่งที่มันพูดกับ `hello-mod` สามสำเนาที่ถูกทำให้พัง (บันทึกจริงบน Mac เครื่องนี้ ตัดบางส่วน):

พิมพ์ชื่อ event ผิด (`'ui.rendr'`):

```
✘ Found 1 error:
  ❯ modules../register.ts: … compiled line 2 `on("ui.rendr", { component: "Spinner" }, …`: "ui.rendr" is not an event; $ is always spelled $.noun.event(...) at the call site, on is always on("<event>", hook), …
✘ Validation failed
```

ชื่อ event อยู่ในตัวแปร (`const EVENT = 'ui.render'; on(EVENT, …)`):

```
✘ Found 1 error:
  ❯ modules../register.ts: … compiled line 3 `on(EVENT, { component: "Spinner" }, …`: the event name passed to on() is not a string literal; …
✘ Validation failed
```

`hooks.json` ใช้ `"module"` แทน `"modules"`:

```
✘ Found 1 error:
  ❯ root: hooks.json must have `hooks` (the hook matchers) or `modules` (hooks modules), or both
✘ Validation failed
```

ที่ถูกปฏิเสธตอนโหลดด้วย: มี `import()` ที่ใดก็ได้ใน module hooks module เป็น ES module ที่ import ไฟล์อื่นของ plugin ได้ด้วย `import` declaration เท่านั้น

## 4 · โหลดแล้วดู spinner

Mod Gym คือ plugin ที่คุณติดตั้งไว้ ระหว่างพัฒนา mod ของคุณเอง ให้โหลดจากโฟลเดอร์ของมันแทน สำหรับหนึ่ง session:

```bash
# on: laptop (a second terminal)
claude --plugin-dir ~/mod-gym-lab/hello-mod
```

ถาม Claude อะไรก็ได้ที่ใช้เวลาสักสองสามวินาที ("list the files here and explain them") แถว spinner จะลงท้ายด้วย `· lifting…`

ทดสอบเร็ว ๆ แบบไม่ interactive ก็ได้ mod จะโหลด และ Claude ตอบตามปกติ:

```bash
# on: laptop
claude -p --plugin-dir ~/mod-gym-lab/hello-mod --model haiku "Reply with the single word ok"
```

**Expected output** (บันทึกจริงบน Mac เครื่องนี้)

```
ok
```

ถ้า module โหลดไม่สำเร็จใน run แบบ `-p` Claude Code จะบอกชื่อ plugin และเหตุผลหนึ่งครั้งทาง stderr

**Hot reload** ใน session แบบ interactive ที่เริ่มด้วย `--plugin-dir` โฟลเดอร์จะถูกเฝ้าดู เมื่อคุณบันทึก `register.ts` module จะ reload: `register` รันใหม่ในสภาพแวดล้อมใหม่ และ transcript พิมพ์บรรทัดสีจางที่ระบุชื่อ plugin ลองเปลี่ยน suffix เป็น `' · deadlifting…'` บันทึก ส่ง prompt อีกครั้ง แล้วดูมันเปลี่ยนโดยไม่ต้อง restart มีสองอย่างที่รีเซ็ตทุกครั้งที่ reload คือตัวแปรของ module เองและ timer ของมัน (Lesson 06 จะแสดงว่าค่าที่ต้องอยู่รอดควรเก็บไว้ที่ไหน)

✓ Checkpoint: คุณเห็น suffix ของคุณบน spinner

## 5 · type อยู่ที่ไหน

ครั้งแรกที่ Claude Code โหลด mod ของคุณจากโฟลเดอร์ที่คุณเป็นเจ้าของ มันจะเขียนไฟล์ประกาศ API ไว้ข้าง ๆ พร้อม `tsconfig.json` ที่ชี้ไปหาไฟล์เหล่านั้น:

```bash
# on: laptop
find ~/mod-gym-lab/hello-mod -type f | sort
```

**Expected output** (บันทึกจริงบน Mac เครื่องนี้ หลังรัน `claude -p --plugin-dir` หนึ่งครั้ง ย่อ path)

```
hello-mod/.claude-plugin/plugin.json
hello-mod/.claude-plugin/types/.gitignore
hello-mod/.claude-plugin/types/claude-code-mcp/index.d.ts
hello-mod/.claude-plugin/types/claude-code-tools/index.d.ts
hello-mod/.claude-plugin/types/claude-code/index.d.ts
hello-mod/.claude-plugin/types/tsconfig.json
hello-mod/hooks/hooks.json
hello-mod/hooks/register.ts
hello-mod/tsconfig.json
```

| ไฟล์ | คืออะไร |
|---|---|
| `types/claude-code/index.d.ts` | mods API ทั้งหมดของ Claude Code build **ของคุณ** ทุก event และทุกเมธอดของ `$` พร้อมคำอธิบายและตัวอย่าง บรรทัดแรกระบุเวอร์ชัน (`// Written by Claude Code 2.1.292.`) เมื่อหน้าเอกสารกับไฟล์นี้ขัดกัน ให้เชื่อไฟล์นี้ |
| `types/claude-code-tools/index.d.ts` | tool ในตัวบนเครื่องนี้ ทำให้ `e.tool === 'Bash'` แคบ type ของ `e` ลงเป็น input ของ Bash |
| `types/claude-code-mcp/index.d.ts` | MCP tool ที่เชื่อมต่ออยู่ตอนที่ mod โหลดครั้งล่าสุด |
| `types/.gitignore` | `*`: โฟลเดอร์นี้ถูกสร้างใหม่ทุกครั้ง ไม่ต้อง commit |
| `tsconfig.json` (ราก mod) | `{ "extends": "./.claude-plugin/types/tsconfig.json" }` |

grep หาชื่อที่ต้องการในไฟล์ API แทนการอ่านทั้งไฟล์:

```bash
# on: laptop
grep -n "Spinner: {" -A 25 ~/mod-gym-lab/hello-mod/.claude-plugin/types/claude-code/index.d.ts
```

## 6 · Type-check

```bash
# on: laptop
npx -y -p typescript@5.6 tsc -p ~/mod-gym-lab/hello-mod
echo "tsc exit $?"
```

**Expected output** (บันทึกจริงบน Mac เครื่องนี้)

```
tsc exit 0
```

ตอนนี้ลองทำผิดแบบที่ validator มองไม่เห็น เปลี่ยน suffix เป็นตัวเลข `suffix: 42` แล้วรัน `tsc` อีกครั้ง:

**Expected output** (บันทึกจริงบน Mac เครื่องนี้ ตัดบางส่วน)

```
hello-mod/hooks/register.ts(5,39): error TS2769: No overload matches this call.
  …
    Type 'number' is not assignable to type 'string'.
tsc exit 2
```

ใส่ string กลับคืน validate กับ `tsc` จับคนละเรื่องกัน: validate ตรวจสิ่งที่ engine จะปฏิเสธไม่ยอมโหลด ส่วน `tsc` ตรวจว่าค่าของคุณมีรูปแบบตามที่ engine คาดไว้

✓ Checkpoint: `tsc exit 0` กับ mod ของคุณ

## ในแผง Mod Gym

รัน `/mod-gym 2` แล้วกด **c** Lesson 02 ตรวจสี่อย่างใน `<lab>/hello-mod`

**Expected output** (`/mod-gym check` บันทึกจริงบน Mac เครื่องนี้ โดยตรวจกับ mod ต้นแบบ)

```
Lesson 02 · Your first mod: three files · 4/4 checks
✓ plugin.json exists (hello-mod/.claude-plugin/plugin.json)
✓ hooks.json names a module (hello-mod/hooks/hooks.json)
✓ register.ts hooks the Spinner (hello-mod/hooks/register.tsx)
✓ claude plugin validate passes (exit 0)
```

ในแล็บของคุณ บรรทัดที่สามจะระบุ `register.ts` ส่วน mod ต้นแบบไปถึง Lesson 08 แล้ว ไฟล์จึงเป็น `register.tsx` ตัวตรวจยอมรับทั้งสองแบบ

ติดอยู่หรือเปล่า กด **a**: Mod Gym จะใส่คำถามโค้ชลงใน prompt ของคุณ และ Claude จะอ่านบทเรียนกับโฟลเดอร์แล็บของคุณผ่าน tool `gym_status`

## ลองทำเอง — spinner ที่รู้ว่า Claude กำลังทำอะไร

prop `mode` ของ spinner บอกว่าตอนนี้ Claude กำลังทำอะไร ให้ suffix เปลี่ยนตาม: `requesting` → `warming up`, `thinking` → `planning the set`, `responding` → `lifting`, `tool-input` → `loading the bar`, `tool-use` → `spotting` ใช้ hook ตัวเดียว แล้ว validate และ type-check

<details><summary>เฉลย</summary>

```ts
import type { Register } from 'claude-code'

const REPS: Record<string, string> = {
  requesting: 'warming up',
  thinking: 'planning the set',
  responding: 'lifting',
  'tool-input': 'loading the bar',
  'tool-use': 'spotting',
}

export const register: Register = on => {
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · ' + (REPS[e.props.mode] ?? 'lifting') + '…' } }),
  )
}
```

**Expected output** (บันทึกจริงบน Mac เครื่องนี้)

```
  ❯ ./register.ts hooks: ui.render{component=Spinner}
  ❯ ./register.ts calls: nothing on $
✔ Validation passed with warnings
tsc exit 0
```

`e.props.mode` มี type กำกับ `tsc` จึงจะฟ้องถ้าสะกดชื่อ prop ผิด ส่วน fallback `?? 'lifting'` ทำให้ spinner ยังดูสมเหตุสมผลถ้า Claude Code รุ่นหลังเพิ่ม mode ใหม่

</details>

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `hooks.json must have hooks … or modules` | สะกด key เป็น `module` หรือไฟล์ว่าง | `{ "modules": ["./register.ts"] }` |
| `… is not an event` | พิมพ์ชื่อ event ผิด | grep ไฟล์ type หา `'ui.render'` แล้วคัดลอกชื่อมา |
| `… is not a string literal` | ชื่อ event หรือ matcher อยู่ในตัวแปร | เขียนตรง ๆ: `on('ui.render', { component: 'Spinner' }, …)` |
| ไม่มี suffix บน spinner | session ไม่ได้เริ่มด้วย `--plugin-dir` หรือ module โหลดไม่สำเร็จ | หาบรรทัดสีจางที่ระบุ `hello-mod` ใน transcript รัน `claude --debug --plugin-dir …` |
| ไม่มีโฟลเดอร์ `.claude-plugin/types` | mod ยังไม่เคยถูกโหลด | รัน `claude -p --plugin-dir <dir> --model haiku "Reply with the single word ok"` หนึ่งครั้ง |
| `tsc` หา `claude-code` ไม่เจอ | ยังไม่มี `tsconfig.json` (สาเหตุเดียวกัน) | โหลด mod หนึ่งครั้ง แล้วรัน `tsc -p <dir>` ใหม่ |

## ต่อไป

[Lesson 03 — hook pipeline: observe, rewrite, answer](../03_hook_pipeline/TUTORIAL.th.md): เพิ่ม hook บนการเรียก tool และบน prompt แล้วดู hook ของคุณเองคู่กับของ Mod Gym ในแท็บ Events
