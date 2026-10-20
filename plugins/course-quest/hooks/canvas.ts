// A tiny true-color character canvas for the terminal's Raster element, and the
// pictures Agentic Quest draws with it: the animated banner, the course metro map,
// a week's flow diagram with a token running along it, and progress bars.
//
// A Raster cell is [codePoint, foreground, background] as little-endian u32s,
// base64 encoded; colors are 0x00RRGGBB, DEFAULT is the terminal's own color.

export const DEFAULT = 0x01000000

export type Canvas = {
  columns: number
  rows: number
  put: (x: number, y: number, ch: string, fg?: number, bg?: number) => void
  text: (x: number, y: number, s: string, fg?: number, bg?: number) => number
  /** the code point drawn at a cell (a space outside the canvas) */
  get: (x: number, y: number) => number
  cells: () => string
}

// Wide (two-cell) and emoji-presentation characters, and zero-width ones, cannot sit in one
// Raster cell: the engine refuses the whole tree. These ranges are skipped.
const NOT_ONE_CELL: Array<[number, number]> = [
  [0x0300, 0x036f], [0x1100, 0x115f], [0x200b, 0x200f], [0x231a, 0x231b], [0x2329, 0x232a],
  [0x23e9, 0x23ec], [0x23f0, 0x23f0], [0x23f3, 0x23f3], [0x25fd, 0x25fe], [0x2614, 0x2615],
  [0x2648, 0x2653], [0x267f, 0x267f], [0x2693, 0x2693], [0x26a1, 0x26a1], [0x26aa, 0x26ab],
  [0x26bd, 0x26be], [0x26c4, 0x26c5], [0x26ce, 0x26ce], [0x26d4, 0x26d4], [0x26ea, 0x26ea],
  [0x26f2, 0x26f3], [0x26f5, 0x26f5], [0x26fa, 0x26fa], [0x26fd, 0x26fd], [0x2705, 0x2705],
  [0x270a, 0x270b], [0x2728, 0x2728], [0x274c, 0x274c], [0x274e, 0x274e], [0x2753, 0x2755],
  [0x2757, 0x2757], [0x2795, 0x2797], [0x27b0, 0x27b0], [0x27bf, 0x27bf], [0x2b1b, 0x2b1c],
  [0x2b50, 0x2b50], [0x2b55, 0x2b55], [0x2e80, 0xa4cf], [0xac00, 0xd7a3], [0xf900, 0xfaff],
  [0xfe00, 0xfe0f], [0xfe30, 0xfe4f], [0xff00, 0xff60], [0xffe0, 0xffe6],
]

export function fitsOneCell(cp: number): boolean {
  if (cp < 0x20 || cp > 0xffff || (cp >= 0x7f && cp < 0xa0)) return false
  return !NOT_ONE_CELL.some(([a, b]) => cp >= a && cp <= b)
}

export function canvas(columns: number, rows: number, bg = DEFAULT): Canvas {
  const words = new Uint32Array(columns * rows * 3)
  for (let i = 0; i < columns * rows; i++) {
    words[i * 3] = 0x20
    words[i * 3 + 1] = DEFAULT
    words[i * 3 + 2] = bg
  }
  const put = (x: number, y: number, ch: string, fg = DEFAULT, back?: number) => {
    if (x < 0 || y < 0 || x >= columns || y >= rows) return
    const cp = ch.codePointAt(0) ?? 0x20
    const i = (y * columns + x) * 3
    words[i] = fitsOneCell(cp) ? cp : 0x20
    words[i + 1] = fg
    if (back !== undefined) words[i + 2] = back
  }
  const text = (x: number, y: number, s: string, fg = DEFAULT, back?: number) => {
    let n = 0
    for (const ch of s) {
      if (!fitsOneCell(ch.codePointAt(0) ?? 0)) continue // only one-cell characters fit
      put(x + n, y, ch, fg, back)
      n += 1
    }
    return n
  }
  const cells = () => {
    const bytes = new Uint8Array(words.buffer)
    let bin = ''
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
    }
    return btoa(bin)
  }
  const get = (x: number, y: number) => (x < 0 || y < 0 || x >= columns || y >= rows ? 0x20 : (words[(y * columns + x) * 3] ?? 0x20))
  return { columns, rows, put, text, get, cells }
}

// ---------- color ----------

export function rgb(r: number, g: number, b: number): number {
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(v)))
  return (c(r) << 16) | (c(g) << 8) | c(b)
}

export function mix(a: number, b: number, t: number): number {
  const ch = (c: number, s: number) => (c >> s) & 0xff
  return rgb(
    ch(a, 16) + (ch(b, 16) - ch(a, 16)) * t,
    ch(a, 8) + (ch(b, 8) - ch(a, 8)) * t,
    ch(a, 0) + (ch(b, 0) - ch(a, 0)) * t,
  )
}

export function hsl(h: number, s: number, l: number): number {
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => {
    const k = (n + h / 30) % 12
    return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))
  }
  return rgb(f(0), f(8), f(4))
}

export function hex(color: number): string {
  return '#' + color.toString(16).padStart(6, '0')
}

/** One color per phase; 7 is the robotics track. */
export const PHASE_COLOR: Record<number, number> = {
  1: 0x4fc3f7,
  2: 0x81c784,
  3: 0xffb74d,
  4: 0xba68c8,
  5: 0xf06292,
  6: 0xffd54f,
  7: 0x4db6ac,
}
export const GREY = 0x5c6370
export const DIM = 0x3a3f4b
export const WHITE = 0xf5f5f5
export const INK = 0x14161c

// ---------- the banner: a friendly unit card with Bolt, the course mascot ----------
//
// Modelled on learning apps' unit cards: one calm, saturated color per phase, a darker
// "lip" under the card, a mascot that talks to you in a speech line, and small chips.

/** Card colors: saturated enough that white text reads on every one. */
export const CARD_COLOR: Record<number, number> = {
  1: 0x1e88e5,
  2: 0x2e9d48,
  3: 0xe8650a,
  4: 0x8e24aa,
  5: 0xd81b60,
  6: 0xb7791f,
  7: 0x00897b,
}

// Bolt, 14 × 11 pixels. o outline · w body · v visor · e eye · E eye highlight ·
// m smile · c cheek · b antenna bulb · h hand (the right one waves) · . card
const BOLT = [
  '......b.......',
  '......o.......',
  '..oooooooooo..',
  '.owwwwwwwwwwo.',
  'howvvvvvvvvwo.',
  '.owvEevvEevwo.',
  '.owveevveevwo.',
  '.owvvvmmvvvwo.',
  '.owcwwwwwwcwo.',
  '..oooooooooo..',
  '...oo....oo...',
]
const WAVE_UP = { x: 13, y: 3 }
const WAVE_DOWN = { x: 13, y: 5 }

function hash(n: number): number {
  const x = Math.sin(n * 127.1) * 43758.5453
  return x - Math.floor(x)
}

export type BannerOptions = {
  frame: number
  phase: number
  /** small caps line, e.g. "WEEK 24 · OPEN STACK" */
  label: string
  title: string
  /** what Bolt says; typed out one letter per frame */
  speech: string
  /** the frame the current speech started, for the typewriter */
  speechFrom: number
  level: number
  xp: number
  /** 0..1 of the way to the next level */
  fraction: number
  streak: number
  /** frames left of a celebration (confetti and a jump); 0 for none */
  party: number
}

export const BANNER_ROWS = 6

export function banner(columns: number, o: BannerOptions): Canvas {
  const rows = BANNER_ROWS
  const c = canvas(columns, rows)
  const card = CARD_COLOR[o.phase] ?? 0x1e88e5
  const lip = mix(card, 0x000000, 0.35)
  const soft = mix(WHITE, card, 0.3)
  const px = rows * 2

  // Bolt's pixels: he bobs gently, jumps when you score, blinks, and waves now and then
  const jump = o.party > 0 && Math.sin(o.party * 0.6) > 0 ? 1 : 0
  const bob = o.party > 0 ? 0 : (o.frame >> 3) % 2
  const top = bob - jump
  const left = 2
  const isBlink = o.frame % 46 < 2
  const isWave = o.party > 0 || o.frame % 120 < 24
  const wave = isWave && (o.frame >> 2) % 2 ? WAVE_UP : WAVE_DOWN
  const bulb = mix(0xffeb3b, 0xff5252, 0.5 + 0.5 * Math.sin(o.frame * 0.4))
  const bolt = (x: number, y: number): number | undefined => {
    const lx = x - left
    const ly = y - top
    if (lx === wave.x && ly === wave.y) return 0xf7f7fb
    const ch = BOLT[ly]?.[lx]
    if (isBlink && (ch === 'e' || ch === 'E')) return ly === 6 ? 0x7cf7ff : 0x1b2340
    switch (ch) {
      case 'o':
        return 0x1b2340
      case 'w':
        return 0xf7f7fb
      case 's':
        return 0xd6dae6
      case 'v':
        return 0x1b2340
      case 'e':
        return 0x7cf7ff
      case 'E':
        return 0xffffff
      case 'm':
        return 0x7cf7ff
      case 'c':
        return 0xff9eb1
      case 'b':
        return bulb
      case 'h':
        return 0xf7f7fb
      default:
        return undefined
    }
  }
  const bg = (y: number) => (y >= px - 1 ? lip : card)
  const pixel = (x: number, y: number) => bolt(x, y) ?? bg(y)

  // the card, with rounded corners, pixel art across the mascot's columns
  const artRight = left + 15
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      if (x <= artRight) c.put(x, y, '▀', pixel(x, y * 2), pixel(x, y * 2 + 1))
      else c.put(x, y, y === rows - 1 ? '▀' : ' ', card, y === rows - 1 ? lip : card)
    }
  }
  c.put(0, 0, '▗', card)
  c.put(columns - 1, 0, '▖', card)
  c.put(0, rows - 1, '▝', card)
  c.put(columns - 1, rows - 1, '▘', card)

  // the text column
  const tx = artRight + 2
  const room = columns - tx - 2
  if (room >= 8) {
    const fit = (s: string, n: number) => ([...s].length > n ? [...s].slice(0, Math.max(0, n - 1)).join('') + '…' : s)
    // chips on the right of the label row
    const chips = `★ Lv ${o.level}  ✦ ${o.xp} xp` + (o.streak > 1 ? `  ↑ ${o.streak} days` : '')
    const showChips = room >= [...chips].length + 12
    const labelRoom = showChips ? room - [...chips].length - 2 : room
    c.text(tx, 1, fit(o.label, labelRoom), soft, card)
    if (showChips) c.text(columns - 2 - [...chips].length, 1, chips, 0xfff3c4, card)
    c.text(tx, 2, fit(o.title, room), WHITE, card)
    // Bolt speaks: a typewriter, then the caret blinks
    const shown = Math.max(0, o.frame - o.speechFrom)
    const words = [...o.speech]
    const typed = words.slice(0, Math.min(words.length, shown)).join('')
    const caret = shown < words.length || (o.frame >> 2) % 2 ? '▏' : ' '
    c.put(tx, 3, '◀', soft, card)
    c.text(tx + 2, 3, fit(typed, room - 3) + caret, 0xfff3c4, card)
    // the level bar
    const label = `${Math.round(o.fraction * 100)}% to Lv ${o.level + 1}`
    const barWidth = Math.max(6, room - [...label].length - 1)
    for (let i = 0; i < barWidth; i++) {
      const filled = i < Math.round(o.fraction * barWidth)
      const shine = filled && (i - (o.frame >> 1)) % 14 === 0
      c.put(tx + i, 4, filled ? '━' : '─', shine ? WHITE : filled ? 0xffd54f : mix(card, 0x000000, 0.25), card)
    }
    if (room - barWidth - 1 >= [...label].length) c.text(tx + barWidth + 1, 4, label, soft, card)
  }

  // a few calm sparkles in the far corner, twinkling slowly
  for (let i = 0; i < 3; i++) {
    const sx = columns - 3 - i * 5
    const sy = i % 2 === 0 ? 4 : 0
    if (sx <= tx + 4 || sy === 4 || c.get(sx, sy) !== 0x20) continue
    const on = Math.sin(o.frame * 0.15 + i * 2) > 0.3
    if (on) c.put(sx, sy, '✦', mix(card, WHITE, 0.6), card)
  }

  // confetti when you score
  if (o.party > 0) {
    const colors = [0xffd54f, 0xff8a80, 0x80d8ff, 0xb9f6ca, 0xffffff]
    for (let i = 0; i < Math.min(28, columns >> 1); i++) {
      const cx = Math.floor(hash(i * 3.1) * columns)
      const cy = Math.floor((hash(i * 5.7) * rows + (30 - o.party) * 0.35) % (rows - 1))
      if (cx <= artRight || [cx - 1, cx, cx + 1].some(x => c.get(x, cy) !== 0x20)) continue
      c.put(cx, cy, i % 3 === 0 ? '✦' : i % 3 === 1 ? '•' : '·', colors[i % colors.length] ?? WHITE, card)
    }
  }
  return c
}

// ---------- the course map: one metro line per phase ----------

export type Station = { id: string; state: 'done' | 'started' | 'new' }
export type MapLine = { phase: number; label: string; stations: Station[] }

export function metro(columns: number, lines: MapLine[], current: string, frame: number): Canvas {
  const c = canvas(columns, Math.max(1, lines.length * 2))
  const labelWidth = Math.min(20, Math.max(8, ...lines.map(l => [...l.label].length)) + 1)
  lines.forEach((line, li) => {
    const y = li * 2
    const color = PHASE_COLOR[line.phase] ?? WHITE
    c.text(0, y, [...line.label].slice(0, labelWidth - 1).join(''), color)
    const startX = labelWidth + 1
    const n = line.stations.length
    const room = columns - startX - 3
    const gap = n > 1 ? Math.max(3, Math.min(8, Math.floor(room / (n - 1)))) : 0
    const end = Math.min(columns - 1, startX + gap * (n - 1) + 2)
    for (let x = startX; x <= end; x++) c.put(x, y, '━', mix(color, INK, 0.45))
    if (line.stations.some(s => s.id === current) && end > startX) {
      c.put(startX + (frame % (end - startX + 1)), y, '▶', mix(color, WHITE, 0.4))
    }
    line.stations.forEach((s, si) => {
      const x = startX + si * gap
      if (x + 1 >= columns) return
      const isCurrent = s.id === current
      const glow = 0.5 + 0.5 * Math.sin(frame * 0.6)
      const dot = s.state === 'done' ? '●' : s.state === 'started' ? '◐' : '○'
      const fg = isCurrent ? mix(color, WHITE, glow) : s.state === 'new' ? GREY : color
      c.put(x, y, isCurrent ? ((frame >> 2) % 2 ? '◉' : '●') : dot, fg)
      const tagFg = isCurrent ? WHITE : s.state === 'new' ? GREY : mix(color, WHITE, 0.3)
      c.text(x, y + 1, s.id, tagFg, isCurrent ? mix(color, INK, 0.5) : undefined)
    })
  })
  return c
}

// ---------- a week's flow: boxes, arrows and a token that runs along them ----------

export function flow(columns: number, nodes: string[], loop: boolean, color: number, frame: number): Canvas {
  const list = nodes.length ? nodes : ['…']
  const longest = Math.max(...list.map(n => [...n].length))
  const fit = (arrow: number) => Math.floor((columns - 1 - arrow * (list.length - 1)) / list.length) - 2
  const arrow = fit(4) >= Math.min(longest, 10) ? 4 : 2
  if (fit(arrow) < Math.min(longest, 9)) return flowDown(columns, list, loop, color, frame)
  const max = Math.max(3, Math.min(14, fit(arrow)))
  const labels = list.map(n => ([...n].length > max ? [...n].slice(0, max - 1).join('') + '…' : n))
  const widths = labels.map(l => [...l].length + 2)
  const xs: number[] = []
  let x = 0
  for (const w of widths) {
    xs.push(x)
    x += w + arrow
  }
  const total = x - arrow
  const c = canvas(Math.max(1, Math.min(columns, total + 1)), loop ? 5 : 3)
  const back = loop ? total : 0
  const pos = frame % Math.max(1, total + back)
  const isLit = (i: number) => pos < total && pos >= (xs[i] ?? 0) && pos < (xs[i] ?? 0) + (widths[i] ?? 0)
  labels.forEach((label, i) => {
    const left = xs[i] ?? 0
    const w = widths[i] ?? 3
    const on = isLit(i)
    box(c, left, 0, w, label, on, color)
    if (i < labels.length - 1) {
      for (let k = 0; k < arrow - 1; k++) c.put(left + w + k, 1, '─', mix(color, INK, 0.35))
      c.put(left + w + arrow - 1, 1, '▶', mix(color, INK, 0.1))
    }
  })
  const lastMid = (xs[xs.length - 1] ?? 0) + Math.floor((widths[widths.length - 1] ?? 2) / 2)
  const firstMid = Math.floor((widths[0] ?? 2) / 2)
  if (loop) {
    const line = mix(color, INK, 0.35)
    c.put(lastMid, 3, '│', line)
    c.put(firstMid, 3, '▲', line)
    c.put(firstMid, 4, '╰', line)
    c.put(lastMid, 4, '╯', line)
    for (let k = firstMid + 1; k < lastMid; k++) c.put(k, 4, '─', line)
    c.text(Math.max(firstMid + 2, Math.floor((firstMid + lastMid) / 2) - 3), 4, ' again ', mix(color, WHITE, 0.2))
  }
  if (pos < total) {
    if (!labels.some((_, i) => isLit(i))) c.put(pos, 1, '◆', 0xffffff)
  } else if (loop) {
    const k = (pos - total) / Math.max(1, back)
    c.put(Math.max(firstMid, lastMid - Math.round(k * (lastMid - firstMid))), 4, '◆', 0xffffff)
  }
  return c
}

function box(c: Canvas, left: number, top: number, w: number, label: string, on: boolean, color: number) {
  const border = on ? mix(color, WHITE, 0.55) : mix(color, INK, 0.25)
  c.put(left, top, '╭', border)
  c.put(left + w - 1, top, '╮', border)
  c.put(left, top + 2, '╰', border)
  c.put(left + w - 1, top + 2, '╯', border)
  for (let k = 1; k < w - 1; k++) {
    c.put(left + k, top, '─', border)
    c.put(left + k, top + 2, '─', border)
  }
  c.put(left, top + 1, '│', border)
  c.put(left + w - 1, top + 1, '│', border)
  c.text(left + 1, top + 1, label, on ? INK : WHITE, on ? mix(color, WHITE, 0.35) : undefined)
}

/** The narrow form: boxes stacked top to bottom, the token falling through them. */
function flowDown(columns: number, list: string[], loop: boolean, color: number, frame: number): Canvas {
  const inner = Math.max(3, Math.min(columns - (loop ? 6 : 2), Math.max(...list.map(n => [...n].length))))
  const labels = list.map(n => ([...n].length > inner ? [...n].slice(0, inner - 1).join('') + '…' : n))
  const step = 4 // a box is 3 rows, then one row of arrow
  const rows = list.length * step - 1
  const c = canvas(Math.min(columns, inner + 2 + (loop ? 4 : 0)), rows)
  const pos = frame % Math.max(1, rows + (loop ? rows : 0))
  const mid = Math.floor((inner + 2) / 2)
  labels.forEach((label, i) => {
    const top = i * step
    box(c, 0, top, inner + 2, label, pos < rows && pos >= top && pos < top + 3, color)
    if (i < labels.length - 1) c.put(mid, top + 3, pos === top + 3 ? '◆' : '▼', pos === top + 3 ? 0xffffff : mix(color, INK, 0.1))
  })
  if (loop) {
    const line = mix(color, INK, 0.35)
    const x = inner + 4
    c.put(inner + 2, 1, '◀', line)
    c.put(inner + 3, 1, '─', line)
    c.put(x, 1, '╮', line)
    c.put(inner + 2, rows - 2, '─', line)
    c.put(inner + 3, rows - 2, '─', line)
    c.put(x, rows - 2, '╯', line)
    for (let y = 2; y < rows - 2; y++) c.put(x, y, '│', line)
    if (pos >= rows) {
      const y = rows - 2 - Math.round(((pos - rows) / rows) * (rows - 3))
      c.put(x, Math.max(1, y), '◆', 0xffffff)
    }
  }
  return c
}

// ---------- bars ----------

const EIGHTHS = ' ▏▎▍▌▋▊▉█'

export function bar(c: Canvas, x: number, y: number, width: number, fraction: number, from: number, to: number, frame = 0) {
  const f = Math.max(0, Math.min(1, fraction))
  const full = f * width
  for (let i = 0; i < width; i++) {
    const t = width > 1 ? i / (width - 1) : 0
    const shimmer = 0.15 * Math.max(0, Math.sin((i - frame * 0.7) * 0.35))
    const color = mix(mix(from, to, t), WHITE, shimmer)
    if (i + 1 <= full) c.put(x + i, y, '█', color)
    else if (i < full) c.put(x + i, y, EIGHTHS[Math.round((full - i) * 8)] ?? ' ', color)
    else c.put(x + i, y, '·', DIM)
  }
}
