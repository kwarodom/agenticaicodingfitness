# Week 23 house style — build spec for a new hub app (Stack Navigator)

Extracted from `week23/01_nemotron_models` (read in full) and `week23/02_nim_microservices`
(view.py/sim.py). Follow this exactly and the new app is indistinguishable from the other 12.

---

## 1. File layout convention (every app is a standalone folder)

```
NN_app_name/
├── README.md              # optional (5 of 12 apps lack it) — but include one
├── requirements.txt       # 3 lines: fastapi>=0.110, uvicorn[standard]>=0.29, openai>=1.30
├── config.py              # COPY FROM 01_nemotron_models AND ADAPT (see §3) — 326 lines
├── dgxsh.py               # COPY VERBATIM — byte-identical (same md5) across all 12 apps
├── tutorial_server.py     # FastAPI control plane, ~330–350 lines (see §2)
├── view.py                # "make it visible" engine — real endpoint (OpenAI SDK) or sim (see §5)
├── sim.py                 # app-specific simulator: installed_models(), tok_s(), stream_generate()
├── demos/
│   ├── step01_….py        # one runnable script per chapter, 4–5 per app
│   └── step0N_….py
└── static/
    └── guide.html         # ONE self-contained file: CSS + HTML + JS, no external libs (see §6)
```

Note: 01 uses `ntsim.py`/`ntview.py` instead of `sim.py`/`view.py` (an outlier). Apps 02–12
use `sim.py`/`view.py` and that is the convention to follow — `view.py`'s docstring says
"Each Week 23 app ships this file plus a `sim.py`".

## 2. tutorial_server.py — FastAPI patterns (mirror 01_nemotron_models/tutorial_server.py)

Header/boilerplate (lines 1–57):
- `#!/usr/bin/env python3` + module docstring describing REAL/SIM dual mode and launch line.
- `from __future__ import annotations`; imports: asyncio, os, shutil, socket, sys, time, pathlib.
- `sys.path.insert(0, str(Path(__file__).resolve().parent))` then `import config  # noqa: E402`.
- `PKG = Path(__file__).resolve().parent`; `ROOT = PKG.parents[1]`;
  `PY = str(ROOT / ".venv" / "bin" / "python")` with fallback to `sys.executable`;
  `DEMOS = PKG / "demos"`.
- Port: `GUIDE_PORT = int(os.environ.get("<PREFIX>_GUIDE_PORT", "<port>"))` — each app has its
  own prefix (NT_, NIM_, DYNAMO_, SKILLS_, AIQ_, CLAW_, GUARD_, RELAY_, ECON_, GYM_,
  FLYWHEEL_, HOTEL_). For Stack Navigator pick e.g. `NAV_GUIDE_PORT` with a default below 8100
  (the team decides; 8099 keeps 8100–8111 intact).
- Port auto-pick: `_port_busy(port)` (socket connect_ex, 0.3 s timeout) +
  `_pick_free_port(preferred, span=40)` scanning preferred..preferred+39.

Content model:
- `STEPS = [ {...}, ... ]` — a module-level list of dicts. Keys:
  `id` ("intro", "step01"…"step0N", "outro", "refs"), `group` (sidebar section label),
  `kind` ("concept" | "run"), `demo` ("step01_xxx.py", run steps only),
  `title` ("Ch N · <name>" / "Appendix · <name>"), `level`
  ("beginner"|"intermediate"|"advanced"|"all levels"), `desc` (long multi-line string;
  intro desc carries "Week 23 · Tutorial NN of 12 · Phase: …", a chapter list, "Why it
  matters", "Where it fits", "How to run"). Always end with an `outro` concept step and a
  `refs` "Appendix · References & real-world applications" concept step under group "Go further".
- `STEP_BY_ID = {s["id"]: s for s in STEPS}`.

App + endpoints (all present in every app):
- `app = FastAPI(title="<App name> — interactive tutorial")`; `_run_lock = asyncio.Lock()`;
  `SELECTED = {"model": config.MODEL}`.
- `GET /` → `FileResponse(PKG / "static" / "guide.html", headers={"Cache-Control": "no-store, max-age=0"})`.
- `GET /api/steps` → `{steps:[public fields], mode, conn, conn_human, model, base_url, models}`;
  when SIM, models come from `sim.installed_models()` (`import sim as dgxsim` inside handler),
  when REAL from `config.list_local_models()`; keeps `SELECTED["model"]` valid.
- `POST /api/select_model` (pydantic `ModelRequest{model:str}`).
- `POST /api/connect` (pydantic `ConnRequest{conn,url,key,auth}`) → calls
  `config.apply_connection(req.model_dump())`, returns `{ok, conn, mode, base_url:
  config.safe_base_url(), endpoint_up, model, models}`.
- `GET /api/source/{step_id}` → `{source: path.read_text(), filename}` for the demo file.
- `POST /api/run` (pydantic `RunRequest{step_id}`) → **StreamingResponse, media_type
  "text/plain"**. Guarded by `_run_lock` ("another demo is already running"). First yield is
  `f"$ {Path(PY).name} demos/{step['demo']}\n\n"`, then `_stream_demo(demo, timeout)`:
  spawns `asyncio.create_subprocess_exec(PY, str(DEMOS/demo), cwd=str(PKG),
  env={**os.environ, "PYTHONUNBUFFERED":"1", "DGX_MODEL": SELECTED["model"]}, stdout=PIPE,
  stderr=STDOUT)`, reads lines with `asyncio.wait_for` against a wall-clock deadline, kills on
  timeout, and ALWAYS terminates the stream with the sentinel line
  `__EXIT__ <returncode> <elapsed_s>\n` (`__EXIT__ 124` on timeout). Timeout:
  `360.0 if config.MODE == "real" else 120.0`.
- `POST /api/cleanup` → removes `PKG/.sandbox` and all `__pycache__`, returns `{"messages":[…]}`.
- DGX console section (verbatim in every app): `import dgxsh`, `_dgx_lock = asyncio.Lock()`,
  `GET /api/dgx/status`, `POST /api/dgx/config` (host/user/port/key), `POST /api/dgx/run`
  streaming one SSH command with a 600 s cap and the same `__EXIT__` sentinel.

`__main__` block:
- `port = _pick_free_port(GUIDE_PORT)`; print a banner list starting
  `["", "  ▣  <App name> — <tagline>"]` with REAL (`✓ REAL endpoint: model @ url`) vs SIM
  (`◈ SIM mode …`) lines, a `⚠ port busy` line naming the real `<PREFIX>_GUIDE_PORT` env var
  (many apps wrongly say DGX_GUIDE_PORT — get this right in the new app), then
  `open  →  http://127.0.0.1:{port}`; finally `uvicorn.run(app, host="127.0.0.1", port=port)`.

## 3. config.py — the connection switch: **copy config.py from 01_nemotron_models and adapt**

Do NOT rewrite it. It is 326 lines and identical in structure across all 12 apps. Copy
`week23/01_nemotron_models/config.py` into the new folder verbatim; the only adaptation
normally needed is the docstring (all 12 currently still say "Week 19" — fix yours) and,
if the app wants different model preferences, the `_PREFERRED` list. What it provides:

- Env contract: `DGX_CONN=local|tunnel|cloud`, `DGX_BASE_URL` (explicit, always wins),
  `DGX_TUNNEL_URL`, `DGX_CLOUD_URL`, `DGX_API_KEY`, `DGX_MODEL`, `DGX_MODE=sim|real|auto`.
- `_resolve_connection()` → `(CONN, BASE_URL, API_KEY)`; infers local/tunnel/cloud from the
  hostname; DEFAULT is `tunnel` to `http://your-spark.your-tailnet.ts.net:11434/v1`, with an
  import-time fallback to local Ollama `http://localhost:11434/v1` if unreachable, then SIM.
- `conn_human()`, `is_sovereign()`, `cost_note()` ("on your DGX · $0.0000" vs "cloud usage billed"),
  `safe_base_url()` (masks URL passwords).
- `apply_connection(dict)` — runtime re-point from the UI: clears/sets env vars (so demo
  subprocesses inherit), auto-appends `/v1`, supports ngrok basic-auth `user:pass`, then
  recomputes globals CONN/BASE_URL/API_KEY/MODE/MODEL.
- `_open()` — urlopen with Basic/Bearer auth (+ Anthropic x-api-key special case).
- `list_local_models()` — probes OpenAI `/v1/models` and Ollama `/api/tags` (order flipped for
  cloud); `endpoint_up()`; `mode()` (honors DGX_MODE, else `real` iff endpoint_up).
- `_PREFERRED` model ranking; `pick_model()`; `DEFAULT_MAX_TOKENS = 1024`, `FAST_MAX_TOKENS = 320`;
  `PKG`, `SANDBOX = PKG/".sandbox"`, `ensure_sandbox()`; `DGX_SPECS` dict (DGX Spark GB10 128 GB /
  DGX Station GB300 784 GB hardware facts).
- Module-level resolution at import: `MODE = mode()`, tunnel→local fallback, `MODEL = pick_model()`.

Also copy `dgxsh.py` verbatim (SSH console backend; env `DGX_SSH_HOST/USER/PORT/KEY`, defaults
`your-spark.your-tailnet.ts.net` / `your-dgx-user`, BatchMode key-only auth).

## 4. requirements.txt

Apps 02–12 (the convention):
```
fastapi>=0.110
uvicorn[standard]>=0.29
openai>=1.30
```
(01 differs: a comment header + `openai>=1.40`, `fastapi>=0.110`, `uvicorn>=0.29`.)
The root README launch recipe: `uv pip install -r week23/<folder>/requirements.txt` then
`.venv/bin/python week23/<folder>/tutorial_server.py`.

## 5. Content/simulation factoring (from 02_nim_microservices)

- `view.py` — shared narration engine, imported by every demo. Glyph vocabulary (the frontend
  colorizer keys off these): `▣` banner/DGX, `»` prompt, `·` answer, `◆` metric, `~ REASON`,
  `→ ACT`, `← OBSERVE`, `━`/`┌─`/`└` banners, `═` result. API:
  - `banner(part, title, level)`, `mode_line(model)`, `is_sim()`.
  - `generate(prompt, *, model, max_tokens=400, title)` — SIM: streams `sim.stream_generate()`
    word-by-word and prints `◆ ~N tok · simulated ~X tok/s · on your DGX · $0.0000`; REAL:
    OpenAI SDK `chat.completions.create(..., stream=True)` against `config.BASE_URL`, merging
    `delta.content` + `delta.reasoning`, then a real tok/s metric + `config.cost_note()`.
  - `classify(prompt, labels, ...)` — terse label extraction; auto-picks a direct-answer
    (non-thinking) model via `_THINKING_PAT`/`_DIRECT_PREF`, strips `<think>` blocks, last
    label mention wins.
  - `_endpoint_error(e)` — friendly 404/405/401 hints pointing at the 🔌 Connection panel.
- `sim.py` — small and app-specific (~35 lines in 02): a domain data table (e.g. the NIM
  CATALOG), `installed_models() -> list[str]`, `tok_s(model) -> float`, and
  `stream_generate(prompt, model)` yielding a canned `[simulated …]` answer word-by-word with
  per-token delay derived from tok_s. All domain-specific simulated "facts" live here.
- `demos/stepNN_*.py` — each imports `view` (and `config`/`sim` as needed), calls
  `view.banner(...)`, `view.mode_line()`, prints concept narration in plain text using the
  glyph vocabulary, shows real DGX shell commands as plain `$ …`/`docker …` lines, and calls
  `view.generate()/classify()` for the live/simulated inference moments. Exit 0 on success —
  the UI turns the sidebar entry green on `__EXIT__ 0`.

## 6. static/guide.html — single file, no external libraries, inline SVG

One self-contained HTML file (~640–770 lines): `<style>` block, HTML skeleton, then three
`<script>` blocks (main app, Week 23 navigator, DGX console). No CDN, no fonts fetched, no
frameworks — "sovereign".

### 6.1 CSS design tokens (copy exactly)

```css
:root{
  --bg:#0c0f0a; --panel:#13160f; --panel2:#1a1f14; --line:#2c3322;
  --txt:#e8efe0; --dim:#8f9a82; --accent:#76b900; --green:#76b900;
  --amber:#d29922; --red:#f85149; --violet:#bc8cff; --cyan:#56d4dd; --term:#080a06;
}
```
Dark olive/black theme with NVIDIA green `#76b900` as accent. Extra literal colors used:
body desc text `#cdd6c2`, terminal text `#d6deca`, reasoning grey-green `#7d8a6a`, smi
`#7fb2b6`, cmd blue `#9fb8d4`, term bar bg `#0f120b`, placeholder `#586049`, hover lime
`#a3e635`, svg edge `#4a5340` / `#3a4230`.

Fonts: body `font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,
Arial,sans-serif`; monospace `ui-monospace,SFMono-Regular,Menlo,monospace` for `.term`,
`pre.src`, seq messages. JS mirror of the palette: `const VC={g:"#76b900",c:"#56d4dd",
v:"#bc8cff",a:"#d29922",r:"#f85149"}`.

### 6.2 Layout skeleton

```
<header>  ▣ <h1> · .sub tagline · .grow spacer · model: <select.model-sel>
          · pill#pill-mode (SIM/REAL) · 🔌 Connection · 🖥️ DGX console · 🧹 Clean sandbox
          · divider · ◀ #nav-prev · #nav-pos "app NN/12" · #nav-next ▶
<div.progress><div#bar>          ← 3px gradient progress bar (accent→cyan)
<div.legendbar>                  ← 💻 laptop vs 🖥️ DGX legend + "where do I run…" link
<div.main>                       ← CSS grid: 340px sidebar | 1fr content (1 col <900px)
  <nav.side#side>                ← groups (.grp uppercase labels) + .step rows (icon+title)
  <section.content#content>      ← crumb, h2 + .lv level chip, diagrams, desc, src, actions, term
overlays: #connpanel (connection), #help (💻vs🖥️ table), #dgxov (DGX console)
```

Key component classes: `.pill`(+`.real`/`.sim` dot colors green/cyan), `.tbtn` toolbar button,
`.run` big green run button, `.step`/`.sel`/`.done`/`.fail` sidebar states, `.sp` spinner,
`.lv.beginner|intermediate|advanced|all` level chips (green/cyan/violet/amber tints),
`.srcwrap`/`.srctog`/`pre.src(.open)` collapsible source, `.term-wrap`/`.term-bar` (3 traffic
dots) /`.term`/`.footer` terminal, `.runhint`, `.hintbar`, `.help-ov`/`.help-box` modal,
`.cf-row` form rows, `.conn-modes` radio list.

### 6.3 SIM/REAL badge (mode pill)

`<span class="pill clickable" id="pill-mode" onclick="toggleConn()"><span class="dot"></span>
<span id="mode-txt">checking…</span></span>`. JS: `pillText()` returns
`REAL · ${CONN} · ${MODEL}` or `SIM · ${CONN}`; `p.classList.toggle("real", MODE==="real")`
etc. Dot is green for real, cyan for sim. Run button label flips
`▶ Run on the DGX` / `▶ Run (simulated)`; a `.note` next to it explains the mode.

### 6.4 Connection panel

Modal `#connpanel` with 3 radios (local / tunnel / cloud) → `connFields()` regenerates
`#conn-fields`: local = Ollama URL; tunnel = tunnel URL + optional basic-auth; cloud =
provider `<select>` (NVIDIA build / Hugging Face / Ollama Cloud / OpenRouter / Anthropic /
custom, prefilled URLs from `const CLOUD_PROV`) + endpoint URL + password-type API key +
"not sovereign" warning. "Connect & refresh" → `applyConn()` POSTs `/api/connect`, then
`refreshConn()` re-fetches `/api/steps` and updates pill/model dropdown; status line shows
`✓ connected · conn · N model(s)` or `⚠ not reachable — running in SIM`.

### 6.5 How the frontend calls the backend

- `jget`/`jpost` fetch helpers; `boot()` loads `/api/steps` into globals
  `STEPS, MODE, MODEL, BASE, CONN, MODELS`, selects step 0.
- Run streaming: `fetch("/api/run", POST {step_id})` → `resp.body.getReader()` +
  `TextDecoder`, accumulate text, split display at the `__EXIT__` marker, regex
  `/__EXIT__ (\-?\d+) ([\d.]+)/` for exit code + duration; per-step `termCache`; footer shows
  `✓ ran on-device (exit 0) · 12.3s · $0 cloud cost` (or `via cloud · cloud usage billed`).
- `colorize(text)` maps the view.py glyph prefixes to `.l-*` span classes (l-dgx, l-gpu,
  l-smi, l-prompt, l-reason, l-answer, l-metric, l-act, l-observe, l-result, l-pass, l-warn,
  l-fail, l-banner, l-cmd) — keep demo output prefixes consistent with §5.
- Source viewer: lazy `GET /api/source/{id}` into `srcCache`, collapsible `<pre.src>`.
- DGX console overlay: quick-command buttons (nvidia-smi, docker ps, ollama, mem/disk,
  gpu stats), streamed via fetch reader with an `AbortController` stop button.

### 6.6 Inline-SVG animated diagram technique (the shared JS viz toolkit)

All charts are generated by pure-JS helper functions returning HTML strings, keyed per step
in `const VIZ = { intro: …, step01: …, … }` and injected in `renderContent()` via
`${VIZ[s.id]||""}`. Helpers (copy them verbatim):

- `vbars(title, rows, unit, hint)` — animated horizontal bars (`.vbar` width animation
  `vgrow` 0.9 s with staggered `animation-delay: i*0.07s`).
- `vmeter(title, used, total, hint)` — capacity meter with gradient fill.
- `vseq(title, actors, steps, hint)` — animated sequence diagram; actors across the top,
  `.seq-msg` rows fade/slide in one-by-one (`seqin`, delay `0.15+i*0.55s`); kinds
  call|ret|reason|answer with distinct border/arrow (`→ ← ↻ ⇒`).
- `vpipe(title, nodes, hint)` — pipeline of `.node` chips (cls s|r|a|f = green/grey/cyan/violet)
  joined by `.arrow` glyphs; `.loopdiag .arrow::after` adds a travelling cyan pulse dot
  (`vflow` 2.4 s infinite).
- `veqn(...)` — the "MODEL + HARNESS = AGENT" equation boxes (`.eqn`, `vpop` fade-in).
- `varch(title, nodes, edges, hint, opts)` — THE software-architecture diagram: a real
  `<svg class="arch" viewBox="0 0 100 H">` with node rects on a 0–100 grid
  (`{id,x,y,w,h,label,sub,cls}`), edges clipped to box borders with an arrowhead
  `<marker id="arw">`, class `edge draw` (stroke-dasharray 220 draw-on animation `adraw`)
  or `edge dashed`, mid-edge `<text.elabel>`, `gnode` pop-in per box, and **animated dataflow
  pulses**: `<circle class="pulse" r="0.85"><animateMotion dur="2.6s" repeatCount="indefinite">
  <mpath href="#edgeId"/></animateMotion></circle>` along each edge. Font sizes are tiny
  (2.3px/1.8px/1.7px) because of the 100-unit viewBox.
- Also two static snippet constants: `FLOW` (the DGX→PROMPT→REASON→ANSWER→tok/s loopdiag with
  "⛔ never leaves the box") shown on every run step, and `STACK` (layer stack) on the intro.
- Convention: `intro` gets `varch(...) + veqn(...)`; each run chapter gets one or two
  `vbars`/`vseq`/`vpipe`; `outro` gets a `vpipe` of the Week 23 stack.

### 6.7 Week 23 navigator (second `<script>`)

Every guide.html embeds the same IIFE with
`const TUTS=[["Nemotron open models",8100],…,["Capstone — smart hotel",8111]]` (12 entries),
finds the current app by `location.port`, wires `#nav-prev`/`#nav-next` hrefs (wrap-around),
`#nav-pos` = "app NN/12", and Alt+←/→ keyboard shortcuts. Hidden if the port isn't in the
list. **A new hub app must decide whether to join this array — changing it means editing all
12 existing guide.html files.** (If Stack Navigator runs on a port not in TUTS, the navigator
simply hides itself — safe default.)

## 7. Voice & microcopy

- Em dashes and middle dots everywhere: "Ch 2 · The Nemotron 3 family — pick by task".
- Glyphs: ▣ (app/brand), ◈ (SIM), 🔌 Connection, 🖥️ DGX console, 🧹 Clean sandbox, 💻 laptop.
- Constant refrain: "$0 cloud cost", "sovereign", "nothing leaves the box",
  "Agent = Model + Harness", cross-references like "(App 5)".
- Every app: intro concept chapter → 4–5 runnable "Ch N" chapters grouped in 2–3 sidebar
  groups → outro "Appendix · How to get started" → "Appendix · References & real-world
  applications" (group "Go further").
- README.md (when present) heading style varies; root README table style:
  `| NN | [folder](folder/) | port | what you learn | prereq |`.
