#!/usr/bin/env python3
"""Lab Runner — the step-by-step web app for Week 23's hands-on tutorial track.

A small control plane that parses the 12 TUTORIAL.md files (01–12), serves them
as a structured course (static/guide.html renders it), and runs each folder's
labs/*.py scripts server-side, streaming their output to the browser line-by-line.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / NIM on this laptop,
    or a DGX you point DGX_BASE_URL at). Labs make genuine on-device inference.
  • SIM  — no endpoint reachable → every lab degrades gracefully and prints the
    real commands plus a labeled expected-output sample. Nobody is blocked.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8113 is taken):

    .venv/bin/python week23/00_lab_runner/tutorial_server.py
    # → http://127.0.0.1:8113
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PKG = Path(__file__).resolve().parent                 # …/week23/00_lab_runner
WEEK20 = PKG.parent                                   # …/week23
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
STATIC = PKG / "static"
MEDIA = WEEK20 / "media"                              # …/week23/media
MEDIA_RE = re.compile(r"^[a-z0-9-]+\.(jpg|png|mp4)$")

GUIDE_PORT = int(os.environ.get("LAB_GUIDE_PORT", "8113"))


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


# ── the 12 hands-on tutorials (num, folder, companion-app port) ────────────────
FOLDERS: list[tuple[str, str, int]] = [
    ("01", "01_nemotron_models", 8100),
    ("02", "02_nim_microservices", 8101),
    ("03", "03_dynamo_serving", 8102),
    ("04", "04_agent_skills", 8103),
    ("05", "05_aiq_research_lab", 8104),
    ("06", "06_nemoclaw", 8105),
    ("07", "07_guardrails_openshell", 8106),
    ("08", "08_nemo_relay", 8107),
    ("09", "09_inference_economics", 8108),
    ("10", "10_nemo_gym_rl", 8109),
    ("11", "11_data_flywheel", 8110),
    ("12", "12_capstone_smart_hotel", 8111),
]
FOLDER_SET = {f for _, f, _ in FOLDERS}

LAB_RE = re.compile(r"^lab\d\d_[a-z0-9_]+\.py$")
RUN_ENV_KEYS = ("DGX_CONN", "DGX_BASE_URL", "DGX_API_KEY", "DGX_MODE", "DGX_MODEL")
# Hard cap per lab run. 150s suits local endpoints; big models over a tunnel
# stream slower — override with LAB_RUN_TIMEOUT (seconds) at launch.
RUN_TIMEOUT = float(os.environ.get("LAB_RUN_TIMEOUT", "150"))


# ── TUTORIAL.md parser — pure-stdlib line scanner over H2 headings ─────────────
def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "section"


def _kind_of(title: str) -> str:
    t = title.strip()
    if re.match(r"^0\s*·", t):
        return "paths"
    if re.match(r"^\d+\s*·", t):
        return "step"
    if t.startswith("Labs"):
        return "labs"
    if t.startswith("Try it yourself"):
        return "exercises"
    if t.startswith("Troubleshooting"):
        return "troubleshooting"
    if t.startswith("Next"):
        return "next"
    return "step"


def _split_sections(text: str) -> list[dict]:
    """Split RAW markdown on top-level '## ' headings, fence-aware.

    Everything before the first H2 becomes the intro section. Each section's
    `md` is the verbatim slice of the file, heading line included.
    """
    lines = text.splitlines(keepends=True)
    bounds: list[tuple[int, str]] = []                 # (line index, H2 title)
    fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence and ln.startswith("## "):
            bounds.append((i, ln[3:].strip()))

    first = bounds[0][0] if bounds else len(lines)
    sections = [{"id": "intro", "kind": "intro", "title": "Introduction",
                 "md": "".join(lines[:first])}]
    seen: dict[str, int] = {"intro": 1}
    for j, (i, title) in enumerate(bounds):
        end = bounds[j + 1][0] if j + 1 < len(bounds) else len(lines)
        sid = _slug(title)
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            sid = f"{sid}-{seen[sid]}"
        sections.append({"id": sid, "kind": _kind_of(title), "title": title,
                         "md": "".join(lines[i:end])})
    return sections


def _doc_title(text: str, folder: str) -> str:
    """The H1 text after the em-dash: '# ▶ Hands-on Lab NN — <title>'."""
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if not m:
        return folder
    h1 = m.group(1).strip()
    if "—" in h1:
        return h1.split("—", 1)[1].strip()
    return h1


def _doc_meta(text: str) -> dict:
    """Parse the '**Time** … · **Difficulty** …' line if present."""
    t = re.search(r"\*\*Time\*\*\s*([^·\n]+)", text)
    d = re.search(r"\*\*Difficulty\*\*\s*([^·\n]+)", text)
    return {"time": t.group(1).strip() if t else None,
            "difficulty": d.group(1).strip() if d else None}


def _lab_title(text: str, fname: str) -> str:
    """Title for a lab file: the TUTORIAL.md '**labs/<file>** — …' blurb,
    else a prettified filename."""
    fallback = fname[:-3].replace("_", " ")
    m = re.search(r"\*\*labs/" + re.escape(fname), text)
    if not m:
        return fallback
    line = text[m.end():].splitlines()[0]
    line = line.lstrip("*").strip().lstrip("—–-·").strip()
    # cut at the first sentence-ish boundary
    cuts = [p for p in (line.find(".**"), line.find(". "), line.find(" Run:"))
            if p > 0]
    if cuts:
        line = line[:min(cuts)]
    line = line.rstrip("*").rstrip(".").strip()
    if len(line) > 110:
        line = line[:107].rstrip() + "…"
    return line or fallback


def _list_labs(folder: str, text: str) -> list[dict]:
    labs_dir = WEEK20 / folder / "labs"
    if not labs_dir.is_dir():
        return []
    return [{"file": p.name, "title": _lab_title(text, p.name)}
            for p in sorted(labs_dir.glob("lab*.py")) if LAB_RE.match(p.name)]


def _next_folder(sections: list[dict]) -> str | None:
    for s in sections:
        if s["kind"] == "next":
            m = re.search(r"\.\./(\d\d_[a-z0-9_]+)/", s["md"])
            if m:
                return m.group(1)
    return None


# ── diagrams.json — optional per-folder visuals (hero + architecture/sequence/charts)
DIAGRAM_KEYS = ("architecture", "sequence", "charts")


def _load_diagrams(folder: str) -> tuple[str | None, dict | None]:
    """(hero, diagrams) from <folder>/diagrams.json.

    Missing or invalid file → (None, None) plus a server-log warning — the
    course must never crash over a bad visuals file.
    """
    path = WEEK20 / folder / "diagrams.json"
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level is not a JSON object")
    except Exception as e:                              # OSError / JSONDecodeError / …
        print(f"  ⚠ {folder}/diagrams.json ignored: {e}", file=sys.stderr, flush=True)
        return None, None
    hero = data.get("hero")
    diagrams = {k: data.get(k) for k in DIAGRAM_KEYS}
    return (hero if isinstance(hero, str) and hero else None), diagrams


def _insert_visualize(sections: list[dict]) -> None:
    """Insert the synthetic diagrams section just before 'labs' (or at the end)."""
    sec = {"id": "visualize", "kind": "diagrams",
           "title": "📊 Visualize it — architecture · sequence · numbers",
           "md": ""}
    for i, s in enumerate(sections):
        if s.get("kind") == "labs":
            sections.insert(i, sec)
            return
    sections.append(sec)


def _build_entry(num: str, folder: str, port: int, path: Path) -> dict:
    hero, diagrams = _load_diagrams(folder)
    entry: dict = {"num": num, "folder": folder, "port": port, "title": folder,
                   "meta": {"time": None, "difficulty": None},
                   "sections": [], "labs": [], "next": None,
                   "hero": hero, "diagrams": diagrams}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        entry["parse_warning"] = f"could not read {path.name}: {e}"
        entry["sections"] = [{"id": "intro", "kind": "intro",
                              "title": "Introduction", "md": ""}]
        entry["labs"] = _list_labs(folder, "")
        if diagrams is not None:
            _insert_visualize(entry["sections"])
        return entry
    try:
        sections = _split_sections(text)
        entry.update(title=_doc_title(text, folder), meta=_doc_meta(text),
                     sections=sections, labs=_list_labs(folder, text),
                     next=_next_folder(sections))
    except Exception as e:                              # never crash the course
        entry["parse_warning"] = f"parse failed: {e}"
        entry["sections"] = [{"id": "intro", "kind": "intro",
                              "title": "Introduction", "md": text}]
        try:
            entry["labs"] = _list_labs(folder, text)
        except Exception:
            pass
    if diagrams is not None:
        _insert_visualize(entry["sections"])
    return entry


# parse once at startup; re-read a folder's TUTORIAL.md / diagrams.json only if
# either file's mtime changed (same pattern for both)
_CACHE: dict[str, tuple[tuple[float, float], dict]] = {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _course_entry(num: str, folder: str, port: int) -> dict:
    path = WEEK20 / folder / "TUTORIAL.md"
    key = (_mtime(path), _mtime(WEEK20 / folder / "diagrams.json"))
    hit = _CACHE.get(folder)
    if hit and hit[0] == key:
        return hit[1]
    entry = _build_entry(num, folder, port, path)
    _CACHE[folder] = (key, entry)
    return entry


def course() -> list[dict]:
    return [_course_entry(num, folder, port) for num, folder, port in FOLDERS]


course()                                                # warm the cache at startup


# ── the app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Lab Runner — Week 23 hands-on tutorial track")
_run_lock = asyncio.Lock()


@app.get("/")
async def index():
    guide = STATIC / "guide.html"
    if guide.exists():
        return FileResponse(guide, headers={"Cache-Control": "no-store, max-age=0"})
    return PlainTextResponse(
        "▣ Lab Runner backend is up — static/guide.html is not built yet.\n\n"
        "API is live:\n"
        "  GET  /api/course   — the parsed 12-tutorial course\n"
        "  GET  /api/source?folder=<folder>&lab=<labNN_x.py>\n"
        "  POST /api/run      — {\"folder\",\"lab\",\"env\"} → streamed output\n"
        "  GET  /api/status   — REAL/SIM connection status\n"
        "  GET  /media/<f>    — week23/media images & clips (allowlisted)\n")


@app.get("/static/{fname:path}")
async def static_file(fname: str):
    base = STATIC.resolve()
    target = (base / fname).resolve()
    if not str(target).startswith(str(base) + os.sep) or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/media/{fname}")
async def media_file(fname: str):
    """Serve week23/media assets — strict allowlist, traversal-safe, cacheable."""
    if not MEDIA_RE.match(fname):
        raise HTTPException(status_code=404, detail="not found")
    base = MEDIA.resolve()
    target = (base / fname).resolve()
    if not str(target).startswith(str(base) + os.sep) or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target, headers={"Cache-Control": "max-age=3600"})


@app.get("/api/course")
async def api_course() -> list[dict]:
    return course()


def _lab_path(folder: str, lab: str) -> Path:
    """Strict allowlist: folder ∈ the 12, lab matches ^lab\\d\\d_[a-z0-9_]+\\.py$."""
    if folder not in FOLDER_SET or not LAB_RE.match(lab):
        raise HTTPException(status_code=404, detail="not found")
    path = WEEK20 / folder / "labs" / lab
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return path


@app.get("/api/source")
async def api_source(folder: str = Query(...), lab: str = Query(...)):
    path = _lab_path(folder, lab)
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


class RunRequest(BaseModel):
    folder: str
    lab: str
    env: dict[str, str] | None = None


# ── run history — feeds the per-lab sparklines in the UI ─────────────────────
HISTORY_PATH = PKG / ".run_history.json"               # gitignored, last 50/lab
HISTORY_MAX = 50
_TOKS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*tok/s")


def _load_history() -> dict:
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _record_run(folder: str, lab: str, model: str, code: int,
                secs: float, tok_s: float | None) -> None:
    try:
        hist = _load_history()
        runs = hist.setdefault(f"{folder}/{lab}", [])
        runs.append({"ts": int(time.time()), "model": model, "code": code,
                     "seconds": round(secs, 1), "tok_s": tok_s})
        del runs[:-HISTORY_MAX]
        HISTORY_PATH.write_text(json.dumps(hist), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — history must never break a run
        print(f"⚠ run history not saved: {e}", file=sys.stderr)


def _stream_lab(folder: str, lab: str, extra_env: dict[str, str]):
    async def gen():
        start = time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        for k in RUN_ENV_KEYS:                          # ONLY the whitelisted keys
            v = extra_env.get(k)
            if v is not None:
                env[k] = str(v)
        model = env.get("DGX_MODEL") or config.MODEL
        tok_s: float | None = None                      # last 'NN tok/s' a lab prints
        proc = await asyncio.create_subprocess_exec(
            PY, str(WEEK20 / folder / "labs" / lab), cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=max(1, start + RUN_TIMEOUT - time.time()))
                except asyncio.TimeoutError:
                    proc.kill()
                    _record_run(folder, lab, model, 124, time.time() - start, tok_s)
                    yield (f"\n⏱  lab exceeded {RUN_TIMEOUT:.0f}s — killed.\n"
                           f"__EXIT__ 124 {time.time()-start:.1f}\n")
                    return
                if not line:
                    break
                text = line.decode(errors="replace")
                m = _TOKS_RE.search(text)
                if m:
                    tok_s = float(m.group(1))
                yield text
            await proc.wait()
            _record_run(folder, lab, model, proc.returncode or 0,
                        time.time() - start, tok_s)
            yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
        finally:
            if proc.returncode is None:
                proc.kill()
    return gen()


@app.get("/api/history")
async def api_history() -> dict:
    return _load_history()


@app.post("/api/run")
async def api_run(req: RunRequest):
    _lab_path(req.folder, req.lab)                      # 404 outside the allowlist
    extra_env = req.env or {}

    async def body():
        if _run_lock.locked():
            yield "⚠  another lab is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            yield f"$ {Path(PY).name} week23/{req.folder}/labs/{req.lab}\n\n"
            async for chunk in _stream_lab(req.folder, req.lab, extra_env):
                yield chunk
    return StreamingResponse(body(), media_type="text/plain")


# ── checkpoint progress — server-side so it survives a browser switch ─────────
PROGRESS_PATH = PKG / "progress.json"                  # gitignored
_progress_lock = threading.Lock()
_CKEY_RE = re.compile(r"^[0-9a-z_]+/[a-z0-9-]+/\d+$")   # "<folder>/<section-id>/<n>"


def _load_progress() -> dict:
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class ProgressRequest(BaseModel):
    key: str
    done: bool


@app.get("/api/progress")
async def api_progress_get() -> dict:
    return {"checkpoints": _load_progress()}


@app.post("/api/progress")
async def api_progress_set(req: ProgressRequest) -> dict:
    if not _CKEY_RE.match(req.key or ""):
        raise HTTPException(400, "bad checkpoint key")
    with _progress_lock:                                # serialize read-modify-write
        cks = _load_progress()
        if req.done:
            cks[req.key] = True
        else:
            cks.pop(req.key, None)
        try:
            PROGRESS_PATH.write_text(json.dumps(cks), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — never 500 over a save miss
            raise HTTPException(500, f"could not save progress: {e}")
    return {"ok": True, "count": len(cks)}


# ── built-in terminal: run any shell command without leaving the web app ──────
# This server binds 127.0.0.1 only; the Host-header check below additionally
# guards against DNS-rebinding. It executes arbitrary commands BY DESIGN — it is
# a local, single-user learning console (same trust model as Jupyter).
SHELL_TIMEOUT = 600.0                        # docker/NIM pulls take a while
SHELL_BIN = os.environ.get("SHELL") or "/bin/zsh"
_shell_lock = asyncio.Lock()
_shell_proc: asyncio.subprocess.Process | None = None


class ShellRequest(BaseModel):
    cmd: str
    env: dict[str, str] | None = None


def _local_only(request: Request) -> None:
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
        raise HTTPException(403, "the terminal is available on localhost only")


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the whole process group — shell commands spawn children."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


@app.post("/api/shell")
async def api_shell(req: ShellRequest, request: Request):
    _local_only(request)
    cmd = (req.cmd or "").strip()
    extra_env = req.env or {}

    async def body():
        global _shell_proc
        if not cmd:
            yield "type a command first.\n__EXIT__ 1 0\n"
            return
        if _shell_lock.locked():
            yield "⚠  another command is already running — wait for it or hit ■ Stop.\n__EXIT__ 1 0\n"
            return
        async with _shell_lock:
            start = time.time()
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "TERM": "dumb"}
            for k in RUN_ENV_KEYS:                       # same whitelist as labs
                v = extra_env.get(k)
                if v is not None:
                    env[k] = str(v)
            proc = await asyncio.create_subprocess_exec(
                SHELL_BIN, "-lc", cmd, cwd=str(ROOT), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True)
            _shell_proc = proc
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(),
                            timeout=max(1, start + SHELL_TIMEOUT - time.time()))
                    except asyncio.TimeoutError:
                        _kill_tree(proc)
                        yield (f"\n⏱  command exceeded {SHELL_TIMEOUT:.0f}s — killed.\n"
                               f"__EXIT__ 124 {time.time()-start:.1f}\n")
                        return
                    if not line:
                        break
                    yield line.decode(errors="replace")
                await proc.wait()
                yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
            finally:
                _shell_proc = None
                if proc.returncode is None:
                    _kill_tree(proc)
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/shell/stop")
async def api_shell_stop(request: Request) -> dict:
    _local_only(request)
    proc = _shell_proc
    if proc is None or proc.returncode is not None:
        return {"stopped": False, "detail": "nothing is running"}
    _kill_tree(proc)
    return {"stopped": True, "detail": "killed the command's process group"}


@app.get("/api/status")
async def api_status() -> dict:
    real = config.MODE == "real"
    if real:
        detail = (f"REAL — {config.conn_human()} · labs run genuine inference "
                  f"against {config.safe_base_url()}")
    else:
        detail = (f"SIM — no endpoint reachable via {config.conn_human()}; labs "
                  "degrade gracefully and print expected-output samples ($0, no GPU)")
    models: list[str] = []
    if real:
        try:
            models = await asyncio.to_thread(config.list_local_models)
        except Exception:
            models = []
    return {"mode": config.MODE, "conn": config.CONN,
            "base_url": config.safe_base_url(),
            "model": config.MODEL if real else None, "models": models,
            "run_timeout": RUN_TIMEOUT, "detail": detail}


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  ▣  Lab Runner — Week 23's hands-on track, one lab at a time"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.safe_base_url()}",
                   "        labs run for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; every lab degrades",
                   "        gracefully and still teaches. Go REAL anytime:",
                   "        ollama serve   (or set DGX_BASE_URL / use the 🔌 panel)"]
    parsed = course()
    warn = sum(1 for e in parsed if e.get("parse_warning"))
    banner += [f"      ▤ course: {len(parsed)} tutorials · "
               f"{sum(len(e['sections']) for e in parsed)} sections · "
               f"{sum(len(e['labs']) for e in parsed)} labs"
               + (f" · ⚠ {warn} parse warning(s)" if warn else "")]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set LAB_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
