#!/usr/bin/env python3
"""Maestro Lab Runner — the step-by-step web app for Week 28 (securing multi-agent systems with the CSA
MAESTRO threat-modeling framework, on the deliberately vulnerable "Hotel Ops Ensemble").

A local-only fork of Week 27's Embed Lab Runner: it parses every week28_maestro_mas_security/NN_*/TUTORIAL.md, serves them as one
continuous course (static/guide.html renders it, EN / ไทย), and runs each module's labs/*.py and
exercises/*.py on THIS machine with the repo's .venv Python, streaming output live.

  • 🧠 SIM / 🦙 LIVE — the agents' brain. SIM is a deterministic stand-in (offline, instant); LIVE uses
    local Ollama. The only env the browser may set is MAS_BRAIN / MAS_MODEL, allowlisted in _child_env.
  • /api/status — what is ready here: the venv, week28_maestro_mas_security/common (maskit · ensemble · maestro), Ollama.
  • /api/maestro — the MAESTRO layer / threat catalogue from week28_maestro_mas_security/common/maestro.py.
  • ⌨ terminal — runs commands on this machine only.

Launch (auto-picks a free port if 8128 is taken):

    .venv/bin/python week28_maestro_mas_security/00_maestro_lab_runner/tutorial_server.py
    # → http://127.0.0.1:8128
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

PKG = Path(__file__).resolve().parent                 # …/week28_maestro_mas_security/00_maestro_lab_runner
WEEK = PKG.parent                                     # …/week28_maestro_mas_security
ROOT = WEEK.parent                                    # …/agenticaicodingfitness
COMMON = WEEK / "common"

VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
PY_LABEL = ".venv/bin/python" if VENV_PY.exists() else "python"
STATIC = PKG / "static"
GUIDE_PORT = int(os.environ.get("MAESTRO_GUIDE_PORT", "8128"))
LANGS = ("en", "th")
RUN_TIMEOUT = float(os.environ.get("LAB_RUN_TIMEOUT", "900"))
MODULE_RE = re.compile(r"^\d\d_[a-z0-9_]+$")
FILE_RE = re.compile(r"^(labs/lab\d\d_[a-z0-9_]+|exercises/ex\d\d_[a-z0-9_]+|"
                     r"exercises/solutions/ex\d\d_[a-z0-9_]+)\.py$")
OLLAMA = "http://localhost:11434"


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


def modules() -> list[str]:
    """Every week28_maestro_mas_security/NN_name/ folder with a TUTORIAL.md, in order (00 is this app)."""
    return [p.parent.name for p in sorted(WEEK.glob("[0-9][0-9]_*/TUTORIAL.md"))
            if MODULE_RE.match(p.parent.name) and not p.parent.name.startswith("00_")]


# ── TUTORIAL.md parser — pure-stdlib line scanner over H2 headings ─────────────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _kind_of(title: str) -> str:
    t = title.strip()
    if re.match(r"^0\s*·", t):
        return "setup"
    if t.startswith("Labs"):
        return "labs"
    if t.startswith("Try it yourself"):
        return "exercises"
    if t.startswith("Troubleshooting"):
        return "troubleshooting"
    if t.startswith("Next") or t.startswith("What to build next"):
        return "next"
    return "step"


def _split_sections(text: str) -> list[dict]:
    """Split raw markdown on top-level '## ' headings, fence-aware."""
    lines = text.splitlines(keepends=True)
    bounds: list[tuple[int, str]] = []
    fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence and ln.startswith("## "):
            bounds.append((i, ln[3:].strip()))
    first = bounds[0][0] if bounds else len(lines)
    sections = [{"id": "intro", "kind": "intro", "title": "Introduction", "md": "".join(lines[:first])}]
    seen: dict[str, int] = {"intro": 1}
    for j, (i, title) in enumerate(bounds):
        end = bounds[j + 1][0] if j + 1 < len(bounds) else len(lines)
        sid = _slug(title)
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            sid = f"{sid}-{seen[sid]}"
        sections.append({"id": sid, "kind": _kind_of(title), "title": title, "md": "".join(lines[i:end])})
    return sections


def _doc_title(text: str, folder: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if not m:
        return folder
    h1 = m.group(1).strip()
    return h1.split("—", 1)[1].strip() if "—" in h1 else h1


def _doc_meta(text: str) -> dict:
    t = re.search(r"\*\*Time\*\*\s*([^·\n]+)", text)
    d = re.search(r"\*\*Difficulty\*\*\s*([^·\n]+)", text)
    c = re.search(r"\*\*(?:Hardware|Sparks)\*\*\s*([^\n]+)", text)
    return {"time": t.group(1).strip() if t else None,
            "difficulty": d.group(1).strip() if d else None,
            "cost": c.group(1).strip() if c else None}


def _docstring_title(path: Path) -> str:
    """First docstring line, minus a 'Lab 01-2 ·' / 'Exercise 01 ·' prefix."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:1500]
    except OSError:
        return ""
    m = re.search(r'"""\s*(.+)', head)
    if not m:
        return ""
    line = m.group(1).strip().rstrip('".').strip()
    return line.split("·", 1)[1].strip() if "·" in line else line


def _file_title(text: str, rel: str, path: Path | None = None) -> str:
    """'**labs/<file>** — title.' blurb from TUTORIAL.md, else the docstring, else the name."""
    fname = rel.rsplit("/", 1)[-1]
    fallback = (_docstring_title(path) if path else "") or fname[:-3].split("_", 1)[-1].replace("_", " ")
    m = re.search(r"\*\*" + re.escape(rel) + r"\*\*", text)
    if not m:
        return fallback
    line = text[m.end():].splitlines()[0].lstrip("*").strip().lstrip("—–-·:").strip()
    cuts = [p for p in (line.find(". "),) if p > 0]
    if cuts:
        line = line[:min(cuts)]
    line = line.rstrip(".").strip()
    return (line[:107].rstrip() + "…") if len(line) > 110 else (line or fallback)


def _list_files(folder: str, text: str) -> tuple[list[dict], list[dict]]:
    base = WEEK / folder
    labs = [{"file": f"labs/{p.name}", "title": _file_title(text, f"labs/{p.name}", p)}
            for p in sorted((base / "labs").glob("lab*.py")) if FILE_RE.match(f"labs/{p.name}")]
    exercises = []
    for p in sorted((base / "exercises").glob("ex*.py")):
        rel = f"exercises/{p.name}"
        if not FILE_RE.match(rel):
            continue
        sol = base / "exercises" / "solutions" / p.name
        exercises.append({"file": rel, "title": _file_title(text, rel, p),
                          "solution": f"exercises/solutions/{p.name}" if sol.is_file() else None})
    return labs, exercises


def _next_folder(sections: list[dict]) -> str | None:
    for s in sections:
        if s["kind"] == "next":
            m = re.search(r"\.\./(\d\d_[a-z0-9_]+)/", s["md"])
            if m:
                return m.group(1)
    return None


DIAGRAM_KEYS = ("architecture", "sequence", "charts")


def _load_diagrams(folder: str) -> dict | None:
    path = WEEK / folder / "diagrams.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level is not a JSON object")
    except Exception as e:  # noqa: BLE001 — a bad visuals file must never break the course
        print(f"  ⚠ {folder}/diagrams.json ignored: {e}", file=sys.stderr, flush=True)
        return None
    return {k: data.get(k) for k in DIAGRAM_KEYS}


def _insert_visualize(sections: list[dict], lang: str = "en") -> None:
    title = ("📊 ภาพรวม — สถาปัตยกรรม · ลำดับขั้น · ตัวเลข" if lang == "th"
             else "📊 Visualize it — architecture · sequence · numbers")
    sec = {"id": "visualize", "kind": "diagrams", "title": title, "md": ""}
    for i, s in enumerate(sections):
        if s.get("kind") == "labs":
            sections.insert(i, sec)
            return
    sections.append(sec)


def _build_entry(folder: str, lang: str = "en") -> dict:
    path = WEEK / folder / "TUTORIAL.md"
    th_path = WEEK / folder / "TUTORIAL.th.md"
    diagrams = _load_diagrams(folder)
    entry: dict = {"num": folder[:2], "folder": folder, "title": folder, "lang": "en",
                   "has_th": th_path.is_file(),
                   "meta": {"time": None, "difficulty": None, "cost": None},
                   "sections": [], "labs": [], "exercises": [], "next": None, "diagrams": diagrams}
    try:
        text = path.read_text(encoding="utf-8")
        sections = _split_sections(text)
        labs, exercises = _list_files(folder, text)
        entry.update(title=_doc_title(text, folder), meta=_doc_meta(text), sections=sections,
                     labs=labs, exercises=exercises, next=_next_folder(sections))
        if lang == "th" and th_path.is_file():
            th_text = th_path.read_text(encoding="utf-8")
            th_secs = _split_sections(th_text)
            if len(th_secs) != len(sections):
                entry["parse_warning"] = (f"Thai translation has {len(th_secs)} sections, English has "
                                          f"{len(sections)} — showing English")
            else:
                # Pair by position: ids + kinds (and so progress keys and lab cards) stay English.
                for en_s, th_s in zip(sections, th_secs):
                    en_s["title_en"] = en_s["title"]
                    en_s["md"] = th_s["md"]
                    if en_s["kind"] != "intro":
                        en_s["title"] = th_s["title"]
                th_labs, th_ex = _list_files(folder, th_text)
                for dst, src in ((labs, th_labs), (exercises, th_ex)):
                    for a, b in zip(dst, src):
                        if b["title"]:
                            a["title"] = b["title"]
                sections[0]["title"] = "บทนำ"
                entry.update(title=_doc_title(th_text, folder), meta=_doc_meta(th_text), lang="th")
    except Exception as e:  # noqa: BLE001 — never crash the course
        entry["parse_warning"] = f"parse failed: {e}"
        entry["sections"] = [{"id": "intro", "kind": "intro", "title": "Introduction", "md": ""}]
    if diagrams is not None:
        _insert_visualize(entry["sections"], entry["lang"])
    return entry


_CACHE: dict[tuple[str, str], tuple[tuple, dict]] = {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _dir_sig(path: Path) -> tuple:
    return tuple(sorted(p.name for p in path.glob("*.py"))) if path.is_dir() else ()


def course(lang: str = "en") -> list[dict]:
    lang = lang if lang in LANGS else "en"
    out = []
    for folder in modules():
        base = WEEK / folder
        key = (_mtime(base / "TUTORIAL.md"), _mtime(base / "TUTORIAL.th.md"), _mtime(base / "diagrams.json"),
               _dir_sig(base / "labs"), _dir_sig(base / "exercises"),
               _dir_sig(base / "exercises" / "solutions"))
        hit = _CACHE.get((folder, lang))
        if not hit or hit[0] != key:
            hit = (key, _build_entry(folder, lang))
            _CACHE[(folder, lang)] = hit
        out.append(hit[1])
    return out


# ── the app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Maestro Lab Runner — Week 28")
_run_lock = asyncio.Lock()
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")


def _local_only(request: Request) -> None:
    """Only this app, from this machine, may call state-changing endpoints.

    Host check → blocks DNS rebinding. Origin check → blocks any other website
    open in the same browser from POSTing to 127.0.0.1 (it would carry its own
    Origin). Scripts/curl on this machine send no Origin and are allowed.
    """
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
    if host not in _LOCAL_HOSTS:
        raise HTTPException(403, "available on localhost only")
    origin = request.headers.get("origin")
    if origin and origin != "null":
        o_host = re.sub(r"^https?://", "", origin.lower()).split("/")[0].rsplit(":", 1)[0]
        if o_host not in _LOCAL_HOSTS:
            raise HTTPException(403, "cross-site request refused")


@app.get("/")
async def index():
    guide = STATIC / "guide.html"
    if guide.exists():
        return FileResponse(guide, headers={"Cache-Control": "no-store, max-age=0"})
    return PlainTextResponse("Maestro Lab Runner backend is up — static/guide.html missing.")


@app.get("/static/{fname:path}")
async def static_file(fname: str):
    base = STATIC.resolve()
    target = (base / fname).resolve()
    if not str(target).startswith(str(base) + os.sep) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target, headers={"Cache-Control": "no-store, max-age=0"})


MEDIA_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
             ".gif": "image/gif", ".svg": "image/svg+xml", ".mp4": "video/mp4", ".webm": "video/webm"}


@app.get("/w28media/{folder}/{rel:path}")
async def module_media(folder: str, rel: str):
    """Serve ONLY week28_maestro_mas_security/<module>/media/* and week28_maestro_mas_security/<module>/.runs/* (files a lab just produced).
    Extension allowlist, no traversal, no directory listings."""
    if folder not in modules() or not re.fullmatch(r"(media|\.runs)/[A-Za-z0-9_.\-/]{1,160}", rel or ""):
        raise HTTPException(404, "not found")
    base = (WEEK / folder).resolve()
    target = (base / rel).resolve()
    ext = target.suffix.lower()
    top = (base / rel.split("/", 1)[0]).resolve()
    if ext not in MEDIA_EXT or not str(target).startswith(str(top) + os.sep) or not target.is_file():
        raise HTTPException(404, "not found")
    cache = "no-store, max-age=0" if rel.startswith(".runs/") else "max-age=3600"
    return FileResponse(target, media_type=MEDIA_EXT[ext], headers={"Cache-Control": cache})


@app.get("/api/course")
async def api_course(lang: str = "en") -> list[dict]:
    return course(lang)


def _file_path(folder: str, rel: str) -> Path:
    """Strict allowlist: folder ∈ discovered modules, rel ∈ labs/ex/solutions patterns."""
    if folder not in modules() or not FILE_RE.match(rel or ""):
        raise HTTPException(404, "not found")
    path = WEEK / folder / rel
    if not path.is_file():
        raise HTTPException(404, "not found")
    return path


@app.get("/api/source")
async def api_source(folder: str, file: str):
    return PlainTextResponse(_file_path(folder, file).read_text(encoding="utf-8", errors="replace"))


# ── run history (sparklines) ───────────────────────────────────────────────────
HISTORY_PATH = PKG / ".run_history.json"
HISTORY_MAX = 50


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _record_run(key: str, mode: str, code: int, secs: float) -> None:
    try:
        hist = _load_json(HISTORY_PATH)
        runs = hist.setdefault(key, [])
        runs.append({"ts": int(time.time()), "mode": mode, "code": code, "seconds": round(secs, 1), "cost": 0})
        del runs[:-HISTORY_MAX]
        HISTORY_PATH.write_text(json.dumps(hist), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — history must never break a run
        print(f"⚠ run history not saved: {e}", file=sys.stderr)


BRAINS = ("sim", "ollama")
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,80}$")


def _brain_env(extra: dict[str, str]) -> dict[str, str]:
    """ALLOWLIST of what the browser may set: MAS_BRAIN ∈ BRAINS, and MAS_MODEL only if it is a
    well-formed name of a model this machine's Ollama actually has. Everything else is dropped."""
    out: dict[str, str] = {}
    brain = extra.get("MAS_BRAIN")
    if isinstance(brain, str) and brain in BRAINS:
        out["MAS_BRAIN"] = brain
    model = extra.get("MAS_MODEL")
    if (out.get("MAS_BRAIN") == "ollama" and isinstance(model, str) and MODEL_RE.match(model)
            and "embed" not in model.lower() and model in _local_models(_ollama_models())):
        out["MAS_MODEL"] = model
    return out


def _child_env(extra: dict[str, str]) -> dict[str, str]:
    """Env for lab subprocesses and the terminal. The browser may set only the brain (see _brain_env)."""
    env = {k: v for k, v in os.environ.items() if k not in ("MAS_BRAIN", "MAS_MODEL")}  # the toggle decides
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}, **_brain_env(extra or {}))
    return env


class RunRequest(BaseModel):
    folder: str
    file: str
    env: dict[str, str] | None = None


@app.get("/api/history")
async def api_history() -> dict:
    return _load_json(HISTORY_PATH)


@app.post("/api/run")
async def api_run(req: RunRequest, request: Request):
    _local_only(request)
    path = _file_path(req.folder, req.file)
    env = await asyncio.to_thread(_child_env, req.env or {})
    mode = env.get("MAS_BRAIN", "sim")
    brain = f"LIVE · {env['MAS_MODEL']}" if "MAS_MODEL" in env else ("LIVE" if mode == "ollama" else "SIM")

    async def body():
        if _run_lock.locked():
            yield "⚠  another lab is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            start = time.time()
            yield f"$ {PY_LABEL} week28_maestro_mas_security/{req.folder}/{req.file}   [THIS MACHINE · brain={brain}]\n\n"
            proc = await asyncio.create_subprocess_exec(
                PY, str(path), cwd=str(ROOT), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(proc.stdout.readline(),
                                                      timeout=max(1, start + RUN_TIMEOUT - time.time()))
                    except asyncio.TimeoutError:
                        proc.kill()
                        _record_run(f"{req.folder}/{req.file}", mode, 124, time.time() - start)
                        yield f"\n⏱  exceeded {RUN_TIMEOUT:.0f}s — killed.\n__EXIT__ 124 {time.time()-start:.1f}\n"
                        return
                    if not line:
                        break
                    yield line.decode(errors="replace")
                await proc.wait()
                _record_run(f"{req.folder}/{req.file}", mode, proc.returncode or 0, time.time() - start)
                yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
            finally:
                if proc.returncode is None:
                    proc.kill()
    return StreamingResponse(body(), media_type="text/plain")


# ── checkpoint progress — server-side so it survives a browser switch ─────────
PROGRESS_PATH = PKG / "progress.json"
_progress_lock = threading.Lock()
_CKEY_RE = re.compile(r"^[0-9a-z_]+/[a-z0-9-]+/\d+$")


class ProgressRequest(BaseModel):
    key: str
    done: bool


@app.get("/api/progress")
async def api_progress_get() -> dict:
    return {"checkpoints": _load_json(PROGRESS_PATH)}


@app.post("/api/progress")
async def api_progress_set(req: ProgressRequest, request: Request) -> dict:
    _local_only(request)
    if not _CKEY_RE.match(req.key or ""):
        raise HTTPException(400, "bad checkpoint key")
    with _progress_lock:
        cks = _load_json(PROGRESS_PATH)
        if req.done:
            cks[req.key] = True
        else:
            cks.pop(req.key, None)
        try:
            PROGRESS_PATH.write_text(json.dumps(cks), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"could not save progress: {e}")
    return {"ok": True, "count": len(cks)}


# ── built-in terminal: this machine only ───────────────────────────────────────
SHELL_TIMEOUT = {"mac": 600.0}
SHELL_BIN = os.environ.get("SHELL") or "/bin/zsh"
_shell_lock = asyncio.Lock()
_shell_proc: asyncio.subprocess.Process | None = None


class ShellRequest(BaseModel):
    cmd: str
    target: str = "mac"
    env: dict[str, str] | None = None


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _shell_argv(cmd: str, target: str) -> tuple[list[str], str]:
    return [SHELL_BIN, "-lc", cmd], "this machine"


@app.post("/api/shell")
async def api_shell(req: ShellRequest, request: Request):
    _local_only(request)
    cmd = (req.cmd or "").strip()
    target = req.target if req.target in SHELL_TIMEOUT else "mac"
    env = {**(await asyncio.to_thread(_child_env, req.env or {})), "TERM": "dumb"}
    try:
        argv, where = _shell_argv(cmd, target) if cmd else ([], "")
        setup_error = ""
    except HTTPException as e:
        argv, where, setup_error = [], "", str(e.detail)
    limit = SHELL_TIMEOUT[target]

    async def body():
        global _shell_proc
        if not cmd:
            yield "type a command first.\n__EXIT__ 1 0\n"
            return
        if setup_error:
            yield f"✕ {setup_error}\n__EXIT__ 2 0\n"
            return
        if _shell_lock.locked():
            yield "⚠  another command is already running — wait for it or hit ■ Stop.\n__EXIT__ 1 0\n"
            return
        async with _shell_lock:
            start = time.time()
            if target != "mac":
                yield f"◆ running on {where}\n"
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=str(ROOT), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            _shell_proc = proc
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(proc.stdout.readline(),
                                                      timeout=max(1, start + limit - time.time()))
                    except asyncio.TimeoutError:
                        _kill_tree(proc)
                        yield f"\n⏱  command exceeded {limit:.0f}s — killed. Run long jobs under tmux or nohup.\n__EXIT__ 124 {time.time()-start:.1f}\n"
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
    return {"stopped": True, "detail": "killed the command's process group (a remote job keeps running if it was nohup'd)"}


# ── status: what is ready on this machine ─────────────────────────────────────
_STATUS: dict = {"checked": 0.0}
_status_lock = asyncio.Lock()


def _ollama_models() -> list[str]:
    from urllib.request import urlopen
    try:
        with urlopen(OLLAMA + "/api/tags", timeout=1.5) as r:   # noqa: S310 — localhost
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def _local_models(models: list[str]) -> list[str]:
    """Never offer a cloud model (`*:cloud`, `*-cloud`) — Week 28's LIVE brain stays on this machine."""
    return [m for m in models if "cloud" not in m]


_TOOLS: dict = {}
KIT_FILES = ("maskit.py", "ensemble.py", "maestro.py")


def _venv_version() -> str:
    if "version" not in _TOOLS:
        try:
            r = subprocess.run([PY, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
                               capture_output=True, text=True, timeout=30)
            _TOOLS["version"] = r.stdout.strip()
        except Exception:  # noqa: BLE001
            _TOOLS["version"] = ""
    return _TOOLS["version"]


def _kit_probe() -> dict:
    """Do week28_maestro_mas_security/common/{maskit,ensemble,maestro}.py exist and import? Cached by their mtimes —
    the kit may still be being written, so a missing or broken file is reported, never raised."""
    sig = tuple(_mtime(COMMON / f) for f in KIT_FILES)
    hit = _TOOLS.get("kit")
    if hit and hit[0] == sig:
        return hit[1]
    files = {f[:-3]: (COMMON / f).is_file() for f in KIT_FILES}
    ok, err = False, ""
    if all(files.values()):
        try:
            r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'week28_maestro_mas_security/common'); "
                                          "import maskit, ensemble, maestro"],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0
            err = "" if ok else ((r.stderr.strip().splitlines() or ["import failed"])[-1])[:300]
        except Exception as e:  # noqa: BLE001
            err = str(e)[:300]
    else:
        err = "missing: " + ", ".join(f"{k}.py" for k, v in files.items() if not v)
    kit = {"files": files, "ok": ok, "error": err}
    _TOOLS["kit"] = (sig, kit)
    return kit


def _probe() -> dict:
    models = _ollama_models()
    local = _local_models(models)
    return {"checked": time.time(), "venv": VENV_PY.exists(), "python": _venv_version(), "kit": _kit_probe(),
            "ollama": bool(models), "ollama_models": local[:20]}


@app.get("/api/status")
async def api_status() -> dict:
    async with _status_lock:
        if time.time() - _STATUS.get("checked", 0) > 30:
            _STATUS.update(await asyncio.to_thread(_probe))
    s = _STATUS
    kit = s.get("kit") or {}
    if not s.get("venv"):
        detail = ".venv missing at the repo root — `uv venv && uv pip install -r requirements.txt`"
    elif not kit.get("ok"):
        detail = f"week28_maestro_mas_security/common kit not ready — {kit.get('error') or 'not probed'}"
    else:
        detail = "ready — SIM brain runs offline; LIVE needs Ollama" + ("" if s.get("ollama") else " (off)")
    # Shape kept compatible with the Week 25–27 page (a = this machine, always reachable).
    return {**s, "default_mode": "sim", "detail": detail, "run_timeout": RUN_TIMEOUT,
            "a": {"host": "this machine", "reachable": True, "gpu": ""},
            "b": {"host": "", "reachable": False}, "endpoints": {}, "laptop_models": s.get("ollama_models", [])}


# ── 🗺 MAESTRO catalogue — read from week28_maestro_mas_security/common/maestro.py in a subprocess ──
_MAESTRO: dict = {}
MAESTRO_TTL = 60.0


def _maestro_catalog() -> dict:
    path = COMMON / "maestro.py"
    sig = _mtime(path)
    hit = _MAESTRO.get("hit")
    if hit and hit[0] == sig and time.time() - hit[1] < MAESTRO_TTL:
        return hit[2]
    if sig < 0:
        data: dict = {"error": "week28_maestro_mas_security/common/maestro.py not found yet"}
    else:
        try:
            r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'week28_maestro_mas_security/common'); "
                                          "import maestro, json; print(json.dumps(maestro.catalog()))"],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                raise RuntimeError((r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1])
            data = json.loads(r.stdout.strip().splitlines()[-1])
            if not isinstance(data, dict):
                raise ValueError("catalog() did not return a JSON object")
        except Exception as e:  # noqa: BLE001 — the dialog shows the error, the app keeps running
            data = {"error": f"maestro.catalog() failed: {e}"[:400]}
    _MAESTRO["hit"] = (sig, time.time(), data)
    return data


@app.get("/api/maestro")
async def api_maestro() -> dict:
    return await asyncio.to_thread(_maestro_catalog)


@app.post("/api/status/refresh")
async def api_status_refresh(request: Request) -> dict:
    _local_only(request)
    _STATUS["checked"] = 0.0
    return await api_status()


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    parsed = course()
    st = _probe()
    _STATUS.update(st)
    kit = st["kit"]
    banner = ["", "  🎼  Maestro Lab Runner — Week 28 · Securing multi-agent systems with MAESTRO"]
    banner += [f"      {'✓' if st['venv'] else '✕'} {PY_LABEL}"
               + (f" · Python {st['python']}" if st["python"] else "") + ("" if st["venv"] else " — repo .venv missing")]
    banner += [f"      {'✓' if kit['ok'] else '◈'} week28_maestro_mas_security/common kit (maskit · ensemble · maestro)"
               + ("" if kit["ok"] else f" — {kit['error']}")]
    banner += [f"      {'✓' if st['ollama'] else '◈'} Ollama "
               + (", ".join(st["ollama_models"][:4]) if st["ollama"] else "not running — 🧠 SIM brain still works")]
    banner += [f"      ▤ course: {len(parsed)} modules · {sum(len(e['sections']) for e in parsed)} sections · "
               f"{sum(len(e['labs']) for e in parsed)} labs · {sum(len(e['exercises']) for e in parsed)} exercises"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set MAESTRO_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
