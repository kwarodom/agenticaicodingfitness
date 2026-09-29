#!/usr/bin/env python3
"""Spark Lab Runner — the step-by-step web app for Week 25 (DGX Spark: fine-tune, serve, build agents).

Same shape as Week 24's Jev Lab Runner: it parses every week25/NN_*/TUTORIAL.md,
serves them as one continuous course (static/guide.html renders it), and runs
each module's labs/*.py and exercises/*.py server-side, streaming output live.

Spark-specific additions:
  • LIVE / DRY switch. LIVE runs commands on your DGX Spark(s) — locally when this
    server runs ON the Spark, else over `ssh $SPARK_HOST` (and `$SPARK_HOST2` for the
    two-Spark modules). DRY runs nothing and shows RECORDED / REFERENCE output, labelled.
  • 🖥 Spark setup dialog: Spark hostnames + HF_TOKEN / NGC_API_KEY / NVIDIA_API_KEY,
    saved server-side to week25/.env.local (gitignored, mode 0600), never sent back.
  • Endpoint probes: which of Ollama / vLLM / SGLang / TRT-LLM / llama.cpp / LiteLLM answer.
  • /api/chat — the proxy behind the inline "⚡ Ask the Spark" blocks.
  • The ⌨ terminal can target this Mac, Spark A or Spark B (ssh), picked per command.

Launch (auto-picks a free port if 8125 is taken):

    .venv/bin/python week25/00_spark_lab_runner/tutorial_server.py
    # → http://127.0.0.1:8125
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import re
import shlex
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

PKG = Path(__file__).resolve().parent                 # …/week25/00_spark_lab_runner
WEEK = PKG.parent                                     # …/week25
ROOT = WEEK.parent                                    # …/agenticaicodingfitness
sys.path.insert(0, str(WEEK / "common"))
import sparkkit  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
STATIC = PKG / "static"
GUIDE_PORT = int(os.environ.get("SPARK_GUIDE_PORT", "8125"))

RUN_ENV_KEYS = ("SPARK_MODE", "SPARK_ALLOW_LAPTOP")   # the ONLY env a browser may set
LANGS = ("en", "th")
# Settings the 🖥 dialog may write. Secrets are write-only; hosts are shown back (they are not secret).
SETTINGS = {
    "SPARK_HOST": {"label": "Spark A — ssh target", "secret": False, "hint": "e.g. spark-abcd or me@spark-abcd.tailnet.ts.net"},
    "SPARK_HOST2": {"label": "Spark B — ssh target (two-Spark modules)", "secret": False, "hint": "e.g. spark-efgh"},
    "SPARK_API_HOST": {"label": "Spark A — HTTP host (optional)", "secret": False,
                       "hint": "leave empty to reuse the ssh host · 'localhost' if you ssh -L tunnel the ports"},
    "HF_TOKEN": {"label": "Hugging Face token", "secret": True, "hint": "gated models (Llama, Gemma) · huggingface.co/settings/tokens"},
    "NGC_API_KEY": {"label": "NGC API key", "secret": True, "hint": "NIM containers + nvcr.io pulls · ngc.nvidia.com"},
    "NVIDIA_API_KEY": {"label": "NVIDIA API key (build.nvidia.com)", "secret": True, "hint": "hosted Nemotron for NemoClaw / NAT fallbacks"},
}
SETTING_RE = {"SPARK_HOST": sparkkit.HOST_RE, "SPARK_HOST2": sparkkit.HOST_RE, "SPARK_API_HOST": sparkkit.HOST_RE}
RUN_TIMEOUT = float(os.environ.get("LAB_RUN_TIMEOUT", "900"))
MODULE_RE = re.compile(r"^\d\d_[a-z0-9_]+$")
FILE_RE = re.compile(r"^(labs/lab\d\d_[a-z0-9_]+|exercises/ex\d\d_[a-z0-9_]+|"
                     r"exercises/solutions/ex\d\d_[a-z0-9_]+)\.py$")


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
    """Every week25/NN_name/ folder with a TUTORIAL.md, in order (00 is this app)."""
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
app = FastAPI(title="Spark Lab Runner — Week 25")
_run_lock = asyncio.Lock()
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")

# Optional remote access through a local reverse proxy such as `tailscale serve`: set SPARK_GUIDE_PASSWORD
# and list the proxy's hostnames in SPARK_GUIDE_HOSTS (comma-separated). Every request then needs HTTP Basic
# auth with that password (any username). Extra hosts are ignored unless a password is set.
_PASSWORD = sparkkit.cfg("SPARK_GUIDE_PASSWORD")
_ALLOWED_HOSTS = _LOCAL_HOSTS + (tuple(h.strip().lower() for h in sparkkit.cfg("SPARK_GUIDE_HOSTS").split(",")
                                       if h.strip()) if _PASSWORD else ())


@app.middleware("http")
async def _password_gate(request: Request, call_next):
    if _PASSWORD:
        given = ""
        auth = request.headers.get("authorization", "")
        if auth[:6].lower() == "basic ":
            try:
                given = base64.b64decode(auth[6:]).decode("utf-8").partition(":")[2]
            except Exception:  # noqa: BLE001
                given = ""
        if not hmac.compare_digest(given.encode(), _PASSWORD.encode()):
            return PlainTextResponse("password required", status_code=401,
                                     headers={"WWW-Authenticate": 'Basic realm="Spark Lab Runner", charset="UTF-8"'})
    return await call_next(request)


def _local_only(request: Request) -> None:
    """Only this app, from this machine, may call state-changing endpoints.

    Host check → blocks DNS rebinding. Origin check → blocks any other website
    open in the same browser from POSTing to 127.0.0.1 (it would carry its own
    Origin). Scripts/curl on this machine send no Origin and are allowed.
    With SPARK_GUIDE_PASSWORD set, the SPARK_GUIDE_HOSTS proxy names count as local too.
    """
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
    if host not in _ALLOWED_HOSTS:
        raise HTTPException(403, "available on localhost only")
    origin = request.headers.get("origin")
    if origin and origin != "null":
        o_host = re.sub(r"^https?://", "", origin.lower()).split("/")[0].rsplit(":", 1)[0]
        if o_host not in _ALLOWED_HOSTS:
            raise HTTPException(403, "cross-site request refused")


@app.get("/")
async def index():
    guide = STATIC / "guide.html"
    if guide.exists():
        return FileResponse(guide, headers={"Cache-Control": "no-store, max-age=0"})
    return PlainTextResponse("Spark Lab Runner backend is up — static/guide.html missing.")


@app.get("/static/{fname:path}")
async def static_file(fname: str):
    base = STATIC.resolve()
    target = (base / fname).resolve()
    if not str(target).startswith(str(base) + os.sep) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target, headers={"Cache-Control": "no-store, max-age=0"})


MEDIA_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
             ".gif": "image/gif", ".svg": "image/svg+xml", ".mp4": "video/mp4", ".webm": "video/webm"}


@app.get("/w25media/{folder}/{rel:path}")
async def module_media(folder: str, rel: str):
    """Serve ONLY week25/<module>/media/* and week25/<module>/.runs/* (files a lab just produced).
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


def _child_env(extra: dict[str, str]) -> dict[str, str]:
    """Env for lab subprocesses and the terminal: settings from .env.local injected, browser limited."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("SPARK_RECORD", None)                     # learners never overwrite recordings
    for name in SETTINGS:
        if not env.get(name) and sparkkit.cfg(name):
            env[name] = sparkkit.cfg(name)
    for k in RUN_ENV_KEYS:
        v = extra.get(k)
        if v is not None:
            env[k] = str(v)
    if env.get("SPARK_MODE") not in ("live", "dry"):
        env.pop("SPARK_MODE", None)
    if env.get("SPARK_ALLOW_LAPTOP") not in ("0", "1"):
        env.pop("SPARK_ALLOW_LAPTOP", None)
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
    env = _child_env(req.env or {})
    mode = env.get("SPARK_MODE") or ("live" if _STATUS.get("a", {}).get("reachable") else "dry")

    async def body():
        if _run_lock.locked():
            yield "⚠  another lab is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            start = time.time()
            yield f"$ {Path(PY).name} week25/{req.folder}/{req.file}   [{mode.upper()}]\n\n"
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


# ── inline "⚡ Ask the Spark" proxy ────────────────────────────────────────────
class ChatRequest(BaseModel):
    target: str = "ollama"
    which: str = "a"
    model: str = ""
    messages: list[dict]
    max_tokens: int | None = 256
    tools: list | None = None
    mode: str | None = None
    allow_laptop: bool | None = True


_chat_sem = asyncio.Semaphore(2)
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/@+\-]{1,160}$")


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request) -> dict:
    _local_only(request)
    if req.target not in sparkkit.PORTS or req.target in ("openwebui", "dashboard"):
        raise HTTPException(422, f"target must be one of: {', '.join(k for k in sparkkit.PORTS if k not in ('openwebui', 'dashboard'))}")
    if req.which not in ("a", "b"):
        raise HTTPException(422, "which must be 'a' or 'b'")
    if req.model and not MODEL_RE.match(req.model):
        raise HTTPException(422, "that does not look like a model id")
    if not req.messages or len(json.dumps(req.messages)) > 40_000:
        raise HTTPException(422, "send 1+ messages, under 40,000 characters")
    for m in req.messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
            raise HTTPException(422, "each message needs a role: system | user | assistant | tool")
    env_mode = req.mode if req.mode in ("live", "dry") else None
    max_tokens = max(8, min(int(req.max_tokens or 256), 2048))

    def call() -> dict:
        saved = {k: os.environ.get(k) for k in ("SPARK_MODE", "SPARK_ALLOW_LAPTOP")}
        try:
            if env_mode:
                os.environ["SPARK_MODE"] = env_mode
            os.environ["SPARK_ALLOW_LAPTOP"] = "1" if req.allow_laptop else "0"
            sparkkit._CACHE.pop("laptop", None)
            return sparkkit.chat_any(req.target, req.model or "default", req.messages, which=req.which,
                                     max_tokens=max_tokens, tools=req.tools, quiet=True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    async with _chat_sem:
        try:
            r = await asyncio.to_thread(call)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"{type(e).__name__}: {str(e)[:200]}")
    r["target"], r["which"] = req.target, req.which
    return r


# ── 🖥 Spark setup: hosts + tokens, saved server-side ─────────────────────────
class SettingRequest(BaseModel):
    name: str
    value: str = ""


LOCAL_ENV = WEEK / ".env.local"
_settings_lock = threading.Lock()


def _settings_view() -> list[dict]:
    rows = []
    for name, meta in SETTINGS.items():
        val = sparkkit.cfg(name)
        rows.append({"name": name, **meta, "set": bool(val), "source": sparkkit.cfg_source(name),
                     "value": "" if meta["secret"] else val})
    return rows


@app.get("/api/settings")
async def api_settings_get(request: Request) -> dict:
    _local_only(request)
    return {"settings": _settings_view()}


@app.post("/api/settings")
async def api_settings_set(req: SettingRequest, request: Request) -> dict:
    """Save (or with value "" remove) one setting in week25/.env.local — gitignored, mode 0600.
    Secret values are never echoed back; the response only says whether one is now set."""
    _local_only(request)
    if req.name not in SETTINGS:
        raise HTTPException(404, "unknown setting")
    value = (req.value or "").strip()
    pattern = SETTING_RE.get(req.name) or re.compile(r"^[A-Za-z0-9._\-:+/=]{8,400}$")
    if value and not pattern.match(value):
        raise HTTPException(422, "unexpected characters or length for this setting")
    with _settings_lock:
        lines = []
        if LOCAL_ENV.is_file():
            lines = [ln for ln in LOCAL_ENV.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.startswith("#")
                     and not re.match(rf"\s*(?:export\s+)?{req.name}\s*=", ln)]
        if value:
            lines.append(f"{req.name}={value}")
        LOCAL_ENV.write_text("# Saved by the Spark Lab Runner 🖥 dialog. Gitignored — never commit this file.\n"
                             + "".join(ln + "\n" for ln in lines), encoding="utf-8")
        try:
            os.chmod(LOCAL_ENV, 0o600)
        except OSError:
            pass
    sparkkit._CACHE.clear()
    _STATUS["checked"] = 0.0
    return {"ok": True, "name": req.name, "set": bool(sparkkit.cfg(req.name))}


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


# ── built-in terminal: this Mac, or Spark A / B over ssh ──────────────────────
SHELL_TIMEOUT = {"mac": 600.0, "a": 3600.0, "b": 3600.0}
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
    """argv + a one-line description of where it runs."""
    if target == "mac" or (target == "a" and sparkkit.on_spark()):
        return [SHELL_BIN, "-lc", cmd], "this machine"
    h = sparkkit.host(target)
    if not h:
        raise HTTPException(422, f"Spark {target.upper()} has no ssh host yet — set it in 🖥 Spark setup")
    # HF_TOKEN / NGC_API_KEY travel inside the remote command's env (never echoed), like the lab subprocesses.
    fwd = " ".join(f"{k}={shlex.quote(sparkkit.cfg(k))}" for k in ("HF_TOKEN", "NGC_API_KEY", "NVIDIA_API_KEY")
                   if sparkkit.cfg(k))
    remote = (f"export {fwd}; " if fwd else "") + f"cd ~ && {cmd}"
    return sparkkit._ssh_base(h) + [f"bash -lc {shlex.quote(remote)}"], f"ssh {h}"


@app.post("/api/shell")
async def api_shell(req: ShellRequest, request: Request):
    _local_only(request)
    cmd = (req.cmd or "").strip()
    target = req.target if req.target in SHELL_TIMEOUT else "mac"
    env = {**_child_env(req.env or {}), "TERM": "dumb"}
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


# ── status: which Sparks answer, which endpoints are up ───────────────────────
_STATUS: dict = {"checked": 0.0}
_status_lock = asyncio.Lock()
ENDPOINT_KINDS = ("ollama", "vllm", "sglang", "trtllm", "llamacpp", "lmstudio", "litellm")


def _probe() -> dict:
    sparkkit._CACHE.clear()
    out: dict = {"checked": time.time(), "on_spark": sparkkit.on_spark()}
    for w in ("a", "b"):
        h = sparkkit.host(w)
        row = {"host": h, "reachable": False, "gpu": "", "detail": ""}
        if w == "a" and out["on_spark"]:
            row.update(host="(this machine)", reachable=True, gpu="GB10")
        elif h:
            row["reachable"] = sparkkit.reachable(w)
            row["detail"] = "ssh ok" if row["reachable"] else "ssh unreachable (is it on, and on your tailnet?)"
        out[w] = row
    eps = {}
    for w in ("a", "b"):
        if not (out[w]["reachable"] or sparkkit.cfg("SPARK_API_HOST" + ("2" if w == "b" else ""))):
            continue
        for k in ENDPOINT_KINDS:
            u = sparkkit.url(k, w)
            ids = sparkkit.models(u, timeout=1.5) if u else []
            eps[f"{w}:{k}"] = {"url": u, "up": bool(ids) or sparkkit.up(u, 1.0), "models": ids[:20]}
    out["endpoints"] = eps
    out["laptop_models"] = [m for m in sparkkit.models(sparkkit.LAPTOP_OLLAMA, timeout=1.5)][:20]
    return out


@app.get("/api/status")
async def api_status() -> dict:
    async with _status_lock:
        if time.time() - _STATUS.get("checked", 0) > 30:
            _STATUS.update(await asyncio.to_thread(_probe))
    a = _STATUS.get("a", {})
    if a.get("reachable"):
        detail = f"LIVE available — Spark A {a.get('host')} answers"
    elif a.get("host"):
        detail = f"DRY — Spark A ({a.get('host')}) is not reachable over ssh right now"
    else:
        detail = "DRY — no Spark configured yet · open 🖥 Spark setup"
    return {**_STATUS, "default_mode": "live" if a.get("reachable") else "dry", "detail": detail,
            "run_timeout": RUN_TIMEOUT, "ports": sparkkit.PORTS,
            "settings": [{"name": r["name"], "set": r["set"]} for r in _settings_view()]}


@app.post("/api/status/refresh")
async def api_status_refresh(request: Request) -> dict:
    _local_only(request)
    _STATUS["checked"] = 0.0
    return await api_status()


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    parsed = course()
    banner = ["", "  🟩  Spark Lab Runner — Week 25 · DGX Spark: fine-tune · serve · build sandboxed agents"]
    st = _probe()
    _STATUS.update(st)
    if st["a"]["reachable"]:
        banner += [f"      ✓ Spark A reachable ({st['a']['host']}) — LIVE: labs run real commands on it."]
    elif st["a"]["host"]:
        banner += [f"      ◈ Spark A ({st['a']['host']}) not reachable — DRY until it is (power on + tailscale)."]
    else:
        banner += ["      ◈ No Spark configured — DRY. Open 🖥 Spark setup in the app to add SPARK_HOST."]
    if st["b"]["host"]:
        banner += [f"      {'✓' if st['b']['reachable'] else '◈'} Spark B: {st['b']['host']}"]
    if st["laptop_models"]:
        banner += [f"      💻 laptop Ollama stand-in: {', '.join(st['laptop_models'][:3])}"]
    banner += [f"      ▤ course: {len(parsed)} modules · {sum(len(e['sections']) for e in parsed)} sections · "
               f"{sum(len(e['labs']) for e in parsed)} labs · {sum(len(e['exercises']) for e in parsed)} exercises"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set SPARK_GUIDE_PORT)."]
    if _PASSWORD:
        banner += [f"      🔒 password required (SPARK_GUIDE_PASSWORD) · also served as: "
                   f"{', '.join(_ALLOWED_HOSTS[len(_LOCAL_HOSTS):]) or '—'}"]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    # SPARK_GUIDE_BIND (e.g. this machine's Tailscale IP) adds a second listening address — only with a password.
    bind = sparkkit.cfg("SPARK_GUIDE_BIND") if _PASSWORD else ""
    if not bind:
        uvicorn.run(app, host="127.0.0.1", port=port)
    else:
        socks = []
        for addr in ("127.0.0.1", bind):
            s = socket.socket(socket.AF_INET6 if ":" in addr else socket.AF_INET)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, port))
            socks.append(s)
        print(f"      🌐 also listening on http://{bind}:{port}", flush=True)
        uvicorn.Server(uvicorn.Config(app, port=port)).run(sockets=socks)
