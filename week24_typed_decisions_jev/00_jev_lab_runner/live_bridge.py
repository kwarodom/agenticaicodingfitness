"""Live console bridge — your laptop mic + speaker + webcam ⇄ gemini-3.8-live, with Jev refereeing.

    browser (mic 16 kHz PCM, webcam JPEG ~1 fps, speaker 24 kHz PCM)
        ⇅  ws://127.0.0.1:8124/ws/live        (this file — localhost only, Host + Origin checked)
    Gemini Live API (gemini-3.8-live, BidiGenerateContent)   ← the key stays HERE, never in the browser
        │ every finished turn: heard (you) + said (assistant)
        ▼
    Jev (Module 14 referee.py): emergency · needs a person · promised an action · invented a fact · language
        ▼
    code: call state machine (calm → handoff / escalated) → proposed actions (execute: False)
          optional auto-correct: tell the live model to say the approved correction line
    optional "Jev watches the camera": every 10 s a vision model describes the latest frame,
          Jev judges it for hazards — an independent second opinion, not the live model's word.

Browser → server messages (JSON): start · audio · frame · text · lang · stop
Server → browser messages (JSON): status · audio · heard · said · interrupted · turn · jev · watch · lang · usage · error
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path

from starlette.websockets import WebSocket, WebSocketDisconnect

PKG = Path(__file__).resolve().parent
WEEK = PKG.parent
MOD14 = WEEK / "14_live_realtime_jev"
RUNS = MOD14 / ".runs"
sys.path.insert(0, str(WEEK / "common"))
sys.path.insert(0, str(MOD14))
import jevkit  # noqa: E402
import livekit  # noqa: E402
import llmkit  # noqa: E402
import mediakit  # noqa: E402
import referee as R  # noqa: E402  (Module 14's questions, prompts and state machine)

MODEL = "gemini-3.8-live"
MAX_SECONDS = 10 * 60                 # hard cap per session (cost + privacy)
WATCH_EVERY = 10.0                    # seconds between "Jev watches the camera" checks
VOICES = ("Kore", "Puck", "Charon", "Aoede", "Fenrir", "Leda", "Orus", "Zephyr")
LOCAL = ("127.0.0.1", "localhost", "[::1]", "::1")
_active = asyncio.Lock()              # one live session at a time on this machine

CAMERA_NOTE = ("You can also see the caller's webcam. If they ask, describe what you see briefly and literally. "
               "Never claim you can act on what you see.")
HAZARD_Q = jevkit.guarded({
    "hazard": jevkit.noul("Does `camera_description` report a possible safety hazard visible now — smoke, fire, "
                          "flames, sparks, water on the floor, a person injured or lying down?"),
    "unclear": jevkit.noul("Does `camera_description` say the image is dark, blurry, blocked or unclear?"),
})
WATCH_PROMPT = ("You are a safety camera checker. In one or two sentences, describe any safety hazard visible "
                "(smoke, fire, sparks, water on the floor, an injured person). If nothing unusual, say 'nothing unusual'.")
LEAVE_OK = re.compile(r"leave|get out|evacuat|call 0|ออกจากห้อง|โทร 0", re.I)

# ── language: Auto / English / Thai, switchable mid-call ──────────────────────
# Verified live: a supervisor instruction switches gemini-3.8-live's reply language
# mid-session in both directions; "English only" is a hard lock, so Auto must say
# "follow the caller". A th-TH languageCode at setup biased later replies to Thai,
# so we steer by instruction only.
FEMALE_VOICES = {"Kore", "Aoede", "Leda", "Zephyr"}          # → polite particle ค่ะ; the others → ครับ
LANG_RULE = {
    "auto": "LANGUAGE: reply in the language the caller is speaking right now (English or Thai). If they switch, switch with them.",
    "en": "LANGUAGE: speak only English, even if the caller speaks another language.",
    "th": "LANGUAGE: speak only Thai, politely, ending sentences with {p}. Even if the caller speaks English, answer in Thai.",
}
LANG_SWITCH = {
    "auto": "From now on reply in whichever language the caller speaks (English or Thai). Acknowledge in one or two words.",
    "en": "From now on speak only English. Acknowledge with one word.",
    "th": "From now on speak only Thai, politely, ending sentences with {p}. Acknowledge in Thai in one or two words.",
}
LINES_TH = {   # Thai versions of Module 14's approved lines. {p}: statement particle (ค่ะ/ครับ);
               # {pn}: the particle after นะ, which is คะ — not ค่ะ — for a female voice (นะคะ / นะครับ)
    R.SAFETY_LINE: "กรุณาออกจากห้องทันที แล้วโทร 0 จากที่ปลอดภัยนะ{pn} ระบบกำลังแจ้งเจ้าหน้าที่แล้ว{p}",
    R.CORRECTION_PROMISE: "ขอแก้ไขนะ{pn} ไม่สามารถส่งเจ้าหน้าที่ไปเองได้ แต่ระบบได้แจ้งเพื่อนร่วมงานแล้ว และจะโทรกลับหาคุณเร็ว ๆ นี้{p}",
    R.CORRECTION_FACT: "ขอแก้ไขนะ{pn} ข้อมูลนี้ยังไม่ได้รับการยืนยัน ขอโอนสายไปที่แผนกต้อนรับนะ{pn}",
}
SUPERVISOR = "[Supervisor system message — not the caller] "


def _particle(voice: str) -> str:
    return "ค่ะ" if voice in FEMALE_VOICES else "ครับ"


def _localize(line: str, lang: str, voice: str) -> str:
    if lang != "th":
        return line
    female = voice in FEMALE_VOICES
    return LINES_TH.get(line, line).format(p="ค่ะ" if female else "ครับ", pn="คะ" if female else "ครับ")


def _allowed(ws: WebSocket) -> bool:
    host = (ws.headers.get("host") or "").rsplit(":", 1)[0].lower()
    origin = ws.headers.get("origin") or ""
    o_host = re.sub(r"^https?://", "", origin.lower()).split("/")[0].rsplit(":", 1)[0] if origin else ""
    return host in LOCAL and (not origin or o_host in LOCAL)


async def live_ws(ws: WebSocket):
    if not _allowed(ws):
        await ws.close(code=4403)
        return
    await ws.accept()
    send = lambda obj: ws.send_text(json.dumps(obj, ensure_ascii=False))
    key, _ = llmkit.api_key("gemini")
    if not key or not jevkit.api_key():
        await send({"type": "error", "detail": "The live console needs GEMINI_API_KEY and TYPESAFE_API_KEY "
                                               "(repo .env or the 🔑 Keys dialog)."})
        await ws.close()
        return
    if _active.locked():
        await send({"type": "error", "detail": "Another live session is already running on this machine — stop it first."})
        await ws.close()
        return
    async with _active:
        try:
            cfg = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=30))
        except Exception:  # noqa: BLE001
            await ws.close()
            return
        if cfg.get("type") != "start" or not cfg.get("consent"):
            await send({"type": "error", "detail": "Tick the consent box first: your mic/camera stream goes to Google Gemini."})
            await ws.close()
            return
        await _session(ws, send, key, cfg)


async def _session(ws: WebSocket, send, key: str, cfg: dict):
    import websockets
    preset = "naive" if cfg.get("preset") == "naive" else "safe"
    voice = cfg.get("voice") if cfg.get("voice") in VOICES else "Kore"
    camera = bool(cfg.get("camera"))
    use_referee = cfg.get("referee", True) is not False
    autocorrect = bool(cfg.get("autocorrect", True)) and use_referee
    watch = bool(cfg.get("watch")) and camera
    lang = cfg.get("lang") if cfg.get("lang") in LANG_RULE else "auto"
    call_ref = time.strftime("LIVE-%Y%m%d-%H%M%S")
    system = R.system_for(R.SAFE_SYSTEM if preset == "safe" else R.NAIVE_SYSTEM, call_ref)
    if camera:
        system += "\n" + CAMERA_NOTE
    system += "\n" + LANG_RULE[lang].format(p=_particle(voice))
    RUNS.mkdir(exist_ok=True)
    log_path = RUNS / f"{call_ref}.jsonl"
    frame_path = RUNS / "live_latest_frame.jpg"
    t0 = time.perf_counter()
    stamp = lambda: round(time.perf_counter() - t0, 2)
    st = {"state": "calm", "turn": 0, "heard": "", "said": "", "speech_end": None, "first_audio": None,
          "last_frame": None, "frames": 0, "watch_busy": False, "last_watch": 0.0, "stop": False,
          "tokens": 0, "corrections": 0, "lang": lang, "system_turns": 0, "speaking": False, "queue": []}

    def log(kind: str, **kw):
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": stamp(), "kind": kind, **kw}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    setup = {"setup": {
        "model": f"models/{MODEL}",
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}},
        "systemInstruction": {"parts": [{"text": system}]},
        "inputAudioTranscription": {}, "outputAudioTranscription": {},
        "contextWindowCompression": {"slidingWindow": {}},       # audio+video sessions stay open longer
    }}
    await send({"type": "status", "state": "connecting", "detail": f"opening {MODEL} · voice {voice} · {preset} prompt"})
    try:
        async with websockets.connect(livekit.GEMINI_WS.format(key=key), max_size=None) as gws:
            await gws.send(json.dumps(setup))
            await asyncio.wait_for(gws.recv(), timeout=20)          # setupComplete
            await send({"type": "status", "state": "live", "call_ref": call_ref, "model": MODEL, "preset": preset, "lang": lang,
                        "max_seconds": MAX_SECONDS, "log": str(log_path.relative_to(WEEK.parent))})
            log("start", model=MODEL, preset=preset, voice=voice, camera=camera, watch=watch, lang=lang)

            async def _send_supervisor(text: str):
                st["system_turns"] += 1
                st["speaking"] = True                                # it will answer — hold further messages
                await gws.send(json.dumps({"clientContent": {"turns": [{"role": "user", "parts": [{"text": SUPERVISOR + text}]}],
                                                             "turnComplete": True}}))

            async def _send_user_text(text: str):
                st["heard"] += (" " if st["heard"] else "") + text   # attributed when actually sent
                st["speech_end"] = stamp()
                st["speaking"] = True
                await gws.send(json.dumps({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}],
                                                             "turnComplete": True}}))

            async def supervise(text: str):
                """Tell the live model something as the system, not the caller. If it is mid-answer,
                queue the message until its turn completes, so it never cuts itself off mid-sentence."""
                if st["speaking"]:
                    st["queue"].append(("system", text))
                else:
                    await _send_supervisor(text)

            async def flush_one():
                if st["queue"] and not st["speaking"]:
                    who, text = st["queue"].pop(0)
                    await (_send_supervisor(text) if who == "system" else _send_user_text(text))

            async def referee_turn(n: int, heard: str, said: str, gap):
                try:
                    r = await asyncio.to_thread(R.referee, heard, said)
                    a = r["answers"]
                    new_state, actions = R.next_state(st["state"], a, heard)
                    changed = new_state != st["state"]
                    st["state"] = new_state
                    corrected = []
                    if autocorrect and not st["stop"]:
                        lines = [x[5:] for x in actions if x.startswith("say: Correction")]
                        if changed and new_state == "escalated" and not LEAVE_OK.search(said or ""):
                            lines.insert(0, R.SAFETY_LINE)
                        say_lang = st["lang"] if st["lang"] != "auto" else (
                            "th" if a["caller_language"]["choice"] == "th" else "en")
                        for line in lines[:1]:                     # at most one injected line per turn
                            line = _localize(line, say_lang, voice)
                            await supervise("Say exactly this to the caller now, then stop: " + line)
                            corrected.append(line)
                            st["corrections"] += 1
                    msg = {"type": "jev", "turn": n, "jev_ms": r["jev_ms"], "state": st["state"], "changed": changed,
                           "actions": actions, "corrected": corrected,
                           "scores": {k: a[k]["noul"] for k in ("caller_emergency", "needs_human",
                                                                 "assistant_promised_action", "assistant_invented_fact")},
                           "language": a["caller_language"]["choice"], "gap": gap}
                    await send(msg)
                    log("jev", **{k: v for k, v in msg.items() if k != "type"})
                except Exception as e:  # noqa: BLE001 — the referee must never kill the call
                    await send({"type": "error", "detail": f"Jev referee failed on turn {n}: {type(e).__name__}: {str(e)[:160]}"})

            async def watch_camera():
                st["watch_busy"] = True
                try:
                    frame_path.write_bytes(st["last_frame"])
                    seen = await asyncio.to_thread(mediakit.see, "gemini", frame_path, WATCH_PROMPT, max_tokens=160)
                    desc = seen.get("text", "")
                    t_j = time.perf_counter()
                    resp = await asyncio.to_thread(jevkit.ask, {"camera_description": desc}, HAZARD_Q, quiet=True)
                    a = jevkit.validate(resp, HAZARD_Q)
                    msg = {"type": "watch", "t": stamp(), "description": desc, "hazard": a["hazard"]["noul"],
                           "unclear": a["unclear"]["noul"], "vision_ms": seen.get("latency_ms"),
                           "jev_ms": round((time.perf_counter() - t_j) * 1000)}
                    await send(msg)
                    log("watch", **{k: v for k, v in msg.items() if k != "type"})
                except Exception as e:  # noqa: BLE001
                    await send({"type": "error", "detail": f"camera check skipped: {type(e).__name__}: {str(e)[:140]}"})
                finally:
                    st["watch_busy"] = False

            async def from_browser():
                while not st["stop"]:
                    if time.perf_counter() - t0 > MAX_SECONDS:
                        await send({"type": "status", "state": "stopping", "detail": f"{MAX_SECONDS // 60}-minute cap reached"})
                        break
                    try:
                        raw = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    m = json.loads(raw)
                    kind = m.get("type")
                    if kind == "stop":
                        break
                    if kind == "audio" and m.get("data"):
                        await gws.send(json.dumps({"realtimeInput": {"audio": {"data": m["data"],
                                                                                "mimeType": "audio/pcm;rate=16000"}}}))
                    elif kind == "frame" and camera and m.get("data"):
                        if len(m["data"]) > 400_000:                  # ~300 KB JPEG cap per frame
                            continue
                        st["frames"] += 1
                        await gws.send(json.dumps({"realtimeInput": {"video": {"data": m["data"], "mimeType": "image/jpeg"}}}))
                        if watch:
                            st["last_frame"] = base64.b64decode(m["data"])
                            if not st["watch_busy"] and time.perf_counter() - st["last_watch"] >= WATCH_EVERY:
                                st["last_watch"] = time.perf_counter()
                                asyncio.create_task(watch_camera())
                    elif kind == "lang" and m.get("lang") in LANG_SWITCH and m["lang"] != st["lang"]:
                        st["lang"] = m["lang"]
                        await supervise(LANG_SWITCH[st["lang"]].format(p=_particle(voice)))
                        await send({"type": "lang", "lang": st["lang"]})
                        log("lang", lang=st["lang"])
                    elif kind == "text" and (m.get("text") or "").strip():
                        text = m["text"].strip()[:1000]
                        if st["speaking"] or st["queue"]:              # keep typed turns in order
                            st["queue"].append(("user", text))
                        else:
                            await _send_user_text(text)
                st["stop"] = True

            async def from_gemini():
                while not st["stop"]:
                    try:
                        m = json.loads(await asyncio.wait_for(gws.recv(), timeout=1.0))
                    except asyncio.TimeoutError:
                        continue
                    if m.get("goAway"):
                        await send({"type": "status", "state": "stopping", "detail": "Gemini is closing the session (goAway)"})
                    if (m.get("voiceActivity") or {}).get("type") == "ACTIVITY_END":
                        st["speech_end"] = stamp()
                    sc = m.get("serverContent") or {}
                    if sc.get("inputTranscription", {}).get("text"):
                        txt = sc["inputTranscription"]["text"]
                        st["heard"] += txt
                        await send({"type": "heard", "text": txt})
                    if sc.get("outputTranscription", {}).get("text"):
                        txt = sc["outputTranscription"]["text"]
                        st["said"] += txt
                        await send({"type": "said", "text": txt})
                    if sc.get("modelTurn"):
                        st["speaking"] = True
                    for p in (sc.get("modelTurn") or {}).get("parts", []):
                        if p.get("inlineData", {}).get("data"):
                            if st["first_audio"] is None:
                                st["first_audio"] = stamp()
                            await send({"type": "audio", "data": p["inlineData"]["data"]})
                    if sc.get("interrupted"):                         # the caller barged in
                        st["speaking"] = False
                        await send({"type": "interrupted"})
                    if m.get("usageMetadata"):
                        st["tokens"] = m["usageMetadata"].get("totalTokenCount", st["tokens"])
                        await send({"type": "usage", "tokens": st["tokens"]})
                    if sc.get("turnComplete"):
                        st["turn"] += 1
                        heard, said = st["heard"].strip(), st["said"].strip()
                        gap = (round(st["first_audio"] - st["speech_end"], 2)
                               if st["first_audio"] is not None and st["speech_end"] is not None else None)
                        system_turn = st["system_turns"] > 0 and not heard
                        if system_turn:                              # the model answering OUR instruction, not the caller
                            st["system_turns"] -= 1
                        await send({"type": "turn", "turn": st["turn"], "heard": heard, "said": said, "gap": gap,
                                    "system": system_turn})
                        log("turn", turn=st["turn"], heard=heard, said=said, gap=gap, system=system_turn)
                        if use_referee and (heard or said) and not system_turn:
                            asyncio.create_task(referee_turn(st["turn"], heard, said, gap))
                        st.update(heard="", said="", first_audio=None, speech_end=None, speaking=False)
                        await flush_one()                            # send the next held message, if any

            tasks = [asyncio.create_task(from_browser()), asyncio.create_task(from_gemini())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            st["stop"] = True
            for t in pending:
                t.cancel()
            for t in done:
                if t.exception() and not isinstance(t.exception(), (WebSocketDisconnect, asyncio.CancelledError)):
                    raise t.exception()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 — report readable errors, never keys
        msg = str(e).replace(key, "***")[:240]
        try:
            await send({"type": "error", "detail": f"live session ended: {type(e).__name__}: {msg}"})
        except Exception:  # noqa: BLE001
            pass
    finally:
        log("stop", seconds=stamp(), turns=st["turn"], frames=st["frames"], tokens=st["tokens"],
            corrections=st["corrections"], final_state=st["state"])
        try:
            frame_path.unlink(missing_ok=True)                        # never keep a webcam frame on disk
        except OSError:
            pass
        try:
            await send({"type": "status", "state": "stopped", "seconds": stamp(), "turns": st["turn"],
                        "frames": st["frames"], "tokens": st["tokens"], "final_state": st["state"]})
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
