#!/usr/bin/env python3
"""livekit — one turn of a LIVE (real-time, streaming) voice session, with timings.

Live models listen to streaming audio and answer while the call is still open:

    gemini   gemini-3.8-live          Gemini Live API (websocket, BidiGenerateContent), server VAD
    chatgpt  gpt-realtime-2.1-mini    OpenAI Realtime API (websocket), manual commit

    from livekit import live_turn
    t = live_turn("gemini", audio="media/call_turn2_en.wav", system="You are a hotel voice assistant.")
    t["heard"]   # what the live model transcribed from the caller
    t["said"]    # what it answered (transcript of its speech, or its text)
    t["timing"]  # seconds: speech_end, first_output, total
    t["events"]  # [(t_seconds, kind, text)] — the stream, in order

Why Jev belongs here: a live model TALKS while it thinks. Jev can judge each
turn's transcript (emergency? promise we can't keep? needs a human?) in well
under a second, in parallel, and code decides what really happens — the live
model never gets to dispatch staff or grant anything by itself.

Needs the `websockets` package (already in the repo .venv). DRY mode
(JEV_MODE=dry or no key) replays a recorded turn — same events, same timings.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llmkit  # noqa: E402

RECORDED = Path(__file__).resolve().parent / "recorded_live"
MODELS = {"gemini": "gemini-3.8-live", "chatgpt": "gpt-realtime-2.1-mini"}
GEMINI_WS = ("wss://generativelanguage.googleapis.com/ws/"
             "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}")
OPENAI_WS = "wss://api.openai.com/v1/realtime?model={model}"


class LiveError(RuntimeError):
    pass


def capable() -> list[str]:
    return [p for p in MODELS if llmkit.available(p)]


def _pcm(audio: Path, rate: int) -> bytes:
    """Any audio file → mono 16-bit PCM at `rate` (ffmpeg if available, else a matching WAV)."""
    try:
        return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(audio), "-ac", "1", "-ar", str(rate),
                               "-f", "s16le", "-"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        with wave.open(str(audio)) as w:
            if w.getframerate() != rate or w.getnchannels() != 1:
                raise LiveError(f"install ffmpeg, or supply a {rate} Hz mono WAV") from None
            return w.readframes(w.getnframes())


def _rid(provider, model, system, audio: Path | None, text: str | None) -> str:
    h = hashlib.sha256(audio.read_bytes()).hexdigest()[:16] if audio else text
    return hashlib.sha256(json.dumps([provider, model, system, h]).encode()).hexdigest()[:20]


async def _gemini(model: str, system: str, audio: Path | None, text: str | None, pace: float) -> dict:
    import websockets
    key, _ = llmkit.api_key("gemini")
    ev, heard, said = [], "", ""
    t0 = time.perf_counter()
    stamp = lambda: round(time.perf_counter() - t0, 2)
    speech_end = first = None
    async with websockets.connect(GEMINI_WS.format(key=key), max_size=None) as ws:
        await ws.send(json.dumps({"setup": {
            "model": f"models/{model}", "generationConfig": {"responseModalities": ["AUDIO"]},
            "systemInstruction": {"parts": [{"text": system}]},
            "inputAudioTranscription": {}, "outputAudioTranscription": {}}}))
        await ws.recv()                                             # setupComplete
        ev.append((stamp(), "session", "setup complete"))
        if audio:
            speech = _pcm(audio, 16000)
            pcm = speech + b"\x00\x00" * 24000                          # + 1.5 s silence → VAD sees the end
            chunk = 3200                                              # 100 ms
            for i in range(0, len(pcm), chunk):
                await ws.send(json.dumps({"realtimeInput": {"audio": {
                    "data": base64.b64encode(pcm[i:i + chunk]).decode(), "mimeType": "audio/pcm;rate=16000"}}}))
                if speech_end is None and i + chunk >= len(speech):
                    speech_end = stamp()                              # the caller's last word has been sent
                    ev.append((speech_end, "caller_stopped", "last word of speech streamed"))
                await asyncio.sleep(0.1 * pace)
            await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
        else:
            await ws.send(json.dumps({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}],
                                                        "turnComplete": True}}))
            speech_end = stamp()
        while True:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            except asyncio.TimeoutError:
                ev.append((stamp(), "timeout", "no turnComplete within 45 s"))
                break
            va = m.get("voiceActivity") or {}
            if va.get("type") == "ACTIVITY_END":
                ev.append((stamp(), "vad_end", "server detected end of speech"))
            sc = m.get("serverContent") or {}
            if sc.get("inputTranscription", {}).get("text"):
                heard += sc["inputTranscription"]["text"]
                ev.append((stamp(), "heard", sc["inputTranscription"]["text"]))
            if sc.get("outputTranscription", {}).get("text"):
                said += sc["outputTranscription"]["text"]
                ev.append((stamp(), "said", sc["outputTranscription"]["text"]))
            if any(p.get("inlineData") for p in (sc.get("modelTurn") or {}).get("parts", [])) and first is None:
                first = stamp()
                ev.append((first, "first_audio", "model started speaking"))
            if sc.get("turnComplete"):
                ev.append((stamp(), "turn_complete", ""))
                break
    return {"heard": heard.strip(), "said": said.strip(), "events": ev,
            "timing": {"speech_end": speech_end, "first_output": first, "total": stamp()}}


async def _openai(model: str, system: str, audio: Path | None, text: str | None, pace: float) -> dict:
    import websockets
    key, _ = llmkit.api_key("chatgpt")
    ev, heard, said = [], "", ""
    t0 = time.perf_counter()
    stamp = lambda: round(time.perf_counter() - t0, 2)
    speech_end = first = None
    async with websockets.connect(OPENAI_WS.format(model=model), max_size=None,
                                  additional_headers={"Authorization": f"Bearer {key}"}) as ws:
        await ws.recv()                                             # session.created
        sess = {"type": "realtime", "instructions": system, "output_modalities": ["text"]}
        if audio:
            sess["audio"] = {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                       "transcription": {"model": "gpt-4o-mini-transcribe"}, "turn_detection": None}}
        await ws.send(json.dumps({"type": "session.update", "session": sess}))
        ev.append((stamp(), "session", "session updated"))
        if audio:
            pcm = _pcm(audio, 24000)
            for i in range(0, len(pcm), 4800):                      # 100 ms
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(pcm[i:i + 4800]).decode()}))
                await asyncio.sleep(0.1 * pace)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        else:
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}))
        speech_end = stamp()
        ev.append((speech_end, "caller_stopped", "audio committed"))
        await ws.send(json.dumps({"type": "response.create"}))
        while True:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            except asyncio.TimeoutError:
                ev.append((stamp(), "timeout", "no response.done within 45 s"))
                break
            t = m.get("type")
            if t == "error":
                raise LiveError(json.dumps(m.get("error", m))[:300])
            if t == "conversation.item.input_audio_transcription.completed":
                heard = m.get("transcript", "")
                ev.append((stamp(), "heard", heard))
            if t == "response.output_text.delta":
                if first is None:
                    first = stamp()
                    ev.append((first, "first_output", "model started answering"))
                said += m.get("delta", "")
            if t == "response.done":
                ev.append((stamp(), "said", said))
                ev.append((stamp(), "turn_complete", ""))
                break
    return {"heard": heard.strip(), "said": said.strip(), "events": ev,
            "timing": {"speech_end": speech_end, "first_output": first, "total": stamp()}}


def live_turn(provider: str | None = None, *, audio: str | Path | None = None, text: str | None = None,
              system: str = "You are a hotel front-desk voice assistant. Reply in one or two short sentences.",
              model: str | None = None, pace: float = 1.0) -> dict:
    """Run ONE caller turn through a live model. pace=1.0 streams in real time (what a phone call does)."""
    if (audio is None) == (text is None):
        raise LiveError("pass exactly one of audio= or text=")
    audio = Path(audio) if audio else None
    if provider is None:
        pref = os.environ.get("JEV_LLM_PROVIDER", "")
        provider = pref if pref in MODELS and llmkit.available(pref) else (capable() or ["gemini"])[0]
    if provider not in MODELS:
        raise LiveError(f"live sessions in this course support: {', '.join(MODELS)}")
    model = model or MODELS[provider]
    rid = _rid(provider, model, system, audio, text)
    cache = RECORDED / f"{rid}.json"
    if os.environ.get("JEV_MODE", "").lower() == "dry" or not llmkit.available(provider):
        if cache.is_file():
            r = json.loads(cache.read_text(encoding="utf-8"))["response"]
            return {**r, "_source": "recorded"}
        return {"heard": "", "said": "[DRY — no recorded live turn for this exact audio/system prompt]",
                "events": [], "timing": {}, "provider": provider, "model": model, "_source": "placeholder"}
    try:
        runner = _gemini if provider == "gemini" else _openai
        r = asyncio.run(runner(model, system, audio, text, pace))
    except ImportError:
        raise LiveError("the live labs need the `websockets` package: .venv/bin/pip install websockets") from None
    except Exception as e:  # noqa: BLE001 — surface a readable error, never a key
        if isinstance(e, LiveError):
            raise
        raise LiveError(f"{type(e).__name__}: {str(e)[:240]}") from None
    r.update(provider=provider, model=model, _source="live")
    if os.environ.get("JEV_RECORD") == "1":
        RECORDED.mkdir(exist_ok=True)
        keep = {k: v for k, v in r.items() if not k.startswith("_")}
        cache.write_text(json.dumps({"recorded_at": time.strftime("%Y-%m-%d"),
                                     "request": {"provider": provider, "model": model, "system": system,
                                                 "audio": audio.name if audio else None, "text": text},
                                     "response": keep}, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def show(t: dict) -> None:
    tag = {"live": "LIVE", "recorded": "RECORDED replay", "placeholder": "PLACEHOLDER"}.get(t.get("_source"), "?")
    for ts, kind, txt in t.get("events", []):
        if kind in ("heard", "said") and not txt.strip():
            continue
        glyph = {"heard": "←", "said": "→", "first_audio": "◆", "first_output": "◆", "caller_stopped": "│",
                 "session": "│", "turn_complete": "│", "timeout": "⚠"}.get(kind, "│")
        print(f"{glyph} {ts:6.2f}s  {kind:<15} {txt[:90]}")
    tm = t.get("timing") or {}
    if tm.get("first_output") is not None and tm.get("speech_end") is not None:
        print(f"◆ {t.get('provider')} · {t.get('model')} · {tag} · caller stopped at {tm['speech_end']:.2f}s → "
              f"model answered at {tm['first_output']:.2f}s (gap {tm['first_output'] - tm['speech_end']:.2f}s) · total {tm['total']:.2f}s")
    else:
        print(f"◆ {t.get('provider')} · {t.get('model')} · {tag}")
