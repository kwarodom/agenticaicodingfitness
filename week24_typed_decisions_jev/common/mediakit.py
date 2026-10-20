#!/usr/bin/env python3
"""mediakit — see, hear, speak and watch with multimodal models, then let Jev judge the text.

Jev reads TEXT (and JSON). Multimodal models turn pixels, sound and video into
text; Jev turns that text into typed, cheap, fast decisions; code decides.

    from mediakit import see, hear, speak, watch
    desc = see("gemini", "media/thermostat.png", "Read every number and label on the display.")
    text = hear("chatgpt", "media/voicemail.wav")                 # speech → transcript
    speak("chatgpt", "We are sending an engineer now.", ".runs/reply.wav")
    events = watch("media/cctv.mp4", "List visible events with timestamps as JSON.")

Capability map (verified live on 2026-09-27 with the keys in this repo):

    capability      provider   default model                 API
    see (image→text) gemini     gemini-3.5-flash              OpenAI-compatible chat, image_url data URI
                     chatgpt    gpt-5.4-mini                  chat completions, image_url
                     claude     claude-sonnet-5               Messages, base64 image block
                     ollama     gemma4:latest (local, $0)     OpenAI-compatible, image_url
    hear (audio→text) chatgpt   gpt-4o-mini-transcribe        /audio/transcriptions (multipart)
                     gemini     gemini-3.5-flash              generateContent, inline audio
    speak (text→audio) chatgpt  gpt-4o-mini-tts               /audio/speech → wav
                     gemini     gemini-3.8-flash-tts          generateContent AUDIO → PCM → wav
    watch (video→text) gemini   gemini-3.8-flash              generateContent, inline mp4
    live (streaming)   see livekit.py (gemini-3.8-live · gpt-realtime-2.1-mini)

Modes follow jevkit/llmkit: JEV_MODE=dry (or a missing key) replays a recording
of the exact same request (model + prompt + SHA-256 of the media file); if none
exists you get a clearly-labelled placeholder. JEV_RECORD=1 records. Keys come
from llmkit (env → week24_typed_decisions_jev/.env.local → repo .env) and are never printed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import struct
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llmkit  # noqa: E402

COMMON = Path(__file__).resolve().parent
RECORDED = COMMON / "recorded_media"
GEMINI_NATIVE = "https://generativelanguage.googleapis.com/v1beta"

DEFAULTS = {
    "see": {"gemini": "gemini-3.5-flash", "chatgpt": "gpt-5.4-mini", "claude": "claude-sonnet-5",
            "ollama": "gemma4:latest", "openrouter": "google/gemini-3.5-flash"},
    "hear": {"chatgpt": "gpt-4o-mini-transcribe", "gemini": "gemini-3.5-flash"},
    "speak": {"chatgpt": "gpt-4o-mini-tts", "gemini": "gemini-3.8-flash-tts"},
    "watch": {"gemini": "gemini-3.8-flash"},
}


class MediaError(RuntimeError):
    pass


def capable(capability: str) -> list[str]:
    """Providers that can do `capability` AND have a key (or need none) on this machine."""
    return [p for p in DEFAULTS[capability] if llmkit.available(p)]


def pick(capability: str, preferred: str | None = None) -> str:
    """The learner's LLM picker choice if it can do this; otherwise the first capable provider."""
    pref = preferred or os.environ.get("JEV_LLM_PROVIDER", "")
    if pref in DEFAULTS[capability] and llmkit.available(pref):
        return pref
    ok = capable(capability)
    if not ok:
        raise MediaError(f"no provider with a key can '{capability}' — add one of: {', '.join(DEFAULTS[capability])}")
    return ok[0]


# ── helpers ───────────────────────────────────────────────────────────────────
def _mime(path: Path) -> str:
    m = mimetypes.guess_type(str(path))[0]
    return m or {".wav": "audio/wav", ".mp4": "video/mp4"}.get(path.suffix.lower(), "application/octet-stream")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rid(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:20]


def _live(provider: str) -> bool:
    return os.environ.get("JEV_MODE", "").lower() != "dry" and llmkit.available(provider)


def _replay(rid: str, capability: str, provider: str, model: str):
    f = RECORDED / f"{rid}.json"
    if f.is_file():
        r = json.loads(f.read_text(encoding="utf-8"))["response"]
        r["_source"] = "recorded"
        return r
    return {"text": f"[DRY — no recorded {capability} answer for this exact request; add a {provider} key "
                    f"and switch to ⚡ Live]", "provider": provider, "model": model, "_source": "placeholder"}


def _record(rid: str, request: dict, r: dict) -> None:
    if os.environ.get("JEV_RECORD") == "1":
        RECORDED.mkdir(exist_ok=True)
        keep = {k: v for k, v in r.items() if not k.startswith("_")}
        (RECORDED / f"{rid}.json").write_text(json.dumps({"recorded_at": time.strftime("%Y-%m-%d"), "request": request,
                                                          "response": keep}, ensure_ascii=False, indent=1), encoding="utf-8")


def _http(url: str, body: bytes, headers: dict, timeout: float = 120) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(Request(url, data=body, method="POST", headers={"User-Agent": "week24-jev-lab/1", **headers}),
                         timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise MediaError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
        except (URLError, TimeoutError) as e:
            if attempt == 0:                                  # one retry: a stuck request often succeeds at once
                continue
            raise MediaError(f"network/timeout after a retry: {e}") from None
    raise MediaError("gave up after 3 attempts")


def _gemini_generate(model: str, parts: list, config: dict | None = None, timeout: float = 180) -> dict:
    key, _ = llmkit.api_key("gemini")
    body = {"contents": [{"role": "user", "parts": parts}]}
    if config:
        body["generationConfig"] = config
    raw = _http(f"{GEMINI_NATIVE}/models/{model}:generateContent", json.dumps(body).encode(),
                {"x-goog-api-key": key, "Content-Type": "application/json"}, timeout)
    return json.loads(raw)


def _gemini_text(d: dict) -> str:
    parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


def _gemini_usage(d: dict) -> dict:
    u = d.get("usageMetadata") or {}
    return {"input_tokens": u.get("promptTokenCount", 0), "output_tokens": u.get("candidatesTokenCount", 0)}


# ── see: image → text ─────────────────────────────────────────────────────────
def see(provider: str | None, image: str | Path, prompt: str, *, model: str | None = None,
        max_tokens: int = 700) -> dict:
    image = Path(image)
    provider = provider or pick("see")
    model = model or DEFAULTS["see"][provider]
    rid = _rid("see", provider, model, prompt, _sha(image), max_tokens)
    if not _live(provider):
        return _replay(rid, "see", provider, model)
    b64 = base64.b64encode(image.read_bytes()).decode()
    mime = _mime(image)
    t0 = time.perf_counter()
    p = llmkit.PROVIDERS[provider]
    key, _ = llmkit.api_key(provider)
    if p["kind"] == "anthropic":
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": prompt}]}]}
        d = json.loads(_http(llmkit.base_url(provider) + "/messages", json.dumps(body).encode(),
                             {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}))
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage") or {}
        usage = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}
    else:
        body = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
            p.get("tokens_field", "max_tokens"): max_tokens}
        body.update(p.get("extra") or {})
        d = json.loads(_http(llmkit.base_url(provider) + "/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}))
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        u = d.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
    r = {"text": text.strip(), "provider": provider, "model": d.get("model", model), "usage": usage,
         "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "_source": "live"}
    _record(rid, {"capability": "see", "provider": provider, "model": model, "prompt": prompt, "media": image.name}, r)
    return r


# ── hear: speech → transcript ─────────────────────────────────────────────────
def hear(provider: str | None, audio: str | Path, *, model: str | None = None,
         prompt: str = "Transcribe this audio verbatim in its original language. Output only the transcript.") -> dict:
    audio = Path(audio)
    provider = provider or pick("hear")
    model = model or DEFAULTS["hear"][provider]
    rid = _rid("hear", provider, model, prompt, _sha(audio))
    if not _live(provider):
        return _replay(rid, "hear", provider, model)
    t0 = time.perf_counter()
    key, _ = llmkit.api_key(provider)
    if provider == "chatgpt":
        b = uuid.uuid4().hex
        data = (f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
                f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{audio.name}"\r\n'
                f'Content-Type: {_mime(audio)}\r\n\r\n').encode() + audio.read_bytes() + f"\r\n--{b}--\r\n".encode()
        d = json.loads(_http(llmkit.base_url("chatgpt") + "/audio/transcriptions", data,
                             {"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={b}"}))
        text, usage = d.get("text", ""), {"input_tokens": (d.get("usage") or {}).get("input_tokens", 0), "output_tokens": (d.get("usage") or {}).get("output_tokens", 0)}
        answered = model
    elif provider == "gemini":
        d = _gemini_generate(model, [{"inline_data": {"mime_type": _mime(audio), "data": base64.b64encode(audio.read_bytes()).decode()}},
                                     {"text": prompt}], {"thinkingConfig": {"thinkingBudget": 0}} if "3.5-flash" in model else None)
        text, usage, answered = _gemini_text(d), _gemini_usage(d), d.get("modelVersion", model)
    else:
        raise MediaError(f"{provider} cannot transcribe audio in this course")
    r = {"text": text.strip(), "provider": provider, "model": answered, "usage": usage,
         "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "_source": "live"}
    _record(rid, {"capability": "hear", "provider": provider, "model": model, "media": audio.name}, r)
    return r


# ── speak: text → speech (wav) ────────────────────────────────────────────────
def _wav(pcm: bytes, rate: int = 24000) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def speak(provider: str | None, text: str, out: str | Path, *, model: str | None = None, voice: str | None = None,
          style: str = "") -> dict:
    """Write a .wav file. DRY mode copies a recorded file if this exact request was recorded."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    provider = provider or pick("speak")
    model = model or DEFAULTS["speak"][provider]
    voice = voice or {"chatgpt": "coral", "gemini": "Kore"}[provider]
    rid = _rid("speak", provider, model, voice, style, text)
    cache_wav = RECORDED / f"{rid}.wav"
    if not _live(provider):
        if cache_wav.is_file():
            out.write_bytes(cache_wav.read_bytes())
            r = json.loads((RECORDED / f"{rid}.json").read_text(encoding="utf-8"))["response"]
            return {**r, "file": str(out), "_source": "recorded"}
        return {"text": "[DRY — no recorded audio for this exact text]", "provider": provider, "model": model,
                "file": None, "_source": "placeholder"}
    t0 = time.perf_counter()
    key, _ = llmkit.api_key(provider)
    if provider == "chatgpt":
        body = {"model": model, "input": text, "voice": voice, "response_format": "wav"}
        if style:
            body["instructions"] = style
        wav = _http(llmkit.base_url("chatgpt") + "/audio/speech", json.dumps(body).encode(),
                    {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    timeout=40)                               # TTS occasionally hangs ~2 min; fail fast + retry once
    else:
        d = _gemini_generate(model, [{"text": (f"{style}\n\n" if style else "") + text}],
                             {"responseModalities": ["AUDIO"],
                              "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}})
        part = next(p for p in d["candidates"][0]["content"]["parts"] if p.get("inlineData"))
        rate = 24000
        mt = part["inlineData"].get("mimeType", "")
        if "rate=" in mt:
            rate = int(mt.split("rate=")[1].split(";")[0])
        wav = _wav(base64.b64decode(part["inlineData"]["data"]), rate)
    out.write_bytes(wav)
    secs = max(0.0, (len(wav) - 44) / (2 * 24000))
    r = {"text": text, "provider": provider, "model": model, "voice": voice, "file": str(out),
         "bytes": len(wav), "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "_source": "live"}
    if os.environ.get("JEV_RECORD") == "1":
        RECORDED.mkdir(exist_ok=True)
        cache_wav.write_bytes(wav)
        _record(rid, {"capability": "speak", "provider": provider, "model": model, "voice": voice, "text": text}, r)
    return r


# ── watch: video → text ───────────────────────────────────────────────────────
def watch(video: str | Path, prompt: str, *, model: str | None = None, json_output: bool = False) -> dict:
    video = Path(video)
    provider = "gemini"
    model = model or DEFAULTS["watch"]["gemini"]
    rid = _rid("watch", provider, model, prompt, _sha(video), json_output)
    if not _live(provider):
        return _replay(rid, "watch", provider, model)
    t0 = time.perf_counter()
    cfg = {"responseMimeType": "application/json"} if json_output else None
    d = _gemini_generate(model, [{"inline_data": {"mime_type": _mime(video), "data": base64.b64encode(video.read_bytes()).decode()}},
                                 {"text": prompt}], cfg, timeout=240)
    r = {"text": _gemini_text(d), "provider": provider, "model": d.get("modelVersion", model), "usage": _gemini_usage(d),
         "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "_source": "live"}
    _record(rid, {"capability": "watch", "provider": provider, "model": model, "prompt": prompt, "media": video.name}, r)
    return r


def show(r: dict, label: str = "", width: int = 96) -> None:
    tag = {"live": "LIVE", "recorded": "RECORDED replay", "placeholder": "PLACEHOLDER"}.get(r.get("_source"), "?")
    for line in (r.get("text") or "").splitlines() or [""]:
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > 20 else width
            print("   │ " + line[:cut]); line = line[cut:].lstrip()
        print("   │ " + line)
    u = r.get("usage") or {}
    lat = f" · {r['latency_ms']:.0f} ms" if r.get("latency_ms") else ""
    tok = f" · {u.get('input_tokens', 0)} in / {u.get('output_tokens', 0)} out tok" if u else ""
    print(f"◆ {label + ' · ' if label else ''}{r.get('provider')} · {r.get('model')} · {tag}{tok}{lat}")


if __name__ == "__main__":                              # python mediakit.py → one call per capability
    M = COMMON.parent / "13_multimodal_jev" / "media"
    for cap in DEFAULTS:
        print(f"{cap:<6} capable here: {', '.join(capable(cap)) or 'none'}")
