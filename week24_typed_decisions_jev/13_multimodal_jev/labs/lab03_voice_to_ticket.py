#!/usr/bin/env python3
"""Lab 13-3 · Voicemail → ticket → spoken reply.

For three voicemails (English flood, Thai burning smell, English towels):
  1. hear   — a transcription model turns speech into text
  2. code   — clean timestamps, find the room number, compare it with CALLER-ID (the trusted fact)
  3. Jev    — department · urgency · safety on the transcript
  4. code   — one policy → ticket proposal; an APPROVED reply template in the guest's language
  5. speak  — text-to-speech of that approved reply, saved under .runs/

Then: a SECOND transcriber on the Thai voicemail — do two models agree on the room number?

Run: .venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab03_voice_to_ticket.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perception as P  # noqa: E402
from jevkit import banner, show, step, table  # noqa: E402
import mediakit as mk  # noqa: E402

hear_p, speak_p = mk.pick("hear"), mk.pick("speak")
banner("Lab 13-3 · voice → ticket", f"hear: {hear_p} · decide: Jev · speak: {speak_p}")

FILES = ["voicemail_flood_en.wav", "voicemail_burning_th.wav", "request_towels_en.wav"]
summary = []
for n, name in enumerate(FILES, 1):
    step(n, f"media/{name}")
    try:
        h = mk.hear(hear_p, P.MEDIA / name)
    except mk.MediaError as e:
        print(f"✕ {e}")
        continue
    mk.show(h, "hear")
    transcript = P.clean_transcript(h["text"])
    lang = "th" if P.THAI.search(transcript) else "en"
    heard_room, trusted_room = P.room_from_transcript(transcript), P.CALLER_ID[name]
    mismatch = heard_room is not None and heard_room != trusted_room
    print(f"» code facts  language={lang} · room heard={heard_room} · caller-ID room={trusted_room}"
          f"{'  ⚠ MISMATCH' if mismatch else ''}")
    resp = P.ask({"transcript": transcript}, P.VOICE_QUESTIONS)
    a = P.validate(resp, P.VOICE_QUESTIONS)
    show(resp, top=2)
    dept = a["department"]["choice"]
    decision = P.policy({"modality": "audio", "queue": dept, "safety": a["safety"]["noul"],
                         "perception_ok": bool(transcript),
                         "facts": {"room_mismatch": mismatch, "room_heard": heard_room, "room_trusted": trusted_room}})
    ticket = {"room": trusted_room, "team": dept, "urgency": round(a["urgency"]["score"], 2), **decision}
    print("→ ticket " + json.dumps(ticket, ensure_ascii=False))
    team = P.TEAM_TH.get(dept, dept) if lang == "th" else dept.replace("_", " ")
    reply = P.REPLY[lang].format(room=trusted_room, team=team)      # the room comes from caller-ID, not the transcript
    try:
        s = mk.speak(speak_p, reply, P.RUNS / f"reply_{Path(name).stem}.wav")
        if s.get("file"):
            print(f"🎧 saved → week24_typed_decisions_jev/13_multimodal_jev/.runs/reply_{Path(name).stem}.wav   “{reply}”")
    except mk.MediaError as e:
        print(f"✕ speak: {e}")
    summary.append([name, lang, heard_room, trusted_room, dept, f"{a['safety']['noul']:.2f}", decision["route"]])

step(len(FILES) + 1, "a second transcriber on the Thai voicemail")
other = [p for p in mk.capable("hear") if p != hear_p]
if other:
    try:
        h2 = mk.hear(other[0], P.MEDIA / "voicemail_burning_th.wav")
        mk.show(h2, f"hear ({other[0]})")
        print(f"» code: room heard by {other[0]} = {P.room_from_transcript(P.clean_transcript(h2['text']))} · "
              f"caller-ID = {P.CALLER_ID['voicemail_burning_th.wav']}")
    except mk.MediaError as e:
        print(f"✕ {e}")
else:
    print("◈ only one transcription provider has a key — add another to compare")

step(len(FILES) + 2, "summary")
table(summary, ["voicemail", "lang", "room heard", "caller-ID", "team", "safety", "route"])
print("═ Two transcribers can agree and still both be wrong. Exact facts (room number) come from a trusted system;")
print("  the transcript is for MEANING, which is what Jev judges.  execute: False — staff approve every ticket and reply.")
