#!/usr/bin/env python3
"""Lab 05-2 · From a real .eml file to Jev state — safely.

Parses data/sample.eml with Python's email package, keeps ONLY the plain-text
body plus the subject, and never follows URLs or opens attachments. Then it
classifies the email with the same questions and policy as lab 05-1.

Run: .venv/bin/python week24/05_email_triage/labs/lab02_eml_to_state.py
"""
import re
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parent))
from jevkit import ask, banner, show, step, validate  # noqa: E402

from lab01_triage_inbox import QUESTIONS, review_policy  # noqa: E402  (same schema + policy)

EML = HERE.parents[1] / "data" / "sample.eml"
URL_RE = re.compile(r"https?://\S+")


def eml_to_state(path: Path) -> dict:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    part = msg.get_body(preferencelist=("plain",))       # plain text only — never HTML
    if part is None:
        raise SystemExit("No plain-text body. Convert/redact HTML with an approved parser first.")
    body = part.get_content()
    attachments = [a.get_filename() for a in msg.iter_attachments()]
    # Minimise: the classifier does not need raw links — replace them with a marker.
    body = URL_RE.sub("[link removed]", body).strip()
    return {"subject": str(msg.get("Subject", "")) or "(no subject)", "body": body}, attachments, msg


def main() -> None:
    banner("Lab 05-2 · .eml → state", "parse · minimise · classify · propose")

    step(1, f"parse {EML.name} with email.parser (standard library)")
    state, attachments, msg = eml_to_state(EML)
    print(f"   From:        {msg['From']}   (NOT sent to Jev — the sender is not needed to classify)")
    print(f"   Subject:     {state['subject']}")
    print(f"   Attachments: {attachments}   (never opened, never sent)")
    print(f"✓ plain-text body kept: {len(state['body'])} chars · URLs replaced with [link removed]")
    print("── state we will send ──")
    for line in state["body"].splitlines():
        print(f"   │ {line}")

    step(2, "classify with the lab 05-1 questions")
    resp = ask(state, QUESTIONS)
    answers = validate(resp, QUESTIONS)
    show(resp, top=3)

    step(3, "apply the same deterministic review policy")
    route, reasons = review_policy(answers)
    print(f"→ proposed route: {route}" + (f"   (because: {', '.join(reasons)})" if reasons else ""))
    print("═ execute: False · no_send: True · no_delete: True · attachments_opened: False · urls_followed: False")


if __name__ == "__main__":
    main()
