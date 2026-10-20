#!/usr/bin/env python3
"""A mock TRUSTED PROVIDER that issues ID-JAGs — the agent-verified half.

In the real world the ID-JAG (Identity Assertion Authorization Grant) is a JWT
signed by a provider the app already trusts — OpenAI, Anthropic, Cursor — that
vouches for *who the human behind the agent is*. The app verifies that signature
against the provider's published JWKS and, if it trusts the issuer, hands back a
scoped credential with NO human in the loop. (Slides A4 / B-verified.)

To keep this example runnable OFFLINE with real crypto (not hand-waving), this
module stands up a tiny self-contained provider:

  • a real RSA keypair, generated in-process at import time;
  • mint_id_jag(...) → a signed RS256 JWT with the claims the app validates
    (iss, sub, aud, iat, exp, jti, email/email_verified);
  • jwks() → the public half as a JWKS the app fetches by `kid`.

`auth.altotech.ai` trusts this provider by listing its issuer + JWKS in
app_server.TRUST_LIST. Swap this file for "verify against the real OpenAI JWKS"
and nothing else in the example changes — that's the point of the abstraction.

Requires: pyjwt[crypto] (PyJWT + cryptography). Both are in requirements.txt.
"""
from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# This provider's stable identity. The app's trust list keys off `ISSUER`.
ISSUER = "https://id.trusted-provider.example"
KID = "authmd-demo-key-1"

# A real trusted provider has a STABLE signing key, not one regenerated each
# boot — and the app (Part A) and agent (Part B) run as separate processes, so
# the key must be shared between them. We persist it to a PEM beside this package
# and generate it only once. Delete the file to rotate. (Real providers rotate
# keys + publish several `kid`s; one persisted key is enough to teach the flow.)
_KEY_PATH = Path(__file__).resolve().parent / "idjag_signing_key.pem"


def _load_or_create_key():
    if _KEY_PATH.exists():
        return serialization.load_pem_private_key(_KEY_PATH.read_bytes(), password=None)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _KEY_PATH.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    return key


_PRIVATE_KEY = _load_or_create_key()
_PUBLIC_KEY = _PRIVATE_KEY.public_key()


def _b64url_uint(n: int) -> str:
    """Encode an int as base64url (no padding) — the JWKS n/e wire format."""
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def jwks() -> dict:
    """The public key as a JWKS document. The app fetches this and selects the
    key whose `kid` matches the ID-JAG header (slide A4 step 3)."""
    pub = _PUBLIC_KEY.public_numbers()
    return {"keys": [{
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": KID,
        "n": _b64url_uint(pub.n),
        "e": _b64url_uint(pub.e),
    }]}


def public_pem() -> bytes:
    """PEM the app can pin directly if it prefers not to call the JWKS URL."""
    return _PUBLIC_KEY.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def mint_id_jag(*, subject: str, email: str, audience: str,
                ttl_seconds: int = 300) -> str:
    """Sign an ID-JAG asserting that `subject` (with verified `email`) authorises
    an agent to act against `audience`. Short-lived by design — the consuming
    agent re-mints a fresh one each time it wakes (slide B3).

    Args:
        subject:  the provider's stable user id (becomes `sub`).
        email:    the user's verified email (the app matches users on this).
        audience: the resource the token is for, e.g. the app's API base (`aud`).
        ttl_seconds: lifetime; deliberately minutes, not days.
    """
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": subject,
        "aud": audience,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid.uuid4().hex,           # unique → app's replay cache rejects reuse
        "email": email,
        "email_verified": True,
    }
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256",
                      headers={"kid": KID})
