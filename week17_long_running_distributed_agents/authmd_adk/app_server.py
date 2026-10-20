#!/usr/bin/env python3
"""PART A — the AltoTech Energy API, made *agent-ready* with auth.md.

This one FastAPI service is everything slides A1–A6 describe an app must publish
and implement so an agent it has never met can discover it, authenticate on a
real user's behalf, and call it with a scoped, short-lived, revocable token:

  Discovery        GET  /auth.md                                  (A1)
                   GET  /.well-known/oauth-protected-resource     (A2 · PRM, RFC 9728)
                   GET  /.well-known/oauth-authorization-server    (A2 · agent_auth block)
                   401 + WWW-Authenticate on the protected API     (A2 · the hint)
  Registration     POST /agent/auth        dispatch on `type`      (A3)
                     type=identity_assertion → verify ID-JAG       (A4 · agent-verified)
                     type=anonymous          → pre-claim creds      (A5 · user-claimed)
                     type=verified_email     → email OTP now        (A5 · user-claimed)
                   POST /agent/auth/claim          supply email      (A5)
                   POST /agent/auth/claim/complete submit OTP        (A5)
                   POST /agent/auth/revoke         logout / token    (A6)
  Protected API    GET  /sites/{id}/energy    needs scope sites.read
                   POST /sites/{id}/setpoint  needs scope control.write

Faithful where it teaches: real RFC 9728 discovery shapes, real RS256/JWKS
ID-JAG verification, replay protection by `jti`, and only SHA-256 *hashes* of
secrets are stored (slide A5). Stubbed where it would only add noise: the store
is an in-process dict and rate-limiting is a comment, not middleware — see the
README's reality check. Runs fully offline; no model, no outbound network.
"""
from __future__ import annotations

import hashlib
import secrets
import time
import uuid

import jwt
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import idjag_provider
from config import APP_BASE

SCOPES_SUPPORTED = {
    "sites.read": "read energy + occupancy data for a site",
    "control.write": "write HVAC setpoints for a site",
}

# Issuers this app trusts for the agent-verified flow → how to get their JWKS.
# Production: an HTTPS JWKS URL fetched + cached. Here: the in-process provider.
TRUST_LIST = {idjag_provider.ISSUER: idjag_provider.jwks}


# ── token + secret stores (in-process; hash everything sensitive) ────────────
def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# token_hash → {user_id, scopes, kind, expires_at, revoked}
_CREDENTIALS: dict[str, dict] = {}
# claim_token_hash → {user_id, scopes, email, kind, otp_hash, completed}
_PENDING_CLAIMS: dict[str, dict] = {}
_SEEN_JTI: set[str] = set()                 # replay cache (slide A6)
_AUDIT: list[dict] = []                      # one event per state change (A6)

# DEMO ONLY: the OTP would be emailed and never touch an API response. To keep
# this runnable offline we stash the plaintext here so the demo driver can
# "read the user's email". Delete this in anything real — see README.
_DEMO_INBOX: dict[str, str] = {}


def _audit(event: str, **fields) -> None:
    _AUDIT.append({"t": round(time.time(), 3), "event": event, **fields})


def _issue(user_id: str, scopes: list[str], kind: str, ttl: int) -> str:
    """Mint an opaque credential, store only its hash, return the plaintext once."""
    token = f"{kind}_{secrets.token_urlsafe(24)}"
    _CREDENTIALS[_sha256(token)] = {
        "user_id": user_id, "scopes": list(scopes), "kind": kind,
        "expires_at": time.time() + ttl, "revoked": False,
    }
    _audit("credential_issued", user_id=user_id, kind=kind, scopes=scopes)
    return token


def _scopes_or_400(requested) -> list[str]:
    scopes = list(requested or ["sites.read"])
    unknown = [s for s in scopes if s not in SCOPES_SUPPORTED]
    if unknown:
        raise HTTPException(400, f"unsupported scopes: {unknown}")
    return scopes


app = FastAPI(title="AltoTech Energy API — agent-ready (auth.md × ADK, Part A)")


# ── A1: auth.md ──────────────────────────────────────────────────────────────
@app.get("/auth.md", response_class=PlainTextResponse)
async def auth_md() -> str:
    """The discovery-friendly Markdown companion to the structured PRM (A1)."""
    scope_lines = "\n".join(f"- {s:<14} {d}" for s, d in SCOPES_SUPPORTED.items())
    return f"""# AltoTech Energy API

Agent registration for the AltoTech energy-optimisation platform.

## Flows

- agent-verified (ID-JAG)
- user-claimed (email required)

## Scopes

{scope_lines}

## Discovery

- PRM:         {APP_BASE}/.well-known/oauth-protected-resource
- AS metadata: {APP_BASE}/.well-known/oauth-authorization-server
- Register:    {APP_BASE}/agent/auth

## Contact

integrations@altotech.ai
"""


# ── A2: discovery documents ──────────────────────────────────────────────────
@app.get("/.well-known/oauth-protected-resource")
async def prm() -> dict:
    """Protected Resource Metadata — RFC 9728 (A2). Points at the AS."""
    return {
        "resource": f"{APP_BASE}/",
        "authorization_servers": [f"{APP_BASE}/"],
        "scopes_supported": list(SCOPES_SUPPORTED),
        "bearer_methods_supported": ["header"],
    }


@app.get("/.well-known/oauth-authorization-server")
async def as_metadata() -> dict:
    """AS metadata carrying the agent_auth block (A2)."""
    return {
        "issuer": f"{APP_BASE}/",
        "agent_auth": {
            "register_uri": f"{APP_BASE}/agent/auth",
            "claim_uri": f"{APP_BASE}/agent/auth/claim",
            "claim_complete_uri": f"{APP_BASE}/agent/auth/claim/complete",
            "revocation_uri": f"{APP_BASE}/agent/auth/revoke",
            "identity_types_supported": ["anonymous", "identity_assertion",
                                         "verified_email"],
            "scopes_supported": list(SCOPES_SUPPORTED),
        },
    }


# ── A3: POST /agent/auth — one endpoint, dispatch on `type` ──────────────────
class AuthRequest(BaseModel):
    type: str                       # identity_assertion | anonymous | verified_email
    id_jag: str | None = None       # verified
    email: str | None = None        # verified_email / matching
    scopes: list[str] | None = None


@app.post("/agent/auth")
async def agent_auth(req: AuthRequest) -> dict:
    if req.type == "identity_assertion":
        return _handle_verified(req)
    if req.type == "anonymous":
        return _handle_anonymous(req)
    if req.type == "verified_email":
        return _handle_email_required(req)
    raise HTTPException(400, f"unknown type {req.type!r}")


# ── A4: agent-verified — verify the ID-JAG, issue creds synchronously ────────
def _handle_verified(req: AuthRequest) -> dict:
    if not req.id_jag:
        raise HTTPException(400, "identity_assertion requires id_jag")
    scopes = _scopes_or_400(req.scopes)

    header = jwt.get_unverified_header(req.id_jag)              # 1. decode header
    issuer = jwt.decode(req.id_jag, options={"verify_signature": False}).get("iss")
    if issuer not in TRUST_LIST:                               # 2. issuer on trust list?
        raise HTTPException(403, f"untrusted issuer {issuer!r}")

    jwks = TRUST_LIST[issuer]()                                # 3. fetch JWKS
    key = next((k for k in jwks["keys"] if k["kid"] == header.get("kid")), None)
    if key is None:
        raise HTTPException(401, "no JWKS key matches the ID-JAG kid")

    try:                                                       # 4+5. verify sig + claims
        claims = jwt.decode(
            req.id_jag,
            key=jwt.algorithms.RSAAlgorithm.from_jwk(key),
            algorithms=["RS256"],
            audience=f"{APP_BASE}/",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"ID-JAG verification failed: {e}")

    jti = claims.get("jti")                                    # replay protection
    if not jti or jti in _SEEN_JTI:
        raise HTTPException(401, "missing or replayed jti")
    _SEEN_JTI.add(jti)
    if not claims.get("email_verified"):
        raise HTTPException(401, "ID-JAG email is not verified")

    user_id = _match_or_provision(issuer, claims["sub"], claims["email"])  # 6
    # NO refresh token (slide A4): the agent re-mints by presenting a fresh ID-JAG.
    token = _issue(user_id, scopes, kind="access_token", ttl=300)
    _audit("verified_issued", user_id=user_id, issuer=issuer)
    return {"flow": "agent_verified", "access_token": token, "token_type": "Bearer",
            "scopes": scopes, "expires_in": 300, "refresh_token": None,
            "user_id": user_id}


def _match_or_provision(issuer: str, sub: str, email: str) -> str:
    """Resolve the asserted identity to a local user, JIT-provisioning if new
    (slide A6). Here: a deterministic id derived from the verified email."""
    return "user_" + _sha256(f"{email}")[:12]


# ── A5: user-claimed — the OTP ceremony ──────────────────────────────────────
def _new_claim(user_id: str, scopes: list[str], email: str | None,
               upgrade_kind: str) -> str:
    claim_token = "claim_" + secrets.token_urlsafe(24)
    _PENDING_CLAIMS[_sha256(claim_token)] = {
        "user_id": user_id, "scopes": scopes, "email": email,
        "kind": upgrade_kind, "otp_hash": None, "completed": False,
    }
    return claim_token


def _send_otp(email: str, claim_hash: str) -> None:
    """Generate a one-time code, store only its hash, 'email' the plaintext.
    Plaintext leaves the server exactly once (slide A5)."""
    otp = f"{secrets.randbelow(1_000_000):06d}"
    _PENDING_CLAIMS[claim_hash]["otp_hash"] = _sha256(otp)
    _DEMO_INBOX[email] = otp                       # DEMO ONLY — see module docstring
    _audit("otp_sent", email=email)


def _handle_anonymous(req: AuthRequest) -> dict:
    """Anonymous start: issue a pre-claim api_key immediately with minimal scope;
    the user runs the OTP claim later to bind an identity + upgrade scopes."""
    requested = _scopes_or_400(req.scopes)
    user_id = "anon_" + uuid.uuid4().hex[:12]
    pre_claim_scopes = [s for s in requested if s == "sites.read"] or ["sites.read"]
    api_key = _issue(user_id, pre_claim_scopes, kind="api_key", ttl=3600)
    claim_token = _new_claim(user_id, requested, email=None, upgrade_kind="api_key")
    _audit("anonymous_preclaim", user_id=user_id)
    return {"flow": "user_claimed", "stage": "pre_claim", "api_key": api_key,
            "claim_token": claim_token, "scopes": pre_claim_scopes,
            "upgrade_scopes": requested, "user_id": user_id}


def _handle_email_required(req: AuthRequest) -> dict:
    """Email-required start: no credential yet. Email the OTP now and withhold
    the credential until /claim/complete."""
    if not req.email:
        raise HTTPException(400, "verified_email requires email")
    scopes = _scopes_or_400(req.scopes)
    user_id = "user_" + _sha256(req.email)[:12]
    claim_token = _new_claim(user_id, scopes, req.email, upgrade_kind="access_token")
    _send_otp(req.email, _sha256(claim_token))
    return {"flow": "user_claimed", "stage": "otp_required", "claim_token": claim_token,
            "credential": None, "scopes": scopes, "user_id": user_id}


class ClaimRequest(BaseModel):
    claim_token: str
    email: str


@app.post("/agent/auth/claim")
async def agent_auth_claim(req: ClaimRequest) -> dict:
    """Anonymous-only: bind an email and trigger the OTP email (A5 step 2)."""
    rec = _PENDING_CLAIMS.get(_sha256(req.claim_token))
    if rec is None:
        raise HTTPException(404, "unknown claim_token")
    rec["email"] = req.email
    _send_otp(req.email, _sha256(req.claim_token))
    return {"stage": "otp_sent", "message": f"OTP emailed to {req.email}"}


class ClaimCompleteRequest(BaseModel):
    claim_token: str
    otp: str


@app.post("/agent/auth/claim/complete")
async def agent_auth_claim_complete(req: ClaimCompleteRequest) -> dict:
    """Submit claim_token + OTP. Anonymous upgrades scopes in place; email-
    required issues a fresh credential (A5 step 4)."""
    chash = _sha256(req.claim_token)
    rec = _PENDING_CLAIMS.get(chash)
    if rec is None:
        raise HTTPException(404, "unknown claim_token")
    if rec["completed"]:
        raise HTTPException(409, "claim already completed")
    if rec["otp_hash"] is None or _sha256(req.otp) != rec["otp_hash"]:
        raise HTTPException(401, "invalid OTP")

    rec["completed"] = True
    _DEMO_INBOX.pop(rec.get("email", ""), None)
    if rec["kind"] == "api_key":
        # Anonymous: upgrade the existing api_key's scopes in place.
        for cred in _CREDENTIALS.values():
            if cred["user_id"] == rec["user_id"] and cred["kind"] == "api_key":
                cred["scopes"] = list(rec["scopes"])
        _audit("claim_upgraded", user_id=rec["user_id"], scopes=rec["scopes"])
        return {"stage": "claimed", "upgraded_scopes": rec["scopes"],
                "user_id": rec["user_id"]}

    # Email-required: issue a fresh scoped credential now.
    token = _issue(rec["user_id"], rec["scopes"], kind="access_token", ttl=300)
    _audit("claim_issued", user_id=rec["user_id"], scopes=rec["scopes"])
    return {"stage": "claimed", "access_token": token, "token_type": "Bearer",
            "scopes": rec["scopes"], "expires_in": 300, "user_id": rec["user_id"]}


# ── A6: revocation ───────────────────────────────────────────────────────────
class RevokeRequest(BaseModel):
    token: str | None = None        # revoke this exact credential
    logout_token: str | None = None  # provider-signed: revoke all for the user


@app.post("/agent/auth/revoke")
async def agent_auth_revoke(req: RevokeRequest) -> dict:
    if req.token:
        rec = _CREDENTIALS.get(_sha256(req.token))
        if rec:
            rec["revoked"] = True
            _audit("revoked_token", user_id=rec["user_id"])
        return {"revoked": bool(rec)}

    if req.logout_token:                       # verify like an ID-JAG, then bulk-revoke
        issuer = jwt.decode(req.logout_token, options={"verify_signature": False}).get("iss")
        if issuer not in TRUST_LIST:
            raise HTTPException(403, "untrusted issuer")
        header = jwt.get_unverified_header(req.logout_token)
        key = next((k for k in TRUST_LIST[issuer]()["keys"]
                    if k["kid"] == header.get("kid")), None)
        try:
            claims = jwt.decode(req.logout_token,
                                key=jwt.algorithms.RSAAlgorithm.from_jwk(key),
                                algorithms=["RS256"], audience=f"{APP_BASE}/")
        except jwt.InvalidTokenError as e:
            raise HTTPException(401, f"logout_token invalid: {e}")
        if claims.get("jti") in _SEEN_JTI:
            raise HTTPException(401, "replayed logout_token")
        _SEEN_JTI.add(claims.get("jti"))
        user_id = _match_or_provision(issuer, claims["sub"], claims["email"])
        n = 0
        for cred in _CREDENTIALS.values():
            if cred["user_id"] == user_id and not cred["revoked"]:
                cred["revoked"] = True
                n += 1
        _audit("revoked_user", user_id=user_id, count=n)
        return {"revoked_count": n, "user_id": user_id}

    raise HTTPException(400, "supply token or logout_token")


# ── the protected resource: scoped Bearer enforcement + the 401 hint ─────────
def _require_scope(authorization: str | None, scope: str) -> dict:
    """Validate the Bearer token and that it carries `scope`. The 401 path
    returns the WWW-Authenticate discovery hint (slide A2)."""
    prm_url = f"{APP_BASE}/.well-known/oauth-protected-resource"
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token",
                            headers={"WWW-Authenticate":
                                     f'Bearer resource_metadata="{prm_url}"'})
    token = authorization.split(" ", 1)[1].strip()
    rec = _CREDENTIALS.get(_sha256(token))
    if rec is None or rec["revoked"] or rec["expires_at"] < time.time():
        raise HTTPException(401, "invalid, expired, or revoked token",
                            headers={"WWW-Authenticate":
                                     f'Bearer resource_metadata="{prm_url}"'})
    if scope not in rec["scopes"]:
        raise HTTPException(403, f"token lacks required scope {scope!r}")
    return rec


@app.get("/sites/{site_id}/energy")
async def site_energy(site_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_scope(authorization, "sites.read")
    # Mock telemetry — the point is the auth, not the data.
    return {"site_id": site_id, "kw_now": 412.5, "occupancy": 0.63,
            "hvac_setpoint_c": 24.0, "recommended_setpoint_c": 25.5}


class SetpointRequest(BaseModel):
    setpoint_c: float


@app.post("/sites/{site_id}/setpoint")
async def site_setpoint(site_id: str, body: SetpointRequest,
                        authorization: str | None = Header(default=None)) -> dict:
    rec = _require_scope(authorization, "control.write")
    _audit("setpoint_applied", site_id=site_id, setpoint_c=body.setpoint_c,
           user_id=rec["user_id"])
    return {"site_id": site_id, "applied_setpoint_c": body.setpoint_c,
            "status": "applied"}


# ── demo-support endpoints (NOT part of auth.md) ─────────────────────────────
@app.get("/_demo/inbox/{email}")
async def demo_inbox(email: str) -> dict:
    """Stand-in for 'the user opens their email and reads the OTP'. Demo only."""
    otp = _DEMO_INBOX.get(email)
    if otp is None:
        raise HTTPException(404, "no OTP pending for that address")
    return {"email": email, "otp": otp}


@app.get("/_demo/audit")
async def demo_audit() -> dict:
    return {"events": _AUDIT}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "scopes_supported": list(SCOPES_SUPPORTED)}


@app.exception_handler(HTTPException)
async def _http_exc(_: Request, exc: HTTPException):
    # Preserve WWW-Authenticate on 401s (the discovery hint).
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code,
                        headers=exc.headers or {})


if __name__ == "__main__":
    import uvicorn
    from config import APP_HOST, APP_PORT
    print(f"✅ AltoTech Energy API (agent-ready) up — {APP_BASE}")
    print(f"   auth.md:   {APP_BASE}/auth.md")
    print(f"   PRM:       {APP_BASE}/.well-known/oauth-protected-resource")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
