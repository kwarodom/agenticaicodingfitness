#!/usr/bin/env python3
"""PART B core — the auth-grant abstraction that bridges the two halves.

This is the single most important file in Part B. Slides B2/B3 make one demand:

    Store the GRANT, not the TOKEN. auth.md tokens are short-lived by design;
    your agent may sleep for days. Persist what the agent is *allowed to do*,
    then RE-MINT a fresh credential each time it wakes — right before it calls
    the API.

So `AuthGrant` is the durable thing that lives in ADK session state (it has no
live token in it), and `AuthMdClient` turns a grant into a fresh credential on
demand by doing real auth.md discovery + running the right flow:

  • acquire(grant)            — agent-verified. Fully autonomous, re-mintable on
                                every wake by presenting a fresh ID-JAG. This is
                                the wake-time re-mint path (B3).
  • start_claim/complete_claim — user-claimed. The human-in-the-loop approval
                                gate (B4): a human confirms an OTP once.
  • revoke                    — hand a token (or provider logout token) back.

Deliberately generic: `acquire(grant)` is the seam where you'd fall back to a
plain OAuth refresh-token exchange for services that don't publish an auth.md
(slide "reality check"). The ADK agent never sees any of this — it just calls
acquire() in a before_tool_callback.

Pure HTTP + crypto; no model, no network beyond the app. Runs offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

import idjag_provider


@dataclass
class AuthGrant:
    """The DURABLE authorisation — safe to persist across days of idle time.
    Note what is NOT here: no access_token, no api_key. Just what the agent is
    allowed to do and how to re-acquire it."""
    service: str                       # logical name, e.g. "altotech_energy"
    flow: str                          # "agent_verified" | "user_claimed"
    scopes: list[str]
    subject: str | None = None         # verified: the provider-stable user id
    email: str | None = None           # matching key for both flows
    claim_ref: str | None = None       # user-claimed: the claim_token reference

    def as_state(self) -> dict:
        """Shape stored under state['auth_grants'][service] (slide B2)."""
        return {"flow": self.flow, "scopes": self.scopes, "subject": self.subject,
                "email": self.email, "claim_ref": self.claim_ref}

    @classmethod
    def from_state(cls, service: str, d: dict) -> "AuthGrant":
        return cls(service=service, flow=d["flow"], scopes=list(d["scopes"]),
                   subject=d.get("subject"), email=d.get("email"),
                   claim_ref=d.get("claim_ref"))


@dataclass
class Credential:
    """A freshly minted, short-lived, scoped token. Held transiently — never
    persisted; re-minted on the next wake."""
    token: str
    scopes: list[str]
    flow: str
    expires_in: int = 0
    extra: dict = field(default_factory=dict)

    def bearer(self) -> str:
        return f"Bearer {self.token}"


class AuthMdClient:
    """Consumes an app's auth.md to mint scoped credentials from a durable grant."""

    def __init__(self, app_base: str, *, timeout: float = 30.0):
        self.app_base = app_base.rstrip("/")
        self._http = httpx.Client(base_url=self.app_base, timeout=timeout)
        self._agent_auth: dict | None = None

    # ── discovery (B-recap / slide A2 consumed from the agent side) ──────────
    def discover(self) -> dict:
        """auth.md → PRM → AS metadata. Returns the agent_auth block (cached).
        This is exactly the chain an agent runs the first time it meets a
        service it found via docs, an SDK, or a 401's WWW-Authenticate hint."""
        if self._agent_auth is not None:
            return self._agent_auth
        # auth.md is the human/agent-readable entry point; the PRM is the
        # machine source of truth that points at the Authorization Server.
        self._http.get("/auth.md").raise_for_status()
        prm = self._http.get("/.well-known/oauth-protected-resource").json()
        as_url = prm["authorization_servers"][0].rstrip("/")
        meta = httpx.get(f"{as_url}/.well-known/oauth-authorization-server",
                         timeout=self._http.timeout).json()
        self._agent_auth = meta["agent_auth"]
        return self._agent_auth

    # ── agent-verified: the autonomous wake-time re-mint (B3) ────────────────
    def acquire(self, grant: AuthGrant, scopes: list[str] | None = None) -> Credential:
        """Re-mint a fresh scoped token for an agent-verified grant — no human.

        Each call signs a NEW short-lived ID-JAG for the granted identity and
        exchanges it for a fresh access token. An expired token is never a
        problem: you simply call acquire() again on the next wake.

        (In production the agent does NOT self-sign — its host platform, e.g.
        OpenAI/Anthropic/Cursor, vouches for the user and issues the ID-JAG.
        Here idjag_provider stands in for that trusted platform.)
        """
        if grant.flow != "agent_verified":
            raise ValueError(f"acquire() is for agent_verified grants, not {grant.flow!r}")
        scopes = scopes or grant.scopes
        meta = self.discover()
        id_jag = idjag_provider.mint_id_jag(
            subject=grant.subject or "agent-user",
            email=grant.email or "ops@altotech.ai",
            audience=f"{self.app_base}/",
        )
        r = self._http.post(meta["register_uri"], json={
            "type": "identity_assertion", "id_jag": id_jag, "scopes": scopes})
        r.raise_for_status()
        d = r.json()
        return Credential(token=d["access_token"], scopes=d["scopes"],
                          flow="agent_verified", expires_in=d.get("expires_in", 0),
                          extra={"user_id": d.get("user_id")})

    # ── user-claimed: the human approval gate (B4) ───────────────────────────
    def start_claim(self, email: str, scopes: list[str]) -> str:
        """Begin the email-required claim: the app emails the user a one-time
        code and withholds the credential. Returns a claim_ref to persist in the
        grant. This is the 'pause for approval' half of slide B4."""
        meta = self.discover()
        r = self._http.post(meta["register_uri"], json={
            "type": "verified_email", "email": email, "scopes": scopes})
        r.raise_for_status()
        return r.json()["claim_token"]

    def complete_claim(self, claim_ref: str, otp: str) -> Credential:
        """Finish the claim with the OTP the human confirmed → fresh credential.
        This is the 'resume + authorize in one motion' half of slide B4."""
        meta = self.discover()
        r = self._http.post(meta["claim_complete_uri"], json={
            "claim_token": claim_ref, "otp": otp})
        r.raise_for_status()
        d = r.json()
        return Credential(token=d["access_token"], scopes=d["scopes"],
                          flow="user_claimed", expires_in=d.get("expires_in", 0),
                          extra={"user_id": d.get("user_id")})

    # ── revocation (A6, consumed from the agent side) ────────────────────────
    def revoke(self, token: str | None = None, logout_token: str | None = None) -> dict:
        meta = self.discover()
        r = self._http.post(meta["revocation_uri"],
                            json={"token": token, "logout_token": logout_token})
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._http.close()
