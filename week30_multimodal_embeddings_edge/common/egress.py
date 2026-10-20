"""egress — prove that a pipeline keeps its data on this machine.

A small, in-process network guard for the Week 30 labs. `guard()` wraps socket connect and DNS lookups so
that only loopback (this machine) is reachable; every attempt is logged with its destination and verdict.
It is a TEACHING tool and a tripwire, not a security boundary: code that spawns another process, or uses
raw syscalls, is not covered. Production sovereignty is enforced outside the app — a host firewall, a
network policy, an air gap (Week 26's OpenShell policies are one way).

    import egress
    with egress.guard(allow_ports={11434}) as log:
        ...                      # embed, search, ask the local LLM
    egress.report(log)           # every connection attempt, allowed or blocked
"""
from __future__ import annotations

import ipaddress
import socket
from contextlib import contextmanager

LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


class EgressBlocked(ConnectionRefusedError):
    """Raised when the guard refuses an outbound connection."""


def is_loopback(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _dest(address) -> tuple[str, int | None]:
    if isinstance(address, tuple) and address:
        return str(address[0]), (int(address[1]) if len(address) > 1 else None)
    return str(address), None                                   # AF_UNIX path


@contextmanager
def guard(allow_ports: set[int] | None = None, allow_unix: bool = True):
    """Only loopback connections (optionally only to `allow_ports`) are allowed inside the block."""
    log: list[dict] = []
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_getaddrinfo = socket.getaddrinfo

    def verdict(sock, address) -> bool:
        if sock.family == getattr(socket, "AF_UNIX", None):
            ok = allow_unix
            log.append({"kind": "unix", "host": str(address), "port": None, "allowed": ok})
            return ok
        host, port = _dest(address)
        ok = is_loopback(host) and (allow_ports is None or port in allow_ports)
        log.append({"kind": "connect", "host": host, "port": port, "allowed": ok})
        return ok

    def connect(sock, address):
        if not verdict(sock, address):
            raise EgressBlocked(f"egress guard: blocked connection to {_dest(address)[0]}:{_dest(address)[1]}")
        return orig_connect(sock, address)

    def connect_ex(sock, address):
        if not verdict(sock, address):
            return 111                                          # ECONNREFUSED
        return orig_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        h = host.decode() if isinstance(host, bytes) else str(host)
        if h and h != "None" and not is_loopback(h):
            log.append({"kind": "dns", "host": h, "port": args[0] if args else None, "allowed": False})
            raise EgressBlocked(f"egress guard: blocked DNS lookup of {h}")
        return orig_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    try:
        yield log
    finally:
        socket.socket.connect = orig_connect
        socket.socket.connect_ex = orig_connect_ex
        socket.getaddrinfo = orig_getaddrinfo


def summary(log: list[dict]) -> dict:
    out = {"attempts": len(log), "allowed": sum(e["allowed"] for e in log),
           "blocked": sum(not e["allowed"] for e in log)}
    out["outbound_allowed"] = sum(e["allowed"] and e["kind"] != "unix" and not is_loopback(e["host"]) for e in log)
    return out


def report(log: list[dict], printer=print) -> dict:
    s = summary(log)
    seen = {}
    for e in log:
        k = (e["kind"], e["host"], e["port"], e["allowed"])
        seen[k] = seen.get(k, 0) + 1
    for (kind, host, port, allowed), n in seen.items():
        printer(f"│ {'allowed' if allowed else 'BLOCKED':8s} {kind:8s} {host}{':' + str(port) if port else ''}  ×{n}")
    return s
