"""Run an async coroutine to completion in its OWN thread + event loop.

The Claude Agent SDK is built on anyio task groups. Calling ``asyncio.run()``
repeatedly in one thread (one foreground turn, then a background consolidation,
then another turn) leaves anyio cancel scopes bound to dead loops and raises
"Attempted to exit cancel scope in a different task". Giving every SDK
interaction a fresh thread with a fresh loop side-steps that entirely — and it
also lets these calls run safely from inside FastAPI's sync-endpoint threadpool.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def run_sync(async_fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    """Execute ``async_fn(*args, **kwargs)`` in a dedicated thread and return its
    result (re-raising any exception in the caller's thread)."""
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(async_fn(*args, **kwargs))
        except BaseException as exc:                    # noqa: BLE001 — re-raised below
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
