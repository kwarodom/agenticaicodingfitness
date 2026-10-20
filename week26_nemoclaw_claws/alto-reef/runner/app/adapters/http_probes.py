"""HTTP health probes for vLLM, NAT, Phoenix, OTel. Uses urllib only."""
from __future__ import annotations
import time, urllib.request

def probe(url: str, timeout: float = 2.0) -> dict:
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return {"url": url, "ok": 200 <= resp.status < 400, "status": resp.status, "ms": int((time.time()-t0)*1000), "at": time.time()}
    except Exception as e:  # noqa: BLE001
        return {"url": url, "ok": False, "error": str(e)[:200], "ms": int((time.time()-t0)*1000), "at": time.time()}
