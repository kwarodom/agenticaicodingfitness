"""route_guest_message — the fine-tuned hotel router as a NAT tool (Week 25 capstone).

The agent does not guess the department or the priority. It asks the model that was fine-tuned for
exactly that (Module 09), served with its LoRA adapter (Module 13), behind the LiteLLM gateway
(Module 08) under the alias `hotel-router`. The router's one line of JSON is parsed and checked in
code; the agent only ever sees a validated result or a clear error it must report.

Plain urllib, no SDK: inside the OpenShell sandbox (Module 15) the only egress the policy opens is the
gateway, and this file is the whole client.
"""
import json
import os
import re
import urllib.error
import urllib.request

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMFrameworkEnum
from nat.plugin_api import register_function

DEPARTMENTS = ("housekeeping", "engineering", "front_desk", "food_beverage", "concierge", "security")
PRIORITIES = ("normal", "urgent")
# The exact system prompt Module 09 trained the router with. A different prompt = a different task.
ROUTER_SYSTEM = ("You are the guest-request router for a hotel. Read the guest's message and answer with one line "
                 "of JSON: {\"department\": one of housekeeping|engineering|front_desk|food_beverage|concierge|security, "
                 "\"priority\": normal|urgent, \"reply\": a short, polite reply to the guest in the guest's language}. "
                 "Do not promise a specific time.")


def parse_router(text: str) -> dict | None:
    """The router's output → a validated dict, or None. Strict on values: no guessing, no repair."""
    for cand in [(text or "").strip()] + re.findall(r"\{.*?\}", text or "", flags=re.S):
        try:
            d = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict) and d.get("department") in DEPARTMENTS and d.get("priority") in PRIORITIES \
                and isinstance(d.get("reply"), str):
            return d
    return None


def call_router(message: str, base_url: str, model: str, api_key: str, timeout: float = 60) -> dict:
    """POST one guest message to the gateway → {"ok", "department", "priority", "reply", "served_by", "fallbacks"}."""
    body = {"model": model, "temperature": 0, "max_tokens": 160,
            "messages": [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": message}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json",
                                                         "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:          # noqa: S310 — the gateway
            data = json.loads(r.read())
            hdr = {k.lower(): v for k, v in r.headers.items()}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"router unreachable: {type(e).__name__}: {str(e)[:120]}"}
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    parsed = parse_router(text)
    if not parsed:
        return {"ok": False, "error": f"router returned invalid JSON: {text[:120]!r}",
                "served_by": data.get("model")}
    return {"ok": True, **parsed, "served_by": data.get("model"),
            "fallbacks": hdr.get("x-litellm-attempted-fallbacks", "0")}


class RouteGuestMessageConfig(FunctionBaseConfig, name="route_guest_message"):
    """Classify a guest message with the fine-tuned hotel router behind the LiteLLM gateway."""
    base_url: str = Field(default="http://127.0.0.1:4000/v1", description="The gateway's OpenAI-compatible base URL.")
    model: str = Field(default="hotel-router", description="The gateway alias of the fine-tuned router.")
    api_key_env: str = Field(default="HOTEL_GATEWAY_KEY", description="Env var holding the gateway key (never in YAML).")


@register_function(config_type=RouteGuestMessageConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def route_guest_message_function(config: RouteGuestMessageConfig, builder: Builder):

    async def _route_guest_message(message: str) -> str:
        """Classify a hotel guest's message with the hotel's fine-tuned router.

        Args:
            message (str): The guest's message, exactly as written (any language).

        Returns:
            str: JSON with department, priority (normal|urgent) and a suggested reply, or an error.
        """
        r = call_router(message, config.base_url, config.model, os.environ.get(config.api_key_env, ""))
        return json.dumps(r, ensure_ascii=False)

    yield FunctionInfo.from_fn(_route_guest_message, description=_route_guest_message.__doc__)
