"""Small provider adapters for the workshop copilot and vision labeling."""

import base64
import json
from dataclasses import dataclass


PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "glm": "https://api.z.ai/api/coding/paas/v4",
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Completion:
    text: str | None
    tool_calls: list[ToolCall]
    model: str
    assistant_message: dict


def complete(settings, system, messages, tools, max_tokens=1000):
    provider = settings["provider"]
    if provider == "anthropic":
        return _complete_anthropic(
            settings, system, messages, tools, max_tokens
        )
    return _complete_openai_compatible(
        settings, system, messages, tools, max_tokens
    )


def label_scene(settings, contact_sheet, object_ids):
    encoded = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
    prompt = (
        "Identify each labeled crop as an ordinary desk object. Return JSON only "
        "with one top-level key named 'objects'. Preserve every ID exactly: "
        f"{object_ids}. Each object must contain id, label, aliases, attributes, "
        "and semantic_confidence from 0 to 1. Use concise labels and useful "
        "aliases. If uncertain, use label 'unknown'. Do not return positions, "
        "dimensions, grasp coordinates, or robot commands."
    )
    provider = settings["provider"]
    if provider == "anthropic":
        content, model = _label_anthropic(settings, encoded, prompt)
    else:
        content, model = _label_openai_compatible(settings, encoded, prompt)
    labels = _parse_labels(content, object_ids)
    return labels, model


def _complete_openai_compatible(settings, system, messages, tools, max_tokens):
    from openai import OpenAI

    provider = settings["provider"]
    client = OpenAI(
        api_key=settings.get("api_key") or "local",
        base_url=_base_url(settings),
        # Some gateways (e.g. ThaiLLM) run a WAF that blocks the default
        # "OpenAI/Python" User-Agent. Send a neutral UA so custom endpoints work.
        default_headers={"User-Agent": "agentic-robotics-portal/1.0"},
    )
    request = {
        "model": settings["model"],
        "messages": [{"role": "system", "content": system}, *messages],
    }
    if tools:
        if provider in {"custom", "glm"}:
            request["tools"] = [
                {
                    "type": tool["type"],
                    "function": {
                        key: value
                        for key, value in tool["function"].items()
                        if key != "strict"
                    },
                }
                for tool in tools
            ]
        else:
            request["tools"] = tools
    if tools and provider in {"openrouter", "openai"}:
        request["parallel_tool_calls"] = False
    if provider == "openai":
        request["max_completion_tokens"] = max_tokens
    else:
        request["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request)
    message = response.choices[0].message
    calls = [
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=_json_object(call.function.arguments),
        )
        for call in (message.tool_calls or [])
    ]
    message_data = message.model_dump()
    assistant_message = {
        "role": "assistant",
        "content": message.content,
    }
    for field in ("tool_calls", "reasoning", "reasoning_details"):
        if message_data.get(field) is not None:
            assistant_message[field] = message_data[field]
    return Completion(
        text=message.content,
        tool_calls=calls,
        model=response.model or settings["model"],
        assistant_message=assistant_message,
    )


def _complete_anthropic(settings, system, messages, tools, max_tokens):
    from anthropic import Anthropic

    client = Anthropic(api_key=settings["api_key"])
    request = {
        "model": settings["model"],
        "system": system,
        "messages": _anthropic_messages(messages),
        "max_tokens": max_tokens,
    }
    if tools:
        request["tools"] = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            }
            for tool in tools
        ]
    response = client.messages.create(**request)
    text_blocks = [
        block.text for block in response.content if block.type == "text"
    ]
    calls = [
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in response.content
        if block.type == "tool_use"
    ]
    assistant_content = []
    for block in response.content:
        if block.type == "text":
            assistant_content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            assistant_content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    text = "\n".join(text_blocks).strip() or None
    return Completion(
        text=text,
        tool_calls=calls,
        model=response.model,
        assistant_message={"role": "assistant", "content": assistant_content},
    )


def _anthropic_messages(messages):
    converted = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
            if (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1]["content"], list)
                and all(
                    item.get("type") == "tool_result"
                    for item in converted[-1]["content"]
                )
            ):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        converted.append({"role": role, "content": message["content"]})
    return converted


def _label_openai_compatible(settings, encoded, prompt):
    from openai import OpenAI

    provider = settings["provider"]
    client = OpenAI(
        api_key=settings.get("api_key") or "local",
        base_url=_base_url(settings),
        # Some gateways (e.g. ThaiLLM) run a WAF that blocks the default
        # "OpenAI/Python" User-Agent. Send a neutral UA so custom endpoints work.
        default_headers={"User-Agent": "agentic-robotics-portal/1.0"},
    )
    request = {
        "model": settings["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        }],
    }
    if provider == "openai":
        request["max_completion_tokens"] = 1200
    else:
        request["max_tokens"] = 1200
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"{provider} returned no semantic labels")
    return content, response.model or settings["model"]


def _label_anthropic(settings, encoded, prompt):
    from anthropic import Anthropic

    client = Anthropic(api_key=settings["api_key"])
    response = client.messages.create(
        model=settings["model"],
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    content = "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if not content:
        raise RuntimeError("anthropic returned no semantic labels")
    return content, response.model


def _base_url(settings):
    if settings["provider"] == "custom":
        base_url = (settings.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise ValueError("a base URL is required for a custom provider")
        # Be forgiving: users often paste the full completions URL. The OpenAI
        # SDK appends "/chat/completions" itself, so a pasted suffix would
        # double up and 404. Strip it back to the API root.
        for suffix in ("/chat/completions", "/completions"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        return base_url
    return PROVIDER_BASE_URLS[settings["provider"]]


def _json_object(value):
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("the model returned invalid tool arguments") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("the model returned non-object tool arguments")
    return parsed


def _parse_labels(content, object_ids):
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("the vision model did not return JSON")
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as error:
            raise RuntimeError("the vision model returned invalid JSON") from error

    labels = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(labels, list):
        raise RuntimeError("the vision model returned an invalid 'objects' list")
    for item in labels:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise RuntimeError("the vision model returned an invalid object ID")
        if not isinstance(item.get("label"), str) or not item["label"].strip():
            raise RuntimeError(f"invalid label for {item['id']}")
        for field in ("aliases", "attributes"):
            item[field] = _normalize_text_list(item.get(field), field, item["id"])
        confidence = item.get("semantic_confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0
        ):
            raise RuntimeError(f"invalid confidence for {item['id']}")

    returned_ids = [item["id"] for item in labels]
    if sorted(returned_ids) != sorted(object_ids):
        raise RuntimeError(
            f"vision model returned IDs {returned_ids}; expected {object_ids}"
        )
    labels_by_id = {item["id"]: item for item in labels}
    return [labels_by_id[object_id] for object_id in object_ids]


def _normalize_text_list(value, field, object_id):
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items()]
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        return [str(item) for item in value]
    raise RuntimeError(f"invalid {field} for {object_id}")
