#!/usr/bin/env python3
"""PART 5 · NeMo Agent Toolkit — register a tool  [INTERMEDIATE]

So far we hand-rolled the agent loop. The NVIDIA NeMo Agent Toolkit (NAT) is the
config-driven framework that builds, connects, evaluates, and observes agentic
systems — and it points at YOUR DGX model, so the whole thing stays sovereign.
The first building block is a TOOL: a Python function registered with NAT via
FunctionBaseConfig + @register_function + FunctionInfo.

This demo writes a real NAT tool module into .sandbox/ and explains each piece.

Run:  python demos/step05_nat_intro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import obsview  # noqa: E402

TOOL_MODULE = '''\
# hvac_tools.py — a NeMo Agent Toolkit tool for our DGX agent
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class RoomTelemetryConfig(FunctionBaseConfig, name="get_room_telemetry"):
    """Read live HVAC telemetry for a hotel room."""
    # config fields appear in the workflow YAML; none needed here.
    pass


@register_function(config_type=RoomTelemetryConfig)
async def get_room_telemetry(config: RoomTelemetryConfig, builder: Builder):
    rooms = {"1203": {"temp_c": 26.4, "setpoint_c": 22.0, "occupied": True},
             "0907": {"temp_c": 22.1, "filter_pa": 265, "occupied": False}}

    async def _inner(room: str) -> str:
        return str(rooms.get(room, {}))

    # FunctionInfo wraps the callable + its description so the agent can call it.
    yield FunctionInfo.from_fn(
        _inner, description="Read live HVAC telemetry for a hotel room.")
'''


def main() -> None:
    obsview.banner("PART 5", "NeMo Agent Toolkit — register a tool", "INTERMEDIATE")
    obsview.mode_line()

    sb = config.ensure_sandbox()
    (sb / "hvac_tools.py").write_text(TOOL_MODULE)
    print("Wrote a real NAT tool → .sandbox/hvac_tools.py\n")
    print(TOOL_MODULE)

    print("The four NAT pieces, decoded:")
    print("  • FunctionBaseConfig(name=…) → declares the tool + its config schema;")
    print("    the `name` is how the YAML workflow refers to it.")
    print("  • @register_function(config_type=…) → registers it in NAT's registry,")
    print("    so `nat` can discover and wire it without you importing it by hand.")
    print("  • Builder → NAT's cross-framework factory (wraps LangChain + LlamaIndex);")
    print("    ask it for an LLM and you get one pointed at your DGX model.")
    print("  • FunctionInfo.from_fn(fn, description=…) → the callable + the description")
    print("    the agent reads to decide WHEN to call it (the description is the API).")

    print("\nInstall NAT on the DGX:  pip install nvidia-nat   (a.k.a. aiqtoolkit)")
    print("\nTakeaway: in NAT a tool is a registered function, not bespoke loop code.")
    print("Next: compose tools + an agent into a YAML workflow on your DGX model.")


if __name__ == "__main__":
    main()
