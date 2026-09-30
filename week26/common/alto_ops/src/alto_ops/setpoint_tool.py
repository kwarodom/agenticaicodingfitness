"""request_setpoint_change — the claw may REQUEST a write, never make one (research tutorial §6.3 design note).

It validates a work_order_id with a Pydantic input schema (Part 3 exercise 2, layer a) and returns a ticket
that a separate, human-approved service would act on. Nothing here talks to a BMS.
"""
import json
import re

from pydantic import BaseModel, Field, field_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class SetpointRequestConfig(FunctionBaseConfig, name="request_setpoint_change"):
    """Create a setpoint-change TICKET for human approval. Never writes to equipment."""
    min_c: float = Field(5.5, description="lowest chilled-water setpoint a ticket may ask for (°C)")
    max_c: float = Field(9.0, description="highest chilled-water setpoint a ticket may ask for (°C)")


class SetpointRequest(BaseModel):
    point: str = Field(description="BMS point, e.g. CH-2.CHWST_SP")
    value_c: float = Field(description="requested setpoint in °C")
    work_order_id: str = Field(description="approved work order id, e.g. WO-2026-0142")

    @field_validator("work_order_id")
    @classmethod
    def _wo(cls, v: str) -> str:
        if not re.fullmatch(r"WO-\d{4}-\d{4}", v or ""):
            raise ValueError("work_order_id must look like WO-2026-0142 — no ticket without an approved work order")
        return v


@register_function(config_type=SetpointRequestConfig)
async def request_setpoint_change(config: SetpointRequestConfig, builder: Builder):

    async def _request(req: SetpointRequest) -> str:
        if not config.min_c <= req.value_c <= config.max_c:
            return json.dumps({"status": "refused", "reason": f"{req.value_c} °C outside {config.min_c}–{config.max_c} °C"})
        return json.dumps({"status": "ticket_created", "ticket": f"T-{req.work_order_id[3:]}", "point": req.point,
                           "value_c": req.value_c, "note": "pending human approval — nothing was written"})

    yield FunctionInfo.from_fn(
        _request,
        input_schema=SetpointRequest,
        description="Request (never apply) a chilled-water setpoint change. Needs an approved work_order_id.",
    )
