"""chiller_kpi_v0 — the research tutorial's Lab 3.4 tool, byte for byte, with its ORIGINAL description.

Lab 04-2 runs it next to the course's alto_ops.chiller_tool (whose description adds "RT (refrigeration tons of
cooling)") to show that a tool description is a prompt. Registered under a different _type (chiller_kpi_v0) so
both can exist. It is not an installed package: natkit.nat_with_tools() imports it before the NAT CLI starts.
"""
from pydantic import Field
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ChillerKpiV0Config(FunctionBaseConfig, name="chiller_kpi_v0"):
    """Compute plant kW/RT and flag anomalies from a chiller CSV export."""
    csv_path: str = Field("/sandbox/data/chiller_plant.csv", description="CSV with columns ts,kw,rt")
    kw_per_rt_alarm: float = Field(0.85, description="Alarm threshold for plant efficiency", gt=0)


@register_function(config_type=ChillerKpiV0Config)
async def chiller_kpi_v0(config: ChillerKpiV0Config, builder: Builder):
    import csv

    async def _kpi(hours: int = 24) -> str:
        rows = list(csv.DictReader(open(config.csv_path)))[-hours * 4:]   # 15-min data
        kw = sum(float(r["kw"]) for r in rows) / len(rows)
        rt = sum(float(r["rt"]) for r in rows) / len(rows)
        eff = kw / rt if rt else float("nan")
        flag = "ALARM" if eff > config.kw_per_rt_alarm else "OK"
        return f"window={hours}h avg_kw={kw:.1f} avg_rt={rt:.1f} kw_per_rt={eff:.3f} status={flag}"

    yield FunctionInfo.from_fn(
        _kpi,
        description="Average chiller plant kW, RT and kW/RT over the last N hours; flags efficiency alarms.",
    )
