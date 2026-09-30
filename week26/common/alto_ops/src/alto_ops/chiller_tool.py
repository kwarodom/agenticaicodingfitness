"""chiller_kpi — plant kW/RT over the last N hours from a 15-minute CSV export (research tutorial Lab 3.4).

Import paths verified against NAT 1.9.0 (the tutorial flags them as "confirm on your version").
"""
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ChillerKpiConfig(FunctionBaseConfig, name="chiller_kpi"):
    """Compute plant kW/RT and flag anomalies from a chiller CSV export."""
    csv_path: str = Field("/sandbox/data/chiller_plant.csv", description="CSV with columns ts,kw,rt")
    kw_per_rt_alarm: float = Field(0.85, description="Alarm threshold for plant efficiency", gt=0)


@register_function(config_type=ChillerKpiConfig)
async def chiller_kpi(config: ChillerKpiConfig, builder: Builder):
    import csv

    async def _kpi(hours: int = 24) -> str:
        with open(config.csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[-max(1, int(hours)) * 4:]      # 15-min data → 4 rows per hour
        kw = sum(float(r["kw"]) for r in rows) / len(rows)
        rt = sum(float(r["rt"]) for r in rows) / len(rows)
        eff = kw / rt if rt else float("nan")
        flag = "ALARM" if eff > config.kw_per_rt_alarm else "OK"
        return f"window={hours}h avg_kw={kw:.1f} avg_rt={rt:.1f} kw_per_rt={eff:.3f} status={flag}"

    yield FunctionInfo.from_fn(
        _kpi,
        # The research tutorial's description, plus what RT means: without it a laptop model read "RT" as
        # "refrigerant temperature" (Week 26 build, nemotron-3-nano on Ollama). Tool descriptions are prompts.
        description="Average chiller plant kW, RT (refrigeration tons of cooling) and kW/RT over the last N hours; "
                    "flags efficiency alarms.",
    )
