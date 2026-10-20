#!/usr/bin/env python3
"""A MOCK BMS exposed over MCP — the "bms.alto.local" of the research tutorial, running on this laptop.

Four tools, the ones the tutorial's policies name: read_point, list_alarms, get_trend, write_setpoint.
Values come from week26/common/data/chiller_plant.csv (synthetic). write_setpoint NEVER changes anything:
it refuses without a work_order_id and otherwise answers "simulated" — it exists so policies (Module 03/07)
and NAT's mcp_client (Module 04) have a dangerous tool to deny.

    week26/.venv-nat/bin/python week26/common/bms_mcp_server.py            # → http://localhost:8443/mcp
    week26/.venv-nat/bin/nat mcp client tool list --url http://localhost:8443/mcp
"""
import csv
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA = Path(__file__).resolve().parent / "data" / "chiller_plant.csv"
PORT = int(os.environ.get("BMS_MCP_PORT", "8443"))
mcp = FastMCP("Alto mock BMS", host="127.0.0.1", port=PORT)


def _rows() -> list[dict]:
    with DATA.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@mcp.tool()
def read_point(point: str) -> str:
    """Read the latest value of a BMS point: PLANT.KW, PLANT.RT or PLANT.KW_PER_RT."""
    last = _rows()[-1]
    vals = {"PLANT.KW": float(last["kw"]), "PLANT.RT": float(last["rt"])}
    vals["PLANT.KW_PER_RT"] = round(vals["PLANT.KW"] / vals["PLANT.RT"], 3)
    if point not in vals:
        return f"unknown point {point!r}; try one of {', '.join(vals)}"
    return f"{point}={vals[point]} at {last['ts']} (mock BMS, synthetic data)"


@mcp.tool()
def list_alarms() -> str:
    """List active plant alarms (kW/RT above 0.85 for the last hour)."""
    last = _rows()[-4:]
    eff = sum(float(r["kw"]) for r in last) / sum(float(r["rt"]) for r in last)
    return (f"ALARM plant efficiency {eff:.3f} kW/RT > 0.85 since {last[0]['ts']}" if eff > 0.85
            else "no active alarms")


@mcp.tool()
def get_trend(point: str = "PLANT.KW_PER_RT", hours: int = 6) -> str:
    """Hourly averages of PLANT.KW_PER_RT for the last N hours."""
    rows = _rows()[-max(1, hours) * 4:]
    out = []
    for i in range(0, len(rows), 4):
        chunk = rows[i:i + 4]
        eff = sum(float(r["kw"]) for r in chunk) / sum(float(r["rt"]) for r in chunk)
        out.append(f"{chunk[0]['ts']} {eff:.3f}")
    return f"{point} hourly: " + "; ".join(out)


@mcp.tool()
def write_setpoint(point: str, value: float, work_order_id: str = "") -> str:
    """WRITE a setpoint to equipment. Dangerous — policies should deny this tool. (Mock: never writes.)"""
    if not work_order_id:
        return "REFUSED: write_setpoint requires an approved work_order_id (mock BMS — nothing written)"
    return f"SIMULATED: would set {point}={value} under {work_order_id} (mock BMS — nothing written)"


if __name__ == "__main__":
    print(f"◆ mock BMS MCP server → http://localhost:{PORT}/mcp (streamable-http) · synthetic data · Ctrl-C to stop",
          flush=True)
    mcp.run(transport="streamable-http")
