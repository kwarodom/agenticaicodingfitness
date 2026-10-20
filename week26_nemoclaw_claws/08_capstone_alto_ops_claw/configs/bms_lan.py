#!/usr/bin/env python3
"""Run the course's MOCK BMS (week26/common/bms_mcp_server.py) so the SANDBOX can reach it: bound to the LAN, not 127.0.0.1.

The shared mock binds 127.0.0.1. A sandbox on the Spark cannot reach the host's loopback: OpenShell always blocks
127.0.0.0/8. So the capstone runs the same mock on the Spark host's LAN address, where `bms.alto.local` resolves.
Lab 08-1 copies this file and bms_mcp_server.py into the bundle root, next to data/chiller_plant.csv.

    # on: spark  (from ~/alto-ops-claw-v1, with the NAT venv that has `mcp`)
    BMS_MCP_PORT=8443 python bms_lan.py            # → http://<spark-lan-ip>:8443/mcp · synthetic data · never writes

It is a mock, and it is on your LAN: stop it when the lab is done.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BMS_MCP_PORT", "8443")
import bms_mcp_server as bms  # noqa: E402

bms.mcp.settings.host = os.environ.get("BMS_MCP_HOST", "0.0.0.0")
print(f"◆ mock BMS on {bms.mcp.settings.host}:{bms.PORT}/mcp (LAN) · synthetic data · write_setpoint never writes",
      flush=True)
bms.mcp.run(transport="streamable-http")
