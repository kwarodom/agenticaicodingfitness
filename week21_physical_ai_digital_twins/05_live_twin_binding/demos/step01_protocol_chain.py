#!/usr/bin/env python3
"""PART 1 · The protocol chain — field bus to prim  [BEGINNER]

Field devices speak BACnet/IP or Modbus, and those field buses almost never touch
the twin directly. A BMS or edge gateway (Niagara Framework-class) normalizes the
reading, publishes it to MQTT or Kafka, and a BINDING SERVICE does the ID join —
BACnet object ↔ point ↔ equipment tag ↔ IFC GlobalId ↔ USD prim path — before a
single USD attribute changes.

Dual-mode: if `paho-mqtt` is installed AND a broker answers on MQTT_BROKER
(default 127.0.0.1:1883 — e.g. a local mosquitto), hops 3–4 ride a REAL broker;
if `usd-core` is installed, hop 5 authors a REAL pxr session layer. Otherwise an
in-process mini bus + printed USD stand in, same chain, same lesson.

Run:  python demos/step01_protocol_chain.py
      MQTT_BROKER=127.0.0.1:1883 python demos/step01_protocol_chain.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

CHAIN = """\
  field device ──BACnet/IP · Modbus──► BMS / edge gateway ──normalize──► MQTT / Kafka
   (sensor,        (field buses —        (Niagara-class:       (point message:
    actuator)       never touch the       polls the bus,        tag·value·unit·ts)
                    twin directly)        maps object IDs)            │
                                                                      ▼
       USD attribute on a prim ◄──author update── BINDING SERVICE (the ID join)
"""


def _real_broker():
    """A connected paho-mqtt client if a broker answers on MQTT_BROKER, else None."""
    host, _, port = os.environ.get("MQTT_BROKER", "127.0.0.1:1883").partition(":")
    try:
        import paho.mqtt.client as mqtt  # noqa: PLC0415
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="week21-binding-demo")
        c.connect(host, int(port or 1883), keepalive=10)
        return c, f"{host}:{port or 1883}"
    except Exception:
        return None, None


class RealBus:
    """sim.Bus-shaped adapter over a genuine MQTT broker (hops 3–4 for real)."""

    def __init__(self, client):
        self.client = client

    def subscribe(self, pattern: str, cb) -> None:
        def on_msg(_cl, _ud, m):
            cb(m.topic, json.loads(m.payload.decode()))
        self.client.on_message = on_msg
        self.client.subscribe(pattern, qos=1)
        self.client.loop_start()

    def publish(self, topic: str, msg: dict) -> None:
        self.client.publish(topic, json.dumps(msg), qos=1).wait_for_publish(3)
        time.sleep(0.3)              # let the round-trip deliver before the next print


def _pxr_stage():
    """A real in-memory pxr stage with the demo's prims, or None without usd-core."""
    try:
        from pxr import Usd  # noqa: PLC0415
    except ImportError:
        return None
    stage = Usd.Stage.CreateInMemory()
    for p in sim.POINTS:
        stage.DefinePrim(p["prim"], "Xform")
    return stage


def trace(bus, stage, point_name: str, raw: float, verbose: bool) -> None:
    p = sim.POINT_BY_NAME[point_name]
    topic = sim.topic_for(point_name)

    def binding_service(t: str, msg: dict) -> None:      # subscriber = hop 4+5
        rec = sim.POINT_BY_NAME[msg["point"]]
        # USD identifiers allow [A-Za-z0-9_] only — 'floor-5 submeter' needs sanitizing
        attr = "iot:" + "".join(c if c.isalnum() else "_" for c in msg["kind"])
        if stage is not None:                             # hop 5 FOR REAL (pxr)
            from pxr import Sdf, Usd  # noqa: PLC0415
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                prim = stage.GetPrimAtPath(rec["prim"])
                prim.CreateAttribute(attr, Sdf.ValueTypeNames.Double).Set(msg["value"])
        if verbose:
            print(f"  HOP 4  binding service ◁ {t}")
            print(f"         the ID join:  {rec['obj']}")
            print(f"                     ↔ point   {rec['point']}")
            print(f"                     ↔ tag     {rec['tag']}")
            print(f"                     ↔ GlobalId {rec['guid']}   (preserved in App 3!)")
            print(f"                     ↔ prim    {rec['prim']}")
            how = "authored on a REAL pxr session layer" if stage is not None \
                  else "session layer — Ch 3 explains why"
            print(f"  HOP 5  USD update ({how}):")
            print(f"         over \"{rec['prim']}\" {{ custom double {attr} = {msg['value']} }}")
        else:
            print(f"    {rec['obj']:<30} → {msg['point']:<16} → {rec['prim']}"
                  f"  iot:… = {msg['value']}")

    bus.subscribe("bldg/#", binding_service)
    if verbose:
        print(f"  HOP 1  field device: {p['bus']} · {p['obj']} presentValue = {raw}")
        print(f"         ({p['kind']} — a raw field-bus object; no names, no geometry)")
        print(f"  HOP 2  gateway normalizes → {{'point': '{p['point']}', 'value': {raw}, "
              f"'unit': '{p['unit']}', 'ts': '…'}}")
        print(f"  HOP 3  publish → MQTT topic {topic}")
    bus.publish(topic, {"point": point_name, "value": raw,
                        "unit": p["unit"], "kind": p["kind"]})
    print()


def main() -> None:
    view.banner("PART 1", "The protocol chain — field bus to prim", "BEGINNER")

    client, broker = _real_broker()
    stage = _pxr_stage()
    if client:
        bus: object = RealBus(client)
        usd_note = "hop 5 authors a REAL pxr session layer" if stage is not None \
                   else "hop 5 printed (pip install usd-core to author it)"
        print(f"▣ MODE: REAL — hops 3–4 ride a genuine MQTT broker at {broker}")
        print(f"  (paho-mqtt · QoS 1 · topic bldg/#) · {usd_note}.\n")
    else:
        bus = sim.Bus()
        print("▣ MODE: SIM — no MQTT broker on MQTT_BROKER (default 127.0.0.1:1883);")
        print("  an in-process mini pub/sub stands in. Same chain, same lesson.")
        print("  # make it real:  docker run -d -p 1883:1883 eclipse-mosquitto \\")
        print("  #                    mosquitto -c /mosquitto-no-auth.conf\n")

    print("The chain every production building twin uses:\n")
    print(CHAIN)
    print("One reading, all five hops (AHU-3 supply air temp, BACnet/IP):\n")
    trace(bus, stage, "AHU-3.SAT", 18.4, verbose=True)

    print("Two more, compact — note the Modbus meter takes the exact same path:\n")
    trace(bus, stage, "VAV-5-01.ZNT", 23.1, verbose=False)
    trace(bus, stage, "MTR-5.kWh", 41230.5, verbose=False)

    if stage is not None:
        print("Proof — the REAL session layer pxr now holds (authored, RAM-only):\n")
        for line in stage.GetSessionLayer().ExportToString().splitlines():
            if line.strip().startswith(("custom", "over")) or line.strip() in ("{", "}"):
                print(f"  │ {line}")
        print("  │  … and the authored discipline layers are untouched (Ch 3's lesson).\n")

    print("Why the twin never speaks BACnet itself:")
    print("  • field buses are chatty, polled, and vendor-quirky — the gateway absorbs that;")
    print("  • MQTT/Kafka decouples producers from consumers (the twin is just one subscriber);")
    print("  • the binding service is a pure translation + ID-join layer — the join chain")
    print("    BACnet object ↔ point ↔ tag ↔ GlobalId ↔ prim is what App 3's GUID")
    print("    preservation bought you.\n")

    print("go further (NVIDIA's reference IoT→USD connector, same architecture):")
    print("  $ git clone https://github.com/NVIDIA-Omniverse/iot-samples")
    print("  # its connector service subscribes to a real MQTT broker and writes USD\n")

    print("Takeaway: binding is plumbing plus one non-negotiable join table. Next: WHERE")
    print("those USD writes land — session layers and Fabric, so live data never dirties")
    print("your authored asset files.")

    if client:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
