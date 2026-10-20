"""Two hotel-operations tools, registered the NAT 1.9 way (`@register_function` from `nat.plugin_api`).

Both tools are FAKE and DETERMINISTIC: the "building" is the ROOMS table below, and a ticket id is a
hash of (room, issue). Nothing touches a real building system, so an agent can call them as often as
it likes. The pure functions (`read_room`, `make_ticket`) are plain Python so the exercise checker and
Module 15's sandbox labs can test them without an LLM.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMFrameworkEnum
from nat.plugin_api import register_function

# room → (measured °C, setpoint °C, HVAC status). A tiny fake hotel.
ROOMS = {
    "402": (22.8, 23.0, "cooling"),
    "808": (27.9, 23.0, "fault: fan coil unit not responding"),
    "1203": (23.5, 23.0, "cooling"),
    "1510": (20.1, 22.0, "idle (guest away mode)"),
}
PRIORITIES = ("low", "normal", "high", "urgent")


def _room_id(room: str) -> str:
    digits = re.sub(r"\D", "", str(room))
    return digits or str(room).strip()


def read_room(room: str, comfort_band_c: float = 1.5) -> str:
    """Pure function behind the room_temperature tool."""
    rid = _room_id(room)
    if rid not in ROOMS:
        return f"Room {rid}: unknown room. Known rooms: {', '.join(sorted(ROOMS, key=int))}."
    temp, setpoint, hvac = ROOMS[rid]
    delta = round(temp - setpoint, 1)
    verdict = "within comfort band" if abs(delta) <= comfort_band_c else ("too warm" if delta > 0 else "too cold")
    return (f"Room {rid}: {temp:.1f} °C, setpoint {setpoint:.1f} °C ({delta:+.1f} °C, {verdict}). "
            f"HVAC status: {hvac}.")


def make_ticket(room: str, issue: str, priority: str = "normal") -> dict:
    """Pure function behind the create_maintenance_ticket tool. Same room + issue → same ticket id."""
    rid = _room_id(room)
    prio = (priority or "normal").strip().lower()
    if prio not in PRIORITIES:
        prio = "normal"
    key = f"{rid}|{' '.join(issue.lower().split())}"
    tid = "MT-" + hashlib.sha1(key.encode()).hexdigest()[:6].upper()
    return {"ticket_id": tid, "room": rid, "priority": prio, "issue": issue.strip()}


class RoomTemperatureConfig(FunctionBaseConfig, name="room_temperature"):
    """Read the current temperature, setpoint and HVAC status of one hotel room (fake data)."""
    comfort_band_c: float = Field(default=1.5, description="±°C around the setpoint that counts as comfortable.")


@register_function(config_type=RoomTemperatureConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def room_temperature_function(config: RoomTemperatureConfig, builder: Builder):

    async def _room_temperature(room: str) -> str:
        """Get the current temperature, setpoint and HVAC status of a hotel room.

        Args:
            room (str): The room number, for example "808".

        Returns:
            str: One line with the temperature, setpoint, comfort verdict and HVAC status.
        """
        return read_room(room, config.comfort_band_c)

    yield FunctionInfo.from_fn(_room_temperature, description=_room_temperature.__doc__)


class MaintenanceTicketConfig(FunctionBaseConfig, name="create_maintenance_ticket"):
    """Open a maintenance ticket for a room (fake: returns a deterministic id, optionally logs to a file)."""
    ticket_log: str | None = Field(default=None,
                                   description="Optional JSONL file to append tickets to. None = keep nothing.")


@register_function(config_type=MaintenanceTicketConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def create_maintenance_ticket_function(config: MaintenanceTicketConfig, builder: Builder):

    async def _create_maintenance_ticket(room: str, issue: str, priority: str = "normal") -> str:
        """Create a maintenance ticket for a hotel room.

        Args:
            room (str): The room number, for example "808".
            issue (str): A short description of the problem.
            priority (str): One of low, normal, high, urgent.

        Returns:
            str: The ticket id and a one-line summary.
        """
        t = make_ticket(room, issue, priority)
        if config.ticket_log:
            p = Path(config.ticket_log).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**t, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        return f"Created ticket {t['ticket_id']} for room {t['room']} (priority {t['priority']}): {t['issue']}"

    yield FunctionInfo.from_fn(_create_maintenance_ticket, description=_create_maintenance_ticket.__doc__)
