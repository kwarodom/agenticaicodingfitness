#!/usr/bin/env python3
"""Wire the whole harness together: state · tools · policy · relay · brain · agents · flywheel."""
from __future__ import annotations

from dataclasses import dataclass

from .world import fresh_state, Tools, HotelState
from .policy import Gateway
from .relay import Relay
from .brain import make_brain
from .agents import Orchestrator
from .flywheel import Flywheel


@dataclass
class Runtime:
    state: HotelState
    tools: Tools
    gateway: Gateway
    relay: Relay
    orch: Orchestrator
    fly: Flywheel
    mode: str


def build() -> Runtime:
    state = fresh_state()
    tools = Tools(state)
    gateway = Gateway(state)
    relay = Relay()
    brain, mode = make_brain()
    orch = Orchestrator(state, tools, gateway, relay, brain)
    return Runtime(state, tools, gateway, relay, orch, Flywheel(), mode)
