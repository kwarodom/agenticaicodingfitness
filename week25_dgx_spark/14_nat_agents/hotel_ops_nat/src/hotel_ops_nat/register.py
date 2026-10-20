# flake8: noqa
# NAT imports this module through the 'nat.components' entry point (see pyproject.toml).
# Importing the tool modules runs their @register_function decorators.
from .tools import create_maintenance_ticket_function
from .tools import room_temperature_function
