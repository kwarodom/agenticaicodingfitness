# NAT discovers this module through the `nat.components` entry point in pyproject.toml.
# Importing a module runs its @register_function decorators.
from . import chiller_tool  # noqa: F401
from . import setpoint_tool  # noqa: F401
