"""WildFakeTools — image-manipulation tools distilled from public web discourse.

Every module in this package (other than ``base_tool``) defines exactly one
:class:`~.base_tool.BaseTool` subclass implementing a single mined technique.
Tools are resolved lazily by name, so importing this package does not import
all of them.

    from tools import get_tool_by_name
    tool = get_tool_by_name("phone_camera_look")
    out = tool.apply(img, {})          # every parameter has a default
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

from .base_tool import BaseTool
from ._diffusion import DIFFUSION_TOOLS, DiffusionBackendUnavailable

__all__ = [
    "BaseTool",
    "available_tools",
    "get_tool_by_name",
    "get_all_tools",
    "DIFFUSION_TOOLS",
    "DiffusionBackendUnavailable",
]

_DIR = Path(__file__).resolve().parent


def available_tools():
    """Sorted names of every tool in the suite."""
    return sorted(
        m.name
        for m in pkgutil.iter_modules([str(_DIR)])
        if m.name != "base_tool" and not m.name.startswith("_")
    )


def get_tool_by_name(name):
    """Instantiate the tool implemented by module ``name``.

    Raises LookupError if the module defines no BaseTool subclass of its own.
    """
    module = importlib.import_module(f".{name}", __name__)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BaseTool)
            and obj is not BaseTool
            and obj.__module__ == module.__name__
        ):
            return obj()
    raise LookupError(f"no BaseTool subclass defined in tool module {name!r}")


def get_all_tools():
    """Instantiate every tool in the suite."""
    return [get_tool_by_name(n) for n in available_tools()]
