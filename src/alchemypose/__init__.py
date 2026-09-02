"""Whole-body human pose visualisation.

``App`` is resolved lazily so that importing this package never pulls in Qt:
``core`` and ``inference`` must stay importable on headless machines and in CI.
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:  # pragma: no cover
    from alchemypose.gui.app import App

__all__ = ["App", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "App":
        from alchemypose.gui.app import App as _App

        return _App
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
