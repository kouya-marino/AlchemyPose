"""The desktop application.

``App`` resolves lazily so that ``import alchemypose.gui.theme`` and friends do
not drag in a QApplication.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from alchemypose.gui.app import App

__all__ = ["App"]


def __getattr__(name: str) -> Any:
    if name == "App":
        from alchemypose.gui.app import App as _App

        return _App
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
