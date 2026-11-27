"""Whole-body human pose visualisation.

``App`` is resolved lazily so that importing this package never pulls in Qt:
``core`` and ``inference`` must stay importable on headless machines and in CI.
"""

from typing import TYPE_CHECKING, Any

#: The one place the version is written. ``pyproject.toml`` declares the version
#: dynamic and reads it back from here, so the distribution metadata and this
#: attribute cannot disagree — and ``tests/unit/test_package.py`` fails loudly if
#: an installed copy ever proves otherwise.
__version__ = "3.0.0"

if TYPE_CHECKING:  # pragma: no cover
    from alchemypose.gui.app import App

__all__ = ["App", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "App":
        from alchemypose.gui.app import App as _App

        return _App
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
