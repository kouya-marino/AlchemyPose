#!/usr/bin/env python3
"""Open the AlchemyPose window from a checkout: ``python main.py``.

For an installed copy, a bare ``alchemypose`` opens the window, and so do
``alchemypose gui`` and ``python -m alchemypose``; any of them is preferable,
because they work from any directory. This file covers the case none of them
does: a bare ``git clone`` with nothing installed, where a ``src/`` layout is
unimportable until something is.

It also exists because the sibling applications in this suite are started with a
plain ``python main.py``, so it is the first thing anyone who knows them will
try here.

The work is delegated to the ``gui`` command rather than reimplemented, so this
cannot drift from ``alchemypose gui`` — same window, same exit code, and the
same message if PySide6 is missing. The command is named rather than left off,
even though the bare form now opens the window too: naming it is what keeps this
launcher meaning one thing if the default ever changes again.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"


def ensure_importable() -> bool:
    """Put ``src/`` on the path if ``alchemypose`` is not already importable.

    Returns whether it had to. An installed copy is deliberately left alone:
    prepending ``src/`` unconditionally would shadow it with the working tree,
    which is a surprising thing for a launcher to do when the two differ.
    """
    try:
        import alchemypose  # noqa: F401
    except ImportError:
        if not SRC.is_dir():
            raise
        sys.path.insert(0, str(SRC))
        return True
    return False


def main() -> None:
    """Run the ``gui`` command, as ``alchemypose gui`` does."""
    ensure_importable()
    # Imported after the path is arranged, or this would fail in the very case
    # the bootstrap above exists to handle.
    from alchemypose.cli import app

    app(["gui"])


if __name__ == "__main__":
    main()
