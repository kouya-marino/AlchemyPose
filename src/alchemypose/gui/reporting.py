"""How the window tells the user something went wrong.

Injected rather than called directly, so a test never constructs a real modal.
Constructing a real modal while a worker thread is alive has segfaulted in
sibling projects.
"""

from __future__ import annotations

from typing import Protocol


class Reporter(Protocol):
    def error(self, title: str, message: str) -> None: ...


class DialogReporter:
    """The real thing: a modal message box."""

    def error(self, title: str, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, title, message)


class SilentReporter:
    """Records instead of showing. For tests."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, str]] = []

    def error(self, title: str, message: str) -> None:
        self.errors.append((title, message))
