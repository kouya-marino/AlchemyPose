"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from alchemypose.gui.main_window import MainWindow


class App(QApplication):
    """A QApplication that knows its own window."""

    def __init__(self, argv: list[str] | None = None) -> None:
        super().__init__(argv if argv is not None else [])
        self.setApplicationName("AlchemyPose")
        self.window = MainWindow()


def main() -> int:
    app = App(sys.argv)
    app.window.show()
    return app.exec()
