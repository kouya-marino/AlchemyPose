"""Getting the result out: the numbers, and the picture.

Three plain functions, each taking what it needs and a path. No dialog, no
window, no status bar — the window layers those on top. That is what lets the
whole of export be tested without a modal, and it is the same reason
``reporting.Reporter`` exists.

The PNG renders the *scene*, not the widget. Rendering the widget would bake in
whatever the user happened to be looking at: the zoom, the scroll position, the
window's aspect ratio, the empty canvas either side of a portrait. The scene is
the photograph plus the skeleton drawn over it in image-pixel coordinates, so
rendering it 1:1 gives back the source image's own resolution and the same file
whatever the view is doing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QFileDialog, QWidget

from alchemypose.core.schema import PoseFrame, PoseResult
from alchemypose.core.serialize import to_csv, to_json
from alchemypose.errors import AlchemyPoseError
from alchemypose.gui.canvas.view import PoseCanvas


class ExportError(AlchemyPoseError):
    """An export could not be written. Carries what to tell the user."""


def _write_text(text: str, path: Path) -> None:
    """Write, turning the filesystem's refusals into one reportable error.

    A path the user typed can name a directory that does not exist, one they
    cannot write to, or a name the filesystem rejects. All three are ordinary
    and none of them is a bug, so none of them may reach the user as a
    traceback.
    """
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"Could not write {path}: {exc.strerror or exc}") from exc


def export_json(result: PoseResult, path: Path) -> None:
    """Every keypoint, with the topology inline so the file outlives this package."""
    _write_text(to_json(result), path)


def export_csv(result: PoseResult, path: Path) -> None:
    """One row per keypoint — the flat form analysts ask for."""
    _write_text(to_csv(result), path)


def export_glb(
    result: PoseResult, path: Path, frames: tuple[PoseFrame, ...] | None = None, track_id: int = 0
) -> None:
    """One person's 3D motion as an animated, skinned skeleton (``core.gltf``)."""
    from alchemypose.core.gltf import NoMotion, skeleton_glb

    try:
        payload = skeleton_glb(result, frames=frames, track_id=track_id)
    except NoMotion as exc:
        raise ExportError(str(exc)) from exc
    try:
        path.write_bytes(payload)
    except OSError as exc:
        raise ExportError(f"Could not write {path}: {exc.strerror or exc}") from exc


def export_png(canvas: PoseCanvas, path: Path) -> None:
    """Render the canvas's scene at 1:1 with the source image.

    Independent of the current zoom and of the window's size: the source
    rectangle is the scene rectangle, which ``PoseCanvas.set_image`` pins to the
    photograph, and the target is the same rectangle at the same size.
    """
    rect: QRectF = canvas.scene().sceneRect()
    width, height = round(rect.width()), round(rect.height())
    if width <= 0 or height <= 0:
        raise ExportError("There is nothing on the canvas to export yet.")

    image = QImage(width, height, QImage.Format.Format_ARGB32)
    if image.isNull():
        raise ExportError(f"Could not allocate a {width}x{height} image to export into.")
    # Transparent, not the canvas colour: the scene's own background is the
    # photograph, and anything outside it is not part of the export.
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        canvas.scene().render(painter, target=QRectF(image.rect()), source=rect)
    finally:
        # The QImage cannot be saved while a painter is still active on it, and
        # an exception from render() would otherwise leave one there.
        painter.end()

    if not image.save(str(path)):
        raise ExportError(f"Could not write {path}. Check the folder exists and is writable.")


class SavePrompt(Protocol):
    """Asks the user where to put a file. Injected, so a test never opens one."""

    def ask(self, parent: QWidget, caption: str, suggested_name: str, file_filter: str) -> Path | None:
        """The chosen path, or ``None`` if the user cancelled."""
        ...


class DialogSavePrompt:
    """The real thing: the platform's save dialog."""

    def ask(self, parent: QWidget, caption: str, suggested_name: str, file_filter: str) -> Path | None:
        name, _ = QFileDialog.getSaveFileName(parent, caption, suggested_name, file_filter)
        return Path(name) if name else None


class FixedSavePrompt:
    """Answers with a path decided in advance, and records what it was asked. For tests."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.asked: list[tuple[str, str, str]] = []

    def ask(self, parent: QWidget, caption: str, suggested_name: str, file_filter: str) -> Path | None:
        self.asked.append((caption, suggested_name, file_filter))
        return self.path
