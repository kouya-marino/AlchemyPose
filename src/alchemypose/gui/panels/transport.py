"""Play, pause and scrub: the strip under the canvas in video mode."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

from alchemypose.gui import theme


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)}:{rest:05.2f}"


class TransportBar(QWidget):
    """A scrubber that reports frame indices, never seconds.

    The slider's range is the frame count and its value *is* ``frame_index``,
    so there is no rounding between what is under the playhead and what the
    result calls that frame. A second, thinner bar under it shows how far the
    analysis has got while it is still running.
    """

    position_changed = Signal(int)
    play_toggled = Signal(bool)
    smoothing_toggled = Signal(bool)
    speed_changed = Signal(float)

    #: Playback rates offered. Slower ones are for checking a move the robot
    #: cannot make at full speed.
    SPEEDS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

    def __init__(self) -> None:
        super().__init__()
        self._fps = 30.0
        self._frame_count = 0

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setFixedWidth(64)
        self.play_button.setToolTip("Play / pause (Space)")
        self.play_button.toggled.connect(self._on_play_toggled)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.position_changed.emit)

        self.time_label = QLabel(format_time(0.0))
        self.time_label.setMinimumWidth(150)
        self.time_label.setStyleSheet(f"color: {theme.TEXT_DIM};")

        self.analysis_label = QLabel("")
        self.analysis_label.setStyleSheet(f"color: {theme.TEXT_DIM};")

        self.smooth_box = QCheckBox("Smooth")
        self.smooth_box.setChecked(True)
        self.smooth_box.setToolTip("Steady the skeleton between frames (display only; exports stay raw)")
        self.smooth_box.toggled.connect(self.smoothing_toggled.emit)

        self.speed_box = QComboBox()
        for speed in self.SPEEDS:
            self.speed_box.addItem(f"{speed:g}×", speed)
        self.speed_box.setToolTip("Playback speed")
        self.speed_box.currentIndexChanged.connect(
            lambda _i: self.speed_changed.emit(float(self.speed_box.currentData()))
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(self.play_button)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.time_label)
        layout.addWidget(self.analysis_label)
        layout.addWidget(self.speed_box)
        layout.addWidget(self.smooth_box)

    def configure(self, frame_count: int, fps: float) -> None:
        """Set up for a newly opened video, parked on frame 0."""
        self._frame_count = frame_count
        self._fps = fps if fps > 0 else 30.0
        self.slider.blockSignals(True)
        try:
            self.slider.setRange(0, max(0, frame_count - 1))
            self.slider.setValue(0)
        finally:
            self.slider.blockSignals(False)
        self.set_playing(False)
        self._update_time(0)

    @property
    def position(self) -> int:
        return int(self.slider.value())

    def show_position(self, index: int) -> None:
        """Move the playhead without announcing it (playback drives this)."""
        self.slider.blockSignals(True)
        try:
            self.slider.setValue(index)
        finally:
            self.slider.blockSignals(False)
        self._update_time(index)

    def set_position(self, index: int) -> None:
        """Move the playhead *and* announce it, as a user's scrub would."""
        self.slider.setValue(index)
        self._update_time(index)

    def set_playing(self, playing: bool) -> None:
        self.play_button.blockSignals(True)
        try:
            self.play_button.setChecked(playing)
            self.play_button.setText("Pause" if playing else "Play")
        finally:
            self.play_button.blockSignals(False)

    @property
    def playing(self) -> bool:
        return bool(self.play_button.isChecked())

    @property
    def speed(self) -> float:
        return float(self.speed_box.currentData())

    def suggest_speed(self, fraction: float) -> None:
        """Pick the fastest offered rate at or below ``fraction`` (1.0 leaves the choice alone)."""
        if fraction >= 1.0:
            return
        fitting = [s for s in self.SPEEDS if s <= fraction] or [self.SPEEDS[-1]]
        self.speed_box.setCurrentIndex(self.SPEEDS.index(max(fitting)))

    def set_analysis_progress(self, done: int, total: int) -> None:
        self.analysis_label.setText("" if done >= total else f"analysing {done}/{total}")

    def _update_time(self, index: int) -> None:
        last = max(0, self._frame_count - 1)
        self.time_label.setText(
            f"{format_time(index / self._fps)} / {format_time(last / self._fps)}  ·  {index}"
        )

    def _on_play_toggled(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        self.play_toggled.emit(playing)
