"""Video mode for the window: a decoded clip, its analysis, and a playhead.

The analysis runs once, in the background, over every frame (see
``inference.video`` for why that is not the latest-request-wins worker). While
it runs, the clip can already be played and scrubbed: frames that have been
analysed draw their skeletons, frames that have not draw the photograph alone.
When it finishes, people are renumbered by how much of the clip they are in,
so "person 1" is whoever is on screen longest, and the smoothed copy used for
drawing is computed once.

Nothing here touches the stored result's coordinates; ``smoothed`` is a
separate tuple that only the display reads.

Threading follows ``gui/bridge.py``: the analysis thread is plain
``threading`` and only ever *emits*; every slot runs on the GUI thread via a
queued connection, and nothing on the analysis thread touches a widget.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from alchemypose.core.schema import PoseFrame, PoseResult
from alchemypose.core.smoothing import smooth_frames
from alchemypose.core.tracking import by_track
from alchemypose.inference.video import AnalysisCancelled, FrameEngine, cached_analysis
from alchemypose.media.video import VideoInfo, VideoReader, open_video

#: A track has to be in at least this share of a clip's frames to get a person
#: chip (and at least ``MIN_TRACK_FRAMES``). A 1080p dance clip picks up
#: one-to-three-frame detections of distant passers-by; they stay in the stored
#: result, but seven chips for one dancer is noise.
MIN_TRACK_SHARE = 0.05
MIN_TRACK_FRAMES = 5


def significant_tracks(frames: tuple[PoseFrame, ...]) -> int:
    """How many leading track ids are worth a chip.

    Ids are dense and ordered by presence (``renumber_by_presence``), so the
    significant ones are a prefix: count until the first that falls short.
    """
    counts: dict[int, int] = {}
    for frame in frames:
        for person in frame.people:
            if person.track_id is not None:
                counts[person.track_id] = counts.get(person.track_id, 0) + 1
    floor = max(MIN_TRACK_FRAMES, int(round(MIN_TRACK_SHARE * len(frames))))
    tracks = 0
    while counts.get(tracks, 0) >= floor:
        tracks += 1
    return tracks


class VideoController(QObject):
    """Owns one open clip: its reader, its analysis, and where the playhead is."""

    #: ``(index, bgr, view)`` — the frame to show and the one-frame result to
    #: draw over it (``None`` while that frame is not analysed yet).
    frame_ready = Signal(int, object, object)
    analysis_progress = Signal(int, int)        # done, total
    analysis_finished = Signal(object, bool)    # PoseResult, was_cached
    analysis_failed = Signal(str)
    playing_changed = Signal(bool)

    # Emitted from the analysis thread; connected to private slots, so queued.
    _frame_analysed = Signal(object, int, int)
    _finished = Signal(object, bool, int)
    _failed = Signal(str, int)

    def __init__(
        self,
        engine_factory: Callable[[], FrameEngine],
        preset: str,
        parent: QObject | None = None,
        reader_factory: Callable[[Path], VideoReader] = open_video,
    ) -> None:
        super().__init__(parent)
        self._engine_factory = engine_factory
        self._preset = preset
        self._reader_factory = reader_factory

        self._reader: VideoReader | None = None
        self._path: Path | None = None
        self._partial: dict[int, PoseFrame] = {}
        self._result: PoseResult | None = None
        self._smoothed: tuple[PoseFrame, ...] | None = None
        self._smoothing = True
        self._tracks = 0
        self._index = 0
        self._generation = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        #: Wall-clock anchor for playback: (monotonic seconds, frame index) at play().
        self._clock: tuple[float, int] = (0.0, 0)
        self.frames_skipped = 0

        self._frame_analysed.connect(self._on_frame_analysed)
        self._finished.connect(self._on_finished)
        self._failed.connect(self._on_failed)

    def set_engine(self, engine_factory: Callable[[], FrameEngine], preset: str) -> None:
        """Use a different engine for the next clip opened. The current one is not re-run here."""
        self._engine_factory = engine_factory
        self._preset = preset

    # -------------------------------------------------------------- queries

    @property
    def is_open(self) -> bool:
        return self._reader is not None

    @property
    def info(self) -> VideoInfo | None:
        return self._reader.info if self._reader is not None else None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def result(self) -> PoseResult | None:
        """The finished analysis, raw. ``None`` until it finishes."""
        return self._result

    @property
    def smoothed_frames(self) -> tuple[PoseFrame, ...] | None:
        return self._smoothed

    @property
    def track_count(self) -> int:
        return self._tracks

    @property
    def index(self) -> int:
        return self._index

    @property
    def analysing(self) -> bool:
        return self._thread is not None and self._result is None

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    # ------------------------------------------------------------ lifecycle

    def open(self, path: Path) -> VideoInfo:
        """Open ``path``, show frame 0, and start analysing in the background."""
        self.close()
        reader = self._reader_factory(path)
        self._reader, self._path = reader, path
        self._generation += 1
        self._partial, self._result, self._smoothed, self._tracks = {}, None, None, 0
        self._index = 0
        self._stop = threading.Event()
        generation, stop = self._generation, self._stop

        # The analysis decodes with its own reader: a reader is a cursor, and
        # the playhead and the analysis must not fight over one.
        def run() -> None:
            try:
                analysis_reader = self._reader_factory(path)
                try:
                    result, cached = cached_analysis(
                        path,
                        self._engine_factory,
                        analysis_reader,
                        preset=self._preset,
                        on_frame=lambda frame, done, total: self._frame_analysed.emit(frame, done, total),
                        should_stop=stop.is_set,
                    )
                finally:
                    analysis_reader.close()
            except AnalysisCancelled:
                return
            except Exception as exc:  # noqa: BLE001 - reported, never raised into Qt
                if not stop.is_set():
                    self._failed.emit(str(exc), generation)
                return
            if not stop.is_set():
                self._finished.emit(result, cached, generation)

        self._thread = threading.Thread(target=run, name="alchemypose-video", daemon=True)
        self._thread.start()
        self.show(0)
        return reader.info

    def close(self, timeout: float = 2.0) -> None:
        self.pause()
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._path = None

    # ------------------------------------------------------------- playback

    def play(self) -> None:
        info = self.info
        if info is None:
            return
        if self._index >= info.frame_count - 1:
            self.show(0)                          # play from the start again
        self._clock = (time.monotonic(), self._index)
        self.frames_skipped = 0
        # Ticks come faster than frames; each tick shows whichever frame the
        # wall clock says is due, so playback keeps real time even when a
        # 1080p frame takes longer than 1/fps to decode and draw.
        self._timer.start(max(1, int(1000.0 / info.fps / 2)))
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playing_changed.emit(False)

    def set_smoothing(self, enabled: bool) -> None:
        self._smoothing = enabled
        if self._reader is not None:
            self.show(self._index)

    def _tick(self) -> None:
        info = self.info
        if info is None:
            self.pause()
            return
        started_at, started_index = self._clock
        due = started_index + int((time.monotonic() - started_at) * info.fps)
        if due >= info.frame_count:
            self.show(info.frame_count - 1)
            self.pause()
            return
        if due <= self._index:
            return                                # this frame is still on screen for a while
        self.frames_skipped += due - self._index - 1
        self.show(due)

    def show(self, index: int) -> None:
        """Decode ``index`` and announce it with whatever pose is known for it."""
        if self._reader is None:
            return
        info = self._reader.info
        index = max(0, min(index, max(0, info.frame_count - 1)))
        try:
            bgr = self._reader.read(index)
        except Exception as exc:  # noqa: BLE001 - a bad frame stops playback, not the app
            self.pause()
            self.analysis_failed.emit(f"could not read frame {index}: {exc}")
            return
        self._index = index
        self.frame_ready.emit(index, bgr, self.view(index))

    def view(self, index: int) -> PoseResult | None:
        """The one-frame result to draw at ``index``, or ``None`` if not analysed yet.

        After the analysis finishes, people are re-ordered so index ``i`` is
        track ``i`` (see ``tracking.by_track``) and the smoothed copy is used
        when smoothing is on. Before that, whatever has been analysed is shown
        as it came, untracked as far as the chips are concerned.
        """
        info = self.info
        if info is None:
            return None
        if self._result is not None:
            frames = self._smoothed if (self._smoothing and self._smoothed is not None) else self._result.frames
            if index >= len(frames):
                return None
            frame = by_track(frames[index], self._tracks, len(self._result.topology))
            return replace(self._result, frames=(frame,))
        partial = self._partial.get(index)
        if partial is None:
            return None
        from alchemypose.core.schema import Camera
        from alchemypose.core.topology import COCO_WHOLEBODY_133

        return PoseResult(
            topology=COCO_WHOLEBODY_133,
            frames=(partial,),
            source=str(self._path),
            model="",
            camera=Camera.default_for(info.width, info.height),
        )

    # ------------------------------------------------------ analysis results

    def _on_frame_analysed(self, frame: PoseFrame, done: int, total: int) -> None:
        if self._result is not None:
            return
        self._partial[frame.frame_index] = frame
        self.analysis_progress.emit(done, total)
        if frame.frame_index == self._index and not self.playing:
            self.show(self._index)        # the frame under a parked playhead just got its pose

    def _on_finished(self, result: PoseResult, cached: bool, generation: int) -> None:
        if generation != self._generation:
            return                                  # a straggler from a clip since replaced
        info = self.info
        fps = info.fps if info is not None else 30.0
        self._result = result
        self._partial = {}
        self._tracks = significant_tracks(result.frames)
        self._smoothed = smooth_frames(result.frames, fps)
        total = len(result.frames)
        self.analysis_progress.emit(total, total)
        self.analysis_finished.emit(result, cached)
        if not self.playing:
            self.show(self._index)

    def _on_failed(self, message: str, generation: int) -> None:
        if generation == self._generation:
            self.analysis_failed.emit(message)

