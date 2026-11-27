"""Synthetic videos for the video tests: small, generated, and self-identifying.

Each frame writes its own index in binary as six vertical stripes, white for a
set bit and black for a clear one, so a test can read back *which* frame it was
handed without any decoder cooperation. Black and white survive any lossy codec
the wheel might fall back to; flat grey levels did not (MPEG-4 shifted them by
up to six levels, enough to misread a neighbouring frame). Nothing here depends
on a file in the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BITS = 6
WIDTH, HEIGHT = 96, 64
_STRIPE = WIDTH // BITS


def frame_for(index: int) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for bit in range(BITS):
        if index >> bit & 1:
            frame[:, bit * _STRIPE : (bit + 1) * _STRIPE] = 255
    return frame


def index_of(frame: np.ndarray) -> int:
    """Which synthetic frame this is (valid for fewer than 2**BITS frames)."""
    index = 0
    for bit in range(BITS):
        # The centre of the stripe only: chroma subsampling blurs its edges.
        stripe = frame[:, bit * _STRIPE + 4 : (bit + 1) * _STRIPE - 4]
        if float(stripe.mean()) > 127.0:
            index |= 1 << bit
    return index


def write_video(path: Path, frames: int = 24, fps: int = 12, gop: int = 6) -> Path:
    """Encode ``frames`` synthetic frames as H.264 MP4 with a keyframe every ``gop``."""
    import av

    # libx264 where the PyAV wheel carries it; MPEG-4 Part 2 otherwise, which
    # every FFmpeg build has. Either gives the GOP structure the seek tests need.
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = gop
        stream.codec_context.max_b_frames = 0
        stream.options = {"crf": "10"} if codec == "libx264" else {"qscale": "2"}
        for index in range(frames):
            video_frame = av.VideoFrame.from_ndarray(frame_for(index), format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
