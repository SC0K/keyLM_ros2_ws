"""Opt-in MP4 capture; preserve wall-clock duration when render frames are late."""

from pathlib import Path
import math
import time

import cv2


class VideoRecording:
    def __init__(self, path, fps, width, height):
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("recording_path must end in .mp4")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Recording FPS must be finite and positive")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("MP4 frame dimensions must be positive even numbers")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve exclusively: never overwrite footage from an earlier run.
        with self.path.open("xb"):
            pass
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Cannot open MP4 encoder for {self.path}")
        self.fps = fps
        self.start = None
        self.frames = 0
        self.last_frame = None

    def write(self, frame, now=None):
        now = time.monotonic() if now is None else now
        if self.start is None:
            self.start = now
        index = int(max(0., now - self.start) * self.fps)
        # Repeat the last image across missed capture slots rather than speeding
        # up the experiment when rendering is slower than the requested FPS.
        while self.frames < index and self.last_frame is not None:
            self.writer.write(self.last_frame)
            self.frames += 1
        if self.frames <= index:
            self.writer.write(frame)
            self.frames += 1
        self.last_frame = frame.copy()

    def close(self):
        self.writer.release()
