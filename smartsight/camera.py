"""Camera module (108) input, S201.

Source can be a webcam index ("0"), a video file, or the glasses' ESP32-CAM
MJPEG stream URL (stock Arduino CameraWebServer example:
"http://<esp32-cam-ip>:81/stream").

A background thread always holds only the NEWEST frame. Without this,
OpenCV's internal buffer makes a ~1 s/frame pipeline process frames that
are seconds old - unacceptable when the output is a collision warning.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)


class FrameGrabber:
    def __init__(self, source: str):
        self.source = int(source) if source.isdigit() else source
        self._is_file = isinstance(self.source, str) and not self.source.startswith(("http", "rtsp"))
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.ok = False

    def start(self) -> "FrameGrabber":
        threading.Thread(target=self._run, daemon=True, name="camera").start()
        return self

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        cap = None
        fps_delay = 0.0
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    log.warning("camera %r unavailable, retrying", self.source)
                    self.ok = False
                    time.sleep(2)
                    continue
                if self._is_file:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    fps_delay = 1.0 / fps           # play files at real speed
            ok, frame = cap.read()
            if not ok:
                if self._is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video files
                    continue
                cap.release()
                cap, self.ok = None, False
                continue
            with self._lock:
                self._frame, self._seq, self.ok = frame, self._seq + 1, True
            if fps_delay:
                time.sleep(fps_delay)
        if cap is not None:
            cap.release()

    def latest(self):
        """Returns (seq, frame) - seq lets callers skip a frame they already processed."""
        with self._lock:
            return self._seq, self._frame

    def stop(self):
        self._stop.set()
