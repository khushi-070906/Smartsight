"""S202/S203 - Frame relevance filter.

Cheap checks (well under 2 ms on a 160 px thumbnail) that decide whether a
frame is worth sending through detection + depth (~750 ms/frame on CPU in
the v2t-graph latency runs). Rejects:
    - frames arriving faster than max_fps
    - motion-blurred frames (variance of Laplacian)
    - too dark / blown-out frames
    - frames nearly identical to the last accepted one (static scene),
      except one forced frame every force_every_s so a static-but-dangerous
      scene is still re-checked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import FilterConfig

THUMB_W = 160


@dataclass
class FilterDecision:
    useful: bool
    reason: str
    sharpness: float = 0.0
    brightness: float = 0.0
    change: float = 0.0


class FrameRelevanceFilter:
    def __init__(self, cfg: FilterConfig, clock=time.monotonic):
        self.cfg = cfg
        self._clock = clock
        self._last_accept_t = -1e9
        self._last_thumb: np.ndarray | None = None

    @staticmethod
    def _thumb(frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        th = max(1, int(h * THUMB_W / w))
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        return cv2.resize(grey, (THUMB_W, th), interpolation=cv2.INTER_AREA)

    def check(self, frame_bgr: np.ndarray) -> FilterDecision:
        now = self._clock()
        if now - self._last_accept_t < 1.0 / self.cfg.max_fps:
            return FilterDecision(False, "rate")

        thumb = self._thumb(frame_bgr)
        brightness = float(thumb.mean())
        if brightness < self.cfg.min_brightness:
            return FilterDecision(False, "dark", brightness=brightness)
        if brightness > self.cfg.max_brightness:
            return FilterDecision(False, "overexposed", brightness=brightness)

        sharpness = float(cv2.Laplacian(thumb, cv2.CV_64F).var())
        if sharpness < self.cfg.min_sharpness:
            return FilterDecision(False, "blur", sharpness, brightness)

        change = 255.0
        if self._last_thumb is not None and self._last_thumb.shape == thumb.shape:
            change = float(cv2.absdiff(thumb, self._last_thumb).mean())
        forced = (now - self._last_accept_t) >= self.cfg.force_every_s
        if change < self.cfg.min_change and not forced:
            return FilterDecision(False, "static", sharpness, brightness, change)

        self._last_accept_t = now
        self._last_thumb = thumb
        return FilterDecision(True, "forced" if change < self.cfg.min_change else "ok",
                              sharpness, brightness, change)
