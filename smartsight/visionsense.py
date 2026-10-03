"""VisionSense scene-understanding pipeline, Fig. 2 (S201-S214).

    S201 capture            -> caller passes a frame
    S202/S203 relevance     -> frame_filter.FrameRelevanceFilter
    S204/S205 detect+depth  -> backend.detect          (v2t-graph detect_depth)
    S210/S211 track+velocity-> v2t-graph tracker.ObjectTracker (confirmed tracks only)
    S206-S209 graph+prune   -> backend.prune           (v2t-graph graph_builder + pruning)
    S212 time-to-contact    -> ttc.classify
    S213a/b/c branch        -> FeedbackPolicy.decide
    S214 delivery           -> audio_out.SpeechQueue (+ haptic command to the cane)

Tracking runs before pruning (as in v2t-graph's live_camera_loop_tracked.py)
so only confirmed tracks enter the graph; this kills single-frame flicker.

The backend is injected so the pipeline logic is testable without
torch/model weights (see tests/).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import i18n
from .config import TTCConfig
from .frame_filter import FilterDecision, FrameRelevanceFilter
from .ttc import Urgency, classify


class VisionBackend(Protocol):
    def detect(self, frame_bgr): ...
    def prune(self, detections, frame_w: int, frame_h: int, heading_rad: float) -> list[tuple[int, float]]: ...
    def make_tracker(self): ...


@dataclass
class SceneObject:
    track_id: int
    label: str
    distance_m: float
    azimuth_deg: float
    priority: float
    velocity_mps: float
    ttc_s: float
    urgency: Urgency
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def approaching(self) -> bool:
        return math.isfinite(self.ttc_s)


@dataclass
class SceneSnapshot:
    t: float
    objects: list[SceneObject]                     # pruned, priority order (what the user hears about)
    confirmed: list[SceneObject] = field(default_factory=list)  # every confirmed track (bus flow, queries)
    frame: np.ndarray | None = None
    frame_size: tuple[int, int] = (0, 0)

    @property
    def max_urgency(self) -> Urgency:
        return max((o.urgency for o in self.objects), default=Urgency.NONE)


def azimuth_from_x(cx: float, w: int) -> float:
    """Same convention as v2t-graph encoders.encode_spatial_audio."""
    return (cx / w - 0.5) * 180.0


class VisionSensePipeline:
    def __init__(self, backend: VisionBackend, frame_filter: FrameRelevanceFilter, ttc_cfg: TTCConfig,
                 clock=time.monotonic):
        self.backend = backend
        self.filter = frame_filter
        self.ttc_cfg = ttc_cfg
        self.tracker = backend.make_tracker()
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: SceneSnapshot | None = None
        self.last_filter: FilterDecision | None = None
        self.stats = {"frames": 0, "processed": 0, "infer_ms": 0.0}

    @property
    def latest(self) -> SceneSnapshot | None:
        with self._lock:
            return self._latest

    def process(self, frame_bgr: np.ndarray, heading_rad: float = 0.0) -> SceneSnapshot | None:
        self.stats["frames"] += 1
        decision = self.filter.check(frame_bgr)
        self.last_filter = decision
        if not decision.useful:
            return None

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        raw = self.backend.detect(frame_bgr)
        det_to_track = self.tracker.update(raw, w, h)          # {raw index: confirmed Track}
        conf_idx = list(det_to_track.keys())
        confirmed_dets = [raw[i] for i in conf_idx]

        confirmed_objs: list[SceneObject] = []
        for new_i, old_i in enumerate(conf_idx):
            det, tr = raw[old_i], det_to_track[old_i]
            urg, ttc = classify(det.depth_m, tr.depth_velocity_mps, self.ttc_cfg)
            confirmed_objs.append(SceneObject(
                track_id=tr.track_id, label=det.label, distance_m=float(det.depth_m),
                azimuth_deg=azimuth_from_x(det.centroid_px[0], w), priority=0.0,
                velocity_mps=float(tr.depth_velocity_mps), ttc_s=ttc, urgency=urg,
                bbox_xyxy=tuple(det.bbox_xyxy),
            ))

        kept = self.backend.prune(confirmed_dets, w, h, heading_rad)
        objects = []
        for idx, importance in kept:
            o = confirmed_objs[idx]
            o.priority = importance
            objects.append(o)
        # Safety override: anything URGENT is reported even if pruning dropped it
        # (e.g. an off-heading car closing fast). Pruning ranks; it must not hide danger.
        kept_ids = {o.track_id for o in objects}
        for o in confirmed_objs:
            if o.urgency == Urgency.URGENT and o.track_id not in kept_ids:
                objects.append(o)
        objects.sort(key=lambda o: (o.urgency, o.priority), reverse=True)

        snap = SceneSnapshot(t=self._clock(), objects=objects, confirmed=confirmed_objs,
                             frame=frame_bgr, frame_size=(w, h))
        with self._lock:
            self._latest = snap
        self.stats["processed"] += 1
        self.stats["infer_ms"] = (time.perf_counter() - t0) * 1000
        return snap


@dataclass
class Feedback:
    haptic: str | None = None                        # pattern for the cane (S213a)
    speech: list[tuple[int, str]] = field(default_factory=list)  # (priority, text)


PRIO_URGENT, PRIO_ALERT, PRIO_RESPONSE, PRIO_NARRATION = 0, 1, 2, 3


class FeedbackPolicy:
    """S213a/b/c. Decides what to say and when, without repeating itself."""

    def __init__(self, lang: str, max_items: int = 3, narration_interval_s: float = 4.0,
                 repeat_suppress_s: float = 8.0, english_phrase=None, clock=time.monotonic):
        self.lang = lang
        self.max_items = max_items
        self.narration_interval_s = narration_interval_s
        self.repeat_suppress_s = repeat_suppress_s
        self.english_phrase = english_phrase       # v2t navigation_planner wording, if available
        self._clock = clock
        self._last_narration = -1e9
        self._alerted: dict[tuple[int, int], float] = {}   # (track_id, urgency) -> time

    def phrase(self, o: SceneObject) -> str:
        urgent = o.urgency == Urgency.URGENT
        if self.lang == "en" and self.english_phrase and not urgent:
            text = self.english_phrase(o.label, o.distance_m, o.azimuth_deg)
        else:
            text = i18n.object_phrase(self.lang, o.label, o.distance_m, o.azimuth_deg, urgent)
        if o.approaching and not urgent:
            text = text.rstrip(".।") + ". " + i18n.t(self.lang, "approaching")
        return text

    def decide(self, snap: SceneSnapshot) -> Feedback:
        now = self._clock()
        fb = Feedback()
        self._alerted = {k: v for k, v in self._alerted.items() if now - v < self.repeat_suppress_s}

        urgent = [o for o in snap.objects if o.urgency == Urgency.URGENT]
        soon = [o for o in snap.objects if o.urgency == Urgency.SOON]

        if urgent:
            fb.haptic = "urgent"                                  # S213a
        elif soon:
            fb.haptic = "soon"

        for o in urgent[:1]:                                      # S213b, most critical only
            if (o.track_id, Urgency.URGENT) not in self._alerted:
                fb.speech.append((PRIO_URGENT, self.phrase(o)))
                self._alerted[(o.track_id, Urgency.URGENT)] = now
        for o in soon[:2]:
            if (o.track_id, Urgency.SOON) not in self._alerted and (o.track_id, Urgency.URGENT) not in self._alerted:
                fb.speech.append((PRIO_ALERT, self.phrase(o)))
                self._alerted[(o.track_id, Urgency.SOON)] = now

        # S213c: calm narration only when nothing is urgent/soon - never talk
        # about a door while a pole is 0.6 m away, even if the pole was already announced.
        if not urgent and not soon and now - self._last_narration >= self.narration_interval_s:
            items = [o for o in snap.objects if o.urgency == Urgency.NONE][: self.max_items]
            if items:
                fb.speech.append((PRIO_NARRATION, " ".join(self.phrase(o) for o in items)))
                self._last_narration = now
        return fb

    def describe(self, snap: SceneSnapshot | None) -> str:
        """Answer to 'what is in front of me?' (S304a): always speaks, no suppression."""
        if snap is None or not snap.objects:
            return i18n.t(self.lang, "nothing_ahead")
        return " ".join(self.phrase(o) for o in snap.objects[: self.max_items])
