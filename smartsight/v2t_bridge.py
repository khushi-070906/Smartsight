"""Bridge to the v2t-graph repo (github.com/khushi-070906/v2t-graph).

SmartSight uses v2t-graph as-is, as the core of the VisionSense pipeline:
    detect_depth.DetectorDepthEstimator  -> S204 detection + S205 depth
    graph_builder.build_graph            -> S206 scene graph (ego node)
    pruning.prune_graph                  -> S207/S208/S209 heading x affordance x distance pruning
    tracker.ObjectTracker                -> S210 tracking + S211 depth velocity
    navigation_planner._format_instruction -> English phrasing (unchanged wording)

Nothing in v2t-graph is edited. The only runtime adjustment is ADDING
outdoor classes to pruning.AFFORDANCE_PRIORITY (existing keys untouched),
so buses/cars/potholes aren't scored as "unknown" (0.3).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .config import VisionConfig

log = logging.getLogger(__name__)


def add_v2t_to_path(path: str) -> Path:
    p = Path(os.environ.get("V2T_GRAPH_PATH", path)).expanduser().resolve()
    if not (p / "pruning.py").exists():
        raise FileNotFoundError(
            f"v2t-graph not found at {p}. Clone it next to this repo or set "
            f"V2T_GRAPH_PATH / vision.v2t_graph_path."
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


class V2TBackend:
    """Detection + depth + graph pruning, delegated to v2t-graph."""

    def __init__(self, cfg: VisionConfig, load_models: bool = True):
        add_v2t_to_path(cfg.v2t_graph_path)
        import detect_depth
        import graph_builder
        import navigation_planner
        import pruning
        import tracker as v2t_tracker

        self._gb, self._pr, self._np, self._tr = graph_builder, pruning, navigation_planner, v2t_tracker
        self.cfg = cfg

        added = {k: v for k, v in cfg.extra_affordance.items() if k not in pruning.AFFORDANCE_PRIORITY}
        pruning.AFFORDANCE_PRIORITY.update(added)
        if added:
            log.info("added affordance classes: %s", added)

        self.pruning_config = pruning.PruningConfig()
        self.detector = None
        if load_models:
            calib = detect_depth.load_calibration(cfg.calibration_path)
            self.detector = detect_depth.DetectorDepthEstimator(
                detector_weights=cfg.detector_weights,
                depth_model_name=cfg.depth_model,
                device=cfg.device,
                conf_threshold=cfg.conf_threshold,
                open_vocab_classes=cfg.classes,
                calibration=calib,
            )

    def make_tracker(self):
        return self._tr.ObjectTracker(
            confirm_frames=self.cfg.confirm_frames, max_missed_frames=self.cfg.max_missed_frames
        )

    def detect(self, frame_bgr):
        return self.detector.run(frame_bgr)

    def prune(self, detections, frame_w: int, frame_h: int, heading_rad: float) -> list[tuple[int, float]]:
        """Returns [(index into detections, importance)] for kept nodes, highest first."""
        if not detections:
            return []
        labels = [d.label for d in detections]
        graph = self._gb.build_graph(detections, frame_size=(frame_w, frame_h), include_object_edges=False)
        pruned = self._pr.prune_graph(graph, detections_labels=labels, heading_rad=heading_rad,
                                      config=self.pruning_config)
        imp = pruned.kept_importance.tolist() if pruned.kept_importance is not None else []
        out = [(i, float(imp[k]) if k < len(imp) else 0.0)
               for k, i in enumerate(pruned.kept_node_indices) if 0 <= i < len(detections)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def english_phrase(self, label: str, distance_m: float, azimuth_deg: float) -> str:
        return self._np._format_instruction(label, distance_m, azimuth_deg)
