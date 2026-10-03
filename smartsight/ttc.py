"""S211/S212 - Depth velocity -> time-to-contact -> urgency class.

Uses tracker.Track.depth_velocity_mps from v2t-graph (smoothed d(depth)/dt,
negative = approaching). TTC = depth / closing_speed.

Urgency (the three branches of Fig. 2):
    URGENT : inside urgent_dist_m, or TTC < urgent_ttc_s   -> S213a haptic+buzzer AND S213b voice
    SOON   : inside soon_dist_m,   or TTC < soon_ttc_s     -> S213b priority voice alert
    NONE   : everything else                               -> S213c contextual narration

Distance-only triggers exist because a static obstacle (pole, wall) has
~zero depth velocity while the *user* is the one moving; with only a
head-mounted camera and no visual odometry the tracker cannot separate
ego-motion from object motion, so proximity must stand on its own.
"""

from __future__ import annotations

import math
from enum import IntEnum

from .config import TTCConfig


class Urgency(IntEnum):
    NONE = 0
    SOON = 1
    URGENT = 2


def time_to_contact(depth_m: float, depth_velocity_mps: float, min_closing_mps: float) -> float:
    closing = -depth_velocity_mps
    if closing < min_closing_mps or depth_m <= 0:
        return math.inf
    return depth_m / closing


def classify(depth_m: float, depth_velocity_mps: float, cfg: TTCConfig) -> tuple[Urgency, float]:
    ttc = time_to_contact(depth_m, depth_velocity_mps, cfg.min_closing_mps)
    if depth_m <= cfg.urgent_dist_m or ttc < cfg.urgent_ttc_s:
        return Urgency.URGENT, ttc
    if depth_m <= cfg.soon_dist_m or ttc < cfg.soon_ttc_s:
        return Urgency.SOON, ttc
    return Urgency.NONE, ttc
