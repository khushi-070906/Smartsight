"""S401a-S403a - Fall detection from MPU6050 acceleration vectors.

Reference implementation. The SAME state machine runs in the cane firmware
(firmware/smartsight_cane/smartsight_cane.ino, FallDetector) at 100 Hz;
this Python copy exists so the thresholds can be unit-tested and tuned on
logged IMU data (the 10 Hz link stream is too coarse to run it app-side).

Not a single threshold (per the disclosure): a fall must show
    1. IMPACT     |a| > impact_g                         (magnitude)
       optionally preceded by free-fall |a| < freefall_g (boosts confidence,
       lowers the tilt requirement)
    2. SETTLE     ignore settle_s after impact (bounces)
    3. STILLNESS  std(|a|) < still_std_g over still_window_s
    4. TILT       angle between pre-impact gravity vector and post-impact
                  mean vector > tilt_deg                  (direction)
A stumble (impact, no lasting tilt) or a cane tap on the ground (impact,
cane stays upright) is rejected. A cane dropped by itself WILL pass; that
is what the S404 confirmation window is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FallParams:
    impact_g: float = 2.2
    freefall_g: float = 0.45
    freefall_lookback_s: float = 0.6
    settle_s: float = 0.8
    still_window_s: float = 1.5
    still_std_g: float = 0.08
    tilt_deg: float = 50.0
    tilt_deg_with_freefall: float = 35.0
    gravity_alpha: float = 0.02      # low-pass weight for reference gravity when steady
    timeout_s: float = 4.0


def _norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _angle_deg(a, b) -> float:
    na, nb = _norm(a), _norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    c = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)))
    return math.degrees(math.acos(c))


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


class FallDetector:
    """Time is handled as integer milliseconds (same as the firmware) so that
    threshold comparisons are exact and identical on both implementations."""

    IDLE, SETTLING, CHECKING = 0, 1, 2

    def __init__(self, p: FallParams | None = None):
        self.p = p or FallParams()
        self.state = self.IDLE
        self.g_ref = [0.0, 0.0, 1.0]
        self._last_low_g_ms: int | None = None
        self._t_impact = 0
        self._g_at_impact = [0.0, 0.0, 1.0]
        self._had_freefall = False
        self._win: list[tuple[float, float, float]] = []

    def update(self, t: float, ax: float, ay: float, az: float) -> bool:
        """Feed one sample (t in seconds, a in g). Returns True exactly once per fall."""
        p = self.p
        tm = _ms(t)
        mag = _norm((ax, ay, az))
        if mag < p.freefall_g:
            self._last_low_g_ms = tm

        if self.state == self.IDLE:
            if abs(mag - 1.0) < 0.1:  # steady -> track gravity direction
                a = p.gravity_alpha
                self.g_ref = [(1 - a) * g + a * v for g, v in zip(self.g_ref, (ax, ay, az))]
            if mag > p.impact_g:
                self.state = self.SETTLING
                self._t_impact = tm
                self._g_at_impact = list(self.g_ref)
                self._had_freefall = (self._last_low_g_ms is not None
                                      and tm - self._last_low_g_ms <= _ms(p.freefall_lookback_s))
                self._win = []
            return False

        if tm - self._t_impact > _ms(p.timeout_s):
            self.state = self.IDLE
            return False

        if self.state == self.SETTLING:
            if tm - self._t_impact >= _ms(p.settle_s):
                self.state = self.CHECKING
            return False

        # CHECKING
        self._win.append((ax, ay, az))
        if tm - self._t_impact < _ms(p.settle_s) + _ms(p.still_window_s):
            return False

        mags = [_norm(v) for v in self._win]
        mean_mag = sum(mags) / len(mags)
        std = math.sqrt(max(0.0, sum(m * m for m in mags) / len(mags) - mean_mag * mean_mag))
        mean_vec = [sum(v[i] for v in self._win) / len(self._win) for i in range(3)]
        tilt = _angle_deg(self._g_at_impact, mean_vec)
        need = p.tilt_deg_with_freefall if self._had_freefall else p.tilt_deg

        self.state = self.IDLE
        return std < p.still_std_g and tilt > need
