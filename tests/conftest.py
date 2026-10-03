"""Test fixtures. VisionSense tests use v2t-graph's REAL tracker.py and
navigation_planner.py (pure Python) with a fake detector/pruner, so they run
without torch or model weights. Set V2T_GRAPH_PATH or keep the v2t-graph
checkout next to this repo."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V2T = Path(os.environ.get("V2T_GRAPH_PATH", ROOT.parent / "v2t-graph")).resolve()
if (V2T / "tracker.py").exists():
    sys.path.insert(0, str(V2T))


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock():
    return FakeClock()


def v2t_available() -> bool:
    try:
        import tracker  # noqa: F401
        return True
    except Exception:
        return False


needs_v2t = pytest.mark.skipif(not v2t_available(), reason="v2t-graph checkout not found")
