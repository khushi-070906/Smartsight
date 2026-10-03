"""Compiles firmware/smartsight_cane/fall_detector.h on the host and checks it
agrees with smartsight/fall_detect.py, on the hand-built vectors AND on random
noise. Skipped if no C++ compiler is available."""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from smartsight.fall_detect import FallDetector

HOST = Path(__file__).parent / "firmware_host"
pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    exe = tmp_path_factory.mktemp("fw") / "fall_cli"
    subprocess.run(["g++", "-std=c++17", "-O1", "-o", str(exe), str(HOST / "fall_cli.cpp")], check=True)
    return exe


def run_cpp(cli, samples, hz=100):
    txt = "\n".join(f"{i / hz:.4f} {ax:.5f} {ay:.5f} {az:.5f}" for i, (ax, ay, az) in enumerate(samples))
    out = subprocess.run([str(cli)], input=txt, capture_output=True, text=True, check=True).stdout
    return [float(l.split()[1]) for l in out.splitlines() if l.startswith("FALL")]


def run_py(samples, hz=100):
    d, hits = FallDetector(), []
    for i, (ax, ay, az) in enumerate(samples):
        if d.update(i / hz, ax, ay, az):
            hits.append(round(i / hz, 2))
    return hits


UP = lambda n: [(0.02, 0.01, 1.0)] * n
VECTORS = {
    "fall": UP(300) + [(0, 0, 0.2)] * 30 + [(1.5, 0.5, 2.8)] * 5 + [(1.0, 0.2, 0.3)] * 40 + [(1.0, 0.0, 0.05)] * 250,
    "tap": UP(300) + [(0.1, 0.1, 3.0)] * 3 + UP(400),
    "idle": UP(1000),
}


@pytest.mark.parametrize("name", VECTORS)
def test_vectors_match(cli, name):
    s = VECTORS[name]
    assert run_cpp(cli, s) == run_py(s)
    assert bool(run_cpp(cli, s)) == (name == "fall")


def test_random_streams_match(cli):
    rng = random.Random(7)
    for _ in range(25):
        s = []
        for _ in range(rng.randint(3, 8)):
            kind = rng.choice(["up", "walk", "impact", "lie"])
            n = rng.randint(50, 300)
            if kind == "up":
                s += UP(n)
            elif kind == "walk":
                s += [(rng.gauss(0, .3), rng.gauss(0, .3), 1 + rng.gauss(0, .3)) for _ in range(n)]
            elif kind == "impact":
                s += [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(2.3, 3.5)) for _ in range(rng.randint(1, 5))]
            else:
                s += [(1.0 + rng.gauss(0, .01), rng.gauss(0, .01), 0.05 + rng.gauss(0, .01)) for _ in range(n)]
        assert run_cpp(cli, s) == run_py(s)


def test_sketch_passes_host_syntax_check():
    r = subprocess.run([str(HOST / "check_sketch.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
