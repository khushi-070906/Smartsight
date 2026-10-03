import math

import cv2
import numpy as np

from smartsight.cane_link import SimCaneLink
from smartsight.config import FilterConfig, TTCConfig, load_config
from smartsight.fall_detect import FallDetector
from smartsight.frame_filter import FrameRelevanceFilter
from smartsight.intent import Intent, classify
from smartsight.protocol import EventPacket, SensorPacket, parse_line
from smartsight.ttc import Urgency, classify as ttc_classify


# ---------------------------------------------------------------- protocol / link
def test_parse_sensor_and_event():
    s = parse_line('{"t":"s","us":120,"tof":800,"ax":0,"ay":0,"az":1,"yaw":5,"bat":3.9}')
    assert isinstance(s, SensorPacket) and s.nearest_m == 0.8
    e = parse_line(b'{"t":"e","ev":"near","cm":42}')
    assert isinstance(e, EventPacket) and e.event == "near" and e.payload == {"cm": 42}
    assert parse_line("garbage") is None and parse_line("{bad json") is None and parse_line("[1]") is None


def test_link_reassembles_chunks():
    link = SimCaneLink()
    got = []
    link.subscribe(got.append)
    msg = b'{"t":"e","ev":"sos_button"}\n{"t":"s","us":-1,"tof":300}\n'
    for i in range(0, len(msg), 7):           # BLE-sized fragments
        link._feed(msg[i:i + 7])
    assert [type(p).__name__ for p in got] == ["EventPacket", "SensorPacket"]
    assert link.last_sensor.nearest_m == 0.3


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"audio": {"language": "hi"}}')
    assert load_config(str(p)).audio.language == "hi"
    p.write_text('{"audio": {"langauge": "hi"}}')
    try:
        load_config(str(p))
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------- S202 filter
def _textured(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(40, 220, (240, 320, 3))).astype(np.uint8)


def test_filter_rejections(clock):
    f = FrameRelevanceFilter(FilterConfig(max_fps=4, force_every_s=2), clock=clock)
    assert f.check(_textured(0)).useful
    clock.advance(0.05)
    assert f.check(_textured(1)).reason == "rate"
    clock.advance(0.5)
    assert f.check(np.zeros((240, 320, 3), np.uint8)).reason == "dark"
    blurred = cv2.GaussianBlur(_textured(2), (31, 31), 15)
    assert f.check(blurred).reason == "blur"
    assert f.check(_textured(0)).reason == "static"         # same as last accepted
    clock.advance(2.0)
    assert f.check(_textured(0)).reason == "forced"         # static, but re-checked periodically
    clock.advance(0.5)
    assert f.check(_textured(3)).useful


# ---------------------------------------------------------------- S211/S212
def test_ttc_classes():
    c = TTCConfig()
    assert ttc_classify(0.6, 0.0, c)[0] == Urgency.URGENT          # too close
    u, ttc = ttc_classify(3.0, -2.5, c)                             # car closing at 2.5 m/s
    assert u == Urgency.URGENT and math.isclose(ttc, 1.2)
    assert ttc_classify(5.0, -1.5, c)[0] == Urgency.SOON            # TTC 3.3 s
    assert ttc_classify(1.5, 0.0, c)[0] == Urgency.SOON             # static but near
    u, ttc = ttc_classify(5.0, -0.05, c)                            # noise, not approaching
    assert u == Urgency.NONE and math.isinf(ttc)


# ---------------------------------------------------------------- fall detection
def _run(det, samples, hz=100):
    fired = False
    for i, (ax, ay, az) in enumerate(samples):
        fired |= det.update(i / hz, ax, ay, az)
    return fired


def _upright(n):
    return [(0.02, 0.01, 1.0)] * n


def test_real_fall_detected():
    s = _upright(300) + [(0, 0, 0.2)] * 30 + [(1.5, 0.5, 2.8)] * 5 + [(1.0, 0.2, 0.3)] * 40 + [(1.0, 0.0, 0.05)] * 250
    assert _run(FallDetector(), s)


def test_cane_tap_not_a_fall():
    s = _upright(300) + [(0.1, 0.1, 3.0)] * 3 + _upright(400)          # impact, stays upright
    assert not _run(FallDetector(), s)


def test_stumble_with_motion_not_a_fall():
    rng = np.random.default_rng(0)
    after = [(1.0 + rng.normal(0, 0.4), rng.normal(0, 0.4), rng.normal(0, 0.4)) for _ in range(300)]
    s = _upright(300) + [(0.5, 0.5, 2.6)] * 4 + after                  # tilted but still moving (walking on)
    assert not _run(FallDetector(), s)


# ---------------------------------------------------------------- S303 intents
def test_intents_english():
    cases = {
        "what is in front of me": Intent.ENVIRONMENT,
        "which direction should I go": Intent.NAVIGATION,
        "which bus is this": Intent.BUS_WHICH,
        "where does this bus go": Intent.BUS_WHERE,
        "read this menu": Intent.OCR,
        "what does that sign say": Intent.OCR,
        "help me": Intent.SOS,
        "cancel": Intent.CANCEL,
        "I'm okay": Intent.CANCEL,
        "where am i": Intent.LOCATION,
        "say that again": Intent.REPEAT,
        "tell me a joke": Intent.UNKNOWN,
    }
    for text, want in cases.items():
        assert classify(text) == want, text


def test_intents_hindi():
    cases = {
        "मेरे सामने क्या है": Intent.ENVIRONMENT,
        "यह कौन सी बस है": Intent.BUS_WHICH,
        "यह बस कहाँ जाएगी": Intent.BUS_WHERE,
        "इसे पढ़ो": Intent.OCR,
        "बचाओ": Intent.SOS,
        "कैंसल करो": Intent.CANCEL,
        "kidhar jaun": Intent.NAVIGATION,
        "ye kaun si bus hai": Intent.BUS_WHICH,
    }
    for text, want in cases.items():
        assert classify(text) == want, text
