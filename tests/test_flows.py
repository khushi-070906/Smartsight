import numpy as np
import pytest

from conftest import needs_v2t
from smartsight.audio_out import SpeechQueue
from smartsight.cane_link import SimCaneLink
from smartsight.config import Config, FilterConfig, SOSConfig, TransportConfig, TTCConfig
from smartsight.frame_filter import FrameRelevanceFilter
from smartsight.navigation import FixedLocation, recommend_direction
from smartsight.ocr import OCRService, TextBox, reading_order
from smartsight.sos import SOSManager, SOSState
from smartsight.transport import BusIdentifier, RouteDirectory, RouteInfo, candidate_numbers
from smartsight.ttc import Urgency
from smartsight.visionsense import (PRIO_NARRATION, PRIO_URGENT, FeedbackPolicy, SceneObject,
                                    SceneSnapshot, VisionSensePipeline)

W, H = 640, 480


class AlwaysUseful(FrameRelevanceFilter):
    def __init__(self):
        super().__init__(FilterConfig())

    def check(self, frame):
        from smartsight.frame_filter import FilterDecision
        return FilterDecision(True, "ok")


class FakeBackend:
    """detect() replays a scripted list of detections per call; prune() keeps
    everything whose label is in `keep`, importance = 1/depth."""

    def __init__(self, script, keep=None):
        self.script, self.i, self.keep = script, 0, keep

    def detect(self, frame):
        d = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return d

    def prune(self, dets, w, h, heading_rad):
        out = [(i, 1.0 / max(d.depth_m, 0.1)) for i, d in enumerate(dets)
               if self.keep is None or d.label in self.keep]
        return sorted(out, key=lambda x: -x[1])

    def make_tracker(self):
        import tracker
        return tracker.ObjectTracker(confirm_frames=3, max_missed_frames=5)


def det(label, cx, depth, i=0, box_h=100):
    from detect_depth import Detection
    return Detection(obj_id=i, label=label, confidence=0.9,
                     bbox_xyxy=(cx - 50, 200 - box_h / 2, cx + 50, 200 + box_h / 2),
                     centroid_px=(cx, 200), depth_m=depth)


@pytest.fixture
def fake_time(monkeypatch):
    """v2t-graph's tracker uses time.time() for velocity; drive it at 4 fps."""
    import tracker
    t = [5000.0]
    monkeypatch.setattr(tracker.time, "time", lambda: t[0])
    return t


def frame():
    return np.zeros((H, W, 3), np.uint8)


# ---------------------------------------------------------------- Fig. 2
@needs_v2t
def test_confirmation_gates_new_objects(fake_time):
    p = VisionSensePipeline(FakeBackend([[det("chair", 320, 3.0)]]), AlwaysUseful(), TTCConfig())
    seen = []
    for _ in range(3):
        seen.append(len(p.process(frame()).objects))
        fake_time[0] += 0.25
    assert seen == [0, 0, 1]          # spoken only once confirmed for 3 frames (S210)


@needs_v2t
def test_approaching_person_escalates_to_urgent(fake_time):
    depths = [6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]      # 2 m/s closing at 4 fps
    p = VisionSensePipeline(FakeBackend([[det("person", 330, d)] for d in depths]), AlwaysUseful(), TTCConfig())
    urg = []
    for _ in depths:
        s = p.process(frame())
        urg.append(s.objects[0].urgency if s.objects else None)
        fake_time[0] += 0.25
    assert urg[:2] == [None, None]
    assert urg[-1] == Urgency.URGENT                       # TTC ~ 2.5 / 2.0 = 1.25 s, still > urgent_dist
    assert p.latest.objects[0].approaching


@needs_v2t
def test_urgent_object_survives_pruning(fake_time):
    # pruner drops cars (e.g. off-heading); a car at 0.6 m must still be reported
    script = [[det("chair", 320, 3.0, 0), det("car", 80, 0.6, 1)]]
    p = VisionSensePipeline(FakeBackend(script, keep={"chair"}), AlwaysUseful(), TTCConfig())
    for _ in range(3):
        s = p.process(frame())
        fake_time[0] += 0.25
    assert [o.label for o in s.objects] == ["car", "chair"]
    assert s.objects[0].urgency == Urgency.URGENT


def _obj(tid, label, d, az, urg=Urgency.NONE, ttc=float("inf")):
    return SceneObject(tid, label, d, az, 1.0, 0.0, ttc, urg, (0, 0, 10, 10))


def test_feedback_policy_branches_and_dedupe(clock):
    pol = FeedbackPolicy("en", narration_interval_s=4, repeat_suppress_s=8, clock=clock)
    snap = SceneSnapshot(clock(), [_obj(1, "pole", 0.6, 0, Urgency.URGENT), _obj(2, "door", 4, 60)])
    fb = pol.decide(snap)
    assert fb.haptic == "urgent" and fb.speech[0][0] == PRIO_URGENT and "Pole" in fb.speech[0][1]
    clock.advance(1)
    assert pol.decide(snap).speech == []                    # same urgent track not repeated
    clock.advance(1)
    calm = SceneSnapshot(clock(), [_obj(2, "door", 4, 60)])
    fb = pol.decide(calm)
    assert fb.haptic is None and fb.speech == [(PRIO_NARRATION, "Door at 2 o'clock, 4 meters away.")]


def test_feedback_hindi():
    pol = FeedbackPolicy("hi")
    text = pol.describe(SceneSnapshot(0, [_obj(1, "bus", 5, -60)]))
    assert text == "बस 10 बजे की दिशा में, 5 मीटर दूर।"


# ---------------------------------------------------------------- Fig. 5
def test_candidate_numbers_rejects_plates_and_words():
    boxes = [TextBox("534", 0.9, (0, 0, 80, 40)), TextBox("DL1PC1234", 0.9, (0, 50, 80, 60)),
             TextBox("ANAND VIHAR", 0.9, (0, 0, 80, 20)), TextBox("99O", 0.8, (0, 0, 30, 15))]
    assert [c for c, _ in candidate_numbers(boxes)] == ["534", "990"]


class FakeOCR:
    """Returns the route board only if the crop it gets is the bus's upper part."""

    def __init__(self, text="534"):
        self.text, self.crops = text, []

    def detect(self, img):
        self.crops.append(img.shape)
        return [TextBox(self.text, 0.9, (0, 0, 60, 40))]


def bus_snap(clock, frame_img, tid=7, d=6.0):
    o = SceneObject(tid, "bus", d, 0, 0.9, 0, float("inf"), Urgency.NONE, (100, 100, 500, 400))
    return SceneSnapshot(clock(), [o], confirmed=[o], frame=frame_img, frame_size=(W, H))


def test_bus_identification_full_flow(clock):
    directory = RouteDirectory({"534": RouteInfo("534", "Anand Vihar ISBT", ["Mehrauli", "AIIMS", "ITO", "Anand Vihar ISBT"])})
    ocr = FakeOCR("534")
    bus = BusIdentifier(TransportConfig(bus_confirm_frames=3), OCRService(ocr), directory, "en", clock=clock)
    img = frame()
    outs = []
    for _ in range(4):
        outs.append(bus.update(bus_snap(clock, img)))
        clock.advance(0.3)
    assert outs[:2] == [None, None]                                  # S505 vision gate
    assert outs[2] == "Bus number 534 is here, going to Anand Vihar ISBT."
    assert outs[3] is None                                           # cooldown, no repeat
    assert ocr.crops[0][:2] == (int(0.45 * 300), 400)                # OCR saw only the board region
    assert "AIIMS" in bus.answer_where()                             # S514


def test_bus_unknown_route_needs_two_reads(clock):
    bus = BusIdentifier(TransportConfig(bus_confirm_frames=1), OCRService(FakeOCR("764")), RouteDirectory(), "en", clock=clock)
    img = frame()
    assert bus.update(bus_snap(clock, img)) is None                  # single read of an unknown number
    assert bus.update(bus_snap(clock, img)) == "Bus number 764 is here."
    assert bus.answer_where() == "I don't have route details for bus 764."


def test_no_bus_no_ocr(clock):
    ocr = FakeOCR()
    bus = BusIdentifier(TransportConfig(), OCRService(ocr), RouteDirectory(), clock=clock)
    o = SceneObject(1, "car", 5, 0, 1, 0, float("inf"), Urgency.NONE, (0, 0, 100, 100))
    assert bus.update(SceneSnapshot(clock(), [o], [o], frame())) is None and ocr.crops == []
    assert bus.answer_which(None) == "I have not seen a bus yet."


def test_gtfs_loader(tmp_path):
    (tmp_path / "routes.txt").write_text("route_id,route_short_name,route_long_name\nR1,534,Mehrauli - Anand Vihar\n")
    (tmp_path / "trips.txt").write_text("route_id,service_id,trip_id,direction_id\nR1,S,T1,0\nR1,S,T2,1\n")
    (tmp_path / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nT1,,,A,1\nT1,,,C,3\nT1,,,B,2\nT2,,,C,1\n")
    (tmp_path / "stops.txt").write_text("stop_id,stop_name\nA,Mehrauli\nB,AIIMS\nC,Anand Vihar ISBT\n")
    r = RouteDirectory.from_gtfs(tmp_path).lookup("534")
    assert r.stops == ["Mehrauli", "AIIMS", "Anand Vihar ISBT"] and r.destination == "Anand Vihar ISBT"
    assert RouteDirectory.from_gtfs(tmp_path).lookup("534").stops == r.stops   # from cache


# ---------------------------------------------------------------- OCR ordering
def test_reading_order():
    boxes = [TextBox("world", 1, (60, 0, 100, 20)), TextBox("line2", 1, (0, 40, 50, 60)), TextBox("hello", 1, (0, 2, 50, 22))]
    assert [[b.text for b in l] for l in reading_order(boxes)] == [["hello", "world"], ["line2"]]


# ---------------------------------------------------------------- Fig. 4
class Spy:
    def __init__(self):
        self.sent = []

    def send(self, text, fix):
        self.sent.append(text)
        return True

    name = "spy"


def make_sos(clock):
    speech = SpeechQueue("console", start_thread=False)
    cane, spy = SimCaneLink(), Spy()
    sos = SOSManager(SOSConfig(confirm_window_s=10, user_name="Khushi"), speech, cane,
                     FixedLocation(28.7, 77.1), [spy], clock=clock, async_dispatch=False)
    return sos, speech, cane, spy


def test_sos_cancel_within_window(clock):
    sos, speech, cane, spy = make_sos(clock)
    sos.on_button()
    assert sos.state == SOSState.PENDING
    clock.advance(4)
    sos.tick()
    sos.on_button()                                                 # second press cancels (S405)
    clock.advance(20)
    sos.tick()
    assert spy.sent == [] and sos.state == SOSState.IDLE
    assert b'"cancelled"' in cane.sent[-1]


def test_fall_dispatches_after_window(clock):
    sos, speech, cane, spy = make_sos(clock)
    sos.trigger("fall")
    for _ in range(11):
        clock.advance(1)
        sos.tick()
    assert len(spy.sent) == 1
    msg = spy.sent[0]
    assert "FALL" in msg and "Khushi" in msg and "maps.google.com/?q=28.700000,77.100000" in msg
    texts = [e[3] for e in speech._heap]
    assert any("fell" in t for t in texts) and any("sent to your guardian" in t for t in texts)
    assert sos.state == SOSState.IDLE


# ---------------------------------------------------------------- navigation
def test_direction_recommendation():
    blocked_ahead = SceneSnapshot(0, [], confirmed=[_obj(1, "pole", 1.0, 0)])
    assert recommend_direction(blocked_ahead, "en") == "Path looks clearest at 10 o'clock."
    assert recommend_direction(SceneSnapshot(0, [], []), "en") == "Path looks clearest at 12 o'clock."


# ---------------------------------------------------------------- speech queue
def test_urgent_preempts_narration():
    q = SpeechQueue("console", start_thread=False)
    q.say("Chair at 1 o'clock.", PRIO_NARRATION)
    q.say("Bus 534 is here.", 1)
    q.say("Stop! Car right ahead.", PRIO_URGENT)
    while q.speak_one(block=False):
        pass
    assert [t for _, t in q.history] == ["Stop! Car right ahead."]


# ---------------------------------------------------------------- app wiring
def test_app_queries_without_vision(clock):
    from smartsight.app import SmartSightApp
    cfg = Config()
    cfg.transport.gtfs_dir = None
    speech = SpeechQueue("console", start_thread=False)
    cane = SimCaneLink()
    app = SmartSightApp(cfg, vision=False, ocr=False, speech=speech, cane=cane,
                        location=FixedLocation(28.7366, 77.1146))
    assert app.handle_query("which bus is this") == "I have not seen a bus yet."
    assert app.handle_query("what is in front of me") == "Nothing important ahead."
    assert app.handle_query("where am i").startswith("You are near latitude 28.7366")
    assert app.handle_query("repeat") .startswith("You are near")
    cane.inject('{"t":"e","ev":"sos_button"}')
    assert app.sos.state == SOSState.PENDING
    app.handle_query("cancel")
    assert app.sos.state == SOSState.IDLE
