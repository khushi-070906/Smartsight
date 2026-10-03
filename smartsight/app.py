"""SmartSight companion app (200): wires every flow together.

Threads:
    camera   - FrameGrabber keeps the newest frame
    main     - VisionSense loop (Fig. 2) + bus identification (Fig. 5) + SOS timer (Fig. 4)
    voice    - push-to-talk queries (Fig. 3)
    cane     - link reader; events dispatched to SOS manager / voice trigger
    tts      - SpeechQueue worker (single audio channel to the bone-conduction headset)
"""

from __future__ import annotations

import logging
import threading
import time

from . import i18n
from .asr import make_asr
from .cane_link import SimCaneLink, make_link
from .config import Config
from .frame_filter import FrameRelevanceFilter
from .intent import Intent, classify_multilingual
from .languages import get as get_language
from .navigation import describe_location, make_location, recommend_direction
from .ocr import OCRService
from .protocol import EventPacket, SensorPacket, haptic_cmd
from .sos import SOSManager, make_dispatchers
from .transport import BusIdentifier, RouteDirectory
from .visionsense import PRIO_ALERT, PRIO_RESPONSE, FeedbackPolicy, VisionSensePipeline
from .audio_out import SpeechQueue

log = logging.getLogger(__name__)

LOW_BATTERY_V = 3.45


class _NoOCR:
    def detect(self, img):
        return []


class SmartSightApp:
    def __init__(self, cfg: Config, *, vision: bool = True, ocr: bool = True,
                 speech: SpeechQueue | None = None, cane=None, location=None,
                 vision_backend=None, ocr_engine=None, asr=None, query_translator=None):
        self.cfg = cfg
        self.language = get_language(cfg.audio.language)
        self.lang = self.language.code
        self.speech = speech or SpeechQueue(cfg.audio.tts_backend, self.lang,
                                            gtts_fallback=cfg.audio.tts_fallback_gtts,
                                            cache_dir=cfg.audio.tts_cache_dir, device=cfg.audio.tts_device)
        self.cane = cane or make_link(cfg.link.cane, ble_name=cfg.link.ble_name,
                                      serial_port=cfg.link.serial_port, serial_baud=cfg.link.serial_baud)
        self.location = location or make_location(cfg.link)

        # --- 202: VisionSense (v2t-graph) ---
        self.pipeline = None
        english = None
        if vision:
            if vision_backend is None:
                from .v2t_bridge import V2TBackend
                vision_backend = V2TBackend(cfg.vision)
            if hasattr(vision_backend, "english_phrase"):
                english = vision_backend.english_phrase
            self.pipeline = VisionSensePipeline(vision_backend, FrameRelevanceFilter(cfg.filter), cfg.ttc)
        self.policy = FeedbackPolicy(self.lang, cfg.vision.max_instructions, cfg.audio.narration_interval_s,
                                     cfg.audio.repeat_suppress_s, english_phrase=english)

        # --- 204: OCR ---
        if ocr_engine is None and ocr:
            from .ocr import make_ocr_engine
            ocr_engine = make_ocr_engine(cfg.ocr, self.lang)
        self.ocr = OCRService(ocr_engine or _NoOCR())

        # --- Fig. 5 transport ---
        directory = RouteDirectory.from_gtfs(cfg.transport.gtfs_dir) if cfg.transport.gtfs_dir else RouteDirectory()
        self.bus = BusIdentifier(cfg.transport, self.ocr, directory, self.lang)

        # --- 210: SOS ---
        self.sos = SOSManager(cfg.sos, self.speech, self.cane, self.location,
                              make_dispatchers(cfg.sos), self.lang)

        self.asr = asr or make_asr(cfg.asr, self.lang)
        # Voice commands in languages without keyword rules: translate to English, then classify.
        self.query_translator = query_translator
        if self.query_translator is None and cfg.audio.translate_queries and self.lang != "en":
            from .translate import QueryTranslator
            self.query_translator = QueryTranslator(self.language, cfg.vision.device)
        self.grabber = None
        self.last_response = ""
        self._talk = threading.Event()
        self._stop = threading.Event()
        self._last_haptic = None
        self._last_haptic_t = 0.0
        self._cane_was_connected = None
        self._battery_warned = False
        self.cane.subscribe(self.on_cane_packet)

    # ------------------------------------------------------------------ cane
    def on_cane_packet(self, pkt) -> None:
        if isinstance(pkt, EventPacket):
            ev = pkt.event
            if ev == "sos_button":
                self.sos.on_button()                    # S401b-S403b
            elif ev == "sos_cancel":
                self.sos.cancel()
            elif ev == "fall":
                self.sos.trigger("fall")                # S403a
            elif ev == "talk":
                self._talk.set()
            # "near": the cane already buzzed locally (first-line alert); nothing to add
        elif isinstance(pkt, SensorPacket):
            if pkt.battery_v and pkt.battery_v < LOW_BATTERY_V and not self._battery_warned:
                self._battery_warned = True
                self.speech.say(i18n.t(self.lang, "battery_low"), PRIO_ALERT)

    def _set_haptic(self, pattern: str | None) -> None:
        """The cane expires an app haptic pattern after 3 s (so a dropped link
        can't leave it buzzing), so an active pattern is refreshed every 1 s."""
        pattern = pattern or "off"
        now = time.monotonic()
        if pattern != self._last_haptic or (pattern != "off" and now - self._last_haptic_t > 1.0):
            self.cane.send(haptic_cmd(pattern))
            self._last_haptic, self._last_haptic_t = pattern, now

    # ---------------------------------------------------------------- queries
    def handle_query(self, text: str) -> str:                   # S303 -> S304 -> S307
        intent, _en = classify_multilingual(text, self.query_translator)
        snap = self.pipeline.latest if self.pipeline else None
        if intent == Intent.CANCEL:
            self.sos.cancel()          # speaks its own confirmation; no-op if nothing pending
            return ""
        if intent == Intent.SOS:
            self.sos.trigger("voice")
            return ""
        if intent == Intent.ENVIRONMENT:
            out = self.policy.describe(snap)
        elif intent == Intent.NAVIGATION:
            out = recommend_direction(snap, self.lang)
        elif intent == Intent.BUS_WHICH:
            out = self.bus.answer_which(snap)
        elif intent == Intent.BUS_WHERE:
            out = self.bus.answer_where()
        elif intent == Intent.OCR:
            out = self.read_text_now()
        elif intent == Intent.LOCATION:
            out = describe_location(self.location.get(), self.lang)
        elif intent == Intent.REPEAT:
            out = self.last_response or i18n.t(self.lang, "nothing_ahead")
        else:
            out = i18n.t(self.lang, "unknown_query")
        if intent != Intent.REPEAT:
            self.last_response = out
        return out

    def read_text_now(self) -> str:                              # S304d-S306
        frame = None
        if self.grabber is not None:
            _, frame = self.grabber.latest()                     # still frame, bypasses the relevance filter
        if frame is None:
            return i18n.t(self.lang, "ocr_none")
        text = self.ocr.read_text(frame)
        return i18n.t(self.lang, "ocr_prefix", text=text) if text else i18n.t(self.lang, "ocr_none")

    # ------------------------------------------------------------------ loops
    def voice_loop(self) -> None:
        keyboard = self.cfg.asr.backend == "keyboard"
        while not self._stop.is_set():
            if keyboard:
                try:
                    text = self.asr.listen("you> ")
                except EOFError:
                    self._stop.set()
                    return
                if self._sim_command(text):
                    continue
            else:
                if not self._talk.wait(0.5):
                    continue
                self._talk.clear()
                self.speech.say(i18n.t(self.lang, "listening"), PRIO_RESPONSE)
                time.sleep(0.6)
                text = self.asr.listen()
            if not text.strip():
                continue
            reply = self.handle_query(text)
            if reply:
                self.speech.say(reply, PRIO_RESPONSE)            # S308/S309

    def _sim_command(self, text: str) -> bool:
        """Dev shortcuts when typing: !sos !fall !cancel !talk !near <cm>"""
        if not text.startswith("!"):
            return False
        parts = text[1:].split()
        cmd = parts[0] if parts else ""
        ev = {"sos": "sos_button", "fall": "fall", "cancel": "sos_cancel", "talk": "talk", "near": "near"}.get(cmd)
        if ev is None:
            print("commands: !sos !fall !cancel !talk !near <cm>")
            return True
        line = f'{{"t":"e","ev":"{ev}"' + (f',"cm":{parts[1]}' if ev == "near" and len(parts) > 1 else "") + "}"
        if isinstance(self.cane, SimCaneLink):
            self.cane.inject(line)
        else:
            self.on_cane_packet(EventPacket(ev, {}))
        return True

    def vision_step(self, frame) -> None:
        snap = self.pipeline.process(frame, heading_rad=0.0)
        if snap is None:
            return
        fb = self.policy.decide(snap)                              # S213a/b/c
        self._set_haptic(fb.haptic)
        for prio, text in fb.speech:
            self.speech.say(text, prio)                            # S214
        announcement = self.bus.update(snap)                       # Fig. 5
        if announcement:
            self.speech.say(announcement, PRIO_ALERT)

    def _check_cane_connection(self) -> None:
        c = self.cane.connected
        if self._cane_was_connected is not None and c != self._cane_was_connected:
            self.speech.say(i18n.t(self.lang, "cane_back" if c else "cane_lost"), PRIO_ALERT)
        self._cane_was_connected = c

    def run(self) -> None:
        self.cane.start()
        self.location.start()
        if self.pipeline:
            from .camera import FrameGrabber
            self.grabber = FrameGrabber(self.cfg.link.camera).start()
        threading.Thread(target=self.voice_loop, daemon=True, name="voice").start()
        log.info("SmartSight running (lang=%s, cane=%s, vision=%s)", self.lang, self.cfg.link.cane,
                 bool(self.pipeline))

        last_seq, last_stats = -1, time.monotonic()
        try:
            while not self._stop.is_set():
                self.sos.tick()
                self._check_cane_connection()
                if self.pipeline and self.grabber:
                    seq, frame = self.grabber.latest()
                    if frame is not None and seq != last_seq:
                        last_seq = seq
                        self.vision_step(frame)
                        continue
                if time.monotonic() - last_stats > 10 and self.pipeline:
                    last_stats = time.monotonic()
                    log.info("vision stats: %s", self.pipeline.stats)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            self.cane.stop()
            if self.grabber:
                self.grabber.stop()
            self.speech.stop()
