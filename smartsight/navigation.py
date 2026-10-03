"""Navigation and GPS module (208): S304b, location for S406b.

Location providers:
    fixed - config lat/lon (bench testing)
    nmea  - USB/serial GPS receiver (pip install pynmea2 pyserial)
    http  - listens on :port for GET/POST ?lat=..&lon=.. ; point the Android
            app "GPSLogger" (custom URL logging) at http://<laptop-ip>:8765/?lat=%LAT&lon=%LON
            so the phone's GPS feeds the hub during prototyping.

Direction guidance ("which way should I go?") is computed from the current
pruned scene: the forward field is split into clock sectors 9..3 and the
sector closest to 12 o'clock with no object inside clear_m is recommended.
This is obstacle-avoidance direction, not turn-by-turn routing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import i18n
from .visionsense import SceneSnapshot

log = logging.getLogger(__name__)


@dataclass
class Fix:
    lat: float
    lon: float
    t: float
    source: str

    def maps_url(self) -> str:
        return f"https://maps.google.com/?q={self.lat:.6f},{self.lon:.6f}"


class FixedLocation:
    def __init__(self, lat: float, lon: float):
        self.fix = Fix(lat, lon, time.time(), "fixed")

    def start(self): ...

    def get(self, max_age_s: float = 120) -> Fix | None:
        return Fix(self.fix.lat, self.fix.lon, time.time(), "fixed")


class HTTPLocation:
    def __init__(self, port: int):
        self.port = port
        self.fix: Fix | None = None

    def start(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def _handle(self):
                q = parse_qs(urlparse(self.path).query)
                if self.command == "POST":
                    body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)).decode()
                    q.update(parse_qs(body))
                try:
                    outer.fix = Fix(float(q["lat"][0]), float(q["lon"][0]), time.time(), "http")
                    self.send_response(200)
                except (KeyError, ValueError, IndexError):
                    self.send_response(400)
                self.end_headers()

            do_GET = do_POST = _handle

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True, name="loc-http").start()
        log.info("location receiver on :%d", self.port)

    def get(self, max_age_s: float = 120) -> Fix | None:
        f = self.fix
        return f if f and time.time() - f.t <= max_age_s else None


class NMEALocation:
    def __init__(self, port: str, baud: int = 9600):
        self.port, self.baud = port, baud
        self.fix: Fix | None = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="loc-nmea").start()

    def _run(self):
        import pynmea2
        import serial
        with serial.Serial(self.port, self.baud, timeout=1) as ser:
            while True:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith(("$GPGGA", "$GNGGA", "$GPRMC", "$GNRMC")):
                    continue
                try:
                    msg = pynmea2.parse(line)
                except pynmea2.ParseError:
                    continue
                if getattr(msg, "latitude", 0) and getattr(msg, "longitude", 0):
                    self.fix = Fix(msg.latitude, msg.longitude, time.time(), "nmea")

    def get(self, max_age_s: float = 120) -> Fix | None:
        f = self.fix
        return f if f and time.time() - f.t <= max_age_s else None


def make_location(link_cfg):
    if link_cfg.location == "http":
        return HTTPLocation(link_cfg.http_location_port)
    if link_cfg.location == "nmea":
        return NMEALocation(link_cfg.nmea_port)
    return FixedLocation(link_cfg.fixed_lat, link_cfg.fixed_lon)


ARC = [9, 10, 11, 12, 1, 2, 3]          # forward half of the clock, left to right
SECTORS = [12, 11, 1, 10, 2, 9, 3]      # preference order: straight first, then widen


def recommend_direction(snap: SceneSnapshot | None, lang: str, clear_m: float = 3.0) -> str:
    objs = snap.confirmed if snap else []
    blocked: set[int] = set()
    for o in objs:
        if o.distance_m >= clear_m:
            continue
        i = ARC.index(i18n.clock_from_azimuth(o.azimuth_deg))
        blocked.add(ARC[i])
        if o.distance_m < 1.2:      # a very close object also covers the neighbouring sectors
            blocked.update(ARC[max(0, i - 1): i + 2])
    for s in SECTORS:
        if s not in blocked:
            return i18n.t(lang, "go_dir", clock=i18n.t(lang, "clock", h=s))
    return i18n.t(lang, "go_blocked")


def describe_location(fix: Fix | None, lang: str) -> str:
    if fix is None:
        return i18n.t(lang, "location_unknown")
    return i18n.t(lang, "location_here", lat=f"{fix.lat:.4f}", lon=f"{fix.lon:.4f}")
