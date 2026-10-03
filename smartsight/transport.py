"""Public transport (bus) identification, Fig. 5 (S501-S514).

    S502/S503  frames come from VisionSense (no separate camera loop)
    S504/S505  vision gate: a CONFIRMED track labelled "bus" (tracker-confirmed
               over bus_confirm_frames) - OCR never runs on a bare frame
    S506       OCR only inside the upper part of that bus's bounding box,
               where Delhi DTC/cluster buses carry the route board.
               Roadside signage outside the box cannot trigger an announcement.
    S507       bus-number extraction; registration plates (DL1PC1234) rejected;
               two agreeing reads required unless the number exists in GTFS
    S508-S510  route lookup in static GTFS (Delhi OTD: otd.delhi.gov.in)
    S513/S514  follow-up questions answered from the active bus record
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import i18n
from .config import TransportConfig
from .ocr import OCRService, TextBox
from .visionsense import SceneObject, SceneSnapshot

log = logging.getLogger(__name__)

PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$")          # DL1PC1234, HR26DK8337
ROUTE_RE = re.compile(r"^(?:[A-Z]{1,3}-?)?\d{1,4}[A-Z]{0,4}$")          # 534, 990A, 764STL, E-14, AC-12


def ascii_digits(s: str) -> str:
    """Route boards outside Latin-script regions often use native digits
    (Devanagari ५३४, Bengali ৫৩৪, Arabic-Indic ٥٣٤, ...). Map every Unicode
    decimal digit to 0-9."""
    return "".join(str(unicodedata.digit(c)) if unicodedata.category(c) == "Nd" else c for c in s)


def normalize(token: str) -> str:
    t = ascii_digits(token).upper().strip()
    t = t.replace("O", "0") if re.fullmatch(r"[\dO]+[A-Z]*", t) and any(c.isdigit() for c in t) else t
    return re.sub(r"[\s.]", "", t)


def candidate_numbers(boxes: list[TextBox]) -> list[tuple[str, float]]:
    """Returns [(route number, text height)], tallest first. Height is a proxy
    for 'the big route number' vs small destination text."""
    out = []
    for b in boxes:
        for tok in re.split(r"[\s,/|:]+", ascii_digits(b.text)):
            tok = normalize(tok)
            if len(tok) < 1 or PLATE_RE.match(tok) or not ROUTE_RE.match(tok):
                continue
            if not any(c.isdigit() for c in tok):
                continue
            out.append((tok, b.height))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


@dataclass
class RouteInfo:
    number: str
    destination: str
    stops: list[str] = field(default_factory=list)


class RouteDirectory:
    """Static GTFS index: route_short_name -> first trip's stop list."""

    def __init__(self, routes: dict[str, RouteInfo] | None = None):
        self.routes = routes or {}

    def lookup(self, number: str) -> RouteInfo | None:
        return self.routes.get(normalize(number))

    def __contains__(self, number: str) -> bool:
        return normalize(number) in self.routes

    @classmethod
    def from_gtfs(cls, gtfs_dir: str | Path) -> "RouteDirectory":
        d = Path(gtfs_dir)
        if not (d / "routes.txt").exists():
            log.warning("GTFS not found at %s; bus announcements will be number-only (S509b)", d)
            return cls()

        cache = d / "_smartsight_routes.json"
        if cache.exists() and cache.stat().st_mtime >= (d / "stop_times.txt").stat().st_mtime:
            raw = json.loads(cache.read_text(encoding="utf-8"))
            log.info("loaded %d bus routes from cache", len(raw))
            return cls({k: RouteInfo(**v) for k, v in raw.items()})

        def rows(name):
            with open(d / name, newline="", encoding="utf-8-sig") as f:
                yield from csv.DictReader(f)

        route_name: dict[str, tuple[str, str]] = {}
        for r in rows("routes.txt"):
            short = normalize(r.get("route_short_name") or r.get("route_long_name") or "")
            if short:
                route_name[r["route_id"]] = (short, r.get("route_long_name", ""))

        trip_for_route: dict[str, tuple[str, str]] = {}          # route_id -> (trip_id, headsign)
        for t in rows("trips.txt"):
            if t["route_id"] in route_name and t["route_id"] not in trip_for_route:
                if t.get("direction_id", "0") in ("0", ""):
                    trip_for_route[t["route_id"]] = (t["trip_id"], t.get("trip_headsign", ""))
        for t in rows("trips.txt"):                               # routes with only direction 1
            if t["route_id"] in route_name and t["route_id"] not in trip_for_route:
                trip_for_route[t["route_id"]] = (t["trip_id"], t.get("trip_headsign", ""))

        wanted = {tid: rid for rid, (tid, _) in trip_for_route.items()}
        seq: dict[str, list[tuple[int, str]]] = {}
        for st in rows("stop_times.txt"):                          # large file: streamed
            rid = wanted.get(st["trip_id"])
            if rid:
                seq.setdefault(rid, []).append((int(st["stop_sequence"]), st["stop_id"]))

        stop_name = {s["stop_id"]: s.get("stop_name", s["stop_id"]) for s in rows("stops.txt")}

        routes = {}
        for rid, (short, long_name) in route_name.items():
            stops = [stop_name.get(sid, sid) for _, sid in sorted(seq.get(rid, []))]
            headsign = trip_for_route.get(rid, ("", ""))[1]
            dest = headsign or (stops[-1] if stops else long_name)
            routes.setdefault(short, RouteInfo(short, dest, stops))
        log.info("loaded %d bus routes from GTFS", len(routes))
        try:
            cache.write_text(json.dumps({k: v.__dict__ for k, v in routes.items()}, ensure_ascii=False),
                             encoding="utf-8")
        except OSError:
            pass
        return cls(routes)


@dataclass
class ActiveBus:
    number: str
    route: RouteInfo | None
    seen_at: float


class BusIdentifier:
    def __init__(self, cfg: TransportConfig, ocr: OCRService, directory: RouteDirectory, lang: str = "en",
                 clock=time.monotonic):
        self.cfg, self.ocr, self.dir, self.lang = cfg, ocr, directory, lang
        self._clock = clock
        self._reads: dict[int, Counter] = {}            # track_id -> number votes
        self._bus_frames: dict[int, int] = {}
        self._announced: dict[str, float] = {}
        self.active: ActiveBus | None = None

    def _crop_board(self, snap: SceneSnapshot, bus: SceneObject):
        x1, y1, x2, y2 = (int(v) for v in bus.bbox_xyxy)
        h = y2 - y1
        x1, y1 = max(0, x1), max(0, y1)
        y2b = y1 + max(1, int(0.45 * h))                 # route board: upper ~45% of the bus
        crop = snap.frame[y1:y2b, x1:x2]
        return crop if crop.size else None

    def _read_number(self, snap: SceneSnapshot, bus: SceneObject) -> str | None:
        crop = self._crop_board(snap, bus)
        if crop is None:
            return None
        cands = candidate_numbers(self.ocr.detect(crop))
        if not cands:
            return None
        known = [c for c, _ in cands if c in self.dir]
        return known[0] if known else cands[0][0]

    def update(self, snap: SceneSnapshot | None, force: bool = False) -> str | None:
        """Call on each processed snapshot. Returns an announcement or None."""
        if snap is None or snap.frame is None:
            return None
        now = self._clock()
        buses = [o for o in snap.confirmed if o.label == "bus"]
        live_ids = {b.track_id for b in buses}
        self._bus_frames = {k: v for k, v in self._bus_frames.items() if k in live_ids}
        self._reads = {k: v for k, v in self._reads.items() if k in live_ids}
        if not buses:
            return None

        bus = min(buses, key=lambda b: b.distance_m)          # nearest bus
        n = self._bus_frames[bus.track_id] = self._bus_frames.get(bus.track_id, 0) + 1
        if n < self.cfg.bus_confirm_frames and not force:
            return None                                       # S505 not yet satisfied

        number = self._read_number(snap, bus)                 # S506/S507
        if number is None:
            return None
        votes = self._reads.setdefault(bus.track_id, Counter())
        votes[number] += 1
        best, count = votes.most_common(1)[0]
        if count < 2 and best not in self.dir and not force:
            return None

        if not force and now - self._announced.get(best, -1e9) < self.cfg.announce_cooldown_s:
            return None
        self._announced[best] = now
        route = self.dir.lookup(best)                          # S508
        self.active = ActiveBus(best, route, now)
        return self.announce(self.active)

    def announce(self, ab: ActiveBus) -> str:                  # S509a/S509b/S510
        if ab.route is None:
            return i18n.t(self.lang, "bus_number_only", num=ab.number)
        return i18n.t(self.lang, "bus_full", num=ab.number, dest=ab.route.destination)

    def _active(self) -> ActiveBus | None:
        if self.active and self._clock() - self.active.seen_at <= self.cfg.active_record_ttl_s:
            return self.active
        return None

    def answer_which(self, snap: SceneSnapshot | None) -> str:
        ab = self._active()
        if ab:
            return self.announce(ab)
        text = self.update(snap, force=True)
        return text or i18n.t(self.lang, "bus_no_active")

    def answer_where(self) -> str:                             # S514
        ab = self._active()
        if ab is None:
            return i18n.t(self.lang, "bus_no_active")
        if ab.route is None:
            return i18n.t(self.lang, "bus_no_route", num=ab.number)
        stops = ab.route.stops
        if len(stops) > 2:
            k = self.cfg.max_stops_spoken
            step = max(1, (len(stops) - 1) // k)
            picks = stops[1:-1:step][:k]
        else:
            picks = stops
        out = i18n.t(self.lang, "bus_full", num=ab.number, dest=ab.route.destination)
        if picks:
            out += " " + i18n.t(self.lang, "bus_stops", stops=", ".join(picks))
        return out
