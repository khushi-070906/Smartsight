"""Multilingual phrase tables, loaded from smartsight/locales/<code>.json.

Every language in languages.LANGUAGES (English + the 22 scheduled
languages) is selectable. A locale file holds:
    meta     : source ("human-draft" | "machine:indictrans2") and review status
    labels   : detector class -> word in that language
    phrases  : system utterances with {placeholders}

Shipped hand-written drafts: en, hi, ta, te, bn. Generate the rest with
    python tools/translate_locales.py --all
(IndicTrans2 on your GPU). A language with no locale file falls back to
English text, with a warning - it still works, it just speaks English words.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from .languages import LANGUAGES

log = logging.getLogger(__name__)

LOCALE_DIR = Path(__file__).with_name("locales")
SUPPORTED = tuple(LANGUAGES)                    # selectable


@lru_cache(maxsize=None)
def _load(lang: str) -> dict | None:
    p = LOCALE_DIR / f"{lang}.json"
    if not p.exists():
        if lang != "en":
            log.warning("no locale file for %r; speaking English. Run: python tools/translate_locales.py --lang %s",
                        lang, lang)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def reload() -> None:
    _load.cache_clear()


def available() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in LOCALE_DIR.glob("*.json")))


def locale_meta(lang: str) -> dict:
    d = _load(lang)
    return d["meta"] if d else {}


def _table(lang: str, kind: str) -> dict:
    d = _load(lang)
    return d[kind] if d else {}


class _View:
    """Read-only {lang: table} mapping over the locale files
    (keeps i18n.PHRASES[lang] / i18n.LABELS[lang] working)."""

    def __init__(self, kind: str):
        self.kind = kind

    def __getitem__(self, lang):
        t = _table(lang, self.kind)
        if not t:
            raise KeyError(lang)
        return t

    def get(self, lang, default=None):
        return _table(lang, self.kind) or default

    def __contains__(self, lang):
        return lang in available()

    def __iter__(self):
        return iter(available())


PHRASES = _View("phrases")
LABELS = _View("labels")


def t(lang: str, key: str, **kw) -> str:
    template = _table(lang, "phrases").get(key) or _table("en", "phrases")[key]
    return template.format(**kw) if kw else template


def label(lang: str, name: str) -> str:
    return _table(lang, "labels").get(name, name)


def clock_from_azimuth(azimuth_deg: float) -> int:
    """Same mapping as v2t-graph navigation_planner.azimuth_to_clock: the
    camera's +-90 deg field maps onto 9..3 o'clock, 30 deg per hour."""
    hour = round(azimuth_deg / 30.0)
    hour = max(-3, min(3, hour))
    return 12 if hour == 0 else (hour if hour > 0 else 12 + hour)


def format_distance(lang: str, d: float) -> str:
    rounded = round(d * 2) / 2 if d < 3 else round(d)
    return t(lang, "meters", d=f"{rounded:g}")


def object_phrase(lang: str, name: str, distance_m: float, azimuth_deg: float, urgent: bool) -> str:
    clock = t(lang, "clock", h=clock_from_azimuth(azimuth_deg))
    lab = label(lang, name)
    if lang == "en" or _load(lang) is None:
        lab = lab.capitalize()
    key = "object_urgent" if urgent else "object"
    return t(lang, key, label=lab, clock=clock, dist=format_distance(lang, distance_m))
