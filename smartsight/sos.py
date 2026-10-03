"""Emergency / SOS manager (210), Fig. 4 (S401-S409).

    S401a-S403a  fall event arrives from the cane ({"ev":"fall"})
    S401b-S403b  SOS button event arrives from the cane ({"ev":"sos_button"}), or voice "help"
    S404/S405    confirmation window: spoken countdown + cane buzzer; cancel by
                 pressing the button again or saying "cancel"
    S406a        abort -> spoken confirmation
    S406b-S408   GPS fix -> compose -> dispatch to every configured backend
    S409         spoken confirmation of dispatch (or failure)

Dispatch backends (stdlib urllib only, no SDKs):
    console  - log only (dev)
    telegram - Bot API sendMessage + sendLocation (free; guardian opens a chat with the bot once)
    twilio   - SMS via Twilio REST API
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from enum import Enum

from . import i18n, protocol
from .config import SOSConfig
from .visionsense import PRIO_URGENT

log = logging.getLogger(__name__)


class SOSState(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    DISPATCHING = "dispatching"


def compose_message(cfg: SOSConfig, source: str, fix) -> str:      # S407
    when = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    reason = {"fall": "a FALL was detected", "button": "the SOS button was pressed",
              "voice": "they asked for help by voice"}.get(source, source)
    lines = [f"SmartSight EMERGENCY: {cfg.user_name} may need help - {reason}.", f"Time: {when}"]
    if fix:
        lines += [f"Location: {fix.lat:.6f}, {fix.lon:.6f} ({fix.source})", fix.maps_url()]
    else:
        lines.append("Location: unavailable")
    return "\n".join(lines)


def _post(url: str, data: dict, headers: dict | None = None, timeout: float = 10) -> int:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


class ConsoleDispatcher:
    name = "console"

    def send(self, text: str, fix) -> bool:
        log.warning("SOS DISPATCH:\n%s", text)
        print("\n=== SOS DISPATCH ===\n" + text + "\n====================\n", flush=True)
        return True


class TelegramDispatcher:
    name = "telegram"

    def __init__(self, token: str, chat_id: str):
        self.base, self.chat_id = f"https://api.telegram.org/bot{token}", chat_id

    def send(self, text: str, fix) -> bool:
        ok = _post(f"{self.base}/sendMessage", {"chat_id": self.chat_id, "text": text}) == 200
        if fix:
            _post(f"{self.base}/sendLocation", {"chat_id": self.chat_id, "latitude": fix.lat, "longitude": fix.lon})
        return ok


class TwilioDispatcher:
    name = "twilio"

    def __init__(self, sid: str, token: str, from_: str, to: str):
        self.url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        self.auth = "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode()
        self.from_, self.to = from_, to

    def send(self, text: str, fix) -> bool:
        return _post(self.url, {"From": self.from_, "To": self.to, "Body": text},
                     {"Authorization": self.auth}) in (200, 201)


def make_dispatchers(cfg: SOSConfig) -> list:
    out = []
    for b in cfg.backends:
        if b == "console":
            out.append(ConsoleDispatcher())
        elif b == "telegram" and cfg.telegram_bot_token and cfg.telegram_chat_id:
            out.append(TelegramDispatcher(cfg.telegram_bot_token, cfg.telegram_chat_id))
        elif b == "twilio" and cfg.twilio_sid and cfg.guardian_phone:
            out.append(TwilioDispatcher(cfg.twilio_sid, cfg.twilio_token, cfg.twilio_from, cfg.guardian_phone))
        else:
            log.error("SOS backend %r not configured - skipped", b)
    if not out:
        out.append(ConsoleDispatcher())
    return out


class SOSManager:
    def __init__(self, cfg: SOSConfig, speech, cane, location, dispatchers, lang: str = "en",
                 clock=time.monotonic, async_dispatch: bool = True):
        self.cfg, self.speech, self.cane, self.location = cfg, speech, cane, location
        self.dispatchers, self.lang, self._clock = dispatchers, lang, clock
        self.async_dispatch = async_dispatch
        self.state = SOSState.IDLE
        self.source = ""
        self._t0 = 0.0
        self._reminded = False
        self._lock = threading.Lock()
        self.last_result: bool | None = None

    def trigger(self, source: str) -> None:                        # S403a/S403b -> S404
        with self._lock:
            if self.state != SOSState.IDLE:
                return
            self.state, self.source = SOSState.PENDING, source
            self._t0, self._reminded = self._clock(), False
        lead = i18n.t(self.lang, "fall_detected") + " " if source == "fall" else ""
        self.speech.say(lead + i18n.t(self.lang, "sos_countdown", s=int(self.cfg.confirm_window_s)), PRIO_URGENT)
        self.cane.send(protocol.sos_cmd("pending"))

    def cancel(self) -> bool:                                       # S405 yes -> S406a
        with self._lock:
            if self.state != SOSState.PENDING:
                return False
            self.state = SOSState.IDLE
        self.speech.say(i18n.t(self.lang, "sos_cancelled"), PRIO_URGENT)
        self.cane.send(protocol.sos_cmd("cancelled"))
        return True

    def on_button(self) -> None:
        """Button semantics: press while idle = SOS; press while pending = cancel."""
        if self.state == SOSState.PENDING:
            self.cancel()
        elif self.state == SOSState.IDLE:
            self.trigger("button")

    def tick(self) -> None:                                         # call ~5-10x per second
        with self._lock:
            if self.state != SOSState.PENDING:
                return
            elapsed = self._clock() - self._t0
            remaining = self.cfg.confirm_window_s - elapsed
            remind = not self._reminded and 0 < remaining <= self.cfg.confirm_window_s / 2
            if remind:
                self._reminded = True
            if remaining > 0:
                go = False
            else:
                self.state, go = SOSState.DISPATCHING, True
        if remind:
            self.speech.say(i18n.t(self.lang, "sos_countdown", s=max(1, round(remaining))), PRIO_URGENT)
        if go:
            if self.async_dispatch:
                threading.Thread(target=self._dispatch, daemon=True, name="sos").start()
            else:
                self._dispatch()

    def _dispatch(self) -> None:                                    # S406b-S409
        try:
            fix = self.location.get()
            text = compose_message(self.cfg, self.source, fix)
            ok = False
            for d in self.dispatchers:
                try:
                    ok = d.send(text, fix) or ok
                except Exception:
                    log.exception("SOS backend %s failed", d.name)
            self.last_result = ok
            self.speech.say(i18n.t(self.lang, "sos_sent" if ok else "sos_failed"), PRIO_URGENT)
            self.cane.send(protocol.sos_cmd("sent" if ok else "failed"))
        finally:
            with self._lock:
                self.state = SOSState.IDLE
