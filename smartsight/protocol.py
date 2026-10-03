"""Wireless link (119) message schema: newline-delimited compact JSON.

Cane -> app:
    {"t":"s","us":123,"tof":850,"ax":0.01,"ay":-0.02,"az":0.99,"yaw":12.5,"bat":3.92}
        us  = ultrasonic cm (-1 = no echo)     tof = ToF mm (-1 = invalid)
        ax/ay/az = acceleration in g           yaw = integrated gyro yaw (deg)
    {"t":"e","ev":"sos_button"}    manual SOS (S401b/S402b)
    {"t":"e","ev":"sos_cancel"}    button pressed during confirmation window (S405)
    {"t":"e","ev":"fall"}          fall detected on the cane (S403a)
    {"t":"e","ev":"talk"}          optional talk button (push-to-talk)
    {"t":"e","ev":"near","cm":42}  cane already fired its local first-line alert

App -> cane:
    {"c":"hap","p":"urgent"|"soon"|"sos"|"ok"|"off"}
    {"c":"sos","state":"pending"|"sent"|"cancelled"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class SensorPacket:
    ultrasonic_cm: float
    tof_mm: float
    ax: float
    ay: float
    az: float
    yaw_deg: float
    battery_v: float

    @property
    def nearest_m(self) -> float | None:
        vals = []
        if self.ultrasonic_cm > 0:
            vals.append(self.ultrasonic_cm / 100.0)
        if self.tof_mm > 0:
            vals.append(self.tof_mm / 1000.0)
        return min(vals) if vals else None


@dataclass
class EventPacket:
    event: str
    payload: dict


def parse_line(line):
    """Returns SensorPacket | EventPacket | None. Malformed lines are dropped."""
    if isinstance(line, (bytes, bytearray)):
        line = bytes(line).decode("utf-8", errors="ignore")
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    t = d.get("t")
    if t == "s":
        try:
            return SensorPacket(
                ultrasonic_cm=float(d.get("us", -1)), tof_mm=float(d.get("tof", -1)),
                ax=float(d.get("ax", 0)), ay=float(d.get("ay", 0)), az=float(d.get("az", 1)),
                yaw_deg=float(d.get("yaw", 0)), battery_v=float(d.get("bat", 0)),
            )
        except (TypeError, ValueError):
            return None
    if t == "e" and "ev" in d:
        return EventPacket(event=str(d["ev"]),
                           payload={k: v for k, v in d.items() if k not in ("t", "ev")})
    return None


def _enc(obj: dict) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def haptic_cmd(pattern: str) -> bytes:
    return _enc({"c": "hap", "p": pattern})


def sos_cmd(state: str) -> bytes:
    return _enc({"c": "sos", "state": state})
