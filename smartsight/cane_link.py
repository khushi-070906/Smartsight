"""Wireless link (119) between the cane ESP32 (110) and the companion app.

Three transports with one interface:
    SimCaneLink    - no hardware; inject packets from code/tests/keyboard
    SerialCaneLink - ESP32 over USB serial (bench development)
    BLECaneLink    - ESP32 over BLE, Nordic UART Service (field use)

All transports deliver parsed packets to the same callback, from a
background thread, and accept the same send() bytes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from .protocol import EventPacket, SensorPacket, parse_line

log = logging.getLogger(__name__)

# Nordic UART Service (same UUIDs as firmware/smartsight_cane.ino)
NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"   # app writes here
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # cane notifies here

PacketHandler = Callable[[SensorPacket | EventPacket], None]


class CaneLink:
    def __init__(self):
        self._handlers: list[PacketHandler] = []
        self._buf = b""
        self.last_sensor: SensorPacket | None = None
        self.connected = False

    def subscribe(self, handler: PacketHandler) -> None:
        self._handlers.append(handler)

    def _feed(self, data: bytes) -> None:
        """Reassembles newline-delimited JSON across BLE/serial chunk boundaries."""
        self._buf += data
        if len(self._buf) > 4096:            # garbage guard
            self._buf = self._buf[-1024:]
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            pkt = parse_line(line)
            if pkt is None:
                continue
            if isinstance(pkt, SensorPacket):
                self.last_sensor = pkt
            for h in self._handlers:
                try:
                    h(pkt)
                except Exception:
                    log.exception("cane packet handler failed")

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def send(self, data: bytes) -> None: ...


class SimCaneLink(CaneLink):
    """In-process cane. inject() behaves exactly like bytes arriving on the wire."""

    def __init__(self):
        super().__init__()
        self.sent: list[bytes] = []
        self.connected = True

    def inject(self, line: str) -> None:
        self._feed(line.encode() + b"\n")

    def send(self, data: bytes) -> None:
        self.sent.append(data)
        log.debug("sim cane <- %s", data.strip())


class SerialCaneLink(CaneLink):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port, self.baud = port, baud
        self._ser = None
        self._stop = threading.Event()

    def start(self) -> None:
        import serial  # pyserial
        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        self.connected = True
        threading.Thread(target=self._reader, daemon=True, name="cane-serial").start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._ser.read(256)
            except Exception:
                log.exception("serial read failed")
                self.connected = False
                return
            if data:
                self._feed(data)

    def send(self, data: bytes) -> None:
        if self._ser:
            self._ser.write(data)

    def stop(self) -> None:
        self._stop.set()
        if self._ser:
            self._ser.close()


class BLECaneLink(CaneLink):
    """Runs a private asyncio loop in a thread; reconnects automatically."""

    def __init__(self, device_name: str = "SmartSight-Cane"):
        super().__init__()
        self.device_name = device_name
        self._loop = asyncio.new_event_loop()
        self._client = None
        self._stopping = False

    def start(self) -> None:
        threading.Thread(target=self._loop.run_forever, daemon=True, name="cane-ble").start()
        asyncio.run_coroutine_threadsafe(self._run(), self._loop)

    async def _run(self) -> None:
        from bleak import BleakClient, BleakScanner
        while not self._stopping:
            try:
                dev = await BleakScanner.find_device_by_name(self.device_name, timeout=10)
                if dev is None:
                    log.warning("cane %r not found, rescanning", self.device_name)
                    continue
                async with BleakClient(dev) as client:
                    self._client = client
                    await client.start_notify(NUS_TX, lambda _h, data: self._feed(bytes(data)))
                    self.connected = True
                    log.info("cane connected: %s", dev.address)
                    while client.is_connected and not self._stopping:
                        await asyncio.sleep(0.5)
            except Exception:
                log.exception("BLE error; retrying")
            finally:
                self.connected = False
                self._client = None
            await asyncio.sleep(2)

    def send(self, data: bytes) -> None:
        client = self._client
        if client is None:
            return
        # BLE default MTU payload is 20 bytes; chunk to be safe.
        async def _write():
            for i in range(0, len(data), 20):
                await client.write_gatt_char(NUS_RX, data[i:i + 20], response=False)
        asyncio.run_coroutine_threadsafe(_write(), self._loop)

    def stop(self) -> None:
        self._stopping = True


def make_link(kind: str, *, ble_name: str = "", serial_port: str = "", serial_baud: int = 115200) -> CaneLink:
    if kind == "sim":
        return SimCaneLink()
    if kind == "serial":
        return SerialCaneLink(serial_port, serial_baud)
    if kind == "ble":
        return BLECaneLink(ble_name)
    raise ValueError(f"unknown cane link {kind!r}")
