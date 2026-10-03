"""Runtime configuration. Load from JSON; every field has a safe default."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class VisionConfig:
    v2t_graph_path: str = "../v2t-graph"           # path to your v2t-graph checkout
    detector_weights: str = "yolov8s-worldv2.pt"
    # Outdoor metric checkpoint: SmartSight is used on streets/bus stops.
    # Switch to ...-Metric-Indoor-Small-hf indoors (what calibration.json was fit on).
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    device: str = "cuda"
    conf_threshold: float = 0.20
    calibration_path: str | None = None
    # Open-vocabulary prompts: v2t-graph's indoor classes + street classes.
    # "bus" is required by the transport flow (S503/S505).
    classes: list[str] = field(default_factory=lambda: [
        "person", "door", "stairs", "wall", "chair", "table", "obstacle",
        "bus", "car", "motorcycle", "bicycle", "auto rickshaw", "truck",
        "pole", "traffic light", "pothole", "dog", "cow",
    ])
    # Affordance priorities for classes v2t-graph's pruning.py doesn't know
    # (they'd otherwise fall back to "unknown" = 0.3). Merged into
    # pruning.AFFORDANCE_PRIORITY at startup; v2t-graph source is not edited.
    extra_affordance: dict[str, float] = field(default_factory=lambda: {
        "bus": 0.95, "car": 0.95, "motorcycle": 0.95, "auto rickshaw": 0.95,
        "truck": 0.95, "bicycle": 0.85, "pothole": 1.0, "pole": 0.7,
        "traffic light": 0.6, "dog": 0.7, "cow": 0.8,
    })
    max_instructions: int = 3
    confirm_frames: int = 3
    max_missed_frames: int = 5


@dataclass
class FilterConfig:                   # S202 frame relevance filter
    max_fps: float = 4.0
    min_sharpness: float = 60.0       # variance of Laplacian on 160px-wide grey
    min_brightness: float = 25.0
    max_brightness: float = 235.0
    min_change: float = 4.0           # mean abs diff vs last accepted frame (0-255)
    force_every_s: float = 2.0        # accept at least this often even if scene is static


@dataclass
class TTCConfig:                      # S211/S212
    urgent_ttc_s: float = 1.5
    soon_ttc_s: float = 4.0
    urgent_dist_m: float = 0.8        # = navigation_planner.IMMEDIATE_RANGE_M
    soon_dist_m: float = 2.0
    min_closing_mps: float = 0.15     # = tracker.is_approaching default


@dataclass
class AudioConfig:
    language: str = "en"              # any BCP-47 / ISO 639 code: en, hi, ta, sw, yo, ar, zh-Hant, ...
    tts_backend: str = "auto"         # auto | pyttsx3 | parler | mms | gtts | console
    tts_fallback_gtts: bool = True    # allow the online gTTS fallback in "auto"
    tts_device: str = "cuda"
    tts_cache_dir: str = "cache/tts"  # synthesized phrases are cached on disk
    translate_queries: bool = True    # unmatched voice queries -> IndicTrans2/NLLB -> English rules
    narration_interval_s: float = 4.0
    repeat_suppress_s: float = 8.0


@dataclass
class ASRConfig:
    backend: str = "keyboard"         # keyboard | auto | vosk | indicconformer | whisper | mms
    vosk_model_paths: dict[str, str] = field(default_factory=lambda: {
        "en": "models/vosk-model-small-en-in-0.4",
        "hi": "models/vosk-model-small-hi-0.22",
    })
    whisper_model: str = "small"      # faster-whisper size: base | small | medium
    whisper_device: str = "cuda"
    record_seconds: float = 4.0
    sample_rate: int = 16000


@dataclass
class OCRConfig:
    # None = English + <audio.language> (EasyOCR can't mix two non-Latin scripts).
    languages: list[str] | None = None
    engine: str = "auto"              # auto | easyocr | tesseract
    gpu: bool = True
    min_confidence: float = 0.35


@dataclass
class TransportConfig:
    gtfs_dir: str | None = "data/delhi_gtfs"   # routes.txt, trips.txt, stop_times.txt, stops.txt
    bus_confirm_frames: int = 3
    announce_cooldown_s: float = 30.0
    active_record_ttl_s: float = 90.0
    max_stops_spoken: int = 4


@dataclass
class SOSConfig:
    confirm_window_s: float = 10.0
    backends: list[str] = field(default_factory=lambda: ["console"])  # console | telegram | twilio
    user_name: str = "SmartSight user"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    twilio_sid: str = ""
    twilio_token: str = ""
    twilio_from: str = ""
    guardian_phone: str = ""


@dataclass
class LinkConfig:
    cane: str = "sim"                 # sim | ble | serial
    ble_name: str = "SmartSight-Cane"
    serial_port: str = "COM5"
    serial_baud: int = 115200
    camera: str = "0"                 # webcam index, video file, or ESP32-CAM stream URL
    location: str = "fixed"           # fixed | nmea | http
    fixed_lat: float = 28.7366        # dev default only
    fixed_lon: float = 77.1146
    nmea_port: str = ""
    http_location_port: int = 8765


@dataclass
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    ttc: TTCConfig = field(default_factory=TTCConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    sos: SOSConfig = field(default_factory=SOSConfig)
    link: LinkConfig = field(default_factory=LinkConfig)


def _merge(obj, data: dict):
    known = {f.name for f in fields(obj)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys in {type(obj).__name__}: {sorted(unknown)}")
    for name, value in data.items():
        cur = getattr(obj, name)
        if is_dataclass(cur) and isinstance(value, dict):
            _merge(cur, value)
        else:
            setattr(obj, name, value)
    return obj


def load_config(path: str | None) -> Config:
    cfg = Config()
    if path:
        _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg
