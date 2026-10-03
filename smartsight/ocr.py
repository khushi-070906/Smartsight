"""On-device OCR engine (204): S304d-S306 (on request) and S506/S507 (bus boards).

Backends: EasyOCR (~80 languages) and Tesseract (~120 languages + script
models), chosen per language by make_ocr_engine; English is always read too; runs on the RTX GPU or CPU). Only invoked on request
or on a confirmed bus crop - never continuously (disclosure advantage 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import OCRConfig

log = logging.getLogger(__name__)


@dataclass
class TextBox:
    text: str
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in the input image

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def preprocess(img_bgr: np.ndarray, min_height: int = 64) -> tuple[np.ndarray, float]:
    """Upscale small crops and boost local contrast (LED bus boards, faded signs)."""
    h = img_bgr.shape[0]
    scale = 1.0
    if h < min_height:
        scale = min_height / max(h, 1)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), scale


def reading_order(boxes: list[TextBox]) -> list[list[TextBox]]:
    """Groups boxes into lines (vertical overlap), lines top->bottom, words left->right."""
    lines: list[list[TextBox]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if lines:
            ref = lines[-1][0]
            if abs(b.cy - ref.cy) < 0.6 * max(ref.height, b.height, 1):
                lines[-1].append(b)
                continue
        lines.append([b])
    return [sorted(line, key=lambda b: b.box[0]) for line in lines]


def ocr_languages(cfg: OCRConfig, ui_lang: str) -> list[str]:
    """EasyOCR language list: English + the user's language (EasyOCR can't mix
    two non-Latin scripts in one reader)."""
    if cfg.languages:
        return list(cfg.languages)
    from .capabilities import easyocr_code
    from .languages import get
    code = easyocr_code(get(ui_lang)) or (ui_lang if ui_lang in ("hi", "ta", "te", "bn") else None)
    return ["en"] if code in (None, "en") else ["en", code]


class EasyOCREngine:
    def __init__(self, cfg: OCRConfig, ui_lang: str = "en"):
        import easyocr
        self.cfg = cfg
        self.languages = ocr_languages(cfg, ui_lang)
        self.reader = easyocr.Reader(self.languages, gpu=cfg.gpu)

    def detect(self, img_bgr: np.ndarray) -> list[TextBox]:
        proc, scale = preprocess(img_bgr)
        out = []
        for pts, text, conf in self.reader.readtext(proc):
            if conf < self.cfg.min_confidence or not text.strip():
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append(TextBox(text.strip(), float(conf),
                               (min(xs) / scale, min(ys) / scale, max(xs) / scale, max(ys) / scale)))
        return out


class TesseractEngine:
    """For scripts EasyOCR lacks (Gujarati, Malayalam, Odia, Gurmukhi, Ethiopic,
    Khmer, Myanmar, Tibetan, Sinhala, ...). Install Tesseract + the traineddata
    for the language (or its script/<Script> model)."""

    def __init__(self, cfg: OCRConfig, model: str):
        import pytesseract
        self.pt, self.cfg = pytesseract, cfg
        self.lang = f"{model}+eng" if model != "eng" else "eng"

    def detect(self, img_bgr: np.ndarray) -> list[TextBox]:
        proc, scale = preprocess(img_bgr)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        d = self.pt.image_to_data(rgb, lang=self.lang, output_type=self.pt.Output.DICT)
        out = []
        for i, text in enumerate(d["text"]):
            conf = float(d["conf"][i]) / 100.0
            if not text.strip() or conf < self.cfg.min_confidence:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append(TextBox(text.strip(), conf, (x / scale, y / scale, (x + w) / scale, (y + h) / scale)))
        return out


def make_ocr_engine(cfg: OCRConfig, ui_lang: str):
    """EasyOCR where it covers the language, else Tesseract, else English-only EasyOCR."""
    from .capabilities import easyocr_code, tesseract_model
    from .languages import get
    lang = get(ui_lang)
    engine = getattr(cfg, "engine", "auto")
    if engine in ("auto", "easyocr") and (easyocr_code(lang) or engine == "easyocr" or lang.code == "en"):
        return EasyOCREngine(cfg, lang.code)
    model = tesseract_model(lang)
    if engine in ("auto", "tesseract") and model:
        return TesseractEngine(cfg, model)
    log.warning("no OCR model for %s (%s script); reading Latin text only", lang.code, lang.script)
    return EasyOCREngine(cfg, "en")


class OCRService:
    def __init__(self, engine):
        self.engine = engine

    def read_text(self, img_bgr: np.ndarray) -> str:
        """S305/S306: full on-request read, returned as spoken-order text."""
        boxes = self.engine.detect(img_bgr)
        return ". ".join(" ".join(b.text for b in line) for line in reading_order(boxes))

    def detect(self, img_bgr: np.ndarray) -> list[TextBox]:
        return self.engine.detect(img_bgr)
