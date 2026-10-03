"""Which engine can do which job for a language - checked against what is
actually installed/available, so `--lang <anything>` either works or says
exactly which piece is missing.

    python -m smartsight --check-lang yo
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from functools import lru_cache

from .languages import Language

log = logging.getLogger(__name__)

# ISO 15924 -> Tesseract "script/<Name>" model (used when no per-language model exists)
TESS_SCRIPT = {
    "Arab": "Arabic", "Armn": "Armenian", "Beng": "Bengali", "Cyrl": "Cyrillic", "Deva": "Devanagari",
    "Ethi": "Ethiopic", "Geor": "Georgian", "Grek": "Greek", "Gujr": "Gujarati", "Guru": "Gurmukhi",
    "Hans": "HanS", "Hant": "HanT", "Hang": "Hangul", "Hebr": "Hebrew", "Jpan": "Japanese",
    "Knda": "Kannada", "Khmr": "Khmer", "Laoo": "Lao", "Latn": "Latin", "Mlym": "Malayalam",
    "Mymr": "Myanmar", "Orya": "Oriya", "Sinh": "Sinhala", "Syrc": "Syriac", "Taml": "Tamil",
    "Telu": "Telugu", "Thaa": "Thaana", "Thai": "Thai", "Tibt": "Tibetan", "Cher": "Cherokee",
    "Cans": "Canadian_Aboriginal", "Vaii": "Vai", "Fraktur": "Fraktur",
}
TESS_SPECIAL = {"zho_Hans": "chi_sim", "zho_Hant": "chi_tra", "yue_Hant": "chi_tra", "aze_Cyrl": "aze_cyrl",
                "uzn_Cyrl": "uzb_cyrl", "srp_Latn": "srp_latn", "kmr_Latn": "kmr", "ckb_Arab": "ckb"}

MMS_TTS = "facebook/mms-tts-{}"
MMS_ASR = "facebook/mms-1b-all"
NLLB = "facebook/nllb-200-distilled-600M"


@lru_cache(maxsize=1)
def whisper_codes() -> frozenset[str]:
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        return frozenset(_LANGUAGE_CODES)
    except Exception:
        return frozenset()


def whisper_code(lang: Language) -> str | None:
    codes = whisper_codes()
    for c in (lang.iso1, lang.iso3, lang.iso3_individual, "jw" if lang.iso3 == "jav" else None,
              "yue" if lang.iso3_individual == "yue" else None):
        if c and c in codes:
            return c
    return None


@lru_cache(maxsize=1)
def tesseract_langs() -> frozenset[str]:
    if not shutil.which("tesseract"):
        return frozenset()
    try:
        import pytesseract
        return frozenset(pytesseract.get_languages(config=""))
    except Exception:
        return frozenset()


def tesseract_model(lang: Language) -> str | None:
    have = tesseract_langs()
    cands = [TESS_SPECIAL.get(lang.flores), lang.iso3_individual, lang.iso3]
    if lang.script in TESS_SCRIPT:
        cands.append(f"script/{TESS_SCRIPT[lang.script]}")
    for c in cands:
        if c and c in have:
            return c
    return None


@lru_cache(maxsize=1)
def easyocr_langs() -> frozenset[str]:
    try:
        from easyocr.config import all_lang_list
        return frozenset(all_lang_list)
    except Exception:
        return frozenset()


def easyocr_code(lang: Language) -> str | None:
    have = easyocr_langs()
    special = {"zho_Hans": "ch_sim", "zho_Hant": "ch_tra", "srp_Latn": "rs_latin", "srp_Cyrl": "rs_cyrillic"}
    for c in (special.get(lang.flores), lang.iso1, lang.iso3):
        if c and c in have:
            return c
    return None


@lru_cache(maxsize=None)
def hf_model_exists(repo: str) -> bool | None:
    """True/False if the Hugging Face hub can tell us; None if offline (unknown)."""
    try:
        from huggingface_hub import model_info
        model_info(repo)
        return True
    except ImportError:
        return None
    except Exception as e:
        name = type(e).__name__
        if name in ("RepositoryNotFoundError", "GatedRepoError"):
            return False
        return None        # network error etc.


def mms_tts_repo(lang: Language) -> str | None:
    """First MMS-TTS repo not known to be missing (unknown when offline -> tried at load)."""
    for c in dict.fromkeys((lang.iso3_individual, lang.iso3)):
        repo = MMS_TTS.format(c)
        if hf_model_exists(repo) is not False:
            return repo
    return None


@dataclass
class Report:
    lang: Language
    locale: bool
    translate: str
    asr: list[str]
    tts: list[str]
    ocr: str | None

    def text(self) -> str:
        l = self.lang
        lines = [f"{l.name} ({l.native}) code={l.code} flores={l.flores} script={l.script}"
                 + (" [RTL]" if l.rtl else ""),
                 f"  phrases : {'locale file present' if self.locale else 'MISSING - run tools/translate_locales.py --lang ' + l.code}",
                 f"  translate (for locale + queries): {self.translate}",
                 f"  speech in : {', '.join(self.asr) or 'NONE (keyboard only)'}",
                 f"  speech out: {', '.join(self.tts) or 'NONE'}",
                 f"  OCR       : {self.ocr or 'NONE for this script'}"]
        return "\n".join(lines)


def report(lang: Language, online_checks: bool = True) -> Report:
    from . import i18n
    asr = []
    if lang.indicconformer:
        asr.append("IndicConformer")
    if whisper_code(lang):
        asr.append(f"Whisper({whisper_code(lang)})")
    asr.append(f"MMS-ASR({lang.iso3_individual}, if adapter exists)")
    tts = ["OS voice (if installed)"]
    if lang.parler:
        tts.append(f"Indic Parler-TTS ({lang.parler})")
    if online_checks and mms_tts_repo(lang):
        repo = mms_tts_repo(lang)
        tts.append(repo if hf_model_exists(repo) else f"{repo} (unverified - offline)")
    tts.append("gTTS (online, if Google supports it)")
    ocr = None
    if easyocr_code(lang):
        ocr = f"EasyOCR:{easyocr_code(lang)}"
    elif tesseract_model(lang):
        ocr = f"Tesseract:{tesseract_model(lang)}"
    translate = "IndicTrans2" if lang.indic else f"NLLB-200 ({lang.flores})"
    return Report(lang, lang.code in i18n.available(), translate, asr, tts, ocr)
