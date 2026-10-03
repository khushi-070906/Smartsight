"""S303 - Query intent classification: English, Hindi (Devanagari + romanised),
Tamil, Telugu, Bengali.

Deterministic keyword rules, deliberately not an LLM: the set of supported
queries is small, latency must be ~0 ms, and it has to work offline. The
order of RULES matters - first match wins (cancel/SOS checked first so an
emergency phrase is never misread as a scene query).
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum


class Intent(str, Enum):
    CANCEL = "cancel"
    SOS = "sos"
    OCR = "ocr"                         # S304d
    BUS_WHERE = "bus_where"             # S514 follow-up: where does it go / stops
    BUS_WHICH = "bus_which"             # S304c
    NAVIGATION = "navigation"           # S304b
    LOCATION = "location"
    ENVIRONMENT = "environment"         # S304a
    REPEAT = "repeat"
    UNKNOWN = "unknown"


RULES: list[tuple[Intent, list[str]]] = [
    (Intent.CANCEL, [r"\bcancel\b", r"\bstop (the )?(alert|sos)\b", r"\bi'?m (ok|okay|fine)\b",
                     r"रद्द", r"कैंसल", r"\bradd\b", r"ठीक हूँ", r"\bthik hu",
                     r"ரத்து", r"கேன்சல்", r"రద్దు", r"క్యాన్సిల్", r"বাতিল", r"ক্যান্সেল"]),
    (Intent.SOS, [r"\bhelp me\b", r"\bemergency\b", r"\bsos\b", r"\bcall (my )?(guardian|family|mom|dad)\b",
                  r"बचाओ", r"मदद", r"इमरजेंसी", r"\bbachao\b", r"\bmadad\b",
                     r"உதவி", r"காப்பாற்று", r"அவசரம்", r"సహాయం", r"కాపాడండి", r"అత్యవసర", r"বাঁচাও", r"সাহায্য", r"জরুরি",
                     r"\bi (have )?fell\b", r"\bi fell\b", r"गिर गया", r"गिर गई", r"விழுந்துவிட்டேன்", r"పడిపోయాను", r"পড়ে গেছি"]),
    (Intent.OCR, [r"\bread\b", r"\bwhat (does|is) (it|this|that) say", r"\bwhat.*written\b", r"\bmenu\b",
                  r"\bsign(board)?\b", r"पढ़", r"पढ", r"लिखा", r"\bpadh", r"\blikha",
                     r"இதைப் ?படி", r"படிங்க", r"படியுங்கள்", r"படிச்சு", r"என்ன எழுதி", r"చదువు", r"చదవండి", r"ఏం రాసి", r"পড়ো", r"পড়ুন", r"পড়ে শোনাও", r"কী লেখা"]),
    (Intent.BUS_WHERE, [r"\bwhere (does|is) (this|that|the) bus go", r"\bwhere.*bus.*go", r"\bbus.*(stops|route|destination)\b",
                        r"\b(which|what) stops\b", r"कहाँ जा", r"कहां जा", r"\bkahan ja", r"\bkaha ja", r"रूट",
                     r"எங்கே போகும்", r"எங்கு செல்லும்", r"ఎక్కడికి వెళ్", r"ఎక్కడికి పోతుంది", r"কোথায় যা"]),
    (Intent.BUS_WHICH, [r"\bwhich bus\b", r"\bwhat bus\b", r"\bbus (number|no)\b", r"\bis (this|that) (the )?bus\b",
                        r"कौन सी बस", r"कौनसी बस", r"बस नंबर", r"\bkaun ?si bus\b",
                     r"எந்தப் ?பேருந்து", r"எந்த பஸ்", r"பேருந்து எண்", r"ఏ బస్సు", r"బస్సు నంబర్", r"কোন বাস", r"বাস নম্বর"]),
    (Intent.NAVIGATION, [r"\bwhich (way|direction)\b", r"\bwhere (should|do) i go\b", r"\bwhich side\b",
                         r"\bwhich way am i facing\b", r"\bdirection\b", r"किधर", r"किस तरफ़", r"किस तरफ",
                         r"\bkidhar\b", r"\bkis taraf\b", r"दिशा",
                     r"எந்தப் ?பக்கம்", r"எந்தத் ?திசை", r"ఏ వైపు", r"ఏ దిశ", r"ఎటు", r"কোন দিকে", r"কোন দিক"]),
    (Intent.LOCATION, [r"\bwhere am i\b", r"\bmy location\b", r"मैं कहाँ", r"मैं कहां", r"\bmain kahan\b",
                     r"நான் எங்கே", r"నేను ఎక్కడ", r"আমি কোথায়"]),
    (Intent.ENVIRONMENT, [r"\bwhat'?s? (is )?(in front|ahead|around)\b", r"\bin front of me\b", r"\bdescribe\b",
                          r"\bwhat do you see\b", r"\bsurroundings\b", r"सामने", r"आसपास", r"\bsamne\b", r"\baas ?paas\b",
                     r"முன்னால்", r"சுற்றி", r"ముందు", r"చుట్టూ", r"সামনে", r"চারপাশে"]),
    (Intent.REPEAT, [r"\brepeat\b", r"\bsay (that|it) again\b", r"\bwhat\?$", r"दोबारा", r"फिर से", r"\bdobara\b",
                     r"மீண்டும்", r"திரும்ப சொல்", r"మళ్ళీ", r"আবার"]),
]


def _nfc(x: str) -> str:
    # Bengali ড় / Hindi ढ़ etc. have composed and decomposed encodings; ASR
    # output and these patterns must compare in the same form.
    return unicodedata.normalize("NFC", x)


_COMPILED = [(intent, [re.compile(_nfc(p), re.IGNORECASE) for p in pats]) for intent, pats in RULES]


def classify(text: str) -> Intent:
    s = _nfc((text or "").strip().lower())
    if not s:
        return Intent.UNKNOWN
    for intent, pats in _COMPILED:
        if any(p.search(s) for p in pats):
            return intent
    return Intent.UNKNOWN


def classify_multilingual(text: str, translator=None) -> tuple[Intent, str | None]:
    """Keyword rules first (instant, works offline for en/hi/ta/te/bn); if
    nothing matches and a translator is available, translate the query to
    English and classify that. This is what makes voice commands work in
    every language without hand-written keyword lists.
    Returns (intent, english_translation_or_None)."""
    intent = classify(text)
    if intent != Intent.UNKNOWN or translator is None:
        return intent, None
    en = translator.to_english(text)
    if not en:
        return intent, None
    return classify(en), en
