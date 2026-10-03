"""Language resolution for ANY language, plus curated data for India's 22.

What "all languages" can honestly mean: SmartSight works in a language when
its pipeline can translate the phrases AND speak them. Coverage is bounded by
the open models, not by this code:

    translation  IndicTrans2 (22 Indian)  +  NLLB-200 (~200 languages)
    speech out   OS voices, Indic Parler-TTS, Meta MMS-TTS (~1,100), gTTS
    speech in    IndicConformer (22 Indian), Whisper (~99), Meta MMS-ASR (~1,100)
    OCR          EasyOCR (~80), Tesseract (~120 + script models)

=> full voice+text support: roughly the ~200 NLLB languages
   (speech-only fallbacks extend further; most of the world's ~7,000
   languages have no written form or no model and cannot be supported).

Any BCP-47 / ISO 639 code is accepted ("hi", "yor", "sw", "zh-Hant",
"ckb"). It is normalised with `langcodes`; the FLORES-200 code used by
NLLB / IndicTrans2 is derived from ISO 639-3 + likely script, with the
macrolanguage fixes NLLB needs (ar -> arb_Arab, sw -> swh_Latn, ...).
Which engines actually exist for the language is decided at runtime by
capabilities.py, against what is installed - never assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import langcodes

# ISO 639-3 macrolanguage -> the individual language NLLB / IndicTrans2 / MMS use
MACRO_TO_INDIVIDUAL = {
    "ara": "arb", "swa": "swh", "fas": "pes", "msa": "zsm", "lav": "lvs", "mon": "khk",
    "uzb": "uzn", "orm": "gaz", "mlg": "plt", "sqi": "als", "nep": "npi", "ori": "ory",
    "kok": "gom", "aze": "azj", "kur": "kmr", "yid": "ydd", "nor": "nob", "que": "quy",
    "pus": "pbt", "aym": "ayr", "kau": "knc", "est": "est", "fil": "tgl", "grn": "grn",
    "zza": "diq", "kln": "kln", "luy": "luy",
}

# The 22 Eighth-Schedule languages: preferred script + which Indic engines cover them.
# parler: AI4Bharat Indic Parler-TTS ("unofficial" = listed by the model card as unofficial).
INDIC = {
    #  code   flores      indicconformer parler
    "as":  ("asm_Beng", True, "official"),
    "bn":  ("ben_Beng", True, "official"),
    "brx": ("brx_Deva", True, "official"),
    "doi": ("doi_Deva", True, "official"),
    "gu":  ("guj_Gujr", True, "official"),
    "hi":  ("hin_Deva", True, "official"),
    "kn":  ("kan_Knda", True, "official"),
    "ks":  ("kas_Arab", True, "unofficial"),
    "kok": ("gom_Deva", True, "official"),
    "mai": ("mai_Deva", True, "official"),
    "ml":  ("mal_Mlym", True, "official"),
    "mni": ("mni_Beng", True, "official"),
    "mr":  ("mar_Deva", True, "official"),
    "ne":  ("npi_Deva", True, "official"),
    "or":  ("ory_Orya", True, "official"),
    "pa":  ("pan_Guru", True, "unofficial"),
    "sa":  ("san_Deva", True, "official"),
    "sat": ("sat_Olck", True, "official"),
    "sd":  ("snd_Arab", True, "official"),
    "ta":  ("tam_Taml", True, "official"),
    "te":  ("tel_Telu", True, "official"),
    "ur":  ("urd_Arab", True, "official"),
}
SCHEDULED = tuple(INDIC)
RTL_SCRIPTS = {"Arab", "Hebr", "Thaa", "Syrc", "Nkoo", "Adlm"}


@dataclass(frozen=True)
class Language:
    code: str            # normalised BCP-47 tag SmartSight uses (locale file name)
    name: str            # English name
    native: str          # autonym
    iso3: str            # ISO 639-3 (as given; may be a macrolanguage)
    iso3_individual: str # individual language code used by NLLB/MMS
    script: str          # ISO 15924
    flores: str          # FLORES-200 code for NLLB / IndicTrans2
    indic: bool
    indicconformer: bool = False
    parler: str | None = None

    @property
    def rtl(self) -> bool:
        return self.script in RTL_SCRIPTS

    @property
    def iso1(self) -> str | None:
        base = self.code.split("-")[0]
        return base if len(base) == 2 else None


@lru_cache(maxsize=None)
def get(code: str) -> Language:
    try:
        tag = langcodes.Language.get(code.replace("_", "-"))
    except Exception as e:
        raise ValueError(f"not a language code: {code!r}") from e
    if not tag.language or not tag.is_valid():
        raise ValueError(f"not a language code: {code!r}")
    # regions are dropped (de-DE -> de); a non-default script is kept (zh-Hant, sr-Latn)
    norm = langcodes.Language.make(language=tag.language, script=tag.script).simplify_script().to_tag()
    base = tag.language
    try:
        iso3 = tag.to_alpha3()
    except LookupError as e:
        raise ValueError(f"unknown language: {code!r}") from e
    script = tag.script or tag.maximize().script or "Latn"
    individual = MACRO_TO_INDIVIDUAL.get(iso3, iso3)
    if base in INDIC:
        flores, ic, parler = INDIC[base]
        script = flores.split("_")[1]
        return Language(norm, tag.display_name(), tag.autonym(), iso3, flores.split("_")[0],
                        script, flores, True, ic, parler)
    flores = f"{individual}_{script}"
    if individual == "zho":
        flores = "zho_Hant" if script == "Hant" else "zho_Hans"
    return Language(norm, tag.display_name(), tag.autonym(), iso3, individual, script, flores,
                    False, False, "official" if base == "en" else None)


# Back-compat for code that iterates the Indian set
LANGUAGES = {c: get(c) for c in ("en",) + SCHEDULED}
