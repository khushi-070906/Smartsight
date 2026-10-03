"""Translation: IndicTrans2 (AI4Bharat) for the 22 Indian languages,
NLLB-200 (Meta, ~200 languages) for everything else.

Two uses:
    1. tools/translate_locales.py: draft locale files for languages nobody has
       hand-written yet (offline, one-time, on the GPU).
    2. intent.classify_multilingual: a query with no keyword match is
       translated to English and classified with the English rules, so voice
       commands work in every language without per-language keyword lists.

Models (Hugging Face, trust_remote_code):
    ai4bharat/indictrans2-en-indic-dist-200M   English -> Indic
    ai4bharat/indictrans2-indic-en-dist-200M   Indic -> English
Swap to the 1B checkpoints (drop "-dist-200M", add "-1B") for better quality.
Requires: pip install IndicTransToolkit (pre/post-processing).

    facebook/nllb-200-distilled-600M           any <-> any of ~200 languages
    (facebook/nllb-200-distilled-1.3B for better quality)
Both use FLORES-200 codes (languages.Language.flores).
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"

_PH = re.compile(r"\{(\w+)\}")


def protect(template: str) -> tuple[str, list[str]]:
    """Replaces {placeholders} with numeric markers MT systems copy through
    unchanged (digits survive translation far better than Latin words)."""
    names: list[str] = []

    def sub(m):
        names.append(m.group(1))
        return f"<{len(names)}00>"
    return _PH.sub(sub, template), names


def restore(text: str, names: list[str]) -> str | None:
    """Puts placeholders back. Returns None if any marker was lost or
    duplicated - the caller keeps the English string and flags it."""
    out = text
    for i, name in enumerate(names, 1):
        marker = re.compile(rf"<\s*{i}\s*0\s*0\s*>|\b{i}00\b")
        found = marker.findall(out)
        if len(found) != 1:
            return None
        out = marker.sub("{" + name + "}", out)
    if re.search(r"<\s*\d+\s*>", out):
        return None
    return out


class IndicTranslator:
    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        self.ip = IndicProcessor(inference=True)
        log.info("loaded %s on %s", model_name, device)

    def translate(self, texts: list[str], src: str, tgt: str, beams: int = 4) -> list[str]:
        """src/tgt are FLORES codes (languages.Language.flores)."""
        if not texts:
            return []
        batch = self.ip.preprocess_batch(texts, src_lang=src, tgt_lang=tgt)
        enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, use_cache=True, min_length=0, max_length=256,
                                      num_beams=beams, num_return_sequences=1)
        dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(dec, lang=tgt)


NLLB = "facebook/nllb-200-distilled-600M"


class NLLBTranslator:
    def __init__(self, model_name: str = NLLB, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        log.info("loaded %s on %s", model_name, device)

    def supports(self, flores: str) -> bool:
        return self.tok.convert_tokens_to_ids(flores) != self.tok.unk_token_id

    def translate(self, texts: list[str], src: str, tgt: str, beams: int = 4) -> list[str]:
        if not texts:
            return []
        for code in (src, tgt):
            if not self.supports(code):
                raise ValueError(f"NLLB-200 does not support {code}")
        self.tok.src_lang = src
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt),
                                      max_length=256, num_beams=beams)
        return self.tok.batch_decode(out, skip_special_tokens=True)


def make_translator(lang, to_english: bool, device: str = "cuda"):
    """IndicTrans2 for Indian languages, NLLB-200 otherwise."""
    if lang.indic:
        return IndicTranslator(INDIC_EN if to_english else EN_INDIC, device)
    return NLLBTranslator(NLLB, device)


class QueryTranslator:
    """Lazy <user language> -> English translator for voice queries."""

    def __init__(self, lang, device: str = "cuda", factory=make_translator):
        self.lang, self.device, self.factory = lang, device, factory
        self._tr = None
        self.failed = False

    def to_english(self, text: str) -> str | None:
        if self.failed or self.lang.code == "en":
            return None
        try:
            if self._tr is None:
                self._tr = self.factory(self.lang, True, self.device)
            return self._tr.translate([text], self.lang.flores, "eng_Latn", beams=1)[0]
        except Exception:
            log.exception("query translation unavailable; keyword matching only")
            self.failed = True
            return None
