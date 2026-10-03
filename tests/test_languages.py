import string
import unicodedata

import numpy as np
import pytest

from smartsight import i18n
from smartsight.asr import candidates
from smartsight.audio_out import _SENT_SPLIT, _Console, make_tts
from smartsight.config import ASRConfig, OCRConfig
from smartsight.intent import Intent, classify, classify_multilingual
from smartsight.languages import SCHEDULED, get
from smartsight.ocr import ocr_languages
from smartsight.translate import protect, restore
from smartsight.transport import ascii_digits, candidate_numbers
from smartsight.ocr import TextBox

SHIPPED = ["en", "hi", "ta", "te", "bn"]


def _fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


# ---------------------------------------------------------------- language resolution
@pytest.mark.parametrize("code,norm,flores", [
    ("hi", "hi", "hin_Deva"), ("hi-IN", "hi", "hin_Deva"), ("kok", "kok", "gom_Deva"),
    ("ne", "ne", "npi_Deva"), ("or", "or", "ory_Orya"), ("sat", "sat", "sat_Olck"),
    ("mni", "mni", "mni_Beng"), ("ks", "ks", "kas_Arab"),
    ("ar", "ar", "arb_Arab"), ("sw", "sw", "swh_Latn"), ("fa", "fa", "pes_Arab"),
    ("zh", "zh", "zho_Hans"), ("zh-Hant", "zh-Hant", "zho_Hant"), ("yo", "yo", "yor_Latn"),
    ("fil", "fil", "tgl_Latn"), ("de-DE", "de", "deu_Latn"), ("pt-BR", "pt", "por_Latn"),
    ("sr-Latn", "sr-Latn", "srp_Latn"), ("am", "am", "amh_Ethi"),
])
def test_language_resolution(code, norm, flores):
    l = get(code)
    assert (l.code, l.flores) == (norm, flores)


def test_all_22_scheduled_languages_resolve_as_indic():
    assert len(SCHEDULED) == 22
    for c in SCHEDULED:
        l = get(c)
        assert l.indic and l.indicconformer and l.flores.count("_") == 1


def test_rtl_and_invalid():
    assert get("ur").rtl and get("ar").rtl and not get("hi").rtl
    for bad in ["xx", "zz", "", "12", "x"]:
        with pytest.raises(ValueError):
            get(bad)


# ---------------------------------------------------------------- locale files
@pytest.mark.parametrize("lang", sorted(set(i18n.available()) - {"en"}))
def test_every_locale_file_is_complete(lang):
    en_p, en_l = i18n.PHRASES["en"], i18n.LABELS["en"]
    p, l = i18n.PHRASES[lang], i18n.LABELS[lang]
    assert set(p) == set(en_p), set(en_p) ^ set(p)
    assert set(l) == set(en_l), set(en_l) ^ set(l)
    for k in en_p:
        assert _fields(p[k]) == _fields(en_p[k]), (lang, k)


def test_shipped_locales_present():
    assert set(SHIPPED) <= set(i18n.available())


def test_labels_cover_every_detector_class():
    from smartsight.config import VisionConfig
    for lang in i18n.available():
        missing = [c for c in VisionConfig().classes if c not in i18n.LABELS[lang]]
        assert not missing, (lang, missing)


def test_language_without_locale_falls_back_to_english():
    assert "yo" not in i18n.available()
    assert i18n.t("yo", "nothing_ahead") == i18n.t("en", "nothing_ahead")
    assert i18n.object_phrase("yo", "bus", 5, 0, False) == "Bus at 12 o'clock, 5 meters away."


@pytest.mark.parametrize("lang", SHIPPED)
def test_every_phrase_renders(lang):
    kw = dict(label="x", clock="y", dist="z", text="t", num="534", dest="d", stops="a, b",
              s=10, lat="28.7", lon="77.1", h=12, d="2")
    for tpl in i18n.PHRASES[lang].values():
        tpl.format(**kw)


# ---------------------------------------------------------------- MT placeholder safety
def test_protect_restore_roundtrip():
    src, names = protect("{label} at {clock}, {dist} away.")
    assert "{" not in src and names == ["label", "clock", "dist"]
    mt = src.replace("at", "में").replace("away", "दूर")          # translator moved words, kept markers
    assert restore(mt, names) == "{label} में {clock}, {dist} दूर."


def test_restore_rejects_lost_or_duplicated_markers():
    src, names = protect("Bus number {num} is here, going to {dest}.")
    assert restore(src.replace("<200>", ""), names) is None          # dropped
    assert restore(src + " <100>", names) is None                     # duplicated
    assert restore(src.replace("<100>", "< 1 00 >"), names) == "Bus number {num} is here, going to {dest}."


def test_translate_locale_marks_bad_phrases_for_review():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "tl", pathlib.Path(__file__).resolve().parents[1] / "tools" / "translate_locales.py")
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)

    class FakeMT:
        def translate(self, texts, src, tgt):
            return [("" if "<200>" in t and "sos" in t.lower() else "[yo] " + t) for t in texts]

    import json
    en = json.loads((i18n.LOCALE_DIR / "en.json").read_text(encoding="utf-8"))
    out = tl.translate_locale(get("yo"), FakeMT(), en)
    assert out["meta"]["source"] == "machine:nllb-200" and out["meta"]["reviewed_by_native_speaker"] is False
    assert out["phrases"]["object"] == "[yo] {label} at {clock}, {dist} away."
    for k, v in out["phrases"].items():
        assert _fields(v) == _fields(en["phrases"][k])
    assert out["labels"]["bus"] == "[yo] bus"


# ---------------------------------------------------------------- queries in any language
class FakeTranslator:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def to_english(self, text):
        self.calls += 1
        return self.table.get(text)


def test_keyword_hit_never_calls_translator():
    tr = FakeTranslator({})
    assert classify_multilingual("यह कौन सी बस है", tr) == (Intent.BUS_WHICH, None)
    assert tr.calls == 0


def test_unmatched_query_is_translated_then_classified():
    tr = FakeTranslator({"Ọkọ̀ akérò wo nìyí?": "Which bus is this?",
                         "ఏమిటి నా దగ్గర ఉంది": "What is near me"})
    assert classify_multilingual("Ọkọ̀ akérò wo nìyí?", tr) == (Intent.BUS_WHICH, "Which bus is this?")
    assert classify_multilingual("something odd", tr)[0] == Intent.UNKNOWN


def test_intents_existing_indic_keywords_still_work():
    assert classify("இது எந்தப் பேருந்து") == Intent.BUS_WHICH
    assert classify("ఇది చదువు") == Intent.OCR
    assert classify("আমি পড়ে গেছি") == Intent.SOS
    assert classify(unicodedata.normalize("NFD", "এটা পড়ো")) == Intent.OCR


# ---------------------------------------------------------------- native digits on bus boards
def test_native_digits_on_route_boards():
    assert ascii_digits("५३४") == "534" and ascii_digits("৫৩৪") == "534" and ascii_digits("٥٣٤") == "534"
    boxes = [TextBox("रूट ७६४", 0.9, (0, 0, 80, 40))]
    assert [c for c, _ in candidate_numbers(boxes)] == ["764"]


# ---------------------------------------------------------------- engine selection
def test_asr_candidates(tmp_path):
    cfg = ASRConfig(backend="auto", vosk_model_paths={"hi": str(tmp_path)})
    assert candidates(cfg, get("hi"))[:2] == ["vosk", "indicconformer"]
    assert candidates(cfg, get("sat"))[0] == "indicconformer"
    assert candidates(cfg, get("yo"))[-1] == "mms"
    assert candidates(ASRConfig(backend="whisper"), get("yo")) == ["whisper"]


def test_tts_auto_degrades_to_console_when_nothing_installed(monkeypatch, tmp_path):
    import smartsight.capabilities as cap
    monkeypatch.setattr(cap, "mms_tts_repo", lambda lang: None)
    eng = make_tts("auto", get("yo"), cache_dir=str(tmp_path), allow_gtts=False)
    assert isinstance(eng, _Console)


def test_neural_cache_roundtrip(tmp_path):
    from smartsight.audio_out import _CachedNeural

    class Fake(_CachedNeural):
        name = "fake"
        calls = 0

        def synth(self, text):
            Fake.calls += 1
            return np.ones(10), 16000

    eng = Fake(get("mr"), str(tmp_path))
    a1, sr = eng.render("थांबा!")
    a2, _ = eng.render("थांबा!")
    assert Fake.calls == 1 and sr == 16000 and np.array_equal(a1, a2)


def test_sentence_split_handles_scripts():
    assert _SENT_SPLIT.split("Pole at 3.5 meters. Stop!") == ["Pole at 3.5 meters.", "Stop!"]
    assert _SENT_SPLIT.split("रुकिए। सामने खंभा है।") == ["रुकिए।", "सामने खंभा है।"]
    assert _SENT_SPLIT.split("停下！前面有柱子。") == ["停下！", "前面有柱子。", ""]
    assert _SENT_SPLIT.split("رکیں۔ آگے کھمبا ہے۔") == ["رکیں۔", "آگے کھمبا ہے۔"]


def test_ocr_language_selection():
    assert ocr_languages(OCRConfig(), "en") == ["en"]
    assert ocr_languages(OCRConfig(), "ta") == ["en", "ta"]
    assert ocr_languages(OCRConfig(languages=["en", "hi"]), "bn") == ["en", "hi"]


# ---------------------------------------------------------------- app in many languages
@pytest.mark.parametrize("lang", ["en", "hi", "ta", "te", "bn", "mr", "sat", "yo", "ar", "zh"])
def test_app_runs_in_any_language(lang):
    from smartsight.app import SmartSightApp
    from smartsight.audio_out import SpeechQueue
    from smartsight.cane_link import SimCaneLink
    from smartsight.config import Config
    from smartsight.navigation import FixedLocation
    cfg = Config()
    cfg.audio.language = lang
    cfg.transport.gtfs_dir = None
    tr = FakeTranslator({"QUERY-IN-" + lang: "which bus is this"})
    app = SmartSightApp(cfg, vision=False, ocr=False, speech=SpeechQueue("console", lang, start_thread=False),
                        cane=SimCaneLink(), location=FixedLocation(28.7, 77.1), query_translator=tr)
    assert app.handle_query("what is in front of me") == i18n.t(lang, "nothing_ahead")
    assert app.handle_query("QUERY-IN-" + lang) == i18n.t(lang, "bus_no_active")
