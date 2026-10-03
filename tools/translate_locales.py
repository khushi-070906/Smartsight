"""Generate locale files (smartsight/locales/<code>.json) for any language
by machine-translating the English locale.

    python tools/translate_locales.py --lang yo sw am         # specific languages
    python tools/translate_locales.py --indian                # all 22 Eighth-Schedule languages
    python tools/translate_locales.py --nllb-all              # every language NLLB-200 supports (~200)
    python tools/translate_locales.py --lang mr --overwrite   # redo an existing file

Engines: IndicTrans2 for Indian languages, NLLB-200 for the rest (both run
on your GPU). Placeholders like {label} are protected through translation;
any phrase whose placeholders don't survive is kept in ENGLISH and listed
under meta.needs_review, so the app never crashes on a bad translation.

Hand-written locales (meta.source == "human" / "human-draft") are never
overwritten unless --overwrite is given.
Machine output is a DRAFT: get a native speaker to review it before users
depend on it - a mistranslated "Stop!" is a safety problem.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartsight import i18n  # noqa: E402
from smartsight.languages import SCHEDULED, get  # noqa: E402
from smartsight.translate import make_translator, protect, restore  # noqa: E402

log = logging.getLogger("translate_locales")


def translate_locale(lang, translator, en: dict) -> dict:
    phrases_en, labels_en = en["phrases"], en["labels"]
    keys = list(phrases_en)
    protected = [protect(phrases_en[k]) for k in keys]
    outs = translator.translate([p for p, _ in protected], "eng_Latn", lang.flores)

    phrases, bad = {}, []
    for k, (src, names), out in zip(keys, protected, outs):
        fixed = restore(out, names) if names else out
        if fixed is None or not fixed.strip():
            phrases[k] = phrases_en[k]
            bad.append(k)
        else:
            phrases[k] = fixed.strip()

    lab_keys = list(labels_en)
    lab_out = translator.translate([labels_en[k] for k in lab_keys], "eng_Latn", lang.flores)
    labels = {k: (v.strip().rstrip(".।") or labels_en[k]) for k, v in zip(lab_keys, lab_out)}

    engine = "indictrans2" if lang.indic else "nllb-200"
    return {"meta": {"language": lang.code, "name": lang.name, "source": f"machine:{engine}",
                     "reviewed_by_native_speaker": False, "needs_review": bad},
            "labels": labels, "phrases": phrases}


def nllb_codes() -> list[str]:
    from transformers import AutoTokenizer
    from smartsight.translate import NLLB
    tok = AutoTokenizer.from_pretrained(NLLB)
    return sorted(t for t in tok.additional_special_tokens if "_" in t and len(t) == 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", nargs="*", default=[])
    ap.add_argument("--indian", action="store_true")
    ap.add_argument("--nllb-all", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    codes = list(a.lang)
    if a.indian:
        codes += list(SCHEDULED)
    if a.nllb_all:
        codes += nllb_codes()
    if not codes:
        ap.error("give --lang, --indian or --nllb-all")

    en = json.loads((i18n.LOCALE_DIR / "en.json").read_text(encoding="utf-8"))
    translators = {}
    done, skipped, failed = [], [], []
    for code in dict.fromkeys(codes):
        try:
            lang = get(code)
        except ValueError:
            failed.append((code, "unknown code"))
            continue
        out = i18n.LOCALE_DIR / f"{lang.code}.json"
        if lang.code == "en":
            continue
        if out.exists() and not a.overwrite:
            skipped.append(lang.code)
            continue
        key = "indic" if lang.indic else "nllb"
        try:
            if key not in translators:
                translators[key] = make_translator(lang, to_english=False, device=a.device)
            data = translate_locale(lang, translators[key], en)
        except Exception as e:
            failed.append((lang.code, str(e)))
            continue
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        done.append(lang.code)
        log.info("%-8s %-22s %d phrases need review", lang.code, lang.name, len(data["meta"]["needs_review"]))

    log.info("\nwritten: %d  skipped (exists): %d  failed: %d", len(done), len(skipped), len(failed))
    for code, why in failed:
        log.info("  FAILED %s: %s", code, why)


if __name__ == "__main__":
    main()
