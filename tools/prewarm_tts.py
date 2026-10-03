"""Pre-synthesize the phrases that must never wait on a neural TTS model:
every urgent object alert (all labels x all 7 clock positions) and the SOS
messages. Run once per language after choosing it:

    python tools/prewarm_tts.py --lang mr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartsight import i18n  # noqa: E402
from smartsight.audio_out import _SENT_SPLIT, _CachedNeural, make_tts  # noqa: E402
from smartsight.config import load_config  # noqa: E402
from smartsight.languages import get  # noqa: E402


def phrases(lang: str) -> list[str]:
    out = []
    labels = i18n.LABELS.get(lang) or i18n.LABELS["en"]
    for name in labels:
        for az in (-90, -60, -30, 0, 30, 60, 90):
            out.append(i18n.object_phrase(lang, name, 0.5, az, urgent=True))
    for key in ("sos_cancelled", "sos_sent", "sos_failed", "fall_detected", "cane_lost", "battery_low"):
        out.append(i18n.t(lang, key))
    for s in range(1, 31):
        out.append(i18n.t(lang, "sos_countdown", s=s))
    # SpeechQueue speaks sentence by sentence, so cache exactly those pieces
    pieces = [p for text in out for p in _SENT_SPLIT.split(text) if p.strip()]
    return list(dict.fromkeys(pieces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    lang = get(a.lang)
    eng = make_tts(cfg.audio.tts_backend, lang, cache_dir=cfg.audio.tts_cache_dir, device=cfg.audio.tts_device,
                   allow_gtts=False)
    if not isinstance(eng, _CachedNeural):
        print(f"{type(eng).__name__} speaks instantly; nothing to pre-render.")
        return
    items = phrases(lang.code)
    t0 = time.time()
    for i, p in enumerate(items, 1):
        eng.render(p)
        if i % 25 == 0:
            print(f"{i}/{len(items)}  ({time.time() - t0:.0f}s)")
    print(f"cached {len(items)} phrases in {eng.cache}")


if __name__ == "__main__":
    main()
