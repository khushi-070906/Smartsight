"""python -m smartsight [--config config.json] [--no-vision] [--no-ocr] [--lang CODE] [--camera URL] [--cane sim|ble|serial]

--lang takes any BCP-47 / ISO 639 code (hi, mr, sat, sw, yo, ar, zh-Hant, ...).
--check-lang CODE prints which engines cover that language on this machine.
"""

from __future__ import annotations

import argparse
import logging

from .app import SmartSightApp
from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="smartsight", description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-vision", action="store_true", help="skip VisionSense (no torch/v2t-graph needed)")
    ap.add_argument("--no-ocr", action="store_true", help="skip EasyOCR")
    ap.add_argument("--lang", help="any language code, e.g. hi, mr, ta, sw, yo, ar, zh-Hant")
    ap.add_argument("--check-lang", metavar="CODE", help="report engine coverage for a language and exit")
    ap.add_argument("--camera", help="webcam index, video file, or ESP32-CAM stream URL")
    ap.add_argument("--cane", choices=["sim", "ble", "serial"])
    ap.add_argument("--device", help="cuda | cpu")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s")
    if a.check_lang:
        from .capabilities import report
        from .languages import get
        print(report(get(a.check_lang)).text())
        return

    cfg = load_config(a.config)
    if a.lang:
        from .languages import get
        cfg.audio.language = get(a.lang).code      # validates + normalises (hi-IN -> hi)
    if a.camera:
        cfg.link.camera = a.camera
    if a.cane:
        cfg.link.cane = a.cane
    if a.device:
        cfg.vision.device = a.device
        cfg.ocr.gpu = a.device == "cuda"

    SmartSightApp(cfg, vision=not a.no_vision, ocr=not a.no_ocr).run()


if __name__ == "__main__":
    main()
