"""SmartSight companion software (Mobile Companion App, 200).

Module map to the Patent Invention Disclosure reference numerals:
    110/119  cane firmware + wireless link -> firmware/, cane_link.py, protocol.py
    202      AI inference engine           -> visionsense.py (wraps v2t-graph), frame_filter.py, ttc.py
    204      on-device OCR engine          -> ocr.py
    206      multilingual voice interface  -> asr.py, intent.py, i18n.py, audio_out.py
    208      navigation and GPS module     -> navigation.py, location.py
    210      emergency / SOS manager       -> sos.py, fall_detect.py
    (Fig.5)  bus identification            -> transport.py
"""

__version__ = "0.1.0"
