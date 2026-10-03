# SmartSight

Companion software for the SmartSight assistive mobility system (Patent Invention Disclosure, 22 Sep 2026).
The VisionSense pipeline is [v2t-graph](https://github.com/khushi-070906/v2t-graph), used unmodified.

```
smartsight/            companion app (200)
firmware/              ESP32 cane unit (10/110)
tests/                 86 tests, run without torch or model weights
tools/                 translate_locales.py (any language), prewarm_tts.py
```

## Module map

| Patent | Step(s) | Module |
|---|---|---|
| 202 AI inference engine | S201–S214 | `visionsense.py` → v2t-graph via `v2t_bridge.py` |
| | S202/S203 frame relevance filter | `frame_filter.py` |
| | S211/S212 time-to-contact | `ttc.py` |
| 204 OCR engine | S304d–S306, S506/S507 | `ocr.py` (EasyOCR / Tesseract, English + user's script) |
| 206 voice interface | S301–S309 | `asr.py`, `intent.py`, `i18n.py`, `audio_out.py`, `translate.py`, `languages.py`, `capabilities.py` |
| 208 navigation / GPS | S304b, S406b | `navigation.py` |
| 210 SOS manager | S401–S409 | `sos.py`, `fall_detect.py` |
| Fig. 5 bus identification | S501–S514 | `transport.py` (Delhi GTFS) |
| 119 wireless link | — | `cane_link.py`, `protocol.py` |
| 110 ESP32 | — | `firmware/smartsight_cane/smartsight_cane.ino` |

## Setup (Windows)

Clone both repos side by side:

```
git clone https://github.com/khushi-070906/v2t-graph
git clone <this repo> smartsight
cd smartsight
python -m venv .venv && .venv\Scripts\activate
pip install -r ..\v2t-graph\requirements.txt
pip install -r requirements.txt -r requirements-dev.txt
copy config.example.json config.json
pytest
```

For the GPU, install the CUDA build of torch from pytorch.org before installing v2t-graph's requirements.

### Optional data and models

The app runs without any of these.

- **Delhi bus routes:** download the static GTFS from https://otd.delhi.gov.in and unzip it into `data/delhi_gtfs/`. It needs `routes.txt`, `trips.txt`, `stop_times.txt` and `stops.txt`. The first load builds a cache file.
- **Telegram SOS:**
  1. Create a bot with @BotFather and put its token in `config.json`.
  2. Have the guardian send the bot one message.
  3. Read their `chat_id` from `https://api.telegram.org/bot<TOKEN>/getUpdates`.

## Languages

`--lang` accepts **any** language code: `hi`, `mr`, `sat`, `sw`, `yo`, `ar`, `zh-Hant`, `pt-BR`. Run `python -m smartsight --check-lang <code>` to see which engines cover a language on your machine.

**What "all languages" means in practice.** SmartSight can work in a language when it can translate its phrases and speak them. That ceiling is set by open models, not by this code:

| Job | Engines (tried in order) | Coverage |
|---|---|---|
| Phrases + understanding queries | IndicTrans2 (Indian languages), NLLB-200 (everything else) | 22 Indian + ~200 world languages |
| Speech in | Vosk → IndicConformer → Whisper → Meta MMS-ASR | ~1,100 languages via MMS |
| Speech out | OS voice → Indic Parler-TTS → Meta MMS-TTS → gTTS (online) | ~1,100 languages via MMS |
| OCR | EasyOCR → Tesseract (language or script model) | ~120 languages; some scripts (e.g. Ol Chiki) have none |

The full voice-and-text experience therefore covers roughly the ~200 NLLB languages, and all 22 Eighth-Schedule Indian languages. Most of the world's ~7,000 languages have no writing system or no model, and can't be supported this way.

**Setting up a new language:**

1. Generate its phrase file:
   ```
   python tools/translate_locales.py --lang mr sw yo
   python tools/translate_locales.py --indian      # all 22 Indian languages
   python tools/translate_locales.py --nllb-all    # every NLLB language (~200)
   ```
   - Placeholders like `{label}` are protected through translation.
   - Any phrase that comes back broken is kept in English and listed under `meta.needs_review`.
   - Hand-written locales are never overwritten without `--overwrite`.
2. If it uses a neural voice (Parler/MMS), pre-render the urgent phrases with `python tools/prewarm_tts.py --lang mr`. Then an urgent alert never waits on synthesis. The cane vibrates instantly either way.
3. Run with `--lang mr`.

**Voice commands in any language.** For en, hi, ta, te and bn, built-in keyword rules match spoken commands instantly. In any other language, an unmatched query is translated to English (IndicTrans2/NLLB) and classified there. Turn this off with `audio.translate_queries: false`.

**Other language details:**
- Locales: `en` is the source. `hi`, `ta`, `te` and `bn` are hand-written drafts; everything else is machine-generated.
- Bus route boards: native digits (५३४, ৫৩৪, ٥٣٤ …) are read as 534.
- Unicode: queries are NFC-normalised before matching.

## Run

```
# no hardware, no models: voice/SOS/transport logic via keyboard
python -m smartsight --config config.json --no-vision --no-ocr

# full pipeline, laptop webcam, simulated cane
python -m smartsight --config config.json --device cuda

# glasses ESP32-CAM (stock CameraWebServer sketch) + real cane over BLE, Hindi
python -m smartsight --config config.json --camera http://192.168.1.50:81/stream --cane ble --lang hi
```

With keyboard ASR, each typed line is a spoken query. Example queries:
- "what is in front of me"
- "which bus is this"
- "where does this bus go"
- "read this"
- "kidhar jaun"
- "help me"
- "cancel"

You can also type these to simulate cane events: `!sos`, `!fall`, `!cancel`, `!near 40`.

## Cane firmware

Build environment:
- Arduino IDE, board "ESP32 Dev Module", **Arduino-ESP32 core 3.x** (the LEDC calls use the 3.x API).
- Library: `Adafruit_VL53L0X`.

The wiring table is at the top of the sketch. **The HC-SR04 ECHO pin outputs 5 V. Use a 1k/2k resistor divider before GPIO18.**

The cane works on its own, without the phone:
- **Obstacle alert:** the ultrasonic and ToF sensors drive the vibration motor and buzzer. Under 50 cm gives continuous vibration plus beeps; under 100 cm gives pulsed vibration.
- **Fall detection:** runs on the ESP32 at 100 Hz.
- **SOS button:** one press triggers SOS. Pressing again during the countdown cancels it. If the phone is disconnected, the button starts a local siren instead.

For bench testing over USB, set `"link": {"cane": "serial", "serial_port": "COMx"}`. Serial uses the same JSON-lines protocol as BLE (see `protocol.py`).

## Design notes

- **Tracking comes before pruning,** same as `live_camera_loop_tracked.py`. An object has to be confirmed for 3 frames before it can be spoken.
- **Pruning ranks; it never hides danger.** An Urgent object is reported even if `prune_graph` dropped it. While anything is Urgent or Soon, calm narration is suppressed.
- **Distance alone can trigger urgency.** With only a head-mounted camera, the tracker can't separate your own walking from object motion. So a static pole closing in would show ~0 depth velocity. A distance threshold catches it anyway.
- **Bus announcements need both vision and OCR.** OCR runs only on the upper 45% of a vision-confirmed bus. A number must also either exist in GTFS or be read twice before it's announced.
- **One audio channel.** Every module speaks through one priority queue. Urgent alerts drop lower-priority items and preempt at sentence boundaries.
- **Haptic refresh.** The app re-sends an active haptic pattern every second. The cane expires it after 3 s, so a dropped BLE link can't leave it buzzing.

## Known limitations

- **Heading is fixed at 0.** The camera is on the glasses and the IMU is on the cane, so head direction isn't measured.
  - Fix: add a head IMU, or use the cane-mounted-camera variant.
  - The MPU6050 has no magnetometer, so the IMU can't give a compass heading.
- **Depth calibration is indoor-only.** `calibration.json` was fitted indoors on person and door.
  - The default outdoor depth checkpoint has no calibration.
  - Collect outdoor ground truth before trusting outdoor distances.
- **No locale has had a native-speaker review.** The hand-written hi/ta/te/bn drafts and every machine-generated locale need checking before user trials. A mistranslated "Stop!" is a safety problem.
- **Model loading untested:** none of the speech or translation models (IndicTrans2, NLLB, IndicConformer, MMS, Parler, Whisper) have been run in this repo's test environment.
  - The tests cover selection, fallbacks, caching and placeholder safety using fakes.
  - Expect a first-run fix or two, especially in the `trust_remote_code` models' call signatures.
- **Low-resource quality:** ASR and MT quality for small languages can be poor. Check `--check-lang` and test with real speakers.
- **Disk and VRAM:** each extra model costs both.
  - NLLB-600M ≈ 2.5 GB, Parler ≈ 3.5 GB, MMS-1B ≈ 4 GB, IndicConformer ≈ 2.5 GB.
  - On an 8 GB GPU, run YOLO + depth + one speech model at a time. Use CPU for whichever stays idle.
- **Navigation is limited.** "Which way" means which direction is clear right now; there's no turn-by-turn routing.
- **Firmware status:** the firmware has not been compiled or flashed yet.
- **Test coverage:** the full YOLO + Depth Anything path is exercised by v2t-graph's own scripts, not by this repo's tests.
