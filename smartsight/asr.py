"""S301/S302 - Speech to text.

Backends (all offline after the first model download):
    vosk           - small streaming models; set a folder per language in asr.vosk_model_paths
    indicconformer - AI4Bharat IndicConformer 600M, all 22 Indian languages
    whisper        - faster-whisper, ~99 languages
    mms            - Meta MMS-1B-all, ~1,100 languages (per-language adapters)
    auto           - first available of: vosk folder -> indicconformer (Indian)
                     -> whisper -> mms. Default.
    keyboard       - type the query (development).

Push-to-talk: recording starts on trigger (cane talk button / Enter) and
stops after ~0.8 s of silence once speech was heard, or at record_seconds.
"""

from __future__ import annotations

import json
import logging
import queue
from pathlib import Path

import numpy as np

from .config import ASRConfig

log = logging.getLogger(__name__)


class KeyboardASR:
    def listen(self, prompt: str = "query> ") -> str:
        return input(prompt)          # EOFError propagates -> app shuts down cleanly


def record_until_silence(cfg: ASRConfig, rms_threshold: float = 500.0) -> np.ndarray:
    """int16 mono at cfg.sample_rate. Simple energy VAD - good enough for push-to-talk."""
    import sounddevice as sd
    block = int(cfg.sample_rate * 0.1)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    chunks, heard, silent = [], False, 0
    with sd.InputStream(samplerate=cfg.sample_rate, blocksize=block, dtype="int16", channels=1, callback=cb):
        for _ in range(int(cfg.record_seconds / 0.1)):
            c = q.get()
            chunks.append(c)
            loud = float(np.sqrt(np.mean(c.astype(np.float32) ** 2))) > rms_threshold
            if loud:
                heard, silent = True, 0
            elif heard:
                silent += 1
                if silent >= 8:
                    break
    return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)


class VoskASR:
    def __init__(self, cfg: ASRConfig, lang: str):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self.model = Model(cfg.vosk_model_paths[lang])
        self.cfg = cfg
        self._Rec = KaldiRecognizer

    def transcribe(self, audio: np.ndarray) -> str:
        rec = self._Rec(self.model, self.cfg.sample_rate)
        rec.AcceptWaveform(audio.tobytes())
        return json.loads(rec.FinalResult()).get("text", "")

    def listen(self, prompt: str = "") -> str:
        text = self.transcribe(record_until_silence(self.cfg))
        log.info("ASR(vosk): %r", text)
        return text


class IndicConformerASR:
    REPO = "ai4bharat/indic-conformer-600m-multilingual"

    def __init__(self, cfg: ASRConfig, lang: str):
        import torch
        from transformers import AutoModel
        self.torch, self.cfg, self.lang = torch, cfg, lang
        self.device = cfg.whisper_device if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.REPO, trust_remote_code=True).to(self.device)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        wav = self.torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return str(self.model(wav, self.lang, "rnnt")).strip()

    def listen(self, prompt: str = "") -> str:
        text = self.transcribe(record_until_silence(self.cfg))
        log.info("ASR(indicconformer/%s): %r", self.lang, text)
        return text


class MMSASR:
    REPO = "facebook/mms-1b-all"

    def __init__(self, cfg: ASRConfig, iso3: str):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.torch, self.cfg = torch, cfg
        self.processor = AutoProcessor.from_pretrained(self.REPO)
        if iso3 not in self.processor.tokenizer.vocab:
            raise LookupError(f"MMS has no ASR adapter for {iso3!r}")
        self.processor.tokenizer.set_target_lang(iso3)
        self.device = cfg.whisper_device if torch.cuda.is_available() else "cpu"
        self.model = Wav2Vec2ForCTC.from_pretrained(self.REPO, target_lang=iso3,
                                                    ignore_mismatched_sizes=True).to(self.device)
        self.iso3 = iso3

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        inputs = self.processor(audio.astype(np.float32) / 32768.0, sampling_rate=16000,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            ids = self.model(**inputs).logits.argmax(-1)[0]
        return self.processor.decode(ids).strip()

    def listen(self, prompt: str = "") -> str:
        text = self.transcribe(record_until_silence(self.cfg))
        log.info("ASR(mms/%s): %r", self.iso3, text)
        return text


class WhisperASR:
    def __init__(self, cfg: ASRConfig, lang: str):
        from faster_whisper import WhisperModel
        self.cfg, self.lang = cfg, lang
        try:
            self.model = WhisperModel(cfg.whisper_model, device=cfg.whisper_device,
                                      compute_type="float16" if cfg.whisper_device == "cuda" else "int8")
        except Exception:
            log.warning("whisper on %s failed, falling back to CPU", cfg.whisper_device)
            self.model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segs, _ = self.model.transcribe(audio.astype(np.float32) / 32768.0, language=self.lang,
                                        beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()

    def listen(self, prompt: str = "") -> str:
        text = self.transcribe(record_until_silence(self.cfg))
        log.info("ASR(whisper/%s): %r", self.lang, text)
        return text


def candidates(cfg: ASRConfig, lang) -> list[str]:
    """Backends to try, in order, for a languages.Language."""
    if cfg.backend != "auto":
        return [cfg.backend]
    out = []
    path = cfg.vosk_model_paths.get(lang.code)
    if path and Path(path).is_dir():
        out.append("vosk")
    if lang.indicconformer:
        out.append("indicconformer")
    from .capabilities import whisper_code, whisper_codes
    if whisper_code(lang) or not whisper_codes():   # unknown list (not installed) -> still try
        out.append("whisper")
    out.append("mms")
    return out


def choose_backend(cfg: ASRConfig, lang) -> str:
    """First candidate (kept for tests / logging)."""
    from .languages import get
    return candidates(cfg, get(lang) if isinstance(lang, str) else lang)[0]


def make_asr(cfg: ASRConfig, lang_code: str):
    from .capabilities import whisper_code
    from .languages import get
    if cfg.backend == "keyboard":
        return KeyboardASR()
    lang = get(lang_code)
    for backend in candidates(cfg, lang):
        try:
            if backend == "vosk":
                return VoskASR(cfg, lang.code)
            if backend == "indicconformer":
                return IndicConformerASR(cfg, lang.code.split("-")[0])
            if backend == "whisper":
                code = whisper_code(lang)
                if code is None:
                    continue
                return WhisperASR(cfg, code)
            if backend == "mms":
                return MMSASR(cfg, lang.iso3_individual)
        except Exception as e:
            log.warning("ASR backend %s unavailable for %s: %s", backend, lang.code, e)
    log.warning("no speech recognition for %s; using keyboard input", lang.code)
    return KeyboardASR()
