"""S214 / S309 / S512 - Single prioritized speech channel to the bone-conduction headset (300).

Every module (VisionSense alerts, voice answers, OCR text, bus announcements,
SOS countdown) speaks through ONE queue so utterances never overlap.
The headset is simply the OS default audio output (Bluetooth bone-conduction
headphones paired to the phone/laptop) - no special driver.

Priorities (lower = more important):
    0 URGENT     preempts: queued lower-priority items are dropped
    1 ALERT
    2 RESPONSE   answers to the user's own questions
    3 NARRATION  dropped if older than narration_ttl_s when its turn comes

Preemption is at sentence granularity: long texts are split into sentences
and the queue is re-checked between them, so an urgent alert waits at most
one short sentence. (pyttsx3 cannot be reliably stopped from another thread.)

Backends ("auto" tries them in this order):
    pyttsx3 - offline OS voice, used only if a voice for the language is installed
    parler  - AI4Bharat Indic Parler-TTS (22 Indian languages + English), GPU
    mms     - Meta MMS-TTS, one small VITS model per language (~1,100 languages)
    gtts    - Google TTS, online (if tts_fallback_gtts)
    console - prints; tests and headless dev
Neural backends are slow to synthesize (~0.3-2 s per sentence), so every
utterance is cached on disk by (backend, language, text). Urgent phrases
repeat a lot; pre-render them with tools/prewarm_tts.py so an urgent alert
never waits on synthesis. The cane's vibration fires instantly regardless.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# sentence ends: Latin, Devanagari danda/double danda, Urdu full stop, Arabic ?,
# and CJK full-width marks (which are not followed by a space)
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥۔؟])\s+|(?<=[。！？])")


class _Pyttsx3:
    SPECIAL = {"hi": ("kalpana", "hemant"), "ta": ("valluvar",), "en": ("zira", "david", "hazel")}

    def __init__(self, lang, rate: int = 170):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.has_voice = self.set_language(lang)

    def set_language(self, lang) -> bool:
        base = lang.code.split("-")[0]
        hints = (lang.name.split(" (")[0].lower(), f"{base}_", f"{base}-", f"{lang.iso3}_") + self.SPECIAL.get(base, ())
        for v in self.engine.getProperty("voices"):
            hay = (v.id + " " + (v.name or "") + " " +
                   " ".join(str(x) for x in (getattr(v, "languages", None) or []))).lower()
            if any(h in hay for h in hints):
                self.engine.setProperty("voice", v.id)
                return True
        return False

    def say(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()


class _CachedNeural:
    """Base for neural TTS: synthesize -> cache -> play via sounddevice."""
    name = "neural"

    def __init__(self, lang, cache_dir: str):
        self.lang = lang
        self.cache = Path(cache_dir) / self.name / lang.code
        self.cache.mkdir(parents=True, exist_ok=True)

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def render(self, text: str) -> tuple[np.ndarray, int]:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]
        hits = list(self.cache.glob(f"{key}_*.npy"))
        if hits:
            sr = int(hits[0].stem.split("_")[1])
            return np.load(hits[0]), sr
        audio, sr = self.synth(text)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        np.save(self.cache / f"{key}_{sr}.npy", audio)
        return audio, sr

    def say(self, text: str) -> None:
        import sounddevice as sd
        audio, sr = self.render(text)
        sd.play(audio, sr)
        sd.wait()


class _Parler(_CachedNeural):
    name = "parler"
    REPO = "ai4bharat/indic-parler-tts"
    DESCRIPTION = ("A female speaker speaks clearly at a moderate speed and pitch. "
                   "The recording is of very high quality, with the voice sounding clear and very close up.")

    def __init__(self, lang, cache_dir: str, device: str = "cuda"):
        super().__init__(lang, cache_dir)
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(self.REPO).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(self.REPO)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self._desc = self.desc_tok(self.DESCRIPTION, return_tensors="pt").to(self.device)

    def synth(self, text: str):
        prompt = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(input_ids=self._desc.input_ids, attention_mask=self._desc.attention_mask,
                                      prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask)
        return gen.cpu().numpy(), self.model.config.sampling_rate


class _MMS(_CachedNeural):
    name = "mms"

    def __init__(self, lang, cache_dir: str, repo: str, device: str = "cuda"):
        super().__init__(lang, cache_dir)
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = VitsModel.from_pretrained(repo).to(self.device)
        self.uroman = None
        if getattr(self.tok, "is_uroman", False):     # model expects romanized input
            import uroman
            self.uroman = uroman.Uroman()

    def synth(self, text: str):
        if self.uroman is not None:
            text = self.uroman.romanize_string(text)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            wav = self.model(**inputs).waveform
        return wav.cpu().numpy(), self.model.config.sampling_rate


class _GTTS:
    def __init__(self, lang):
        from gtts.lang import tts_langs
        import pygame
        supported = tts_langs()
        base = lang.code.split("-")[0]
        self.code = next((c for c in (lang.code, base, "zh-TW" if lang.flores == "zho_Hant" else None,
                                      "zh-CN" if lang.flores == "zho_Hans" else None) if c and c in supported), None)
        if self.code is None:
            raise LookupError(f"gTTS does not support {lang.code}")
        pygame.mixer.init()

    def say(self, text: str) -> None:
        import io
        import pygame
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=self.code).write_to_fp(buf)
        buf.seek(0)
        pygame.mixer.music.load(buf, "mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)


class _Console:
    def __init__(self, lang=None):
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)
        print(f"[SPEAK] {text}", flush=True)


def make_tts(backend: str, lang, *, cache_dir: str = "cache/tts", device: str = "cuda",
             allow_gtts: bool = True):
    """Returns a speaking engine for languages.Language `lang`."""
    from .capabilities import mms_tts_repo
    if backend == "console":
        return _Console(lang)
    order = [backend] if backend != "auto" else ["pyttsx3", "parler", "mms"] + (["gtts"] if allow_gtts else [])
    fallback = None
    for b in order:
        try:
            if b == "pyttsx3":
                eng = _Pyttsx3(lang)
                if eng.has_voice or backend == "pyttsx3":
                    return eng
                fallback = eng
            elif b == "parler" and (lang.parler or backend == "parler"):
                return _Parler(lang, cache_dir, device)
            elif b == "mms":
                repo = mms_tts_repo(lang)
                if repo:
                    return _MMS(lang, cache_dir, repo, device)
            elif b == "gtts":
                return _GTTS(lang)
        except Exception as e:
            log.warning("TTS backend %s unavailable for %s: %s", b, lang.code, e)
    if fallback is not None:
        log.error("no voice for %s; using the default OS voice - speech may be unintelligible", lang.code)
        return fallback
    log.error("no TTS available for %s; printing instead", lang.code)
    return _Console(lang)


class SpeechQueue:
    def __init__(self, backend: str = "auto", lang: str = "en", narration_ttl_s: float = 2.0,
                 start_thread: bool = True, gtts_fallback: bool = True, cache_dir: str = "cache/tts",
                 device: str = "cuda"):
        self.backend_name = backend
        self.gtts_fallback = gtts_fallback
        self.cache_dir, self.device = cache_dir, device
        self.lang = lang
        self.narration_ttl_s = narration_ttl_s
        self._heap: list = []
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._engine = None
        self._stop = False
        self.history: list[tuple[int, str]] = []
        if start_thread:
            threading.Thread(target=self._worker, daemon=True, name="tts").start()

    def _make_engine(self):
        from .languages import get
        return make_tts(self.backend_name, get(self.lang), cache_dir=self.cache_dir,
                        device=self.device, allow_gtts=self.gtts_fallback)

    def say(self, text: str, priority: int = 2) -> None:
        if not text:
            return
        with self._cv:
            if priority == 0:  # preempt: drop everything less important
                self._heap = [e for e in self._heap if e[0] == 0]
                heapq.heapify(self._heap)
            heapq.heappush(self._heap, (priority, next(self._seq), time.monotonic(), text))
            self._cv.notify()

    def pending(self) -> int:
        with self._cv:
            return len(self._heap)

    def _next(self, block: bool = True):
        with self._cv:
            while not self._heap:
                if not block or self._stop:
                    return None
                self._cv.wait(0.5)
            return heapq.heappop(self._heap)

    def _preempted(self, prio: int) -> bool:
        with self._cv:
            return bool(self._heap) and self._heap[0][0] < prio

    def speak_one(self, block: bool = True) -> bool:
        """Pops and speaks one item. Returns False if nothing was spoken."""
        if self._engine is None:
            self._engine = self._make_engine()   # created on the speaking thread (pyttsx3 needs that)
        item = self._next(block)
        if item is None:
            return False
        prio, _, t_enq, text = item
        if prio == 3 and time.monotonic() - t_enq > self.narration_ttl_s:
            return False
        self.history.append((prio, text))
        for sentence in _SENT_SPLIT.split(text):
            if self._preempted(prio):
                log.debug("preempted after partial: %r", sentence)
                break
            try:
                self._engine.say(sentence)
            except Exception:
                log.exception("TTS failed")
                break
        return True

    def _worker(self) -> None:
        while not self._stop:
            try:
                self.speak_one(block=True)
            except Exception:
                log.exception("speech worker error")
                time.sleep(0.5)

    def stop(self) -> None:
        self._stop = True
        with self._cv:
            self._cv.notify_all()
