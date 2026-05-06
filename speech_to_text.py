"""
Speech-to-text module (offline) using Vosk + sounddevice.

Usage:
    stt = SpeechToTextManager()
    stt.toggle()   # Start listening
    text = stt.get_display_text()
    stt.toggle()   # Stop listening
"""

import json
import importlib
import os
import queue
import threading
import urllib.request
import zipfile

import sounddevice as sd

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


class SpeechToTextManager:
    """Background speech recognizer with start/stop toggle control."""

    def __init__(self, model_root="assets", sample_rate=16000):
        self.model_root = model_root
        self.sample_rate = sample_rate
        self.model_path = os.path.join(model_root, MODEL_NAME)

        self._model = None
        self._kaldi_recognizer_cls = None
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self.enabled = False
        self.last_text = ""
        self.partial_text = ""
        self.status = "OFF"

    def _ensure_model(self):
        """Download and extract the Vosk model if it's missing."""
        if os.path.exists(self.model_path):
            return

        os.makedirs(self.model_root, exist_ok=True)
        zip_path = os.path.join(self.model_root, f"{MODEL_NAME}.zip")

        print(f"[STT] Downloading model: {MODEL_NAME}")
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        print("[STT] Extracting model...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.model_root)
        try:
            os.remove(zip_path)
        except OSError:
            pass
        print("[STT] Model ready.")

    def _ensure_engine(self):
        if self._model is not None and self._kaldi_recognizer_cls is not None:
            return

        try:
            vosk_mod = importlib.import_module("vosk")
        except Exception as exc:
            raise RuntimeError(
                "Vosk is not installed. Run: pip install vosk"
            ) from exc

        self._ensure_model()
        self._model = vosk_mod.Model(self.model_path)
        self._kaldi_recognizer_cls = vosk_mod.KaldiRecognizer

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        if self.enabled:
            self._audio_queue.put(bytes(indata))

    def _listen_loop(self):
        recognizer = self._kaldi_recognizer_cls(self._model, self.sample_rate)

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                while not self._stop_event.is_set():
                    try:
                        data = self._audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            with self._lock:
                                self.last_text = text
                                self.partial_text = ""
                    else:
                        partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                        with self._lock:
                            self.partial_text = partial
        except Exception as exc:
            with self._lock:
                self.status = f"ERROR: {exc}"
            self.enabled = False

    def start(self):
        """Start speech recognition in a background thread."""
        if self.enabled:
            return

        try:
            self._ensure_engine()
        except Exception as exc:
            with self._lock:
                self.status = f"ERROR: {exc}"
            print(f"[STT] {exc}")
            return

        self._stop_event.clear()
        self._audio_queue = queue.Queue()

        with self._lock:
            self.status = "ON"

        self.enabled = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[STT] Listening enabled.")

    def stop(self):
        """Stop speech recognition and keep the most recent text."""
        if not self.enabled and not self._thread:
            with self._lock:
                self.status = "OFF"
            return

        self.enabled = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        with self._lock:
            self.partial_text = ""
            if not self.status.startswith("ERROR"):
                self.status = "OFF"

        print("[STT] Listening disabled.")

    def toggle(self):
        """Toggle speech recognition ON/OFF using the same key."""
        if self.enabled:
            self.stop()
        else:
            self.start()

    def get_display_text(self):
        """Return the text to render on screen (partial preferred while listening)."""
        with self._lock:
            if self.enabled and self.partial_text:
                return self.partial_text
            return self.last_text

    def get_status(self):
        with self._lock:
            return self.status
