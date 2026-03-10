"""
Text-to-speech module using Kokoro TTS.
Call speak(text) from any script — it runs in a background thread.

Install:
    pip install kokoro sounddevice
    pip install misaki[en]    # English phonemizer backend
"""

import threading
import sounddevice as sd
from kokoro import KPipeline

SAMPLE_RATE = 24000
VOICE = "af_heart"   # American English female — change to e.g. "am_adam" for male

_pipeline = None
_lock = threading.Lock()


def _get_pipeline():
    """Lazy-load the Kokoro pipeline (thread-safe)."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                print("[TTS] Loading Kokoro pipeline ...")
                _pipeline = KPipeline(lang_code="a")  # 'a' = American English
                print("[TTS] Kokoro ready.")
    return _pipeline


def _prewarm():
    """Load the pipeline and run a silent warmup synthesis in the background."""
    try:
        pipeline = _get_pipeline()
        # Consume one chunk to JIT-compile the model
        for _, _, _ in pipeline(".", voice=VOICE, speed=1.0):
            break
    except Exception:
        pass

# Start pre-warming immediately when this module is imported
threading.Thread(target=_prewarm, daemon=True).start()


def _speak_blocking(text: str):
    """Synthesize and stream audio chunks as they arrive — minimises latency."""
    try:
        pipeline = _get_pipeline()
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for _, _, audio in pipeline(text, voice=VOICE, speed=1.0):
                stream.write(audio.reshape(-1, 1))
    except Exception as exc:
        print(f"[TTS] Error: {exc}")


def speak(text: str):
    """
    Speak text asynchronously. Returns immediately; audio plays in background.

    Args:
        text: The string to synthesize and play.
    """
    text = text.strip()
    if not text:
        return
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


if __name__ == "__main__":
    # Run synchronously so the process doesn't exit before audio finishes
    print("[TTS] Testing... (may download models on first run)")
    _speak_blocking("Hello! Text to speech is working.")
    print("[TTS] Done.")


