"""core/tts.py — Supertonic V3 TTS client.

speak() returns (t_first_audio, success) for accurate TTS latency measurement.
t_first_audio = perf_counter() when first audio chunk was written (synthesis time only).
"""
import io, os, queue, subprocess, tempfile, threading, time
from typing import Optional, Callable
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np


class TTSClient:
    def __init__(self, cfg: dict):
        port               = cfg["supertonic"]["port"]
        self.voice         = cfg["supertonic"]["default_voice"]
        base_url = cfg["supertonic"].get("url")
        if base_url:
            base_url = base_url.rstrip('/')
        else:
            base_url = f"http://localhost:{port}"
        self.url_compat    = f"{base_url}/v1/audio/speech"
        self.url_native    = f"{base_url}/v1/tts"
        self.current_play_rms = 0.0
        self.demo_audio_dir = None
        self.demo_meta = {}


    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(self, text: str, lang: str,
              stop_event: Optional[threading.Event] = None,
              on_audio_start: Optional[Callable[[float], None]] = None,
              on_audio_chunk: Optional[Callable[[np.ndarray, int], None]] = None) -> tuple:
        """Synthesize and play text.
        Returns (t_first_audio: float|None, success: bool).
        t_first_audio is perf_counter() at first audio chunk — use for TTS latency.
        success=False means no audio was played (synthesis failed or stop_event set).
        """
        if not text.strip():
            return None, False
        if stop_event and stop_event.is_set():
            return None, False
        wav = self._synthesize(text, lang)
        if wav:
            return self._play(wav, stop_event, on_audio_start=on_audio_start, on_audio_chunk=on_audio_chunk)
        return None, False

    def speak_streaming(
        self,
        sentence_queue: queue.Queue,
        lang: str,
        stop_event: Optional[threading.Event] = None,
        on_tts_send: Optional[Callable[[], None]] = None,
        on_audio_start: Optional[Callable[[float], None]] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> Optional[float]:
        """Drain sentence queue, synthesize+play each.
        Returns t_first_audio of the first sentence, or None if nothing played."""
        t_first = None
        while True:
            if stop_event and stop_event.is_set():
                self._drain_queue(sentence_queue)
                break
            try:
                sentence = sentence_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            if sentence is None:      # LLM done sentinel
                break
            if not sentence.strip():
                continue

            if on_tts_send:
                on_tts_send()
                on_tts_send = None

            wav = self._synthesize(sentence, lang)
            if wav and not (stop_event and stop_event.is_set()):
                t, ok = self._play(
                    wav, stop_event,
                    on_audio_start=on_audio_start if t_first is None else None,
                    on_audio_chunk=on_audio_chunk
                )
                if ok and t_first is None:
                    t_first = t
                if stop_event and stop_event.is_set():
                    self._drain_queue(sentence_queue)
                    break
        return t_first

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_demo_audio(self, wav_bytes: bytes):
        if not self.demo_audio_dir:
            return
        try:
            os.makedirs(self.demo_audio_dir, exist_ok=True)
            sc = self.demo_meta.get("scenario", 0)
            tn = self.demo_meta.get("turn", 0)
            role = self.demo_meta.get("role", "unknown")
            part = self.demo_meta.get("part", 0)
            filename = f"scenario_{sc}_turn_{tn}_{role}_part_{part}.wav"
            filepath = os.path.join(self.demo_audio_dir, filename)
            with open(filepath, "wb") as f:
                f.write(wav_bytes)
            self.demo_meta["part"] = part + 1
        except Exception:
            pass

    def _synthesize(self, text: str, lang: str) -> Optional[bytes]:
        """Return WAV bytes or None on failure, and save copy in demo mode."""
        wav = self._synthesize_raw(text, lang)
        if wav:
            self._save_demo_audio(wav)
        return wav

    def _synthesize_raw(self, text: str, lang: str) -> Optional[bytes]:
        """Return WAV bytes or None on failure."""
        # OpenAI-compat endpoint
        try:
            r = requests.post(self.url_compat,
                json={"model": "supertonic-3", "input": text,
                      "voice": self.voice, "language": lang}, timeout=15)
            if r.ok and r.content:
                return r.content
        except Exception:
            pass
        # Native endpoint
        try:
            r = requests.post(self.url_native,
                json={"text": text, "lang": lang, "voice": self.voice}, timeout=15)
            if r.ok and r.content:
                return r.content
        except Exception:
            pass
        # CLI fallback
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(["supertonic", "tts", "--text", text,
                            "--lang", lang, "--voice", self.voice,
                            "--output", tmp_path],
                           check=True, capture_output=True)
            with open(tmp_path, "rb") as f:
                data = f.read()
            os.unlink(tmp_path)
            return data
        except Exception:
            pass
        return None


    def _play(self, wav_bytes: bytes,
              stop_event: Optional[threading.Event] = None,
              on_audio_start: Optional[Callable[[float], None]] = None,
              on_audio_chunk: Optional[Callable[[np.ndarray, int], None]] = None) -> tuple:
        """Play audio in 50ms chunks.
        Returns (t_first_audio, success).
        t_first_audio = perf_counter() at first chunk write = TTS synthesis latency.
        """
        t_first = None
        success = False
        try:
            arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            chunk_frames = int(sr * 0.05)    # 50ms chunks for responsive stop_event checks
            idx = 0
            channels = arr.shape[1] if arr.ndim > 1 else 1
            
            # Try to open output stream, fall back to simulation if no audio device
            stream = None
            try:
                stream = sd.OutputStream(samplerate=sr, channels=channels, dtype="float32")
                stream.start()
            except Exception:
                stream = None

            try:
                while idx < len(arr):
                    if stop_event and stop_event.is_set():
                        break
                    end   = min(idx + chunk_frames, len(arr))
                    chunk = arr[idx:end]
                    
                    chunk_rms = np.sqrt(np.mean(chunk ** 2))
                    self.current_play_rms = float(chunk_rms * 32767.0)

                    if arr.ndim == 1:
                        chunk = chunk.reshape(-1, 1)
                    
                    if stream is not None:
                        stream.write(chunk)
                    else:
                        # Simulate audio play timing for the duration of the chunk
                        time.sleep(len(chunk) / sr)

                    if t_first is None:
                        t_first = time.perf_counter()   # ← actual first audio out
                        success = True
                        if on_audio_start:
                            on_audio_start(t_first)
                    if on_audio_chunk:
                        on_audio_chunk(arr[idx:end], sr)
                    idx = end
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self.current_play_rms = 0.0
        return t_first, success

    @staticmethod
    def _drain_queue(q: queue.Queue):
        """Drain stale items from a sentence queue after interruption."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
