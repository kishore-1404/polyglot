"""core/pipeline.py — Voice pipeline orchestrator.

State machine: LISTENING → RECORDING → PROCESSING → SPEAKING → LISTENING

Architecture notes:
- Main thread runs _tick() loop, which BLOCKS during _process().
- Barge-in detection runs in the mic callback (audio thread) — the only
  thread that runs concurrently with _process().
- _stop_play event propagates from mic-callback → tts._play() loop (50ms latency).
- _drain_mic_queue() after TTS prevents echo feedback.
  On barge-in it skips the full drain so user speech is preserved.
"""
import io, queue, threading, time, wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from core.vad  import VADProcessor
from core.llm  import LLMClient
from core.tts  import TTSClient
from utils.latency import TurnLatency

# Try to import WebRTC audio processing
try:
    from aec_audio_processing import AudioProcessor
    AEC_AVAILABLE = True
except ImportError:
    AEC_AVAILABLE = False

_resamplers = {}

def resample_chunk(chunk: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample chunk from orig_sr to target_sr and convert float32 to int16."""
    if chunk.ndim > 1:
        chunk = np.mean(chunk, axis=1)
        
    if orig_sr == target_sr:
        resampled = chunk
    else:
        try:
            import torch
            import torchaudio
            key = (orig_sr, target_sr)
            if key not in _resamplers:
                _resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
            tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            resampled = _resamplers[key](tensor).squeeze(0).numpy()
        except Exception:
            try:
                from scipy.signal import resample
                num_samples = int(len(chunk) * target_sr / orig_sr)
                resampled = resample(chunk, num_samples)
            except Exception:
                x_old = np.linspace(0, len(chunk) - 1, len(chunk))
                num_samples = int(len(chunk) * target_sr / orig_sr)
                x_new = np.linspace(0, len(chunk) - 1, num_samples)
                resampled = np.interp(x_new, x_old, chunk)
                
    return np.clip(resampled * 32768.0, -32768.0, 32767.0).astype(np.int16)


class AudioBuffer:
    """Thread-safe ring-like audio buffer for speaker reference stream."""
    def __init__(self):
        self._buffer = np.zeros(0, dtype=np.int16)
        self._lock = threading.Lock()
        
    def extend(self, data: np.ndarray):
        with self._lock:
            self._buffer = np.concatenate((self._buffer, data))
            if len(self._buffer) > 160000:
                self._buffer = self._buffer[-160000:]
                
    def read(self, n: int) -> np.ndarray:
        with self._lock:
            if len(self._buffer) >= n:
                data = self._buffer[:n]
                self._buffer = self._buffer[n:]
                return data
            else:
                data = np.zeros(n, dtype=np.int16)
                if len(self._buffer) > 0:
                    data[:len(self._buffer)] = self._buffer
                    self._buffer = np.zeros(0, dtype=np.int16)
                return data
                
    def clear(self):
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.int16)


class State:
    LISTENING   = "listening"
    RECORDING   = "recording"
    PROCESSING  = "processing"
    SPEAKING    = "speaking"
    MUTED       = "muted"


class Pipeline:
    def __init__(
        self,
        cfg:           dict,
        on_state:      Optional[Callable[[str],  None]] = None,
        on_turn:       Optional[Callable[[dict], None]] = None,
        on_text_ready: Optional[Callable[[dict], None]] = None,
        on_log:        Optional[Callable[[str],  None]] = None,
        on_latency_update: Optional[Callable[[dict], None]] = None,
    ):
        self.cfg  = cfg
        self.vad  = VADProcessor(cfg)
        self.llm  = LLMClient(cfg)
        self.tts  = TTSClient(cfg)

        self.on_state      = on_state      or (lambda _: None)
        self.on_turn       = on_turn       or (lambda _: None)
        self.on_text_ready = on_text_ready or (lambda _: None)
        self.on_log        = on_log        or (lambda _: None)
        self.on_latency_update = on_latency_update or (lambda _: None)

        # WebRTC AEC state
        self._ref_audio_buffer = AudioBuffer()
        self._aec_processor = None
        self._aec_mic_buf = np.zeros(0, dtype=np.int16)
        self._aec_ref_buf = np.zeros(0, dtype=np.int16)
        self._aec_out_buf = np.zeros(0, dtype=np.int16)

        if AEC_AVAILABLE:
            try:
                self._aec_processor = AudioProcessor(enable_aec=True, enable_ns=True, enable_agc=True)
                self._aec_processor.set_stream_format(16000, 1)
                self._aec_processor.set_reverse_stream_format(16000, 1)
                delay_ms = int(cfg.get("features", {}).get("webrtc_aec_delay_ms", 50))
                self._aec_processor.set_stream_delay(delay_ms)
                self._log(f"🎙️ WebRTC AEC initialized successfully (delay: {delay_ms}ms).")
            except Exception as e:
                self._aec_processor = None
                self._log(f"⚠️ Failed to initialize WebRTC AEC: {e}. Falling back to RMS thresholding.")
        else:
            self._log("ℹ️ WebRTC AEC not available (aec-audio-processing package missing). Falling back to RMS thresholding.")

        vad      = cfg["vad"]
        fms      = self.vad.frame_ms
        self._silence_limit  = int(float(vad["silence_trigger"]) * 1000 / fms)
        self._max_frames     = int(float(vad["max_speech_secs"]) * 1000 / fms)
        self._overlap_frames = int(float(vad["overlap_secs"])    * 1000 / fms)

        feat = cfg.get("features", {})
        self._barge_enabled   = feat.get("barge_in", True)
        self._barge_threshold = float(feat.get("barge_in_threshold", 0.85))
        self._barge_n         = int(feat.get("barge_in_frames", 4))
        self._streaming_tts   = feat.get("streaming_tts", True)
        self._history_limit   = int(feat.get("history_turns", 20))
        self._echo_decay_ms   = int(cfg.get("features", {}).get("echo_decay_ms", 280))

        self._audio_q        = queue.Queue()
        self._is_playing     = threading.Event()   # TTS active → gate mic
        self._stop_play      = threading.Event()   # interrupt TTS
        self._barge_in_flag  = threading.Event()   # set by mic-callback on barge-in
        self._barge_count_cb = 0                   # consecutive barge frames (mic-callback)

        self._state         = State.LISTENING
        self._speech_frames = []
        self._silence_count = 0
        self._muted         = False
        self._running       = False

        self.history      = []
        self.current_lang = cfg["languages"]["default"]

        # Demo Mode state
        self._demo_active = False
        self._demo_scenario = None
        self._demo_turn_idx = 0
        self._demo_auto_advance = True
        self._demo_thread = None
        self._demo_stop_event = threading.Event()
        self._demo_turn_done = threading.Event()
        self.on_demo_state = lambda _: None


    # ── Public ────────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        try:
            with sd.InputStream(
                samplerate=16000, channels=1, dtype="int16",
                blocksize=self.vad.frame_samples,
                callback=self._mic_callback,
            ):
                self._set_state(State.LISTENING)
                while self._running:
                    self._tick()
        except Exception as e:
            self._log(f"⚠️ sounddevice.InputStream failed: {e}. Running in headless/demo-only mode.")
            self._set_state(State.LISTENING)
            while self._running:
                self._tick()

    def stop(self):
        self._running = False
        self._stop_play.set()

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        self._set_state(State.MUTED if self._muted else State.LISTENING)
        self._log(f"🔇 Muted" if self._muted else "🎙️  Unmuted")
        return self._muted

    def handle_control(self, action: str):
        if action == "mute_toggle":
            self.toggle_mute()

    # ── Mic callback (audio thread) ───────────────────────────────────────────

    def _mic_callback(self, indata, frames, time_info, status):
        if self._demo_active:
            return
            
        # 1. Gate mic input during TTS playback if barge-in is disabled
        if self._is_playing.is_set() and not self._barge_enabled:
            return
            
        frame = indata[:, 0].copy()

        # 2. WebRTC AEC path (if processor is active)
        if self._aec_processor is not None:
            self._aec_mic_buf = np.concatenate((self._aec_mic_buf, frame))
            ref_frame = self._ref_audio_buffer.read(len(frame))
            self._aec_ref_buf = np.concatenate((self._aec_ref_buf, ref_frame))
            
            while len(self._aec_mic_buf) >= 160 and len(self._aec_ref_buf) >= 160:
                mic_chunk = self._aec_mic_buf[:160]
                ref_chunk = self._aec_ref_buf[:160]
                self._aec_mic_buf = self._aec_mic_buf[160:]
                self._aec_ref_buf = self._aec_ref_buf[160:]
                
                try:
                    self._aec_processor.process_reverse_stream(ref_chunk.tobytes())
                    clean_chunk_bytes = self._aec_processor.process_stream(mic_chunk.tobytes())
                    clean_chunk = np.frombuffer(clean_chunk_bytes, dtype=np.int16)
                except Exception:
                    clean_chunk = mic_chunk
                    
                self._aec_out_buf = np.concatenate((self._aec_out_buf, clean_chunk))
                
            while len(self._aec_out_buf) >= 512:
                out_frame = self._aec_out_buf[:512]
                self._aec_out_buf = self._aec_out_buf[512:]
                self._audio_q.put(out_frame)
                
                # Check for barge-in on the cleaned frame
                if (self._is_playing.is_set()
                        and self._barge_enabled
                        and not self._stop_play.is_set()):
                    try:
                        prob = self.vad.get_prob(out_frame)
                    except Exception:
                        prob = 0.0

                    mic_rms = np.sqrt(np.mean(out_frame.astype(np.float32) ** 2))
                    play_rms = getattr(self.tts, "current_play_rms", 0.0)

                    coeff = float(self.cfg.get("features", {}).get("barge_in_echo_coeff", 0.3))
                    offset = float(self.cfg.get("features", {}).get("barge_in_rms_offset", 400.0))

                    is_louder = mic_rms > (coeff * play_rms + offset)

                    if prob >= self._barge_threshold and is_louder:
                        self._barge_count_cb += 1
                        if self._barge_count_cb >= self._barge_n:
                            self._barge_count_cb = 0
                            self._barge_in_flag.set()
                            self._stop_play.set()
                    else:
                        self._barge_count_cb = 0
        else:
            # 3. Fallback RMS path (if AEC not available)
            self._audio_q.put(frame)

            if (self._is_playing.is_set()
                    and self._barge_enabled
                    and not self._stop_play.is_set()):
                try:
                    prob = self.vad.get_prob(frame)
                except Exception:
                    prob = 0.0

                mic_rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
                play_rms = getattr(self.tts, "current_play_rms", 0.0)

                coeff = float(self.cfg.get("features", {}).get("barge_in_echo_coeff", 0.3))
                offset = float(self.cfg.get("features", {}).get("barge_in_rms_offset", 400.0))

                is_louder = mic_rms > (coeff * play_rms + offset)

                if prob >= self._barge_threshold and is_louder:
                    self._barge_count_cb += 1
                    if self._barge_count_cb >= self._barge_n:
                        self._barge_count_cb = 0
                        self._barge_in_flag.set()
                        self._stop_play.set()
                else:
                    self._barge_count_cb = 0

    # ── Main loop tick ────────────────────────────────────────────────────────

    def _tick(self):
        try:
            frame = self._audio_q.get(timeout=0.5)
        except queue.Empty:
            return

        if self._muted:
            return

        # After _process() returns, _is_playing is already cleared.
        # This guard handles the edge case where clearing is slightly delayed.
        if self._is_playing.is_set():
            return

        self._stop_play.clear()
        prob     = self.vad.get_prob(frame)
        is_voice = prob >= self.vad.threshold

        if is_voice:
            if self._state != State.RECORDING:
                self._set_state(State.RECORDING)
                self._speech_frames = []
                self._silence_count = 0

            self._speech_frames.append(frame)
            self._silence_count = 0

            if len(self._speech_frames) >= self._max_frames:
                dur = len(self._speech_frames) * self.vad.frame_ms / 1000
                self._log(f"⚡ Force-flush at {dur:.1f}s")
                flush   = self._speech_frames[:-self._overlap_frames]
                overlap = self._speech_frames[-self._overlap_frames:]
                self._process(flush)
                self._speech_frames = list(overlap)
                self._silence_count = 0
                if not self._speech_frames:
                    self._set_state(State.LISTENING)

        else:
            if self._state == State.RECORDING:
                self._speech_frames.append(frame)
                self._silence_count += 1

                if self._silence_count >= self._silence_limit:
                    if self._speech_frames:
                        self._process(self._speech_frames)
                    self._speech_frames = []
                    self._silence_count = 0
                    self._set_state(State.LISTENING)

    # ── Turn processing ───────────────────────────────────────────────────────

    def _process(self, frames: list):
        if self._demo_active:
            self.tts.demo_audio_dir = "logs/demo_audio"
            self.tts.demo_meta = {
                "scenario": self._demo_scenario,
                "turn": self._demo_turn_idx + 1,
                "role": "bot",
                "part": 0
            }

        lat = TurnLatency()
        lat.t_end_of_speech = time.perf_counter()
        wav = self._frames_to_wav(frames)
        self._barge_in_flag.clear()

        self._set_state(State.PROCESSING)
        lat.t_llm_send = time.perf_counter()

        if self._streaming_tts:
            lang, transcript, response, tts_ok, interrupted = \
                self._process_streaming(wav, lat)
        else:
            lang, transcript, response, tts_ok, interrupted = \
                self._process_standard(wav, lat)

        self.current_lang = lang

        if transcript:
            self.history.append({"role": "user",      "content": f"[{lang.upper()}] {transcript}"})
        self.history.append(    {"role": "assistant",  "content": response})
        if len(self.history) > self._history_limit:
            self.history = self.history[-self._history_limit:]

        self.on_turn({
            "lang":        lang,
            "transcript":  transcript,
            "response":    response,
            "tts_ok":      tts_ok,
            "interrupted": interrupted,
            "latency":     lat.to_dict(),
        })

        if self._demo_active:
            self._demo_turn_done.set()


    def _process_standard(self, wav: bytes, lat: TurnLatency):
        """LLM → on_text_ready → TTS. Returns (lang, transcript, response, tts_ok, interrupted)."""
        cancel = threading.Event()
        lang, transcript, response = self.llm.query(
            wav, self.history, self.current_lang,
            cancel_event=cancel,
            on_first_token=lambda: setattr(lat, "t_first_token", time.perf_counter()),
        )
        lat.t_llm_done = time.perf_counter()

        # Show text in UI before TTS starts
        self.on_text_ready({
            "lang": lang, "transcript": transcript, "response": response,
            "tts_ok": None, "interrupted": False,
            "latency": lat.to_dict(),
        })

        self._set_state(State.SPEAKING)
        lat.t_tts_send = time.perf_counter()
        self._is_playing.set()
        tts_ok      = False
        interrupted = False
        try:
            def on_audio_start(t_first):
                lat.t_audio_start = t_first
                self.on_latency_update(lat.to_dict())

            self._clear_aec_buffers()
            t_first, tts_ok = self.tts.speak(
                response, lang, self._stop_play,
                on_audio_start=on_audio_start,
                on_audio_chunk=self._on_play_audio_chunk
            )
            lat.t_audio_start = t_first if t_first else lat.t_tts_send
            interrupted = self._barge_in_flag.is_set()
        finally:
            self._drain_mic_queue()
            self._is_playing.clear()

        return lang, transcript, response, tts_ok, interrupted

    def _process_streaming(self, wav: bytes, lat: TurnLatency):
        """Concurrent LLM + TTS. on_text_ready fires from LLM thread when response ready."""
        sentence_queue = queue.Queue()
        result         = {}
        cancel         = threading.Event()
        text_ready_ev  = threading.Event()

        def llm_thread():
            lang, transcript, response = self.llm.query(
                wav, self.history, self.current_lang,
                sentence_queue=sentence_queue,
                cancel_event=cancel,
                on_first_token=lambda: setattr(lat, "t_first_token", time.perf_counter()),
            )
            lat.t_llm_done = time.perf_counter()
            result.update({"lang": lang, "transcript": transcript, "response": response})

            if not text_ready_ev.is_set():
                text_ready_ev.set()
                self.on_text_ready({
                    "lang": lang, "transcript": transcript, "response": response,
                    "tts_ok": None, "interrupted": False,
                    "latency": lat.to_dict(),
                })
            else:
                self.on_latency_update(lat.to_dict())

        t = threading.Thread(target=llm_thread, daemon=True)
        t.start()

        self._set_state(State.SPEAKING)
        self._is_playing.set()
        tts_ok      = False
        interrupted = False
        try:
            def on_tts_send():
                lat.t_tts_send = time.perf_counter()

            def on_audio_start(t_first):
                lat.t_audio_start = t_first
                self.on_latency_update(lat.to_dict())

            self._clear_aec_buffers()
            t_first = self.tts.speak_streaming(
                sentence_queue, self.current_lang, self._stop_play,
                on_tts_send=on_tts_send, on_audio_start=on_audio_start,
                on_audio_chunk=self._on_play_audio_chunk
            )
            tts_ok            = t_first is not None
            lat.t_audio_start = t_first if t_first else lat.t_tts_send
            interrupted       = self._barge_in_flag.is_set()
        finally:
            self._drain_mic_queue()
            self._is_playing.clear()

        t.join(timeout=5)

        # Fallback: if streaming TTS failed to play anything (tts_ok is False) but we have a response, play it standard way now
        if not tts_ok and result.get("response") and not (self._stop_play.is_set() or self._barge_in_flag.is_set()):
            self._set_state(State.SPEAKING)
            self._is_playing.set()
            try:
                def on_audio_start_fallback(t_first):
                    lat.t_audio_start = t_first
                    self.on_latency_update(lat.to_dict())

                if getattr(lat, "t_tts_send", None) is None:
                    lat.t_tts_send = time.perf_counter()

                self._clear_aec_buffers()
                t_first, success = self.tts.speak(
                    result["response"], result.get("lang", self.current_lang), self._stop_play,
                    on_audio_start=on_audio_start_fallback,
                    on_audio_chunk=self._on_play_audio_chunk
                )
                if success:
                    tts_ok = True
                    if t_first:
                        lat.t_audio_start = t_first
            finally:
                self._drain_mic_queue()
                self._is_playing.clear()

        # Fire on_text_ready if LLM thread died before it could
        if not text_ready_ev.is_set() and result:
            text_ready_ev.set()
            self.on_text_ready({
                "lang": result.get("lang", self.current_lang),
                "transcript": result.get("transcript", ""),
                "response": result.get("response", ""),
                "tts_ok": tts_ok, "interrupted": interrupted,
                "latency": lat.to_dict(),
            })

        return (result.get("lang",       self.current_lang),
                result.get("transcript", ""),
                result.get("response",   ""),
                tts_ok, interrupted)


    # ── Helpers ────────────────────────────────────────────────────────────────

    def _clear_aec_buffers(self):
        if hasattr(self, "_ref_audio_buffer"):
            self._ref_audio_buffer.clear()
        self._aec_mic_buf = np.zeros(0, dtype=np.int16)
        self._aec_ref_buf = np.zeros(0, dtype=np.int16)
        self._aec_out_buf = np.zeros(0, dtype=np.int16)

    def _on_play_audio_chunk(self, chunk: np.ndarray, sr: int):
        if self._aec_processor is not None:
            resampled = resample_chunk(chunk, sr, 16000)
            self._ref_audio_buffer.extend(resampled)

    def _set_state(self, state: str):
        self._state = state
        self.on_state(state)

    def _log(self, msg: str):
        self.on_log(msg)

    def _drain_mic_queue(self):
        """After TTS: wait for echo decay, discard captured frames.
        On barge-in: minimal delay (preserve user speech in queue)."""
        if self._barge_in_flag.is_set():
            time.sleep(0.05)    # minimal — user already speaking, don't drain their speech
            frames = []
            while not self._audio_q.empty():
                try:
                    frames.append(self._audio_q.get_nowait())
                except queue.Empty:
                    break
            keep_n = self._barge_n + 15
            keep_frames = frames[-keep_n:] if len(frames) > keep_n else frames
            for f in keep_frames:
                self._audio_q.put(f)
        else:
            time.sleep(self._echo_decay_ms / 1000.0)
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except Exception:
                    break

    @staticmethod
    def _frames_to_wav(frames: list) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            for f in frames:
                wf.writeframes(np.asarray(f, dtype=np.int16).tobytes())
        return buf.getvalue()

    # ── Demo Mode ─────────────────────────────────────────────────────────────

    def handle_demo_control(self, msg: dict):
        action = msg.get("action")
        if action == "start":
            scenario_id = int(msg.get("scenario_id", 1))
            auto_advance = bool(msg.get("auto_advance", True))
            self.start_demo(scenario_id, auto_advance)
        elif action == "next":
            self.next_demo_turn()
        elif action == "stop":
            self.stop_demo()

    def start_demo(self, scenario_id: int, auto_advance: bool):
        self.stop_demo()
        self._demo_active = True
        self._demo_scenario = scenario_id
        self._demo_turn_idx = 0
        self._demo_auto_advance = auto_advance
        self._demo_stop_event.clear()
        self._demo_turn_done.clear()
        self.tts.demo_audio_dir = "logs/demo_audio"

        from core.demo import SCENARIOS
        if scenario_id not in SCENARIOS:
            self._log(f"❌ Demo scenario {scenario_id} not found.")
            self._demo_active = False
            return
        
        self._log(f"🎬 Starting Demo Scenario {scenario_id}: {SCENARIOS[scenario_id]['name']}")
        self._demo_thread = threading.Thread(target=self._demo_loop, daemon=True)
        self._demo_thread.start()

    def stop_demo(self):
        self._demo_stop_event.set()
        self._demo_turn_done.set() # wake up waiting thread
        if self._demo_thread and self._demo_thread.is_alive():
            self._demo_thread.join(timeout=2)
        self._demo_active = False
        self._demo_scenario = None
        self._demo_turn_idx = 0
        self.tts.demo_audio_dir = None
        self._set_state(State.LISTENING)
        self._broadcast_demo_state()
        self._log("⏹️ Demo stopped.")

    def next_demo_turn(self):
        # In manual mode, wake up the loop to trigger the next turn
        self._demo_turn_done.set()

    def _broadcast_demo_state(self):
        if self.on_demo_state:
            from core.demo import SCENARIOS
            state = {
                "active": self._demo_active,
                "scenario_id": self._demo_scenario,
                "turn_idx": self._demo_turn_idx,
                "auto_advance": self._demo_auto_advance,
            }
            if self._demo_active and self._demo_scenario in SCENARIOS:
                sc = SCENARIOS[self._demo_scenario]
                state["scenario_name"] = sc["name"]
                state["total_turns"] = len(sc["turns"])
                if self._demo_turn_idx < len(sc["turns"]):
                    state["next_turn_text"] = sc["turns"][self._demo_turn_idx]["text"]
                    state["next_turn_lang"] = sc["turns"][self._demo_turn_idx]["lang"]
                else:
                    state["next_turn_text"] = None
            else:
                state["scenario_name"] = None
                state["total_turns"] = 0
                state["next_turn_text"] = None
            self.on_demo_state(state)

    def _demo_loop(self):
        from core.demo import SCENARIOS
        sc = SCENARIOS[self._demo_scenario]
        turns = sc["turns"]

        while self._demo_active and self._demo_turn_idx < len(turns):
            if self._demo_stop_event.is_set():
                break

            self._broadcast_demo_state()
            turn = turns[self._demo_turn_idx]
            raw_text = turn["text"]
            lang = turn["lang"]

            self._log(f"👉 [Turn {self._demo_turn_idx+1}/{len(turns)}] User: \"{raw_text}\" ({lang})")
            
            # Step 1: Transliterate to Native script
            self._log(f"🧠 Transliterating to native script...")
            native_text = self.llm.transliterate(raw_text)
            self._log(f"📝 Native script: \"{native_text}\"")

            # Step 2: Set demo metadata for user audio synthesis
            self.tts.demo_meta = {
                "scenario": self._demo_scenario,
                "turn": self._demo_turn_idx + 1,
                "role": "user",
                "part": 0
            }

            # Step 3: Synthesize User audio
            self._log(f"🔊 Synthesizing user speech...")
            user_wav = self.tts._synthesize(native_text, lang)
            if not user_wav:
                self._log(f"❌ Failed to synthesize user speech.")
                break

            # Play user audio concurrently with VAD feeding
            def play_user_audio():
                try:
                    self.tts._play(user_wav, self._demo_stop_event)
                except Exception as e:
                    self._log(f"⚠️ Error playing user speech: {e}")
            
            play_thread = threading.Thread(target=play_user_audio, daemon=True)
            play_thread.start()

            # Stream user WAV audio into self._audio_q
            import io, soundfile as sf
            import torch, torchaudio
            try:
                data, samplerate = sf.read(io.BytesIO(user_wav), dtype="float32")
                if data.ndim > 1:
                    data = data[:, 0]
                
                if samplerate != 16000:
                    tensor = torch.from_numpy(data).unsqueeze(0).float()
                    resampler = torchaudio.transforms.Resample(samplerate, 16000)
                    data_16k = resampler(tensor).squeeze(0).numpy()
                else:
                    data_16k = data

                data_int16 = (data_16k * 32767.0).astype(np.int16)
                chunk_size = 512
                self._log(f"🎙️ Streaming user audio to VAD pipeline...")
                
                # Drain queue first
                while not self._audio_q.empty():
                    try:
                        self._audio_q.get_nowait()
                    except Exception:
                        break
                
                for idx in range(0, len(data_int16), chunk_size):
                    if self._demo_stop_event.is_set():
                        break
                    chunk = data_int16[idx:idx+chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                    self._audio_q.put(chunk)
                    time.sleep(0.032)

                # Feed silence to trigger VAD end-of-speech
                self._log(f"🤫 Feeding silence to trigger VAD end-of-speech...")
                for _ in range(self._silence_limit + 5):
                    if self._demo_stop_event.is_set():
                        break
                    self._audio_q.put(np.zeros(chunk_size, dtype=np.int16))
                    time.sleep(0.032)

            except Exception as e:
                self._log(f"❌ Error streaming audio: {e}")
                break


            # Wait for turn processing and bot speech to complete
            self._demo_turn_done.clear()
            self._log(f"⏳ Waiting for bot response...")
            self._demo_turn_done.wait()

            if self._demo_stop_event.is_set():
                break

            self._demo_turn_idx += 1
            self._broadcast_demo_state()

            # Advance logic
            if self._demo_turn_idx < len(turns):
                if self._demo_auto_advance:
                    self._log(f"⏱️ Waiting 2.5s before next turn...")
                    for _ in range(25):
                        if self._demo_stop_event.is_set():
                            break
                        time.sleep(0.1)
                else:
                    self._log(f"⏸️ Manual mode: Click 'Next Turn' to proceed.")
                    self._demo_turn_done.clear()
                    self._demo_turn_done.wait()
                    if self._demo_stop_event.is_set():
                        break

        # Finished all turns
        self._demo_active = False
        self._demo_scenario = None
        self._demo_turn_idx = 0
        self.tts.demo_audio_dir = None
        self._set_state(State.LISTENING)
        self._broadcast_demo_state()
        self._log("🏁 Demo scenario finished.")

