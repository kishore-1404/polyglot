"""core/vad.py — Silero VAD v5 wrapper using ONNX Runtime.

CRITICAL: Silero v5 requires exactly 512, 1024, or 1536 samples at 16kHz.
512 samples (32ms) = 31.25 Hz exactly → minimum valid frame.
"""
import os
import urllib.request
import numpy as np
import onnxruntime as ort


class VADProcessor:
    VALID_FRAME_SIZES = {512, 1024, 1536}

    def __init__(self, cfg: dict):
        frame_sz = int(cfg["vad"]["frame_samples"])
        if frame_sz not in self.VALID_FRAME_SIZES:
            raise ValueError(
                f"vad.frame_samples must be one of {self.VALID_FRAME_SIZES}, "
                f"got {frame_sz}. 480 (30ms) is NOT supported by Silero v5."
            )
        self.frame_samples = frame_sz
        self.threshold     = float(cfg["vad"]["threshold"])
        self.sample_rate   = 16000

        # Download Silero VAD ONNX model to cache if not present
        cache_dir = os.path.expanduser("~/.cache/polyglot")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "silero_vad.onnx")

        if not os.path.exists(model_path):
            print("📥 Downloading Silero VAD ONNX model (~3MB)...")
            url = "https://models.silero.ai/vad_models/silero_vad.onnx"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("✅ VAD Model download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download VAD model: {e}")

        # Initialize ONNX inference session on CPU execution provider
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        # Initialize RNN state (2 layers, batch size 1, 64 dimension)
        self._state = np.zeros((2, 1, 64), dtype=np.float32)

    @property
    def frame_ms(self) -> float:
        return self.frame_samples / self.sample_rate * 1000.0

    def reset(self):
        """Reset the VAD RNN state."""
        self._state = np.zeros((2, 1, 64), dtype=np.float32)

    def get_prob(self, frame_int16: np.ndarray) -> float:
        """Return speech probability [0..1] for a single frame."""
        # Convert int16 to float32 normalized in [-1.0, 1.0]
        f32 = frame_int16.astype(np.float32) / 32768.0
        f32_input = np.expand_dims(f32, axis=0)  # Shape (1, 512)
        sr_input = np.array([self.sample_rate], dtype=np.int64)

        # Run ONNX inference
        inputs = {
            "input": f32_input,
            "state": self._state,
            "sr": sr_input
        }
        
        ort_outs = self._session.run(None, inputs)
        out_prob = ort_outs[0][0][0]
        self._state = ort_outs[1]  # Keep updated RNN state for the next frame
        
        return float(out_prob)

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        return self.get_prob(frame_int16) >= self.threshold
