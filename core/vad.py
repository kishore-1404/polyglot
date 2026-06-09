"""core/vad.py — Silero VAD v5 wrapper.

CRITICAL: Silero v5 requires exactly 512, 1024, or 1536 samples at 16kHz.
Using 480 samples (30ms) gives sr/samples = 33.33 > 31.25 → crash.
512 samples (32ms) = 31.25 exactly → minimum valid frame.
"""
import torch
import numpy as np


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

        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        self._model.eval()

    @property
    def frame_ms(self) -> float:
        return self.frame_samples / self.sample_rate * 1000.0

    def get_prob(self, frame_int16: np.ndarray) -> float:
        """Return speech probability [0..1] for a single frame."""
        f32 = torch.tensor(frame_int16.astype(np.float32) / 32768.0)
        with torch.no_grad():
            return self._model(f32, self.sample_rate).item()

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        return self.get_prob(frame_int16) >= self.threshold
