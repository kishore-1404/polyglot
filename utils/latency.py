"""utils/latency.py — Per-stage latency tracker for one pipeline turn."""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TurnLatency:
    """All timing stamps for a single utterance."""
    t_end_of_speech: float = 0.0   # VAD detected silence end
    t_llm_send:      float = 0.0   # audio sent to llama-server
    t_first_token:   float = 0.0   # first streamed token received
    t_llm_done:      float = 0.0   # last token received (JSON complete)
    t_tts_send:      float = 0.0   # text sent to supertonic
    t_audio_start:   float = 0.0   # first audio byte played

    @property
    def llm_ttft_ms(self) -> float:
        """Time from audio send → first token (model "thinking" lag)."""
        if self.t_first_token and self.t_llm_send:
            return (self.t_first_token - self.t_llm_send) * 1000
        return 0.0

    @property
    def llm_total_ms(self) -> float:
        """Full LLM generation time."""
        if self.t_llm_done and self.t_llm_send:
            return (self.t_llm_done - self.t_llm_send) * 1000
        return 0.0

    @property
    def tts_ms(self) -> float:
        """TTS synthesis + buffering latency."""
        if self.t_audio_start and self.t_tts_send:
            return (self.t_audio_start - self.t_tts_send) * 1000
        return 0.0

    @property
    def total_ms(self) -> float:
        """Processing latency: audio send → audio playback start."""
        if self.t_audio_start and self.t_llm_send:
            return (self.t_audio_start - self.t_llm_send) * 1000
        return 0.0

    @property
    def target_status(self) -> str:
        """Return 'stretch', 'pass', or 'slow' vs the case targets."""
        t = self.total_ms
        if t <= 0:    return "pending"
        if t < 800:   return "stretch"   # <800ms stretch goal
        if t < 1200:  return "pass"      # <1200ms main target
        return "slow"

    def to_dict(self) -> dict:
        return {
            "llm_ttft_ms":  round(self.llm_ttft_ms,  1) if self.t_first_token else None,
            "llm_total_ms": round(self.llm_total_ms, 1) if self.t_llm_done else None,
            "tts_ms":       round(self.tts_ms,        1) if (self.t_audio_start and self.t_tts_send) else None,
            "total_ms":     round(self.total_ms,      1) if self.t_audio_start else None,
            "status":       self.target_status,
        }
