import numpy as np
import librosa
import soundfile as sf
from typing import Literal

QualityPreset = Literal["balanced", "max_clean", "fast"]

def _normalize_loudness(wav: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(wav))
    return wav / (peak + 1e-9)

def _high_pass_filter(wav: np.ndarray, sr: int, cutoff: float = 30.0) -> np.ndarray:
    # Remove DC and low-end rumble
    # Using simple pre-emphasis as a safe, dependency-light proxy for the filter
    return librosa.effects.preemphasis(wav, coef=0.97)

def _trim_silence(wav: np.ndarray) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(wav, top_db=60)
    return trimmed

def enhance_tts_output(audio: np.ndarray, sample_rate: int, quality_preset: QualityPreset = "balanced") -> np.ndarray:
    """Enhance a VITS/MMS waveform with configurable DSP steps."""
    audio = audio.astype(np.float32)
    
    # 1. Normalize
    audio = _normalize_loudness(audio)
    
    # 2. High-pass (Rumble removal)
    audio = _high_pass_filter(audio, sample_rate)
    
    # 3. Silence Trimming
    if quality_preset != "fast":
        audio = _trim_silence(audio)
        
    # 4. Final Normalize
    audio = _normalize_loudness(audio)
    
    return audio
