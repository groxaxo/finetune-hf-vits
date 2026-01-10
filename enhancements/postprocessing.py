"""
Audio post-processing for enhanced output quality.

Includes:
- Loudness normalization
- DC offset removal
- High-pass filtering
- De-essing
- Noise reduction
"""

import numpy as np
import scipy.signal
from typing import Optional


class AudioPostProcessor:
    """
    Post-processing pipeline for TTS output quality enhancement.
    
    Can be applied after synthesis to improve the final audio quality.
    """
    
    def __init__(self, sampling_rate=22050):
        """
        Initialize post-processor.
        
        Args:
            sampling_rate: Audio sampling rate in Hz
        """
        self.sr = sampling_rate
        
    def normalize_loudness(self, audio: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target loudness (LUFS/dB).
        
        Args:
            audio: Input audio array
            target_lufs: Target loudness in LUFS (typically -20 to -14)
            
        Returns:
            Loudness-normalized audio
        """
        try:
            import pyloudnorm as pyln
            
            # Measure loudness
            meter = pyln.Meter(self.sr)
            loudness = meter.integrated_loudness(audio)
            
            # Normalize
            normalized = pyln.normalize.loudness(
                audio, loudness, target_lufs
            )
            
            return normalized
            
        except ImportError:
            # Fallback to peak normalization if pyloudnorm not available
            return self._normalize_peak(audio, target_peak=0.9)
    
    def _normalize_peak(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """
        Simple peak normalization fallback.
        
        Args:
            audio: Input audio
            target_peak: Target peak amplitude (0-1)
            
        Returns:
            Peak-normalized audio
        """
        peak = np.abs(audio).max()
        if peak > 1e-8:  # Avoid division by zero
            return audio * (target_peak / peak)
        return audio
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset by centering audio around zero.
        
        Args:
            audio: Input audio
            
        Returns:
            DC-corrected audio
        """
        return audio - np.mean(audio)
    
    def apply_highpass_filter(
        self, 
        audio: np.ndarray, 
        cutoff: float = 50.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency rumble.
        
        Args:
            audio: Input audio
            cutoff: Cutoff frequency in Hz
            order: Filter order (higher = steeper)
            
        Returns:
            Filtered audio
        """
        # Design Butterworth high-pass filter
        sos = scipy.signal.butter(
            order, cutoff, 'hp', 
            fs=self.sr, output='sos'
        )
        
        # Apply filter
        filtered = scipy.signal.sosfilt(sos, audio)
        
        return filtered
    
    def apply_lowpass_filter(
        self,
        audio: np.ndarray,
        cutoff: float = 8000.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply low-pass filter to remove high-frequency noise.
        
        Args:
            audio: Input audio
            cutoff: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered audio
        """
        sos = scipy.signal.butter(
            order, cutoff, 'lp',
            fs=self.sr, output='sos'
        )
        
        filtered = scipy.signal.sosfilt(sos, audio)
        
        return filtered
    
    def apply_deesser(
        self,
        audio: np.ndarray,
        freq_range: tuple = (4000, 8000),
        threshold_db: float = -20.0,
        ratio: float = 4.0
    ) -> np.ndarray:
        """
        Reduce harsh sibilance (s/sh sounds).
        
        This uses multiband compression to reduce excessive high-frequency energy.
        
        Args:
            audio: Input audio
            freq_range: Frequency range to de-ess (Hz)
            threshold_db: Compression threshold
            ratio: Compression ratio
            
        Returns:
            De-essed audio
        """
        # Extract sibilant frequency band
        sos_band = scipy.signal.butter(
            4, freq_range, 'bp',
            fs=self.sr, output='sos'
        )
        sibilant_band = scipy.signal.sosfilt(sos_band, audio)
        
        # Detect loud sibilants
        threshold_linear = 10 ** (threshold_db / 20)
        envelope = np.abs(sibilant_band)
        
        # Apply compression to loud sibilants
        gain = np.ones_like(audio)
        mask = envelope > threshold_linear
        
        if mask.any():
            # Reduce gain in sibilant regions
            excess = envelope[mask] / threshold_linear
            gain[mask] = 1.0 / (1.0 + (excess - 1.0) / ratio)
        
        # Smooth gain changes
        gain = scipy.signal.savgol_filter(gain, window_length=51, polyorder=3)
        
        # Apply gain
        processed = audio * gain
        
        return processed
    
    def trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim leading and trailing silence.
        
        Args:
            audio: Input audio
            threshold_db: Silence threshold in dB
            frame_length: Frame size for energy calculation
            hop_length: Hop size between frames
            
        Returns:
            Trimmed audio
        """
        # Calculate frame energies
        frames = np.array([
            audio[i:i+frame_length]
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        
        energies = np.sum(frames ** 2, axis=1)
        energies_db = 10 * np.log10(energies + 1e-10)
        
        # Find non-silent frames
        threshold = threshold_db
        non_silent = energies_db > threshold
        
        if not non_silent.any():
            return audio  # All silent, don't trim
        
        # Find first and last non-silent frames
        first_frame = np.argmax(non_silent)
        last_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        # Convert to sample indices
        start_idx = first_frame * hop_length
        end_idx = min((last_frame + 1) * hop_length + frame_length, len(audio))
        
        return audio[start_idx:end_idx]
    
    def process(
        self,
        audio: np.ndarray,
        normalize: bool = True,
        remove_dc: bool = True,
        highpass: bool = True,
        lowpass: bool = False,
        deess: bool = False,
        trim: bool = False,
        target_lufs: float = -20.0
    ) -> np.ndarray:
        """
        Apply full post-processing pipeline.
        
        Args:
            audio: Input audio array
            normalize: Apply loudness normalization
            remove_dc: Remove DC offset
            highpass: Apply high-pass filter
            lowpass: Apply low-pass filter
            deess: Apply de-esser
            trim: Trim silence
            target_lufs: Target loudness for normalization
            
        Returns:
            Processed audio
        """
        processed = audio.copy()
        
        # 1. Remove DC offset first
        if remove_dc:
            processed = self.remove_dc_offset(processed)
        
        # 2. Apply filters
        if highpass:
            processed = self.apply_highpass_filter(processed, cutoff=50.0)
        
        if lowpass:
            processed = self.apply_lowpass_filter(processed, cutoff=8000.0)
        
        # 3. De-ess if needed
        if deess:
            processed = self.apply_deesser(processed)
        
        # 4. Trim silence
        if trim:
            processed = self.trim_silence(processed)
        
        # 5. Normalize last (after all processing)
        if normalize:
            processed = self.normalize_loudness(processed, target_lufs=target_lufs)
        
        return processed


def enhance_tts_output(
    audio: np.ndarray,
    sampling_rate: int = 22050,
    quality_preset: str = "balanced"
) -> np.ndarray:
    """
    Convenience function to enhance TTS output with preset configurations.
    
    Args:
        audio: Input audio from TTS synthesis
        sampling_rate: Audio sampling rate
        quality_preset: One of "minimal", "balanced", "maximum"
        
    Returns:
        Enhanced audio
        
    Presets:
        - minimal: Just normalization and DC removal (fast)
        - balanced: Standard enhancement (recommended)
        - maximum: All enhancements (slower, highest quality)
    """
    processor = AudioPostProcessor(sampling_rate)
    
    presets = {
        "minimal": {
            "normalize": True,
            "remove_dc": True,
            "highpass": False,
            "lowpass": False,
            "deess": False,
            "trim": False,
        },
        "balanced": {
            "normalize": True,
            "remove_dc": True,
            "highpass": True,
            "lowpass": False,
            "deess": False,
            "trim": True,
        },
        "maximum": {
            "normalize": True,
            "remove_dc": True,
            "highpass": True,
            "lowpass": True,
            "deess": True,
            "trim": True,
        },
    }
    
    if quality_preset not in presets:
        raise ValueError(f"Unknown preset: {quality_preset}. Choose from {list(presets.keys())}")
    
    return processor.process(audio, **presets[quality_preset])
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
