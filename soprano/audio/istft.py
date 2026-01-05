"""
ISTFT postprocessing for Soprano TTS decoder output.

This module implements the Inverse Short-Time Fourier Transform (ISTFT)
as a CPU postprocess step after ONNX/OpenVINO decoder inference.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional


class ISTFTConfig:
    """Configuration for ISTFT parameters.
    
    These parameters must match the original ISTFTHead configuration
    from the Soprano model.
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        length: Optional[int] = None,
    ):
        """Initialize ISTFT configuration.
        
        Args:
            n_fft: FFT size
            hop_length: Number of samples between successive frames
            win_length: Window length, defaults to n_fft if None
            window: Window function name ('hann', 'hamming', etc.)
            center: Whether input was centered during STFT
            normalized: Whether to normalize by 1/sqrt(n_fft)
            onesided: Whether input is one-sided FFT (only positive frequencies)
            length: Optional output length to crop/pad to
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.length = length
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ISTFTConfig":
        """Create config from dictionary."""
        # Explicitly list valid parameters instead of using co_varnames
        valid_params = {
            'n_fft', 'hop_length', 'win_length', 'window',
            'center', 'normalized', 'onesided', 'length'
        }
        return cls(**{k: v for k, v in config_dict.items() if k in valid_params})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
            "length": self.length,
        }


def istft_postprocess(
    spectral_tensor: np.ndarray,
    config: ISTFTConfig,
    use_torch: bool = True,
) -> np.ndarray:
    """Convert spectral tensor to audio waveform using ISTFT.
    
    This function implements ISTFT postprocessing on CPU to convert
    the spectral output from the decoder ONNX model into audio.
    
    Args:
        spectral_tensor: Complex spectral tensor from decoder.
            Expected shapes:
            - [B, F, T, 2] where last dim is [real, imag]
            - [B, 2, F, T] where dim=1 is [real, imag]
            F = number of frequency bins
            T = number of time frames
            B = batch size
        config: ISTFTConfig with ISTFT parameters
        use_torch: Whether to use PyTorch (more accurate) or NumPy
        
    Returns:
        Audio waveform as float32 numpy array with shape [B, samples]
        Values are roughly in range [-1, 1]
    """
    # Determine input format and convert to [B, F, T, 2] if needed
    if spectral_tensor.ndim != 4:
        raise ValueError(
            f"Expected 4D spectral tensor, got shape {spectral_tensor.shape}"
        )
    
    # Check if format is [B, 2, F, T] and convert to [B, F, T, 2]
    if spectral_tensor.shape[1] == 2:
        # Transpose from [B, 2, F, T] to [B, F, T, 2]
        spectral_tensor = np.transpose(spectral_tensor, (0, 2, 3, 1))
    
    batch_size, n_freq, n_frames, n_complex = spectral_tensor.shape
    
    if n_complex != 2:
        raise ValueError(
            f"Expected last dimension to be 2 (real/imag), got {n_complex}"
        )
    
    if use_torch:
        return _istft_torch(spectral_tensor, config)
    else:
        return _istft_numpy(spectral_tensor, config)


def _istft_torch(spectral_tensor: np.ndarray, config: ISTFTConfig) -> np.ndarray:
    """Perform ISTFT using PyTorch (CPU).
    
    This is the reference implementation that matches the original model exactly.
    """
    # Convert to torch tensor
    spectral_torch = torch.from_numpy(spectral_tensor)
    
    # Combine real and imaginary parts into complex tensor
    # spectral_torch shape: [B, F, T, 2]
    complex_spec = torch.complex(spectral_torch[..., 0], spectral_torch[..., 1])
    # complex_spec shape: [B, F, T]
    
    # Get window function
    if config.window == "hann":
        window = torch.hann_window(config.win_length)
    elif config.window == "hamming":
        window = torch.hamming_window(config.win_length)
    elif config.window == "blackman":
        window = torch.blackman_window(config.win_length)
    else:
        window = torch.ones(config.win_length)
    
    # Perform ISTFT for each batch element
    batch_size = complex_spec.shape[0]
    waveforms = []
    
    for i in range(batch_size):
        waveform = torch.istft(
            complex_spec[i],
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=window,
            center=config.center,
            normalized=config.normalized,
            onesided=config.onesided,
            length=config.length,
            return_complex=False,
        )
        waveforms.append(waveform)
    
    # Stack and convert to numpy
    waveforms = torch.stack(waveforms, dim=0)
    audio = waveforms.numpy().astype(np.float32)
    
    # Ensure values are in reasonable range (clamp if needed)
    # Note: Only clamp if original pipeline clamps
    # For now, we don't clamp to preserve exact values
    
    return audio


def _istft_numpy(spectral_tensor: np.ndarray, config: ISTFTConfig) -> np.ndarray:
    """Perform ISTFT using NumPy.
    
    This is a fallback implementation for environments without PyTorch.
    Note: May have slight numerical differences from PyTorch version.
    """
    # This is a simplified implementation
    # For production, consider using scipy.signal.istft or similar
    try:
        from scipy.signal import istft as scipy_istft
    except ImportError:
        raise ImportError(
            "NumPy-based ISTFT requires scipy. "
            "Install scipy or use use_torch=True (recommended)"
        )
    
    batch_size, n_freq, n_frames, _ = spectral_tensor.shape
    
    # Combine real and imaginary parts
    complex_spec = spectral_tensor[..., 0] + 1j * spectral_tensor[..., 1]
    
    # Get window
    if config.window == "hann":
        from scipy.signal import get_window
        window = get_window("hann", config.win_length)
    else:
        window = config.window
    
    waveforms = []
    for i in range(batch_size):
        # scipy istft expects [freq, time]
        _, waveform = scipy_istft(
            complex_spec[i],
            fs=22050,  # This should be configurable
            window=window,
            nperseg=config.win_length,
            noverlap=config.win_length - config.hop_length,
            nfft=config.n_fft,
            input_onesided=config.onesided,
            boundary=config.center,
        )
        waveforms.append(waveform)
    
    audio = np.stack(waveforms, axis=0).astype(np.float32)
    
    # Apply length constraint if specified
    if config.length is not None:
        if audio.shape[1] > config.length:
            audio = audio[:, :config.length]
        elif audio.shape[1] < config.length:
            # Pad with zeros
            padding = np.zeros((batch_size, config.length - audio.shape[1]), dtype=np.float32)
            audio = np.concatenate([audio, padding], axis=1)
    
    return audio


def create_default_istft_config() -> ISTFTConfig:
    """Create default ISTFT config for Soprano model.
    
    These defaults should match the Soprano-80M model configuration.
    Adjust based on actual model parameters.
    """
    return ISTFTConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window="hann",
        center=True,
        normalized=False,
        onesided=True,
        length=None,
    )
