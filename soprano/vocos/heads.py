"""
Vocos decoder heads for Soprano TTS.

This module defines the decoder head that produces spectral output
before ISTFT transformation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


class ISTFTHead(nn.Module):
    """ISTFT Head for Vocos decoder.
    
    This head produces spectral output (magnitude or complex spectrogram)
    that can be converted to audio via ISTFT.
    
    For ONNX export, we split this into:
    1. SpectralHead: Produces spectral tensor (exported to ONNX)
    2. ISTFT postprocess: Converts spectral to audio (done in Python/CPU)
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        """Initialize ISTFT head.
        
        Args:
            hidden_size: Size of hidden features from decoder
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length, defaults to n_fft
            window: Window function name
            center: Whether to center frames
            normalized: Whether to normalize ISTFT
            onesided: Whether to use one-sided FFT
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        
        # Calculate output frequency bins
        if onesided:
            self.n_freq_bins = n_fft // 2 + 1
        else:
            self.n_freq_bins = n_fft
        
        # Projection layer: hidden_size -> spectral features
        # Output is complex (real + imaginary), so we need 2 * n_freq_bins
        self.out_proj = nn.Conv1d(
            hidden_size,
            self.n_freq_bins * 2,  # 2 for real and imaginary parts
            kernel_size=1,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full forward pass with ISTFT (for PyTorch baseline).
        
        Args:
            hidden_states: Hidden features from decoder, shape [B, H, T]
                          where H = hidden_size, T = number of frames
        
        Returns:
            Audio waveform, shape [B, samples]
        """
        spectral = self.forward_spectral(hidden_states)
        audio = self._istft(spectral)
        return audio
    
    def forward_spectral(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass producing only spectral output (for ONNX export).
        
        Args:
            hidden_states: Hidden features from decoder, shape [B, H, T]
        
        Returns:
            Spectral tensor, shape [B, F, T, 2]
            where F = n_freq_bins, last dim is [real, imag]
        """
        # Project to spectral features
        # hidden_states: [B, H, T]
        spectral_flat = self.out_proj(hidden_states)  # [B, F*2, T]
        
        batch_size, _, n_frames = spectral_flat.shape
        
        # Reshape to [B, F, T, 2] where last dim is [real, imag]
        spectral = spectral_flat.view(batch_size, self.n_freq_bins, 2, n_frames)
        spectral = spectral.permute(0, 1, 3, 2)  # [B, F, T, 2]
        
        return spectral
    
    def _istft(self, spectral: torch.Tensor) -> torch.Tensor:
        """Internal ISTFT implementation (for PyTorch baseline).
        
        Args:
            spectral: Spectral tensor, shape [B, F, T, 2]
        
        Returns:
            Audio waveform, shape [B, samples]
        """
        batch_size = spectral.shape[0]
        
        # Convert to complex
        complex_spec = torch.complex(spectral[..., 0], spectral[..., 1])
        # complex_spec: [B, F, T]
        
        # Get window
        if self.window == "hann":
            window = torch.hann_window(self.win_length, device=spectral.device)
        elif self.window == "hamming":
            window = torch.hamming_window(self.win_length, device=spectral.device)
        else:
            window = torch.ones(self.win_length, device=spectral.device)
        
        # Apply ISTFT to each batch element
        waveforms = []
        for i in range(batch_size):
            waveform = torch.istft(
                complex_spec[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                normalized=self.normalized,
                onesided=self.onesided,
                return_complex=False,
            )
            waveforms.append(waveform)
        
        return torch.stack(waveforms, dim=0)
    
    def get_istft_config(self) -> Dict[str, Any]:
        """Get ISTFT configuration for export.
        
        Returns:
            Dictionary with ISTFT parameters
        """
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
        }


class SpectralHead(nn.Module):
    """Standalone spectral head for ONNX export.
    
    This is a simplified version of ISTFTHead that only produces
    spectral output without ISTFT.
    """
    
    def __init__(self, istft_head: ISTFTHead):
        """Initialize from existing ISTFTHead.
        
        Args:
            istft_head: ISTFTHead instance to extract parameters from
        """
        super().__init__()
        self.out_proj = istft_head.out_proj
        self.n_freq_bins = istft_head.n_freq_bins
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass producing spectral output.
        
        Args:
            hidden_states: Hidden features, shape [B, H, T]
        
        Returns:
            Spectral tensor, shape [B, F, T, 2]
        """
        spectral_flat = self.out_proj(hidden_states)
        batch_size, _, n_frames = spectral_flat.shape
        spectral = spectral_flat.view(batch_size, self.n_freq_bins, 2, n_frames)
        spectral = spectral.permute(0, 1, 3, 2)
        return spectral
