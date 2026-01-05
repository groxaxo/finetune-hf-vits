"""
Vocos decoder for Soprano TTS.

This module defines the main decoder that converts hidden states
to spectral output.
"""

import torch
import torch.nn as nn
from typing import Optional
from .heads import ISTFTHead


class VocosDecoder(nn.Module):
    """Vocos-based decoder for Soprano TTS.
    
    Converts hidden states from language model to audio via
    spectral representation and ISTFT.
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_layers: int = 4,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
    ):
        """Initialize Vocos decoder.
        
        Args:
            hidden_size: Size of input hidden states
            intermediate_size: Size of intermediate layers
            num_layers: Number of decoder layers
            n_fft: FFT size for ISTFT
            hop_length: Hop length for ISTFT
            win_length: Window length for ISTFT
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        
        # Build decoder layers
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    hidden_size if i == 0 else intermediate_size,
                    intermediate_size,
                    kernel_size=7,
                    padding=3,
                )
            )
            layers.append(nn.ReLU())
        
        self.layers = nn.ModuleList(layers)
        
        # Final projection to hidden_size for ISTFTHead
        self.final_proj = nn.Conv1d(intermediate_size, hidden_size, kernel_size=1)
        
        # ISTFT head
        self.head = ISTFTHead(
            hidden_size=hidden_size,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full forward pass with ISTFT.
        
        Args:
            hidden_states: Input hidden states, shape [B, T, H] or [B, H, T]
        
        Returns:
            Audio waveform, shape [B, samples]
        """
        spectral = self.forward_spectral(hidden_states)
        audio = self.head._istft(spectral)
        return audio
    
    def forward_spectral(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass producing only spectral output (for ONNX export).
        
        Args:
            hidden_states: Input hidden states, shape [B, T, H] or [B, H, T]
        
        Returns:
            Spectral tensor, shape [B, F, T, 2]
        """
        # Ensure [B, H, T] format
        if hidden_states.dim() == 3:
            if hidden_states.shape[1] != self.hidden_size:
                # Assume [B, T, H], transpose to [B, H, T]
                hidden_states = hidden_states.transpose(1, 2)
        
        # Pass through decoder layers
        x = hidden_states
        for layer in self.layers:
            x = layer(x)
        
        # Final projection
        x = self.final_proj(x)
        
        # Generate spectral output
        spectral = self.head.forward_spectral(x)
        
        return spectral
    
    def get_istft_config(self):
        """Get ISTFT configuration."""
        return self.head.get_istft_config()
