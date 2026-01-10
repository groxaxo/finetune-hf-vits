"""
Enhancement modules for improved VITS/MMS output quality.

This package contains advanced features for:
- Multi-tier discriminators
- Prosody control
- Data augmentation
- Post-processing
- Quality evaluation
"""

__version__ = "1.0.0"
VITS Enhancement Module

This module provides optional architectural enhancements for VITS training:
- Multi-Tier Discriminator (MTD) for multi-resolution audio analysis
- Wave-U-Net Discriminator for lightweight, fast discrimination
- DSP-based postprocessing for output enhancement
"""

from .discriminators import MultiTierDiscriminator, WaveUNetDiscriminator, create_discriminator
from .postprocessing import enhance_tts_output

__all__ = [
    "MultiTierDiscriminator",
    "WaveUNetDiscriminator", 
    "create_discriminator",
    "enhance_tts_output",
]
