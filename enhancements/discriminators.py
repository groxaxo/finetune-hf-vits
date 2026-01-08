"""
Advanced discriminator architectures for improved audio quality.

Based on latest research:
- VNet Multi-Tier Discriminator (2024)
- Wave-U-Net Discriminator (2023)
- HiFi-GAN v3 improvements (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class WaveformDiscriminator(nn.Module):
    """
    Time-domain waveform discriminator.
    Focuses on temporal patterns and phase information.
    """
    
    def __init__(self, channels=1):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, 32, 15, stride=1, padding=7),
            nn.Conv1d(32, 64, 41, stride=4, padding=20, groups=4),
            nn.Conv1d(64, 128, 41, stride=4, padding=20, groups=16),
            nn.Conv1d(128, 256, 41, stride=4, padding=20, groups=64),
            nn.Conv1d(256, 512, 41, stride=4, padding=20, groups=256),
            nn.Conv1d(512, 1024, 5, stride=1, padding=2),
        ])
        
        self.conv_post = nn.Conv1d(1024, 1, 3, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: Waveform tensor [batch, channels, time]
            
        Returns:
            output: Discriminator scores
            feature_maps: Intermediate features for feature matching loss
        """
        feature_maps = []
        
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, feature_maps


class SpectralDiscriminator(nn.Module):
    """
    Frequency-domain spectral discriminator.
    Analyzes mel-spectrogram patterns.
    """
    
    def __init__(self, n_mels=80):
        super().__init__()
        
        # 2D convolutions for spectrogram
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (3, 9), stride=(1, 1), padding=(1, 4)),
            nn.Conv2d(32, 64, (3, 9), stride=(2, 2), padding=(1, 4)),
            nn.Conv2d(64, 128, (3, 9), stride=(2, 2), padding=(1, 4)),
            nn.Conv2d(128, 256, (3, 9), stride=(2, 2), padding=(1, 4)),
            nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
        ])
        
        self.conv_post = nn.Conv2d(512, 1, (3, 3), padding=(1, 1))
        
    def forward(self, x):
        """
        Args:
            x: Waveform [batch, 1, time] - will convert to spectrogram
            
        Returns:
            output: Discriminator scores
            feature_maps: Intermediate features
        """
        # Convert to mel-spectrogram
        x = self._to_mel_spectrogram(x)  # [batch, 1, n_mels, time]
        
        feature_maps = []
        
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, feature_maps
    
    def _to_mel_spectrogram(self, waveform):
        """Convert waveform to mel-spectrogram."""
        # Simplified - in practice, use torchaudio.transforms.MelSpectrogram
        # or integrate with VitsFeatureExtractor
        spec = torch.stft(
            waveform.squeeze(1),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024).to(waveform.device),
            return_complex=True
        )
        
        mag = torch.abs(spec).unsqueeze(1)  # [batch, 1, freq, time]
        return mag


class TemporalDiscriminator(nn.Module):
    """
    Multi-scale temporal discriminator.
    Analyzes patterns at different time scales.
    """
    
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            self._create_scale_discriminator()
            for _ in scales
        ])
        
        self.poolings = nn.ModuleList([
            nn.AvgPool1d(scale, stride=scale) if scale > 1 else nn.Identity()
            for scale in scales
        ])
        
    def _create_scale_discriminator(self):
        """Create a single-scale discriminator."""
        return nn.ModuleList([
            nn.Conv1d(1, 16, 15, stride=1, padding=7),
            nn.Conv1d(16, 64, 41, stride=4, padding=20),
            nn.Conv1d(64, 256, 41, stride=4, padding=20),
            nn.Conv1d(256, 1024, 41, stride=4, padding=20),
            nn.Conv1d(1024, 1024, 5, stride=1, padding=2),
            nn.Conv1d(1024, 1, 3, padding=1),
        ])
    
    def forward(self, x):
        """
        Args:
            x: Waveform [batch, 1, time]
            
        Returns:
            outputs: List of discriminator scores for each scale
            feature_maps: List of feature maps for each scale
        """
        outputs = []
        all_feature_maps = []
        
        for pooling, disc in zip(self.poolings, self.discriminators):
            # Downsample
            x_scale = pooling(x)
            
            # Apply discriminator
            feature_maps = []
            for layer in disc:
                x_scale = layer(x_scale)
                if isinstance(layer, nn.Conv1d):
                    x_scale = F.leaky_relu(x_scale, 0.1)
                feature_maps.append(x_scale)
            
            outputs.append(torch.flatten(x_scale, 1, -1))
            all_feature_maps.append(feature_maps)
        
        return outputs, all_feature_maps


class MultiTierDiscriminator(nn.Module):
    """
    Multi-Tier Discriminator combining multiple analysis approaches.
    
    Based on VNet (arXiv:2408.06906v1).
    
    Combines:
    1. Waveform discriminator (time-domain)
    2. Spectral discriminator (frequency-domain)
    3. Temporal discriminator (multi-scale)
    
    This provides comprehensive audio quality assessment from multiple
    perspectives, leading to better training signal and higher quality output.
    """
    
    def __init__(
        self,
        use_waveform=True,
        use_spectral=True,
        use_temporal=True,
        tier_weights=None
    ):
        super().__init__()
        
        self.tiers = nn.ModuleList()
        self.tier_names = []
        
        if use_waveform:
            self.tiers.append(WaveformDiscriminator())
            self.tier_names.append("waveform")
        
        if use_spectral:
            self.tiers.append(SpectralDiscriminator())
            self.tier_names.append("spectral")
        
        if use_temporal:
            self.tiers.append(TemporalDiscriminator())
            self.tier_names.append("temporal")
        
        # Default equal weights
        if tier_weights is None:
            tier_weights = [1.0] * len(self.tiers)
        
        self.tier_weights = nn.Parameter(
            torch.tensor(tier_weights, dtype=torch.float32),
            requires_grad=False
        )
        
    def forward(self, x):
        """
        Args:
            x: Waveform tensor [batch, 1, time]
            
        Returns:
            outputs: List of discriminator scores from each tier
            feature_maps: List of feature maps from each tier
        """
        outputs = []
        all_feature_maps = []
        
        for tier in self.tiers:
            out, fmaps = tier(x)
            outputs.append(out)
            all_feature_maps.append(fmaps)
        
        return outputs, all_feature_maps
    
    def compute_loss(self, real_outputs, fake_outputs):
        """
        Compute discriminator loss across all tiers.
        
        Args:
            real_outputs: List of outputs for real audio
            fake_outputs: List of outputs for generated audio
            
        Returns:
            loss: Weighted sum of tier losses
            loss_dict: Individual tier losses for logging
        """
        total_loss = 0
        loss_dict = {}
        
        for i, (real, fake, weight, name) in enumerate(
            zip(real_outputs, fake_outputs, self.tier_weights, self.tier_names)
        ):
            # Standard GAN loss
            if isinstance(real, list):  # Multi-scale
                tier_loss = 0
                for r, f in zip(real, fake):
                    tier_loss += (
                        torch.mean((r - 1) ** 2) +  # Real should be close to 1
                        torch.mean(f ** 2)            # Fake should be close to 0
                    )
                tier_loss /= len(real)
            else:
                tier_loss = (
                    torch.mean((real - 1) ** 2) +
                    torch.mean(fake ** 2)
                )
            
            weighted_loss = weight * tier_loss
            total_loss += weighted_loss
            
            loss_dict[f"disc_loss_{name}"] = tier_loss.item()
        
        loss_dict["disc_loss_total"] = total_loss.item()
        
        return total_loss, loss_dict


class WaveUNetDiscriminator(nn.Module):
    """
    Wave-U-Net style discriminator.
    
    Lightweight alternative to ensemble discriminators.
    Based on arXiv:2303.13909.
    
    Uses skip connections and multi-level feature extraction
    for expressive discrimination with lower computational cost.
    """
    
    def __init__(self, channels=1, base_channels=32, depth=4):
        super().__init__()
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        in_ch = channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 15, padding=7),
                nn.LeakyReLU(0.1),
                nn.Conv1d(out_ch, out_ch, 15, padding=7),
                nn.LeakyReLU(0.1),
            ))
            
            self.downsamplers.append(
                nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)
            )
            
            in_ch = out_ch
        
        # Decoder (upsampling path)
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i in range(depth - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            
            self.upsamplers.append(
                nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
            )
            
            # Skip connection doubles channels
            self.decoders.append(nn.Sequential(
                nn.Conv1d(out_ch * 2, out_ch, 15, padding=7),
                nn.LeakyReLU(0.1),
                nn.Conv1d(out_ch, out_ch, 15, padding=7),
                nn.LeakyReLU(0.1),
            ))
        
        # Output layer
        self.conv_post = nn.Conv1d(base_channels, 1, 3, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: Waveform [batch, 1, time]
            
        Returns:
            output: Discriminator scores
            feature_maps: Multi-level features
        """
        # Encoder
        skip_connections = []
        feature_maps = []
        
        for encoder, downsampler in zip(self.encoders, self.downsamplers):
            x = encoder(x)
            skip_connections.append(x)
            feature_maps.append(x)
            x = downsampler(x)
        
        # Decoder with skip connections
        skip_connections = skip_connections[:-1]  # Remove last (no skip needed)
        
        for upsampler, decoder, skip in zip(
            self.upsamplers, self.decoders, reversed(skip_connections)
        ):
            x = upsampler(x)
            
            # Match dimensions if needed
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            feature_maps.append(x)
        
        # Output
        x = self.conv_post(x)
        feature_maps.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, feature_maps


# Utility function for integration
def create_discriminator(discriminator_type="multi_tier", **kwargs):
    """
    Factory function to create discriminators.
    
    Args:
        discriminator_type: One of "multi_tier", "wave_unet", "standard"
        **kwargs: Additional arguments for discriminator
        
    Returns:
        Discriminator module
    """
    if discriminator_type == "multi_tier":
        return MultiTierDiscriminator(**kwargs)
    elif discriminator_type == "wave_unet":
        return WaveUNetDiscriminator(**kwargs)
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")
