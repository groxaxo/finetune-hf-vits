import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Tuple

class DiscriminatorSTFT(nn.Module):
    """Sub-module for Multi-Tier Discriminator: Processes specific STFT resolutions."""
    def __init__(self, filters=32, n_fft=1024, hop_length=120, win_length=600):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, filters, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(filters, filters, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(filters, filters, (3, 7), stride=(1, 2), padding=(1, 3))),
            weight_norm(nn.Conv2d(filters, filters, (3, 7), stride=(1, 2), padding=(1, 3))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(filters, 1, (3, 3), (1, 1), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        # Compute STFT on the fly
        x_spec = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length, 
                            return_complex=True, normalized=True)
        x_spec = torch.abs(x_spec).unsqueeze(1)
        
        for l in self.convs:
            x_spec = l(x_spec)
            x_spec = F.leaky_relu(x_spec, 0.1)
            fmap.append(x_spec)
        
        x_spec = self.conv_post(x_spec)
        fmap.append(x_spec)
        return torch.flatten(x_spec, 1, -1), fmap

class MultiTierDiscriminator(nn.Module):
    """
    MTD: Analyzes audio at 3 different resolutions (High, Mid, Low).
    See VNet paper for details.
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(n_fft=2048, hop_length=512, win_length=2048), # Tier 1
            DiscriminatorSTFT(n_fft=1024, hop_length=256, win_length=1024), # Tier 2
            DiscriminatorSTFT(n_fft=512, hop_length=128, win_length=512)    # Tier 3
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class WaveUNetDiscriminator(nn.Module):
    """Lightweight, fast discriminator."""
    def __init__(self, base_channels=64, depth=4):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = base_channels
        
        # Downsampling
        for i in range(depth):
            self.downs.append(nn.Sequential(
                nn.Conv1d(1 if i==0 else ch//2, ch, 15, 4, 7),
                nn.LeakyReLU(0.1, inplace=True)
            ))
            ch *= 2
            
        # Upsampling
        for i in range(depth):
            ch //= 2
            self.ups.append(nn.Sequential(
                nn.ConvTranspose1d(ch*2, ch, 4, 4, 0),
                nn.LeakyReLU(0.1, inplace=True)
            ))
            
        self.final = nn.Conv1d(base_channels, 1, 3, 1, 1)

    def forward(self, x):
        fmap = []
        h = x
        downs_out = []
        for down in self.downs:
            h = down(h)
            downs_out.append(h)
        for i, up in enumerate(self.ups):
            h = up(h)
            if i < len(downs_out): h = h + downs_out[-(i+1)] # Skip connection
            fmap.append(h)
        return self.final(h), fmap

def create_discriminator(discriminator_type="default"):
    if discriminator_type == "multi_tier": return MultiTierDiscriminator()
    if discriminator_type == "wave_unet": return WaveUNetDiscriminator()
    return None # Defaults to original VITS discriminator in training script
