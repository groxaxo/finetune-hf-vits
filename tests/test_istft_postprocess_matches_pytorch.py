"""
Test ISTFT postprocessing matches PyTorch baseline.

This test verifies that the ISTFT postprocessing function
produces the same audio output as the PyTorch ISTFTHead.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from soprano.vocos.heads import ISTFTHead
from soprano.audio.istft import istft_postprocess, ISTFTConfig


def test_istft_postprocess_matches_pytorch():
    """Test that ISTFT postprocess produces same audio as PyTorch."""
    
    # Create ISTFT head
    hidden_size = 512
    n_fft = 1024
    hop_length = 256
    
    istft_head = ISTFTHead(
        hidden_size=hidden_size,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=1024,
        window="hann",
    )
    istft_head.eval()
    
    # Create test hidden states
    torch.manual_seed(42)
    batch_size = 2
    seq_length = 100
    hidden_states = torch.randn(batch_size, hidden_size, seq_length)
    
    # Get spectral output from PyTorch
    with torch.no_grad():
        spectral_pytorch = istft_head.forward_spectral(hidden_states)
        audio_pytorch = istft_head._istft(spectral_pytorch)
    
    # Convert spectral to numpy
    spectral_np = spectral_pytorch.numpy()
    audio_pytorch_np = audio_pytorch.numpy()
    
    # Create ISTFT config from head
    istft_config = ISTFTConfig(
        n_fft=istft_head.n_fft,
        hop_length=istft_head.hop_length,
        win_length=istft_head.win_length,
        window=istft_head.window,
        center=istft_head.center,
        normalized=istft_head.normalized,
        onesided=istft_head.onesided,
    )
    
    # Apply ISTFT postprocessing using torch backend
    audio_postprocess = istft_postprocess(
        spectral_np,
        config=istft_config,
        use_torch=True,
    )
    
    # Compare outputs
    print(f"\nPyTorch audio shape: {audio_pytorch_np.shape}")
    print(f"Postprocess audio shape: {audio_postprocess.shape}")
    
    # Check shapes match
    assert audio_pytorch_np.shape == audio_postprocess.shape, \
        f"Shape mismatch: {audio_pytorch_np.shape} vs {audio_postprocess.shape}"
    
    # Check values match within tolerance
    abs_diff = np.abs(audio_pytorch_np - audio_postprocess)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    print(f"Max absolute difference: {max_abs_diff}")
    print(f"Mean absolute difference: {mean_abs_diff}")
    
    # Very tight tolerance since both use PyTorch
    atol = 1e-6
    rtol = 1e-6
    
    np.testing.assert_allclose(
        audio_pytorch_np,
        audio_postprocess,
        rtol=rtol,
        atol=atol,
        err_msg=f"Audio outputs differ beyond tolerance (rtol={rtol}, atol={atol})"
    )
    
    print("✓ ISTFT postprocess matches PyTorch!")


def test_istft_different_spectral_formats():
    """Test ISTFT with different spectral tensor formats."""
    
    n_fft = 512  # Smaller for faster testing
    hop_length = 128
    n_freq_bins = n_fft // 2 + 1
    
    config = ISTFTConfig(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
    )
    
    batch_size = 1
    n_frames = 50
    
    # Create test spectral in [B, F, T, 2] format
    np.random.seed(42)
    spectral_bft2 = np.random.randn(batch_size, n_freq_bins, n_frames, 2).astype(np.float32)
    
    # Process it
    audio1 = istft_postprocess(spectral_bft2, config, use_torch=True)
    
    # Create same data in [B, 2, F, T] format
    spectral_b2ft = np.transpose(spectral_bft2, (0, 3, 1, 2))
    
    # Process it (should auto-detect and convert)
    audio2 = istft_postprocess(spectral_b2ft, config, use_torch=True)
    
    # Both should produce same audio
    np.testing.assert_allclose(
        audio1,
        audio2,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Different spectral formats should produce same audio"
    )
    
    print(f"✓ ISTFT handles both [B,F,T,2] and [B,2,F,T] formats")
    print(f"  Input format 1: {spectral_bft2.shape}")
    print(f"  Input format 2: {spectral_b2ft.shape}")
    print(f"  Output audio: {audio1.shape}")


def test_istft_audio_properties():
    """Test that ISTFT produces valid audio."""
    
    config = ISTFTConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window="hann",
    )
    
    # Create random spectral
    np.random.seed(42)
    batch_size = 1
    n_freq_bins = config.n_fft // 2 + 1
    n_frames = 100
    
    spectral = np.random.randn(batch_size, n_freq_bins, n_frames, 2).astype(np.float32)
    
    # Process
    audio = istft_postprocess(spectral, config, use_torch=True)
    
    # Check properties
    assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"
    assert audio.ndim == 2, f"Expected 2D audio, got {audio.ndim}D"
    assert audio.shape[0] == batch_size
    assert audio.shape[1] > 0, "Audio should have samples"
    
    # Audio values should be finite
    assert np.all(np.isfinite(audio)), "Audio contains non-finite values"
    
    print(f"✓ ISTFT produces valid audio")
    print(f"  Shape: {audio.shape}")
    print(f"  Dtype: {audio.dtype}")
    print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"  Mean: {audio.mean():.3f}")
    print(f"  Std: {audio.std():.3f}")


if __name__ == "__main__":
    print("Running ISTFT postprocess tests...\n")
    test_istft_postprocess_matches_pytorch()
    print()
    test_istft_different_spectral_formats()
    print()
    test_istft_audio_properties()
    print("\n✓ All ISTFT tests passed!")
