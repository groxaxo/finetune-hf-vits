"""
Test decoder ONNX export and parity with PyTorch.

This test verifies that the ONNX exported decoder produces
the same spectral output as the PyTorch version.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import tempfile
import pytest

from soprano.vocos.decoder import VocosDecoder
from soprano.export.decoder_export import export_decoder_to_onnx
from soprano.backends.onnx_decoder import ONNXDecoder


def test_decoder_spectral_parity():
    """Test that ONNX decoder produces same spectral output as PyTorch."""
    
    # Create decoder
    hidden_size = 512
    decoder = VocosDecoder(
        hidden_size=hidden_size,
        intermediate_size=256,  # Smaller for faster testing
        num_layers=2,
        n_fft=1024,
        hop_length=256,
    )
    decoder.eval()
    
    # Create test input with fixed seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 2
    seq_length = 50
    test_input = torch.randn(batch_size, hidden_size, seq_length)
    
    # Get PyTorch spectral output
    with torch.no_grad():
        pytorch_spectral = decoder.forward_spectral(test_input)
    
    pytorch_spectral_np = pytorch_spectral.numpy()
    
    # Export to ONNX
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    
    try:
        export_decoder_to_onnx(
            decoder=decoder,
            output_path=onnx_path,
            opset_version=14,
            hidden_size=hidden_size,
        )
        
        # Load ONNX decoder
        try:
            onnx_decoder = ONNXDecoder(
                model_path=onnx_path,
                use_torch_istft=True,
            )
        except ImportError as e:
            pytest.skip(f"ONNX Runtime not available: {e}")
            return
        
        # Get ONNX spectral output (without ISTFT)
        test_input_np = test_input.numpy()
        
        # Run ONNX inference to get spectral only
        onnx_spectral = onnx_decoder._run_onnx(test_input_np)
        
        # Compare outputs
        print(f"\nPyTorch spectral shape: {pytorch_spectral_np.shape}")
        print(f"ONNX spectral shape: {onnx_spectral.shape}")
        
        # Check shapes match
        assert pytorch_spectral_np.shape == onnx_spectral.shape, \
            f"Shape mismatch: PyTorch {pytorch_spectral_np.shape} vs ONNX {onnx_spectral.shape}"
        
        # Check values match within tolerance
        abs_diff = np.abs(pytorch_spectral_np - onnx_spectral)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        print(f"Max absolute difference: {max_abs_diff}")
        print(f"Mean absolute difference: {mean_abs_diff}")
        
        # Tolerance based on float32 precision
        atol = 1e-4
        rtol = 1e-4
        
        np.testing.assert_allclose(
            pytorch_spectral_np,
            onnx_spectral,
            rtol=rtol,
            atol=atol,
            err_msg=f"Spectral outputs differ beyond tolerance (rtol={rtol}, atol={atol})"
        )
        
        print("✓ Decoder ONNX parity test passed!")
        
    finally:
        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        config_path = onnx_path.replace(".onnx", "_istft_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)


def test_decoder_full_pipeline():
    """Test full decoder pipeline: hidden states -> spectral -> audio."""
    
    hidden_size = 512
    decoder = VocosDecoder(
        hidden_size=hidden_size,
        intermediate_size=256,
        num_layers=2,
    )
    decoder.eval()
    
    # Create test input
    torch.manual_seed(42)
    batch_size = 1
    seq_length = 100
    test_input = torch.randn(batch_size, hidden_size, seq_length)
    
    # Get spectral output
    with torch.no_grad():
        spectral = decoder.forward_spectral(test_input)
    
    # Verify spectral shape
    assert spectral.ndim == 4, f"Expected 4D spectral, got {spectral.ndim}D"
    assert spectral.shape[0] == batch_size
    assert spectral.shape[-1] == 2, "Last dim should be [real, imag]"
    
    print(f"✓ Spectral shape: {spectral.shape}")
    
    # Get full audio output
    with torch.no_grad():
        audio = decoder.forward(test_input)
    
    # Verify audio shape
    assert audio.ndim == 2, f"Expected 2D audio, got {audio.ndim}D"
    assert audio.shape[0] == batch_size
    assert audio.shape[1] > 0, "Audio should have samples"
    
    print(f"✓ Audio shape: {audio.shape}")
    print(f"✓ Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    print("✓ Decoder full pipeline test passed!")


if __name__ == "__main__":
    print("Running decoder ONNX parity tests...\n")
    test_decoder_spectral_parity()
    print()
    test_decoder_full_pipeline()
    print("\n✓ All decoder tests passed!")
