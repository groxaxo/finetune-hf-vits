"""
End-to-end CPU pipeline test for Soprano TTS.

This test verifies the complete pipeline:
text -> LM hidden states -> decoder spectral -> ISTFT -> audio
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import tempfile
import pytest

from soprano.backends.lm_step import create_dummy_lm, SopranoLMStep
from soprano.vocos.decoder import VocosDecoder
from soprano.export.lm_step_export import export_lm_to_onnx
from soprano.export.decoder_export import export_decoder_to_onnx
from soprano.backends.onnx_lm_step import ONNXLM
from soprano.backends.onnx_decoder import ONNXDecoder


def test_e2e_cpu_pipeline():
    """Test complete end-to-end CPU pipeline."""
    
    print("\n" + "="*60)
    print("End-to-End CPU Pipeline Test")
    print("="*60)
    
    hidden_size = 256  # Smaller for faster testing
    vocab_size = 1000
    
    # Step 1: Create and export LM
    print("\n[1/5] Creating and exporting LM...")
    base_lm = create_dummy_lm(hidden_size=hidden_size)
    base_lm.vocab_size = vocab_size
    base_lm.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    
    lm_step = SopranoLMStep(
        model=base_lm,
        hidden_size=hidden_size,
    )
    lm_step.eval()
    
    with tempfile.NamedTemporaryFile(suffix="_lm.onnx", delete=False) as f:
        lm_onnx_path = f.name
    
    export_lm_to_onnx(
        lm_step=lm_step,
        output_path=lm_onnx_path,
        opset_version=14,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    print(f"✓ LM exported to: {lm_onnx_path}")
    
    # Step 2: Create and export decoder
    print("\n[2/5] Creating and exporting decoder...")
    decoder = VocosDecoder(
        hidden_size=hidden_size,
        intermediate_size=128,
        num_layers=2,
        n_fft=512,  # Smaller for faster testing
        hop_length=128,
    )
    decoder.eval()
    
    with tempfile.NamedTemporaryFile(suffix="_decoder.onnx", delete=False) as f:
        decoder_onnx_path = f.name
    
    export_decoder_to_onnx(
        decoder=decoder,
        output_path=decoder_onnx_path,
        opset_version=14,
        hidden_size=hidden_size,
    )
    print(f"✓ Decoder exported to: {decoder_onnx_path}")
    
    try:
        # Step 3: Load ONNX models
        print("\n[3/5] Loading ONNX models...")
        try:
            onnx_lm = ONNXLM(model_path=lm_onnx_path)
            onnx_decoder = ONNXDecoder(
                model_path=decoder_onnx_path,
                use_torch_istft=True,
            )
        except ImportError as e:
            pytest.skip(f"ONNX Runtime not available: {e}")
            return
        
        print("✓ ONNX models loaded")
        
        # Step 4: Generate hidden states from "text"
        print("\n[4/5] Generating hidden states from text tokens...")
        
        # Simulate tokenized text
        text_tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        
        hidden_states = onnx_lm.generate_hidden_states(
            input_ids=text_tokens,
            max_new_tokens=10,
            temperature=1.0,
            top_p=0.95,
            seed=42,
        )
        
        print(f"✓ Generated hidden states:")
        print(f"  Shape: {hidden_states.shape}")
        print(f"  Hidden size: {hidden_states.shape[2]}")
        print(f"  Sequence length: {hidden_states.shape[1]}")
        
        # Verify hidden states
        assert hidden_states.shape[0] == 1, "Batch size should be 1"
        assert hidden_states.shape[2] == hidden_size, \
            f"Hidden size should be {hidden_size}"
        assert hidden_states.dtype == np.float32, \
            f"Hidden states should be float32, got {hidden_states.dtype}"
        
        # Step 5: Generate audio from hidden states
        print("\n[5/5] Generating audio from hidden states...")
        
        # Decoder expects [B, H, T] format
        # LM outputs [B, T, H], so we need to transpose
        hidden_states_transposed = np.transpose(hidden_states, (0, 2, 1))
        
        audio = onnx_decoder.infer(hidden_states_transposed)
        
        print(f"✓ Generated audio:")
        print(f"  Shape: {audio.shape}")
        print(f"  Dtype: {audio.dtype}")
        print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
        print(f"  Duration (samples): {audio.shape[1]}")
        
        # Verify audio properties
        assert audio.ndim == 2, f"Audio should be 2D, got {audio.ndim}D"
        assert audio.shape[0] == 1, "Batch size should be 1"
        assert audio.shape[1] > 0, "Audio should have samples"
        assert audio.dtype == np.float32, f"Audio should be float32, got {audio.dtype}"
        assert np.all(np.isfinite(audio)), "Audio contains non-finite values"
        
        # Check that audio is not all zeros (actually generated)
        assert np.abs(audio).max() > 0, "Audio is all zeros"
        
        print("\n" + "="*60)
        print("✓ End-to-End CPU Pipeline Test PASSED!")
        print("="*60)
        print("\nPipeline summary:")
        print(f"  Text tokens: {len(text_tokens)}")
        print(f"  Generated tokens: 10")
        print(f"  Hidden states: {hidden_states.shape}")
        print(f"  Audio samples: {audio.shape[1]}")
        print(f"  Audio duration (at 22050 Hz): {audio.shape[1] / 22050:.2f}s")
        
    finally:
        # Cleanup
        for path in [lm_onnx_path, decoder_onnx_path]:
            if os.path.exists(path):
                os.remove(path)
            config_path = path.replace(".onnx", "_config.json")
            if os.path.exists(config_path):
                os.remove(config_path)
            istft_config_path = path.replace(".onnx", "_istft_config.json")
            if os.path.exists(istft_config_path):
                os.remove(istft_config_path)


def test_e2e_with_different_lengths():
    """Test E2E pipeline with different sequence lengths."""
    
    hidden_size = 128  # Very small for fast testing
    vocab_size = 500
    
    # Create models
    base_lm = create_dummy_lm(hidden_size=hidden_size)
    base_lm.vocab_size = vocab_size
    base_lm.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    lm_step = SopranoLMStep(model=base_lm, hidden_size=hidden_size)
    lm_step.eval()
    
    decoder = VocosDecoder(
        hidden_size=hidden_size,
        intermediate_size=64,
        num_layers=1,
        n_fft=256,
        hop_length=64,
    )
    decoder.eval()
    
    with tempfile.NamedTemporaryFile(suffix="_lm.onnx", delete=False) as f:
        lm_path = f.name
    with tempfile.NamedTemporaryFile(suffix="_dec.onnx", delete=False) as f:
        dec_path = f.name
    
    try:
        export_lm_to_onnx(lm_step, lm_path, hidden_size=hidden_size, vocab_size=vocab_size)
        export_decoder_to_onnx(decoder, dec_path, hidden_size=hidden_size)
        
        try:
            onnx_lm = ONNXLM(lm_path)
            onnx_decoder = ONNXDecoder(dec_path, use_torch_istft=True)
        except ImportError as e:
            pytest.skip(f"ONNX Runtime not available: {e}")
            return
        
        # Test with different input lengths
        for num_tokens in [3, 5, 10]:
            print(f"\nTesting with {num_tokens} input tokens...")
            
            text_tokens = np.arange(1, num_tokens + 1, dtype=np.int64)
            
            hidden_states = onnx_lm.generate_hidden_states(
                input_ids=text_tokens,
                max_new_tokens=3,
                temperature=1.0,
                seed=42,
            )
            
            hidden_states_transposed = np.transpose(hidden_states, (0, 2, 1))
            audio = onnx_decoder.infer(hidden_states_transposed)
            
            print(f"  Hidden states: {hidden_states.shape}")
            print(f"  Audio: {audio.shape}")
            
            assert audio.shape[1] > 0, f"No audio generated for {num_tokens} tokens"
            
        print("\n✓ E2E works with different sequence lengths")
        
    finally:
        for path in [lm_path, dec_path]:
            if os.path.exists(path):
                os.remove(path)
            for suffix in ["_config.json", "_istft_config.json"]:
                config_path = path.replace(".onnx", suffix)
                if os.path.exists(config_path):
                    os.remove(config_path)


if __name__ == "__main__":
    print("Running end-to-end CPU pipeline tests...")
    test_e2e_cpu_pipeline()
    print()
    test_e2e_with_different_lengths()
    print("\n✓ All E2E tests passed!")
