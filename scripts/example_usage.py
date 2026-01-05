#!/usr/bin/env python3
"""
Example script showing how to use Soprano TTS with ONNX backend.

This demonstrates the complete workflow:
1. Export models to ONNX
2. Run CPU inference
3. Save audio output

Usage:
    python scripts/example_usage.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tempfile
from pathlib import Path


def main():
    print("="*70)
    print("Soprano TTS Example Usage")
    print("="*70)
    
    # Step 1: Create and export models
    print("\n[Step 1] Creating and exporting models...")
    
    from soprano.backends.lm_step import create_dummy_lm, SopranoLMStep
    from soprano.vocos.decoder import VocosDecoder
    from soprano.export.lm_step_export import export_lm_to_onnx
    from soprano.export.decoder_export import export_decoder_to_onnx
    
    import torch
    
    hidden_size = 256  # Small for demo
    vocab_size = 1000
    
    # Create LM
    base_lm = create_dummy_lm(hidden_size=hidden_size)
    base_lm.vocab_size = vocab_size
    base_lm.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    lm_step = SopranoLMStep(base_lm, hidden_size=hidden_size)
    lm_step.eval()
    
    # Create decoder
    decoder = VocosDecoder(
        hidden_size=hidden_size,
        intermediate_size=128,
        num_layers=2,
        n_fft=512,
        hop_length=128,
    )
    decoder.eval()
    
    # Export to temporary directory
    tmpdir = tempfile.mkdtemp()
    lm_path = os.path.join(tmpdir, "lm.onnx")
    decoder_path = os.path.join(tmpdir, "decoder.onnx")
    
    print(f"  Exporting to: {tmpdir}")
    
    export_lm_to_onnx(lm_step, lm_path, hidden_size=hidden_size, vocab_size=vocab_size)
    export_decoder_to_onnx(decoder, decoder_path, hidden_size=hidden_size)
    
    print("✓ Models exported")
    
    # Step 2: Load ONNX backend
    print("\n[Step 2] Loading ONNX backend...")
    
    try:
        from soprano.backends.onnx_lm_step import ONNXLM
        from soprano.backends.onnx_decoder import ONNXDecoder
        
        onnx_lm = ONNXLM(lm_path)
        onnx_decoder = ONNXDecoder(decoder_path, use_torch_istft=True)
        
        print("✓ ONNX models loaded")
    except ImportError as e:
        print(f"⚠ ONNX Runtime not available: {e}")
        print("  Install with: pip install onnxruntime")
        return
    
    # Step 3: Generate audio
    print("\n[Step 3] Generating audio...")
    
    # Create dummy text tokens (in production, use proper tokenizer)
    text = "Hello, this is a test."
    tokens = np.array([ord(c) % vocab_size for c in text[:50]], dtype=np.int64)
    if len(tokens) == 0:
        tokens = np.array([1], dtype=np.int64)
    
    print(f"  Input tokens: {len(tokens)}")
    
    # Generate hidden states
    hidden_states = onnx_lm.generate_hidden_states(
        input_ids=tokens,
        max_new_tokens=20,
        temperature=1.0,
        top_p=0.95,
        seed=42,
    )
    
    print(f"  Hidden states shape: {hidden_states.shape}")
    
    # Generate audio
    hidden_states_transposed = np.transpose(hidden_states, (0, 2, 1))
    audio = onnx_decoder.infer(hidden_states_transposed)
    
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"  Duration: {audio.shape[1] / 22050:.2f} seconds")
    
    # Step 4: Save audio (optional)
    print("\n[Step 4] Saving audio...")
    
    try:
        import scipy.io.wavfile
        output_path = "soprano_output.wav"
        scipy.io.wavfile.write(output_path, 22050, audio[0])
        print(f"✓ Audio saved to: {output_path}")
    except ImportError:
        print("⚠ scipy not available, skipping audio save")
        print("  Install with: pip install scipy")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    print("\n" + "="*70)
    print("✓ Example completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  - Replace dummy LM with actual Soprano-80M model")
    print("  - Use proper tokenizer for text input")
    print("  - Try OpenVINO backend for better performance")
    print("  - Run benchmarks: scripts/bench_cpu_rtf.py")


if __name__ == "__main__":
    main()
