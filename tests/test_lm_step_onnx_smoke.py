"""
Test LM step ONNX export smoke test.

This test verifies that the LM ONNX export works and can perform
basic inference operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import tempfile
import pytest

from soprano.backends.lm_step import SopranoLMStep, create_dummy_lm
from soprano.export.lm_step_export import export_lm_to_onnx
from soprano.backends.onnx_lm_step import ONNXLM


def test_lm_step_onnx_smoke():
    """Smoke test: verify LM ONNX export and basic inference."""
    
    hidden_size = 512
    vocab_size = 1000  # Smaller for faster testing
    
    # Create dummy LM
    base_lm = create_dummy_lm(hidden_size=hidden_size)
    lm_step = SopranoLMStep(
        model=base_lm,
        hidden_size=hidden_size,
        num_layers=6,
        num_heads=8,
    )
    lm_step.eval()
    
    # Export to ONNX
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    
    try:
        export_lm_to_onnx(
            lm_step=lm_step,
            output_path=onnx_path,
            opset_version=14,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        
        # Load ONNX LM
        try:
            onnx_lm = ONNXLM(
                model_path=onnx_path,
                num_threads=1,
            )
        except ImportError as e:
            pytest.skip(f"ONNX Runtime not available: {e}")
            return
        
        # Test prefill
        prompt_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        logits, hidden_states = onnx_lm.prefill(prompt_ids)
        
        print(f"\nPrefill test:")
        print(f"  Prompt shape: {prompt_ids.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Hidden states shape: {hidden_states.shape}")
        
        # Check shapes
        assert logits.shape[0] == 1, "Batch size should be 1"
        assert logits.shape[1] == prompt_ids.shape[1], "Seq length should match"
        assert hidden_states.shape[0] == 1, "Batch size should be 1"
        assert hidden_states.shape[1] == prompt_ids.shape[1], "Seq length should match"
        assert hidden_states.shape[2] == hidden_size, f"Hidden size should be {hidden_size}"
        
        print("✓ Prefill works correctly")
        
        # Test step inference (3 steps)
        for step in range(3):
            # Use last token from prompt or previous step
            if step == 0:
                next_token = prompt_ids[0, -1]
            else:
                # Just use a dummy token for testing
                next_token = step + 10
            
            step_logits, step_hidden, hidden_states = onnx_lm.step(
                next_token_id=int(next_token),
                past_hidden_states=hidden_states,
            )
            
            print(f"\nStep {step + 1}:")
            print(f"  Input token: {next_token}")
            print(f"  Step logits shape: {step_logits.shape}")
            print(f"  Step hidden shape: {step_hidden.shape}")
            print(f"  Total hidden states shape: {hidden_states.shape}")
            
            # Check shapes
            assert step_logits.shape[0] == 1, "Batch size should be 1"
            assert step_logits.shape[1] == 1, "Step should process 1 token"
            assert step_hidden.shape == (1, 1, hidden_size), \
                f"Step hidden should be [1, 1, {hidden_size}]"
            assert hidden_states.shape[1] == prompt_ids.shape[1] + step + 1, \
                "Hidden states should accumulate"
        
        print("\n✓ Step inference works correctly")
        
        # Verify KV cache effect (sequence grows)
        final_seq_len = hidden_states.shape[1]
        expected_seq_len = prompt_ids.shape[1] + 3  # 5 + 3 steps
        assert final_seq_len == expected_seq_len, \
            f"Expected seq len {expected_seq_len}, got {final_seq_len}"
        
        print(f"✓ Hidden states accumulated correctly: {final_seq_len} tokens")
        
    finally:
        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        config_path = onnx_path.replace(".onnx", "_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)


def test_lm_hidden_state_dimensions():
    """Test that LM produces hidden states with correct dimensions."""
    
    hidden_size = 512
    
    # Create dummy LM
    base_lm = create_dummy_lm(hidden_size=hidden_size)
    lm_step = SopranoLMStep(
        model=base_lm,
        hidden_size=hidden_size,
    )
    lm_step.eval()
    
    # Create test input
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Run forward
    with torch.no_grad():
        logits, hidden_states, past_kv = lm_step(input_ids)
    
    print(f"\nLM output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Hidden states: {hidden_states.shape}")
    
    # Check dimensions
    assert logits.shape == (batch_size, seq_len, base_lm.vocab_size)
    assert hidden_states.shape == (batch_size, seq_len, hidden_size), \
        f"Expected hidden states shape [1, {seq_len}, {hidden_size}], got {hidden_states.shape}"
    
    print(f"✓ LM produces hidden states with correct dimension: {hidden_size}")


def test_lm_generate_hidden_states():
    """Test LM hidden state generation for decoder."""
    
    hidden_size = 256  # Smaller for faster testing
    vocab_size = 1000
    
    # Create and export LM
    base_lm = create_dummy_lm(hidden_size=hidden_size)
    base_lm.vocab_size = vocab_size
    base_lm.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    
    lm_step = SopranoLMStep(
        model=base_lm,
        hidden_size=hidden_size,
    )
    lm_step.eval()
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    
    try:
        export_lm_to_onnx(
            lm_step=lm_step,
            output_path=onnx_path,
            opset_version=14,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        
        try:
            onnx_lm = ONNXLM(model_path=onnx_path)
        except ImportError as e:
            pytest.skip(f"ONNX Runtime not available: {e}")
            return
        
        # Generate hidden states
        input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        
        hidden_states = onnx_lm.generate_hidden_states(
            input_ids=input_ids,
            max_new_tokens=5,
            temperature=1.0,
            seed=42,
        )
        
        print(f"\nGenerated hidden states:")
        print(f"  Input tokens: {len(input_ids)}")
        print(f"  Generated tokens: 5")
        print(f"  Total sequence length: {hidden_states.shape[1]}")
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Hidden size: {hidden_states.shape[2]}")
        
        # Check shape
        assert hidden_states.shape[0] == 1, "Batch size should be 1"
        assert hidden_states.shape[1] >= len(input_ids), \
            "Sequence should include at least input tokens"
        assert hidden_states.shape[2] == hidden_size, \
            f"Hidden size should be {hidden_size}"
        
        print("✓ Hidden state generation works correctly")
        
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        config_path = onnx_path.replace(".onnx", "_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)


if __name__ == "__main__":
    print("Running LM step ONNX smoke tests...\n")
    test_lm_step_onnx_smoke()
    print()
    test_lm_hidden_state_dimensions()
    print()
    test_lm_generate_hidden_states()
    print("\n✓ All LM step tests passed!")
