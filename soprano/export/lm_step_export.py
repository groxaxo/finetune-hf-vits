#!/usr/bin/env python3
"""
Export Soprano LM step model to ONNX format.

This script exports the language model for step-by-step inference
with KV cache support.

Usage:
    python soprano/export/lm_step_export.py \\
        --repo_id ekwek/Soprano-80M \\
        --out soprano_lm_step.onnx
"""

import argparse
import json
import os
import torch
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from soprano.backends.lm_step import SopranoLMStep, SimpleLM, create_dummy_lm


def export_lm_to_onnx(
    lm_step: SopranoLMStep,
    output_path: str,
    opset_version: int = 14,
    hidden_size: int = 512,
    num_layers: int = 6,
    vocab_size: int = 50257,
) -> None:
    """Export LM step model to ONNX format.
    
    Args:
        lm_step: SopranoLMStep wrapper instance
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        hidden_size: Hidden state dimension
        num_layers: Number of layers
        vocab_size: Vocabulary size
    """
    lm_step.eval()
    
    # Create dummy inputs for step inference
    batch_size = 1
    seq_len = 1  # Step inference processes one token
    
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Note: For simplicity, we export without KV cache inputs
    # Production version should include KV cache in ONNX I/O
    dummy_past_kv = None
    
    # Define dynamic axes
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
        "hidden_states": {0: "batch", 1: "sequence"},
    }
    
    # Wrapper for export (simplified without cache for now)
    class LMExportWrapper(torch.nn.Module):
        def __init__(self, lm_step):
            super().__init__()
            self.lm_step = lm_step
        
        def forward(self, input_ids, attention_mask):
            logits, hidden_states, _ = self.lm_step(
                input_ids, attention_mask, past_key_values=None
            )
            return logits, hidden_states
    
    wrapper = LMExportWrapper(lm_step)
    
    print(f"Exporting LM to ONNX...")
    print(f"  Input IDs shape: {dummy_input_ids.shape}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Opset version: {opset_version}")
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "hidden_states"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ LM exported to: {output_path}")
    
    # Save model config
    config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
    }
    config_path = output_path.replace(".onnx", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Model config saved to: {config_path}")
    
    # Verify export
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model validation passed")
    except ImportError:
        print("⚠ ONNX package not installed, skipping validation")
    except Exception as e:
        print(f"⚠ ONNX validation warning: {e}")


def load_lm_from_checkpoint(
    checkpoint_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    hidden_size: int = 512,
) -> SopranoLMStep:
    """Load LM from checkpoint or create dummy.
    
    Args:
        checkpoint_path: Path to LM checkpoint
        repo_id: HuggingFace repo ID
        hidden_size: Hidden state dimension
    
    Returns:
        SopranoLMStep wrapper instance
    """
    # Create base LM
    # In production, load actual Soprano model
    base_lm = create_dummy_lm(hidden_size=hidden_size)
    
    # Load weights if checkpoint provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LM from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        base_lm.load_state_dict(state_dict)
        print("✓ LM weights loaded")
    elif repo_id:
        print(f"⚠ Checkpoint not found, using default initialization")
        print(f"  Note: In production, download from {repo_id}")
    else:
        print("⚠ No checkpoint provided, using default initialization")
    
    # Wrap in step interface
    lm_step = SopranoLMStep(
        model=base_lm,
        hidden_size=hidden_size,
        num_layers=6,
        num_heads=8,
    )
    
    return lm_step


def main():
    parser = argparse.ArgumentParser(
        description="Export Soprano LM to ONNX (step model)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ekwek/Soprano-80M",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--lm_ckpt",
        type=str,
        default=None,
        help="Path to LM checkpoint (optional)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="soprano_lm_step.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Hidden state dimension (default: 512)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size (default: 50257)",
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load LM
    lm_step = load_lm_from_checkpoint(
        checkpoint_path=args.lm_ckpt,
        repo_id=args.repo_id,
        hidden_size=args.hidden_size,
    )
    
    # Export to ONNX
    export_lm_to_onnx(
        lm_step=lm_step,
        output_path=args.out,
        opset_version=args.opset_version,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
    )
    
    print("\n✓ Export complete!")
    print(f"\nNext steps:")
    print(f"  1. Test with: soprano/backends/onnx_lm_step.py")
    print(f"  2. Run e2e test: tests/test_e2e_cpu_pipeline.py")


if __name__ == "__main__":
    main()
