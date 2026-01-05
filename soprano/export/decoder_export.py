#!/usr/bin/env python3
"""
Export Soprano decoder (pre-ISTFT) to ONNX format.

This script exports the decoder model that produces spectral output,
ending before ISTFT transformation. The ISTFT is done as a CPU
postprocess step.

Usage:
    python soprano/export/decoder_export.py \\
        --repo_id ekwek/Soprano-80M \\
        --decoder_ckpt decoder.pth \\
        --out soprano_decoder_preistft.onnx
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# Import decoder components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from soprano.vocos.decoder import VocosDecoder


def export_decoder_to_onnx(
    decoder: VocosDecoder,
    output_path: str,
    opset_version: int = 14,
    hidden_size: int = 512,
    max_seq_length: int = 1000,
) -> None:
    """Export decoder to ONNX format.
    
    Args:
        decoder: VocosDecoder instance to export
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (14 is well-supported)
        hidden_size: Hidden state dimension
        max_seq_length: Maximum sequence length for dummy input
    """
    decoder.eval()
    
    # Create dummy input: [B, H, T]
    batch_size = 1
    seq_length = 100  # Moderate length for export
    dummy_input = torch.randn(batch_size, hidden_size, seq_length)
    
    # Define dynamic axes for flexibility
    dynamic_axes = {
        "hidden_states": {
            0: "batch",
            2: "sequence",
        },
        "spectral": {
            0: "batch",
            2: "time_frames",
        },
    }
    
    # Wrap forward_spectral for export
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
        
        def forward(self, hidden_states):
            return self.decoder.forward_spectral(hidden_states)
    
    wrapper = DecoderWrapper(decoder)
    
    print(f"Exporting decoder to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Opset version: {opset_version}")
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["hidden_states"],
        output_names=["spectral"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ Decoder exported to: {output_path}")
    
    # Save ISTFT config alongside ONNX model
    istft_config = decoder.get_istft_config()
    config_path = output_path.replace(".onnx", "_istft_config.json")
    with open(config_path, "w") as f:
        json.dump(istft_config, f, indent=2)
    print(f"✓ ISTFT config saved to: {config_path}")
    
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


def load_decoder_from_checkpoint(
    checkpoint_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    hidden_size: int = 512,
) -> VocosDecoder:
    """Load decoder from checkpoint or create default.
    
    Args:
        checkpoint_path: Path to decoder checkpoint file
        repo_id: HuggingFace repo ID (if available)
        hidden_size: Hidden state dimension
    
    Returns:
        VocosDecoder instance
    """
    # Create decoder with default parameters
    decoder = VocosDecoder(
        hidden_size=hidden_size,
        intermediate_size=1024,
        num_layers=4,
        n_fft=1024,
        hop_length=256,
    )
    
    # Load weights if checkpoint provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading decoder from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        decoder.load_state_dict(state_dict)
        print("✓ Decoder weights loaded")
    elif repo_id:
        print(f"⚠ Checkpoint path not found, using default initialization")
        print(f"  Note: In production, download from {repo_id}")
    else:
        print("⚠ No checkpoint provided, using default initialization")
    
    return decoder


def main():
    parser = argparse.ArgumentParser(
        description="Export Soprano decoder to ONNX (pre-ISTFT)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ekwek/Soprano-80M",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--decoder_ckpt",
        type=str,
        default=None,
        help="Path to decoder checkpoint (optional)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="soprano_decoder_preistft.onnx",
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
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load decoder
    decoder = load_decoder_from_checkpoint(
        checkpoint_path=args.decoder_ckpt,
        repo_id=args.repo_id,
        hidden_size=args.hidden_size,
    )
    
    # Export to ONNX
    export_decoder_to_onnx(
        decoder=decoder,
        output_path=args.out,
        opset_version=args.opset_version,
        hidden_size=args.hidden_size,
    )
    
    print("\n✓ Export complete!")
    print(f"\nNext steps:")
    print(f"  1. Test with: soprano/backends/onnx_decoder.py")
    print(f"  2. Benchmark with: scripts/bench_cpu_rtf.py")


if __name__ == "__main__":
    main()
