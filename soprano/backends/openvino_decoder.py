"""
OpenVINO backend for Soprano decoder.

This module provides CPU inference for the decoder using OpenVINO.
Requires OpenVINO 2025+ (no openvino-dev, uses openvino package).
"""

import numpy as np
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import openvino as ov
except ImportError:
    ov = None

from soprano.audio.istft import istft_postprocess, ISTFTConfig


class OpenVINODecoder:
    """OpenVINO decoder backend.
    
    Loads OpenVINO IR decoder model and performs inference with ISTFT postprocessing.
    """
    
    def __init__(
        self,
        model_path: str,
        istft_config: Optional[ISTFTConfig] = None,
        istft_config_path: Optional[str] = None,
        num_threads: Optional[int] = None,
        use_torch_istft: bool = True,
        device: str = "CPU",
    ):
        """Initialize OpenVINO decoder.
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            istft_config: ISTFTConfig instance (optional)
            istft_config_path: Path to ISTFT config JSON (optional)
            num_threads: Number of threads (None = default)
            use_torch_istft: Whether to use PyTorch for ISTFT
            device: Device to run on ("CPU", "GPU", etc.)
        """
        if ov is None:
            raise ImportError(
                "openvino is required for OpenVINO backend. "
                "Install with: pip install openvino"
            )
        
        self.model_path = model_path
        self.use_torch_istft = use_torch_istft
        self.device = device
        
        # Load ISTFT config
        if istft_config is None:
            istft_config = self._load_istft_config(istft_config_path, model_path)
        self.istft_config = istft_config
        
        # Load and compile model
        self.compiled_model = self._load_model(num_threads)
        
        print(f"✓ OpenVINO decoder loaded: {model_path}")
        print(f"  Device: {device}")
        print(f"  ISTFT config: n_fft={istft_config.n_fft}, "
              f"hop_length={istft_config.hop_length}")
    
    def _load_istft_config(
        self,
        config_path: Optional[str],
        model_path: str,
    ) -> ISTFTConfig:
        """Load ISTFT config from file or use default."""
        # Try explicit path first
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return ISTFTConfig.from_dict(config_dict)
        
        # Try to find config next to model
        xml_path = Path(model_path)
        istft_config_path = xml_path.parent / (xml_path.stem + "_istft_config.json")
        
        if istft_config_path.exists():
            with open(istft_config_path, "r") as f:
                config_dict = json.load(f)
            return ISTFTConfig.from_dict(config_dict)
        
        # Also check for ONNX config path
        onnx_config_path = str(model_path).replace(".xml", ".onnx").replace(
            ".onnx", "_istft_config.json"
        )
        if os.path.exists(onnx_config_path):
            with open(onnx_config_path, "r") as f:
                config_dict = json.load(f)
            return ISTFTConfig.from_dict(config_dict)
        
        # Fall back to default
        print("⚠ ISTFT config not found, using defaults")
        from soprano.audio.istft import create_default_istft_config
        return create_default_istft_config()
    
    def _load_model(self, num_threads: Optional[int]):
        """Load and compile OpenVINO model.
        
        Args:
            num_threads: Number of threads
        
        Returns:
            Compiled OpenVINO model
        """
        # Create OpenVINO Core
        core = ov.Core()
        
        # Load model
        model = core.read_model(self.model_path)
        
        # Set threading if specified
        config = {}
        if num_threads is not None:
            config["CPU_THREADS_NUM"] = str(num_threads)
        
        # Compile model
        compiled_model = core.compile_model(model, self.device, config=config)
        
        return compiled_model
    
    def infer(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run inference: hidden states -> audio.
        
        Args:
            hidden_states: Hidden states from LM, shape [B, T, H] or [B, H, T]
        
        Returns:
            Audio waveform as float32, shape [B, samples]
        """
        # Ensure correct input format [B, H, T]
        hidden_states = self._prepare_input(hidden_states)
        
        # Run OpenVINO inference to get spectral output
        spectral = self._run_openvino(hidden_states)
        
        # Apply ISTFT postprocessing
        audio = istft_postprocess(
            spectral,
            config=self.istft_config,
            use_torch=self.use_torch_istft,
        )
        
        return audio
    
    def _prepare_input(self, hidden_states: np.ndarray) -> np.ndarray:
        """Prepare hidden states for OpenVINO model."""
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected 3D hidden states, got shape {hidden_states.shape}"
            )
        
        # Convert to float32
        hidden_states = hidden_states.astype(np.float32)
        
        # Detect format and convert to [B, H, T] if needed
        batch_size, dim1, dim2 = hidden_states.shape
        
        if dim1 > dim2 * 2:  # Heuristic: sequence length >> hidden size
            hidden_states = np.transpose(hidden_states, (0, 2, 1))
        
        return hidden_states
    
    def _run_openvino(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run OpenVINO model inference.
        
        Args:
            hidden_states: Input array, shape [B, H, T]
        
        Returns:
            Spectral tensor, shape [B, F, T, 2]
        """
        # Get input/output info
        input_layer = self.compiled_model.input(0)
        output_layer = self.compiled_model.output(0)
        
        # Run inference
        result = self.compiled_model([hidden_states])[output_layer]
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        inputs = self.compiled_model.inputs
        outputs = self.compiled_model.outputs
        
        return {
            "model_path": self.model_path,
            "device": self.device,
            "inputs": [
                {
                    "name": inp.any_name,
                    "shape": list(inp.shape),
                }
                for inp in inputs
            ],
            "outputs": [
                {
                    "name": out.any_name,
                    "shape": list(out.shape),
                }
                for out in outputs
            ],
            "istft_config": self.istft_config.to_dict(),
        }


def convert_onnx_to_openvino(
    onnx_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """Convert ONNX model to OpenVINO IR format.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory (defaults to same as ONNX)
    
    Returns:
        Path to generated .xml file
    """
    if ov is None:
        raise ImportError("openvino is required. Install with: pip install openvino")
    
    if output_dir is None:
        output_dir = os.path.dirname(onnx_path)
    
    # Convert model
    print(f"Converting ONNX to OpenVINO IR...")
    print(f"  Input: {onnx_path}")
    
    model = ov.convert_model(onnx_path)
    
    # Generate output path
    onnx_name = Path(onnx_path).stem
    output_path = os.path.join(output_dir, f"{onnx_name}.xml")
    
    # Save model
    ov.save_model(model, output_path)
    
    print(f"  Output: {output_path}")
    print(f"✓ Conversion complete")
    
    return output_path
