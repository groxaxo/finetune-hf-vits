"""
ONNX Runtime backend for Soprano decoder.

This module provides CPU inference for the decoder using ONNX Runtime.
"""

import numpy as np
import json
import os
from typing import Optional, Dict, Any

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from soprano.audio.istft import istft_postprocess, ISTFTConfig


class ONNXDecoder:
    """ONNX Runtime decoder backend.
    
    Loads ONNX decoder model and performs inference with ISTFT postprocessing.
    """
    
    def __init__(
        self,
        model_path: str,
        istft_config: Optional[ISTFTConfig] = None,
        istft_config_path: Optional[str] = None,
        num_threads: Optional[int] = None,
        use_torch_istft: bool = True,
    ):
        """Initialize ONNX decoder.
        
        Args:
            model_path: Path to ONNX decoder model
            istft_config: ISTFTConfig instance (optional)
            istft_config_path: Path to ISTFT config JSON (optional)
            num_threads: Number of threads for ORT (None = default)
            use_torch_istft: Whether to use PyTorch for ISTFT (recommended)
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime"
            )
        
        self.model_path = model_path
        self.use_torch_istft = use_torch_istft
        
        # Load ISTFT config
        if istft_config is None:
            istft_config = self._load_istft_config(istft_config_path, model_path)
        self.istft_config = istft_config
        
        # Create ORT session
        self.session = self._create_session(num_threads)
        
        print(f"✓ ONNX decoder loaded: {model_path}")
        print(f"  ISTFT config: n_fft={istft_config.n_fft}, "
              f"hop_length={istft_config.hop_length}")
    
    def _load_istft_config(
        self,
        config_path: Optional[str],
        model_path: str,
    ) -> ISTFTConfig:
        """Load ISTFT config from file or use default.
        
        Args:
            config_path: Explicit path to config file
            model_path: ONNX model path (used to infer config path)
        
        Returns:
            ISTFTConfig instance
        """
        # Try explicit path first
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return ISTFTConfig.from_dict(config_dict)
        
        # Try to find config next to ONNX model
        auto_config_path = model_path.replace(".onnx", "_istft_config.json")
        if os.path.exists(auto_config_path):
            with open(auto_config_path, "r") as f:
                config_dict = json.load(f)
            return ISTFTConfig.from_dict(config_dict)
        
        # Fall back to default
        print("⚠ ISTFT config not found, using defaults")
        from soprano.audio.istft import create_default_istft_config
        return create_default_istft_config()
    
    def _create_session(self, num_threads: Optional[int]) -> ort.InferenceSession:
        """Create ONNX Runtime session with optimizations.
        
        Args:
            num_threads: Number of threads (None = default)
        
        Returns:
            ONNX Runtime InferenceSession
        """
        sess_options = ort.SessionOptions()
        
        # Set thread count if specified
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        
        # Enable all optimizations
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        # Create session with CPU provider
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
        )
        
        return session
    
    def infer(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run inference: hidden states -> audio.
        
        Args:
            hidden_states: Hidden states from LM, shape [B, T, H] or [B, H, T]
                         Will be converted to [B, H, T] format
        
        Returns:
            Audio waveform as float32, shape [B, samples]
        """
        # Ensure correct input format [B, H, T]
        hidden_states = self._prepare_input(hidden_states)
        
        # Run ONNX inference to get spectral output
        spectral = self._run_onnx(hidden_states)
        
        # Apply ISTFT postprocessing
        audio = istft_postprocess(
            spectral,
            config=self.istft_config,
            use_torch=self.use_torch_istft,
        )
        
        return audio
    
    def infer_spectral(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run inference: hidden states -> spectral (without ISTFT).
        
        Args:
            hidden_states: Hidden states from LM, shape [B, T, H] or [B, H, T]
        
        Returns:
            Spectral tensor as float32, shape [B, F, T, 2]
        """
        hidden_states = self._prepare_input(hidden_states)
        return self._run_onnx(hidden_states)
    
    # Heuristic threshold for detecting [B, T, H] vs [B, H, T] format
    _FORMAT_DETECTION_RATIO = 2
    
    def _prepare_input(self, hidden_states: np.ndarray) -> np.ndarray:
        """Prepare hidden states for ONNX model.
        
        Args:
            hidden_states: Input array, shape [B, T, H] or [B, H, T]
        
        Returns:
            Array in [B, H, T] format as float32
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected 3D hidden states, got shape {hidden_states.shape}"
            )
        
        # Convert to float32
        hidden_states = hidden_states.astype(np.float32)
        
        # Detect format and convert to [B, H, T] if needed
        batch_size, dim1, dim2 = hidden_states.shape
        
        # Assume if dim1 is much larger than dim2, it's [B, T, H]
        # Otherwise assume [B, H, T]
        if dim1 > dim2 * self._FORMAT_DETECTION_RATIO:
            # Transpose from [B, T, H] to [B, H, T]
            hidden_states = np.transpose(hidden_states, (0, 2, 1))
        
        return hidden_states
    
    def _run_onnx(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run ONNX model inference.
        
        Args:
            hidden_states: Input array, shape [B, H, T]
        
        Returns:
            Spectral tensor, shape [B, F, T, 2]
        """
        # Get input/output names
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        # Run inference
        outputs = self.session.run(
            [output_name],
            {input_name: hidden_states},
        )
        
        spectral = outputs[0]
        return spectral
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            "model_path": self.model_path,
            "inputs": [
                {
                    "name": inp.name,
                    "shape": inp.shape,
                    "type": inp.type,
                }
                for inp in inputs
            ],
            "outputs": [
                {
                    "name": out.name,
                    "shape": out.shape,
                    "type": out.type,
                }
                for out in outputs
            ],
            "istft_config": self.istft_config.to_dict(),
        }
