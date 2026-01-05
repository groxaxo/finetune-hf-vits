"""
OpenVINO backend for Soprano LM.

This module provides CPU inference for the language model using OpenVINO.
Requires OpenVINO 2025+ (no openvino-dev, uses openvino package).
"""

import numpy as np
import json
import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

try:
    import openvino as ov
except ImportError:
    ov = None

from soprano.backends.sampling import sample_next_token


class OpenVINOLM:
    """OpenVINO LM backend.
    
    Loads OpenVINO IR LM model and performs step-by-step inference with sampling.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        num_threads: Optional[int] = None,
        device: str = "CPU",
    ):
        """Initialize OpenVINO LM.
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            config_path: Path to model config JSON (optional)
            num_threads: Number of threads (None = default)
            device: Device to run on ("CPU", "GPU", etc.)
        """
        if ov is None:
            raise ImportError(
                "openvino is required for OpenVINO backend. "
                "Install with: pip install openvino"
            )
        
        self.model_path = model_path
        self.device = device
        
        # Load config
        self.config = self._load_config(config_path, model_path)
        
        # Load and compile model
        self.compiled_model = self._load_model(num_threads)
        
        print(f"✓ OpenVINO LM loaded: {model_path}")
        print(f"  Device: {device}")
        print(f"  Hidden size: {self.config['hidden_size']}")
        print(f"  Vocab size: {self.config['vocab_size']}")
    
    def _load_config(
        self,
        config_path: Optional[str],
        model_path: str,
    ) -> Dict[str, Any]:
        """Load model config from file or use default."""
        # Try explicit path first
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        
        # Try to find config next to model
        xml_path = Path(model_path)
        auto_config_path = xml_path.parent / (xml_path.stem + "_config.json")
        
        if auto_config_path.exists():
            with open(auto_config_path, "r") as f:
                return json.load(f)
        
        # Also check for ONNX config path
        onnx_config_path = str(model_path).replace(".xml", ".onnx").replace(
            ".onnx", "_config.json"
        )
        if os.path.exists(onnx_config_path):
            with open(onnx_config_path, "r") as f:
                return json.load(f)
        
        # Fall back to default
        print("⚠ Model config not found, using defaults")
        return {
            "hidden_size": 512,
            "vocab_size": 50257,
            "num_layers": 6,
        }
    
    def _load_model(self, num_threads: Optional[int]):
        """Load and compile OpenVINO model."""
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
    
    def prefill(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prefill: process full prompt to initialize generation."""
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        
        logits, hidden_states = self._run_openvino(input_ids, attention_mask)
        
        return logits, hidden_states
    
    def step(
        self,
        next_token_id: int,
        past_hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step: generate next token."""
        input_ids = np.array([[next_token_id]], dtype=np.int64)
        
        if attention_mask is None:
            seq_len = past_hidden_states.shape[1] + 1
            attention_mask = np.ones((1, seq_len), dtype=np.int64)
        
        logits, hidden_state_step = self._run_openvino(input_ids, attention_mask)
        
        all_hidden_states = np.concatenate(
            [past_hidden_states, hidden_state_step],
            axis=1,
        )
        
        return logits, hidden_state_step, all_hidden_states
    
    def generate_hidden_states(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate hidden states sequence for decoder."""
        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        
        batch_size, prompt_len = input_ids.shape
        
        # Prefill with prompt
        _, hidden_states = self.prefill(input_ids)
        
        # Track generated tokens
        generated_tokens = input_ids[0].tolist()
        
        # Generate tokens one by one
        for i in range(max_new_tokens):
            input_ids_step = np.array([[generated_tokens[-1]]], dtype=np.int64)
            logits, hidden_step = self._run_openvino(input_ids_step, None)
            
            # Sample next token
            next_logits = logits[0, -1, :]
            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_token_ids=generated_tokens,
                seed=seed,
            )
            
            generated_tokens.append(next_token)
            
            # Append hidden state
            hidden_states = np.concatenate(
                [hidden_states, hidden_step],
                axis=1,
            )
            
            # Check for EOS token
            if next_token == 50256:
                break
        
        return hidden_states
    
    def _run_openvino(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run OpenVINO model inference."""
        input_ids = input_ids.astype(np.int64)
        
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        else:
            attention_mask = attention_mask.astype(np.int64)
        
        # Get input layers
        input_layers = list(self.compiled_model.inputs)
        output_layers = list(self.compiled_model.outputs)
        
        # Run inference
        inputs = {
            input_layers[0]: input_ids,
            input_layers[1]: attention_mask,
        }
        
        results = self.compiled_model(inputs)
        
        logits = results[output_layers[0]]
        hidden_states = results[output_layers[1]]
        
        return logits, hidden_states
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        inputs = self.compiled_model.inputs
        outputs = self.compiled_model.outputs
        
        return {
            "model_path": self.model_path,
            "device": self.device,
            "config": self.config,
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
    
    print(f"Converting ONNX to OpenVINO IR...")
    print(f"  Input: {onnx_path}")
    
    model = ov.convert_model(onnx_path)
    
    onnx_name = Path(onnx_path).stem
    output_path = os.path.join(output_dir, f"{onnx_name}.xml")
    
    ov.save_model(model, output_path)
    
    print(f"  Output: {output_path}")
    print(f"✓ Conversion complete")
    
    return output_path
