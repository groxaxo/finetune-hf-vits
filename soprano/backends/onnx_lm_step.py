"""
ONNX Runtime backend for Soprano LM.

This module provides CPU inference for the language model using ONNX Runtime.
"""

import numpy as np
import json
import os
from typing import Optional, Dict, Any, List, Tuple

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from soprano.backends.sampling import sample_next_token, greedy_decode


class ONNXLM:
    """ONNX Runtime LM backend.
    
    Loads ONNX LM model and performs step-by-step inference with sampling.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        num_threads: Optional[int] = None,
    ):
        """Initialize ONNX LM.
        
        Args:
            model_path: Path to ONNX LM model
            config_path: Path to model config JSON (optional)
            num_threads: Number of threads for ORT (None = default)
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime"
            )
        
        self.model_path = model_path
        
        # Load config
        self.config = self._load_config(config_path, model_path)
        
        # Create ORT session
        self.session = self._create_session(num_threads)
        
        print(f"✓ ONNX LM loaded: {model_path}")
        print(f"  Hidden size: {self.config['hidden_size']}")
        print(f"  Vocab size: {self.config['vocab_size']}")
    
    def _load_config(
        self,
        config_path: Optional[str],
        model_path: str,
    ) -> Dict[str, Any]:
        """Load model config from file or use default.
        
        Args:
            config_path: Explicit path to config file
            model_path: ONNX model path
        
        Returns:
            Config dictionary
        """
        # Try explicit path first
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        
        # Try to find config next to ONNX model
        auto_config_path = model_path.replace(".onnx", "_config.json")
        if os.path.exists(auto_config_path):
            with open(auto_config_path, "r") as f:
                return json.load(f)
        
        # Fall back to default
        print("⚠ Model config not found, using defaults")
        return {
            "hidden_size": 512,
            "vocab_size": 50257,
            "num_layers": 6,
        }
    
    def _create_session(self, num_threads: Optional[int]) -> ort.InferenceSession:
        """Create ONNX Runtime session.
        
        Args:
            num_threads: Number of threads
        
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
    
    def prefill(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prefill: process full prompt to initialize generation.
        
        Args:
            input_ids: Token IDs, shape [B, seq_len]
            attention_mask: Attention mask, shape [B, seq_len]
        
        Returns:
            Tuple of (logits, hidden_states_sequence)
            - logits: shape [B, seq_len, vocab_size]
            - hidden_states_sequence: shape [B, seq_len, hidden_size]
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        
        # Run ONNX inference
        logits, hidden_states = self._run_onnx(input_ids, attention_mask)
        
        return logits, hidden_states
    
    def step(
        self,
        next_token_id: int,
        past_hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step: generate next token.
        
        Args:
            next_token_id: Next input token ID
            past_hidden_states: Hidden states from previous steps
            attention_mask: Attention mask
        
        Returns:
            Tuple of (logits, hidden_state_step, all_hidden_states)
            - logits: Next token logits, shape [1, 1, vocab_size]
            - hidden_state_step: Hidden state for this step, shape [1, 1, hidden_size]
            - all_hidden_states: Updated hidden states sequence
        """
        # Prepare input
        input_ids = np.array([[next_token_id]], dtype=np.int64)
        
        if attention_mask is None:
            seq_len = past_hidden_states.shape[1] + 1
            attention_mask = np.ones((1, seq_len), dtype=np.int64)
        
        # Run inference
        logits, hidden_state_step = self._run_onnx(input_ids, attention_mask)
        
        # Append to past hidden states
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
        """Generate hidden states sequence for decoder.
        
        This is the main interface for TTS: text tokens -> hidden states.
        
        Args:
            input_ids: Input token IDs, shape [B, seq_len] or [seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Repetition penalty
            seed: Random seed
        
        Returns:
            Hidden states tensor, shape [B, total_seq_len, hidden_size]
        """
        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        
        batch_size, prompt_len = input_ids.shape
        
        # Prefill with prompt
        _, hidden_states = self.prefill(input_ids)
        
        # Track generated tokens for repetition penalty
        generated_tokens = input_ids[0].tolist()
        
        # Generate tokens one by one
        for i in range(max_new_tokens):
            # Get logits for next token
            input_ids_step = np.array([[generated_tokens[-1]]], dtype=np.int64)
            logits, hidden_step = self._run_onnx(input_ids_step, None)
            
            # Sample next token
            next_logits = logits[0, -1, :]  # [vocab_size]
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
            
            # Check for EOS token (assume 50256 for GPT-2)
            # In production, use actual EOS token ID
            if next_token == 50256:
                break
        
        return hidden_states
    
    def _run_onnx(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ONNX model inference.
        
        Args:
            input_ids: Token IDs, shape [B, seq_len]
            attention_mask: Attention mask, shape [B, seq_len]
        
        Returns:
            Tuple of (logits, hidden_states)
        """
        # Ensure correct dtype
        input_ids = input_ids.astype(np.int64)
        
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        else:
            attention_mask = attention_mask.astype(np.int64)
        
        # Get input/output names
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]
        
        # Run inference
        outputs = self.session.run(
            output_names,
            {
                input_names[0]: input_ids,
                input_names[1]: attention_mask,
            },
        )
        
        logits = outputs[0]
        hidden_states = outputs[1]
        
        return logits, hidden_states
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            "model_path": self.model_path,
            "config": self.config,
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
        }
