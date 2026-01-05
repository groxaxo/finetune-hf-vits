"""
Soprano TTS main interface with backend selection.

This module provides a unified interface for text-to-speech
with support for different backends (ONNX, OpenVINO).
"""

import numpy as np
from typing import Optional, Literal, Dict, Any
from pathlib import Path


class SopranoTTS:
    """Soprano TTS interface with backend selection.
    
    Supports:
    - PyTorch (GPU/CPU)
    - ONNX Runtime (CPU)
    - OpenVINO (CPU)
    """
    
    def __init__(
        self,
        lm_path: str,
        decoder_path: str,
        backend: Literal["pytorch", "onnx_cpu", "openvino_cpu"] = "pytorch",
        device: str = "cpu",
        num_threads: Optional[int] = None,
    ):
        """Initialize Soprano TTS.
        
        Args:
            lm_path: Path to LM model
            decoder_path: Path to decoder model
            backend: Backend to use
                - "pytorch": PyTorch (GPU/CPU based on device)
                - "onnx_cpu": ONNX Runtime on CPU
                - "openvino_cpu": OpenVINO on CPU
            device: Device for PyTorch backend ("cpu" or "cuda")
            num_threads: Number of threads for CPU backends
        """
        self.backend = backend
        self.device = device
        
        if backend == "pytorch":
            self._init_pytorch(lm_path, decoder_path, device)
        elif backend == "onnx_cpu":
            self._init_onnx(lm_path, decoder_path, num_threads)
        elif backend == "openvino_cpu":
            self._init_openvino(lm_path, decoder_path, num_threads)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        print(f"✓ Soprano TTS initialized with backend: {backend}")
    
    def _init_pytorch(self, lm_path: str, decoder_path: str, device: str):
        """Initialize PyTorch backend."""
        import torch
        from soprano.backends.lm_step import SopranoLMStep, create_dummy_lm
        from soprano.vocos.decoder import VocosDecoder
        
        # Load LM
        base_lm = create_dummy_lm(hidden_size=512)
        if Path(lm_path).exists():
            state_dict = torch.load(lm_path, map_location=device)
            base_lm.load_state_dict(state_dict)
        
        self.lm = SopranoLMStep(base_lm, hidden_size=512)
        self.lm.to(device)
        self.lm.eval()
        
        # Load decoder
        self.decoder = VocosDecoder(hidden_size=512)
        if Path(decoder_path).exists():
            state_dict = torch.load(decoder_path, map_location=device)
            self.decoder.load_state_dict(state_dict)
        
        self.decoder.to(device)
        self.decoder.eval()
        
        self.torch = torch
    
    def _init_onnx(self, lm_path: str, decoder_path: str, num_threads: Optional[int]):
        """Initialize ONNX Runtime backend."""
        from soprano.backends.onnx_lm_step import ONNXLM
        from soprano.backends.onnx_decoder import ONNXDecoder
        
        self.lm = ONNXLM(lm_path, num_threads=num_threads)
        self.decoder = ONNXDecoder(decoder_path, num_threads=num_threads)
    
    def _init_openvino(self, lm_path: str, decoder_path: str, num_threads: Optional[int]):
        """Initialize OpenVINO backend."""
        from soprano.backends.openvino_lm_step import OpenVINOLM
        from soprano.backends.openvino_decoder import OpenVINODecoder
        
        self.lm = OpenVINOLM(lm_path, num_threads=num_threads)
        self.decoder = OpenVINODecoder(decoder_path, num_threads=num_threads)
    
    def synthesize(
        self,
        text: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Repetition penalty
            seed: Random seed for deterministic output
        
        Returns:
            Dictionary with:
            - "audio": Audio waveform as numpy array, shape [samples]
            - "sample_rate": Sample rate (22050)
            - "text": Input text
        """
        # Tokenize text (simplified - in production use proper tokenizer)
        tokens = self._tokenize(text)
        
        if self.backend == "pytorch":
            return self._synthesize_pytorch(
                tokens, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, seed
            )
        else:
            return self._synthesize_cpu(
                tokens, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, seed
            )
    
    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize text to token IDs.
        
        This is a placeholder. In production, use proper tokenizer.
        """
        # Simple character-based tokenization for demo
        tokens = [ord(c) % 1000 for c in text[:100]]  # Limit length
        if not tokens:
            tokens = [1]  # At least one token
        return np.array(tokens, dtype=np.int64)
    
    def _synthesize_pytorch(
        self,
        tokens: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Synthesize using PyTorch backend."""
        import torch
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Convert tokens to tensor
        input_ids = torch.from_numpy(tokens).unsqueeze(0).to(self.device)
        
        # Generate hidden states (simplified - no proper generation loop)
        with torch.no_grad():
            logits, hidden_states, _ = self.lm(input_ids)
        
        # Generate audio
        with torch.no_grad():
            audio = self.decoder(hidden_states)
        
        # Convert to numpy
        audio_np = audio.cpu().numpy()[0]  # Remove batch dim
        
        return {
            "audio": audio_np,
            "sample_rate": 22050,
            "text": "",  # Original text would be stored here
        }
    
    def _synthesize_cpu(
        self,
        tokens: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Synthesize using ONNX/OpenVINO backend."""
        # Generate hidden states
        hidden_states = self.lm.generate_hidden_states(
            input_ids=tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        
        # Generate audio
        # Transpose from [B, T, H] to [B, H, T]
        hidden_states_transposed = np.transpose(hidden_states, (0, 2, 1))
        audio = self.decoder.infer(hidden_states_transposed)
        
        # Remove batch dimension
        audio_np = audio[0]
        
        return {
            "audio": audio_np,
            "sample_rate": 22050,
            "text": "",
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the TTS system."""
        info = {
            "backend": self.backend,
            "device": self.device if self.backend == "pytorch" else "cpu",
        }
        
        if hasattr(self.lm, "get_model_info"):
            info["lm"] = self.lm.get_model_info()
        
        if hasattr(self.decoder, "get_model_info"):
            info["decoder"] = self.decoder.get_model_info()
        
        return info
