"""
Language Model step wrapper for Soprano TTS.

This module wraps the LM for step-by-step inference with KV cache.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union


class SopranoLMStep(nn.Module):
    """Wrapper for Soprano LM to support step-by-step inference.
    
    This wrapper enables:
    1. Prefill: Process full prompt to initialize KV cache
    2. Step: Generate one token at a time with KV cache
    
    Returns both logits (for sampling) and hidden states (for decoder).
    """
    
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
    ):
        """Initialize LM step wrapper.
        
        Args:
            model: Base language model (e.g., GPT-2 style)
            hidden_size: Hidden state dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        """Forward pass for one step.
        
        Args:
            input_ids: Token IDs, shape [B, seq_len]
                      For step inference, seq_len = 1
                      For prefill, seq_len = prompt length
            attention_mask: Attention mask, shape [B, total_seq_len]
            past_key_values: Cached key/value pairs from previous steps
                            Tuple of length num_layers, each containing
                            (key, value) tensors
        
        Returns:
            Tuple of (logits, hidden_states, new_past_key_values)
            - logits: Next token logits, shape [B, seq_len, vocab_size]
            - hidden_states: Last hidden state, shape [B, seq_len, hidden_size]
            - new_past_key_values: Updated KV cache
        """
        # Run model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        
        # Extract outputs
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        new_past_key_values = outputs.past_key_values
        
        return logits, hidden_states, new_past_key_values


class SimpleLM(nn.Module):
    """Simple reference LM for testing and export.
    
    This is a minimal transformer-based language model for demonstration.
    In production, replace with actual Soprano-80M model.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_position_embeddings: int = 2048,
    ):
        """Initialize simple LM.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_position_embeddings: Maximum sequence length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass.
        
        Args:
            input_ids: Token IDs, shape [B, seq_len]
            attention_mask: Attention mask
            past_key_values: KV cache (not used in this simple version)
            use_cache: Whether to return cache
            output_hidden_states: Whether to return hidden states
        
        Returns:
            Object with .logits, .hidden_states, .past_key_values attributes
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Add position embeddings
        if past_key_values is not None and len(past_key_values) > 0:
            # For step inference, position is offset by cache length
            position_offset = past_key_values[0][0].shape[2]  # Cache seq length
        else:
            position_offset = 0
        
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_embeds = self.position_embeddings(positions).unsqueeze(0)
        
        hidden_states = token_embeds + position_embeds
        
        # Apply transformer
        # Note: This simplified version doesn't properly implement KV cache
        # In production, use a proper transformer with cache support
        hidden_states = self.transformer(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        class Output:
            pass
        
        output = Output()
        output.logits = logits
        output.hidden_states = (hidden_states,) if output_hidden_states else None
        output.past_key_values = past_key_values  # Simplified: just return same cache
        
        return output


def create_dummy_lm(hidden_size: int = 512) -> SimpleLM:
    """Create a dummy LM for testing.
    
    Args:
        hidden_size: Hidden state dimension
    
    Returns:
        SimpleLM instance
    """
    return SimpleLM(
        vocab_size=50257,
        hidden_size=hidden_size,
        num_layers=6,
        num_heads=8,
    )
