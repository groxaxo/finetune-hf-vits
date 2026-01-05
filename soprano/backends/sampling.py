"""
Token sampling utilities for Soprano LM inference.

Implements temperature, top-p, and repetition penalty sampling
in pure Python (outside ONNX).
"""

import numpy as np
from typing import Optional, List


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    generated_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> int:
    """Sample next token from logits with various sampling strategies.
    
    Args:
        logits: Logits for next token prediction, shape [vocab_size]
        temperature: Sampling temperature (0.0 = greedy, >1.0 = more random)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        top_k: Top-k sampling threshold (0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        generated_token_ids: Previously generated tokens for repetition penalty
        seed: Random seed for deterministic sampling
        
    Returns:
        Sampled token ID as integer
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Ensure logits is 1D
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}")
    
    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_token_ids:
        logits = _apply_repetition_penalty(
            logits, generated_token_ids, repetition_penalty
        )
    
    # Greedy decoding for temperature near zero
    if temperature < 1e-8:
        return int(np.argmax(logits))
    
    # Apply temperature
    logits = logits / temperature
    
    # Convert to probabilities
    # Use log-softmax for numerical stability
    logits = logits - np.max(logits)  # Subtract max for numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    
    # Apply top-k filtering
    if top_k > 0:
        probs = _apply_top_k(probs, top_k)
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        probs = _apply_top_p(probs, top_p)
    
    # Renormalize after filtering
    probs = probs / np.sum(probs)
    
    # Sample from distribution
    token_id = np.random.choice(len(probs), p=probs)
    
    return int(token_id)


def _apply_repetition_penalty(
    logits: np.ndarray,
    generated_token_ids: List[int],
    penalty: float,
) -> np.ndarray:
    """Apply repetition penalty to logits.
    
    Reduces probability of tokens that have already been generated.
    """
    logits = logits.copy()
    
    for token_id in set(generated_token_ids):
        # If logit is positive, divide by penalty (reduce it)
        # If logit is negative, multiply by penalty (make more negative)
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    
    return logits


def _apply_top_k(probs: np.ndarray, top_k: int) -> np.ndarray:
    """Apply top-k filtering to probabilities.
    
    Keep only top-k highest probability tokens, set others to 0.
    """
    probs = probs.copy()
    
    # Get indices of top-k values
    top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
    
    # Create mask
    mask = np.zeros_like(probs, dtype=bool)
    mask[top_k_indices] = True
    
    # Zero out non-top-k probabilities
    probs[~mask] = 0.0
    
    return probs


def _apply_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Apply top-p (nucleus) filtering to probabilities.
    
    Keep only tokens whose cumulative probability is less than top_p.
    """
    probs = probs.copy()
    
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Calculate cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)
    
    # Find cutoff point
    # Keep tokens until cumulative probability exceeds top_p
    cutoff_idx = np.searchsorted(cumsum_probs, top_p, side="right")
    
    # Ensure at least one token is kept
    cutoff_idx = max(1, cutoff_idx)
    
    # Create mask for tokens to keep
    keep_indices = sorted_indices[:cutoff_idx]
    mask = np.zeros_like(probs, dtype=bool)
    mask[keep_indices] = True
    
    # Zero out filtered probabilities
    probs[~mask] = 0.0
    
    return probs


def greedy_decode(logits: np.ndarray) -> int:
    """Greedy decoding: select token with highest probability.
    
    Args:
        logits: Logits for next token prediction, shape [vocab_size]
        
    Returns:
        Token ID with highest probability
    """
    return int(np.argmax(logits))
