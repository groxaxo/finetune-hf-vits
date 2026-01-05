#!/usr/bin/env python3
"""
Benchmark CPU RTF (Real-Time Factor) for Soprano TTS.

This script measures performance of the ONNX/OpenVINO CPU pipeline.

Usage:
    python scripts/bench_cpu_rtf.py \\
        --lm soprano_lm_step.onnx \\
        --decoder soprano_decoder_preistft.onnx \\
        --backend onnx \\
        --num_threads 4
"""

import argparse
import time
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def benchmark_cpu_pipeline(
    lm_path: str,
    decoder_path: str,
    backend: str = "onnx",
    num_threads: int = None,
    num_tokens: int = 50,
    max_new_tokens: int = 100,
    sample_rate: int = 22050,
):
    """Benchmark CPU inference pipeline.
    
    Args:
        lm_path: Path to LM model (ONNX or OpenVINO IR)
        decoder_path: Path to decoder model (ONNX or OpenVINO IR)
        backend: Backend to use ("onnx" or "openvino")
        num_threads: Number of threads (None = default)
        num_tokens: Number of input tokens
        max_new_tokens: Max tokens to generate
        sample_rate: Audio sample rate for RTF calculation
    """
    print("="*70)
    print("Soprano TTS CPU Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backend: {backend}")
    print(f"  LM model: {lm_path}")
    print(f"  Decoder model: {decoder_path}")
    print(f"  Threads: {num_threads if num_threads else 'default'}")
    print(f"  Input tokens: {num_tokens}")
    print(f"  Max new tokens: {max_new_tokens}")
    print()
    
    # Load models based on backend
    if backend == "onnx":
        from soprano.backends.onnx_lm_step import ONNXLM
        from soprano.backends.onnx_decoder import ONNXDecoder
        
        print("[1/3] Loading ONNX models...")
        lm = ONNXLM(lm_path, num_threads=num_threads)
        decoder = ONNXDecoder(decoder_path, num_threads=num_threads)
        
    elif backend == "openvino":
        from soprano.backends.openvino_lm_step import OpenVINOLM
        from soprano.backends.openvino_decoder import OpenVINODecoder
        
        print("[1/3] Loading OpenVINO models...")
        lm = OpenVINOLM(lm_path, num_threads=num_threads)
        decoder = OpenVINODecoder(decoder_path, num_threads=num_threads)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    print("✓ Models loaded\n")
    
    # Create input tokens
    input_ids = np.arange(1, num_tokens + 1, dtype=np.int64)
    
    # Warmup run (not timed)
    print("[2/3] Warmup run...")
    hidden_states = lm.generate_hidden_states(
        input_ids=input_ids,
        max_new_tokens=5,  # Short warmup
        temperature=1.0,
        seed=42,
    )
    hidden_states_transposed = np.transpose(hidden_states, (0, 2, 1))
    audio = decoder.infer(hidden_states_transposed)
    print("✓ Warmup complete\n")
    
    # Benchmark run
    print("[3/3] Benchmark run...")
    print("-"*70)
    
    # LM prefill
    prefill_start = time.perf_counter()
    logits, hidden_states = lm.prefill(input_ids[np.newaxis, :])
    prefill_time = time.perf_counter() - prefill_start
    
    print(f"LM Prefill:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Time: {prefill_time*1000:.2f} ms")
    print(f"  Throughput: {num_tokens/prefill_time:.1f} tokens/sec")
    
    # LM generation (step-by-step)
    generated_tokens = input_ids.tolist()
    step_times = []
    
    for i in range(max_new_tokens):
        step_start = time.perf_counter()
        
        input_id_step = np.array([[generated_tokens[-1]]], dtype=np.int64)
        logits, hidden_step = lm._run_openvino(input_id_step, None) if hasattr(lm, '_run_openvino') else lm._run_onnx(input_id_step, None)
        
        # Simple greedy decode for benchmarking
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_tokens.append(next_token)
        
        hidden_states = np.concatenate([hidden_states, hidden_step], axis=1)
        
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        # Check for EOS
        if next_token == 50256:
            break
    
    total_step_time = sum(step_times)
    mean_step_time = np.mean(step_times)
    
    print(f"\nLM Generation:")
    print(f"  Generated tokens: {len(step_times)}")
    print(f"  Total time: {total_step_time*1000:.2f} ms")
    print(f"  Mean per-token: {mean_step_time*1000:.2f} ms")
    print(f"  Throughput: {1/mean_step_time:.1f} tokens/sec")
    
    # Decoder inference
    decoder_start = time.perf_counter()
    hidden_states_transposed = np.transpose(hidden_states, (0, 2, 1))
    spectral = decoder._run_openvino(hidden_states_transposed) if hasattr(decoder, '_run_openvino') else decoder._run_onnx(hidden_states_transposed)
    decoder_time = time.perf_counter() - decoder_start
    
    print(f"\nDecoder (spectral):")
    print(f"  Sequence length: {hidden_states.shape[1]}")
    print(f"  Time: {decoder_time*1000:.2f} ms")
    
    # ISTFT postprocess
    istft_start = time.perf_counter()
    from soprano.audio.istft import istft_postprocess
    audio = istft_postprocess(spectral, decoder.istft_config, use_torch=True)
    istft_time = time.perf_counter() - istft_start
    
    print(f"\nISTFT Postprocess:")
    print(f"  Time: {istft_time*1000:.2f} ms")
    
    # Calculate RTF
    total_time = prefill_time + total_step_time + decoder_time + istft_time
    audio_duration = audio.shape[1] / sample_rate
    rtf = audio_duration / total_time
    
    print("-"*70)
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Total time: {total_time*1000:.2f} ms")
    print(f"  - LM prefill: {prefill_time*1000:.2f} ms ({prefill_time/total_time*100:.1f}%)")
    print(f"  - LM generation: {total_step_time*1000:.2f} ms ({total_step_time/total_time*100:.1f}%)")
    print(f"  - Decoder: {decoder_time*1000:.2f} ms ({decoder_time/total_time*100:.1f}%)")
    print(f"  - ISTFT: {istft_time*1000:.2f} ms ({istft_time/total_time*100:.1f}%)")
    print()
    print(f"Generated audio:")
    print(f"  Samples: {audio.shape[1]}")
    print(f"  Duration: {audio_duration:.2f} seconds")
    print()
    print(f"Real-Time Factor (RTF): {rtf:.3f}")
    print(f"  (Lower is better, <1.0 means faster than real-time)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Soprano TTS CPU inference"
    )
    parser.add_argument(
        "--lm",
        type=str,
        required=True,
        help="Path to LM model (ONNX or OpenVINO .xml)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to decoder model (ONNX or OpenVINO .xml)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["onnx", "openvino"],
        default="onnx",
        help="Backend to use (default: onnx)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of threads (default: auto)",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=50,
        help="Number of input tokens (default: 50)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Max tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Audio sample rate (default: 22050)",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.lm).exists():
        print(f"Error: LM model not found: {args.lm}")
        return 1
    
    if not Path(args.decoder).exists():
        print(f"Error: Decoder model not found: {args.decoder}")
        return 1
    
    # Run benchmark
    try:
        benchmark_cpu_pipeline(
            lm_path=args.lm,
            decoder_path=args.decoder,
            backend=args.backend,
            num_threads=args.num_threads,
            num_tokens=args.num_tokens,
            max_new_tokens=args.max_new_tokens,
            sample_rate=args.sample_rate,
        )
        return 0
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
