# Soprano TTS ONNX Export and CPU Inference

This repository includes support for **Soprano TTS** ONNX export and CPU inference using ONNX Runtime and OpenVINO.

## Features

- ✅ **Decoder ONNX Export** (pre-ISTFT) with CPU postprocessing
- ✅ **LM Step-Model ONNX Export** with KV cache support
- ✅ **ONNX Runtime Backend** for CPU inference
- ✅ **OpenVINO Backend** for accelerated CPU inference (optional)
- ✅ **Benchmarking Tools** for RTF (Real-Time Factor) measurement
- ✅ **Unified TTS Interface** with backend selection

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### With ONNX Runtime (for CPU inference)

```bash
pip install -e ".[onnx]"
```

### With OpenVINO (for accelerated CPU inference)

```bash
pip install -e ".[openvino]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Export Models to ONNX

#### Export Decoder (Pre-ISTFT)

```bash
python soprano/export/decoder_export.py \
    --repo_id ekwek/Soprano-80M \
    --decoder_ckpt decoder.pth \
    --out soprano_decoder_preistft.onnx \
    --hidden_size 512
```

This exports the decoder that produces spectral output (before ISTFT). The ISTFT is performed as a CPU postprocess step.

**Output files:**
- `soprano_decoder_preistft.onnx` - ONNX model
- `soprano_decoder_preistft_istft_config.json` - ISTFT configuration

#### Export Language Model (Step Model)

```bash
python soprano/export/lm_step_export.py \
    --repo_id ekwek/Soprano-80M \
    --lm_ckpt lm.pth \
    --out soprano_lm_step.onnx \
    --hidden_size 512 \
    --vocab_size 50257
```

This exports the LM for step-by-step inference with KV cache support.

**Output files:**
- `soprano_lm_step.onnx` - ONNX model
- `soprano_lm_step_config.json` - Model configuration

### 2. Convert to OpenVINO (Optional)

For better CPU performance, convert ONNX models to OpenVINO IR format:

#### Using Python API

```python
from soprano.backends.openvino_decoder import convert_onnx_to_openvino

# Convert decoder
decoder_ir = convert_onnx_to_openvino("soprano_decoder_preistft.onnx")

# Convert LM
from soprano.backends.openvino_lm_step import convert_onnx_to_openvino
lm_ir = convert_onnx_to_openvino("soprano_lm_step.onnx")
```

#### Using CLI (requires OpenVINO tools)

```bash
# Convert decoder
ovc soprano_decoder_preistft.onnx

# Convert LM
ovc soprano_lm_step.onnx
```

**Note:** OpenVINO 2025+ uses `ovc` (OpenVINO Converter), not the deprecated `mo` (Model Optimizer).

**Output files:**
- `soprano_decoder_preistft.xml` - OpenVINO IR model
- `soprano_decoder_preistft.bin` - OpenVINO weights

### 3. Run Inference

#### Using ONNX Runtime Backend

```python
from soprano.tts import SopranoTTS

# Initialize TTS with ONNX backend
tts = SopranoTTS(
    lm_path="soprano_lm_step.onnx",
    decoder_path="soprano_decoder_preistft.onnx",
    backend="onnx_cpu",
    num_threads=4,
)

# Synthesize speech
result = tts.synthesize(
    text="Hello, this is a test.",
    max_new_tokens=100,
    temperature=1.0,
    seed=42,
)

# Save audio
import scipy.io.wavfile
scipy.io.wavfile.write(
    "output.wav",
    rate=result["sample_rate"],
    data=result["audio"],
)
```

#### Using OpenVINO Backend

```python
from soprano.tts import SopranoTTS

# Initialize TTS with OpenVINO backend
tts = SopranoTTS(
    lm_path="soprano_lm_step.xml",  # Note: .xml for OpenVINO
    decoder_path="soprano_decoder_preistft.xml",
    backend="openvino_cpu",
    num_threads=4,
)

# Synthesize speech
result = tts.synthesize(
    text="Hello, this is a test.",
    max_new_tokens=100,
    temperature=1.0,
    seed=42,
)
```

### 4. Benchmark Performance

Measure CPU inference performance and RTF (Real-Time Factor):

```bash
python scripts/bench_cpu_rtf.py \
    --lm soprano_lm_step.onnx \
    --decoder soprano_decoder_preistft.onnx \
    --backend onnx \
    --num_threads 4 \
    --num_tokens 50 \
    --max_new_tokens 100
```

**For OpenVINO:**

```bash
python scripts/bench_cpu_rtf.py \
    --lm soprano_lm_step.xml \
    --decoder soprano_decoder_preistft.xml \
    --backend openvino \
    --num_threads 4
```

The benchmark reports:
- LM prefill time
- LM per-token generation latency
- Decoder spectral generation time
- ISTFT postprocessing time
- **RTF (Real-Time Factor)**: audio_duration / wall_time
  - RTF < 1.0 means faster than real-time ✅

## Architecture

### Two-Model Pipeline

```
Text → Tokenizer → LM (step-by-step) → Hidden States → Decoder → Spectral → ISTFT → Audio
                     ^                                   ^          ^         ^
                     |                                   |          |         |
                  ONNX/OV                            ONNX/OV        |      CPU Post
                                                                    |
                                                            (pre-ISTFT export)
```

### Design Decisions

1. **Decoder exports WITHOUT ISTFT inside ONNX/OpenVINO**
   - ONNX/OpenVINO export ends at spectral frames
   - ISTFT is performed as CPU postprocess (PyTorch or NumPy)
   - **Rationale:** ISTFT is complex and error-prone in ONNX; CPU postprocess is more reliable

2. **LM ONNX is "step-model" only**
   - One-token forward with KV cache (when fully implemented)
   - Sampling loop (temperature/top_p/repetition penalty) in Python
   - **Rationale:** More flexible and easier to debug than ONNX sampling ops

3. **Spectral Tensor Format**
   - Shape: `[B, F, T, 2]` where last dim is `[real, imag]`
   - Alternative: `[B, 2, F, T]` (auto-detected and converted)
   - F = frequency bins (n_fft//2 + 1 for one-sided)
   - T = time frames

## Testing

Run all tests:

```bash
# Test decoder ONNX parity
python tests/test_decoder_onnx_parity.py

# Test ISTFT postprocessing
python tests/test_istft_postprocess_matches_pytorch.py

# Test LM ONNX export
python tests/test_lm_step_onnx_smoke.py

# Test end-to-end CPU pipeline
python tests/test_e2e_cpu_pipeline.py
```

Or use pytest:

```bash
pytest tests/ -v
```

## Performance Tips

### Threading

Set the number of threads for optimal performance:

```python
# ONNX Runtime
tts = SopranoTTS(..., backend="onnx_cpu", num_threads=4)

# OpenVINO
tts = SopranoTTS(..., backend="openvino_cpu", num_threads=4)
```

Or via environment variables:

```bash
# ONNX Runtime
export OMP_NUM_THREADS=4

# OpenVINO
export OMP_NUM_THREADS=4
```

### Weight Format (OpenVINO)

When using Optimum Intel for OpenVINO export, specify weight format:

```bash
# FP32 (no quantization)
optimum-cli export openvino --model ekwek/Soprano-80M --weight-format fp32

# INT8 (quantized, faster but may lose quality)
optimum-cli export openvino --model ekwek/Soprano-80M --weight-format int8
```

**Note:** Large models may default to quantization. Explicitly set `--weight-format fp32` for full precision.

## ISTFT Configuration

The ISTFT configuration is automatically saved during export:

```json
{
  "n_fft": 1024,
  "hop_length": 256,
  "win_length": 1024,
  "window": "hann",
  "center": true,
  "normalized": false,
  "onesided": true,
  "length": null
}
```

This ensures the ISTFT postprocess exactly matches the original model behavior.

## Troubleshooting

### ONNX Runtime not found

```bash
pip install onnxruntime
```

### OpenVINO not found

```bash
pip install openvino
```

**Important:** Use `openvino` package, NOT `openvino-dev` (deprecated in 2025+).

### Shape mismatch errors

Ensure hidden states are in correct format:
- LM outputs: `[B, T, H]` (batch, time, hidden)
- Decoder expects: `[B, H, T]` (batch, hidden, time)

The backends handle this automatically, but if manually processing:

```python
hidden_states = np.transpose(hidden_states, (0, 2, 1))
```

### Audio quality issues

1. Check ISTFT config matches original model
2. Verify spectral output parity (run tests)
3. Try PyTorch ISTFT backend: `use_torch_istft=True`

## References

- [OpenVINO 2025+ Documentation](https://docs.openvino.ai/2025/)
- [OpenVINO Model Conversion (ovc)](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/conversion-parameters.html)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Optimum Intel OpenVINO Export](https://huggingface.co/docs/optimum-intel/en/openvino/export)

## License

The Soprano TTS implementation follows the same license as the base repository (see LICENSE file).
