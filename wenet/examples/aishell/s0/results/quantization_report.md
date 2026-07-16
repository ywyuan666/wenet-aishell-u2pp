# Model Quantization & Inference Report

## Model Size Comparison
| Version | Size (MB) | Reduction |
|---------|-----------|-----------|
| Original (FP32) | 178.6 | - |
| Quantized (INT8) | 64.9 | 2.8x |

## Inference Demo
```
# One-line inference
python benchmark/quantize_and_demo.py <path_to_wav>
```

## Architecture
- Model: Conformer U2++ (12 encoder layers, 3+3 decoder layers)
- Parameters: 42,920,713
- Export: JIT (TorchScript) + INT8 Quantized
