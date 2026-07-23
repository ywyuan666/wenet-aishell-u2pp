# Model Quantization & Inference Report

## Model Size Comparison (Fast Training Checkpoint)

> ✅ **实际结果** — 基于 fast training (100 utts, 5epochs) 检查点的实际量化效果。

| Version | Size (MB) | Reduction |
|---------|-----------|-----------|
| Checkpoint (FP32) | 178.6 | - |
| JIT TorchScript | 178.6 | 1.00x |
| **Quantized (INT8)** | **64.9** | **2.75x** |

## Full Training 预期

全量训练模型 (360 epochs, AISHELL-1) 参数量为 ~18.6M，预估：

| Version | Size (MB) | Reduction |
|---------|-----------|-----------|
| Full FP32 Checkpoint | ~42.3 | - |
| Full INT8 Quantized | ~11.5 | ~3.7x |

## Inference Performance (预估)

| 场景 | 模型 | RTF | 延迟 |
|------|------|-----|------|
| 离线高精度 | Full FP32 + attention_rescoring | ~0.025 | 全句 |
| 在线实时 | Full FP32 + ctc_greedy + chunk=16 | ~0.007 | 640ms |
| 端侧部署 | Full INT8 + ctc_greedy + chunk=8 | ~0.005 | 320ms |

## 量化部署命令

```bash
# INT8 动态量化
python benchmark/quantize_and_demo.py

# Docker 容器化推理
docker build -t wenet-inference .
docker run -p 10086:10086 wenet-inference:latest
```
