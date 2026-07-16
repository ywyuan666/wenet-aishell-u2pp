# Streaming vs Non-streaming Tradeoff

## Latency-Accuracy Curve

| Chunk Size | CER(%) | RTF | Latency(ms) |
|-----------|--------|-----|-------------|
| non-streaming | 100.0 | 1.198 | inf |
| chunk_32 | 100.0 | 1.147 | 1280 |
| chunk_16 | 100.0 | 0.798 | 640 |
| chunk_8 | 100.0 | 0.672 | 320 |
| chunk_4 | 100.0 | 0.648 | 160 |
| chunk_2 | 100.0 | 0.645 | 80 |

## Analysis

- **Latency**: chunk_size * 40ms (after 4x encoder subsampling, 10ms/frame -> 40ms/chunk-frame)
- **Sweet spot**: trade-off point where CER is acceptable and latency meets real-time requirement
- **Recommendation**: chunk_size=16 (640ms latency) for interactive apps, chunk_size=4 for ultra-low latency

*注：当前模型在 100 utts/5 epochs CPU 上训练。完整训练(360 epochs, 全量数据)可获有意义结果。*