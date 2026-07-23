# Streaming vs Non-streaming Tradeoff

## Full Training Results (AISHELL-1, 360 epochs, GPU/AutoDL)

> ✅ **实际结果** — 动态 chunk 训练在 AISHELL-1 全量训练 (Attention Rescoring 解码) 的 CER 对比。

| Decoding Method | CER(%) | 说明 |
|----------------|--------|------|
| ctc_greedy + non-streaming | 4.73% | 非流式 CTC 贪婪 |
| attention + non-streaming | 4.72% | 非流式 attention |
| **attention_rescoring + non-streaming** | **4.61%** | **✅ 最优方案** |

## Streaming 理论预期

基于 Conformer U2++ 动态 chunk 训练原理，chunk size 与延迟/精度的权衡关系如下：

| Chunk Size | 理论延迟(ms) | 预期 CER 退化 | 推荐场景 |
|-----------|-------------|--------------|---------|
| non-streaming (-1) | inf | 基准 4.61% | 离线转写 |
| chunk=32 | 1280 | +0.3~0.5% | 准实时 |
| chunk=16 | 640 | +0.5~1.0% | ⭐ 实时字幕 |
| chunk=8 | 320 | +1.5~2.5% | 语音指令 |
| chunk=4 | 160 | +2.5~4.0% | 超低延迟 |

**延迟计算**: chunk_size × 4 (encoder subsampling) × 10ms (frame shift) = chunk_size × 40ms

## 推荐配置

| 场景 | Chunk | 延迟 | 预期 CER |
|------|-------|------|---------|
| 离线转写 | -1 (non-streaming) | 无约束 | **4.61%** |
| 实时字幕 | 16 | 640ms | ~5.2% |
| 语音指令 | 8 | 320ms | ~6.5% |

---

*流式结果需在 GPU 上以不同 chunk_size 解码获取。chunk_size 越小 → CER 越高 → 延迟越低。*
