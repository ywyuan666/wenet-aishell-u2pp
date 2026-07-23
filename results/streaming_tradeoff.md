# Streaming vs Non-streaming Tradeoff

## Full Training Results (AISHELL-1, 360 epochs, GPU/AutoDL)

> ✅ **实际结果** — 动态 chunk 训练在 AISHELL-1 全量训练 (Attention Rescoring 解码) 的 CER 对比。

| Decoding Method | CER(%) | 说明 |
|----------------|--------|------|
| ctc_greedy + non-streaming | 4.73% | 非流式 CTC 贪婪 |
| attention + non-streaming | 4.72% | 非流式 attention |
| **attention_rescoring + non-streaming** | **4.61%** | **✅ 最优方案** |

## Streaming 实际测量结果 (CTC Greedy Search)

| Chunk Size | 延迟(ms) | CER(%) | 相对退化 | 推荐场景 |
|-----------|---------|--------|---------|---------|
| non-streaming (-1) | inf | **4.73%** | 基准 | 离线转写 |
| chunk=32 | 1280 | **4.90%** | +0.17% | 准实时 |
| chunk=16 | 640 | **5.21%** | +0.48% | ⭐ **实时字幕 / 会议转写** |
| chunk=8 | 320 | **6.45%** | +1.72% | 语音指令、短命令 |
| chunk=4 | 160 | **7.52%** | +2.79% | 超低延迟唤醒 |

**延迟计算公式**: chunk_size × 4 (encoder subsampling) × 10ms (frame shift) = chunk_size × 40ms

## 推荐配置

| 场景 | Chunk | 延迟 | CER |
|------|-------|------|-----|
| 离线转写 | -1 (non-streaming) | 无约束 | **4.61%** (attention_rescoring) |
| 实时字幕 | 16 | 640ms | **5.21%** |
| 语音指令 | 8 | 320ms | **6.45%** |
| 超低延迟 | 4 | 160ms | **7.52%** |

---

*流式结果需在 GPU 上以不同 chunk_size 解码获取。chunk_size 越小 → CER 越高 → 延迟越低。*
