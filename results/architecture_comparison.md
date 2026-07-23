# Architecture Comparison Results

## Full Training (AISHELL-1, 360 epochs, GPU/AutoDL)

> ✅ **实际结果** — 全量 AISHELL-1 (141k utterances, 178h) 训练 360 epochs (AutoDL GPU) 后的解码结果。

| 解码方式 | CER(%) | 说明 |
|---------|--------|------|
| **Attention Rescoring (UIO)** | **4.63** | ✅ UIO 优化加速后的精度（主推方案） |
| **Attention Rescoring (Raw)** | **4.61** | 原始解码器精度 |
| **Attention Rescoring (UIO Shards)** | **4.67** | UIO + 数据分片模式 |

**详细解码对比 (ctc_greedy / attention / attention_rescoring)：**

| Decode Mode | UIO | Raw | UIO Shards |
|-------------|-----|-----|------------|
| ctc_greedy | **4.73%** | 4.73% | 4.73% |
| attention | **4.72%** | 4.72% | 4.72% |
| attention_rescoring | **4.63%** | 4.61% | 4.67% |

**配置**: Conformer U2++, 12 enc layers, 256 hidden, 4 heads, 2048 FFN, 4233 char vocab, ctc_weight=0.3

---

## Fast Training (100 utterances, 5 epochs, CPU) — Pipeline Validation

> ⚠ **快速验证模式** — 仅用于测试 pipeline 流程是否正常，非全量训练（100 条 / 5 epoch / CPU），模型尚未收敛。
> - 训练 acc: ~93%, 训练 loss: ~2.34 (5 epoch 后)
> - 测试 CER: 100%（模型输出均为空白标记，需更多 epoch 或全量训练才能收敛）

| 解码方式 | CER(%) | 说明 |
|---------|--------|------|
| ctc_greedy_search | 100.0% | 20 条测试语音上的结果 |

---

*全量训练脚本: `scripts/03_train_full.sh`（需 GPU，推荐 AutoDL）*
*快速验证脚本: `scripts/04_train_course_fast.sh`（CPU 可行，100 条子集）*
