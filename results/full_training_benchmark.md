# Full Training Benchmark — Conformer U2++ on AISHELL-1

> **训练平台**: AutoDL GPU  
> **训练数据**: AISHELL-1 (141,600 utterances, ~178 hours, 4233 chars)  
> **训练轮数**: 360 epochs  
> **模型架构**: Conformer Encoder (12层, 256 hidden, 4 heads, 2048 FFN) + Bi-Transformer Decoder (各3层)  
> **损失函数**: CTC/Attention 联合损失 (λ=0.3) + Label Smoothing (0.1)  
> **解码策略**: U2++ 两遍解码 (CTC greedy 第一遍 + Attention rescoring 第二遍)

---

## 解码结果对比

| Decode Mode | Raw CER | UIO CER | UIO+Shards | 说明 |
|-------------|---------|---------|-----------|------|
| ctc_greedy_search | **4.73%** | **4.73%** | **4.73%** | ✅ 最快解码，适合纯流式场景 |
| attention | **4.72%** | **4.72%** | **4.72%** | 使用 attention decoder 自回归解码 |
| **attention_rescoring** | **4.61%** | **4.63%** | **4.67%** | ✅ **最优方案**：CTC 第一遍 + Attention 重打分 |

**结论**: attention_rescoring (raw) 获得最佳 CER 4.61%，相比 CTC greedy 降低 0.12%。

### 各解码方式详解

| 方式 | 特点 | 推荐场景 |
|------|------|---------|
| **CTC Greedy Search** | 每帧 argmax，无注意力交互，速度最快 | 实时字幕、语音指令 |
| **Attention Decoder** | Transformer 自回归解码，精度更高 | 离线转写（精度折中） |
| **Attention Rescoring** | CTC 生成 N-best 候选 → Attention 重打分排序 | ✅ **离线转写（最优精度）** |

---

## 流式延迟-精度权衡（全量训练）

| Chunk Size | 帧数 | 理论延迟 | 预期 CER | 推荐场景 |
|-----------|------|---------|---------|---------|
| -1 (non-streaming) | 全句 | inf | **4.61%** | 离线转写、高精度 |
| 32 | 32 | 1280ms | ~4.9% | 准实时 |
| **16** | **16** | **640ms** | **~5.2%** | ⭐ **实时字幕 / 会议转写** |
| 8 | 8 | 320ms | ~6.5% | 语音指令、短命令 |
| **4** | **4** | **160ms** | **~7.5%** | 按键说话、超低延迟唤醒 |
| 2 | 2 | 80ms | ~9.0% | 极限低延迟场景 |

**延迟计算公式**: 理论延迟 = chunk_size × 4 (subsampling) × 10ms (frame_shift) = chunk_size × 40ms

---

## 训练过程

| 阶段 | 说明 |
|------|------|
| 初始模型 | ~187 MB (FP32 checkpoint) |
| 训练集 | AISHELL-1 train (141,600 utts), 178h |
| 验证集 | AISHELL-1 dev (17,000+ utts) |
| 测试集 | AISHELL-1 test (7,176 utts) |
| 训练时长 | ~若干天 (GPU, 360 epochs) |

**对比参考**: WeNet 官方 Interspeech 2021 论文报道 AISHELL-1 test CER 为 4.7%~5.0%，本项目结果 4.61% 处于合理范围。

---

## 模型量化 (Fast Training Checkpoint 实测)

| 版本 | 大小 | 压缩比 |
|------|------|--------|
| FP32 Checkpoint | 178.6 MB | - |
| JIT TorchScript | 178.6 MB | 1.00x |
| **INT8 动态量化** | **64.9 MB** | **2.75x** |

> 全量训练模型预估: FP32 ~42.3 MB, INT8 ~11.5 MB (~3.7x 压缩)

---

*完整实验脚本见 `scripts/` 目录，消融实验分析见 `benchmark/` 目录。*
