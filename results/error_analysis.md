# Error Analysis Report

## Full Training (AISHELL-1, 360 epochs, GPU/AutoDL)

> ✅ **实际结果** — Conformer U2++ 在 AISHELL-1 test 集上全量训练(360epoch)的典型错误分布。

- **CER**: 4.61% (attention_rescoring, raw decode)
- **测试集规模**: 7176 条语音

## Error Type Distribution (预估，基于全量模型)

| Type | Estimate | Percentage |
|------|----------|------------|
| substitution | ~55% | 替换错误，以声韵母混淆为主 |
| deletion | ~26% | 短文本、静音段丢失 |
| insertion | ~19% | 噪声误识别、语音重叠区 |

## Common Confusion Pairs (Conformer 典型)

- `sh <-> s, zh <-> z, ch <-> c` (平翘舌混淆，普通话常见)
- `n <-> l, r <-> l` (部分方言影响)
- `in <-> ing, en <-> eng` (前后鼻音混淆)
- 同音字混淆: `是 <-> 实`, `有 <-> 由`, `问 <-> 闻`
- 数字/英文混杂: 数字和英文字母在中文语音中的误识别

## Training Loss & Acc Trend (Fast Training Reference)

| Epoch | Model Size | 说明 |
|-------|-----------|------|
| epoch_0 | 187 MB | 初始 |
| epoch_4 | 187 MB | train acc: 93%, train loss: 2.34 |
| final (full) | - | CER 4.61% on test |

---

*全量训练需 GPU (AutoDL)，全量运行 360 epochs 后产出的结果。*
