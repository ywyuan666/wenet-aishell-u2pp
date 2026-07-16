# Error Analysis Report

## Summary

- Total errors: 647
- CER: 100.00%
- Test utterances analyzed: 50

## Error Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
| substitution | 50 | 7.7% |
| deletion | 597 | 92.3% |
| insertion | 0 | 0.0% |

## Error Examples

### substitution
- ref='情况' hyp='[1]' => '况'->'1'
- ref='整中' hyp='[1]' => '中'->'1'
- ref='资源' hyp='[1]' => '源'->'1'
### deletion
- ref='的情况' missing '情'
- ref='滞的情况' missing '的'
- ref='停滞的情' missing '滞'
### insertion
- (no examples collected)

## CER by Text Length

| Length Range (chars) | CER(%) |
|----------------------|--------|
| 0-9 | 100.0% |
| 10-19 | 100.0% |
| 20-29 | 100.0% |

## Improvement Suggestions

1. **高频替换错误** -> 考虑 LM 浅融合(shallow fusion)或引入 N-gram 语言模型
2. **短文本删除错误** -> 可能是 VAD 或静音段处理问题，考虑调整 blank penalty
3. **插入错误** -> 可能噪声被误识别，考虑数据增强(noise augmentation)或 beam search 消歧
4. **长文本 CER 升高** -> 注意力漂移，考虑 MoChA 或 monotonic attention
5. **整体优化** -> 增加训练数据(全量 AISHELL)、更多 epochs、使用预训练模型

*注：当前模型在 100 utts/5 epochs CPU 上训练，CER 尚未收敛，以上分析展示方法论框架。*