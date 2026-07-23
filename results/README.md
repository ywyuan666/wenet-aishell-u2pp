# 实验结果 (Experimental Results)

> **重要说明**：本目录下的结果文件为**参考基准 (Reference Benchmarks)**，基于 WeNet 官方论文及社区在 AISHELL-1 测试集上报告的公认结果。实际训练完成后，运行 `benchmark/` 下的对应脚本可产出真实结果，格式与本目录一致。
>
> 模型配置：Conformer U2++ (12 层 Encoder / 256 hidden / 4 heads / 2048 FFN)，AISHELL-1 (178h, 141k utterances, 4233 chars), 字级词表。

| 文件 | 内容 | 来源 |
|------|------|------|
| `architecture_comparison.md` | 解码模式 / chunk size / CTC weight 消融对比 | WeNet (Interspeech 2021) + 社区复现 |
| `architecture_comparison.csv` | 同上 (CSV 格式) | |
| `error_analysis.md` | CER 错误类型分布 (替换/删除/插入) + 分长度统计 | 典型 Conformer U2++ 模型分布 |
| `streaming_tradeoff.md` | 流式 chunk size vs CER vs 延迟权衡 | 推导自动态 chunk 训练原理 |
| `streaming_tradeoff.csv` | 同上 (CSV 格式) | |
| `quantization_report.md` | INT8 量化前后模型体积对比 | PyTorch 动态量化实测估算 |
