# WeNet Conformer U2++ AISHELL-1 端到端语音识别

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green.svg)]()
[![Arch](https://img.shields.io/badge/Model-Conformer%20U2++-orange.svg)](https://arxiv.org/abs/2005.08100)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)]()

基于 WeNet 框架的 **Conformer U2++** 端到端中文语音识别项目，使用 AISHELL-1 (178h) 数据集，完成从数据处理、模型训练、解码评估、消融分析到 JIT 导出/量化的全流程。

## 架构概览

```
  Audio (16kHz)
       |
  [Fbank 80-dim + CMVN + SpecAugment]
       |
  [Conv2d Subsampling (4x)]
       |
  [Conformer Encoder x12]        ──CTC──→ [ctc_greedy_search]  (1st pass, streaming)
       |                                         |
  [Linear → CTC]                               top-N
                                               |
  [Bi-Transformer Decoder x3+3] ←──Cross-Attn──┘
       |
  [Attention Rescoring]           (2nd pass, re-ranking)
       |
  最终识别文本
```

**核心特性**：
- **Conformer Encoder**：CNN + 多头自注意力 + FFN (Macaron)，12 层，相对位置编码
- **U2++ 两遍解码**：CTC 贪婪搜索（第1遍/流式）+ Attention 重打分（第2遍/高精度）
- **动态 Chunk 训练**：统一流式/非流式建模，通过 `decoding_chunk_size` 控制延迟-精度权衡
- **联合 CTC/Attention Loss** (λ=0.3) + Label Smoothing (lsm=0.1)

## 项目结构

```
project/
├── scripts/                     # 11 个流水线脚本
│   ├── 00_prepare_autodl.sh     # 环境检测 + 依赖安装
│   ├── 01_fetch_wenet.sh        # 克隆 + 安装 WeNet
│   ├── 02_prepare_aishell.sh    # 解压 + 准备 AISHELL-1
│   ├── 03_train_course_fast.sh  # 子集快速训练
│   ├── 03_train_full.sh         # 全量训练
│   ├── 03_finetune_from_ckpt.sh # 从 ckpt 微调
│   ├── 04_decode_eval.sh        # 解码 + CER 评估
│   ├── 05_export_model.sh       # JIT 导出
│   ├── 06_package_runtime_model.sh # 打包部署模型
│   ├── 07_start_runtime_docker.sh  # Docker 推理服务
│   └── 08_collect_results.sh    # 汇总结果
├── tools/                       # Python 工具
├── benchmark/                   # 消融实验 & 分析
│   ├── compare_architectures.py  # 架构/解码/chunk对比
│   ├── error_analysis.py         # CER 错误分类
│   ├── streaming_tradeoff.py     # 流式延迟精度曲线
│   └── quantize_and_demo.py      # 量化 + 推理 Demo
├── env_autodl.sh                # 环境配置 (AutoDL/本地自适应)
├── eval_cer.py                  # CER 评估脚本
├── setup_local.ps1              # Windows 一键环境搭建
├── run_pipeline.ps1             # Windows 流水线运行
└── .github/workflows/ci.yml     # CI/CD 自动化
```

## 快速开始

### Windows 本地
```powershell
# 1. 环境搭建
powershell -File .\setup_local.ps1

# 2. 快速训练 (100 utts, 5 epochs, CPU)
cd wenet\examples\aishell\s0
$env:PYTHONIOENCODING = 'UTF-8'
python ..\..\..\wenet\bin\train.py --config conf/train_cpu_fast.yaml `
    --data_type raw --train_data data/train_subset/data.list `
    --cv_data data/dev/data.list --model_dir exp/u2pp_conformer_course `
    --num_workers 1 --prefetch 2 --device cpu

# 3. CER 评估
cd D:\wenet && python eval_cer.py

# 4. JIT 导出
python wenet/bin/export_jit.py --config exp/u2pp_conformer_course/train.yaml `
    --checkpoint exp/u2pp_conformer_course/epoch_4.pt `
    --output_file exp/u2pp_conformer_course/final.zip
```

### AutoDL (Linux GPU)
```bash
cd /root/autodl-tmp/wenet_aishell_autodl_project
screen -S wenet
bash scripts/00_prepare_autodl.sh
bash scripts/01_fetch_wenet.sh
bash scripts/02_prepare_aishell.sh
bash scripts/03_train_course_fast.sh
bash scripts/04_decode_eval.sh
bash scripts/05_export_model.sh
```

## 消融实验

运行以下命令进行系统的模型对比与分析：

```bash
# 架构/解码/chunk 对比
python benchmark/compare_architectures.py

# CER 错误分析 (按类别、长度)
python benchmark/error_analysis.py

# 流式延迟-精度权衡曲线
python benchmark/streaming_tradeoff.py

# 模型量化 + 推理 Demo
python benchmark/quantize_and_demo.py <audio.wav>
```

生成的报告位于 `results/` 目录。

## 技术亮点

| 类别 | 内容 |
|------|------|
| **平台适配** | 10+ 兼容修复：Windows Bash、Python 3.14、torch.jit、deepspeed 可选导入、torchaudio→soundfile |
| **CPU/GPU 自适应** | 自动检测 CUDA，CPU 模式自动调整 workers/batch/nj |
| **流式/非流式统一** | 动态 chunk 训练 + 可调 `decoding_chunk_size` |
| **模型优化** | INT8 动态量化 (3-4x 压缩) + JIT TorchScript 导出 |
| **工程化** | CI/CD 自动化测试 + PowerShell/Git Bash 双入口 |

## 兼容性修复清单

| 问题 | 修复方案 |
|------|----------|
| Python 3.14 移除 `__annotations__` | monkey-patch `torch.jit._check` |
| torch.jit.script() 失败 | 注入 `__annotations__` 到 nn.Module |
| deepspeed 不可安装 | try/except 可选导入 |
| torchaudio 无法加载音频 | soundfile 回退方案 |
| torchrun 不支持 Windows | 单进程直接调用 train.py |
| MSYS 路径不兼容 | sed 替换为 Windows 路径 |
| GBK 编码读取中文 | FileOpener encoding='utf-8' |
| lscpu 命令不存在 Windows | platform.system() 检测 |

## 参考资料

- [WeNet: Production First and Production Ready End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2102.01547)
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [U2++: Unified Streaming and Non-streaming Two-pass End-to-End Model](https://arxiv.org/abs/2106.05633)
- [AISHELL-1: An Open-Source Mandarin Speech Corpus](https://arxiv.org/abs/1709.05522)

## 简历写法

```text
基于 WeNet 的 Conformer U2++ 端到端语音识别系统
- 基于 AISHELL-1 (178h) 构建完整 ASR pipeline：数据准备(141k utts)、CMVN、字级词表(4233 chars)、
  Conformer U2++ 训练、CTC/Attention 联合解码、JIT 导出与 INT8 量化。
- 系统消融实验：对比 CTC greedy/prefix beam/Attention rescoring 三种解码模式，
  分析 chunk size(4/8/16/32/-1) 对流式延迟与 CER 的权衡关系，输出错误分类报告(替换/删除/插入)。
- 解决 10+ 跨平台兼容问题：Windows Git Bash 适配、Python 3.14 torch.jit 修复、
  deepspeed 可选导入、torchaudio→soundfile 回退方案、GBK 编码修复。
- 完成模型部署：JIT TorchScript 导出(fina.zip 178MB)、INT8 量化(3-4x 压缩)、
  Docker/WebSocket 推理服务入口，支持流式(chunk=4/8/16)与非流式(chunk=-1)两种模式。
```
