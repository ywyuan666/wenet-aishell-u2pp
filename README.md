# WeNet AISHELL AutoDL Project

这个项目包用于在 AutoDL RTX 5090 实例上跑一个完整的 WeNet 中文语音识别课程大作业：AISHELL-1 数据准备、WeNet Conformer/U2++ 训练、CER 评测、checkpoint averaging、JIT 模型导出、runtime 模型打包和 Docker/WebSocket 部署入口。

项目包本身不包含 AISHELL 数据和 WeNet 官方大仓库。上传后脚本会把 WeNet 官方代码拉到 `/root/autodl-tmp/wenet`，所有大文件、缓存、数据集、checkpoint 都放到 `/root/autodl-tmp`，避免撑爆系统盘。

## AutoDL 目录策略

你的机器配置：CPU 25 核，内存 90 GB，RTX 5090，系统盘 30 GB，数据盘 `/root/autodl-tmp` 200 GB。

默认路径：

```text
/root/autodl-tmp/wenet_aishell_autodl_project   # 本项目脚本（上传后路径）
/root/autodl-tmp/wenet                          # WeNet 官方源码
/root/autodl-tmp/datasets/aishell               # AISHELL-1 数据（从 data/*.tgz 解压）
/root/autodl-tmp/wenet_runtime_models           # 导出的部署模型
/root/autodl-tmp/.cache                         # pip/torch/huggingface/modelscope 缓存
```

### 本地项目结构

```text
project/            # 项目根目录
├── data/           # AISHELL-1 数据压缩包 (data_aishell.tgz, resource_aishell.tgz)
├── scripts/        # 流水线脚本（11 个 .sh）
├── tools/          # Python 工具脚本
├── env_autodl.sh   # 环境变量（自动适配 AutoDL / 本地）
├── run_fast_pipeline.sh  # 一键运行快速流水线
└── README.md
```

## 上传后快速开始

把项目文件夹打包上传到 AutoDL 的 `/root/autodl-tmp`，然后执行：

```bash
cd /root/autodl-tmp
unzip -o wenet_aishell_autodl_project.zip -d wenet_aishell_autodl_project
cd wenet_aishell_autodl_project
source ./env_autodl.sh    # 首次手动加载环境变量
screen -S wenet

bash scripts/00_prepare_autodl.sh
bash scripts/01_fetch_wenet.sh
bash scripts/02_prepare_aishell.sh
```

先跑小规模训练闭环：

```bash
bash scripts/03_train_course_fast.sh
bash scripts/04_decode_eval.sh
bash scripts/05_export_model.sh
bash scripts/06_package_runtime_model.sh
```

确认 pipeline 没问题后，再跑全量训练：

```bash
EXP_DIR=exp/u2pp_conformer_full TRAIN_SET=train bash scripts/03_train_full.sh
EXP_DIR=exp/u2pp_conformer_full AVERAGE_NUM=10 bash scripts/04_decode_eval.sh
EXP_DIR=exp/u2pp_conformer_full AVERAGE_NUM=10 bash scripts/05_export_model.sh
EXP_DIR=exp/u2pp_conformer_full bash scripts/06_package_runtime_model.sh
```

## 脚本说明

```text
env_autodl.sh                      统一路径、缓存、训练参数
scripts/00_prepare_autodl.sh       检查 CUDA/磁盘，安装基础 Python 依赖
scripts/01_fetch_wenet.sh          拉取 WeNet 官方源码并安装，不覆盖 torch
scripts/02_prepare_aishell.sh      下载并准备 AISHELL-1
scripts/03_train_course_fast.sh    用训练子集快速跑通课程闭环
scripts/03_train_full.sh           全量 AISHELL-1 训练
scripts/03_finetune_from_ckpt.sh   从已有 checkpoint 微调
scripts/04_decode_eval.sh          解码并计算 CER
scripts/05_export_model.sh         导出 final.zip
scripts/06_package_runtime_model.sh 打包 final.zip + units.txt
scripts/07_start_runtime_docker.sh 启动 Docker/WebSocket 推理服务
tools/*.py                         子集、jsonl 转换、结果汇总工具
```

## 自有数据微调

自有数据整理成 jsonl：

```json
{"key":"utt_000001","wav_path_abs":"/root/autodl-tmp/my_data/wavs/utt_000001.wav","text":"今天天气很好"}
```

转换成 WeNet data.list：

```bash
python tools/jsonl_to_wenet_data.py \
  --jsonl /root/autodl-tmp/my_data/train.jsonl \
  --out-dir /root/autodl-tmp/wenet/examples/aishell/s0/data/my_train
```

从已有 checkpoint 微调：

```bash
TRAIN_SET=my_train \
EXP_DIR=exp/my_finetune \
CHECKPOINT=/root/autodl-tmp/wenet/examples/aishell/s0/exp/u2pp_conformer_full/avg_10.pt \
bash scripts/03_finetune_from_ckpt.sh
```

注意：如果自有文本出现 AISHELL 字表没有的字符，需要重建字表，否则会有 OOV 风险。

## 简历写法

```text
基于 WeNet 的中文端到端语音识别系统
- 基于 AISHELL-1 构建完整 ASR 训练 pipeline，完成数据准备、CMVN 统计、字级词表构建、data.list 生成与模型训练。
- 使用 WeNet Conformer/U2++ 架构进行端到端建模，支持 CTC、Attention、Attention Rescoring 等多种解码方式，并以 CER 作为核心评估指标。
- 对比流式与非流式识别配置，分析 chunk size、beam size、checkpoint averaging 对识别性能的影响。
- 完成模型导出与部署验证，将训练模型导出为 final.zip，并基于 WeNet runtime/WebSocket 服务实现语音识别推理。
```

## 常见注意事项

1. RTX 5090 上不要乱装 torch。AutoDL 镜像里的 PyTorch/CUDA 通常已经适配显卡，本项目安装 WeNet 时默认不覆盖 torch/torchaudio。
2. 系统盘小，所有数据、缓存、checkpoint 都放 `/root/autodl-tmp`。
3. 长时间训练用 `screen -S wenet`，断开用 `Ctrl-a d`，恢复用 `screen -r wenet`。
4. fast 版本只证明 pipeline 完整，不追求低 CER；简历和报告建议最终跑 full 版本。