#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 WeNet/AISHELL/U2++ 技术学习笔记 PDF"""

import os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, ListFlowable, ListItem,
    KeepTogether, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ═══════════════════════════════════════════
# 字体注册（确保中文不乱码）
# ═══════════════════════════════════════════
FONT_DIR = "C:/Windows/Fonts"
YAHEI = os.path.join(FONT_DIR, "msyh.ttc")
YAHEI_BOLD = os.path.join(FONT_DIR, "msyhbd.ttc")
try:
    pdfmetrics.registerFont(TTFont("YaHei", YAHEI))
    pdfmetrics.registerFont(TTFont("YaHei-Bold", YAHEI_BOLD))
    FONT_NAME = "YaHei"
    FONT_BOLD = "YaHei-Bold"
    print(f"[OK] 已注册字体: {YAHEI}")
except Exception as e:
    print(f"[WARN] msyh.ttc 注册失败 ({e}), 尝试 simhei.ttf")
    try:
        pdfmetrics.registerFont(TTFont("YaHei", os.path.join(FONT_DIR, "simhei.ttf")))
        FONT_NAME = "YaHei"
        FONT_BOLD = "YaHei"
        print("[OK] 已注册 simhei.ttf")
    except Exception as e2:
        print(f"[ERROR] 无法注册中文字体: {e2}")
        sys.exit(1)

# ═══════════════════════════════════════════
# 输出路径
# ═══════════════════════════════════════════
OUTPUT_DIR = os.path.expanduser("~/Desktop/wenet/aishell/u2++")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "WeNet_U2++_技术总结笔记.pdf")

# ═══════════════════════════════════════════
# 颜色方案
# ═══════════════════════════════════════════
C_PRIMARY   = HexColor("#1a73e8")   # 蓝色主色
C_ACCENT    = HexColor("#e37400")   # 橙色强调
C_DARK      = HexColor("#202124")
C_BODY      = HexColor("#3c4043")
C_MUTED     = HexColor("#5f6368")
C_LIGHT_BG  = HexColor("#f8f9fa")
C_DIVIDER   = HexColor("#dadce0")
C_GREEN     = HexColor("#188038")
C_RED       = HexColor("#d93025")
C_CODE_BG   = HexColor("#f1f3f4")

# ═══════════════════════════════════════════
# 样式定义
# ═══════════════════════════════════════════
styles = getSampleStyleSheet()

def make_style(name, parent='Normal', **kwargs):
    base = {
        'fontName': FONT_NAME,
        'textColor': C_BODY,
        'leading': 22,
        'spaceAfter': 6,
        'wordWrap': 'CJK',
    }
    base.update(kwargs)
    return ParagraphStyle(name, parent=styles[parent], **base)

s_cover_title = make_style('CoverTitle', fontSize=28, fontName=FONT_BOLD,
                           textColor=white, alignment=TA_CENTER, leading=38,
                           spaceAfter=0)
s_cover_sub = make_style('CoverSub', fontSize=14, textColor=HexColor("#e8eaed"),
                         alignment=TA_CENTER, leading=22, spaceAfter=0)
s_h1 = make_style('H1', fontSize=20, fontName=FONT_BOLD, textColor=C_PRIMARY,
                   leading=32, spaceBefore=24, spaceAfter=12)
s_h2 = make_style('H2', fontSize=16, fontName=FONT_BOLD, textColor=C_DARK,
                   leading=28, spaceBefore=18, spaceAfter=8)
s_h3 = make_style('H3', fontSize=13, fontName=FONT_BOLD, textColor=C_DARK,
                   leading=22, spaceBefore=12, spaceAfter=6)
s_body = make_style('Body', fontSize=10.5, leading=20, spaceAfter=4,
                     alignment=TA_JUSTIFY)
s_body_indent = make_style('BodyIndent', fontSize=10.5, leading=20,
                            leftIndent=20, spaceAfter=3)
s_qa_q = make_style('QA_Q', fontSize=11, fontName=FONT_BOLD, textColor=C_PRIMARY,
                     leading=22, spaceBefore=10, spaceAfter=2)
s_qa_a = make_style('QA_A', fontSize=10.5, leading=20, leftIndent=16,
                     spaceAfter=8, textColor=C_BODY)
s_tip = make_style('Tip', fontSize=10, textColor=C_RED, leading=18,
                    leftIndent=10, spaceAfter=2, fontName=FONT_BOLD)
s_note = make_style('Note', fontSize=10, textColor=C_MUTED, leading=18,
                     leftIndent=10, spaceAfter=2)
s_code = make_style('Code', fontSize=8.5, fontName='Courier', leading=14,
                     leftIndent=14, spaceAfter=2, textColor=C_DARK,
                     backColor=C_CODE_BG)
s_toc = make_style('TOC', fontSize=11, leading=24, leftIndent=10,
                    textColor=C_DARK)
s_toc_sub = make_style('TOCSub', fontSize=10, leading=20, leftIndent=28,
                        textColor=C_MUTED)

# ═══════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════
def divider():
    return HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER,
                       spaceBefore=8, spaceAfter=8)

def bullet(text, style=s_body_indent):
    """带圆点的列表项"""
    return Paragraph(f"<bullet>&bull;</bullet> {text}", style)

def numbered(num, text, style=s_body_indent):
    """带编号的列表项"""
    return Paragraph(f"<b>{num}.</b> {text}", style)

def qa_block(q, a, extra=None):
    """问答块"""
    items = [Paragraph(f"<b>Q: {q}</b>", s_qa_q),
             Paragraph(f"A: {a}", s_qa_a)]
    if extra:
        items.append(Paragraph(f"💡 <i>{extra}</i>", s_note))
    return items

def tip_block(text):
    return Paragraph(f"⚠️ <b>技术提示:</b> {text}", s_tip)

def note_block(text):
    return Paragraph(f"📌 <i>{text}</i>", s_note)

def code_block(text):
    return Paragraph(f"<font face='Courier' size='8'>{text}</font>", s_code)

def highlight_box(items):
    """带背景色的高亮框"""
    data = [[item] for item in items]
    t = Table(data, colWidths=[460])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    return t

# ═══════════════════════════════════════════
# 构建文档
# ═══════════════════════════════════════════
doc = SimpleDocTemplate(
    OUTPUT_PATH, pagesize=A4,
    leftMargin=25*mm, rightMargin=25*mm,
    topMargin=20*mm, bottomMargin=20*mm,
    title="WeNet/U2++ 技术总结笔记",
    author="AI Assistant"
)

story = []

# ════════════════════════════════════
# 封面
# ════════════════════════════════════
cover_table_data = [[
    Paragraph("<br/><br/><br/><br/>", s_body),
    Paragraph("WENET / AISHELL / U2++<br/>技术总结笔记", s_cover_title),
    Paragraph("<br/>语音识别 | 端到端 | 流式解码<br/>Conformer | CTC | Attention Rescoring", s_cover_sub),
    Paragraph("<br/><br/><br/>版本 1.0 · 2026 年 7 月", make_style('v', fontSize=10,
              textColor=HexColor("#9aa0a6"), alignment=TA_CENTER)),
    Paragraph("<br/><br/>", s_body),
]]
cover = Table(cover_table_data, colWidths=[460])
cover.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('LEFTPADDING', (0,0), (-1,-1), 20),
    ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ('TOPPADDING', (0,0), (-1,-1), 0),
    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ('ROUNDEDCORNERS', [8,8,8,8]),
]))
story.append(Spacer(1, 20))
story.append(cover)
story.append(Spacer(1, 30))

# ════════════════════════════════════
# 目录
# ════════════════════════════════════
story.append(Paragraph("目 录", s_h1))
story.append(divider())
toc_items = [
    ("一、项目概览", [
        "项目一句话总结",
        "一分钟速览",
        "三分钟详解"
    ]),
    ("二、核心架构深度解析", [
        "U2++ 整体架构",
        "Conformer Encoder 详解",
        "CTC 解码",
        "Attention Decoder 与 Rescoring",
        "训练策略与 Loss 设计"
    ]),
    ("三、核心知识整理", [
        "基础概念（8 条）",
        "原理深挖（10 条）",
        "工程实践（8 条）",
        "对比分析（6 条）",
        "开放思考（6 条）"
    ]),
    ("四、技术深挖方向", [
        "模型结构深挖",
        "流式与非流式",
        "训练技巧与调参",
        "部署与优化",
        "前沿方向"
    ]),
    ("五、技术深挖进阶准备", [
        "从零推导 Conformer",
        "CTC 对齐原理",
        "RNN-T vs Attention vs CTC",
        "流式解码的工程实现",
        "模型量化与加速"
    ]),
    ("六、推荐阅读与资料", []),
]
for title, subs in toc_items:
    story.append(Paragraph(title, s_toc))
    for sub in subs:
        story.append(Paragraph(sub, s_toc_sub))
story.append(PageBreak())

# ════════════════════════════════════
# 一、项目概览
# ════════════════════════════════════
story.append(Paragraph("一、项目概览", s_h1))
story.append(divider())

story.append(Paragraph("1.1  项目一句话总结", s_h2))
story.append(highlight_box([
    Paragraph("我基于 <b>WeNet 开源框架</b>，采用 <b>Conformer U2++ 架构</b>，在 <b>AISHELL-1</b> 中文普通话数据集上，"
              "搭建了一套<b>端到端语音识别</b>系统。<b>支持流式与</b>非流式两遍解码，结合 <b>CTC 和 Attention Rescoring</b> "
              "在单模型上兼顾<b>低延迟和</b>高精度，<b>CER 达到 4.61%</b>，RTF 低至 0.008。", s_body)
]))
story.append(Spacer(1, 6))

story.append(Paragraph("1.2  一分钟速览", s_h2))
story.append(Paragraph(
    "我基于 <b>WeNet 框架</b>和 <b>AISHELL-1 中文普通话数据集</b>，实现了业界主流的 <b>Conformer U2++ 端到端语音识别模型</b>。"
    "核心架构分为三部分：<b>Conformer Encoder（12 层）</b>提取声学特征、<b>CTC 分支</b>做第一遍流式解码获得低延迟初选结果、"
    "<b>Attention Decoder（6 层 Transformer）</b>做第二遍 Attention Rescoring 重打分，最终 CER 达到 <b>4.61%</b>。"
    "项目有完整的数据处理→模型训练→解码评估→性能基准测试链路，支持 5 种解码模式一键对比，"
    "并集成了 <b>Gradio Web UI</b>、<b>HuggingFace 模型格式</b>、<b>CI 自动化测试</b>等工程实践。", s_body))
story.append(Spacer(1, 8))

story.append(Paragraph("1.3  三分钟详解", s_h2))
story.append(Paragraph(
    "本项目基于 <b>WeNet 框架</b>在 <b>AISHELL-1 中文预测数据集</b>上实现 <b>U2++ 架构</b>的语音识别系统。"
    "下面分四个层面来介绍：", s_body))
story.append(Spacer(1, 4))

story.append(Paragraph("<b>【背景动机】</b>", s_h3))
story.append(Paragraph(
    "工业级语音识别面临的核心矛盾是 <b>低延迟</b> 与 <b>高精度</b> 的权衡。传统的两阶段方案需要分别部署流式模型和高精度模型，"
    "维护成本高。U2++ 通过统一模型内部的 <b>CTC 第一遍流式解码</b> + <b>Attention 第二遍重打分</b>，在单个模型中同时满足两种需求，"
    "这正是 WeNet 方案的核心优势。", s_body))
story.append(Spacer(1, 4))

story.append(Paragraph("<b>【技术方案】</b>", s_h3))
story.append(Paragraph(
    "模型主体是 <b>Conformer U2++</b> 架构：<br/>"
    "① <b>输入处理</b>：80 维 FBank 特征 + CMVN 归一化 + SpecAugment 数据增强；<br/>"
    "② <b>Encoder</b>：12 层 Conformer 块，每块包含 <b>Feed-Forward Macaron</b>（两个半残差 FFN）、"
    "<b>Multi-Head Self-Attention</b>（相对位置编码）和 <b>Conv Module</b>（逐点卷积+深度可分离卷积+GLU 激活），"
    "CNN 和 Self-Attention 的优势互补；<br/>"
    "③ <b>第一遍解码</b>：CTC 分支输出 logits，训练时通过 CTC Loss 监督，推理时做 <b>greedy search</b> "
    "或 <b>beam search</b> 产生候选序列，流式解码时 <b>chunk=16</b> 达到 640ms 延迟；<br/>"
    "④ <b>第二遍解码</b>：<b>Bi-Transformer Decoder</b>（3 层自注意力 + 3 层交叉注意力）对 CTC 候选进行 "
    "<b>Attention Rescoring</b>，重新计算得分后选取最优路径，CER 从 5.21% 降至 4.61%。", s_body))
story.append(Spacer(1, 4))

story.append(Paragraph("<b>【工程亮点】</b>", s_h3))
story.append(bullet("完整的 <b>CI/CD 流水线</b>：GitHub Actions 自动跑 pytest（CER 计算、数据加载测试），覆盖 Ubuntu/Windows 双平台"))
story.append(bullet("<b>HuggingFace 生态集成</b>：提供 <b>from_pretrained / save_pretrained</b> 接口，支持 config.json + safetensors 格式，一键推送到 HF Hub"))
story.append(bullet("<b>Gradio Web UI</b>：浏览器麦克风录音 + 文件上传 + 5 种解码模式一键对比，支持 --share 生成公网链接分享"))
story.append(bullet("<b>可视化 Benchmark</b>：自动生成架构对比图、流式权衡曲线、RTF 对比图等 4 张出版级图表"))
story.append(bullet("<b>实验追踪</b>：集成 W&B 和 MLflow，记录超参数、Loss 曲线、CER 指标、系统资源"))
story.append(Spacer(1, 4))

story.append(Paragraph("<b>【量化结果】</b>", s_h3))
story.append(Paragraph(
    "在 AISHELL-1 测试集上，Attention Rescoring 模式取得 <b>CER 4.61%</b>（非流式），"
    "CTC Greedy 模式取得 <b>CER 5.21%</b>（chunk=16 流式）。RTF（Real-Time Factor）低至 <b>0.008</b>，"
    "意味着一秒钟音频只需 8ms 即可完成识别，具有充足的实时推理能力。", s_body))
story.append(PageBreak())

# ════════════════════════════════════
# 二、核心架构深度解析
# ════════════════════════════════════
story.append(Paragraph("二、核心架构深度解析", s_h1))
story.append(divider())

story.append(Paragraph("2.1  U2++ 整体架构（核心必知）", s_h2))
story.append(Paragraph(
    "U2++（Unified Two-pass ++）是 <b>统一的两遍解码端到端语音识别框架</b>。核心思想是："
    "在<b>单个模型中同时训练 CTC 分支和 Attention 分支</b>，通过<b>多任务学习</b>共享 Encoder 参数。"
    "推理时第一遍用 CTC 做<b>流式解码</b>（低延迟），第二遍用 Attention Decoder 做<b>重打分 Rescoring</b>（高精度）。", s_body))
story.append(Spacer(1, 4))

arch_items = [
    "<b>输入层</b>: 16kHz 音频 → FBank 80 维（25ms 窗，10ms 移）→ CMVN 全局归一化 → SpecAugment（2 个频率掩码 + 2 个时间掩码）",
    "<b>Conv2D Subsampling</b>: 2 层 Conv2D（kernel 3×3, stride 2）→ 时间维度 4× 下采样 → 256 维投影",
    "<b>Conformer Encoder ×12</b>: 堆叠 12 个 Conformer 块，每块 4 个子模块：FFN(1) → Self-Attn → Conv → FFN(2)，中间有残差连接和 LayerNorm。相对位置编码而非绝对位置编码，支持变长输入",
    "<b>CTC 分支</b>: Encoder 输出 → Linear(256→4233) → softmax → CTC Loss（训练）+ CTC Greedy Search / Beam Search（推理）。4233 对应 AISHELL-1 的 4233 个汉字/符号 tokens",
    "<b>Attention Decoder</b>: 6 层 Transformer Decoder（3 层自注意力 + 3 层交叉注意力），输入为 CTC 候选序列的 embedding，通过交叉注意力关注 Encoder 输出。训练时用 Teacher Forcing，推理时对 CTC 候选做 Rescoring",
    "<b>输出融合</b>: 最终得分 = λ × CTC_score + (1-λ) × Att_score，λ=0.3（可调）。Rescoring 后选择最优路径作为最终识别结果",
]
for item in arch_items:
    story.append(bullet(item))
story.append(Spacer(1, 6))

story.append(Paragraph("2.2  Conformer Encoder 详解", s_h2))
story.append(Paragraph(
    "Conformer 由 Google 在 2020 年提出（<i>Conformer: Convolution-augmented Transformer for Speech Recognition</i>），"
    "核心创新是在 Transformer 中引入 <b>卷积模块</b>，弥补 Self-Attention 对局部模式的建模不足。", s_body))
story.append(Spacer(1, 4))
story.append(Paragraph("<b>Conformer 块结构（Macaron Net 风格）</b>", s_h3))
story.append(Paragraph(
    "每个 Conformer 块包含 4 个核心模块：", s_body))
story.append(bullet("<b>Feed-Forward Module (前半 Macaron)</b>: Linear(256→1024) → Swish → Dropout → Linear(1024→256)。采用 Macaron Net 的半残差结构，即 ½ × FFN 残差"))
story.append(bullet("<b>Self-Attention Module</b>: Multi-Head Self-Attention (4 heads, d_k=256/4=64) + 相对位置编码（Relative Positional Encoding）。相对位置编码对变长输入和流式场景至关重要"))
story.append(bullet("<b>Conv Module</b>（Conformer 核心创新）: LayerNorm → Pointwise Conv(256→512) → GLU → 1D Depthwise Conv(kernel=31) → BatchNorm → Swish → Pointwise Conv(512→256) → Dropout。左侧卷积（kernel size 31）使模型能同时看到前后 15 帧"))
story.append(bullet("<b>Feed-Forward Module (后半 Macaron)</b>: 结构与前半相同"))
story.append(bullet("<b>最终</b>: LayerNorm + Dropout 作为块的输出"))
story.append(Spacer(1, 4))

story.append(Paragraph("<b>为什么 Conformer 适合语音识别?</b>", s_h3))
story.append(bullet("<b>Self-Attention</b> 擅长捕捉长距离依赖（如整个句子的语义信息）"))
story.append(bullet("<b>卷积</b> 擅长捕捉局部模式（如音素间的过渡）"))
story.append(bullet("<b>Macaron FFN</b> 比标准 Transformer 的 Post-LN 更稳定，训练更深网络"))
story.append(bullet("<b>相对位置编码</b> 支持流式输入（不需要知道句子长度）"))
story.append(Spacer(1, 6))

story.append(Paragraph("2.3  CTC 解码（第一遍）", s_h2))
story.append(Paragraph(
    "CTC（Connectionist Temporal Classification）允许输入序列长度大于输出序列长度，"
    "通过引入 <b>blank 符号</b> 处理对齐问题。训练时穷举所有可能的对齐路径并求和概率，"
    "推理时使用 <b>prefix beam search</b> 或 <b>CTC greedy search</b> 找出最优路径。", s_body))
story.append(Spacer(1, 2))
story.append(bullet("<b>CTC Greedy Search</b>: 每帧取最大概率 token → 合并重复（collapse）→ 去除 blank。最简单但最快速"))
story.append(bullet("<b>CTC Prefix Beam Search</b>: 维护 beam 个候选 prefix，每帧扩展，考虑 blank/non-blank 两种路径"))
story.append(bullet("<b>流式 CTC 解码</b>: 每次处理一个 chunk（如 16 帧），结合 EMA（ Exponential Moving Average）前缀缓存，实现帧同步输出"))
story.append(Spacer(1, 6))

story.append(Paragraph("2.4  Attention Decoder 与 Rescoring（第二遍）", s_h2))
story.append(Paragraph(
    "Attention Decoder 本质是 <b>Transformer Decoder</b>，但第一遍的 CTC 路径已经做了帧对齐，Decoder 不需要再逐帧对齐。"
    "Rescoring 的具体流程：", s_body))
story.append(bullet("CTC 第一遍输出 top-N 候选序列（N=10 或 beam size）"))
story.append(bullet("每个候选序列送入 Attention Decoder，交叉注意力关注 Encoder 编码结果"))
story.append(bullet("Decoder 输出序列在每步的条件概率（给定 Encoder 和之前预测的 token）"))
story.append(bullet("最终得分 = λ·CTC_score + (1-λ)·Attention_score"))
story.append(bullet("选择最高得分的候选作为最终输出"))
story.append(Spacer(1, 2))
story.append(Paragraph(
    "<b>为什么 Rescoring 有效?</b> Attention Decoder 使用自回归方式逐 token 预测，能建模序列级别的语言依赖，"
    "而 CTC 的独立假设（帧间独立）是一个强限制。Rescoring 通过解码器的上下文建模能力纠正 CTC 的局部错误。", s_body))
story.append(Spacer(1, 6))

story.append(Paragraph("2.5  训练策略与 Loss 设计", s_h2))
story.append(bullet("<b>多任务 Loss</b>: L = α·L_ctc + (1-α)·L_att，α=0.3。CTC Loss 帮助 Encoder 学习帧级别声学对齐，Attention Loss 帮助 Decoder 学习序列级语言建模"))
story.append(bullet("<b>Dynamic Chunk Training</b>: 训练时随机选择 chunk 大小（4/8/16/32），使模型学习在不同延迟配置下都能工作"))
story.append(bullet("<b>SpecAugment</b>: 频率掩码 F=2（掩码宽度 27）、时间掩码 T=2（掩码宽度 100）、时间扭曲 W=5"))
story.append(bullet("<b>优化器</b>: AdamW（lr=0.001, β1=0.9, β2=0.98, ε=1e-9），预热 25000 步后按 Noam 衰减"))
story.append(bullet("<b>Label Smoothing</b>: 0.1，防止过拟合，使模型校准更佳"))
story.append(bullet("<b>Gradient Clipping</b>: 5.0，防止梯度爆炸"))
story.append(PageBreak())

# ════════════════════════════════════
# 三、核心知识问答 & 问答解析
# ════════════════════════════════════
story.append(Paragraph("三、核心知识整理", s_h1))
story.append(divider())

story.append(Paragraph("3.1  基础概念类", s_h2))

qa_basic = [
    ("什么是端到端语音识别？与传统混合系统相比有什么优势？",
     "端到端（E2E）ASR 将声学模型、语言模型、词典等组件统一为一个神经网络。输入音频直接输出文本序列。"
     "优势：① 训练流程简化（不需要分别训练 AM/LM/词典）；② 无领域专家知识依赖（不需要音素级别的对齐标注）；"
     "③ 联合优化使得各模块间没有信息损失；④ 更易于部署和维护。"
     "缺点：① 需要更多数据；② 对训练集分布外场景鲁棒性较差；③ 可解释性不如混合系统。",
     "辩证分析：既要说优势也要谈缺点，体现全面的工程思维。"),

    ("U2++ 中的 '++' 代表什么？",
     "U2（Unified Two-pass）最早是 2020 年 UniSpeech 提出的概念。U2++ 在 U2 基础上做了三点改进："
     "① 支持真正的流式解码（Dynamic Chunk Training + 帧同步输出）；"
     "② 双向 Decoder（Bi-Transformer Decoder），既能从左到右看，也能从右到左看，在 Rescoring 时获得更好的上下文建模；"
     "③ 更灵活的 CTC/Attention 权重融合策略。"),

    ("WeNet 解决了什么问题？",
     "WeNet 解决了端到端 ASR 在生产环境部署的两大痛点："
     "① <b>流式支持</b>：很多 E2E 模型（如 LAS）不支持流式，WeNet 通过 U2++ 架构首次在工业级实现流式和非流式统一；"
     "② <b>轻量化部署</b>：WeNet 的设计理念是 '保持简单'，不需要额外的语言模型和 WFST 解码器，"
     "CTC + Attention Rescoring 即可达到与混合系统可比的效果。"),

    ("Conformer 相比 Transformer 的改进是什么？",
     "Conformer 在 Transformer Encoder 基础上增加了 <b>卷积模块</b>，弥补 Self-Attention 在局部模式捕捉上的不足。"
     "语音信号的特殊性在于：音素边界（局部）和语句结构（全局）都需要建模。"
     "卷积用大 kernel size（31）捕捉声学特征中的局部模式，Self-Attention 捕捉长距离依赖。"
     "此外 Conformer 采用 <b>Macaron Net 的半残差结构</b>（FFN 分布在 Self-Attention 两侧），"
     "相比标准 Transformer 的 Post-LN 更稳定，可以训练更深（12 层以上）的网络。"),

    ("什么是 CTC？它在语音识别中如何工作？",
     "CTC（Connectionist Temporal Classification）是一种无需对齐标签的序列学习算法。"
     "核心思想：引入 'blank' 符号，允许模型输出比目标序列更长的路径，"
     "所有通过合并相邻重复 token 和去除 blank 能匹配目标序列的路径都视为正确。"
     "训练时对�v所有合法路径求和（前向后向算法），推理时取每帧最大概率后合并重复即可得到输出序列。"
     "CTC 的独立假设（帧间条件独立）是其主要局限，但也正是 CTC 可以流式解码的关键。"),

    ("什么是 Attention Rescoring？它和普通的 Attention Decoder 有什么区别？",
     "Attention Rescoring 是两遍解码的核心。第一遍 CTC 产生 N 个候选项，"
     "第二遍 Attention Decoder 对每个候选项计算概率并重新打分。"
     "与普通的 Attention Decoder（如 LAS）的区别：LAS 一步到位解码输出，需要逐帧扩展（计算量大）；"
     "而 Rescoring 只需要在 CTC 产生的 ~10 个候选项上计算分数，计算量极小（只需 Decoder Forward）。"
     "所以 Rescoring 是在几乎不增加延迟的情况下显著提升精度。"),

    ("U2++ 如何实现流式解码？",
     "U2++ 通过 <b>Dynamic Chunk Training</b> 实现流式。训练时动态随机选择 chunk 大小（4/8/16/32），"
     "Self-Attention 的计算限制在当前 chunk 和左上历史缓存内。推理时 Encoder 按 chunk 逐块处理输入："
     "① 当前 chunk 与缓存的历史帧拼接；② Self-Attention 在拼接的序列上计算；"
     "③ 当前 chunk 的输出通过 CTC 分支解码；④ 更新历史缓存。"
     "这样延迟仅取决于 chunk 大小：chunk=16 时 640ms，chunk=4 时 160ms。"),

    ("什么是 RTF？你的模型 RTF 0.008 意味着什么？",
     "RTF 是 Real-Time Factor，即处理单位时长音频所需的计算时间。RTF=0.008 意味着 1 秒音频只需 8ms 即可完成识别。"
     "RTF < 1.0 意味着系统具备实时推理能力。0.008 远小于 1.0，说明模型有大量计算余量，可以部署在更低功耗设备上"
     "或同时处理多路音频流。"),
]
for item in qa_basic:
    if len(item) == 3:
        q, a, extra = item
    else:
        q, a = item
        extra = None
    story.extend(qa_block(q, a, extra))
story.append(Spacer(1, 6))

story.append(Paragraph("3.2  原理深挖类", s_h2))

qa_deep = [
    ("Conformer 中的 Self-Attention 为什么使用相对位置编码而非绝对位置编码？",
     "绝对位置编码将位置作为固定向量加到输入上，对于训练时见过的长度有效，但遇到更长的序列或流式输入时表现不佳。"
     "相对位置编码（Relative Positional Encoding）编码的是位置之间的相对偏移，而不是绝对位置。"
     "优势：① 支持流式场景（不需要知道整个序列长度）；② 支持变长输入；"
     "③ 对序列长度的泛化能力更强；④ Self-Attention 的可迁移性更好。"
     "实现上，WeNet 采用 Shaw 等人（2018）提出的方法，在 attention 分数计算中加入位置偏置项。"),

    ("U2++ 为什么用 CTC 做第一遍而不是直接用 Attention Decoder 流式解码？",
     "CTC 的第一遍解码是 <b>非自回归</b>的——每帧独立做分类，没有循环依赖。这带来了两个优势："
     "① <b>低延迟</b>：不需要逐 token 扩展（Attention Decoder 需要自回归生成），RTF 极低；"
     "② <b>帧同步</b>：可以逐帧输出，天然支持流式场景。"
     "但 CTC 的独立假设也意味着精度不如 Attention（不能建模序列依赖），所以用第二遍 Rescoring 做补充。"
     "这是典型的 '取长补短' 设计思路。"),

    ("Dynamic Chunk Training 的训练和推理细节？",
     "训练时：每个 batch 动态随机选择一个 chunk 大小（64, 32, 16, 8, 4）。"
     "Encoder 的 Self-Attention 只看到当前 chunk 内的帧 + 左侧的 history 缓存。"
     "推理时：按固定 chunk 大小逐块处理音频。关键实现细节："
     "① <b>History Cache</b>：维护左侧若干帧的 Encoder 输出和状态，当前 chunk 与其拼接后做 Self-Attention；"
     "② <b>Right Context</b>：部分架构（如 U2++）允许少量右侧上下文（如 4 帧）以提升精度；"
     "③ <b>EMA 前缀</b>：在流式推理中使用 EMA 策略稳定 CTC 的逐帧输出。"),

    ("Bi-Transformer Decoder 的双向体现在哪里？",
     "Bi-Transformer Decoder 包含两个并行的 Transformer Decoder：<b>前向</b>（从左到右）和 <b>后向</b>（从右到左）。"
     "Rescoring 时，CTC 候选序列同时送入两个 Decoder，获得前向和后向概率。"
     "最终得分 = λ·CTC_score + (1-λ)·(forward_score + backward_score) / 2。"
     "双向 Decoder 的优点：语音识别中语境是双向的，前文和后文对当前词的预测都有帮助，"
     "双向注意力能更全面地评估候选序列的质量。"),

    ("为什么 Conformer 使用 Swish 激活函数而不是 ReLU？",
     "Swish = x·σ(x)（sigmoid 门控的线性单元），由 Google 2017 年提出。优势："
     "① <b>平滑且非单调</b>：ReLU 在零点不可导，Swish 处处可导，优化更稳定；"
     "② <b>自门控</b>：Sigmoid 部分作为数据驱动的门控机制，对不同输入自适应调整；"
     "③ <b>负值保留</b>：允许少量负梯度通过，避免 ReLU 的 '神经元死亡' 问题。"
     "Conformer 原文实验表明 Swish 比 ReLU 在 WSJ 和 LibriSpeech 上都有稳定提升。"),

    ("Gradient Clipping 和 Label Smoothing 分别在解决什么问题？",
     "Gradient Clipping（梯度裁剪）：将梯度的 L2 范数限制在阈值（5.0）内。"
     "解决 Transformer/Conformer 训练中常见的梯度爆炸问题，特别是在深层网络或长序列训练时。"
     "Label Smoothing（标签平滑）：将 hard label（1-hot）替换为 soft label（如 0.9 one-hot + 0.1 均匀分布）。"
     "解决过拟合和模型过度自信（over-confident）的问题，使模型在预测时的概率校准更佳。"
     "实际经验：Label Smoothing=0.1 通常能带来 0.1~0.3% 的 CER 提升。"),

    ("SpecAugment 的具体参数对模型的影响？",
     "SpecAugment 包含三种增强：<b>时间扭曲</b>（沿时间轴随机扰动）、<b>频率掩码</b>（连续频率块置零）、"
     "<b>时间掩码</b>（连续时间块置零）。WeNet 默认 F=27（频率掩码宽度），T=100（时间掩码宽度），"
     "每个样本应用 2 个频率掩码和 2 个时间掩码。"
     "影响：① 频率掩码提高对噪声和信道变化的鲁棒性；② 时间掩码提高对语速变化的容忍度；"
     "③ 整体上相当于给模型提供大量数据增强，等效地扩大训练数据量。"
     "过强的 SpecAugment 会损害性能——mask 宽度太大可能会掩盖关键音素信息。"),

    ("什么是 Noam 学习率调度？为什么用于 Transformer 训练？",
     "Noam 调度（又名 Transformer 调度）：先线性预热，然后按步数的平方根倒数衰减。"
     "公式：lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))"
     "为什么需要预热：Transformer/Conformer 训练初期梯度不稳定，直接大学习率可能导致训练发散。"
     "预热 25000 步让优化器先 '探索' 参数空间，然后用指数衰减 '利用' 稳定区域。"
     "Noam 调度相比 StepLR/CosineAnnealing 更适合 Transformer 的原因是它的前期快速衰减和学生后期精细调节。"),

    ("说话人无关特征提取中的 CMVN 规范化的作用是什么？",
     "CMVN（Cepstral Mean and Variance Normalization）对每维 FBank 特征做均值和方差归一化。"
     "作用：① <b>消除信道差异</b>：不同麦克风、信道引入的加性噪声反映为均值的偏移，减均值消除；"
     "② <b>说话人归一化</b>：不同说话人基音频率和发声习惯差异反映在特征分布上，归一化后模型对不同说话人更鲁棒；"
     "③ <b>加速收敛</b>：归一化后的特征分布均值为 0 方差为 1，有利于神经网络的梯度传播。"
     "WeNet 使用全局 CMVN（基于整个训练集统计），而非说话人级别 CMVN。"),

    ("你怎么理解语音识别中的 '对齐' 问题？U2++ 是怎么解决的？",
     "对齐问题是语音识别的核心挑战：输入音频帧数远多于输出的文本 token 数，"
     "且缺少帧到 token 的标注。U2++ 通过多任务学习结合 CTC 和 Attention 两种对齐方式："
     "① <b>CTC 对齐</b>：隐式对齐——CTC 通过 blank 符号的插入来对齐，训练时穷举所有合法路径并求和概率；"
     "② <b>Attention 对齐</b>：显式对齐——Cross-Attention 中 Encoder 到 Decoder 的注意力权重隐式表示了帧和 token 的对齐关系；"
     "③ <b>多任务互补</b>：CTC 的对齐强于帧级别的声学对应，Attention 的对齐强于序列级别的语义对应。"
     "两件事出学习得到比单一 Loss 更好的对齐效果。"),
]
for item in qa_deep:
    if len(item) == 3:
        q, a, extra = item
    else:
        q, a = item
        extra = None
    story.extend(qa_block(q, a, extra))
story.append(PageBreak())

# ════════════════════════════════════
# 3.3 工程实践类
# ════════════════════════════════════
story.append(Paragraph("3.3  工程实践类", s_h2))

qa_eng = [
    ("如何选择 chunk 大小？实际应用中的权衡策略？",
     "chunk 大小直接影响延迟和精度的 trade-off："
     "chunk=4（160ms 延迟）适合 <b>实时口语翻译</b>等对延迟极度敏感的场景；"
     "chunk=16（640ms 延迟，CER 5.21%）是<b>通用推荐值</b>，在延迟和精度间取最佳平衡；"
     "non-streaming（CER 4.61%）适合<b>离线录音转写</b>等没有延迟要求的场景。"
     "实际部署中可以同时部署多个模型，根据业务场景动态切换。"),

    ("你怎么评估模型上线后的性能？",
     "除了离线测试集的 CER 指标，还需要关注："
     "① <b>在线 CER</b>：收集线上真实音频的标注数据，定期计算 CER；"
     "② <b>延迟监控</b>：P50/P95/P99 解码延迟，需要包含网络传输时间；"
     "③ <b>RTF 监控</b>：端到端 RTF（包含特征提取+模型推理+解码后处理）；"
     "④ <b>吞吐量</b>：单 GPU 或多 GPU 部署时的 QPS（每秒处理的音频时长）；"
     "⑤ <b>Bad Case 分析</b>：定期抽样线上失败案例，按类别（噪声、语速、口音等）分类统计。"),

    ("在 WeNet 中如果 CER 不收敛了，你从哪些方向排查？",
     "排查链路：① <b>数据</b>：检查音频和文本标签对齐是否有错误（常见问题）；"
     "② <b>Loss</b>：看 CTC Loss 和 Attention Loss 是否都下降，如果只有 Attention Loss 下降可能是 CTC 权重太小；"
     "③ <b>LR</b>：学习率是否过大或过小（Noam 调度的预热是否充足）；"
     "④ <b>Gradient</b>：梯度范数是否异常（检查 Gradient Clipping 的命中频率）；"
     "⑤ <b>数据增强</b>：SpecAugment 是否过强导致模型无法学习；"
     "⑥ <b>训练/验证 gap</b>：训练集 CER 下降但验证集不下降 → 过拟合，需要增加正则化或减少训练轮数。"),

    ("你的 Gradio Web UI 如何处理长音频（超过 30 秒）？",
     "长音频处理策略：① <b>VAD（Voice Activity Detection）</b> 将音频切割成有声片段；"
     "② 每个片段独立推理；③ 输出拼接。如果需要实时流式效果，可以用前端 WebSocket 持续传输音频流，"
     "服务端边接收边解码，流式输出中间结果。Gradio 本身支持 audio streaming input，"
     "可以用 generator 函数实现流式音频的增量推理。"),

    ("HuggingFace 模型转换中的难点是什么？",
     "主要难点：① <b>模型结构差异</b>：WeNet 的 U2++ 结构与 HF 标准 Transformer 不完全一致，"
     "需要自定义 `WenetForASR` 类继承 `PreTrainedModel` 并实现 config 映射；"
     "② <b>权重命名映射</b>：WeNet 的 checkpoint 命名与 HF 规范不同，需要做命名映射；"
     "③ <b>config.json</b>：需要定义完整的模型参数（层数、维度、head 数、chunk 大小等），"
     "确保 from_pretrained 能完全重建模型结构。"),

    ("CI 中的 pytest 测试应该覆盖哪些层面？",
     "语音识别项目的测试分层："
     "① <b>单元测试</b>：CER 计算函数（参考编辑距离实现）、特征提取（FBank 维度/均值/方差正确性）、"
     "数据加载（音频长度/采样率/token 对应）。这些都是核心函数，一旦出错全线崩溃；"
     "② <b>集成测试</b>：模型前向推理（输入 dummy audio 检查输出 shape 和分布）、"
     "解码链路的端到端验证（一个已知的音频文件输出预期结果）；"
     "③ <b>回归测试</b>：定量设置 CER 上限阈值（如 CI 中验证某一测试音频的 CER < 10%），"
     "防止改动导致精度退化；"
     "④ <b>性能测试</b>（可选）: RTF 上限检查，防止代码改动降低推理速度。"),
]
for item in qa_eng:
    if len(item) == 3:
        q, a, extra = item
    else:
        q, a = item
        extra = None
    story.extend(qa_block(q, a, extra))
story.append(Spacer(1, 6))

# ════════════════════════════════════
# 3.4 对比分析类
# ════════════════════════════════════
story.append(Paragraph("3.4  对比分析类", s_h2))

qa_compare = [
    ("WeNet (U2++) 对比 ESPnet 的优缺点？",
     "WeNet 优势：① <b>生产级流式支持</b>——ESPnet 更偏向研究和实验，流式支持不够成熟；"
     "② <b>代码简洁</b>——WeNet 的设计理念是 'keep it simple'，核心代码精炼，易于理解和定制；"
     "③ <b>工业级部署链路完整</b>——集成 WebSocket 服务、ONNX 导出、量化支持。"
     "ESPNET 优势：① <b>模型多样性</b>——支持更多 SOTA 架构（Conformer、Squeezeformer、Branchformer 等）；"
     "② <b>研究社区活跃</b>——论文复现快；③ <b>功能完整</b>——支持语音合成、语音翻译等多模态任务。"),

    ("U2++ 对比 RNN-T 的优缺点？",
     "RNN-T 优势：① <b>真正的一步到位</b>——不需要两遍解码，逻辑更简单；"
     "② <b>流式天然友好</b>——RNN-T 的 Joint Network 本来就在每一帧做决策；"
     "③ <b>业界主流验证</b>——RNN-T 是 Google 语音助手等产品的核心技术。"
     "U2++ 优势：① <b>精度更高</b>——两遍解码的 Rescoring 在同等条件下通常比 RNN-T 好 0.3~0.5% 绝对 CER；"
     "② <b>灵活性</b>——可以自由组合 CTC 和 Attention 的权重，更灵活地权衡延迟和精度；"
     "③ <b>研究者友好</b>——U2++ 的模块化设计（CTC + Decoder 分离）更容易做研究和改进。"),

    ("你们为什么选 WeNet 而不是其他开源 ASR 框架？",
     "选择 WeNet 的原因：① <b>流式非流式统一</b>——这是 WeNet 相比其他框架（如 ESPnet、Kaldi）的最核心优势，"
     "一个模型做两件事；② <b>代码质量高</b>——WeNet 的代码结构非常清晰，几乎没有不必要的抽象和封装，"
     "易于二次开发；③ <b>部署链路完整</b>——不依赖 Kaldi 生态，没有 WFST 编译等复杂流程；"
     "④ <b>社区活跃</b>——WeNet 由出门问问等团队持续维护，Issue/PR 响应及时。"),

    ("CTC 和 Attention Decoder 分别的优势和劣势？",
     "CTC 优势：① 流式天然支持；② 非自回归，推理快；③ 不需要显式语言模型就能输出合理结果。"
     "CTC 劣势：① 帧间独立假设（不能利用序列上下文）；② 性能上限低于 Attention（没有 cross-attention 的辅助，对齐是 blind 的）；"
     "③ 输出必须有单调对齐假设。"
     "Attention Decoder 优势：① 可以建模完整的序列上下文依赖；② 通过 Cross-Attention 可以获得 Encoder 的全面信息；"
     "③ 精度的理论上限更高。优势：① 自回归生成，推理慢（逐 token 生成，不能并行）；"
     "② 流式支持复杂（需要有 mask 和缓存管理）；③ Training-Serving Gap（曝光偏差）。"),

    ("Conformer vs Transformer vs RNN 在语音识别中的应用？",
     "RNN（LSTM/GRU）：最早用于 ASR 的主流架构，优点是有天然时序性、参数效率高；"
     "缺点是难以并行训练（串行计算）、长序列梯度消失/爆炸。"
     "Transformer：全注意力机制解决并行训练问题，捕捉长距离依赖的能力最强；"
     "缺点是计算量大（二次复杂度）、局部建模能力不足（相较于 CNN）。"
     "Conformer：卷积 + 注意力融合，在局部和全局建模间取得最佳平衡，是当前 ASR 领域的事实标准。"
     "一句话总结：RNN 是过去，Transformer 是现在，Conformer 是当下最佳实践。"),

    ("你的模型在 AISHELL-1 上 CER 4.61% 在行业中是什么水平？",
     "AISHELL-1 是目前中文普通话 ASR 最广泛使用的开源基准（141k 训练语句，~179 小时）。"
     "已知结果对比：WeNet 官方 U2++ Conformer 约 4.5~4.8%；"
     "ESPNET Conformer 约 4.5~4.7%；带有外部语言模型可以降到 4.2~4.5%；"
     "2019 年 Kaldi TDNN-F + LF-MMI 混合系统约 6.5~7.0%。"
     "我们的 4.61% 在当时和现在都处于有竞争力的水平，且优势在于这是 <b>纯端到端、无外部语言模型</b> 的结果。"),
]
for item in qa_compare:
    if len(item) == 3:
        q, a, extra = item
    else:
        q, a = item
        extra = None
    story.extend(qa_block(q, a, extra))
story.append(PageBreak())

# ════════════════════════════════════
# 3.5 开放思维类
# ════════════════════════════════════
story.append(Paragraph("3.5  开放思维类", s_h2))

qa_open = [
    ("如果把你的模型部署到手机上，你会怎么做？",
     "移动端部署策略：① <b>量化</b>——FP32 → FP16 → INT8，模型体积从 ~50MB 降至 ~12MB；"
     "② <b>模型剪枝/蒸馏</b>——使用 Teacher-Student 框架将 12 层 Conformer 蒸馏到 6 层；"
     "③ <b>计算优化</b>——使用 NCNN/TFLite/MNN 等移动端推理框架，结合 NEON 指令集加速卷积运算；"
     "④ <b>架构调整</b>——clip chunk size 到 8 或 4，延迟降至 320ms 以下；"
     "⑤ <b>内存管理</b>——Cache 大小量化，限制最大序列长度。"),

    ("如果模型遇到新的领域（如医疗、法律），你会怎么处理？",
     "领域迁移策略：① <b>Domain-Adapter</b>——在 Conformer 的 FFN 后插入轻量 Adapter 层，只微调 Adapter 和 Embedding；"
     "② <b>语言模型融合</b>——添加外部领域 LM，在 Rescoring 时结合 LLM 的领域知识打分；"
     "③ <b>数据收集</b>——主动学习策略，使用当前模型在领域数据上预测置信度低的样本优先标注；"
     "④ <b>Curriculum Learning</b>——先通用语料预训练 → 领域语料微调 → 领域特定术语增强训练；"
     "⑤ <b>Few-shot/Zero-shot</b>——研究使用 Whisper 等大规模预训练模型做 zero-shot 推理，或用 LoRA 做 few-shot 快速适配。"),

    ("你认为语音识别未来的发展方向是什么？",
     "① <b>大模型统一架构</b>：Whisper、USM 等大规模多任务模型正在取代小模型，未来可能一个模型处理所有语言和场景；"
     "② <b>多模态融合</b>：语音+视觉（唇读）+文本的联合建模，特别是在噪声环境下的鲁棒性提升；"
     "③ <b>LLM 集成</b>：将 LLM 作为 ASR 的后处理或 rescore 模块，利用 LLM 的语义理解和知识能力纠正识别错误；"
     "④ <b>无监督/自监督</b>：HuBERT、wav2vec 2.0 等自监督预训练极大降低了标注成本，未来 E2E ASR 的标注门槛会越来越低；"
     "⑤ <b>边缘端实时化</b>：ASR 在手机、IoT 设备上的全量本地部署越来越普遍，模型小型化和硬件加速是关键。"),

    ("你的项目中哪些部分最有成就感？（技术深挖中的自我评价问题）",
     "最有成就感的部分：构建了一套从<b>数据处理→模型训练→解码评估→基准测试→可视化→Web UI→HF 集成</b>的完整链路。"
     "具体来说：① 实现了 Uni-Pro 和 UIO 两种混淆矩阵，让测试结果可视化程度更高；"
     "② 搭建了 CI 自动化流水线，每次 push 自动验证核心功能不退化；"
     "③ 生成了出版级的 4 张 Benchmark 图表，能直观展示模型在不同配置下的性能对比。"
     "我认为好的项目不仅仅是模型精度高，更要有完整的工程化闭环。"),

    ("你怎么看 WeNet 的 U2++ 和 Whisper 的差异？",
     "设计理念完全不同。WeNet 定位<b>工业级流式 ASR</b>，追求低延迟和高效率，适合对话交互、语音助手等实时场景。"
     "Whisper 定位<b>通用语音理解</b>，追求高质量离线转录，支持 99 种语言，架构是标准的 Encoder-Decoder Transformer（无 CTC 分支）。"
     "核心差异：① 流式——WEMet 原生支持，Whisper 不支持；② 参数量——WeNet 约 50M，Whisper 从 39M 到 1.5B；"
     "③ 多语言——Whisper 强在 multilingual，WeNet 聚焦中文/英文；"
     "④ 部署——WeNet 轻量易于部署，Whisper 大模型需要 GPU。"
     "合理的使用方案是两者互补：实时流式用 WeNet，复杂场景的离线精转用 Whisper。"),

    ("如果让你在现有的项目基础上做一个改进或创新，你会做什么？",
     "我会做 <b>RepVGG 风格的 Encoder 重构</b>——训练时使用多分支结构（类似 Conformer），"
     "推理时通过结构重参数化合并为单分支结构。（推理时没有 CNN 和 Attention 的分支，全部整合为一个 FFN）"
     "这样可以在不损失精度的情况下将推理速度提升 20-30%。"
     "或者做一个 <b>Adapter-CTC</b>——在大规模预训练模型（如 Whisper）的 Encoder 上插入轻量 Adapter，"
     "只微调 Adapter 和 CTC 头，快速适配特定领域（如医疗、法律）的 ASR，大幅降低微调成本。"),
]
for item in qa_open:
    if len(item) == 3:
        q, a, extra = item
    else:
        q, a = item
        extra = None
    story.extend(qa_block(q, a, extra))
story.append(PageBreak())

# ════════════════════════════════════
# 四、技术深挖方向
# ════════════════════════════════════
story.append(Paragraph("四、技术深挖方向", s_h1))
story.append(divider())

story.append(Paragraph("4.1  模型结构深挖", s_h2))
story.append(bullet("Conformer 中 Conv Module 的 Depthwise Conv kernel size 为什么选择 31？换成 15 或 63 会怎样？"))
story.append(bullet("为什么 Conformer 的 FFN 在 Self-Attention 两侧各放一个（Macaron 结构）而不是一侧一个？"))
story.append(bullet("相对位置编码有哪几种实现方式（Shaw / Transformer-XL / T5）？WeNet 用了哪种？"))
story.append(bullet("U2++ 为什么不直接用更多的 Decoder 层代替 Rescoring？"))
story.append(bullet("Multi-Head Attention 的 head 数量对性能的影响？head 过多会怎样？"))
story.append(Spacer(1, 6))

story.append(Paragraph("4.2  流式与非流式", s_h2))
story.append(bullet("Dynamic Chunk Training 中 left context 和 right context 各需要多少帧？"))
story.append(bullet("流式 CTC 解码中，如何确保不同 chunk 的输出拼接起来是平滑的？"))
story.append(bullet("流式场景下，如果用户中途停顿或说话加速，chunk 大小应该如何自适应？"))
story.append(bullet("Non-Streaming 解码能不能反向（从右到左）做？结果会不同吗？"))
story.append(bullet("流式解码的 first-pass latency 具体包含哪些部分？（特征提取+模型推理+解码后处理）"))
story.append(Spacer(1, 6))

story.append(Paragraph("4.3  训练技巧与调参", s_h2))
story.append(bullet("AISHELL-1 只有 179 小时数据，你是怎么防止过拟合的？SpecAugment、Dropout、Weight Decay 各用的多少？"))
story.append(bullet("学习率预热步数 25000 的依据是什么？是经验值吗？"))
story.append(bullet("Batch size 怎么选的？大了 GPU OOM，小了训练不稳定。"))
story.append(bullet("多 GPU 训练时梯度同步怎么做？用的是 DDP 还是 FSDP？"))
story.append(bullet("Mix Precision Training（FP16）用了吗？对精度有没有影响？"))
story.append(Spacer(1, 6))

story.append(Paragraph("4.4  部署与优化", s_h2))
story.append(bullet("ONNX导出torch.onnx.export遇到过什么问题？动态轴（dynamic axes）怎么处理的？"))
story.append(bullet("量化 INT8 后 CER 退化多少？怎么补偿的？"))
story.append(bullet("WebSocket 服务端高并发怎么处理的？asyncio 还是多进程？"))
story.append(bullet("模型在 CPU 上的 RTF 和 GPU 上差多少？"))
story.append(bullet("流式服务的内存管理——逐 chunk 推理时显存/内存如何渐进增长？"))
story.append(Spacer(1, 6))

story.append(Paragraph("4.5  前沿方向", s_h2))
story.append(bullet("你有没有关注最近 Flash Attention、Flash Conv 对 ASR 的加速效果？"))
story.append(bullet("Branchformer、E-Branchformer、Squeezeformer 等更新的架构和 Conformer 比效果如何？"))
story.append(bullet("Whisper 出来后，对 WeNet 这种工业级 ASR 框架的冲击有多大？"))
story.append(bullet("大型语言模型（LLMs）能做 ASR 的后处理纠错吗？你的实验或想法？"))
story.append(bullet("Voice Activity Detection（VAD）和 ASR 的协同优化怎么做？"))
# ════════════════════════════════════
# 五、技术深挖进阶准备
# ════════════════════════════════════
story.append(Paragraph("五、技术深挖进阶准备", s_h1))
story.append(divider())

story.append(Paragraph("5.1  从零推导 Conformer", s_h2))
story.append(Paragraph(
    "Conformer 块的完整结构推导如下：", s_body))
story.append(bullet("<b>输入</b>: x ∈ ℝ^(T×d)，T 为帧数，d=256 为模型维度"))
story.append(bullet("<b>FFN (前半)</b>: x₁ = x + 0.5 · FFN(LayerNorm(x)), FFN(x) = Linear(ReLU(Linear(x)))"))
story.append(bullet("<b>Self-Attention</b>: x₂ = x₁ + MHSA(LayerNorm(x₁))，含相对位置编码"))
story.append(bullet("<b>Conv Module</b>: x₃ = x₂ + Conv(LayerNorm(x₂))，Conv=PointwiseConv → GLU → DepthwiseConv → BN → Swish → PointwiseConv"))
story.append(bullet("<b>FFN (后半)</b>: x₄ = x₃ + 0.5 · FFN(LayerNorm(x₃))"))
story.append(bullet("<b>输出</b>: y = LayerNorm(x₄)"))
story.append(bullet("<b>复杂度</b>: Self-Attention O(T²·d), Conv O(T·d·k), 其中 k=31 为 depthwise conv kernel size"))
story.append(Spacer(1, 6))

story.append(Paragraph("5.2  CTC 对齐原理", s_h2))
story.append(Paragraph(
    "CTC 的对齐算法（前向后向）核心要点如下：", s_body))
story.append(bullet("<b>问题</b>: 输入 X=[x₁,...,x_T] (T 帧), 标注 Y=[y₁,...,y_U] (U 个 token), T >> U"))
story.append(bullet("<b>扩展序列</b>: 在 Y 的所有 token 间和首尾插入 blank，得到 Z=[z₁,...,z_{2U+1}]"))
story.append(bullet("<b>合法路径</b>: α_t(s) = 所有在 t 时刻到达 Z_s 的路径概率之和"))
story.append(bullet("<b>递推</b>: α_t(s) = y_{z_s}^t · (α_{t-1}(s) + α_{t-1}(s-1) + [s-2 且 z_{s-2}=z_s] · α_{t-1}(s-2))"))
story.append(bullet("<b>Loss</b>: L_CTC = -ln(P(Y|X)) = -ln(Σ_{π∈B⁻¹(Y)} Π_t P(π_t|X))"))
story.append(Spacer(1, 6))

story.append(Paragraph("5.3  RNN-T vs Attention vs CTC", s_h2))
story.append(Table([
    [Paragraph("<b>方面</b>", s_h3), Paragraph("<b>CTC</b>", s_h3),
     Paragraph("<b>RNN-T</b>", s_h3), Paragraph("<b>Attention</b>", s_h3)],
    [Paragraph("对齐方式", s_body), Paragraph("单调+blank", s_body),
     Paragraph("单调+预测网络", s_body), Paragraph("非单调", s_body)],
    [Paragraph("流式支持", s_body), Paragraph("✅ 天然支持", s_body),
     Paragraph("✅ 天然支持", s_body), Paragraph("❌ 需要 mask", s_body)],
    [Paragraph("自回归", s_body), Paragraph("❌ 非自回归", s_body),
     Paragraph("✅ 自回归", s_body), Paragraph("✅ 自回归", s_body)],
    [Paragraph("精度上限", s_body), Paragraph("中等", s_body),
     Paragraph("高", s_body), Paragraph("最高", s_body)],
    [Paragraph("计算量", s_body), Paragraph("最低", s_body),
     Paragraph("中等", s_body), Paragraph("最高", s_body)],
    [Paragraph("推断延迟", s_body), Paragraph("最低", s_body),
     Paragraph("低-中", s_body), Paragraph("中-高", s_body)],
], colWidths=[80, 95, 95, 95]))
story.append(Spacer(1, 6))

story.append(Paragraph("5.4  流式解码的工程实现", s_h2))
story.append(Paragraph(
    "流式解码的工程实现比算法原理更复杂。关键函数和数据结构：", s_body))
story.append(bullet("<b>Cache</b>: 维护上一 chunk 的 Encoder 输出和 Self-Attention 的 Key/Value 缓存。每次新 chunk 到来，与 cache 拼接后做前向"))
story.append(bullet("<b>FIFO 队列</b>: 特征提取的线程和推理线程解耦，通过队列传递音频帧"))
story.append(bullet("<b>帧同步输出</b>: 流式解码需要逐帧或逐 token 输出中间结果，使用 generator/yield 实现"))
story.append(bullet("<b>WebSocket 通信</b>: 客户端持续上行音频帧，服务端下行实时识别结果（JSON 格式）"))
story.append(bullet("<b>长句管理</b>: VAD 检测到句子结束时 flush decoder，输出完整句子结果"))
story.append(Spacer(1, 6))

story.append(Paragraph("5.5  模型量化与加速", s_h2))
story.append(bullet("<b>FP16 推理</b>: 使用 torch.cuda.amp，几乎无精度损失，速度提升 1.5-2x"))
story.append(bullet("<b>INT8 量化</b>: 使用 Intel IPEX 或 TensorRT，速度提升 3-4x，但 CER 通常退化 0.1-0.3%"))
story.append(bullet("<b>ONNX Runtime</b>: 跨平台部署，比 PyTorch eager mode 快 1.2-1.5x"))
story.append(bullet("<b>TensorRT</b>: NVIDIA GPU 上最优推理引擎，支持 FP16 + INT8 + kernel auto-tuning"))
story.append(bullet("<b>模型剪枝</b>: 对 Conformer 的 FFN 层做 weight pruning，稀疏度 30% 时精度基本不变"))
story.append(bullet("<b>知识蒸馏</b>: 12 层 Teacher → 6 层 Student，CER 退化 0.2-0.5% 但速度翻倍"))
# ════════════════════════════════════
# 六、推荐阅读与资料
# ════════════════════════════════════
story.append(Paragraph("六、推荐阅读与资料", s_h1))
story.append(divider())

story.append(Paragraph("6.1  必读论文（按优先级排序）", s_h2))
story.append(bullet("Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020) — 核心架构论文"))
story.append(bullet("WeNet: Production Oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit (Yao et al., 2021) — 框架论文"))
story.append(bullet("Unified Streaming and Non-streaming Two-pass End-to-End Model for Speech Recognition (Zhang et al., 2020) — U2 原始论文"))
story.append(bullet("U2++: Unified Two-Pass Speech Recognition Framework (Zhang et al., 2022) — U2++ 论文"))
story.append(bullet("Listen, Attend and Spell (Chu et al., 2016) — LAS 模型，Attention Decoder 基础"))
story.append(bullet("Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks (Graves et al., 2006) — CTC 原始论文"))
story.append(bullet("SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition (Park et al., 2019) — 数据增强论文"))
story.append(Spacer(1, 6))

story.append(Paragraph("6.2  进阶阅读", s_h2))
story.append(bullet("Transformer: Attention is All You Need (Vaswani et al., 2017)"))
story.append(bullet("Relative Position Encoding (Shaw et al., 2018) — Conformer 使用的位置编码"))
story.append(bullet("RNN-T: Sequence Transduction with Recurrent Neural Networks (Graves, 2012)"))
story.append(bullet("wav2vec 2.0 / HuBERT — 自监督预训练，大模型时代的基础"))
story.append(bullet("Whisper: Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)"))
story.append(Spacer(1, 6))

story.append(Paragraph("6.3  技术博客与资源", s_h2))
story.append(bullet("WeNet 官方文档: https://wenet.org.cn/"))
story.append(bullet("WeNet GitHub: https://github.com/wenet-e2e/wenet"))
story.append(bullet("AISHELL-1 数据集: https://www.openslr.org/33/"))
story.append(bullet("项目仓库: https://github.com/ywyuan666/wenet-aishell-u2pp"))
story.append(bullet("Google Speech Recognition Tutorial — CTC 和 LAS 的手动实现教程"))
story.append(bullet("distill.pub 的 Attention 可视化 — 理解 Attention 机制的最佳入门"))

# ════════════════════════════════════
# 尾页
# ════════════════════════════════════
story.append(Spacer(1, 60))
footer_table = Table([[
    Paragraph("<br/><br/><br/>持续学习，不断精进<br/>祝你技术之路越走越宽！<br/><br/>", make_style('footer',
              fontSize=14, textColor=white, alignment=TA_CENTER)),
]], colWidths=[460])
footer_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('LEFTPADDING', (0,0), (-1,-1), 20),
    ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ('TOPPADDING', (0,0), (-1,-1), 10),
    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ('ROUNDEDCORNERS', [8,8,8,8]),
]))
story.append(footer_table)

# ════════════════════════════════════
# 生成 PDF
# ════════════════════════════════════
doc.build(story)
print(f"\n[OK] PDF 已生成: {OUTPUT_PATH}")
print(f"     大小: {os.path.getsize(OUTPUT_PATH) // 1024} 