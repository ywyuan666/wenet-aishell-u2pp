#!/usr/bin/env python3
"""Generate architectural diagram for README (English labels only)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#f8f9fa')

colors = {
    'input': '#4CAF50',
    'encoder': '#2196F3',
    'encoder_block': '#42A5F5',
    'ctc': '#FF9800',
    'decoder': '#AB47BC',
    'output': '#E53935',
    'text': '#37474F',
    'streaming': '#FF7043',
    'rescoring': '#26A69A',
}

def draw_box(ax, x, y, w, h, color, label, sublabel='', edgecolor=None, alpha=0.9, fontsize=12):
    edgecolor = edgecolor or color
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=edgecolor,
                          linewidth=2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.05, label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.35, sublabel,
                ha='center', va='center', fontsize=8, color='white', alpha=0.9)

# Title
ax.text(7, 9.5, 'WeNet Conformer U2++ Architecture', ha='center', va='center',
        fontsize=22, fontweight='bold', color=colors['text'])
ax.text(7, 9.1, 'Two-Pass Streaming + Non-Streaming Unified End-to-End ASR', ha='center', va='center',
        fontsize=13, color='#78909C')

# Layer 0: Audio Input
draw_box(ax, 5.5, 8.3, 3, 0.6, colors['input'], 'Audio Input', '16kHz WAV')
ax.annotate('', xy=(7, 8.3), xytext=(7, 7.8),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=0'))

# Layer 1: Fbank + CMVN
draw_box(ax, 4, 7.2, 6, 0.55, '#66BB6A', 'Feature Extraction', '80-dim Fbank + CMVN + SpecAug')
ax.annotate('', xy=(7, 7.2), xytext=(7, 6.7),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=0'))

# Layer 2: Subsampling
draw_box(ax, 4, 6.2, 6, 0.45, '#81C784', 'Conv2D Subsampling', '4x downsampling')

# Layer 3: Conformer Encoder x12
y_enc = 4.3
enc_w = 8
enc_h = 1.4
draw_box(ax, 3, y_enc, enc_w, enc_h, colors['encoder'], 'Conformer Encoder x12 Layers', '', alpha=0.85, fontsize=13)
for i in range(12):
    bx = 3.5 + i * 0.65
    draw_box(ax, bx, y_enc + 0.15, 0.5, 1.1, colors['encoder_block'], str(i+1), '', alpha=0.7, fontsize=7)

# Encoder annotations
ax.text(3 + enc_w + 0.3, y_enc + enc_h/2 - 0.35, 'CNN', fontsize=9, color='#42A5F5', va='center', fontweight='bold')
ax.text(3 + enc_w + 0.3, y_enc + enc_h/2, 'Self-Attn', fontsize=9, color='#42A5F5', va='center', fontweight='bold')
ax.text(3 + enc_w + 0.3, y_enc + enc_h/2 + 0.35, 'FFN', fontsize=9, color='#42A5F5', va='center', fontweight='bold')
ax.text(3 + enc_w + 2.0, y_enc + enc_h/2, 'Relative Position Encoding', fontsize=9, color='#42A5F5', va='center', fontweight='bold')

ax.annotate('', xy=(7, y_enc), xytext=(7, 3.4),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=0'))

# Layer 4: Two paths
draw_box(ax, 4.5, 2.7, 3, 0.65, colors['streaming'], '1st Pass: CTC Greedy', 'Streaming / Low Latency')
draw_box(ax, 8.5, 2.7, 3, 0.65, colors['decoder'], 'Bi-Transformer Decoder', 'Cross-Attention x3+3')

ax.annotate('', xy=(6, 3.35), xytext=(7, y_enc),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=-0.3'))
ax.annotate('', xy=(10, 3.35), xytext=(7, y_enc),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=0.3'))

# Layer 5: CTC output
draw_box(ax, 4.5, 1.8, 3, 0.55, '#F44336', 'CTC Logits (blank-aware)', 'Vocabulary: 4234 tokens')
ax.annotate('', xy=(6, 2.35), xytext=(6, 2.7),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=0'))

# Layer 6: Output
draw_box(ax, 4, 0.6, 6, 0.8, colors['output'], 'Attention Rescoring (2nd Pass)', 'Re-rank CTC candidates')
ax.annotate('', xy=(6, 1.4), xytext=(6, 1.8),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=0'))
ax.annotate('', xy=(10, 1.4), xytext=(10, 3.025),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=2.5, connectionstyle='arc3,rad=0'))

# Final result
final_box = FancyBboxPatch((5, 0.1), 4, 0.45, boxstyle="round,pad=0.08", facecolor='#2E7D32', edgecolor='white', linewidth=3)
ax.add_patch(final_box)
ax.text(7, 0.325, 'Final Recognition Text', ha='center', va='center', fontsize=13, fontweight='bold', color='white')

# Legend
ax.text(0.3, 0, 'Legend:', fontsize=10, fontweight='bold', color=colors['text'])
legend_items = [
    ('#4CAF50', 'Input/Feature'),
    ('#2196F3', 'Encoder'),
    ('#FF9800', 'CTC (Streaming)'),
    ('#AB47BC', 'Decoder'),
    ('#E53935', 'CTC Output'),
    ('#26A69A', 'Rescoring'),
]
for i, (c, l) in enumerate(legend_items):
    ax.add_patch(mpatches.FancyBboxPatch((1.8 + i*1.8, -0.05), 0.3, 0.25,
                                          boxstyle="round,pad=0.02", facecolor=c, alpha=0.85))
    ax.text(2.15 + i*1.8, 0.1, l, ha='left', va='center', fontsize=8, color=c)

plt.tight_layout()
plt.savefig('figures/architecture_diagram.png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
print('Generated: figures/architecture_diagram.png')
