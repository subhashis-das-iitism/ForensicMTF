"""Generates the four architecture block diagrams for the paper (overall_architecture,
stream_1_arch, stream_2_arch, stream_3_arch) as clean draw.io-style box/arrow figures,
built with matplotlib. Every box, shape, and connection here is transcribed directly
from the verified source code (models/detector.py, models/forensicmtf.py) - not a
free-hand sketch. See the block-by-block comments below for which code object each
box corresponds to.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parent / 'records' / 'DFB' / 'figures'

COLOR_TEMPORAL = '#cfe2ff'
COLOR_NOISE = '#ffe5b4'
COLOR_SPATIAL = '#d4f7d4'
COLOR_FUSION = '#f0d9ff'
COLOR_HEAD = '#ffd6d6'
COLOR_IO = '#eeeeee'
EDGE = '#333333'


def box(ax, xy, w, h, text, color, fontsize=9, edgecolor=EDGE, lw=1.2, ha='center'):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                           linewidth=lw, edgecolor=edgecolor, facecolor=color, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va='center', fontsize=fontsize, zorder=3, wrap=True)
    return (x, y, w, h)


def arrow(ax, b1, b2, side1='right', side2='left', text=None, style='-|>', color=EDGE, lw=1.3, rad=0.0, fontsize=8):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {
        'right': (x1 + w1, y1 + h1 / 2), 'left': (x1, y1 + h1 / 2),
        'top': (x1 + w1 / 2, y1 + h1), 'bottom': (x1 + w1 / 2, y1),
    }
    pts2 = {
        'right': (x2 + w2, y2 + h2 / 2), 'left': (x2, y2 + h2 / 2),
        'top': (x2 + w2 / 2, y2 + h2), 'bottom': (x2 + w2 / 2, y2),
    }
    p1, p2 = pts[side1], pts2[side2]
    fa = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, linewidth=lw,
                         color=color, zorder=1, connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(fa)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.15, text, ha='center', va='bottom', fontsize=fontsize, style='italic', zorder=4)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


# =============================================================================
# Fig: overall_architecture.png  (models/forensicmtf.py::ForensicMTF.forward)
# =============================================================================
def gen_overall_architecture():
    fig, ax = new_fig(13, 8)

    b_in = box(ax, (0.3, 6.6), 2.4, 1.0, 'DeepFakeBench\nframes+landmarks+masks', COLOR_IO, 8.5)
    b_prep = box(ax, (0.3, 4.6), 2.4, 1.6,
                'Preprocessing\n(Sec. IV-B)\nT=32 sampling\nlandmark bbox crop\nheatmap H', COLOR_IO, 8)

    b_s1 = box(ax, (3.3, 6.4), 2.6, 1.4, 'Stream 1: Temporal\nR3D-18 + Temporal\nAttention  →  $f_v$', COLOR_TEMPORAL, 9)
    b_s2 = box(ax, (3.3, 4.6), 2.6, 1.4, 'Stream 2: Noise/Freq.\nBayarConv + FFT + DWT\n→  $f_n$', COLOR_NOISE, 9)
    b_s3 = box(ax, (3.3, 2.8), 2.6, 1.4, 'Stream 3: Spatial\nConv stem + Landmark\nAttention  →  $f_s$, $\\mathbf{M}\'$', COLOR_SPATIAL, 9)

    b_ca = box(ax, (6.7, 6.0), 2.3, 1.2, 'Cross-Attention\nQ=$f_v$, K=V=$f_n$\n→  $f_{fused}$', COLOR_FUSION, 8.5)
    b_add = box(ax, (9.4, 6.0), 2.0, 1.2, 'Additive inject.\n$f_{final}=f_{fused}{+}s_3 f_s$', COLOR_FUSION, 8.5)

    b_cls = box(ax, (11.7, 6.6), 1.0, 1.0, 'Classifier\nMLP\n→ logits', COLOR_HEAD, 8)
    b_loc = box(ax, (11.7, 4.9), 1.0, 1.0, 'Localization\nHead\n→ mask', COLOR_HEAD, 8)
    b_id = box(ax, (11.7, 3.2), 1.0, 1.0, 'Identification\nHead\n→ method', COLOR_HEAD, 8)

    arrow(ax, b_in, b_prep, 'bottom', 'top')
    arrow(ax, b_prep, b_s1, 'right', 'left', rad=0.15)
    arrow(ax, b_prep, b_s2, 'right', 'left')
    arrow(ax, b_prep, b_s3, 'right', 'left', rad=-0.15)

    arrow(ax, b_s1, b_ca, 'right', 'left', text='$f_v$ (Q)')
    arrow(ax, b_s2, b_ca, 'right', 'left', text='$f_n$ (K,V)', rad=-0.1)
    arrow(ax, b_ca, b_add, 'right', 'left')
    arrow(ax, b_s3, b_add, 'right', 'bottom', text='$s_3 f_s$', rad=0.2)
    arrow(ax, b_s3, b_loc, 'right', 'left', text="$\\mathbf{M}'$", rad=0.3)

    arrow(ax, b_add, b_cls, 'right', 'left', rad=0.15)
    arrow(ax, b_add, b_id, 'right', 'left', rad=-0.15)

    ax.text(6.5, 2.0, 'MLG-MTF: landmark-attention gate (Stream 3) + Localization Head + Identification Head\n'
                      'jointly, directly supervised - replaces the earlier design\'s self-supervised LRCL/PSM losses',
           ha='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='#fffbe0', edgecolor='#999999'))

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'overall_architecture.png', dpi=200)
    plt.close(fig)


# =============================================================================
# Fig: stream_1_arch.png  (models/detector.py::Stream1, TemporalAttention)
# =============================================================================
def gen_stream1():
    fig, ax = new_fig(9.8, 2.6)
    b_in = box(ax, (0.3, 1.1), 1.8, 1.0, 'Video clip\n$T{=}32{\\times}3{\\times}224{\\times}224$', COLOR_IO, 8.5)
    b_r3d = box(ax, (2.5, 1.0), 2.2, 1.2, 'R3D-18\n(Kinetics-400\npretrained)\ngrad-checkpointed', COLOR_TEMPORAL, 8.5)
    b_feat = box(ax, (5.1, 1.1), 1.4, 1.0, '$\\mathbf{f}_{R3D}$\n$\\in\\mathbb{R}^{512}$', COLOR_TEMPORAL, 9)
    b_attn = box(ax, (6.9, 1.0), 1.6, 1.2, 'Temporal\nAttention\n$w=\\sigma(\\text{MLP}(\\mathbf{f}_{R3D}))$', COLOR_TEMPORAL, 8)
    arrow(ax, b_in, b_r3d, 'right', 'left')
    arrow(ax, b_r3d, b_feat, 'right', 'left')
    arrow(ax, b_feat, b_attn, 'right', 'left')
    ax.text(8.75, 1.75, '$\\mathbf{f}_v = w\\cdot\\mathbf{f}_{R3D}$', fontsize=9, ha='left', va='center')
    ax.annotate('', xy=(8.65, 1.6), xytext=(8.5, 1.6),
               arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.3))
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'stream_1_arch.png', dpi=200)
    plt.close(fig)


# =============================================================================
# Fig: stream_2_arch.png  (models/detector.py::Stream2, BayarConv2d, FFTBranch, HaarDWTBranch)
# =============================================================================
def gen_stream2():
    fig, ax = new_fig(10, 5)
    b_in = box(ax, (0.3, 2.0), 1.6, 1.0, '$\\mathbf{F}_{raw}\\in[0,1]$\n$3{\\times}112{\\times}112$', COLOR_IO, 8.5)

    b_bay = box(ax, (2.3, 3.4), 2.3, 1.1, 'BayarConv2d\n(constrained 5x5)\n+ 4-layer CNN', COLOR_NOISE, 8.5)
    b_fft = box(ax, (2.3, 2.0), 2.3, 1.1, 'FFT branch\nlog(1+|FFT|), DC-\nzeroed + 3-layer CNN', COLOR_NOISE, 8.5)
    b_dwt = box(ax, (2.3, 0.6), 2.3, 1.1, 'Haar DWT branch\ngrouped conv (4 sub-\nbands) + 2-layer CNN', COLOR_NOISE, 8.5)

    b_fb = box(ax, (5.3, 3.5), 1.2, 0.9, '$\\mathbf{f}_b$\n$\\mathbb{R}^{512}$', COLOR_NOISE, 8.5)
    b_ff = box(ax, (5.3, 2.05), 1.2, 0.9, '$\\mathbf{f}_f$\n$\\mathbb{R}^{128}$', COLOR_NOISE, 8.5)
    b_fd = box(ax, (5.3, 0.6), 1.2, 0.9, '$\\mathbf{f}_d$\n$\\mathbb{R}^{128}$', COLOR_NOISE, 8.5)

    b_cat = box(ax, (7.1, 1.8), 1.3, 1.4, 'Concat\n+ Linear\n+ LayerNorm', COLOR_NOISE, 8.5)
    b_out = box(ax, (8.9, 2.05), 0.9, 0.9, '$\\mathbf{f}_n$\n$\\mathbb{R}^{512}$', COLOR_NOISE, 9)

    for b_branch in (b_bay, b_fft, b_dwt):
        arrow(ax, b_in, b_branch, 'right', 'left')
    arrow(ax, b_bay, b_fb, 'right', 'left')
    arrow(ax, b_fft, b_ff, 'right', 'left')
    arrow(ax, b_dwt, b_fd, 'right', 'left')
    for b_f in (b_fb, b_ff, b_fd):
        arrow(ax, b_f, b_cat, 'right', 'left')
    arrow(ax, b_cat, b_out, 'right', 'left')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'stream_2_arch.png', dpi=200)
    plt.close(fig)


# =============================================================================
# Fig: stream_3_arch.png  (models/forensicmtf.py::Stream3V2, LandmarkAttention,
# SpatialFeatureExtractor, LocalizationHead)
# =============================================================================
def gen_stream3():
    fig, ax = new_fig(11, 5.4)
    b_in = box(ax, (0.3, 3.4), 1.7, 1.0, '$\\mathbf{F}_{norm}$\n$3{\\times}112{\\times}112$', COLOR_IO, 8.5)
    b_hm = box(ax, (0.3, 1.4), 1.7, 1.0, 'Landmark heatmap\n$H$, $112{\\times}112$', COLOR_IO, 8.5)

    b_ext = box(ax, (2.4, 3.3), 2.2, 1.2, 'SpatialFeature-\nExtractor\n4-layer conv stem', COLOR_SPATIAL, 8.5)
    b_map = box(ax, (5.0, 3.4), 1.4, 1.0, '$\\mathbf{M}$\n$128{\\times}14{\\times}14$', COLOR_SPATIAL, 9)

    b_gate = box(ax, (2.4, 1.3), 2.2, 1.2, 'Landmark gate\nAvgPool(H)→14x14\nConv-ReLU-Conv-Sig.', COLOR_SPATIAL, 8)

    b_mult = box(ax, (6.8, 2.6), 1.1, 1.1, '$\\times$\n$(1{+}\\text{gate})$', COLOR_SPATIAL, 9)
    b_mp = box(ax, (8.3, 3.3), 1.5, 1.0, "$\\mathbf{M}'$\ngated map", COLOR_SPATIAL, 8.5)

    b_pool = box(ax, (8.3, 1.7), 1.5, 1.0, 'AvgPool+Linear\n+Proj+LN', COLOR_SPATIAL, 8)
    b_fs = box(ax, (10.1, 1.9), 0.7, 0.7, '$\\mathbf{f}_s$', COLOR_SPATIAL, 9)

    b_loc = box(ax, (8.3, 4.6), 2.5, 0.8, 'Localization Head → mask logits ($28{\\times}28$)', COLOR_HEAD, 8.5)

    arrow(ax, b_in, b_ext, 'right', 'left')
    arrow(ax, b_ext, b_map, 'right', 'left')
    arrow(ax, b_map, b_mult, 'top', 'top', rad=-0.3)
    arrow(ax, b_hm, b_gate, 'right', 'left')
    arrow(ax, b_gate, b_mult, 'right', 'left')
    arrow(ax, b_mult, b_mp, 'right', 'left')
    arrow(ax, b_mp, b_pool, 'bottom', 'top')
    arrow(ax, b_pool, b_fs, 'right', 'left')
    arrow(ax, b_mp, b_loc, 'top', 'left', rad=0.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'stream_3_arch.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_overall_architecture()
    gen_stream1()
    gen_stream2()
    gen_stream3()
    print('Wrote overall_architecture.png, stream_1_arch.png, stream_2_arch.png, stream_3_arch.png')
