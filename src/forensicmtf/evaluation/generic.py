from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    return float((fpr + fnr)[np.argmin(np.abs(fpr - fnr))] / 2)


def youden_threshold(fpr, tpr, thresholds):
    return float(thresholds[np.argmax(tpr - fpr)])


def _plot_roc(fpr, tpr, auc_val, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color='#4f8ef7', label=f'AUC = {auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_score_dist(probs, labels, thr, out_path, title):
    probs, labels = np.array(probs), np.array(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs[labels == 0], bins=40, alpha=0.6, color='#2ecc71', label='Real')
    ax.hist(probs[labels == 1], bins=40, alpha=0.6, color='#e74c3c', label='Fake')
    ax.axvline(thr, color='black', linestyle='--', lw=1.5, label=f'Thr={thr:.3f}')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion(cm, out_path, title):
    disp = ConfusionMatrixDisplay(cm, display_labels=['Real', 'Fake'])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
