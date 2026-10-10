from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cpu.amp.autocast.*")
FF_SUBSETS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _prune_checkpoints(checkpoints_dir: Path, keep_last: int = 10) -> None:
    checkpoints = sorted(checkpoints_dir.glob('epoch_*.pth'))
    if len(checkpoints) <= keep_last:
        return
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha_real=0.35, alpha_fake=0.65, gamma=2.0, label_smoothing=0.08):
        super().__init__()
        self.alpha = torch.tensor([alpha_real, alpha_fake])
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, labels, sample_weight=None):
        alpha = self.alpha.to(logits.device)
        ce = F.cross_entropy(logits, labels, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        loss = alpha[labels] * (1 - pt) ** self.gamma * ce
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for k, v in self.ema.state_dict().items():
            mv = model.state_dict()[k].detach()
            if not torch.is_floating_point(v):
                v.copy_(mv)
            else:
                v.mul_(self.decay).add_(mv, alpha=1 - self.decay)


def compute_auc(y_true, y_prob):
    y_prob = np.nan_to_num(np.array(y_prob, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float('nan')


def select_threshold(probs, labels):
    probs = np.nan_to_num(np.array(probs, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[np.argmax(tpr - fpr)])


def per_method_metrics(probs, labels, methods, thr, metric_names=None):
    probs = np.nan_to_num(np.array(probs, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    labels = np.array(labels)
    methods = np.array(methods)
    preds = (probs >= thr).astype(int)
    out = {}
    metric_names = metric_names or (['real'] + FF_SUBSETS)
    rm = methods == 'real'
    if rm.any():
        out['real'] = {'acc': float((preds[rm] == labels[rm]).mean()), 'auc': compute_auc((labels != 0).astype(int), probs)}
    for method in metric_names:
        if method == 'real':
            continue
        mm = methods == method
        if not mm.any():
            continue
        combo = rm | mm
        out[method] = {
            'acc': float((preds[mm] == labels[mm]).mean()),
            'auc': compute_auc(labels[combo], probs[combo]) if combo.sum() >= 2 else float('nan'),
        }
    return out
