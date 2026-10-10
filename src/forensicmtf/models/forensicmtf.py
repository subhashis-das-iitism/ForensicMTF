from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from forensicmtf.data.dfb_index import NUM_MANIPULATION_CLASSES
from forensicmtf.models.detector import (
    EMBED_DIM,
    AdditionFusion,
    AttentionFusion,
    ConcatFusion,
    CrossAttentionFusion,
    Stream1,
    Stream2,
)


class SpatialFeatureExtractor(nn.Module):
    """Same conv stem as detector.py::SpatialCNN, without the final pooling/Linear head
    so the pre-pool 14x14x128 map is available for the localization head."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LandmarkAttention(nn.Module):
    """Multiplicative gating of the spatial feature map by a landmark heatmap.
    Residual (1 + gate) so it never zeroes the map even if the heatmap is degenerate
    (e.g. all-zero, from a sample with no usable landmarks). That residual guard only
    protects against a *zero* heatmap, though - it does nothing against a NaN/Inf one
    (NaN propagates through (1+gate) and poisons feat_map regardless of the residual
    structure). A single such sample, redrawn repeatedly by the balanced sampler
    (WeightedRandomSampler(..., replacement=True)), can NaN-poison a large fraction of
    an epoch's batches once this module is enabled - invisible whenever
    use_landmark_attention=False, since the heatmap is never consumed at all then.
    Sanitizing at the point of consumption is defense-in-depth regardless of whether the
    root cause is a corrupted landmark file, a degenerate detection, or something else."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(True), nn.Conv2d(8, 1, 1), nn.Sigmoid(),
        )

    def forward(self, feat_map: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(heatmap).all():
            heatmap = torch.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        h = F.adaptive_avg_pool2d(heatmap, feat_map.shape[-2:])
        gate = self.gate(h)
        if not torch.isfinite(gate).all():
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)
        out = feat_map * (1.0 + gate)
        # Sigmoid bounds gate to [0,1], so (1+gate) can amplify feat_map by at most 2x -
        # bounded on its own, but a feat_map that was already borderline-large (e.g. an
        # early-training BatchNorm spike, which Base never risks since it has no extra
        # headroom from gating) can still be pushed into fp16 overflow (>65504) by that
        # 2x. A scan of all 586,830 FF++ landmark files found zero NaN/Inf/malformed
        # entries, ruling out corrupted input data as the cause of the NaN-cascade
        # observed in the MLG-MTF ablation's landmark-attention variant - this clamp
        # targets the amplification-into-overflow mechanism directly. Purely a safety
        # net: values in the normal operating range are far below this bound and are
        # completely unaffected.
        return out.clamp(max=1e4)


class LocalizationHead(nn.Module):
    """Coarse forgery-localization decoder: 14x14x128 -> 1x28x28 mask logits.
    Intentionally shallow (one upsample step) - cheap, sufficient for IoU evaluation
    via bilinear upsampling to the GT mask resolution at loss/metric time."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.net(feat_map)


class IdentificationHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM, num_classes: int = NUM_MANIPULATION_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.fc(embedding)


class Stream3V2(nn.Module):
    """Spatial stream without the redundant frequency/texture branches from
    detector.py::Stream3 (see plan: FrequencyCNN duplicated Stream2's FFTBranch,
    TextureResidual used unconstrained kernels that could collapse toward identity).
    Optional landmark-guided attention; exposes the raw feature map for localization."""

    def __init__(self, face_size: int = 112, use_landmark_attention: bool = True):
        super().__init__()
        self.extractor = SpatialFeatureExtractor()
        self.landmark_attn = LandmarkAttention(128) if use_landmark_attention else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(128, 128)
        self.proj = nn.Sequential(nn.Linear(128, EMBED_DIM), nn.ReLU(True))
        self.norm = nn.LayerNorm(EMBED_DIM)

    def forward(self, face_norm: torch.Tensor, landmark_heatmap: torch.Tensor | None = None):
        feat_map = self.extractor(face_norm)
        if self.landmark_attn is not None and landmark_heatmap is not None:
            feat_map = self.landmark_attn(feat_map, landmark_heatmap)
        pooled = self.embed(self.pool(feat_map).flatten(1))
        embedding = self.norm(self.proj(pooled))
        return embedding, feat_map


def _make_classifier(hidden_dims: Sequence[int]) -> nn.Sequential:
    dims = [EMBED_DIM] + list(hidden_dims) + [2]
    layers: list[nn.Module] = []
    dropouts = [0.5, 0.3]
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropouts[min(idx, len(dropouts) - 1)]))
    return nn.Sequential(*layers)


class ForensicMTF(nn.Module):
    """DeepFakeBench-oriented three-stream deepfake detector: temporal (Stream1),
    noise/frequency (Stream2), and landmark-guided spatial (Stream3V2), fused and
    classified, with a mask-supervised localization head and a manipulation-method
    identification head providing directly-supervised multi-task regularization."""

    def __init__(self, pretrained: bool = True, s3_weight: float = 0.3, face_size: int = 112,
                 use_spatial: bool = True, use_noise: bool = True, use_temporal: bool = True,
                 fusion_mode: str = 'dual_cross_attention', denoise_noise_stream: bool = False,
                 denoise_kernel: int = 3, denoise_sigma: float = 0.6, feature_dropout_prob: float = 0.0,
                 use_noise_attention: bool = False, classifier_hidden_dims: Sequence[int] | None = None,
                 use_localization_head: bool = True, use_identification_head: bool = True,
                 use_landmark_attention: bool = True, num_manipulation_classes: int = NUM_MANIPULATION_CLASSES):
        super().__init__()
        if not any([use_spatial, use_noise, use_temporal]):
            raise ValueError('At least one stream must be enabled.')
        self.s3_weight = s3_weight
        self.use_spatial = use_spatial
        self.use_noise = use_noise
        self.use_temporal = use_temporal
        self.fusion_mode = fusion_mode
        self.feature_dropout_prob = feature_dropout_prob

        self.stream1 = Stream1(pretrained=pretrained) if use_temporal else None
        self.stream2 = Stream2(
            face_size=face_size,
            denoise_input=denoise_noise_stream,
            denoise_kernel=denoise_kernel,
            denoise_sigma=denoise_sigma,
            use_bayar_attention=use_noise_attention,
        ) if use_noise else None
        self.stream3 = Stream3V2(face_size=face_size, use_landmark_attention=use_landmark_attention) if use_spatial else None

        self.fusion = CrossAttentionFusion()
        self.addition_fusion = AdditionFusion()
        self.concat_fusion = ConcatFusion()
        self.attention_fusion = AttentionFusion()

        hidden_dims = list(classifier_hidden_dims) if classifier_hidden_dims is not None else [256, 128]
        self.classifier = _make_classifier(hidden_dims)

        self.use_localization_head = use_localization_head and use_spatial
        self.localization_head = LocalizationHead(128) if self.use_localization_head else None
        self.use_identification_head = use_identification_head
        self.identification_head = IdentificationHead(EMBED_DIM, num_manipulation_classes) if use_identification_head else None

    def classify_embedding(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused_embedding)

    def extract_features(self, video: torch.Tensor, face_01: torch.Tensor, face_norm: torch.Tensor,
                         landmark_heatmap: torch.Tensor | None = None):
        f_v = self.stream1(video) if self.stream1 is not None else None
        f_n = self.stream2(face_01) if self.stream2 is not None else None
        f_l, spatial_map = (self.stream3(face_norm, landmark_heatmap) if self.stream3 is not None else (None, None))

        if self.training and self.feature_dropout_prob > 0.0:
            active = [(name, feat) for name, feat in [('temporal', f_v), ('noise', f_n), ('spatial', f_l)] if feat is not None]
            if len(active) > 1 and torch.rand(1, device=video.device).item() < self.feature_dropout_prob:
                drop_idx = torch.randint(low=0, high=len(active), size=(1,), device=video.device).item()
                drop_name = active[drop_idx][0]
                if drop_name == 'temporal':
                    f_v = torch.zeros_like(f_v)
                elif drop_name == 'noise':
                    f_n = torch.zeros_like(f_n)
                elif drop_name == 'spatial':
                    f_l = torch.zeros_like(f_l)

        feats = [f for f in (f_v, f_n, f_l) if f is not None]
        if self.fusion_mode == 'dual_cross_attention':
            if f_v is not None and f_n is not None:
                f_fused, f_clean = self.fusion(f_v, f_n)
                f_final = f_fused + self.s3_weight * f_l if f_l is not None else f_fused
            elif len(feats) == 1:
                f_final, f_clean = feats[0], feats[0]
            else:
                f_final, f_clean = self.attention_fusion(feats)
        elif self.fusion_mode == 'addition':
            f_final, f_clean = self.addition_fusion(feats)
        elif self.fusion_mode == 'concatenation':
            f_final, f_clean = self.concat_fusion(feats)
        elif self.fusion_mode == 'attention':
            f_final, f_clean = self.attention_fusion(feats)
        else:
            raise ValueError(f'Unsupported fusion mode: {self.fusion_mode}')
        f_final = F.dropout(f_final, p=0.3, training=self.training)

        return {
            'fused': f_final,
            'clean': f_clean,
            'temporal': f_v,
            'noise': f_n,
            'spatial': f_l,
            'spatial_map': spatial_map,
        }

    def forward(self, video: torch.Tensor, face_01: torch.Tensor, face_norm: torch.Tensor,
               landmark_heatmap: torch.Tensor | None = None):
        details = self.extract_features(video, face_01, face_norm, landmark_heatmap)
        logits = self.classify_embedding(details['fused'])
        mask_logits = None
        if self.localization_head is not None and details['spatial_map'] is not None:
            mask_logits = self.localization_head(details['spatial_map'])
        id_logits = None
        if self.identification_head is not None:
            id_logits = self.identification_head(details['fused'])
        return logits, mask_logits, id_logits, details
