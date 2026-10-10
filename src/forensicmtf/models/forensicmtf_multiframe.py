from __future__ import annotations

import torch
import torch.nn.functional as F

from forensicmtf.models.forensicmtf import ForensicMTF

# Fair single-stream ablation fix: Stream2 (Frequency/Noise) and Stream3 (Spatial) normally
# see a crop from only ONE frame (see dfb_dataset_multiframe.py's docstring for how this
# confound was found), while Stream1 (Temporal) sees the full n_frames clip. This subclass
# accepts a (B,K,3,H,W) stack for face_01/face_norm instead of (B,3,H,W), runs Stream2/Stream3
# once per frame, and mean-pools the resulting embeddings over K before fusion -- giving those
# two streams a comparable (if still naively-pooled, not motion-aware) information budget.
# Only extract_features is overridden; forward() (inherited) dispatches to it polymorphically,
# and the localization head, fusion modules, and Stream1 are all reused completely unchanged.
# Used only by main_dfb_stream_ablation_multiframe.py -- ForensicMTF itself, and every
# existing checkpoint/script built on it, is untouched.


class ForensicMTFMultiFrame(ForensicMTF):
    def extract_features(self, video: torch.Tensor, face_01_stack: torch.Tensor, face_norm_stack: torch.Tensor,
                          landmark_heatmap: torch.Tensor | None = None):
        f_v = self.stream1(video) if self.stream1 is not None else None

        f_n = None
        if self.stream2 is not None:
            b, k = face_01_stack.shape[:2]
            flat = face_01_stack.reshape(b * k, *face_01_stack.shape[2:])
            f_n = self.stream2(flat).reshape(b, k, -1).mean(dim=1)

        f_l, spatial_map = None, None
        if self.stream3 is not None:
            b, k = face_norm_stack.shape[:2]
            flat = face_norm_stack.reshape(b * k, *face_norm_stack.shape[2:])
            heat_flat = None
            if landmark_heatmap is not None:
                heat_flat = landmark_heatmap.unsqueeze(1).expand(-1, k, *([-1] * (landmark_heatmap.dim() - 1)))
                heat_flat = heat_flat.reshape(b * k, *landmark_heatmap.shape[1:])
            f_l_flat, map_flat = self.stream3(flat, heat_flat)
            f_l = f_l_flat.reshape(b, k, -1).mean(dim=1)
            # Localization stays single-frame (the mask ground truth is for one frame too) --
            # use the middle frame's spatial map, matching the mask-generation convention.
            spatial_map = map_flat.reshape(b, k, *map_flat.shape[1:])[:, k // 2]

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
            'fused': f_final, 'clean': f_clean, 'temporal': f_v, 'noise': f_n, 'spatial': f_l,
            'spatial_map': spatial_map,
        }
