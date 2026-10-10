from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from forensicmtf.data.dfb_dataset import (
    FACE_NORM,
    DFBTriStreamDataset,
    _bbox_from_landmarks,
    _landmark_path_for,
    _list_frame_files,
    collate_dfb_batch,
    make_dfb_balanced_sampler,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Fixes a real confound found in the T/S/N stream ablation: Stream1 (Temporal) sees the full
# n_frames (32) clip, but Stream2 (Frequency/Noise) and Stream3 (Spatial) both only ever see a
# crop from ONE frame (the clip's middle frame) -- see DFBTriStreamDataset._load. So "Temporal
# only" vs "Frequency only"/"Spatial only" was measuring "32 frames vs 1 frame", not "motion
# cues vs frequency/spatial cues". This subclass replaces the single-frame face_01/face_norm
# with a (stream23_frames, 3, H, W) stack (own bbox/landmarks per selected frame) so a fair
# single-stream ablation can be re-run with a comparable information budget -- paired with
# ForensicMTFMultiFrame, which mean-pools Stream2/Stream3 over that stack.
# Deliberately keeps the 'face_01'/'face_norm' key names (rather than adding new keys) so the
# existing, unmodified collate_dfb_batch/run_epoch_dfb/evaluate_dfb_loader all work as-is --
# they just end up carrying (B,K,3,H,W) tensors instead of (B,3,H,W) for this dataset only.
# `video`/mask/heatmap/label are untouched -- purely additive; DFBTriStreamDataset itself
# stays exactly as-is for every other caller.


class DFBTriStreamDatasetMultiFrame(DFBTriStreamDataset):
    def __init__(self, index: list, n_frames: int = 32, face_size: int = 112, img_size: int = 224,
                 training: bool = False, augment_cfg: dict | None = None, stream23_frames: int = 8):
        super().__init__(index, n_frames, face_size, img_size, training, augment_cfg)
        self.stream23_frames = stream23_frames

    def _load(self, item: dict):
        sample = super()._load(item)  # keeps video/mask/heatmap/label logic byte-identical

        frame_files = _list_frame_files(item['frames_dir'])
        n_avail = len(frame_files)
        k = min(self.stream23_frames, n_avail)
        sel_idx = np.linspace(0, n_avail - 1, k).astype(int)
        sel_files = [frame_files[i] for i in sel_idx]

        crops = []
        for fp in sel_files:
            img = cv2.imread(str(fp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            nh, nw = img.shape[:2]
            lm_path = _landmark_path_for(item['landmarks_dir'], fp)
            landmarks = np.load(lm_path).astype(np.float32) if lm_path.exists() else None
            bbox = _bbox_from_landmarks(landmarks, nw, nh) if landmarks is not None else (0, 0, nw, nh)
            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                crop = img
            crop = cv2.resize(crop, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
            crops.append(crop)
        while len(crops) < self.stream23_frames:
            crops.append(crops[-1])

        # Deliberately no flip/geometric augmentation on the stack (unlike the single-frame
        # face_01 the parent _load may have flipped) -- this is a supplementary ablation, not
        # the headline model, and keeping the multi-frame sampling simple avoids a second,
        # easy-to-get-wrong source of train/eval mismatch.
        stack01 = torch.from_numpy(np.stack(crops, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # (K,3,H,W)
        stack_norm = FACE_NORM(stack01.clone())

        sample['face_01'] = stack01
        sample['face_norm'] = stack_norm
        return sample


def _make_dataset_mf(items: list, cfg: dict, training: bool) -> DFBTriStreamDatasetMultiFrame:
    return DFBTriStreamDatasetMultiFrame(
        items,
        n_frames=cfg.get('n_frames', 32),
        face_size=cfg.get('face_size', 112),
        img_size=cfg.get('img_size', 224),
        training=training,
        augment_cfg=cfg.get('augmentations') if training else None,
        stream23_frames=cfg.get('stream23_frames', 8),
    )


def get_dfb_loaders_multiframe(index_by_split: dict, cfg: dict):
    train_ds = _make_dataset_mf(index_by_split['train'], cfg, training=True)
    val_ds = _make_dataset_mf(index_by_split['val'], cfg, training=False)
    test_ds = _make_dataset_mf(index_by_split['test'], cfg, training=False)
    sampler = make_dfb_balanced_sampler(train_ds)
    kw = dict(num_workers=cfg.get('num_workers', 4), pin_memory=True, collate_fn=collate_dfb_batch)
    train_dl = DataLoader(train_ds, batch_size=cfg.get('batch_size', 4), sampler=sampler, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=cfg.get('batch_size', 4), shuffle=False, **kw)
    test_dl = DataLoader(test_ds, batch_size=cfg.get('batch_size', 4), shuffle=False, **kw)
    print(f'  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  (stream23_frames={train_ds.stream23_frames})')
    return train_dl, val_dl, test_dl


def get_dfb_eval_loader_multiframe(items: list, cfg: dict) -> DataLoader:
    ds = _make_dataset_mf(items, cfg, training=False)
    return DataLoader(
        ds, batch_size=cfg.get('batch_size', 4), shuffle=False,
        num_workers=cfg.get('num_workers', 4), pin_memory=True, collate_fn=collate_dfb_batch,
    )
