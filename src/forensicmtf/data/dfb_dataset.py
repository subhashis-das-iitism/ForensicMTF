from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from forensicmtf.data.dfb_index import MANIPULATION_CLASSES

VIDEO_NORM = T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
FACE_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _temporal_dropout(video: torch.Tensor, k_max: int = 4) -> torch.Tensor:
    t_dim = video.shape[1]
    k = random.randint(0, k_max)
    if k == 0:
        return video
    drop = set(random.sample(range(t_dim), min(k, t_dim)))
    keep = [i for i in range(t_dim) if i not in drop]
    out = video.clone()
    for d in drop:
        out[:, d] = video[:, min(keep, key=lambda i: abs(i - d))]
    return out


def _apply_block_compression(image: torch.Tensor, min_scale: float = 0.5, max_scale: float = 0.85,
                             quant_levels: int = 32) -> torch.Tensor:
    c, h, w = image.shape
    scale = random.uniform(min_scale, max_scale)
    down_h = max(8, int(round(h * scale)))
    down_w = max(8, int(round(w * scale)))
    compressed = TF.resize(image, [down_h, down_w], interpolation=InterpolationMode.BILINEAR, antialias=True)
    compressed = TF.resize(compressed, [h, w], interpolation=InterpolationMode.NEAREST)
    levels = max(8, int(quant_levels))
    compressed = torch.round(compressed * (levels - 1)) / float(levels - 1)
    return compressed.clamp(0.0, 1.0)


def _adjust_lighting(image: torch.Tensor, brightness_delta: float = 0.2, contrast_delta: float = 0.2) -> torch.Tensor:
    brightness = random.uniform(max(0.5, 1.0 - brightness_delta), 1.0 + brightness_delta)
    contrast = random.uniform(max(0.5, 1.0 - contrast_delta), 1.0 + contrast_delta)
    image = TF.adjust_brightness(image, brightness)
    image = TF.adjust_contrast(image, contrast)
    return image.clamp(0.0, 1.0)


def _apply_rotation_crop(image: torch.Tensor, degrees: float = 5.0, crop_scale: float = 0.92) -> torch.Tensor:
    c, h, w = image.shape
    angle = random.uniform(-degrees, degrees)
    rotated = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    if crop_scale < 1.0:
        crop_h = max(8, int(round(h * random.uniform(crop_scale, 1.0))))
        crop_w = max(8, int(round(w * random.uniform(crop_scale, 1.0))))
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))
        rotated = TF.resized_crop(rotated, top, left, crop_h, crop_w, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return rotated.clamp(0.0, 1.0)


def _apply_extra_augmentations(video_01: torch.Tensor, face_01: torch.Tensor, augment_cfg: dict | None):
    if not augment_cfg:
        return video_01, face_01

    video_frames = [video_01[:, i] for i in range(video_01.shape[1])]

    if random.random() < augment_cfg.get('noise_prob', 0.0):
        sigma = float(augment_cfg.get('noise_sigma', 0.02))
        face_01 = (face_01 + torch.randn_like(face_01) * sigma).clamp(0.0, 1.0)
        video_frames = [(frame + torch.randn_like(frame) * sigma).clamp(0.0, 1.0) for frame in video_frames]

    if random.random() < augment_cfg.get('blur_prob', 0.0):
        kernel = int(augment_cfg.get('blur_kernel', 5))
        sigma = float(augment_cfg.get('blur_sigma', 1.0))
        kernel = kernel if kernel % 2 == 1 else kernel + 1
        face_01 = TF.gaussian_blur(face_01, kernel_size=kernel, sigma=sigma)
        video_frames = [TF.gaussian_blur(frame, kernel_size=kernel, sigma=sigma) for frame in video_frames]

    if random.random() < augment_cfg.get('compression_prob', 0.0):
        min_scale = float(augment_cfg.get('compression_scale_min', 0.5))
        max_scale = float(augment_cfg.get('compression_scale_max', 0.85))
        quant_levels = int(augment_cfg.get('compression_quant_levels', 32))
        face_01 = _apply_block_compression(face_01, min_scale=min_scale, max_scale=max_scale, quant_levels=quant_levels)
        video_frames = [
            _apply_block_compression(frame, min_scale=min_scale, max_scale=max_scale, quant_levels=quant_levels)
            for frame in video_frames
        ]

    if random.random() < augment_cfg.get('lighting_prob', 0.0):
        brightness_delta = float(augment_cfg.get('brightness_delta', 0.2))
        contrast_delta = float(augment_cfg.get('contrast_delta', 0.2))
        face_01 = _adjust_lighting(face_01, brightness_delta=brightness_delta, contrast_delta=contrast_delta)
        video_frames = [
            _adjust_lighting(frame, brightness_delta=brightness_delta, contrast_delta=contrast_delta)
            for frame in video_frames
        ]

    if random.random() < augment_cfg.get('rotate_crop_prob', 0.0):
        degrees = float(augment_cfg.get('rotate_degrees', 5.0))
        crop_scale = float(augment_cfg.get('crop_scale_min', 0.92))
        face_01 = _apply_rotation_crop(face_01, degrees=degrees, crop_scale=crop_scale)
        video_frames = [_apply_rotation_crop(frame, degrees=degrees, crop_scale=crop_scale) for frame in video_frames]

    video_01 = torch.stack(video_frames, dim=1).clamp(0.0, 1.0)
    face_01 = face_01.clamp(0.0, 1.0)
    return video_01, face_01


def _list_frame_files(frames_dir) -> list:
    return sorted(Path(frames_dir).glob('*.png'))


def _landmark_path_for(landmarks_dir, frame_path: Path) -> Path:
    return Path(landmarks_dir) / (frame_path.stem + '.npy')


def _mask_path_for(masks_dir, frame_path: Path):
    if masks_dir is None:
        return None
    p = Path(masks_dir) / frame_path.name
    return p if p.exists() else None


def _bbox_from_landmarks(landmarks: np.ndarray, width: int, height: int, margin: float = 0.25):
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)
    w = max(float(x_max - x_min), 1.0)
    h = max(float(y_max - y_min), 1.0)
    pad_x = w * margin
    pad_y = h * margin
    x1 = int(max(0, x_min - pad_x))
    y1 = int(max(0, y_min - pad_y))
    x2 = int(min(width, x_max + pad_x))
    y2 = int(min(height, y_max + pad_y))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return (0, 0, width, height)
    return (x1, y1, x2, y2)


def _make_landmark_heatmap(landmarks: np.ndarray, bbox, out_size: int = 112, sigma: float = 2.0) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    scale_x = out_size / bw
    scale_y = out_size / bh
    pts = np.empty_like(landmarks, dtype=np.float32)
    pts[:, 0] = (landmarks[:, 0] - x1) * scale_x
    pts[:, 1] = (landmarks[:, 1] - y1) * scale_y
    yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
    diff_x = xx[None, :, :] - pts[:, 0, None, None]
    diff_y = yy[None, :, :] - pts[:, 1, None, None]
    gauss = np.exp(-(diff_x ** 2 + diff_y ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return gauss.max(axis=0)


def _apply_geometric_aug_with_mask(face_01: torch.Tensor, mask: torch.Tensor, heatmap: torch.Tensor,
                                   degrees: float = 5.0, crop_scale: float = 0.92):
    _, h, w = face_01.shape
    angle = random.uniform(-degrees, degrees)
    face_01 = TF.rotate(face_01, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)
    heatmap = TF.rotate(heatmap, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    if crop_scale < 1.0:
        crop_h = max(8, int(round(h * random.uniform(crop_scale, 1.0))))
        crop_w = max(8, int(round(w * random.uniform(crop_scale, 1.0))))
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))
        face_01 = TF.resized_crop(face_01, top, left, crop_h, crop_w, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resized_crop(mask, top, left, crop_h, crop_w, [h, w], interpolation=InterpolationMode.NEAREST)
        heatmap = TF.resized_crop(heatmap, top, left, crop_h, crop_w, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return face_01.clamp(0.0, 1.0), mask.clamp(0.0, 1.0), heatmap.clamp(0.0, 1.0)


class DFBTriStreamDataset(Dataset):
    """Reads DeepFakeBench pre-extracted frames/landmarks/masks directly from disk.

    No video decoding, no face re-detection (face crop is derived from the provided
    ground-truth landmarks). Always emits 'mask'/'mask_valid'/'method_id'/
    'landmark_heatmap' keys so downstream code never key-presence-branches, only
    value-branches on mask_valid / method_id == -1.
    """

    def __init__(self, index: list, n_frames: int = 32, face_size: int = 112, img_size: int = 224,
                 training: bool = False, augment_cfg: dict | None = None):
        self.index = index
        self.n_frames = n_frames
        self.face_size = face_size
        self.img_size = img_size
        self.training = training
        self.augment_cfg = augment_cfg or {}

    def __len__(self):
        return len(self.index)

    def get_method(self, idx: int) -> str:
        return self.index[idx]['method']

    def __getitem__(self, idx: int):
        try:
            return self._load(self.index[idx])
        except Exception:
            return self.__getitem__(random.randint(0, len(self.index) - 1))

    def _load(self, item: dict):
        frame_files = _list_frame_files(item['frames_dir'])
        n_avail = len(frame_files)
        if n_avail == 0:
            raise RuntimeError(f"no frames in {item['frames_dir']}")
        n_take = min(self.n_frames, n_avail)
        sel_idx = np.linspace(0, n_avail - 1, n_take).astype(int)
        sel_files = [frame_files[i] for i in sel_idx]
        while len(sel_files) < self.n_frames:
            sel_files.append(sel_files[-1])

        frames = []
        for fp in sel_files:
            img = cv2.imread(str(fp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            frames.append(img)
        video_np = np.stack(frames, axis=0)

        mid = sel_files[len(sel_files) // 2]
        native_img = cv2.imread(str(mid))
        native_img = cv2.cvtColor(native_img, cv2.COLOR_BGR2RGB)
        nh, nw = native_img.shape[:2]

        lm_path = _landmark_path_for(item['landmarks_dir'], mid)
        landmarks = np.load(lm_path).astype(np.float32) if lm_path.exists() else None
        bbox = _bbox_from_landmarks(landmarks, nw, nh) if landmarks is not None else (0, 0, nw, nh)
        x1, y1, x2, y2 = bbox

        face_crop = native_img[y1:y2, x1:x2]
        if face_crop.shape[0] < 4 or face_crop.shape[1] < 4:
            face_crop = native_img
        face_crop = cv2.resize(face_crop, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)

        if landmarks is not None:
            heatmap = _make_landmark_heatmap(landmarks, bbox, out_size=self.face_size)
        else:
            heatmap = np.zeros((self.face_size, self.face_size), dtype=np.float32)

        mask_path = _mask_path_for(item.get('masks_dir'), mid)
        if mask_path is not None:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_crop = mask_img[y1:y2, x1:x2]
            if mask_crop.shape[0] < 4 or mask_crop.shape[1] < 4:
                mask_crop = mask_img
            mask_crop = cv2.resize(mask_crop, (self.face_size, self.face_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask_crop > 127).astype(np.float32)
            mask_valid = True
        else:
            mask = np.zeros((self.face_size, self.face_size), dtype=np.float32)
            mask_valid = False

        video_01 = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
        face_01 = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        if self.training:
            video_01 = _temporal_dropout(video_01, k_max=4)
            if random.random() > 0.5:
                video_01 = torch.flip(video_01, dims=[-1])
                face_01 = torch.flip(face_01, dims=[-1])
                heatmap_t = torch.flip(heatmap_t, dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if random.random() < 0.5:
                face_01 = (face_01 + torch.randn_like(face_01) * 0.02).clamp(0.0, 1.0)
            video_01, face_01 = _apply_extra_augmentations(video_01, face_01, self.augment_cfg)
            if random.random() < self.augment_cfg.get('rotate_crop_prob', 0.0):
                face_01, mask_t, heatmap_t = _apply_geometric_aug_with_mask(
                    face_01, mask_t, heatmap_t,
                    degrees=self.augment_cfg.get('rotate_degrees', 5.0),
                    crop_scale=self.augment_cfg.get('crop_scale_min', 0.92),
                )

        face_norm = FACE_NORM(face_01.clone())
        video = video_01.permute(1, 0, 2, 3)
        video = VIDEO_NORM(video)
        video = video.permute(1, 0, 2, 3)

        return {
            'video': video,
            'face_01': face_01,
            'face_norm': face_norm,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'method': item['method'],
            'mask': mask_t,
            'mask_valid': torch.tensor(mask_valid),
            'method_id': torch.tensor(MANIPULATION_CLASSES.get(item['method'], -1), dtype=torch.long),
            'landmark_heatmap': heatmap_t,
        }


def collate_dfb_batch(batch):
    return {
        'video': torch.stack([b['video'] for b in batch]),
        'face_01': torch.stack([b['face_01'] for b in batch]),
        'face_norm': torch.stack([b['face_norm'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'method': [b['method'] for b in batch],
        'mask': torch.stack([b['mask'] for b in batch]),
        'mask_valid': torch.stack([b['mask_valid'] for b in batch]),
        'method_id': torch.stack([b['method_id'] for b in batch]),
        'landmark_heatmap': torch.stack([b['landmark_heatmap'] for b in batch]),
    }


def make_dfb_balanced_sampler(ds: DFBTriStreamDataset) -> WeightedRandomSampler:
    methods = [ds.get_method(i) for i in range(len(ds))]
    counts = Counter(methods)
    weights = torch.tensor([1.0 / counts[m] for m in methods], dtype=torch.float32)
    n_samples = min(counts.values()) * len(counts)
    print(f'  Balanced sampler: {dict(counts)}')
    print(f'  Samples per epoch: {n_samples}')
    return WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)


def _make_dataset(items: list, cfg: dict, training: bool) -> DFBTriStreamDataset:
    return DFBTriStreamDataset(
        items,
        n_frames=cfg.get('n_frames', 32),
        face_size=cfg.get('face_size', 112),
        img_size=cfg.get('img_size', 224),
        training=training,
        augment_cfg=cfg.get('augmentations') if training else None,
    )


def get_dfb_loaders(index_by_split: dict, cfg: dict):
    train_ds = _make_dataset(index_by_split['train'], cfg, training=True)
    val_ds = _make_dataset(index_by_split['val'], cfg, training=False)
    test_ds = _make_dataset(index_by_split['test'], cfg, training=False)
    sampler = make_dfb_balanced_sampler(train_ds)
    kw = dict(num_workers=cfg.get('num_workers', 4), pin_memory=True, collate_fn=collate_dfb_batch)
    train_dl = DataLoader(train_ds, batch_size=cfg.get('batch_size', 4), sampler=sampler, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=cfg.get('batch_size', 4), shuffle=False, **kw)
    test_dl = DataLoader(test_ds, batch_size=cfg.get('batch_size', 4), shuffle=False, **kw)
    print(f'  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}')
    return train_dl, val_dl, test_dl


def get_dfb_eval_loader(items: list, cfg: dict) -> DataLoader:
    ds = _make_dataset(items, cfg, training=False)
    return DataLoader(
        ds, batch_size=cfg.get('batch_size', 4), shuffle=False,
        num_workers=cfg.get('num_workers', 4), pin_memory=True, collate_fn=collate_dfb_batch,
    )
