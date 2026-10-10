from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN as _MTCNN_CLS
    MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_CLS = None
    MTCNN_AVAILABLE = False

FF_SUBSETS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
_mtcnn_pool = {}
_mtcnn_pool_lock = threading.Lock()
_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_ff_split_metadata(root: Path, split: str) -> tuple:
    split_file = root / 'splits' / f'{split}.json'
    if not split_file.exists():
        raise FileNotFoundError(f'Split file not found: {split_file}')
    pairs = json.loads(split_file.read_text())
    split_pairs, split_real_ids = set(), set()
    for pair in pairs:
        if len(pair) != 2:
            continue
        src = f'{int(pair[0]):03d}'
        dst = f'{int(pair[1]):03d}'
        split_pairs.update([f'{src}_{dst}', f'{dst}_{src}'])
        split_real_ids.update([src, dst])
    return split_pairs, split_real_ids


def build_ff_list(root: Path, split: str, compression: str = 'c40', fake_subsets: list | None = None) -> list:
    split_pairs, split_real_ids = load_ff_split_metadata(root, split)
    fake_subsets = fake_subsets or FF_SUBSETS
    items, seen = [], set()
    real_dir = root / 'original_sequences' / 'youtube' / compression / 'videos'
    if real_dir.exists():
        for video_path in sorted(real_dir.glob('*.mp4')):
            stem = video_path.stem
            video_id = (
                f'{int(stem.split("__", 1)[0]):03d}' if '__' in stem else
                (f'{int(stem):03d}' if stem.isdigit() else None)
            )
            if video_id and video_id in split_real_ids and str(video_path) not in seen:
                items.append((str(video_path), 0, 'real'))
                seen.add(str(video_path))
    for subset in fake_subsets:
        fake_dir = root / 'manipulated_sequences' / subset / compression / 'videos'
        if fake_dir.exists():
            for video_path in sorted(fake_dir.glob('*.mp4')):
                if video_path.stem in split_pairs:
                    items.append((str(video_path), 1, subset))
    return items


def uniform_select(video_path: str, t_cand: int = 256, t_sel: int = 64) -> list:
    """Plain uniform frame selection.

    Returns indices in "candidate-index space" (i.e. indices into the t_cand-frame
    linspace sample of the video, not raw video frame numbers), which is what
    decode_frames_at_indices expects.
    """
    selected = np.linspace(0, t_cand - 1, t_sel, dtype=int).tolist()
    return selected


def decode_frames_at_indices(video_path: str, indices: list, size: int = 224, t_candidates: int = 256) -> np.ndarray:
    frames_rgb = []
    if DECORD_AVAILABLE:
        try:
            vr = VideoReader(video_path, ctx=decord_cpu(0))
            total = len(vr)
            real_idx = [min(int(i * total / t_candidates), total - 1) for i in indices]
            for frame in vr.get_batch(real_idx).asnumpy():
                frames_rgb.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))
        except Exception:
            frames_rgb = []
    if not frames_rgb:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in indices:
            real_i = min(int(i * total / t_candidates), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, real_i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            else:
                frame = frames_rgb[-1] if frames_rgb else np.zeros((size, size, 3), dtype=np.uint8)
            frames_rgb.append(frame)
        cap.release()
    return np.stack(frames_rgb).astype(np.uint8)


def _get_mtcnn():
    if not MTCNN_AVAILABLE:
        return None
    tid = threading.get_ident()
    with _mtcnn_pool_lock:
        if tid not in _mtcnn_pool:
            _mtcnn_pool[tid] = _MTCNN_CLS(
                keep_all=False,
                device=torch.device('cpu'),
                select_largest=True,
                post_process=False,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
            )
            for param in _mtcnn_pool[tid].parameters():
                param.requires_grad_(False)
    return _mtcnn_pool[tid]


def detect_face_bbox_with_meta(frame_rgb: np.ndarray) -> tuple:
    height, width = frame_rgb.shape[:2]
    detector = _get_mtcnn()
    if detector is not None:
        try:
            boxes, probs = detector.detect(frame_rgb)
            if boxes is not None and len(boxes) > 0 and probs[0] is not None:
                x1, y1, x2, y2 = [float(v) for v in boxes[0]]
                pad_x = (x2 - x1) * 0.05
                pad_y = (y2 - y1) * 0.05
                x1 = int(max(0, x1 - pad_x))
                y1 = int(max(0, y1 - pad_y))
                x2 = int(min(width - 1, x2 + pad_x))
                y2 = int(min(height - 1, y2 + pad_y))
                if x2 - x1 >= 10 and y2 - y1 >= 10:
                    return (x1, y1, x2, y2), 'mtcnn'
        except Exception:
            pass
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) > 0:
        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
        pad_x = bw * 0.05
        pad_y = bh * 0.05
        return (
            int(max(0, x - pad_x)),
            int(max(0, y - pad_y)),
            int(min(width - 1, x + bw + pad_x)),
            int(min(height - 1, y + bh + pad_y)),
        ), 'haar'
    return (width // 4, height // 4, 3 * width // 4, 3 * height // 4), 'fallback'


def detect_face_bbox(frame_rgb: np.ndarray) -> tuple:
    return detect_face_bbox_with_meta(frame_rgb)[0]


def _face_candidate_score(frame_shape: tuple, bbox: tuple, source: str) -> tuple:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area_ratio = float(bw * bh) / float(max(height * width, 1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_dx = abs(cx - width / 2.0) / max(width / 2.0, 1.0)
    center_dy = abs(cy - height / 2.0) / max(height / 2.0, 1.0)
    center_score = 1.0 - min(1.0, (center_dx + center_dy) / 2.0)
    source_bonus = 1.0 if source == 'mtcnn' else 0.5 if source == 'haar' else 0.0
    return (area_ratio, center_score, source_bonus)


def _extract_face_patch(frame_rgb: np.ndarray, bbox: tuple, img_size: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    patch = frame_rgb[y1:y2, x1:x2]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        patch = frame_rgb[img_size // 4: 3 * img_size // 4, img_size // 4: 3 * img_size // 4]
    return patch


def select_adaptive_face_crop(frames: np.ndarray, face_size: int = 112, img_size: int = 224,
                              search_frames: int = 7, min_face_width: int = 20,
                              min_face_height: int = 20, min_face_area_ratio: float = 0.02) -> np.ndarray:
    n_frames = len(frames)
    if n_frames == 0:
        return np.zeros((face_size, face_size, 3), dtype=np.uint8)
    probe_count = max(1, min(search_frames, n_frames))
    probe_indices = np.linspace(0, n_frames - 1, probe_count, dtype=int).tolist()
    best_valid = None
    best_detected = None
    for idx in probe_indices:
        frame = frames[idx]
        bbox, source = detect_face_bbox_with_meta(frame)
        patch = _extract_face_patch(frame, bbox, img_size)
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area_ratio = float(bw * bh) / float(max(frame.shape[0] * frame.shape[1], 1))
        score = _face_candidate_score(frame.shape, bbox, source)
        candidate = (score, frame, bbox)
        if source != 'fallback':
            if best_detected is None or score > best_detected[0]:
                best_detected = candidate
            if bw >= min_face_width and bh >= min_face_height and area_ratio >= min_face_area_ratio:
                if best_valid is None or score > best_valid[0]:
                    best_valid = candidate
    chosen = best_valid or best_detected
    if chosen is not None:
        _, frame, bbox = chosen
        face_patch = _extract_face_patch(frame, bbox, img_size)
    else:
        mid = frames[len(frames) // 2]
        face_patch = mid[img_size // 4: 3 * img_size // 4, img_size // 4: 3 * img_size // 4]
    return cv2.resize(face_patch, (face_size, face_size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _out_name(label: int, method: str, video_path: str) -> str:
    stem = Path(video_path).stem
    return f'real_{stem}.npz' if label == 0 else f'fake_{method}_{stem}.npz'


def _is_complete(npz_path: Path, t_select: int = 64, img_size: int = 224, face_size: int = 112) -> bool:
    if not npz_path.exists():
        return False
    try:
        data = np.load(npz_path, allow_pickle=False)
        return (
            'x' in data and data['x'].shape == (t_select, img_size, img_size, 3) and
            'face' in data and data['face'].shape == (face_size, face_size, 3) and
            'frame_idx' in data and data['frame_idx'].shape == (t_select,)
        )
    except Exception:
        return False


def _save_npz_atomic(out_path: Path, **arrays) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_stem = out_path.parent / (out_path.stem + '_writing')
    np.savez_compressed(str(tmp_stem), **arrays)
    tmp_file = tmp_stem.parent / (tmp_stem.name + '.npz')
    tmp_file.replace(out_path)


def process_one_video(video_path: str, label: int, method: str, out_dir: Path,
                      t_candidates: int = 256, t_select: int = 64,
                      img_size: int = 224, face_size: int = 112,
                      face_search_frames: int = 7, min_face_width: int = 20,
                      min_face_height: int = 20, min_face_area_ratio: float = 0.02) -> str:
    out_path = out_dir / _out_name(label, method, video_path)
    if _is_complete(out_path, t_select=t_select, img_size=img_size, face_size=face_size):
        return 'skipped'
    try:
        frame_idx = uniform_select(video_path, t_candidates, t_select)
        frames = decode_frames_at_indices(video_path, frame_idx, size=img_size, t_candidates=t_candidates)
        while len(frames) < t_select:
            frames = np.concatenate([frames, frames[-1:]], axis=0)
        frames = frames[:t_select]
        face_crop = select_adaptive_face_crop(
            frames,
            face_size=face_size,
            img_size=img_size,
            search_frames=face_search_frames,
            min_face_width=min_face_width,
            min_face_height=min_face_height,
            min_face_area_ratio=min_face_area_ratio,
        )
        _save_npz_atomic(out_path, x=frames.astype(np.uint8), face=face_crop, frame_idx=np.array(frame_idx, dtype=np.int32))
        return 'done'
    except Exception as exc:
        return f'error:{exc}'


def build_cache(items: list, split: str, cache_root: Path, num_workers: int = 4,
                t_candidates: int = 256, t_select: int = 64,
                img_size: int = 224, face_size: int = 112,
                face_search_frames: int = 7, min_face_width: int = 20,
                min_face_height: int = 20, min_face_area_ratio: float = 0.02) -> dict:
    out_dir = cache_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    todo, skipped = [], 0
    for video_path, label, method in items:
        out = out_dir / _out_name(label, method, video_path)
        if _is_complete(out, t_select=t_select, img_size=img_size, face_size=face_size):
            skipped += 1
        else:
            todo.append((
                video_path,
                label,
                method,
                out_dir,
                t_candidates,
                t_select,
                img_size,
                face_size,
                face_search_frames,
                min_face_width,
                min_face_height,
                min_face_area_ratio,
            ))
    print(f'[{split}]  total={len(items)}  todo={len(todo)}  done={skipped}')
    if not todo:
        return {'done': 0, 'skipped': skipped, 'errors': 0}
    counts = {'done': 0, 'skipped': skipped, 'errors': 0}
    errors = []
    def _handle(result, path):
        if result == 'done':
            counts['done'] += 1
        elif result == 'skipped':
            counts['skipped'] += 1
        else:
            counts['errors'] += 1
            errors.append((path, result))
    if num_workers <= 1:
        bar = tqdm(todo, desc=f'Cache [{split}]', unit='video', dynamic_ncols=True)
        for args in bar:
            _handle(process_one_video(*args), args[0])
            bar.set_postfix(done=counts['done'], err=counts['errors'])
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(process_one_video, *args): args[0] for args in todo}
            bar = tqdm(as_completed(futures), total=len(futures), desc=f'Cache [{split}]', unit='video', dynamic_ncols=True)
            for fut in bar:
                _handle(fut.result(), futures[fut])
                bar.set_postfix(done=counts['done'], err=counts['errors'])
    print(f"  done={counts['done']}  skipped={counts['skipped']}  errors={counts['errors']}")
    if errors:
        print('  First 5 errors:')
        for path, err in errors[:5]:
            print(f'    {Path(path).name}: {err}')
    return counts
