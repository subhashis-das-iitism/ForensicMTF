from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from forensicmtf.data.cache_builder import load_ff_split_metadata

FF_METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter', 'DeepFakeDetection']
MANIPULATION_CLASSES = {'real': 0, **{m: i + 1 for i, m in enumerate(FF_METHODS)}}
NUM_MANIPULATION_CLASSES = len(MANIPULATION_CLASSES)

_ID_TOKEN_RE = re.compile(r'id\d+')


def build_dfb_ffpp_index(root: Path, split: str, compressions: list, fake_subsets: list | None = None,
                         max_pairs: int | None = None) -> list:
    """Build an index of DeepFakeBench pre-extracted FF++ samples (frames/landmarks/masks dirs).

    No pixel decoding happens here - only directory listing + splits/*.json parsing.
    Fake-sample masks are always resolved against c23 (the only compression DFB ships
    masks for), regardless of which compression's frames the item points to, since the
    forgery pixel region is compression-invariant and frame filenames match across
    compressions for the same pair id.
    """
    root = Path(root)
    fake_subsets = fake_subsets or ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    split_pairs, split_real_ids = load_ff_split_metadata(root, split)
    if max_pairs:
        # Pick the first `max_pairs` distinct (unordered) id pairs and derive the real-id
        # subset from their endpoints, so fake items are guaranteed to exist for the
        # truncated set (picking the first N real ids independently can easily miss
        # every pair, since a fake sample needs BOTH its src and dst id present).
        seen_unordered = set()
        chosen_pairs = []
        for p in sorted(split_pairs):
            a, b = p.split('_')
            key = tuple(sorted((a, b)))
            if key in seen_unordered:
                continue
            seen_unordered.add(key)
            chosen_pairs.append((a, b))
            if len(chosen_pairs) >= max_pairs:
                break
        allowed_pairs = set()
        real_id_subset = set()
        for a, b in chosen_pairs:
            allowed_pairs.update([f'{a}_{b}', f'{b}_{a}'])
            real_id_subset.update([a, b])
        real_ids = sorted(real_id_subset)
    else:
        allowed_pairs = split_pairs
        real_ids = sorted(split_real_ids)

    items = []
    real_root = root / 'original_sequences' / 'youtube'
    for c in compressions:
        frames_root = real_root / c / 'frames'
        landmarks_root = real_root / c / 'landmarks'
        for rid in real_ids:
            fd = frames_root / rid
            if fd.exists():
                items.append({
                    'frames_dir': str(fd),
                    'landmarks_dir': str(landmarks_root / rid),
                    'masks_dir': None,
                    'label': 0,
                    'method': 'real',
                    'compression': c,
                    'pair_id': rid,
                })

    for subset in fake_subsets:
        masks_root = root / 'manipulated_sequences' / subset / 'c23' / 'masks'
        for c in compressions:
            frames_root = root / 'manipulated_sequences' / subset / c / 'frames'
            landmarks_root = root / 'manipulated_sequences' / subset / c / 'landmarks'
            if not frames_root.exists():
                continue
            for fd in sorted(frames_root.iterdir()):
                if not fd.is_dir():
                    continue
                pair_id = fd.name
                if pair_id not in allowed_pairs:
                    continue
                masks_dir = masks_root / pair_id
                items.append({
                    'frames_dir': str(fd),
                    'landmarks_dir': str(landmarks_root / pair_id),
                    'masks_dir': str(masks_dir) if masks_dir.exists() else None,
                    'label': 1,
                    'method': subset,
                    'compression': c,
                    'pair_id': pair_id,
                })
    return items


def build_dfb_ffpp_family_index(root: Path, compression: str, fake_subset: str, real_dir_name: str = 'youtube') -> list:
    """Flat (non-split) index for one FF++ manipulation family, used as a zero-shot
    cross-method eval target for a model that never had this fake_subset in its training
    fake_subsets. Unlike build_dfb_ffpp_index, this does not consult splits/*.json or
    pair real videos by id - some families (DeepFakeDetection) pair with a disjoint real
    pool (original_sequences/actors) that the standard youtube-based split doesn't cover,
    so every real/fake clip found on disk for this family is included."""
    root = Path(root)
    items = []
    real_frames_root = root / 'original_sequences' / real_dir_name / compression / 'frames'
    real_landmarks_root = root / 'original_sequences' / real_dir_name / compression / 'landmarks'
    if real_frames_root.exists():
        for fd in sorted(real_frames_root.iterdir()):
            if fd.is_dir():
                items.append({
                    'frames_dir': str(fd),
                    'landmarks_dir': str(real_landmarks_root / fd.name),
                    'masks_dir': None,
                    'label': 0,
                    'method': 'real',
                    'compression': compression,
                    'pair_id': fd.name,
                })
    fake_frames_root = root / 'manipulated_sequences' / fake_subset / compression / 'frames'
    fake_landmarks_root = root / 'manipulated_sequences' / fake_subset / compression / 'landmarks'
    masks_root = root / 'manipulated_sequences' / fake_subset / 'c23' / 'masks'
    if fake_frames_root.exists():
        for fd in sorted(fake_frames_root.iterdir()):
            if fd.is_dir():
                masks_dir = masks_root / fd.name
                items.append({
                    'frames_dir': str(fd),
                    'landmarks_dir': str(fake_landmarks_root / fd.name),
                    'masks_dir': str(masks_dir) if masks_dir.exists() else None,
                    'label': 1,
                    'method': fake_subset,
                    'compression': compression,
                    'pair_id': fd.name,
                })
    return items


def build_dfb_ffpp_family_index_stratified(root: Path, compression: str, fake_subset: str,
                                           real_dir_name: str = 'youtube',
                                           split_ratio: dict | None = None, seed: int = 42) -> dict:
    """Train/val/test split of one FF++ manipulation family (e.g. DeepFakeDetection) for
    in-domain (trained-and-tested-on-this-family) training, matching a baseline's own
    same-protocol evaluation (e.g. MVIM's Table III). No official split exists for this
    family (unlike the four core FF++ methods, which have splits/*.json), so this is a
    plain per-class stratified random split, not identity-disjoint -- disclosed as a
    limitation, same as build_dfb_celebdf_index_stratified."""
    items = build_dfb_ffpp_family_index(root, compression, fake_subset, real_dir_name=real_dir_name)
    split_ratio = split_ratio or {'train': 0.8, 'val': 0.1, 'test': 0.1}
    by_label = defaultdict(list)
    for it in items:
        by_label[it['label']].append(it)

    rng = random.Random(seed)
    out = {'train': [], 'val': [], 'test': []}
    for label, clips in by_label.items():
        clips = list(clips)
        rng.shuffle(clips)
        n = len(clips)
        n_val = max(1, round(n * split_ratio['val']))
        n_test = max(1, round(n * split_ratio['test']))
        out['val'].extend(clips[:n_val])
        out['test'].extend(clips[n_val:n_val + n_test])
        out['train'].extend(clips[n_val + n_test:])
    return out


def build_dfb_uadfv_index(root: Path) -> list:
    """Flat, test-only index for UADFV (real/fake dirs, no masks, no official split -
    used entirely as a zero-shot cross-domain eval set)."""
    root = Path(root)
    items = []
    for label, sub, method in [(0, 'real', 'real'), (1, 'fake', 'UADFV')]:
        frames_root = root / sub / 'frames'
        landmarks_root = root / sub / 'landmarks'
        if not frames_root.exists():
            continue
        for fd in sorted(frames_root.iterdir()):
            if fd.is_dir():
                items.append({
                    'frames_dir': str(fd),
                    'landmarks_dir': str(landmarks_root / fd.name),
                    'masks_dir': None,
                    'label': label,
                    'method': method,
                    'compression': None,
                    'pair_id': fd.name,
                })
    return items


def build_dfb_dfdc_index(root: Path) -> list:
    """Flat, test-only index for DFDC (DeepFakeBench ships only the 'test' split).
    Real/fake label comes from test/metadata.json's is_fake field, not directory
    structure - DFDC's frames all live under one shared test/frames/<video_id>/ pool.
    No landmarks are shipped for this dataset; DFBTriStreamDataset already falls back to
    a whole-frame crop and a zero landmark heatmap when landmarks_dir has no matching
    file, so this is a normal (if less-precise) eval, not a crash."""
    root = Path(root)
    meta_path = root / 'test' / 'metadata.json'
    frames_root = root / 'test' / 'frames'
    landmarks_root = root / 'test' / 'landmarks'
    meta = json.loads(meta_path.read_text())
    items = []
    for fname, info in meta.items():
        stem = fname[:-4] if fname.endswith('.mp4') else fname
        frames_dir = frames_root / stem
        if not frames_dir.is_dir():
            continue
        label = 1 if info.get('is_fake') == 1 else 0
        items.append({
            'frames_dir': str(frames_dir),
            'landmarks_dir': str(landmarks_root / stem),
            'masks_dir': None,
            'label': label,
            'method': 'real' if label == 0 else 'DFDC',
            'compression': None,
            'pair_id': stem,
        })
    return items


def _read_celebdf_test_list(list_path: Path) -> dict:
    """Parse List_of_testing_videos.txt. Label convention: label = 1 - int(prefix),
    since Celeb-DF-v1's official list uses prefix 1 for real and 0 for fake."""
    tests = {}
    for line in list_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        label = 1 - int(parts[0])
        rel = parts[1]
        category = rel.split('/')[0]
        stem = Path(rel).stem
        tests[(category, stem)] = label
    return tests


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_dfb_celebdf_index(root: Path, split_ratio: dict | None = None, seed: int = 42) -> dict:
    """Build a DeepFakeBench Celeb-DF-v1 index. Test split = official
    List_of_testing_videos.txt. Train/val = identity-disjoint split of the remaining
    clips, grouping by shared 'idN' identity tokens (connected components) so that
    neither a real identity nor either side of a synthesis swap pair leaks across
    train/val."""
    root = Path(root)
    split_ratio = split_ratio or {'train': 0.9, 'val': 0.1}
    test_list_path = root / 'List_of_testing_videos.txt'
    test_map = _read_celebdf_test_list(test_list_path) if test_list_path.exists() else {}

    categories = [('Celeb-real', 0), ('YouTube-real', 0), ('Celeb-synthesis', 1)]
    all_clips = []
    for category, label in categories:
        frames_dir = root / category / 'frames'
        if not frames_dir.exists():
            continue
        for d in sorted(frames_dir.iterdir()):
            if d.is_dir():
                all_clips.append((category, d.name, label))

    def _make_item(category, stem, label):
        return {
            'frames_dir': str(root / category / 'frames' / stem),
            'landmarks_dir': str(root / category / 'landmarks' / stem),
            'masks_dir': None,
            'label': label,
            'method': 'real' if label == 0 else 'fake',
            'compression': None,
            'pair_id': stem,
        }

    test_items = []
    trainval_clips = []
    for category, stem, label in all_clips:
        key = (category, stem)
        if key in test_map:
            test_items.append(_make_item(category, stem, test_map[key]))
        else:
            trainval_clips.append((category, stem, label))

    uf = _UnionFind()

    def _clip_key(category, stem):
        return f'{category}/{stem}'

    for category, stem, _label in trainval_clips:
        key = _clip_key(category, stem)
        tokens = _ID_TOKEN_RE.findall(stem)
        if not tokens:
            continue
        for t in tokens[1:]:
            uf.union(tokens[0], t)
        uf.union(key, tokens[0])

    groups = defaultdict(list)
    for category, stem, label in trainval_clips:
        key = _clip_key(category, stem)
        tokens = _ID_TOKEN_RE.findall(stem)
        root_key = uf.find(tokens[0]) if tokens else key
        groups[root_key].append((category, stem, label))

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    train_r = split_ratio.get('train', 0.9)
    val_r = split_ratio.get('val', 0.1)
    total = sum(len(groups[k]) for k in group_keys)
    train_target = total * train_r / max(train_r + val_r, 1e-6)

    train_items, val_items = [], []
    running = 0
    for k in group_keys:
        clips = groups[k]
        bucket = train_items if running < train_target else val_items
        bucket.extend(_make_item(c, s, l) for c, s, l in clips)
        running += len(clips)

    return {'train': train_items, 'val': val_items, 'test': test_items}


def build_dfb_celebdf_index_stratified(root: Path, val_ratio: float = 0.1, seed: int = 42) -> dict:
    """Build a DeepFakeBench Celeb-DF-v1 index for in-domain (trained-and-tested-on-Celeb-DF)
    training, matching CUTA's own evaluation protocol. Test = official
    List_of_testing_videos.txt (identity-disjoint from train/val by construction, since it is
    a fixed external list). Train/val = a per-class stratified random split of everything else.

    Unlike build_dfb_celebdf_index's identity-disjoint train/val grouping, this does NOT
    guarantee identity separation between train and val: Celeb-DF-v1's Celeb-synthesis clips
    encode two source identities per swap pair, so nearly the entire fake set forms one
    connected component under identity-based union-find, making a class-balanced
    identity-disjoint val split infeasible with so few real identities (~59). We accept
    train/val identity overlap as a disclosed limitation (Section~ref{sec:limitations})
    since it only affects early-stopping/LR-scheduling signal, not the reported test numbers,
    which remain fully identity-disjoint via the untouched official test list."""
    root = Path(root)
    test_list_path = root / 'List_of_testing_videos.txt'
    test_map = _read_celebdf_test_list(test_list_path) if test_list_path.exists() else {}

    categories = [('Celeb-real', 0), ('YouTube-real', 0), ('Celeb-synthesis', 1)]

    def _make_item(category, stem, label):
        return {
            'frames_dir': str(root / category / 'frames' / stem),
            'landmarks_dir': str(root / category / 'landmarks' / stem),
            'masks_dir': None,
            'label': label,
            'method': 'real' if label == 0 else 'fake',
            'compression': None,
            'pair_id': stem,
        }

    test_items, trainval_by_label = [], defaultdict(list)
    for category, label in categories:
        frames_dir = root / category / 'frames'
        if not frames_dir.exists():
            continue
        for d in sorted(frames_dir.iterdir()):
            if not d.is_dir():
                continue
            key = (category, d.name)
            if key in test_map:
                test_items.append(_make_item(category, d.name, test_map[key]))
            else:
                trainval_by_label[label].append((category, d.name, label))

    rng = random.Random(seed)
    train_items, val_items = [], []
    for label, clips in trainval_by_label.items():
        clips = list(clips)
        rng.shuffle(clips)
        n_val = max(1, round(len(clips) * val_ratio))
        for category, stem, lbl in clips[:n_val]:
            val_items.append(_make_item(category, stem, lbl))
        for category, stem, lbl in clips[n_val:]:
            train_items.append(_make_item(category, stem, lbl))

    return {'train': train_items, 'val': val_items, 'test': test_items}


def save_index(index_by_split: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(index_by_split, fh)


def load_index(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)
