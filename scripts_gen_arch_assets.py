"""Render every thumbnail used inside the TriForensics architecture diagram from the real
FF++ data and the trained checkpoint, so no panel in the figure is a stock illustration.

Each asset is written as a borderless square PNG at ASSET_PX so it can be dropped straight
into the diagram. Everything is produced by the same code paths the model uses at inference:
DFBTriStreamDataset for the preprocessing panels, Stream2's own BayarConv/FFTBranch._spectrum/
Haar kernels for the noise-frequency panels, and Stream3V2 + LocalizationHead (loaded from
best.pth) for the M' and predicted-mask panels. Nothing here is re-implemented approximately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import load_config                                    # noqa: E402
from forensicmtf.data.dfb_dataset import (DFBTriStreamDataset, _bbox_from_landmarks,  # noqa: E402
                                          _list_frame_files, _make_landmark_heatmap)
from forensicmtf.dfb_runner import _eval_model_kwargs, resolve_path           # noqa: E402
from forensicmtf.models.forensicmtf import ForensicMTF                 # noqa: E402

ASSET_PX = 512          # every square panel is rendered at this resolution
STRIP_FRAMES = 8        # frames shown in the Stream-1 filmstrip / stack


# --------------------------------------------------------------------------- io helpers
def save(path: Path, img: np.ndarray, size: int | None = ASSET_PX, interp=cv2.INTER_NEAREST):
    """`img` is HxW (gray) or HxWx3 RGB, uint8. Upscaled with NEAREST by default so the
    native resolution of a feature map stays visible rather than being blurred away."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if size is not None and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=interp)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f'  {path.name:34s} {img.shape[1]}x{img.shape[0]}')


def norm8(x: np.ndarray) -> np.ndarray:
    """Per-image min-max to uint8. Used for every feature-map / spectrum panel so the
    structure is visible regardless of the raw activation scale."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x, dtype=np.uint8) if hi - lo < 1e-8 else \
        ((x - lo) / (hi - lo) * 255.0).astype(np.uint8)


def colormap(x8: np.ndarray, cmap=cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    return cv2.cvtColor(cv2.applyColorMap(x8, cmap), cv2.COLOR_BGR2RGB)


def tile(images: list[np.ndarray], rows: int, cols: int, pad: int = 6,
         bg: int = 255) -> np.ndarray:
    """Contact sheet of equal-sized panels (used for the BayarConv channel grid and the
    DWT sub-band quad that appear as small tile rows in the diagram)."""
    h, w = images[0].shape[:2]
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i, im in enumerate(images[:rows * cols]):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im
    return canvas


# --------------------------------------------------------------------------- sample pick
def pick_sample(index: list, method: str, require_mask: bool) -> dict:
    for item in index:
        if item['method'] != method:
            continue
        if require_mask and not item.get('masks_dir'):
            continue
        if _list_frame_files(item['frames_dir']):
            return item
    raise SystemExit(f'no usable {method} sample in the index')


def pick_fake_with_good_mask(index: list, method: str, n_scan: int = 40) -> dict:
    """Prefer a sample whose ground-truth mask actually shows the manipulated region.

    Some FF++ masks degenerate to the full crop or to a plain rectangle, which makes a
    useless figure panel. Score candidates by how much of the frame they cover (want a
    clear but not total region) and by how far the mask is from filling its own bounding
    box, and take the best.
    """
    best, best_score = None, -1.0
    for item in index:
        if item['method'] != method or not item.get('masks_dir'):
            continue
        files = _list_frame_files(item['frames_dir'])
        if not files:
            continue
        mid = files[len(files) // 2]
        mp = Path(item['masks_dir']) / mid.name
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
        area = float(m.mean())
        if not 0.05 < area < 0.55:
            continue
        ys, xs = np.where(m)
        bb = max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        fill = m.sum() / bb                       # 1.0 == a perfect rectangle
        score = (1.0 - abs(area - 0.25) / 0.25) + 2.0 * (1.0 - fill)
        if score > best_score:
            best, best_score = item, score
        n_scan -= 1
        if n_scan <= 0:
            break
    return best if best is not None else pick_sample(index, method, require_mask=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config_dfb.yaml')
    ap.add_argument('--compression', default='c23', choices=['c23', 'c40'],
                    help='c23 reads more clearly at thumbnail size; c40 matches the headline model.')
    ap.add_argument('--method', default='Deepfakes')
    ap.add_argument('--checkpoint', default=None, help='defaults to records/DFB/models_ffpp_<compression>/best.pth')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    records = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / 'records' / 'DFB' / 'figures' / 'arch_assets'
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    index = json.load(open(records / 'index' / f'ffpp_{args.compression}.json'))['test']
    fake = pick_fake_with_good_mask(index, args.method)
    # the fake's own source video (pair_id '128_896' -> real '128'), so the real/fake
    # panels in the figure show the same identity
    src_id = fake['pair_id'].split('_')[0]
    real = next((it for it in index if it['method'] == 'real' and it['pair_id'] == src_id),
                pick_sample(index, 'real', require_mask=False))
    print(f'fake sample: {fake["pair_id"]} ({fake["method"]}, {args.compression})')
    print(f'real sample: {real["pair_id"]}')

    ffpp = cfg['dfb']['ffpp']
    face_size, img_size = ffpp.get('face_size', 112), ffpp.get('img_size', 224)
    n_frames = ffpp.get('n_frames', 32)
    ds = DFBTriStreamDataset([fake, real], n_frames=n_frames, face_size=face_size,
                             img_size=img_size, training=False)
    s_fake, s_real = ds[0], ds[1]

    # ---------------------------------------------------------------- 1. preprocessing
    print('\n[1/5] Data & Preprocessing')
    frame_files = _list_frame_files(fake['frames_dir'])
    sel = np.linspace(0, len(frame_files) - 1, min(n_frames, len(frame_files))).astype(int)
    mid_path = frame_files[sel[len(sel) // 2]]
    native = cv2.cvtColor(cv2.imread(str(mid_path)), cv2.COLOR_BGR2RGB)
    nh, nw = native.shape[:2]
    save(out_dir / '01_raw_frame.png', native, interp=cv2.INTER_AREA)

    lm = np.load(Path(fake['landmarks_dir']) / (mid_path.stem + '.npy')).astype(np.float32)
    bbox = _bbox_from_landmarks(lm, nw, nh)
    # Landmark + bbox overlay: exactly the crop window _load() uses (25% margin box).
    ov = native.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(ov, (x1, y1), (x2, y2), (66, 133, 244), max(2, nw // 160))
    for px, py in lm:
        cv2.circle(ov, (int(px), int(py)), max(1, nw // 220), (234, 67, 53), -1)
    save(out_dir / '02_landmark_bbox.png', ov, interp=cv2.INTER_AREA)

    face_u8 = (s_fake['face_01'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    save(out_dir / '03_face_crop.png', face_u8)
    save(out_dir / '03b_face_crop_real.png',
         (s_real['face_01'].permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    heat = _make_landmark_heatmap(lm, bbox, out_size=face_size)
    save(out_dir / '04_landmark_heatmap.png', colormap(norm8(heat), cv2.COLORMAP_INFERNO))
    save(out_dir / '04b_landmark_heatmap_gray.png', norm8(heat))
    # Heatmap over the face, i.e. what the Stream-3 gate is actually conditioned on.
    save(out_dir / '04c_heatmap_on_face.png',
         cv2.addWeighted(face_u8, 0.55, colormap(norm8(heat), cv2.COLORMAP_INFERNO), 0.45, 0))

    gt = (s_fake['mask'][0].numpy() * 255).astype(np.uint8)
    save(out_dir / '05_gt_mask.png', gt)
    save(out_dir / '05b_gt_mask_on_face.png',
         cv2.addWeighted(face_u8, 0.6, cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB), 0.4, 0))

    # ---------------------------------------------------------------- 2. stream 1
    print('\n[2/5] Stream 1 (temporal)')
    strip_idx = np.linspace(0, len(sel) - 1, STRIP_FRAMES).astype(int)
    vid01 = s_fake['video']  # normalised; re-read the raw frames instead for display
    frames_u8 = []
    for i in sel[strip_idx]:
        f = cv2.cvtColor(cv2.imread(str(frame_files[i])), cv2.COLOR_BGR2RGB)
        frames_u8.append(cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA))
    strip = np.concatenate([np.pad(f, ((0, 0), (0, 8), (0, 0)), constant_values=255)
                            for f in frames_u8], axis=1)[:, :-8]
    cv2.imwrite(str(out_dir / '10_filmstrip.png'), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f'  10_filmstrip.png                   {strip.shape[1]}x{strip.shape[0]}')

    # Perspective "stack of frames" for the Stream-1 box: later frames drawn behind and
    # offset, so T=32 sampling reads as a clip rather than a single image.
    n_stack, off = 5, 34
    canvas = np.full((img_size + off * (n_stack - 1) + 12, img_size + off * (n_stack - 1) + 12, 3), 255, np.uint8)
    for k in range(n_stack - 1, -1, -1):
        f = frames_u8[int(k * (len(frames_u8) - 1) / (n_stack - 1))]
        y, x = k * off, (n_stack - 1 - k) * off
        canvas[y:y + img_size, x:x + img_size] = f
        cv2.rectangle(canvas, (x, y), (x + img_size - 1, y + img_size - 1), (40, 40, 40), 2)
    cv2.imwrite(str(out_dir / '11_frame_stack.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f'  11_frame_stack.png                 {canvas.shape[1]}x{canvas.shape[0]}')

    # ---------------------------------------------------------------- 3. stream 2
    print('\n[3/5] Stream 2 (noise / frequency)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.checkpoint) if args.checkpoint else records / f'models_ffpp_{args.compression}' / 'best.pth'
    model_kwargs = _eval_model_kwargs(dict(cfg.get('train', {})))
    model = ForensicMTF(pretrained=False, **model_kwargs).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) and ('ema' in ckpt or 'model' in ckpt) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  loaded {ckpt_path.name}  (missing={len(missing)}, unexpected={len(unexpected)})')

    face01 = s_fake['face_01'].unsqueeze(0).to(device)
    with torch.no_grad():
        bayar = model.stream2.bayar(face01).clamp(-2.0, 2.0)[0].cpu().numpy()   # 32 x 112 x 112
        spec = model.stream2.fft._spectrum(face01)[0, 0].cpu().numpy()          # DC-notched log spectrum
        sub = F.conv2d(face01, model.stream2.dwt.haar, stride=2, groups=3)[0].cpu().numpy()

    save(out_dir / '20_bayar_residual.png', norm8(np.abs(bayar).mean(0)))
    # The 8 highest-energy Bayar channels, i.e. the filters actually carrying the residual.
    order = np.argsort(-np.abs(bayar).reshape(len(bayar), -1).std(1))
    save(out_dir / '21_bayar_grid.png',
         tile([norm8(bayar[i]) for i in order[:8]], rows=2, cols=4), size=None)
    save(out_dir / '22_fft_spectrum.png', colormap(norm8(spec), cv2.COLORMAP_VIRIDIS))
    save(out_dir / '22b_fft_spectrum_gray.png', norm8(spec))
    # Haar sub-bands: 12 channels = 4 bands x 3 colour planes; show the luminance-ish mean.
    bands = ['LL', 'LH', 'HL', 'HH']
    band_imgs = [cv2.resize(norm8(sub[[b, b + 4, b + 8]].mean(0)), (224, 224),
                            interpolation=cv2.INTER_NEAREST) for b in range(4)]
    for name, im in zip(bands, band_imgs):
        save(out_dir / f'23_dwt_{name}.png', im)
    save(out_dir / '23_dwt_subbands.png', tile(band_imgs, rows=2, cols=2), size=None)

    # ---------------------------------------------------------------- 4. stream 3 + heads
    print('\n[4/5] Stream 3 (spatial) + heads')
    face_norm = s_fake['face_norm'].unsqueeze(0).to(device)
    heat_t = s_fake['landmark_heatmap'].unsqueeze(0).to(device)
    with torch.no_grad():
        raw_map = model.stream3.extractor(face_norm)                       # pre-gate 14x14x128
        h_ds = F.adaptive_avg_pool2d(heat_t, raw_map.shape[-2:])
        gate = model.stream3.landmark_attn.gate(h_ds)                      # the learned gate
        _, m_prime = model.stream3(face_norm, heat_t)                      # M' = gated map
        mask_logits = model.localization_head(m_prime)
        logits, _, id_logits, _ = model(s_fake['video'].unsqueeze(0).to(device),
                                        face01, face_norm, landmark_heatmap=heat_t)

    save(out_dir / '30_spatial_prepool.png', colormap(norm8(raw_map[0].mean(0).cpu().numpy())))
    save(out_dir / '31_landmark_gate.png', colormap(norm8(gate[0, 0].cpu().numpy()), cv2.COLORMAP_INFERNO))
    mp = norm8(m_prime[0].mean(0).cpu().numpy())
    save(out_dir / '32_spatial_map_Mprime.png', colormap(mp))
    save(out_dir / '32b_spatial_map_Mprime_gray.png', mp)
    save(out_dir / '32c_Mprime_on_face.png',
         cv2.addWeighted(face_u8, 0.55,
                         cv2.resize(colormap(mp), (face_size, face_size), interpolation=cv2.INTER_CUBIC),
                         0.45, 0))

    prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()                  # 28x28 -> upsampled
    save(out_dir / '40_pred_mask_prob.png', colormap(norm8(prob), cv2.COLORMAP_MAGMA))
    save(out_dir / '41_pred_mask_binary.png', ((prob > 0.5) * 255).astype(np.uint8))
    save(out_dir / '42_pred_mask_on_face.png',
         cv2.addWeighted(face_u8, 0.6,
                         cv2.resize(colormap(norm8(prob), cv2.COLORMAP_MAGMA), (face_size, face_size),
                                    interpolation=cv2.INTER_CUBIC), 0.4, 0))
    # Bicubic companions: the panels above keep the true 14x14 / 28x28 grid visible, these
    # read better when shrunk into a diagram box.
    save(out_dir / '32d_spatial_map_Mprime_smooth.png', colormap(mp), interp=cv2.INTER_CUBIC)
    save(out_dir / '31b_landmark_gate_smooth.png',
         colormap(norm8(gate[0, 0].cpu().numpy()), cv2.COLORMAP_INFERNO), interp=cv2.INTER_CUBIC)
    save(out_dir / '40b_pred_mask_prob_smooth.png',
         colormap(norm8(prob), cv2.COLORMAP_MAGMA), interp=cv2.INTER_CUBIC)

    # Identification head: there is no natural "picture" for a 7-way method label, so the
    # figure panel is the head's actual softmax over the classes. All seven outputs are
    # plotted (the head is built with NUM_MANIPULATION_CLASSES=7); FaceShifter and
    # DeepFakeDetection sit at ~0 because config_dfb.yaml trains on the other five only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from forensicmtf.data.dfb_index import MANIPULATION_CLASSES

    id_p = torch.softmax(id_logits, dim=-1)[0].cpu().numpy()
    short = {'real': 'Real', 'Deepfakes': 'DF', 'Face2Face': 'F2F', 'FaceSwap': 'FS',
             'NeuralTextures': 'NT', 'FaceShifter': 'FSh', 'DeepFakeDetection': 'DFD'}
    names = [short[k] for k, _ in sorted(MANIPULATION_CLASSES.items(), key=lambda kv: kv[1])]
    win = int(id_p.argmax())
    def _style(ax):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # (a) vertical bars, square — only legible if the diagram box is >= ~140 px wide.
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=ASSET_PX / 3.4)
    ax.bar(range(len(id_p)), id_p, color=['#c9302c' if i == win else '#b8bcc4' for i in range(len(id_p))],
           edgecolor='#333', linewidth=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.text(win, id_p[win] + 0.03, f'{id_p[win]:.2f}', ha='center', fontsize=14, fontweight='bold')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.4)
    ax.tick_params(axis='x', length=0, pad=4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / '50_id_head_softmax.png', facecolor='white')
    plt.close(fig)

    # (b) horizontal bars: the class name sits on its own row, so the labels stay legible
    # when the panel is shrunk into a diagram box -- text height scales with the box height
    # rather than with 1/7th of its width.
    fig, ax = plt.subplots(figsize=(3.0, 3.2), dpi=ASSET_PX / 3.2)
    y = range(len(id_p))[::-1]
    ax.barh(list(y), id_p, height=0.72,
            color=['#c9302c' if i == win else '#c3c7cf' for i in range(len(id_p))],
            edgecolor='#333', linewidth=1.0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=19, fontweight='bold')
    ax.set_xlim(0, 1.18)
    ax.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(1.6)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.text(id_p[win] + 0.04, list(y)[win], f'{id_p[win]:.2f}', va='center',
            fontsize=18, fontweight='bold', color='#c9302c')
    fig.tight_layout(pad=0.25)
    fig.savefig(out_dir / '52_id_head_barh.png', facecolor='white')
    plt.close(fig)

    # (c) readout: the predicted class as a word, above a 7-cell strip. The dominant element
    # is one large word, so this is the only variant that still reads at ~66 px.
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=ASSET_PX / 3.4)
    ax.axis('off')
    ax.text(0.5, 0.93, 'predicted method', ha='center', va='top', fontsize=15, color='#555')
    ax.text(0.5, 0.60, names[win], ha='center', va='center', fontsize=40,
            fontweight='bold', color='#c9302c')
    ax.text(0.5, 0.34, f'p = {id_p[win]:.2f}', ha='center', va='center', fontsize=17, color='#333')
    for i, v in enumerate(id_p):
        x0 = 0.06 + i * (0.88 / len(id_p))
        w = 0.88 / len(id_p) - 0.012
        ax.add_patch(plt.Rectangle((x0, 0.06), w, 0.14, transform=ax.transData,
                                   facecolor='#c9302c' if i == win else '#d7dae0',
                                   edgecolor='#555', linewidth=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_dir / '53_id_head_readout.png', facecolor='white')
    plt.close(fig)
    print(f"  50/52/53_id_head_*.png             argmax={names[win]} p={id_p[win]:.3f}")

    p_fake = torch.softmax(logits, dim=-1)[0, 1].item()
    print(f'  p(fake)={p_fake:.4f}   arg id-head={int(id_logits.argmax(-1))}  '
          f'(true method={fake["method"]})')

    # ---------------------------------------------------------------- 5. contact sheet
    print('\n[5/5] contact sheet')
    names = sorted(p.name for p in out_dir.glob('*.png') if p.name != '99_contact_sheet.png')
    thumbs = []
    for n in names:
        im = cv2.cvtColor(cv2.imread(str(out_dir / n)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (180, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(im, n[:22], (4, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
        cv2.putText(im, n[:22], (4, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        thumbs.append(im)
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    thumbs += [np.full((180, 180, 3), 255, np.uint8)] * (rows * cols - len(thumbs))
    sheet = tile(thumbs, rows, cols, pad=8)
    cv2.imwrite(str(out_dir / '99_contact_sheet.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f'  99_contact_sheet.png               {sheet.shape[1]}x{sheet.shape[0]}')

    (out_dir / 'SOURCE.txt').write_text(
        f'FF++ {args.compression}\n'
        f'fake : {fake["method"]} / {fake["pair_id"]}  frames={fake["frames_dir"]}\n'
        f'real : {real["pair_id"]}\n'
        f'frame: {mid_path.name}   bbox={bbox}   T={n_frames}  face={face_size}  img={img_size}\n'
        f'ckpt : {ckpt_path}\n'
        f'p(fake)={p_fake:.4f}\n')
    print(f'\nAll assets in {out_dir}')


if __name__ == '__main__':
    main()
