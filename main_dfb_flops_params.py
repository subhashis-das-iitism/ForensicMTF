from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.models.forensicmtf import ForensicMTF


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _measure_flops(model: torch.nn.Module, video, face_01, face_norm, landmark_heatmap) -> int:
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(model, display=False) as flop_counter:
            model(video, face_01, face_norm, landmark_heatmap=landmark_heatmap)
        return int(flop_counter.get_total_flops())


def main():
    parser = argparse.ArgumentParser(description='Report parameters and FLOPs for ForensicMTF variants')
    parser.add_argument('--out-dir', default='records/DFB/model_stats')
    parser.add_argument('--face-size', type=int, default=112)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--video-size', type=int, default=224)
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video = torch.randn(1, 3, args.frames, args.video_size, args.video_size, device=device)
    face_01 = torch.randn(1, 3, args.face_size, args.face_size, device=device)
    face_norm = torch.randn(1, 3, args.face_size, args.face_size, device=device)
    heatmap = torch.randn(1, 1, args.face_size, args.face_size, device=device)

    # Stream-composition variants (same axis as the old paper's S/N/T ablation) and
    # head-composition variants (the new MLG-MTF axis: landmark attention / localization
    # / identification), each independently toggleable on ForensicMTF.
    variants = [
        ('Full (S+N+T, all heads)', dict(use_spatial=True, use_noise=True, use_temporal=True,
                                          use_localization_head=True, use_identification_head=True, use_landmark_attention=True)),
        ('Temporal only', dict(use_spatial=False, use_noise=False, use_temporal=True,
                               use_localization_head=False, use_identification_head=False, use_landmark_attention=False)),
        ('Noise only', dict(use_spatial=False, use_noise=True, use_temporal=False,
                            use_localization_head=False, use_identification_head=False, use_landmark_attention=False)),
        ('Spatial only', dict(use_spatial=True, use_noise=False, use_temporal=False,
                              use_localization_head=False, use_identification_head=False, use_landmark_attention=False)),
        ('S+N', dict(use_spatial=True, use_noise=True, use_temporal=False,
                    use_localization_head=False, use_identification_head=False, use_landmark_attention=False)),
        ('S+T', dict(use_spatial=True, use_noise=False, use_temporal=True,
                    use_localization_head=False, use_identification_head=False, use_landmark_attention=False)),
        ('N+T', dict(use_spatial=False, use_noise=True, use_temporal=True,
                    use_localization_head=False, use_identification_head=False, use_landmark_attention=False)),
        ('MLG-MTF: base (S+N+T, no heads)', dict(use_spatial=True, use_noise=True, use_temporal=True,
                                                  use_localization_head=False, use_identification_head=False, use_landmark_attention=False)),
        ('MLG-MTF: + landmark attention', dict(use_spatial=True, use_noise=True, use_temporal=True,
                                                use_localization_head=False, use_identification_head=False, use_landmark_attention=True)),
        ('MLG-MTF: + localization head', dict(use_spatial=True, use_noise=True, use_temporal=True,
                                               use_localization_head=True, use_identification_head=False, use_landmark_attention=True)),
        ('MLG-MTF: + identification head (full)', dict(use_spatial=True, use_noise=True, use_temporal=True,
                                                        use_localization_head=True, use_identification_head=True, use_landmark_attention=True)),
    ]

    rows = []
    for name, kwargs in variants:
        model = ForensicMTF(pretrained=False, face_size=args.face_size, **kwargs).to(device)
        params = _count_params(model)
        flops = _measure_flops(model, video, face_01, face_norm, heatmap)
        flops_per_frame = flops / float(max(1, args.frames))
        rows.append({
            'method': name,
            'params': params,
            'params_m': round(params / 1e6, 3),
            'flops_clip': flops,
            'flops_clip_g': round(flops / 1e9, 3),
            'flops_frame': int(round(flops_per_frame)),
            'flops_frame_g': round(flops_per_frame / 1e9, 3),
        })

    csv_path = out_dir / 'model_flops_params.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    txt_path = out_dir / 'model_flops_params.txt'
    lines = ['FLOPs and Parameters for ForensicMTF Variants', '']
    lines.append(f'Counting convention: clip-level input with T={args.frames}, video={args.video_size}x{args.video_size}, face={args.face_size}x{args.face_size}')
    lines.append('Per-frame FLOPs are reported as clip FLOPs divided by T.')
    lines.append('')
    lines.append('method,params,params_m,flops_clip,flops_clip_g,flops_frame,flops_frame_g')
    for row in rows:
        lines.append(
            f"{row['method']},{row['params']},{row['params_m']},"
            f"{row['flops_clip']},{row['flops_clip_g']},{row['flops_frame']},{row['flops_frame_g']}"
        )
    txt_path.write_text('\n'.join(lines) + '\n')
    print(f'[done] wrote model stats to {csv_path}')
    for row in rows:
        print(f"  {row['method']:<40} params={row['params_m']:>8}M  flops/clip={row['flops_clip_g']:>8}G")


if __name__ == '__main__':
    main()
