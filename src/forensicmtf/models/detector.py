from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import functional as TF
from torchvision.models.video import R3D_18_Weights, r3d_18

EMBED_DIM = 512
FFT_DIM = 128
DWT_DIM = 128


class TemporalAttention(nn.Module):
    def __init__(self, d: int = EMBED_DIM):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class Stream1(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        r3d = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
        r3d.fc = nn.Identity()
        self.backbone = r3d
        self.attn = TemporalAttention(EMBED_DIM)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        feat = checkpoint(self.backbone, video, use_reentrant=False)
        return self.attn(feat.unsqueeze(1))


class BayarConv2d(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 32, k: int = 5):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        mask = torch.ones(k, k)
        mask[k // 2, k // 2] = 0.0
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w * self.mask
        w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-4)
        wf = w.clone()
        wf[:, :, self.k // 2, self.k // 2] = -1.0
        return F.conv2d(x, wf, padding=self.k // 2)


class FFTBranch(nn.Module):
    def __init__(self, out_dim: int = FFT_DIM, face_size: int = 112):
        super().__init__()
        self.dc_r = face_size // 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        gray = 0.299 * x32[:, 0] + 0.587 * x32[:, 1] + 0.114 * x32[:, 2]
        fft = torch.fft.fftshift(torch.fft.fft2(gray))
        logm = torch.log1p(torch.abs(fft))
        h, w = logm.shape[-2:]
        cy, cx = h // 2, w // 2
        r = self.dc_r
        mask = torch.ones_like(logm)
        mask[:, max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)] = 0.0
        logm = logm * mask
        mn = logm.flatten(1).min(1).values.view(-1, 1, 1)
        mx = logm.flatten(1).max(1).values.view(-1, 1, 1)
        return ((logm - mn) / (mx - mn + 1e-8)).unsqueeze(1).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self._spectrum(x)))


class HaarDWTBranch(nn.Module):
    def __init__(self, out_dim: int = DWT_DIM):
        super().__init__()
        s = 0.5
        haar = torch.tensor([
            [[s, s], [s, s]],
            [[s, s], [-s, -s]],
            [[s, -s], [s, -s]],
            [[s, -s], [-s, s]],
        ], dtype=torch.float32).unsqueeze(1).repeat(3, 1, 1, 1)
        self.register_buffer('haar', haar)
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subbands = F.conv2d(x, self.haar, stride=2, groups=3)
        return self.norm(self.encoder(subbands))


class Stream2(nn.Module):
    def __init__(self, face_size: int = 112, denoise_input: bool = False, denoise_kernel: int = 3,
                 denoise_sigma: float = 0.6, use_bayar_attention: bool = False):
        super().__init__()
        self.denoise_input = denoise_input
        self.denoise_kernel = denoise_kernel if denoise_kernel % 2 == 1 else denoise_kernel + 1
        self.denoise_sigma = denoise_sigma
        self.use_bayar_attention = use_bayar_attention
        self.bayar = BayarConv2d(3, 32, k=5)
        self.bayar_attention = nn.Sequential(
            nn.Conv2d(32, 16, 1), nn.ReLU(True), nn.Conv2d(16, 1, 1), nn.Sigmoid()
        ) if use_bayar_attention else None
        self.bayar_enc = nn.Sequential(
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, EMBED_DIM),
        )
        self.fft = FFTBranch(FFT_DIM, face_size=face_size)
        self.dwt = HaarDWTBranch(DWT_DIM)
        self.proj = nn.Sequential(nn.Linear(EMBED_DIM + FFT_DIM + DWT_DIM, EMBED_DIM), nn.ReLU(True))
        self.norm = nn.LayerNorm(EMBED_DIM)

    def forward(self, face_01: torch.Tensor) -> torch.Tensor:
        if self.denoise_input:
            face_01 = F.avg_pool2d(face_01, kernel_size=3, stride=1, padding=1)
            face_01 = TF.gaussian_blur(
                face_01,
                kernel_size=[self.denoise_kernel, self.denoise_kernel],
                sigma=[self.denoise_sigma, self.denoise_sigma],
            )
        bayar_feat = self.bayar(face_01).clamp(-2.0, 2.0)
        if self.bayar_attention is not None:
            bayar_feat = bayar_feat * self.bayar_attention(bayar_feat)
        f_b = self.bayar_enc(bayar_feat)
        f_f = self.fft(face_01)
        f_d = self.dwt(face_01)
        return self.norm(self.proj(torch.cat([f_b, f_f, f_d], dim=-1)))


class CrossAttentionFusion(nn.Module):
    def __init__(self, d: int = EMBED_DIM, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(d * 2, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, f_video: torch.Tensor, f_noise: torch.Tensor) -> tuple:
        q = f_video.unsqueeze(1)
        kv = f_noise.unsqueeze(1)
        a, _ = self.attn(q, kv, kv)
        f = self.norm1(q + a)
        f = self.norm2(f + self.mlp(f)).squeeze(1)
        return self.drop(f), f


class AdditionFusion(nn.Module):
    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        fused = torch.stack(feats, dim=0).sum(dim=0)
        return fused, fused


class ConcatFusion(nn.Module):
    def __init__(self, d: int = EMBED_DIM, max_streams: int = 3, dropout: float = 0.3):
        super().__init__()
        self.max_streams = max_streams
        self.proj = nn.Sequential(
            nn.Linear(d * max_streams, d),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat(feats, dim=-1)
        full_dim = EMBED_DIM * self.max_streams
        if cat.shape[-1] < full_dim:
            pad = torch.zeros(cat.shape[0], full_dim - cat.shape[-1], device=cat.device, dtype=cat.dtype)
            cat = torch.cat([cat, pad], dim=-1)
        fused = self.proj(cat)
        return fused, fused


class AttentionFusion(nn.Module):
    def __init__(self, d: int = EMBED_DIM, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(d * 2, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack(feats, dim=1)
        attended, _ = self.attn(tokens, tokens, tokens)
        fused = self.norm1(tokens + attended)
        pooled = fused.mean(dim=1)
        pooled = self.norm2(pooled + self.mlp(pooled))
        return self.drop(pooled), pooled


