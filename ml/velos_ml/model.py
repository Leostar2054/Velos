"""Residual U-Net for complex-image refinement (small enough to train on CPU)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class RefineUNet(nn.Module):
    """x[B,3,H,W] (Re, Im, mask) → y[B,2,H,W] (Re, Im). Predicts a residual on (Re, Im)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 2, base: int = 32, depth: int = 3, residual: bool = True):
        super().__init__()
        self.residual = residual
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList(); c = in_ch
        for ch in chs[:-1]:
            self.enc.append(ConvBlock(c, ch)); c = ch
        self.bottom = ConvBlock(c, chs[-1])
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        c = chs[-1]
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(ConvBlock(ch * 2, ch)); c = ch
        self.head = nn.Conv2d(c, out_ch, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # start as identity

    def forward(self, x):
        H, W = x.shape[-2:]
        pad_h = (-H) % 2 ** len(self.enc); pad_w = (-W) % 2 ** len(self.enc)
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        skips = []; h = xp
        for blk in self.enc:
            h = blk(h); skips.append(h); h = F.max_pool2d(h, 2)
        h = self.bottom(h)
        for up, blk, s in zip(self.up, self.dec, reversed(skips)):
            h = up(h); h = blk(torch.cat([h, s], 1))
        out = self.head(h)[..., :H, :W]
        if self.residual:
            out = out + x[:, :2]
        return out


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    return RefineUNet(base=int(m.get("base", 32)), depth=int(m.get("depth", 3)),
                      residual=bool(m.get("residual", True)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
