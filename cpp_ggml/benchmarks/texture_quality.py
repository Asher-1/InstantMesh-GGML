#!/usr/bin/env python3
"""Texture "snowflake" quality comparison: before (single-point bake) vs after
(2x2 supersample + Gaussian low-pass, --tex-smooth). Reports noise metrics and
content-preservation PSNR/SSIM, and writes a side-by-side figure with a zoomed
patch highlighting the removed speckle.

Usage:
  python3 benchmarks/texture_quality.py <before.png> <after.png> [--out out.png]
"""
import sys, os
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0


def noise_metric(im):
    """High-frequency proxy: mean |gradient| and fraction of pixels whose value
    deviates strongly (median-filter residual > 0.10) — captures snowflake speckle."""
    gx = np.abs(np.diff(im, axis=1)).mean()
    gy = np.abs(np.diff(im, axis=0)).mean()
    med = ndimage.median_filter(im, 3)
    dev = np.abs(im - med).mean(axis=2)
    frac = np.mean(dev > 0.10)
    return (gx + gy) / 2, frac


def main():
    before_p, after_p = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
        os.path.join(os.path.dirname(before_p), "texture_before_after.png")

    b = load(before_p)
    a = load(after_p)
    mask = b.sum(axis=2) > 0.05
    H = W = b.shape[0]

    g_b, s_b = noise_metric(b)
    g_a, s_a = noise_metric(a)
    mse = ((b - a) ** 2)[mask].mean()
    psnr = 20 * np.log10(1 / np.sqrt(mse)) if mse > 0 else float("inf")
    ss = ssim(np.round(b * 255).astype(np.uint8),
              np.round(a * 255).astype(np.uint8), channel_axis=2)

    print(f"before: mean-grad={g_b:.4f}  noise-frac={s_b*100:.3f}%")
    print(f"after : mean-grad={g_a:.4f}  noise-frac={s_a*100:.3f}%")
    print(f"  -> grad {g_b/max(g_a,1e-9):.2f}x lower, noise-frac "
          f"{s_b/max(s_a,1e-9):.2f}x lower")
    print(f"content preservation: PSNR(optimized vs original)={psnr:.2f} dB  SSIM={ss:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, im, t in ((axes[0], b, "Before (single-point bake)"),
                      (axes[1], a, "After (2x2 supersample + Gaussian)")):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(t, fontsize=13)
        ax.axis("off")
    # zoom inset ~ center 25%
    zo = slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4)
    axin = axes[1].inset_axes([0.55, 0.05, 0.4, 0.4])
    axin.imshow(np.clip(a[zo], 0, 1))
    axin.set_title("smooth detail", fontsize=9)
    axin.axis("off")
    fig.suptitle(f"Texture snowflake fix  |  grad {g_b:.4f}->{g_a:.4f}  "
                 f"noise {s_b*100:.2f}%->{s_a*100:.2f}%  |  "
                 f"PSNR={psnr:.1f}dB  SSIM={ss:.3f}",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
