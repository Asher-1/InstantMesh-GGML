#!/usr/bin/env python3
"""Definitive bake audit using the dumped ground truth (texdump.*).

  rgb.bin  = raw synthesizer_texture_forward output at covered pixels (ordered)
  gb.bin   = rasterized world positions (TR*TR*3)
  mask.bin = coverage
  ch_dump.png = final texture written to disk

Checks:
  (1) tex[p] == affine_clamp(rgb[covered.index(p)])  -> bake write correctness
  (2) rgb[covered] == PyTorch net_rgb at gb[covered] -> synthesizer parity at gb
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pytorch_reference import load_model  # noqa: E402
from pytorch_reference import MV  # noqa: E402

TR = 1024
IMG = "cute_horse"
D = "/tmp/texdump"


def main():
    mask = np.fromfile(D + ".mask.bin", np.uint8).reshape(TR, TR) > 0
    gb = np.fromfile(D + ".gb.bin", np.float32).reshape(TR, TR, 3)
    rgb = np.fromfile(D + ".rgb.bin", np.float32).reshape(-1, 3)
    covered = np.flatnonzero(mask)
    assert len(rgb) == len(covered), (len(rgb), len(covered))

    # (1) bake write correctness: tex vs clamp(rgb). PNG is vertically flipped
    # vs the in-memory tex (PNG row y = tex row tr-1-y), so unflip to compare.
    tex = np.asarray(Image.open("/tmp/ch_dump.png").convert("RGB"), np.float32) / 255.0
    tex_unflip = tex[::-1].reshape(-1, 3)
    clamp_rgb = np.clip(rgb * (1.0 + 2.0 * 0.001) - 0.001, 0, 1)
    d1 = np.abs(tex_unflip[covered] - clamp_rgb).mean()
    print(f"[1] bake write: tex(unflipped) vs clamp(rgb) over {len(covered)} px: MAE={d1:.5f} "
          f"(expect ~0 -> write correct)")

    # (2) synthesizer parity at gb positions
    model = load_model()
    images = np.fromfile(os.path.join(MV, IMG, "image.bin"), dtype=np.float32)[:6*3*224*224].reshape(1, 6, 3, 224, 224)
    cameras = np.fromfile(os.path.join(MV, IMG, "camera.bin"), dtype=np.float32)[:6*16].reshape(1, 6, 16)
    with torch.no_grad():
        planes = model.forward_planes(torch.from_numpy(images).cuda(), torch.from_numpy(cameras).cuda())
    pts = gb.reshape(-1, 3)[covered]
    rgb_pt = np.zeros((len(pts), 3), np.float32)
    B = 65536
    with torch.no_grad():
        for o in range(0, len(pts), B):
            v = torch.from_numpy(pts[o:o+B]).float().unsqueeze(0).cuda()
            rgb_pt[o:o+B] = model.synthesizer.get_texture_prediction(planes, v).clamp(0, 1).squeeze(0).cpu().numpy()
    d2 = np.abs(rgb - rgb_pt).mean()
    corr = np.corrcoef(rgb.ravel(), rgb_pt.ravel())[0, 1]
    mse = ((rgb - rgb_pt) ** 2).mean()
    psnr = 20 * np.log10(1 / np.sqrt(mse)) if mse > 0 else float("inf")
    print(f"[2] synth parity at gb: ggml raw rgb vs PyTorch net_rgb: PSNR={psnr:.2f} dB "
          f"MAE={d2:.4f} r={corr:.4f}")
    # compare using the affine-clamped final (what ends up in texture)
    d2b = np.abs(clamp_rgb - rgb_pt).mean()
    print(f"[2b] clamp(rgb) vs PyTorch: MAE={d2b:.4f}")

    # where do the gb positions actually live? histogram
    print(f"  gb range x=[{gb[:,:,0].min():.2f},{gb[:,:,0].max():.2f}] y=[{gb[:,:,1].min():.2f},{gb[:,:,1].max():.2f}] z=[{gb[:,:,2].min():.2f},{gb[:,:,2].max():.2f}]")


if __name__ == "__main__":
    main()
