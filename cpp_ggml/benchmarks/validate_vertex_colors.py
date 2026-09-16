#!/usr/bin/env python3
"""Isolate synthesizer_texture_forward correctness: compare ggml vertex colors
(exported at mesh vertices) against PyTorch net_rgb queried at the exact same
world positions (the mesh vertices). If they match, the texture color source is
correct and any texture bug lies in the UV/raster/bake path."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pytorch_reference import load_model  # noqa: E402
from pytorch_reference import MV  # noqa: E402

IMG = "cute_horse"
OBJ = "/tmp/ch_vcol.obj"   # ggml vertex-color export (same verts used by texmap)


def load_obj(path):
    verts, cols = [], []
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
            cols.append([float(t[4]), float(t[5]), float(t[6])])
    return np.asarray(verts, np.float32), np.asarray(cols, np.float32)


def main():
    model = load_model()
    images = np.fromfile(os.path.join(MV, IMG, "image.bin"), dtype=np.float32)[:6*3*224*224].reshape(1, 6, 3, 224, 224)
    cameras = np.fromfile(os.path.join(MV, IMG, "camera.bin"), dtype=np.float32)[:6*16].reshape(1, 6, 16)
    with torch.no_grad():
        planes = model.forward_planes(torch.from_numpy(images).cuda(),
                                      torch.from_numpy(cameras).cuda())

    V, gg = load_obj(OBJ)                 # ggml vertex colors (clamped 0..1)
    N = len(V)
    # PyTorch net_rgb at the same world positions (inner synthesizer query,
    # same sample_from_planes + net_rgb path the official texmap uses)
    v = torch.from_numpy(V).float().unsqueeze(0).cuda()  # (1,N,3)
    with torch.no_grad():
        pt = model.synthesizer.get_texture_prediction(planes, v).clamp(0, 1).squeeze(0).cpu().numpy()
    pt = np.asarray(pt, np.float32)
    d = (gg - pt) ** 2
    mse = d.mean(); psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    mae = np.abs(gg - pt).mean()
    corr = np.corrcoef(gg.ravel(), pt.ravel())[0, 1]
    print(f"ggml vertex colors vs PyTorch net_rgb at {N} verts:")
    print(f"  PSNR = {psnr:.2f} dB   MAE = {mae:.4f}   pearson r = {corr:.4f}")
    print(f"  per-channel MAE = {np.abs(gg-pt).mean(axis=0).round(4)}")
    print(f"  ggml mean={gg.mean(axis=0).round(3)} py mean={pt.mean(axis=0).round(3)}")


if __name__ == "__main__":
    main()
