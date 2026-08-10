#!/usr/bin/env python3
"""Decisive end-to-end color parity: query PyTorch net_rgb on the EXACT ggml
mesh vertices (with PyTorch's own planes from the same input) and compare to
the colors ggml stored on those same vertices.

- If they match per-vertex  -> planes + color function align; the earlier mean
  difference is purely because the two pipelines produce different vertex sets.
- If they differ            -> ggml and pytorch produce different planes or the
  ggml color query uses a different coordinate frame (a real bug to chase).

Usage: python3 benchmarks/color_parity_e2e.py [image ...]
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import benchmarks.pytorch_reference as PR  # noqa: E402  (loads model, shims)


def read_obj_verts_colors(path):
    verts, cols = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if t and t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
                if len(t) >= 7:
                    cols.append([float(t[4]), float(t[5]), float(t[6])])
                else:
                    cols.append([0.6, 0.6, 0.6])
    return np.asarray(verts, np.float32), np.asarray(cols, np.float32)


def main():
    imgs = sys.argv[1:] or ["blue_cat", "fox", "robot", "cute_horse"]
    model = PR.load_model()
    for img in imgs:
        images = np.fromfile(os.path.join(PR.MV, img, "image.bin"), dtype=np.float32)
        cameras = np.fromfile(os.path.join(PR.MV, img, "camera.bin"), dtype=np.float32)
        images = torch.from_numpy(images[:6 * 3 * 224 * 224].reshape(1, 6, 3, 224, 224)).cuda()
        cameras = torch.from_numpy(cameras[:6 * 16].reshape(1, 6, 16)).cuda()

        gg_verts, gg_cols = read_obj_verts_colors(
            os.path.join(ROOT, "benchmarks", "results", f"{img}__cuda__f16.obj"))
        with torch.no_grad():
            planes = model.forward_planes(images, cameras)
            vt = torch.from_numpy(gg_verts).float().cuda().unsqueeze(0)
            rgb = model.synthesizer.get_texture_prediction(planes, vt)  # [1,M,3] incl clamp
            pt_cols = rgb.squeeze(0).cpu().numpy()  # [M,3] in [0,1]

        d = np.abs(pt_cols - gg_cols)
        rel = d.mean() / (pt_cols.std() + 1e-9)
        print(f"{img}: M={len(gg_verts)} color mean_abs={d.mean():.4f} max_abs={d.max():.4f} "
              f"rel={rel:.4f}  (py_mean={pt_cols.mean(0).round(3)} gg_mean={gg_cols.mean(0).round(3)})")


if __name__ == "__main__":
    main()
