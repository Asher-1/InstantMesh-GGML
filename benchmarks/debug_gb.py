#!/usr/bin/env python3
"""Debug the ggml texture bake using the dumped ground truth.

Compares:
  A) ggml rasterized world positions (gb.bin) vs a Python barycentric
     reconstruction from the same UVs (uvs.bin) + verts (verts.bin) + faces.
  B) ggml texture colors (at ggml gb positions) vs PyTorch net_rgb at the
     SAME ggml gb positions.  (proves whether the bake placement/colors are
     correct given ggml's own world positions)
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


def load_obj(path):
    verts, faces = [], []
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64)


def main():
    uvs = np.fromfile(D + ".uvs.bin", np.float32)
    gb = np.fromfile(D + ".gb.bin", np.float32).reshape(TR, TR, 3)
    mask = np.fromfile(D + ".mask.bin", np.uint8).reshape(TR, TR) > 0
    verts = np.fromfile(D + ".verts.bin", np.float32).reshape(-1, 3)
    V, F = load_obj("/tmp/ch_dump.obj")
    nv = len(verts)
    uv = uvs.reshape(nv, 2)
    ncover = mask.sum()
    print(f"nv={nv} verts_mesh={len(V)} covered={ncover}")

    # ---- A) reconstruct world positions from uv via barycentric -------------
    gb_rec = np.zeros_like(gb)
    done = np.zeros((TR, TR), bool)
    for fi in range(len(F)):
        f = F[fi]
        p = uv[f, 0] * TR
        q = uv[f, 1] * TR
        w0, w1, w2 = verts[f[0]], verts[f[1]], verts[f[2]]
        minx = max(0, int(p.min())); maxx = min(TR-1, int(np.ceil(p.max())))
        miny = max(0, int(q.min())); maxy = min(TR-1, int(np.ceil(q.max())))
        if minx > maxx or miny > maxy: continue
        gx = np.arange(minx, maxx+1, dtype=np.float32)+0.5
        gy = np.arange(miny, maxy+1, dtype=np.float32)+0.5
        GX, GY = np.meshgrid(gx, gy)
        den = (p[1]-p[2])*(q[0]-q[2]) + (p[2]-p[0])*(q[1]-q[2])
        if abs(den) < 1e-9: continue
        l0 = ((p[1]-p[2])*(GX-p[2]) + (p[2]-p[0])*(GY-q[2])) / den
        l1 = ((p[2]-p[0])*(GX-p[0]) + (p[0]-p[2])*(GY-q[0])) / den
        l2 = 1.0-l0-l1
        inside = (l0>=-1e-4)&(l1>=-1e-4)&(l2>=-1e-4)
        if not inside.any(): continue
        wp = l0[...,None]*w0 + l1[...,None]*w1 + l2[...,None]*w2
        reg = gb_rec[miny:maxy+1, minx:maxx+1]
        dreg = done[miny:maxy+1, minx:maxx+1]
        sel = inside & ~dreg
        reg[sel] = wp[sel]
        dreg[sel] = True
    dif = np.abs(gb - gb_rec)
    d = dif[mask]
    print(f"[A] ggml gb vs python reconstruction: MAE={d.mean():.5f} max={d.max():.5f} "
          f"mean_ggml_gb={gb[mask].mean(axis=0).round(4)} mean_rec={gb_rec[mask].mean(axis=0).round(4)}")
    # sample a few covered pixels
    idx = np.flatnonzero(mask)
    for k in [0, len(idx)//2, len(idx)-1]:
        p = idx[k]; py, px = divmod(p, TR)
        print(f"  px={px} py={py} ggml={gb[py,px].round(3)} rec={gb_rec[py,px].round(3)} "
              f"diff={dif[py,px].round(3)} uv={uvs.reshape(nv,2)[np.searchsorted(np.arange(1,TR+1)*0,0)] if False else '--'}")
    # is ggml gb a scaled/offset version of rec? fit per-channel affine on covered
    A = np.stack([gb_rec[mask], np.ones(mask.sum())], axis=1)
    coeff, *_ = np.linalg.lstsq(A, gb[mask], rcond=None)
    pred = A @ coeff
    resid = np.abs(gb[mask] - pred).mean()
    print(f"  best affine fit rec->ggml: slope={coeff[0].round(3)} bias={coeff[1].round(3)}  resid_MAE={resid:.5f}")

    # ---- B) PyTorch net_rgb at ggml gb positions vs ggml texture colors -----
    model = load_model()
    images = np.fromfile(os.path.join(MV, IMG, "image.bin"), dtype=np.float32)[:6*3*224*224].reshape(1, 6, 3, 224, 224)
    cameras = np.fromfile(os.path.join(MV, IMG, "camera.bin"), dtype=np.float32)[:6*16].reshape(1, 6, 16)
    with torch.no_grad():
        planes = model.forward_planes(torch.from_numpy(images).cuda(), torch.from_numpy(cameras).cuda())
    pts = gb[mask]
    # batch query
    rgb_pt = np.zeros((len(pts), 3), np.float32)
    B = 65536
    with torch.no_grad():
        for o in range(0, len(pts), B):
            v = torch.from_numpy(pts[o:o+B]).float().unsqueeze(0).cuda()
            rgb_pt[o:o+B] = model.synthesizer.get_texture_prediction(planes, v).clamp(0,1).squeeze(0).cpu().numpy()
    tex_gg = np.asarray(Image.open("/tmp/ch_dump.png").convert("RGB"), np.float32)/255.0
    rgb_gg = tex_gg[mask]
    diff = (rgb_gg - rgb_pt)**2
    mse = diff.mean(); psnr = 20*np.log10(1/np.sqrt(mse)) if mse>0 else float("inf")
    corr = np.corrcoef(rgb_gg.ravel(), rgb_pt.ravel())[0,1]
    print(f"[B] ggml texture vs PyTorch net_rgb at ggml gb positions: PSNR={psnr:.2f} dB MAE={np.abs(rgb_gg-rgb_pt).mean():.4f} r={corr:.4f}")


if __name__ == "__main__":
    main()
