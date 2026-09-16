#!/usr/bin/env python3
"""Definitive textured-render check independent of UV.

Render the mesh from the fixed VIEW producing, per pixel, the *surface world
position* (barycentric of the 3D vertices in screen space). Query PyTorch
net_rgb at those positions -> the TRUE appearance of the mesh. Compare against
render_textured() of the ggml textured OBJ. If they match, the ggml texture
render is correct in content and orientation (no UV/texture involvement)."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pytorch_reference import load_model  # noqa: E402
from pytorch_reference import MV  # noqa: E402
from render_textured import render_textured, rot, VIEW  # noqa: E402

IMG = "cute_horse"
OBJ = "/tmp/ch_dump.obj"


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


def worldpos_render(V, F, size=512):
    v = V - V.mean(axis=0); v = v / (np.max(np.abs(v)) + 1e-9)
    R = rot(VIEW["elev"], VIEW["azim"]); v = v @ R.T
    half = size / 2.0
    img = np.zeros((size, size, 3), np.float32)
    zbuf = np.full((size, size), np.inf, np.float32)
    xs = v[:,0]*half + half; ys = v[:,1]*half + half; zs = v[:,2]
    for fi in range(len(F)):
        f = F[fi]; p = xs[f]; q = ys[f]; z = zs[f]
        w0, w1, w2 = V[f[0]], V[f[1]], V[f[2]]
        minx = max(0, int(np.floor(p.min()))); maxx = min(size-1, int(np.ceil(p.max())))
        miny = max(0, int(np.floor(q.min()))); maxy = min(size-1, int(np.ceil(q.max())))
        if minx > maxx or miny > maxy: continue
        gx = np.arange(minx, maxx+1, dtype=np.float32)+0.5
        gy = np.arange(miny, maxy+1, dtype=np.float32)+0.5
        GX, GY = np.meshgrid(gx, gy)
        den = (p[1]-p[2])*(q[0]-q[2]) + (p[2]-p[0])*(q[1]-q[2])
        if abs(den) < 1e-12: continue
        l0 = ((p[1]-p[2])*(GX-p[2]) + (p[2]-p[0])*(GY-q[2])) / den
        l1 = ((p[2]-p[0])*(GX-p[0]) + (p[0]-p[2])*(GY-q[0])) / den
        l2 = 1.0-l0-l1
        inside = (l0>=-1e-4)&(l1>=-1e-4)&(l2>=-1e-4)
        if not inside.any(): continue
        wp = l0[...,None]*w0 + l1[...,None]*w1 + l2[...,None]*w2
        dz = l0*z[0]+l1*z[1]+l2*z[2]
        reg = zbuf[miny:maxy+1, minx:maxx+1]
        m = inside & (dz < reg)
        if not m.any(): continue
        reg[m] = dz[m]
        img[miny:maxy+1, minx:maxx+1][m] = wp[m]
    return img


def main():
    V, F = load_obj(OBJ)
    wp = worldpos_render(V, F)
    mask = wp.sum(axis=2) != 0  # (a covered pixel has nonzero world pos)
    # filter a few zero-ish edge pixels that are inside silhouette
    mask = mask & (np.abs(wp).sum(axis=2) > 1e-6)

    model = load_model()
    images = np.fromfile(os.path.join(MV, IMG, "image.bin"), dtype=np.float32)[:6*3*224*224].reshape(1, 6, 3, 224, 224)
    cameras = np.fromfile(os.path.join(MV, IMG, "camera.bin"), dtype=np.float32)[:6*16].reshape(1, 6, 16)
    with torch.no_grad():
        planes = model.forward_planes(torch.from_numpy(images).cuda(), torch.from_numpy(cameras).cuda())
    pts = wp[mask]
    rgb = np.zeros((len(pts), 3), np.float32)
    B = 65536
    with torch.no_grad():
        for o in range(0, len(pts), B):
            v = torch.from_numpy(pts[o:o+B]).float().unsqueeze(0).cuda()
            rgb[o:o+B] = model.synthesizer.get_texture_prediction(planes, v).clamp(0,1).squeeze(0).cpu().numpy()
    true_img = np.zeros_like(wp)
    true_img[mask] = rgb

    r_tex = render_textured(OBJ)
    m = mask
    corr = np.corrcoef(true_img[m].ravel(), r_tex[m].ravel())[0, 1]
    mse = ((true_img[m] - r_tex[m]) ** 2).mean()
    psnr = 20*np.log10(1/np.sqrt(mse)) if mse > 0 else float("inf")
    print(f"ggml textured render vs TRUE appearance (PyTorch net_rgb at surface):")
    print(f"  PSNR = {psnr:.2f} dB   pearson r = {corr:.4f}   over {m.sum()} px")
    print(f"  true mean={true_img[m].mean(axis=0).round(3)}  tex mean={r_tex[m].mean(axis=0).round(3)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(true_img); ax[0].set_title("true appearance (net_rgb at surface)"); ax[0].axis("off")
    ax[1].imshow(r_tex); ax[1].set_title("ggml textured render"); ax[1].axis("off")
    ax[2].imshow(np.abs(true_img - r_tex)); ax[2].set_title(f"|diff| PSNR={psnr:.1f}dB"); ax[2].axis("off")
    fig.tight_layout()
    out = os.path.join(ROOT, "benchmarks", "results", "tex_render_vs_true.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
