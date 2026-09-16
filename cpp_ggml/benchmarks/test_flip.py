#!/usr/bin/env python3
"""Determine if the ggml textured RENDER has a vertical-flip issue by comparing
the normal and vertically-flipped textured renders against the correct flat
vertex-color render (same geometry). The orientation that correlates higher is
correct."""
import os
import sys
import shutil

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from render_textured import render_textured  # noqa: E402


def load_obj(path):
    verts, cols = [], []
    faces = []
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "v" and len(t) >= 7:
            verts.append([float(t[1]), float(t[2]), float(t[3])])
            cols.append([float(t[4]), float(t[5]), float(t[6])])
        elif t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    return np.asarray(verts, np.float32), np.asarray(cols, np.float32), np.asarray(faces, np.int64)


def make_tex_obj(src_obj, tex_png, out_dir, flip_v=False):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(src_obj)
    new_obj = os.path.join(out_dir, base)
    new_png = os.path.join(out_dir, "tex.png")
    img = Image.open(tex_png).convert("RGB")
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(new_png)
    shutil.copy(src_obj, new_obj)
    with open(os.path.join(out_dir, "tex.mtl"), "w") as f:
        f.write("newmtl mat\nKa 1 1 1\nKd 1 1 1\nmap_Kd tex.png\n")
    return new_obj


def main():
    V, C, F = load_obj("/tmp/ch_vcol.obj")
    from render_textured import rot, VIEW
    v = V - V.mean(axis=0); v = v / (np.max(np.abs(v)) + 1e-9)
    R = rot(VIEW["elev"], VIEW["azim"]); v = v @ R.T
    half = 256
    img = np.zeros((512, 512, 3), np.float32)
    zbuf = np.full((512, 512), np.inf, np.float32)
    xs = v[:,0]*half + half; ys = v[:,1]*half + half; zs = v[:,2]
    tri = (C[F[:,0]] + C[F[:,1]] + C[F[:,2]]) / 3.0
    for fi in range(len(F)):
        f = F[fi]; p = xs[f]; q = ys[f]; z = zs[f]
        minx = max(0, int(np.floor(p.min()))); maxx = min(511, int(np.ceil(p.max())))
        miny = max(0, int(np.floor(q.min()))); maxy = min(511, int(np.ceil(q.max())))
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
        dz = l0*z[0]+l1*z[1]+l2*z[2]
        reg = zbuf[miny:maxy+1, minx:maxx+1]
        m = inside & (dz < reg)
        if not m.any(): continue
        reg[m] = dz[m]
        img[miny:maxy+1, minx:maxx+1][m] = tri[fi]
    mask = img.sum(axis=2) > 0.01
    ref = img[mask].ravel()

    tex_png = "/tmp/ch_dump.png"
    src_obj = "/tmp/ch_dump.obj"
    r_norm = render_textured(src_obj)
    r_flip = render_textured(make_tex_obj(src_obj, tex_png, "/tmp/flip_tex", flip_v=True))
    for name, r in [("normal", r_norm), ("flipped", r_flip)]:
        v_ = r[mask].ravel()
        corr = np.corrcoef(ref, v_)[0, 1]
        mse = ((r[mask] - img[mask]) ** 2).mean()
        psnr = 20*np.log10(1/np.sqrt(mse)) if mse > 0 else float("inf")
        print(f"  {name:8s} texture: r={corr:.3f}  PSNR={psnr:.1f} dB vs flat vertex render")


if __name__ == "__main__":
    main()
