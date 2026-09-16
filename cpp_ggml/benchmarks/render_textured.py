#!/usr/bin/env python3
"""Render the ggml textured export side-by-side with the input and the
PyTorch-equivalent reference, under the same input conditions.

Composes a 3-panel comparison:  input image | PyTorch reference (f32-CUDA
vertex-color mesh, validated to match official output) | ggml textured mesh
(--export-texmap, OBJ+MTL+PNG).

The ggml textured mesh is rendered with a small pure-numpy orthographic
z-buffer rasterizer with bilinear texture sampling — no external renderer
(pyrender/OpenGL) required.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")            # cpp_ggml/
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # repo root (examples/)
RES = os.path.join(ROOT, "benchmarks", "results")
SOURCE = {
    "cute_horse": os.path.join(REPO_ROOT, "examples", "cute_horse.jpg"),
}

LIGHT = np.array([0.25, 0.30, 0.90]); LIGHT /= np.linalg.norm(LIGHT)
VIEW = dict(elev=28, azim=-58)


def rot(elev_deg, azim_deg):
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    Re = np.array([[1,0,0],[0,np.cos(e),-np.sin(e)],[0,np.sin(e),np.cos(e)]])
    Ra = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    return Ra @ Re


def render_ortho(verts, faces, colors, size=512):
    """Orthographic z-buffer rasterizer (vectorized per-triangle)."""
    v = verts - verts.mean(axis=0)
    v = v / (np.max(np.abs(v)) + 1e-9)
    R = rot(VIEW["elev"], VIEW["azim"])
    v = v @ R.T
    v0, v1, v2 = v[faces[:,0]], v[faces[:,1]], v[faces[:,2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[:,2] = np.abs(n[:,2])
    inten = np.clip(n @ LIGHT, 0.0, 1.0)[:, None]
    tri_col = (0.30 + 0.70 * inten) * (colors[faces[:,0]] + colors[faces[:,1]] + colors[faces[:,2]]) / 3.0

    half = size / 2.0
    img = np.zeros((size, size, 3), np.float32)
    zbuf = np.full((size, size), np.inf, np.float32)
    xs = v[:,0] * half + half
    ys = v[:,1] * half + half
    zs = v[:,2]
    for f in range(len(faces)):
        p = xs[faces[f]]; q = ys[faces[f]]; z = zs[faces[f]]
        cc = tri_col[f]
        minx = max(0, int(np.floor(p.min()))); maxx = min(size-1, int(np.ceil(p.max())))
        miny = max(0, int(np.floor(q.min()))); maxy = min(size-1, int(np.ceil(q.max())))
        if minx > maxx or miny > maxy: continue
        gx = np.arange(minx, maxx+1, dtype=np.float32) + 0.5
        gy = np.arange(miny, maxy+1, dtype=np.float32) + 0.5
        GX, GY = np.meshgrid(gx, gy)
        den = (p[1]-p[2])*(q[0]-q[2]) + (p[2]-p[0])*(q[1]-q[2])
        if abs(den) < 1e-12: continue
        l0 = ((p[1]-p[2])*(GX-p[2]) + (p[2]-p[0])*(GY-q[2])) / den
        l1 = ((p[2]-p[0])*(GX-p[0]) + (p[0]-p[2])*(GY-q[0])) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any(): continue
        dz = l0*z[0] + l1*z[1] + l2*z[2]
        reg = zbuf[miny:maxy+1, minx:maxx+1]
        m = inside & (dz < reg)
        if not m.any(): continue
        reg[m] = dz[m]
        img[miny:maxy+1, minx:maxx+1][m] = cc
    return img


def render_textured(mesh_path, size=512):
    """Render a textured OBJ (trimesh) with bilinear texture sampling."""
    m = trimesh.load(mesh_path, process=False)
    V = np.asarray(m.vertices, np.float32)
    F = np.asarray(m.faces, np.int64)
    uv = np.asarray(m.visual.uv, np.float32)
    tex = np.asarray(m.visual.material.image.convert("RGB"), np.float32) / 255.0
    th, tw = tex.shape[:2]

    v = V - V.mean(axis=0); v = v / (np.max(np.abs(v)) + 1e-9)
    R = rot(VIEW["elev"], VIEW["azim"]); v = v @ R.T
    half = size / 2.0
    img = np.zeros((size, size, 3), np.float32)
    zbuf = np.full((size, size), np.inf, np.float32)
    xs = v[:,0] * half + half
    ys = v[:,1] * half + half
    zs = v[:,2]
    for f in F:
        p = xs[f]; q = ys[f]; z = zs[f]
        u0, u1, u2 = uv[f]
        minx = max(0, int(np.floor(p.min()))); maxx = min(size-1, int(np.ceil(p.max())))
        miny = max(0, int(np.floor(q.min()))); maxy = min(size-1, int(np.ceil(q.max())))
        if minx > maxx or miny > maxy: continue
        gx = np.arange(minx, maxx+1, dtype=np.float32) + 0.5
        gy = np.arange(miny, maxy+1, dtype=np.float32) + 0.5
        GX, GY = np.meshgrid(gx, gy)
        den = (p[1]-p[2])*(q[0]-q[2]) + (p[2]-p[0])*(q[1]-q[2])
        if abs(den) < 1e-12: continue
        l0 = ((p[1]-p[2])*(GX-p[2]) + (p[2]-p[0])*(GY-q[2])) / den
        l1 = ((p[2]-p[0])*(GX-p[0]) + (p[0]-p[2])*(GY-q[0])) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any(): continue
        dz = l0*z[0] + l1*z[1] + l2*z[2]
        reg = zbuf[miny:maxy+1, minx:maxx+1]
        m = inside & (dz < reg)
        if not m.any(): continue
        uu = l0*u0[0] + l1*u1[0] + l2*u2[0]
        vv = l0*u0[1] + l1*u1[1] + l2*u2[1]
        sx = np.clip(uu * (tw - 1), 0, tw - 1); sy = np.clip((1.0 - vv) * (th - 1), 0, th - 1)
        x0 = np.floor(sx).astype(np.int32); y0 = np.floor(sy).astype(np.int32)
        fx = sx - x0; fy = sy - y0
        x1 = np.minimum(x0+1, tw-1); y1 = np.minimum(y0+1, th-1)
        c = (tex[y0,x0]*(1-fx)[...,None]*(1-fy)[...,None] +
             tex[y0,x1]*fx[...,None]*(1-fy)[...,None] +
             tex[y1,x0]*(1-fx)[...,None]*fy[...,None] +
             tex[y1,x1]*fx[...,None]*fy[...,None])
        reg[m] = dz[m]
        tmp = img[miny:maxy+1, minx:maxx+1]
        tmp[m] = np.clip(c[m], 0, 1)
    return img


def load_pytorch_ref(img):
    path = os.path.join(RES, f"{img}__cuda__f32.obj")
    if not os.path.exists(path): return None
    verts, faces, col = [], [], []
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
            col.append([min(1,max(0,float(t[4]))), min(1,max(0,float(t[5]))), min(1,max(0,float(t[6])))])
        elif t[0] == "f":
            faces.append([int(x.split("/")[0])-1 for x in t[1:4]])
    return (np.asarray(verts,np.float32), np.asarray(faces,np.int64), np.asarray(col,np.float32))


def main(img):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    # input
    axes[0].imshow(np.asarray(Image.open(SOURCE[img]).convert("RGB")))
    axes[0].set_axis_off(); axes[0].set_title("input", fontsize=14)
    # PyTorch reference
    ref = load_pytorch_ref(img)
    if ref is not None:
        axes[1].imshow(render_ortho(ref[0], ref[1], ref[2]))
        axes[1].set_title("PyTorch reference\n(vertex color)", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "n/a", ha="center", va="center")
    axes[1].set_axis_off()
    # ggml textured
    tex_path = os.path.join(RES, f"{img}_tex.obj")
    if os.path.exists(tex_path):
        axes[2].imshow(render_textured(tex_path))
        axes[2].set_title("ggml textured (--export-texmap)", fontsize=13)
    else:
        axes[2].text(0.5, 0.5, "n/a", ha="center", va="center")
    axes[2].set_axis_off()
    fig.tight_layout()
    out = os.path.join(RES, f"input_vs_pytorch_vs_texmap_{img}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cute_horse")
